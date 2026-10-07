"""Account-engine evaluation pipeline: one spec, every scenario, one report card."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import mmap
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from src.backtest.engine import DelistPolicy
from src.backtest.market import MarketArrays, load_market_arrays
from src.core.pit import PITDataError
from src.data.research_protocol import ResearchProtocol, WindowAuthorization, WindowGuard
from src.research.book import (
    BookSpec,
    build_sleeve_targets,
    combine_sleeve_targets,
    sleeve_capital_krw,
)
from src.research.cube import ResearchCube
from src.research.evaluation import (
    EvaluationEvidence,
    EvaluationPolicy,
    ReportCard,
    build_report_card,
)
from src.research.hedge import (
    BetaNeutralOverlay,
    HedgeInputs,
    HedgeSpec,
    derivative_config,
)
from src.research.ledger_bridge import LedgerOutcome, overlay_market_from_inputs
from src.research.model import ScoreMatrix, ScorerConfig, walk_forward_scores
from src.research.panel import FEATURE_NAMES, HORIZONS, FeaturePanel, build_panel
from src.research.policy import TrendCashPolicy, universe_mask
from src.research.registry import RunRegistry, RunReturns
from src.research.simulator import SimConfig, simulate
from src.research.stats import annualized_log_growth

__all__ = [
    "EvaluationRun",
    "Pipeline",
    "PipelineContext",
    "StrategySpec",
    "load_strategy_spec",
    "strategy_spec_from_canonical_json",
]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """One evaluated spec: the report card plus the evidence behind it.

    ``spec_json`` is the canonical JSON the run was produced from; champion/challenger comparisons re-read it
    to attribute a neighbor run to the knob it probes.
    """

    report: ReportCard
    evidence: EvaluationEvidence
    spec_json: str | None = None


class StrategySpec(BaseModel):
    """One pre-registered strategy: selection policy plus frozen scorer, sleeves and hedge overlay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: TrendCashPolicy
    scorer: ScorerConfig
    book: BookSpec
    hedge: HedgeSpec

    @model_validator(mode="after")
    def _check_sleeves(self) -> StrategySpec:
        if int(self.book.sleeves) != int(self.policy.rebalance_every_sessions):
            raise ValueError("book.sleeves must equal policy.rebalance_every_sessions")
        return self

    def canonical_json(self) -> str:
        payload = {
            "book": json.loads(self.book.canonical_json()),
            "feature_names": list(FEATURE_NAMES),
            "hedge": json.loads(self.hedge.canonical_json()),
            "horizons": list(HORIZONS),
            "policy": json.loads(self.policy.canonical_json()),
            "scorer": json.loads(self.scorer.canonical_json()),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_strategy_spec(path: Path) -> StrategySpec:
    """Load a TOML with ``[policy]``, ``[scorer]``, ``[book]`` and ``[hedge]`` tables."""
    import tomllib

    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError:
        raise
    except ValueError as exc:
        raise ValueError(f"invalid strategy TOML: {path}: {exc}") from exc
    if not isinstance(raw, dict):  # pragma: no cover - tomllib always returns a dict
        raise ValueError(f"invalid strategy TOML: {path}")
    allowed = {"policy", "scorer", "book", "hedge"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown strategy keys: {sorted(unknown)}")
    if "policy" not in raw or "scorer" not in raw or "book" not in raw or "hedge" not in raw:
        raise ValueError("strategy TOML must declare [policy], [scorer], [book] and [hedge] tables")
    try:
        policy = TrendCashPolicy.model_validate(raw["policy"])
    except Exception as exc:
        raise ValueError(f"invalid [policy] table: {exc}") from exc
    try:
        scorer = ScorerConfig.model_validate(raw["scorer"])
    except Exception as exc:
        raise ValueError(f"invalid [scorer] table: {exc}") from exc
    try:
        book = BookSpec.model_validate(raw["book"])
    except Exception as exc:
        raise ValueError(f"invalid [book] table: {exc}") from exc
    try:
        hedge = HedgeSpec.model_validate(raw["hedge"])
    except Exception as exc:
        raise ValueError(f"invalid [hedge] table: {exc}") from exc
    try:
        return StrategySpec(policy=policy, scorer=scorer, book=book, hedge=hedge)
    except Exception as exc:
        raise ValueError(f"invalid strategy spec: {exc}") from exc


def strategy_spec_from_canonical_json(payload: str) -> StrategySpec:
    """Rebuild a spec from ``StrategySpec.canonical_json`` (what the champion store keeps as the identity).

    Raises ValueError when the payload is not canonical strategy JSON or violates a spec invariant.
    """
    try:
        raw = json.loads(payload)
        return StrategySpec(
            policy=TrendCashPolicy.model_validate(raw["policy"]),
            scorer=ScorerConfig.model_validate(raw["scorer"]),
            book=BookSpec.model_validate(raw["book"]),
            hedge=HedgeSpec.model_validate(raw["hedge"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid strategy canonical JSON: {exc}") from exc


@dataclass(frozen=True, slots=True)
class PipelineContext:
    protocol: ResearchProtocol
    cube: ResearchCube
    registry: RunRegistry
    guard: WindowGuard
    panel_dir: Path
    dividends: pl.DataFrame
    hedge_inputs: HedgeInputs
    rules: Any
    engine_config_path: Path
    market_cache_root: Path
    reports_root: Path
    scores_root: Path
    ledger_runner: Callable[..., LedgerOutcome]
    now: Callable[[], datetime]
    dataset_ids: Mapping[str, str]
    cash_returns: NDArray[np.float64] | None = None


#: Datasets the account engine reads beside the cube. The account goes straight to them, so rebuilding any of
#: them changes the run even when the cube id is unchanged.
_RUN_INPUT_DATASETS = ("market_panel", "dividend_events", "hedge_series", "cash_series")


def _sessions_of(cube: ResearchCube) -> list[date]:
    return list(cube.sessions)


def _scores_identity(spec: StrategySpec) -> str:
    """Cache identity of a score matrix: the scorer config plus the universe it trains and predicts on."""
    payload = f"{spec.scorer.config_hash}|{spec.policy.universe.model_dump_json()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evaluation_run_id(
    *,
    spec: StrategySpec,
    capital_krw: int,
    protocol_hash: str,
    cube_id: str,
    dataset_ids: Mapping[str, str],
    start: date,
    end: date,
    engine_config_bytes: bytes,
) -> str:
    """Identity of one evaluation run.

    ``capital_krw`` is part of it because integer shares, the ``min_units_per_slot`` affordability filter and
    price impact make every stream capital-dependent: two capitals are two runs and must never collide in the
    registry or in the reports directory. The four dataset ids are part of it because the account reads those
    inputs directly instead of through the cube, so a rebuilt input is a different run even when the cube id
    survived the rebuild.
    """
    missing = [name for name in _RUN_INPUT_DATASETS if not dataset_ids.get(name)]
    if missing:
        raise ValueError(f"pipeline dataset_ids are missing: {missing}")
    payload = {
        "capital_krw": int(capital_krw),
        "cube_id": cube_id,
        "dataset_ids": {name: str(dataset_ids[name]) for name in _RUN_INPUT_DATASETS},
        "end": end.isoformat(),
        "engine_config_sha256": hashlib.sha256(engine_config_bytes).hexdigest(),
        "protocol_hash": protocol_hash,
        "spec_json": spec.canonical_json(),
        "start": start.isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class _OverlayDecision:
    """One overlay decision the account engine asked for; ``None`` means "keep the current position"."""

    session_idx: int
    contracts: int | None
    inverse_value_krw: int | None

    def key(self) -> tuple[int | None, int | None]:
        return (self.contracts, self.inverse_value_krw)


class _RecordingOverlay:
    """``OverlayPolicy`` decorator that keeps every decision the engine asks for.

    The causal perturbation compares these decisions against a replay whose inputs after the cut row were
    corrupted; a decision at a row ``<= cut`` must be identical.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.decisions: list[_OverlayDecision] = []

    def target(self, state: Any) -> Any:
        decision = self._inner.target(state)
        self.decisions.append(
            _OverlayDecision(
                session_idx=int(state.session_idx),
                contracts=None if decision is None else int(decision.contracts),
                inverse_value_krw=None if decision is None else int(decision.inverse_value_krw),
            )
        )
        return decision


def _nav_series(outcome: LedgerOutcome) -> NDArray[np.float64]:
    return np.ascontiguousarray(np.asarray(outcome.nav_krw, dtype=np.float64), dtype=np.float64)


def _log_memory_phase(phase: str) -> None:
    """One INFO telemetry line with RSS, anonymous memory and peak RSS in GiB (``n/a`` when unavailable)."""
    gib = float(1 << 30)
    rss_gb = anon_gb = peak_gb = "n/a"
    with contextlib.suppress(Exception):
        import psutil

        proc = psutil.Process()
        rss = float(proc.memory_info().rss)
        rss_gb = f"{rss / gib:.2f}"
        with contextlib.suppress(Exception):
            shared = getattr(proc.memory_full_info(), "shared", None)
            if shared is not None:
                anon_gb = f"{max(rss - float(shared), 0.0) / gib:.2f}"
    with contextlib.suppress(Exception):
        import resource

        peak_gb = f"{float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 / 1024.0:.2f}"
    _LOG.info(
        "[SYS] evaluate memory phase=%s rss_gb=%s anon_gb=%s peak_gb=%s",
        phase,
        rss_gb,
        anon_gb,
        peak_gb,
    )


def _corrupt_hedge_inputs(
    inputs: HedgeInputs, cut: int, *, sessions: Sequence[date], rng: Any
) -> HedgeInputs:
    """Scale the index level and inverse close of every session after the cube row ``cut`` (NaN stays NaN).

    Sessions are matched by date rather than by position: the hedge series may carry a lead session the cube
    timeline does not have.
    """
    levels = np.asarray(inputs.index_level, dtype=np.float64).copy()
    inverse = np.asarray(inputs.inverse_close, dtype=np.float64).copy()
    cut_session = sessions[cut]
    tail = [pos for pos, day in enumerate(inputs.sessions) if day > cut_session]
    for arr in (levels, inverse):
        block = arr[tail]
        noise = np.exp(0.1 * rng.standard_normal(len(tail)))
        arr[tail] = np.where(np.isfinite(block), block * noise, block)
    return HedgeInputs(
        sessions=inputs.sessions,
        index_level=np.ascontiguousarray(levels, dtype=np.float64),
        inverse_close=np.ascontiguousarray(inverse, dtype=np.float64),
    )


def _corrupt_cash_returns(values: NDArray[np.float64], cut: int, rng: Any) -> NDArray[np.float64]:
    """Jitter the cash return after ``cut`` (NaN stays NaN)."""
    out = np.asarray(values, dtype=np.float64).copy()
    tail = slice(int(cut) + 1, int(out.shape[0]))
    noise = np.exp(0.05 * rng.standard_normal(out[tail].shape)) - 1.0
    out[tail] = np.where(np.isfinite(out[tail]), out[tail] + 0.01 * noise, out[tail])
    return np.ascontiguousarray(out, dtype=np.float64)


def _decision_mismatches(
    clean: Sequence[_OverlayDecision], perturbed: Sequence[_OverlayDecision], cut: int
) -> int:
    """Overlay decisions taken at rows ``<= cut`` that differ."""
    mismatches = 0
    for before, after in zip(clean, perturbed, strict=False):
        if before.session_idx > cut:
            break
        if before.key() != after.key():
            mismatches += 1
    return mismatches


def _corrupt_market_arrays(arrays: MarketArrays, cut: int, rng: Any) -> MarketArrays:
    """Corrupt every int, float and bool field of the engine's own arrays at rows after ``cut``.

    The replay must stay a market the engine accepts, so integer fields keep their sign (``0`` stays absent, a
    price stays a positive price the limit clamp can use) and ``present`` may only gain rows: the ledger refuses
    to mark a holding in an instrument the arrays report as missing, which would turn an artificial input into
    an unrelated failure instead of the causal comparison this exists for.
    """
    tail = slice(int(cut) + 1, None)

    def _ints(block: NDArray[np.int64]) -> NDArray[np.int64]:
        work = np.array(block, dtype=np.int64, copy=True)
        values = work[tail]
        if values.size:
            scaled = np.rint(values * np.exp(0.05 * rng.standard_normal(values.shape))).astype(np.int64)
            work[tail] = np.where(values > 0, np.maximum(scaled, 1), values)
        return work

    def _floats(block: NDArray[np.float64]) -> NDArray[np.float64]:
        work = np.array(block, dtype=np.float64, copy=True)
        values = work[tail]
        if values.size:
            work[tail] = np.where(np.isfinite(values), values * np.exp(0.3 * rng.standard_normal(values.shape)), values)
        return work

    def _bools(block: NDArray[np.bool_]) -> NDArray[np.bool_]:
        work = np.array(block, dtype=bool, copy=True)
        values = work[tail]
        if values.size:
            work[tail] = np.logical_xor(values, rng.random(values.shape) < 0.5)
        return work

    def _presence(block: NDArray[np.bool_]) -> NDArray[np.bool_]:
        work = np.array(block, dtype=bool, copy=True)
        work[tail] = True
        return work

    return MarketArrays(
        dataset_id=arrays.dataset_id,
        sessions=arrays.sessions,
        instrument_ids=arrays.instrument_ids,
        int_fields={name: _ints(block) for name, block in arrays.int_fields.items()},
        float_fields={name: _floats(block) for name, block in arrays.float_fields.items()},
        bool_fields={
            name: _presence(block) if name == "present" else _bools(block)
            for name, block in arrays.bool_fields.items()
        },
        market=arrays.market,
    )


def _corrupt_dividends(dividends: pl.DataFrame, cut_session: date, rng: Any) -> pl.DataFrame:
    """Scale the payout of every dividend whose ex-date falls after the cut.

    ``dps_krw`` stays a positive integer because the engine rejects anything else; only the amount changes, so
    the replay is the same corporate-action calendar paying a different cash flow.
    """
    if not dividends.height or "ex_session" not in dividends.columns:
        return dividends
    factors = rng.integers(2, 5, size=dividends.height)
    return dividends.with_columns(
        pl.when(pl.col("ex_session") > cut_session)
        .then(pl.col("dps_krw") * pl.Series(factors, dtype=pl.Int64))
        .otherwise(pl.col("dps_krw"))
        .alias("dps_krw")
    )


class _LazyCorruptCubeArrays(Mapping[str, NDArray[Any]]):
    """Read-only mapping of a cube's arrays, corrupted past a cut row, materialised on first access.

    A memory-mapped source is reopened copy-on-write (``mmap_mode="c"``) so only the pages of the corrupted
    rows are charged to private memory; an in-memory source is copied. Each array has its own generator keyed
    by ``(seed, cut, name)``, so the corrupted bytes never depend on which consumer touched the array first.
    """

    __slots__ = ("_cache", "_cut", "_seed", "_source")

    def __init__(self, source: Mapping[str, NDArray[Any]], cut: int, seed: int) -> None:
        self._source = source
        self._cut = int(cut)
        self._seed = int(seed)
        self._cache: dict[str, NDArray[Any]] = {}

    def __getitem__(self, name: str) -> NDArray[Any]:
        corrupted = self._cache.get(name)
        if corrupted is None:
            corrupted = _corrupt_tail(self._source[name], self._cut, self._seed, name)
            self._cache[name] = corrupted
        return corrupted

    def __iter__(self) -> Iterator[str]:
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)

    def materialised(self) -> tuple[str, ...]:
        """Names corrupted so far, in first-access order."""
        return tuple(self._cache)


def _corrupt_tail(source: NDArray[Any], cut: int, seed: int, name: str) -> NDArray[Any]:
    """Copy ``source`` and corrupt only its rows after ``cut``; the prefix stays byte-identical.

    The generator is seeded from ``(seed, cut, crc32(name))`` alone, so the result is a function of the
    array's identity rather than of access order, and a memory-mapped source pays only for the written pages.
    """
    # Only a top-level map (its buffer is the mmap itself) can be reopened from filename/offset/shape; a view of
    # a memmap inherits those attributes from its parent and would reopen the wrong bytes.
    if isinstance(source, np.memmap) and source.filename is not None and isinstance(source.base, mmap.mmap):
        work: NDArray[Any] = np.memmap(
            str(source.filename), dtype=source.dtype, shape=source.shape, mode="c", order="C", offset=source.offset
        )
    else:
        work = np.array(source, copy=True, order="C")
    rng = np.random.default_rng([seed, cut, zlib.crc32(name.encode("utf-8"))])
    if work.ndim == 2 and work.flags.c_contiguous:
        # C-order rows make the tail one contiguous byte range, so only it is written and only it faults in.
        block = work[cut + 1 :, :]
        shape = block.shape
        if work.dtype == np.bool_:
            block ^= rng.random(shape) < 0.5
        elif np.issubdtype(work.dtype, np.floating):
            # x * (1 + 0.5 * N(0, 1)), applied in place: the noise buffer is the only transient allocation, so
            # the cost is the corrupted tail plus one tail-sized buffer, never a second copy of the tail.
            scale = rng.standard_normal(shape)
            np.multiply(scale, 0.5, out=scale)
            np.add(scale, 1.0, out=scale)
            block *= scale
        elif np.issubdtype(work.dtype, np.integer):
            block += rng.integers(-2, 3, size=shape)
    work.flags.writeable = False
    return work


class Pipeline:
    """Account-engine evaluation pipeline for one pre-registered spec."""

    def __init__(self, context: PipelineContext) -> None:
        self._ctx = context
        self._sim_configs: dict[int, SimConfig] = {}

    def _sim_config(self, capital_krw: int) -> SimConfig:
        """Weight-space screening config at ``capital_krw`` (cached per capital).

        ``primary_capital_krw`` is only the default: a run evaluated at another capital must diagnose that
        capital too, or the fast-simulator gap would describe a different book.
        """
        capital = int(capital_krw)
        if capital <= 0:
            raise ValueError(f"capital_krw must be positive, got {capital_krw!r}")
        config = self._sim_configs.get(capital)
        if config is None:
            base = SimConfig.from_engine_toml(Path(self._ctx.engine_config_path), capital_krw=capital)
            config = base.model_copy(update={"halted_exit_value": 0.0})
            self._sim_configs[capital] = config
        return config

    def _hedge_series_by_row(
        self, values: NDArray[np.float64], sessions: Sequence[date]
    ) -> NDArray[np.float64]:
        """One hedge series re-posed onto the cube timeline (NaN where the session is absent)."""
        inputs = self._ctx.hedge_inputs
        series = np.asarray(values, dtype=np.float64)
        if series.size != len(inputs.sessions):
            raise PITDataError("hedge series length differs from its sessions")
        position = {day: pos for pos, day in enumerate(inputs.sessions)}
        out = np.full(len(sessions), math.nan, dtype=np.float64)
        cube_rows: list[int] = []
        hedge_rows: list[int] = []
        for cube_row, day in enumerate(sessions):
            hit = position.get(day)
            if hit is not None:
                cube_rows.append(cube_row)
                hedge_rows.append(int(hit))
        if cube_rows:
            out[cube_rows] = series[hedge_rows]
        return out

    def _data_end_row(self, *, lo: int, hi: int, sessions: Sequence[date]) -> int:
        """Last cube row whose cash return and futures-underlying level are both usable.

        The cash and hedge series may lag the cube by a few sessions, so the window ends where every input
        exists. Any earlier hole inside the window is an internal gap and fails here — before the scorer runs —
        instead of surfacing as a non-deterministic account much later. The index level is needed from
        ``lo - 1`` (the overlay marks against the previous close) while the engine first reads the cash return
        at ``lo``, so the two are checked over their own ranges.

        Raises PITDataError: no usable session, a misaligned cash series, or an internal gap (the first such
        session is named, together with any inverse-close gap inside the window).
        """
        if self._ctx.cash_returns is None:
            raise PITDataError("evaluate requires the cash return series")
        cash = np.asarray(self._ctx.cash_returns, dtype=np.float64)
        if cash.shape != (len(sessions),):
            raise PITDataError(
                f"cash returns {cash.shape} are not aligned to the {len(sessions)} cube sessions"
            )
        levels = self._hedge_series_by_row(self._ctx.hedge_inputs.index_level, sessions)
        inverse = self._hedge_series_by_row(self._ctx.hedge_inputs.inverse_close, sessions)
        usable = np.isfinite(cash) & np.isfinite(levels) & (levels > 0.0)
        hits = np.flatnonzero(usable[lo : hi + 1])
        if hits.size == 0:
            raise PITDataError(
                f"no session in [{sessions[lo]}, {sessions[hi]}] has both a cash return and an index level"
            )
        data_end = lo + int(hits[-1])
        inverse_gaps = [row for row in range(lo, data_end + 1) if not np.isfinite(inverse[row])]
        note = f"; the inverse close is also missing at {sessions[inverse_gaps[0]].isoformat()}" if inverse_gaps else ""
        for row in range(max(lo - 1, 0), data_end + 1):
            if not (math.isfinite(float(levels[row])) and float(levels[row]) > 0.0):
                raise PITDataError(
                    f"index level is missing at {sessions[row].isoformat()} inside the run window{note}"
                )
        for row in range(lo, data_end + 1):
            if not math.isfinite(float(cash[row])):
                raise PITDataError(
                    f"cash return is missing at {sessions[row].isoformat()} inside the run window{note}"
                )
        return data_end

    def _window_rows(self, spec: StrategySpec) -> tuple[int, int, WindowAuthorization]:
        """Resolve ``(lo, hi, authorization)`` of the evaluation window: the first session of
        ``scorer.first_test_year`` through the data end.

        Raises ValueError when the first test year has no sessions, PITDataError when the inputs cannot
        cover the window, WindowError when the window leaves certified data.
        """
        ctx = self._ctx
        sessions = _sessions_of(ctx.cube)
        fty = int(spec.scorer.first_test_year)
        firsts = [day for day in sessions if day.year == fty]
        if not firsts:
            raise ValueError(f"first_test_year {fty} has no sessions in the cube")
        lo = sessions.index(firsts[0])
        hi = self._data_end_row(lo=lo, hi=len(sessions) - 1, sessions=sessions)
        auth = ctx.guard.authorize(start=sessions[lo], end=sessions[hi])
        return lo, hi, auth

    def panel_for(self, last_row: int) -> FeaturePanel:
        n = len(self._ctx.cube.sessions)
        if isinstance(last_row, bool) or not isinstance(last_row, int) or not 0 <= last_row < n:
            raise ValueError(f"last_row {last_row!r} outside [0, {n - 1}]")
        return build_panel(self._ctx.cube, last_row=last_row)

    def scores(self, spec: StrategySpec, *, panel: FeaturePanel | None = None) -> ScoreMatrix:
        """Walk-forward scores from the first test year through the data-end year.

        ``panel`` lets a caller that already holds ``panel_for(hi)`` reuse it on a cache miss instead of building
        a second full panel (≈2 GB resident, ≈5 GB transient at the full universe). When it is given, its
        ``last_row`` must equal the window's data-end row.

        Raises: ValueError when ``panel.last_row`` differs from the data-end row; otherwise as before.
        """
        ctx = self._ctx
        sessions = _sessions_of(ctx.cube)
        _, hi, auth = self._window_rows(spec)
        if panel is not None and panel.last_row != hi:
            raise ValueError(f"panel last_row {panel.last_row} differs from the data-end row {hi}")
        test_years = list(range(int(spec.scorer.first_test_year), sessions[hi].year + 1))
        identity = _scores_identity(spec)
        cached = self._load_scores_cache(
            cube_id=ctx.cube.cube_id,
            identity=identity,
            scorer_hash=spec.scorer.config_hash,
            test_years=tuple(test_years),
            last_row=hi,
        )
        if cached is not None:
            return cached
        if panel is None:
            panel = self.panel_for(hi)
        uni_full = np.asarray(universe_mask(ctx.cube, spec.policy.universe), dtype=bool)
        universe = np.ascontiguousarray(uni_full[: hi + 1])
        result = walk_forward_scores(
            panel,
            universe,
            sessions,
            spec.scorer,
            test_years=test_years,
            authorization=auth,
        )
        self._save_scores_cache(result, identity=identity, test_years=tuple(test_years))
        return result

    def _stock_targets(
        self,
        *,
        spec: StrategySpec,
        cube: ResearchCube,
        panel: FeaturePanel,
        scores: NDArray[np.float64],
        universe: NDArray[np.bool_],
        sessions: Sequence[date],
        lo: int,
        hi: int,
        capital: int,
    ) -> dict[int, NDArray[np.float64]]:
        """Combined 5-sleeve stock-book targets at ``capital``, executable at rows ``[lo - 1, hi - 1]``.

        ``cube`` may be a corrupted copy: only ``arrays["close"]`` is read, for the ``min_units_per_slot``
        affordability filter.
        """
        buffer = float(self._sim_config(capital).cash_buffer)
        view = SimpleNamespace(arrays={"close": np.asarray(cube.arrays["close"], dtype=np.float64)[: hi + 1]})
        matrix = ScoreMatrix(
            scores=np.ascontiguousarray(np.asarray(scores, dtype=np.float64), dtype=np.float32),
            test_years=(),
            config_hash="pipeline",
            last_row=hi,
        )
        sleeves = build_sleeve_targets(
            spec.policy,
            view,  # type: ignore[arg-type]
            panel,
            matrix,
            np.asarray(universe, dtype=bool),
            sessions=sessions,
            sleeves=int(spec.book.sleeves),
            lo=lo,
            hi=hi + 1,
            sleeve_capital_krw=sleeve_capital_krw(capital, spec.book),
            cash_buffer=buffer,
        )
        combined = combine_sleeve_targets(sleeves, lo=max(lo, 1), hi=hi + 1)
        return {row: weights for row, weights in combined.items() if lo - 1 <= row <= hi - 1}

    def evaluate(self, spec: StrategySpec, *, capital_krw: int | None = None) -> EvaluationRun:
        """Evaluate one spec at ``capital_krw`` (default ``protocol.primary_capital_krw``) on the account engine
        over [first session of scorer.first_test_year, data end] and write
        ``<reports_root>/<spec_hash>_<run_id>.json``.

        The data end is the last cube session on which the cash return and the futures-underlying level are
        both finite: those series may lag the cube by a few sessions and the window ends where every input
        exists. A hole earlier than that is an internal gap and raises before any scoring.

        Runs (all through ``ctx.ledger_runner`` with ``cash_returns``, the beta-neutral overlay at offset 0,
        and combined 5-sleeve stock targets): base; stress_slippage (extra stock slippage + hedge extra cost);
        stress_delay (stock targets and overlay beta delayed by ``stress_delay_sessions``); one run per
        ``cost_grid_ticks``; unhedged (hedge_ratio 0); placebo (same policy on seeded uniform scores over the
        universe). Plus the account-level causal perturbation test, the index and equal-weight universe
        benchmarks and the weight-simulator stock-only growth — every one of them at ``capital_krw``.

        Raises: WindowError, PITDataError, ValueError (propagated, never swallowed).
        """
        ctx = self._ctx
        protocol = ctx.protocol
        spy = int(protocol.sessions_per_year)
        capital = int(capital_krw) if capital_krw is not None else int(protocol.primary_capital_krw)
        if capital <= 0:
            raise ValueError(f"capital_krw must be positive, got {capital_krw!r}")
        sessions = _sessions_of(ctx.cube)
        lo, hi, auth = self._window_rows(spec)
        start, end = sessions[lo], sessions[hi]
        window_sessions = tuple(sessions[lo : hi + 1])
        scenarios = protocol.scenarios
        engine_bytes = Path(ctx.engine_config_path).read_bytes()
        engine_hash = hashlib.sha256(engine_bytes).hexdigest()
        run_id = _evaluation_run_id(
            spec=spec,
            capital_krw=capital,
            protocol_hash=protocol.content_hash,
            cube_id=ctx.cube.cube_id,
            dataset_ids=ctx.dataset_ids,
            start=start,
            end=end,
            engine_config_bytes=engine_bytes,
        )
        policy = EvaluationPolicy.model_validate(protocol.evaluation.model_dump())
        market_arrays = load_market_arrays(
            panel_dir=Path(ctx.panel_dir), cache_root=Path(ctx.market_cache_root)
        )
        _log_memory_phase("inputs")

        panel = self.panel_for(hi)
        _log_memory_phase("panel")
        base_scores = np.asarray(self.scores(spec, panel=panel).scores, dtype=np.float64)
        _log_memory_phase("scores")
        uni_full = np.asarray(universe_mask(ctx.cube, spec.policy.universe), dtype=bool)
        uni = np.ascontiguousarray(uni_full[: hi + 1])
        close_full = np.asarray(ctx.cube.arrays["close"], dtype=np.float64)[: hi + 1]

        executable = self._stock_targets(
            spec=spec,
            cube=ctx.cube,
            panel=panel,
            scores=base_scores,
            universe=uni,
            sessions=sessions,
            lo=lo,
            hi=hi,
            capital=capital,
        )
        delay_n = int(scenarios.stress_delay_sessions)
        delayed = {row + delay_n: w for row, w in executable.items() if row + delay_n <= hi - 1}

        placebo_scores = np.where(
            np.asarray(uni, dtype=bool),
            np.random.default_rng(int(scenarios.placebo_seed)).random(base_scores.shape),
            np.nan,
        )
        placebo_executable = self._stock_targets(
            spec=spec,
            cube=ctx.cube,
            panel=panel,
            scores=placebo_scores,
            universe=uni,
            sessions=sessions,
            lo=lo,
            hi=hi,
            capital=capital,
        )
        del panel
        _log_memory_phase("targets")

        overlay_market = overlay_market_from_inputs(ctx.hedge_inputs)
        derivatives = derivative_config(spec.hedge)
        stressed_derivatives = derivative_config(
            spec.hedge.model_copy(
                update={
                    "futures_cost_rate": float(spec.hedge.futures_cost_rate)
                    + float(scenarios.stress_hedge_extra_cost),
                    "inverse_cost_rate": float(spec.hedge.inverse_cost_rate)
                    + float(scenarios.stress_hedge_extra_cost),
                }
            )
        )
        unhedged_derivatives = derivative_config(spec.hedge.model_copy(update={"hedge_ratio": 0.0}))

        scenario_names = (
            ["base", "stress_slippage", "stress_delay"]
            + [f"cost_{tick}" for tick in scenarios.cost_grid_ticks]
            + ["unhedged", "placebo"]
        )
        n_runs = len(scenario_names)
        outcomes: dict[str, LedgerOutcome] = {}
        base_recorder: _RecordingOverlay | None = None
        for pos, name in enumerate(scenario_names):
            _LOG.info("[ALGO] evaluate scenario=%s %d/%d", name, pos + 1, n_runs)
            overlay: Any
            slip_ticks: float | None
            if name == "stress_delay":
                targets = delayed
                overlay = BetaNeutralOverlay(spec.hedge, rebalance_offset=0, execution_delay=delay_n)
                outcome_derivatives = derivatives
                extra = 0.0
                slip_ticks = None
            elif name == "stress_slippage":
                targets = executable
                overlay = BetaNeutralOverlay(spec.hedge, rebalance_offset=0)
                outcome_derivatives = stressed_derivatives
                extra = float(scenarios.stress_extra_slippage)
                slip_ticks = None
            elif name.startswith("cost_"):
                targets = executable
                overlay = BetaNeutralOverlay(spec.hedge, rebalance_offset=0)
                outcome_derivatives = derivatives
                extra = 0.0
                slip_ticks = float(name[len("cost_") :])
            elif name == "unhedged":
                targets = executable
                overlay = BetaNeutralOverlay(
                    spec.hedge.model_copy(update={"hedge_ratio": 0.0}), rebalance_offset=0
                )
                outcome_derivatives = unhedged_derivatives
                extra = 0.0
                slip_ticks = None
            elif name == "placebo":
                targets = placebo_executable
                overlay = BetaNeutralOverlay(spec.hedge, rebalance_offset=0)
                outcome_derivatives = derivatives
                extra = 0.0
                slip_ticks = None
            else:
                targets = executable
                outcome_derivatives = derivatives
                extra = 0.0
                slip_ticks = None
                base_recorder = _RecordingOverlay(BetaNeutralOverlay(spec.hedge, rebalance_offset=0))
                overlay = base_recorder
            outcomes[name] = ctx.ledger_runner(
                cube=ctx.cube,
                targets=targets,
                panel_dir=Path(ctx.panel_dir),
                dividends=ctx.dividends,
                engine_config_path=Path(ctx.engine_config_path),
                rules=ctx.rules,
                market_cache_root=Path(ctx.market_cache_root),
                capital_krw=capital,
                halted_exit_policy=DelistPolicy.ZERO,
                start=start,
                end=end,
                authorization=auth,
                cash_returns=ctx.cash_returns,
                overlay=overlay,
                overlay_market=overlay_market,
                derivatives=outcome_derivatives,
                extra_slippage=extra,
                auction_slippage_ticks=slip_ticks,
                sessions_per_year=spy,
                market_arrays=market_arrays,
                rebalance_band=float(spec.book.rebalance_band),
            )
        _log_memory_phase("scenarios")

        index_log_returns = self._index_log_returns(window_sessions)
        universe_ew = self._universe_ew_log_returns(lo, hi, uni_full, close_full)
        float_result = simulate(
            ctx.cube, executable, start=start, end=end, config=self._sim_config(capital), authorization=auth,
            rebalance_band=float(spec.book.rebalance_band),
        )
        fast_sim_growth = float(
            annualized_log_growth(np.asarray(float_result.log_returns), sessions_per_year=spy)
        )
        mismatches = self._perturbation_mismatches(
            spec,
            base_scores=base_scores,
            universe=uni,
            executable=executable,
            clean_decisions=tuple(base_recorder.decisions) if base_recorder is not None else (),
            clean_nav=_nav_series(outcomes["base"]),
            market_arrays=market_arrays,
            sessions=sessions,
            lo=lo,
            hi=hi,
            capital=capital,
            derivatives=derivatives,
            auth=auth,
        )
        evidence = EvaluationEvidence(
            sessions=window_sessions,
            base=outcomes["base"],
            stress_slippage=outcomes["stress_slippage"],
            stress_delay=outcomes["stress_delay"],
            cost_grid={
                float(tick): outcomes[f"cost_{tick}"] for tick in scenarios.cost_grid_ticks
            },
            unhedged=outcomes["unhedged"],
            placebo=outcomes["placebo"],
            index_log_returns=index_log_returns,
            universe_ew_log_returns=universe_ew,
            fast_sim_growth=fast_sim_growth,
            perturbation_mismatches=int(mismatches),
        )
        report = build_report_card(
            evidence, policy, spec_hash=spec.spec_hash, run_id=run_id, protocol_version=protocol.version
        )
        for name in scenario_names:
            outcome = outcomes[name]
            self._ctx.registry.record(
                family=spec.policy.family,
                spec_hash=spec.spec_hash,
                spec_json=spec.canonical_json(),
                run_id=run_id,
                scenario=name,
                capital_krw=capital,
                sim_config_json=json.dumps(
                    {"scenario": name, "engine_config_hash": engine_hash}, sort_keys=True
                ),
                cube_id=ctx.cube.cube_id,
                protocol_hash=protocol.content_hash,
                engine_config_hash=engine_hash,
                report_digest=report.digest,
                returns=RunReturns(
                    sessions=tuple(outcome.sessions),
                    net=np.ascontiguousarray(np.asarray(outcome.log_returns, dtype=np.float64)),
                    benchmarks={},
                ),
                metrics={"objective_j": float(report.objective_j)},
                now=ctx.now(),
            )
        self._write_report(spec.spec_hash, run_id, report)
        _log_memory_phase("done")
        _LOG.info(
            "[ALGO] evaluate done spec=%s run=%s J=%.4f g=%.4f mdd=%.4f passed=%s",
            spec.spec_hash[:12],
            run_id,
            float(report.objective_j),
            float(report.metrics["g"]),
            float(report.metrics["mdd"]),
            bool(report.passed),
        )
        return EvaluationRun(report=report, evidence=evidence, spec_json=spec.canonical_json())

    def _index_log_returns(self, window_sessions: tuple[date, ...]) -> NDArray[np.float64]:
        """Daily log returns of the futures underlying over the window.

        The first window session has no in-window predecessor, so it uses the session before the window;
        when even that is absent the first element is 0.0 (the benchmark starts flat, not silently wrong).
        """
        inputs = self._ctx.hedge_inputs
        days = list(inputs.sessions)
        position = {day: pos for pos, day in enumerate(days)}
        levels = np.asarray(inputs.index_level, dtype=np.float64)
        if levels.size != len(days):
            raise PITDataError("hedge index level length differs from its sessions")
        out = np.zeros(len(window_sessions), dtype=np.float64)
        hits: list[int] = []
        for day in window_sessions:
            hit = position.get(day)
            if hit is None:
                raise PITDataError(f"index level missing at {day.isoformat()}")
            hits.append(int(hit))
        for k, day in enumerate(window_sessions):
            prev = hits[k] - 1 if k == 0 else hits[k - 1]
            if prev < 0:
                out[k] = 0.0
                continue
            current, before = float(levels[hits[k]]), float(levels[prev])
            if not math.isfinite(current) or not math.isfinite(before) or current <= 0.0 or before <= 0.0:
                raise PITDataError(f"index level missing or non-positive at {day.isoformat()}")
            out[k] = math.log(current / before)
        return np.ascontiguousarray(out, dtype=np.float64)

    def _universe_ew_log_returns(
        self,
        lo: int,
        hi: int,
        universe_full: NDArray[np.bool_],
        close_full: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Frictionless equal-weight universe benchmark; delistings ignored (optimistic diagnostic)."""
        uni = np.asarray(universe_full, dtype=bool)
        close = np.asarray(close_full, dtype=np.float64)
        out = np.zeros(hi - lo + 1, dtype=np.float64)
        for k in range(lo, hi + 1):
            if k < 1:
                out[k - lo] = 0.0
                continue
            members = uni[k - 1]
            if not bool(np.any(members)):
                out[k - lo] = 0.0
                continue
            today = close[k][members]
            yesterday = close[k - 1][members]
            valid = (
                np.isfinite(today)
                & np.isfinite(yesterday)
                & (yesterday > 0.0)
                & (today > 0.0)
            )
            if not bool(np.any(valid)):
                out[k - lo] = 0.0
                continue
            out[k - lo] = math.log1p(float(np.mean(today[valid] / yesterday[valid] - 1.0)))
        return np.ascontiguousarray(out, dtype=np.float64)

    def _perturbation_cuts(
        self, *, sessions: Sequence[date], lo: int, hi: int, spec: StrategySpec
    ) -> tuple[int, ...]:
        """Cut rows for the causal perturbation: rows inside a test year that has a preceding test year.

        The preceding year is what guarantees the rescore of the cut's year still sees uncorrupted training
        rows. A window holding a single test year has no such row and is left unperturbed.
        """
        protocol = self._ctx.protocol
        wanted = int(protocol.scenarios.perturbation_cuts)
        first_year = int(spec.scorer.first_test_year)
        test_years = set(range(first_year, sessions[hi].year + 1))
        pool = [
            row
            for row in range(lo, hi)
            if sessions[row].year in test_years and (sessions[row].year - 1) in test_years
        ]
        if not pool:
            return ()
        rng = np.random.default_rng(int(protocol.scenarios.perturbation_seed))
        drawn = rng.choice(len(pool), size=min(wanted, len(pool)), replace=False)
        return tuple(sorted(pool[int(pos)] for pos in np.atleast_1d(drawn)))

    def _perturb_cut(
        self,
        spec: StrategySpec,
        *,
        position: int,
        cut: int,
        base_scores: NDArray[np.float64],
        executable: Mapping[int, NDArray[np.float64]],
        clean_decisions: Sequence[_OverlayDecision],
        clean_nav: NDArray[np.float64],
        market_arrays: MarketArrays,
        sessions: Sequence[date],
        lo: int,
        hi: int,
        capital: int,
        derivatives: Any,
        auth: WindowAuthorization,
    ) -> int:
        """Mismatch count of one causal-perturbation cut (see ``_perturbation_mismatches``).

        Why a separate method: every corrupted input of a cut (cube, panel, ``MarketArrays``, dividends, hedge and
        cash series, ledger outcome) is a full-size copy. Ending their lifetime at this method's return means two
        cuts never coexist in memory. Within the cut, the corrupted ``MarketArrays`` are created only after the
        perturbed panel and its score matrix are released, so the engine copy never coexists with a panel build.
        """
        ctx = self._ctx
        spy = int(ctx.protocol.sessions_per_year)
        seed = int(ctx.protocol.scenarios.perturbation_seed) + position
        rng = np.random.default_rng(seed)
        corrupted_cube = self._corrupt_cube(ctx.cube, cut, seed)
        cut_year = sessions[cut].year
        perturbed_panel = build_panel(corrupted_cube, last_row=hi)
        perturbed_universe = np.ascontiguousarray(
            np.asarray(universe_mask(corrupted_cube, spec.policy.universe), dtype=bool)[: hi + 1]
        )
        rescored = walk_forward_scores(
            perturbed_panel,
            perturbed_universe,
            sessions,
            spec.scorer,
            test_years=[cut_year],
            authorization=auth,
        )
        merged = np.asarray(base_scores, dtype=np.float64).copy()
        year_rows = [row for row in range(hi + 1) if sessions[row].year == cut_year]
        merged[year_rows] = np.asarray(rescored.scores, dtype=np.float64)[year_rows]
        perturbed_targets = self._stock_targets(
            spec=spec,
            cube=corrupted_cube,
            panel=perturbed_panel,
            scores=merged,
            universe=perturbed_universe,
            sessions=sessions,
            lo=lo,
            hi=hi,
            capital=capital,
        )
        del perturbed_panel, rescored
        corrupted_inputs = _corrupt_hedge_inputs(ctx.hedge_inputs, cut, sessions=sessions, rng=rng)
        assert ctx.cash_returns is not None  # guaranteed by _data_end_row
        corrupted_cash = _corrupt_cash_returns(ctx.cash_returns, cut, rng)
        cut_session = sessions[cut]
        corrupted_arrays = _corrupt_market_arrays(market_arrays, cut, rng)
        corrupted_dividends = _corrupt_dividends(ctx.dividends, cut_session, rng)
        recorder = _RecordingOverlay(BetaNeutralOverlay(spec.hedge, rebalance_offset=0))
        rerun = ctx.ledger_runner(
            cube=corrupted_cube,
            targets=perturbed_targets,
            panel_dir=Path(ctx.panel_dir),
            dividends=corrupted_dividends,
            engine_config_path=Path(ctx.engine_config_path),
            rules=ctx.rules,
            market_cache_root=Path(ctx.market_cache_root),
            capital_krw=capital,
            halted_exit_policy=DelistPolicy.ZERO,
            start=sessions[lo],
            end=sessions[hi],
            authorization=auth,
            cash_returns=corrupted_cash,
            overlay=recorder,
            overlay_market=overlay_market_from_inputs(corrupted_inputs),
            derivatives=derivatives,
            extra_slippage=0.0,
            auction_slippage_ticks=None,
            sessions_per_year=spy,
            market_arrays=corrupted_arrays,
            rebalance_band=float(spec.book.rebalance_band),
        )
        total = 0
        for row, weights in executable.items():
            if row > cut:
                continue
            other = perturbed_targets.get(row)
            if other is None:  # pragma: no cover - decision rows do not depend on the corruption
                total += int(np.asarray(weights).size)
                continue
            total += int(
                np.count_nonzero(
                    np.abs(np.asarray(weights, dtype=np.float64) - np.asarray(other, dtype=np.float64)) > 1e-12
                )
            )
        total += _decision_mismatches(clean_decisions, recorder.decisions, cut)
        stop = cut - lo + 1
        clean_tail = np.asarray(clean_nav, dtype=np.float64)[0:stop]
        rerun_tail = _nav_series(rerun)[0:stop]
        width = min(int(clean_tail.shape[0]), int(rerun_tail.shape[0]))
        total += abs(int(clean_tail.shape[0]) - int(rerun_tail.shape[0]))
        total += int(np.count_nonzero(clean_tail[:width] != rerun_tail[:width]))
        return int(total)

    def _perturbation_mismatches(
        self,
        spec: StrategySpec,
        *,
        base_scores: NDArray[np.float64],
        universe: NDArray[np.bool_],
        executable: Mapping[int, NDArray[np.float64]],
        clean_decisions: Sequence[_OverlayDecision],
        clean_nav: NDArray[np.float64],
        market_arrays: MarketArrays,
        sessions: Sequence[date],
        lo: int,
        hi: int,
        capital: int,
        derivatives: Any,
        auth: WindowAuthorization,
    ) -> int:
        """Account-level causal check: corrupt every input after each cut row, then require rows ``<= cut``
        to be identical in the stock targets, the overlay decisions and the account's NAV records.

        Every input the evaluated account reads is corrupted: the cube arrays (which feed the scorer and the
        ``min_units_per_slot`` filter), the engine's own ``MarketArrays``, the dividend events with an ex-date
        after the cut, the index level, the inverse close and the cash return. A rescoring or replay error
        propagates — the integrity check must never pass vacuously.
        """
        cuts = self._perturbation_cuts(sessions=sessions, lo=lo, hi=hi, spec=spec)
        total = 0
        for position, cut in enumerate(cuts):
            _LOG.info("[ALGO] perturbation cut=%d/%d row=%d", position + 1, len(cuts), cut)
            total += self._perturb_cut(
                spec,
                position=position,
                cut=cut,
                base_scores=base_scores,
                executable=executable,
                clean_decisions=clean_decisions,
                clean_nav=clean_nav,
                market_arrays=market_arrays,
                sessions=sessions,
                lo=lo,
                hi=hi,
                capital=capital,
                derivatives=derivatives,
                auth=auth,
            )
            _log_memory_phase(f"cut{position}")
        return int(total)


    @staticmethod
    def _corrupt_cube(cube: ResearchCube, cut: int, seed: int) -> ResearchCube:
        """A cube equal to ``cube`` at rows ``<= cut`` and randomly corrupted at rows ``> cut``, built lazily.

        Each array is corrupted on first access and cached. A memory-mapped source is reopened copy-on-write
        (``mmap_mode="c"``), so only the pages of the corrupted rows become private memory; an in-memory
        source is copied. Arrays nobody reads are never materialised.

        Why lazy: the perturbation must corrupt every input the account can read, but the evaluate path reads
        only part of the cube. An unread array has no consumer to leak through, and if a future consumer does
        read it, it is corrupted on that access, so the check stays complete without paying for every array.

        The corruption of array ``name`` depends only on ``(seed, cut, crc32(name))``, never on access order,
        so the corrupted cube is deterministic whichever consumer touches it first. By dtype: floats
        ``x * (1 + 0.5 * N(0, 1))``; bools xor ``U < 0.5``; integers ``+ U{-2..2}``. Exit arrays are copied
        unchanged.

        Raises:
            ValueError: ``cut`` outside ``[0, S - 1)``.
        """
        n_s = len(cube.sessions)
        if not 0 <= cut < n_s - 1:
            raise ValueError(f"perturbation cut {cut} outside [0, {n_s - 1})")
        # Built directly rather than through ``from_arrays``, which would iterate and materialise every array.
        exit_at = np.array(cube.exit_at, dtype=np.int64, copy=True)
        exit_halted = np.array(cube.exit_halted, dtype=np.bool_, copy=True)
        exit_at.flags.writeable = False
        exit_halted.flags.writeable = False
        return ResearchCube(
            cube_id=cube.cube_id,
            sessions=cube.sessions,
            instrument_ids=cube.instrument_ids,
            arrays=_LazyCorruptCubeArrays(cube.arrays, cut, seed),
            exit_at=exit_at,
            exit_halted=exit_halted,
        )

    def _scores_cache_path(self, cube_id: str, identity: str, test_years: tuple[int, ...], last_row: int) -> Path:
        years = "-".join(str(y) for y in test_years)
        name = f"{cube_id}__{identity[:16]}__{years}__{last_row}.npz"
        return Path(self._ctx.scores_root) / name

    def _load_scores_cache(
        self, *, cube_id: str, identity: str, scorer_hash: str, test_years: tuple[int, ...], last_row: int
    ) -> ScoreMatrix | None:
        path = self._scores_cache_path(cube_id, identity, test_years, last_row)
        if not path.is_file():
            return None
        try:
            with np.load(str(path), allow_pickle=False) as store:
                ok = (
                    str(store["cube_id"]) == cube_id
                    and str(store["config_hash"]) == identity
                    and tuple(int(v) for v in store["test_years"].tolist()) == tuple(test_years)
                    and int(store["last_row"]) == int(last_row)
                )
                if not ok:
                    return None
                scores = np.ascontiguousarray(store["scores"])
        except Exception:  # noqa: BLE001 - corrupt cache is ignored
            return None
        arr = np.ascontiguousarray(scores, dtype=np.float32)
        arr.flags.writeable = False
        return ScoreMatrix(scores=arr, test_years=tuple(test_years), config_hash=scorer_hash, last_row=last_row)

    def _save_scores_cache(self, matrix: ScoreMatrix, *, identity: str, test_years: tuple[int, ...]) -> None:
        root = Path(self._ctx.scores_root)
        root.mkdir(parents=True, exist_ok=True)
        path = self._scores_cache_path(self._ctx.cube.cube_id, identity, test_years, matrix.last_row)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            cube_id=np.array(self._ctx.cube.cube_id),
            config_hash=np.array(identity),
            test_years=np.asarray(list(test_years), dtype=np.int64),
            last_row=np.asarray(int(matrix.last_row), dtype=np.int64),
            scores=np.ascontiguousarray(matrix.scores),
        )
        tmp.replace(path)

    def _write_report(self, spec_hash: str, run_id: str, report: ReportCard) -> Path:
        root = Path(self._ctx.reports_root)
        root.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = json.loads(report.canonical_json())
        payload["digest"] = report.digest
        path = root / f"{spec_hash}_{run_id}.json"
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return path
