"""ML trend-cash evaluation pipeline over a sealed research cube."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from src.backtest.engine import DelistPolicy
from src.core.pit import PITDataError
from src.data.research_protocol import (
    FinalistRecord,
    LockboxAuthorization,
    LockboxError,
    ResearchProtocol,
    Segment,
)
from src.research.criteria import (
    CriteriaReport,
    DiscoveryEvidence,
)
from src.research.criteria import (
    evaluate_discovery as _evaluate_discovery_criteria,
)
from src.research.criteria import (
    evaluate_holdout as _evaluate_holdout_criteria,
)
from src.research.cube import ResearchCube
from src.research.ledger_bridge import LedgerOutcome
from src.research.model import ScoreMatrix, ScorerConfig, walk_forward_scores
from src.research.panel import FEATURE_NAMES, HORIZONS, FeaturePanel, build_panel
from src.research.policy import TrendCashPolicy, build_targets, decision_rows, universe_mask
from src.research.registry import TrialRegistry, TrialReturns
from src.research.simulator import SimConfig, simulate
from src.research.stats import (
    annualized_log_growth,
    effective_trial_count,
    point_metrics,
)

__all__ = [
    "PRODUCTION_PHASE",
    "Pipeline",
    "PipelineContext",
    "StrategySpec",
    "load_strategy_spec",
]

PRODUCTION_PHASE: Final = 0

_LOG = logging.getLogger(__name__)


class StrategySpec(BaseModel):
    """One pre-registered strategy: selection policy plus frozen scorer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: TrendCashPolicy
    scorer: ScorerConfig

    def canonical_json(self) -> str:
        payload = {
            "feature_names": list(FEATURE_NAMES),
            "horizons": list(HORIZONS),
            "policy": json.loads(self.policy.canonical_json()),
            "scorer": json.loads(self.scorer.canonical_json()),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_strategy_spec(path: Path) -> StrategySpec:
    """Load a TOML with ``[policy]`` and ``[scorer]`` tables."""
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
    allowed = {"policy", "scorer"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown strategy keys: {sorted(unknown)}")
    if "policy" not in raw or "scorer" not in raw:
        raise ValueError("strategy TOML must declare [policy] and [scorer] tables")
    try:
        policy = TrendCashPolicy.model_validate(raw["policy"])
    except Exception as exc:
        raise ValueError(f"invalid [policy] table: {exc}") from exc
    try:
        scorer = ScorerConfig.model_validate(raw["scorer"])
    except Exception as exc:
        raise ValueError(f"invalid [scorer] table: {exc}") from exc
    return StrategySpec(policy=policy, scorer=scorer)


@dataclass(frozen=True, slots=True)
class PipelineContext:
    protocol: ResearchProtocol
    cube: ResearchCube
    registry: TrialRegistry
    lockbox: Any
    panel_dir: Path
    dividends: pl.DataFrame
    rules: Any
    engine_config_path: Path
    market_cache_root: Path
    reports_root: Path
    scores_root: Path
    ledger_runner: Callable[..., LedgerOutcome]
    now: Callable[[], datetime]


def _sessions_of(cube: ResearchCube) -> list[date]:
    return list(cube.sessions)


def _window_indices(sessions: Sequence[date], start: date, end: date) -> tuple[int, int]:
    index_of = {day: idx for idx, day in enumerate(sessions)}
    if start not in index_of or end not in index_of or index_of[start] > index_of[end]:
        raise ValueError(f"run window [{start}, {end}] is not within the cube sessions")
    return (index_of[start], index_of[end])


def _first_at_or_after(sessions: Sequence[date], day: date) -> date:
    for session in sessions:
        if session >= day:
            return session
    raise ValueError(f"window start {day} is after the last cube session")


def _last_at_or_before(sessions: Sequence[date], day: date) -> date:
    for session in reversed(list(sessions)):
        if session <= day:
            return session
    raise ValueError(f"window end {day} is before the first cube session")


def _annualized_mean(per_session: NDArray[np.float64], sessions_per_year: int) -> float:
    arr = np.asarray(per_session, dtype=np.float64)
    return float(np.mean(arr)) * int(sessions_per_year) if arr.size else 0.0


def _trial_metrics(
    log_returns: NDArray[np.float64],
    turnover: NDArray[np.float64],
    cost: NDArray[np.float64],
    sessions_per_year: int,
) -> dict[str, float]:
    values = np.asarray(log_returns, dtype=np.float64)
    pm = point_metrics(values, sessions_per_year=sessions_per_year)
    return {
        "cagr": float(pm.cagr),
        "mdd": float(pm.max_drawdown),
        "calmar": float(pm.calmar),
        "sharpe": float(pm.sharpe),
        "turnover": float(_annualized_mean(turnover, sessions_per_year)),
        "cost": float(_annualized_mean(cost, sessions_per_year)),
    }


def _sim_json_for_trial(config: SimConfig, phase: int) -> str:
    payload = {"phase": int(phase), "sim_config": json.loads(config.canonical_json())}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _scores_identity(spec: StrategySpec) -> str:
    """Cache identity of a score matrix: the scorer config plus the universe it trains and predicts on."""
    payload = f"{spec.scorer.config_hash}|{spec.policy.universe.model_dump_json()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _effective_from_columns(columns: list[NDArray[np.float64]]) -> float:
    if not columns:
        return 1.0
    try:
        return float(effective_trial_count(np.column_stack(columns)))
    except ValueError:
        return float(len(columns))


def _report_payload(report: CriteriaReport) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(report.canonical_json())
    return payload


def _rebuild_report(raw: Mapping[str, Any]) -> CriteriaReport:
    from src.research.criteria import CriterionCheck

    try:
        checks = tuple(
            CriterionCheck(
                criterion=str(item["criterion"]),
                name=str(item["name"]),
                value=float(item["value"]),
                threshold=float(item["threshold"]),
                passed=bool(item["passed"]),
                blocking=bool(item["blocking"]),
                detail=str(item["detail"]),
            )
            for item in raw["checks"]
        )
        return CriteriaReport(
            spec_hash=str(raw["spec_hash"]),
            trial_id=str(raw["trial_id"]),
            segment=Segment(str(raw["segment"])),
            protocol_version=str(raw["protocol_version"]),
            checks=checks,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid criteria report payload: {exc}") from exc


class Pipeline:
    """Sealed evaluation pipeline for one pre-registered ML trend-cash spec."""

    def __init__(self, context: PipelineContext) -> None:
        self._ctx = context
        self._base_config: SimConfig | None = None

    def _base_sim_config(self) -> SimConfig:
        if self._base_config is None:
            base = SimConfig.from_engine_toml(
                self._ctx.engine_config_path,
                capital_krw=self._ctx.protocol.primary_capital_krw,
            )
            self._base_config = base.model_copy(update={"halted_exit_value": 0.0})
        return self._base_config

    def panel_for(self, last_row: int) -> FeaturePanel:
        n = len(self._ctx.cube.sessions)
        if isinstance(last_row, bool) or not isinstance(last_row, int) or not 0 <= last_row < n:
            raise ValueError(f"last_row {last_row!r} outside [0, {n - 1}]")
        return build_panel(self._ctx.cube, last_row=last_row)

    def scores(self, spec: StrategySpec, *, segment: Segment) -> ScoreMatrix:
        ctx = self._ctx
        sessions = _sessions_of(ctx.cube)
        if segment is Segment.DISCOVERY:
            start, end = self._window_for_spec(spec, Segment.DISCOVERY)
            auth = ctx.lockbox.authorize(start=start, end=end, spec_hash=None)
            test_years = list(range(spec.scorer.first_test_year, end.year + 1))
        elif segment is Segment.HOLDOUT:
            start, end = self._holdout_window()
            auth = ctx.lockbox.authorize(start=start, end=end, spec_hash=spec.spec_hash)
            if not auth.evidence:
                raise LockboxError(f"holdout is not authorized for spec: {spec.spec_hash}")
            test_years = sorted({day.year for day in sessions if start <= day <= end})
        else:
            raise ValueError(f"segment {segment.value} is sealed for scoring")
        if not test_years:  # pragma: no cover - window always spans a calendar year
            raise ValueError("scoring window has no test years")
        _, hi = _window_indices(sessions, start, end)
        last_row = hi
        identity = _scores_identity(spec)
        cached = self._load_scores_cache(
            cube_id=ctx.cube.cube_id,
            identity=identity,
            scorer_hash=spec.scorer.config_hash,
            test_years=tuple(test_years),
            last_row=last_row,
        )
        if cached is not None:
            return cached
        panel = self.panel_for(last_row)
        uni_full = np.asarray(universe_mask(ctx.cube, spec.policy.universe), dtype=bool)
        universe = np.ascontiguousarray(uni_full[: last_row + 1])
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

    def evaluate_discovery(self, spec: StrategySpec) -> CriteriaReport:
        ctx = self._ctx
        sessions = _sessions_of(ctx.cube)
        protocol = ctx.protocol
        spy = protocol.sessions_per_year
        fty = int(spec.scorer.first_test_year)
        firsts = [day for day in sessions if day.year == fty]
        if not firsts:
            raise ValueError(f"first_test_year {fty} has no sessions in the cube")
        start = firsts[0]
        ends = [day for day in sessions if day <= protocol.discovery_end]
        if not ends:  # pragma: no cover - protocol discovery_end always overlaps the cube
            raise ValueError("discovery window has no cube sessions")
        end = ends[-1]
        if start > end:
            raise ValueError(f"first_test_year {fty} is after the discovery end {protocol.discovery_end}")
        auth = ctx.lockbox.authorize(start=start, end=end, spec_hash=None)
        lo_sim, hi_sim = _window_indices(sessions, start, end)
        last_row = hi_sim
        panel = self.panel_for(last_row)
        score_mat = self.scores(spec, segment=Segment.DISCOVERY)
        base_scores = np.asarray(score_mat.scores, dtype=np.float64)
        uni_full = np.asarray(universe_mask(ctx.cube, spec.policy.universe), dtype=bool)
        uni = np.ascontiguousarray(uni_full[: last_row + 1])
        close_full = np.asarray(ctx.cube.arrays["close"], dtype=np.float64)[: last_row + 1]
        base_config = self._base_sim_config()
        buffer = float(base_config.cash_buffer)
        primary = int(protocol.primary_capital_krw)
        every = int(spec.policy.rebalance_every_sessions)
        n_phases = every

        base_targets: dict[int, dict[int, NDArray[np.float64]]] = {}
        base_streams: list[NDArray[np.float64]] = []
        slip_streams: list[NDArray[np.float64]] = []
        delay_streams: list[NDArray[np.float64]] = []
        turnovers: list[NDArray[np.float64]] = []
        costs: list[NDArray[np.float64]] = []
        trial_ids: list[str] = []
        slip_extra = float(protocol.criteria.c1.stress_extra_slippage)
        delay_n = int(protocol.criteria.c1.stress_execution_delay)
        for phase in range(n_phases):
            _LOG.info("[ALGO] discovery phase=%d/%d", phase, n_phases)
            rows = decision_rows(sessions, lo=lo_sim, hi=hi_sim + 1, every=every, phase=phase)
            targets = self._targets_for(spec.policy, close_full, panel, base_scores, uni, list(rows), primary, buffer)
            base_targets[phase] = targets
            exec_base = {r: w for r, w in targets.items() if lo_sim - 1 <= int(r) <= hi_sim - 1}
            res_base = simulate(ctx.cube, exec_base, start=start, end=end, config=base_config, authorization=auth)
            base_streams.append(np.ascontiguousarray(res_base.log_returns))
            slip_cfg = base_config.model_copy(update={"extra_slippage": slip_extra})
            res_slip = simulate(ctx.cube, exec_base, start=start, end=end, config=slip_cfg, authorization=auth)
            slip_streams.append(np.ascontiguousarray(res_slip.log_returns))
            delay_cfg = base_config.model_copy(update={"execution_delay": delay_n})
            exec_delay = {r: w for r, w in targets.items() if lo_sim - 1 - delay_n <= int(r) <= hi_sim - 1 - delay_n}
            res_delay = simulate(ctx.cube, exec_delay, start=start, end=end, config=delay_cfg, authorization=auth)
            delay_streams.append(np.ascontiguousarray(res_delay.log_returns))
            metrics = _trial_metrics(res_base.log_returns, res_base.turnover, res_base.cost, spy)
            record = ctx.registry.record(
                family=spec.policy.family,
                spec_hash=spec.spec_hash,
                spec_json=spec.canonical_json(),
                segment=Segment.DISCOVERY,
                sim_config_json=_sim_json_for_trial(base_config, phase),
                cube_id=ctx.cube.cube_id,
                returns=TrialReturns(
                    sessions=res_base.sessions,
                    net=np.ascontiguousarray(res_base.log_returns),
                    benchmarks={},
                ),
                metrics=metrics,
                now=ctx.now(),
            )
            trial_ids.append(record.trial_id)
            turnovers.append(np.ascontiguousarray(res_base.turnover))
            costs.append(np.ascontiguousarray(res_base.cost))

        phase0_targets = base_targets[PRODUCTION_PHASE]
        fast_growth: dict[tuple[int, str], float] = {}
        ledger_returns: dict[tuple[int, str], NDArray[np.float64]] = {}
        for capital in protocol.criteria.c3.ledger_capitals:
            cap = int(capital)
            for policy_name, halted_val, delist in (
                ("zero", 0.0, DelistPolicy.ZERO),
                ("last_close", 1.0, DelistPolicy.LAST_CLOSE),
            ):
                match_cfg = base_config.model_copy(update={"capital_krw": cap, "halted_exit_value": halted_val})
                match_res = simulate(
                    ctx.cube, phase0_targets, start=start, end=end, config=match_cfg, authorization=auth
                )
                fast_growth[(cap, policy_name)] = float(
                    annualized_log_growth(np.asarray(match_res.log_returns), sessions_per_year=spy)
                )
                _LOG.info("[EXEC] ledger replay capital=%d policy=%s", cap, policy_name)
                outcome = ctx.ledger_runner(
                    cube=ctx.cube,
                    targets=phase0_targets,
                    panel_dir=Path(ctx.panel_dir),
                    dividends=ctx.dividends,
                    engine_config_path=Path(ctx.engine_config_path),
                    rules=ctx.rules,
                    market_cache_root=Path(ctx.market_cache_root),
                    capital_krw=cap,
                    halted_exit_policy=delist,
                    start=start,
                    end=end,
                    authorization=auth,
                )
                ledger_returns[(cap, policy_name)] = np.ascontiguousarray(outcome.log_returns)

        nocap_policy = spec.policy.model_copy(update={"min_units_per_slot": 0})
        rows0 = decision_rows(sessions, lo=lo_sim, hi=hi_sim + 1, every=every, phase=PRODUCTION_PHASE)
        nocap_targets = self._targets_for(
            nocap_policy, close_full, panel, base_scores, uni, list(rows0), primary, buffer
        )
        exec_nocap = {r: w for r, w in nocap_targets.items() if lo_sim - 1 <= int(r) <= hi_sim - 1}
        g_base = float(annualized_log_growth(base_streams[PRODUCTION_PHASE], sessions_per_year=spy))
        res_nocap = simulate(ctx.cube, exec_nocap, start=start, end=end, config=base_config, authorization=auth)
        g_nocap = float(annualized_log_growth(np.asarray(res_nocap.log_returns), sessions_per_year=spy))
        price_delta = float(g_base - g_nocap) if np.isfinite(g_base) and np.isfinite(g_nocap) else float("nan")

        mismatches = self._perturbation_mismatches(
            spec, panel, base_scores, uni, close_full, base_targets, sessions, start, end, last_row, auth
        )
        effective = self._effective_registry_trials(spec.policy.family)
        evidence = DiscoveryEvidence(
            sessions=tuple(sessions[lo_sim : hi_sim + 1]),
            base=tuple(base_streams),
            stress_slippage=tuple(slip_streams),
            stress_delay=tuple(delay_streams),
            fast_growth=dict(fast_growth),
            ledger_returns={k: np.ascontiguousarray(v) for k, v in ledger_returns.items()},
            perturbation_mismatches=int(mismatches),
            price_cap_growth_delta=float(price_delta),
            effective_registry_trials=float(effective),
        )
        report = _evaluate_discovery_criteria(
            evidence, protocol, spec_hash=spec.spec_hash, trial_id=trial_ids[PRODUCTION_PHASE]
        )
        self._write_report(spec.spec_hash, "discovery", report)
        return report

    def register_finalist(self, spec: StrategySpec) -> None:
        ctx = self._ctx
        path = Path(ctx.reports_root) / f"{spec.spec_hash}_discovery.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"missing discovery report: {spec.spec_hash}") from exc
        except ValueError as exc:
            raise ValueError(f"invalid discovery report: {spec.spec_hash}: {exc}") from exc
        if not isinstance(raw, dict) or "digest" not in raw:
            raise ValueError(f"discovery report digest mismatch: {spec.spec_hash}")
        stored = str(raw["digest"])
        body = {k: v for k, v in raw.items() if k != "digest"}
        rebuilt = _rebuild_report(body)
        if rebuilt.spec_hash != spec.spec_hash:
            raise ValueError(f"discovery report spec mismatch: {spec.spec_hash}")
        if rebuilt.digest != stored:
            raise ValueError(f"discovery report digest mismatch: {spec.spec_hash}")
        if not rebuilt.passed:
            raise ValueError(f"discovery report did not pass: {spec.spec_hash}")
        if ctx.lockbox.finalists():
            raise LockboxError("only one finalist can be registered")
        ctx.lockbox.register_finalists(
            [
                FinalistRecord(
                    spec_hash=spec.spec_hash,
                    family=spec.policy.family,
                    discovery_trial_id=rebuilt.trial_id,
                    gate_report_digest=rebuilt.digest,
                    registered_at=ctx.now(),
                )
            ]
        )

    def holdout(self, spec: StrategySpec) -> CriteriaReport:
        ctx = self._ctx
        protocol = ctx.protocol
        sessions = _sessions_of(ctx.cube)
        if not any(r.spec_hash == spec.spec_hash for r in ctx.lockbox.finalists()):
            raise LockboxError(f"holdout is sealed for spec: {spec.spec_hash}")
        start, end = self._holdout_window()
        auth = ctx.lockbox.authorize(start=start, end=end, spec_hash=spec.spec_hash)
        if not auth.evidence:  # pragma: no cover - finalist check above guards this
            raise LockboxError(f"holdout is not authorized for spec: {spec.spec_hash}")
        lo_sim, hi_sim = _window_indices(sessions, start, end)
        last_row = hi_sim
        panel = self.panel_for(last_row)
        score_mat = self.scores(spec, segment=Segment.HOLDOUT)
        scores_arr = np.asarray(score_mat.scores, dtype=np.float64)
        uni_full = np.asarray(universe_mask(ctx.cube, spec.policy.universe), dtype=bool)
        uni = np.ascontiguousarray(uni_full[: last_row + 1])
        close_full = np.asarray(ctx.cube.arrays["close"], dtype=np.float64)[: last_row + 1]
        every = int(spec.policy.rebalance_every_sessions)
        rows = decision_rows(sessions, lo=lo_sim, hi=hi_sim + 1, every=every, phase=PRODUCTION_PHASE)
        base_config = self._base_sim_config()
        buffer = float(base_config.cash_buffer)
        primary = int(protocol.primary_capital_krw)
        targets = self._targets_for(spec.policy, close_full, panel, scores_arr, uni, list(rows), primary, buffer)
        executable = {r: w for r, w in targets.items() if lo_sim - 1 <= int(r) <= hi_sim - 1}
        _LOG.info("[ALGO] holdout phase=%d rows=%d", PRODUCTION_PHASE, len(executable))
        result = simulate(ctx.cube, executable, start=start, end=end, config=base_config, authorization=auth)
        metrics = _trial_metrics(result.log_returns, result.turnover, result.cost, protocol.sessions_per_year)
        record = ctx.registry.record(
            family=spec.policy.family,
            spec_hash=spec.spec_hash,
            spec_json=spec.canonical_json(),
            segment=Segment.HOLDOUT,
            sim_config_json=_sim_json_for_trial(base_config, PRODUCTION_PHASE),
            cube_id=ctx.cube.cube_id,
            returns=TrialReturns(sessions=result.sessions, net=np.ascontiguousarray(result.log_returns), benchmarks={}),
            metrics=metrics,
            now=ctx.now(),
        )
        report = _evaluate_holdout_criteria(
            np.ascontiguousarray(result.log_returns), protocol, spec_hash=spec.spec_hash, trial_id=record.trial_id
        )
        self._write_report(spec.spec_hash, "holdout", report)
        ctx.lockbox.record_holdout_verdict(spec_hash=spec.spec_hash, passed=report.passed, report_digest=report.digest)
        return report

    def _window_for_spec(self, spec: StrategySpec, segment: Segment) -> tuple[date, date]:
        sessions = _sessions_of(self._ctx.cube)
        if segment is Segment.DISCOVERY:
            fty = int(spec.scorer.first_test_year)
            firsts = [day for day in sessions if day.year == fty]
            if not firsts:
                raise ValueError(f"first_test_year {fty} has no sessions in the cube")
            start = firsts[0]
            ends = [day for day in sessions if day <= self._ctx.protocol.discovery_end]
            if not ends:  # pragma: no cover - protocol discovery_end always overlaps the cube
                raise ValueError("discovery window has no cube sessions")
            end = ends[-1]
            if start > end:
                raise ValueError(f"first_test_year {fty} is after the discovery end")
            return (start, end)
        if segment is Segment.HOLDOUT:  # pragma: no cover - scores() resolves holdout directly
            return self._holdout_window()
        raise ValueError(f"segment {segment.value} is sealed for scoring")  # pragma: no cover - scores() guards this

    def _holdout_window(self) -> tuple[date, date]:
        sessions = _sessions_of(self._ctx.cube)
        protocol = self._ctx.protocol
        try:
            start = _first_at_or_after(sessions, protocol.holdout_start)
            end = _last_at_or_before(sessions, protocol.holdout_end)
        except ValueError as exc:  # pragma: no cover - protocol windows always overlap the cube in tests
            raise ValueError(f"holdout window has no cube sessions: {exc}") from exc
        if start > end:  # pragma: no cover - protocol windows always overlap the cube in tests
            raise ValueError("holdout window has no cube sessions")
        return (start, end)

    def _targets_for(
        self,
        policy: TrendCashPolicy,
        close_sliced: NDArray[np.float64],
        panel: FeaturePanel,
        scores_arr: NDArray[np.float64],
        universe_sliced: NDArray[np.bool_],
        rows: list[int],
        capital_krw: int,
        cash_buffer: float,
    ) -> dict[int, NDArray[np.float64]]:
        mat = np.ascontiguousarray(np.asarray(scores_arr, dtype=np.float64), dtype=np.float32)
        mat.flags.writeable = False
        fake_scores = ScoreMatrix(scores=mat, test_years=(0,), config_hash="pipeline", last_row=mat.shape[0] - 1)
        proxy = SimpleNamespace(arrays={"close": np.asarray(close_sliced, dtype=np.float64)})
        return build_targets(
            policy,
            proxy,  # type: ignore[arg-type]
            panel,
            fake_scores,
            np.asarray(universe_sliced, dtype=bool),
            rows=rows,
            capital_krw=int(capital_krw),
            cash_buffer=float(cash_buffer),
        )

    def _perturbation_mismatches(
        self,
        spec: StrategySpec,
        panel: FeaturePanel,
        base_scores: NDArray[np.float64],
        universe_sliced: NDArray[np.bool_],
        close_sliced: NDArray[np.float64],
        base_targets: Mapping[int, Mapping[int, NDArray[np.float64]]],
        sessions: list[date],
        start: date,
        end: date,
        last_row: int,
        auth: LockboxAuthorization,
    ) -> int:
        protocol = self._ctx.protocol
        cuts_n = int(protocol.criteria.c1.perturbation_cuts)
        seed = int(protocol.criteria.c1.perturbation_seed)
        lo_sim, hi_sim = _window_indices(sessions, start, end)
        if hi_sim <= lo_sim:  # pragma: no cover - discovery windows always span sessions
            edges = [lo_sim]
        else:
            edges = sorted(
                {
                    min(hi_sim, max(lo_sim, round(v)))
                    for v in np.linspace(float(lo_sim), float(hi_sim - 1), num=cuts_n).tolist()
                }
            )
            if not edges:  # pragma: no cover - linspace always yields a cut
                edges = [lo_sim]
        every = int(spec.policy.rebalance_every_sessions)
        buffer = float(self._base_sim_config().cash_buffer)
        primary = int(protocol.primary_capital_krw)
        total = 0
        for pos, cut in enumerate(edges):
            rng = np.random.default_rng(seed + pos)
            corrupted = self._corrupt_cube(self._ctx.cube, cut, rng)
            pert_panel = build_panel(corrupted, last_row=last_row)
            pert_uni_full = np.asarray(universe_mask(corrupted, spec.policy.universe), dtype=bool)
            pert_uni = np.ascontiguousarray(pert_uni_full[: last_row + 1])
            pert_close = np.asarray(corrupted.arrays["close"], dtype=np.float64)[: last_row + 1]
            cut_year = sessions[int(cut)].year
            try:
                rescored = walk_forward_scores(
                    pert_panel,
                    pert_uni,
                    sessions,
                    spec.scorer,
                    test_years=[cut_year],
                    authorization=auth,
                )
            except (ValueError, LockboxError):  # pragma: no cover - stubbed scorers always rescore
                continue
            pert_arr = np.asarray(rescored.scores, dtype=np.float64)
            merged = np.asarray(base_scores, dtype=np.float64).copy()
            year_rows = [r for r in range(last_row + 1) if sessions[r].year == cut_year]
            for r in year_rows:
                merged[r] = pert_arr[r]
            for phase, base_map in base_targets.items():
                rows = decision_rows(sessions, lo=lo_sim, hi=hi_sim + 1, every=every, phase=int(phase))
                pert_map = self._targets_for(
                    spec.policy, pert_close, pert_panel, merged, pert_uni, list(rows), primary, buffer
                )
                for row, base_w in base_map.items():
                    if int(row) > int(cut):
                        continue
                    other = pert_map.get(int(row))
                    if other is None:  # pragma: no cover - decision rows are identical
                        total += int(np.asarray(base_w).size)
                        continue
                    diff = np.abs(np.asarray(base_w, dtype=np.float64) - np.asarray(other, dtype=np.float64)) > 1e-12
                    total += int(np.count_nonzero(diff))
        return int(total)

    @staticmethod
    def _corrupt_cube(cube: ResearchCube, cut: int, rng: Any) -> ResearchCube:
        sessions = list(cube.sessions)
        n_s = len(sessions)
        rebuilt: dict[str, NDArray[Any]] = {}
        for name, arr in cube.arrays.items():
            values = np.asarray(arr)
            if values.ndim != 2 or values.shape[0] != n_s:  # pragma: no cover - synthetic cubes are dense 2-D
                rebuilt[name] = np.asarray(values).copy()
                continue
            work = values.copy()
            tail = (slice(cut + 1, n_s), slice(None))
            if values.dtype == bool or values.dtype == np.bool_:
                flips = rng.random(work[tail].shape) < 0.5
                work[tail] = np.logical_xor(work[tail], flips)
            elif np.issubdtype(values.dtype, np.floating):
                noise = rng.standard_normal(work[tail].shape)
                work[tail] = work[tail] * (1.0 + 0.5 * noise)
            elif np.issubdtype(values.dtype, np.integer):  # pragma: no cover - synthetic cubes carry floats/bools
                jitter = rng.integers(-2, 3, size=work[tail].shape)
                work[tail] = work[tail] + jitter
            rebuilt[name] = work
        return ResearchCube.from_arrays(
            cube_id=cube.cube_id,
            sessions=sessions,
            instrument_ids=list(cube.instrument_ids),
            arrays=rebuilt,
            exit_at=np.asarray(cube.exit_at).copy(),
            exit_halted=np.asarray(cube.exit_halted).copy(),
        )

    def _effective_registry_trials(self, family: str) -> float:
        trials = self._ctx.registry.trials(segment=Segment.DISCOVERY, family=family)
        groups: dict[tuple[str, ...], list[str]] = {}
        for trial in trials:
            try:
                stored = self._ctx.registry.returns(trial.trial_id)
            except PITDataError:  # pragma: no cover - registry returns are intact in tests
                continue
            key = tuple(day.isoformat() for day in stored.sessions)
            groups.setdefault(key, []).append(trial.trial_id)
        if not groups:  # pragma: no cover - discovery always records trials before this call
            return 1.0
        largest = max(groups.values(), key=len)
        columns: list[NDArray[np.float64]] = []
        for trial_id in largest:
            try:
                stored = self._ctx.registry.returns(trial_id)
            except PITDataError:  # pragma: no cover - registry returns are intact in tests
                continue
            columns.append(np.asarray(stored.net, dtype=np.float64))
        return float(_effective_from_columns(columns))

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

    def _write_report(self, spec_hash: str, kind: str, report: CriteriaReport) -> None:
        root = Path(self._ctx.reports_root)
        root.mkdir(parents=True, exist_ok=True)
        payload = _report_payload(report)
        payload["digest"] = report.digest
        (root / f"{spec_hash}_{kind}.json").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
