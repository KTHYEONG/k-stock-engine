"""Virtual ledger over sealed operational decisions with revision detection."""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.backtest.engine import DelistPolicy
from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.research.cube import ResearchCube
from src.research.ledger_bridge import LedgerFill, LedgerReject
from src.research.operate import SealedDecision

if TYPE_CHECKING:
    from src.research.pipeline import Pipeline, StrategySpec

__all__ = [
    "ShadowHistoryChanged",
    "ShadowRun",
    "append_shadow",
    "executable_targets",
    "run_shadow",
]


@dataclass(frozen=True, slots=True)
class ShadowRun:
    sessions: tuple[date, ...]
    nav_krw: NDArray[np.float64]
    log_returns: NDArray[np.float64]
    fills: tuple[LedgerFill, ...]
    rejects: tuple[LedgerReject, ...]
    late_sessions: tuple[date, ...]
    ledger_hash: str
    spec_hash: str = ""
    dataset_ids: Mapping[str, str] = field(default_factory=dict)


class ShadowHistoryChanged(PITDataError):  # noqa: N818 - name fixed by the ops-6 spec
    """A previously recorded shadow session no longer reproduces from current data."""


def _open_at(session: date) -> datetime:
    return datetime(session.year, session.month, session.day, 9, 0, tzinfo=KRX_TZ)


def _is_late(sealed_at: datetime, execution_session: date) -> bool:
    if sealed_at.utcoffset() is None:
        raise PITDataError("sealed_at must be timezone-aware")
    return sealed_at >= _open_at(execution_session)


def executable_targets(
    chain: Sequence[SealedDecision], cube: ResearchCube, *, on_time_only: bool
) -> tuple[dict[int, NDArray[np.float64]], tuple[date, ...]]:
    """Cube-row -> weight-vector targets usable by the ledger and late data sessions."""
    ordered = sorted(chain, key=lambda item: item.data_session)
    sessions = list(cube.sessions)
    row_of = {day: pos for pos, day in enumerate(sessions)}
    index_of = {inst: col for col, inst in enumerate(cube.instrument_ids)}
    width = len(cube.instrument_ids)
    seen_rows: set[int] = set()
    vectors: list[tuple[int, Any, NDArray[np.float64], bool]] = []
    for decision in ordered:
        row = row_of.get(decision.data_session)
        if row is None:
            raise PITDataError(f"sealed data session {decision.data_session.isoformat()} is not in the cube")
        if row in seen_rows:
            raise PITDataError(f"duplicate sealed data session {decision.data_session.isoformat()}")
        seen_rows.add(row)
        weights = np.zeros(width, dtype=np.float64)
        for inst, weight in dict(decision.targets).items():
            col = index_of.get(str(inst))
            if col is None:
                raise PITDataError(f"sealed instrument {inst!r} is not in the cube")
            value = float(weight)
            if not math.isfinite(value) or value < 0.0:
                raise PITDataError(f"sealed weight for {inst!r} must be finite and non-negative")
            weights[col] = value
        if float(weights.sum()) > 1.0 + 1e-9:
            raise PITDataError("sealed weights sum above one")
        vectors.append((row, decision, np.ascontiguousarray(weights), _is_late(decision.sealed_at, decision.execution_session)))
    targets: dict[int, NDArray[np.float64]] = {}
    late = [decision.data_session for _, decision, _, is_late in vectors if is_late]
    event_rows = set(seen_rows)
    if on_time_only:
        for row, decision, _, is_late in vectors:
            if is_late:
                for follower in range(row + 1, len(sessions) - 1):
                    if decision.sealed_at < _open_at(sessions[follower + 1]):
                        event_rows.add(follower)
                        break
    for row in sorted(event_rows):
        cutoff = _open_at(sessions[row + 1]) if row + 1 < len(sessions) else None
        usable = [
            weights for source, decision, weights, is_late in vectors
            if source <= row and (
                not on_time_only
                or (cutoff is not None and decision.sealed_at < cutoff)
                or (cutoff is None and not is_late)
            )
        ]
        if usable:
            targets[row] = np.ascontiguousarray(usable[-1].copy())
    return targets, tuple(late)


def run_shadow(
    pipeline: Pipeline, spec: StrategySpec, chain: Sequence[SealedDecision], *, capital_krw: int, on_time_only: bool = True
) -> ShadowRun:
    """Run the certified ledger over the shadow window with sealed targets."""
    if isinstance(capital_krw, bool) or not isinstance(capital_krw, int) or capital_krw < 1:
        raise ValueError(f"capital_krw must be an int >= 1, got {capital_krw!r}")
    if not chain:
        raise PITDataError("no sealed decisions to shadow")
    ctx = pipeline._ctx
    cube = ctx.cube
    targets, late = executable_targets(chain, cube, on_time_only=on_time_only)
    ordered = sorted(chain, key=lambda item: item.data_session)
    start = ordered[0].execution_session
    end = cube.sessions[-1]
    if start not in set(cube.sessions):
        raise PITDataError(f"shadow start {start.isoformat()} is not in the cube")
    auth = ctx.guard.authorize(start=start, end=end)
    targets = {row: weights for row, weights in targets.items() if row < len(cube.sessions) - 1}
    from src.research.cube import _load_tail_market

    market_arrays = _load_tail_market(Path(ctx.panel_dir), cube.sessions[0])
    if getattr(spec, "trend_overlay", None) is not None:
        try:
            delay_n = int(ctx.protocol.scenarios.stress_delay_sessions)
        except (AttributeError, TypeError, ValueError):
            delay_n = 0
        overlay, scenario_market, derivatives, leg_derivatives = pipeline._trend_scenario_overlay(
            spec, "base", delay_n=delay_n, extra_cost=0.0
        )
    else:
        from src.research.hedge import BetaNeutralOverlay, derivative_config
        from src.research.ledger_bridge import overlay_market_from_inputs

        overlay = BetaNeutralOverlay(spec.hedge, rebalance_offset=0)
        scenario_market = overlay_market_from_inputs(ctx.hedge_inputs)
        derivatives = derivative_config(spec.hedge)
        leg_derivatives = None
    outcome = ctx.ledger_runner(
        cube=cube,
        targets=targets,
        panel_dir=Path(ctx.panel_dir),
        dividends=ctx.dividends,
        engine_config_path=Path(ctx.engine_config_path),
        rules=ctx.rules,
        market_cache_root=Path(ctx.market_cache_root),
        capital_krw=int(capital_krw),
        halted_exit_policy=DelistPolicy.ZERO,
        start=start,
        end=end,
        authorization=auth,
        cash_returns=ctx.cash_returns,
        overlay=overlay,
        overlay_market=scenario_market,
        derivatives=derivatives,
        leg_derivatives=leg_derivatives,
        extra_slippage=0.0,
        auction_slippage_ticks=None,
        sessions_per_year=int(ctx.protocol.sessions_per_year),
        rebalance_band=float(spec.book.rebalance_band),
        market_arrays=market_arrays,
        record_orders=True,
        settle_final_taxes=False,
    )
    return ShadowRun(
        sessions=tuple(outcome.sessions),
        nav_krw=np.ascontiguousarray(np.asarray(outcome.nav_krw, dtype=np.float64)),
        log_returns=np.ascontiguousarray(np.asarray(outcome.log_returns, dtype=np.float64)),
        fills=tuple(outcome.fills),
        rejects=tuple(outcome.rejects),
        late_sessions=tuple(late),
        ledger_hash=str(outcome.ledger_hash),
        spec_hash=str(getattr(spec, "spec_hash", "")),
        dataset_ids=dict(getattr(ctx, "dataset_ids", {}) or {}),
    )


_NAV_SCHEMA = {"session": pl.Date, "nav_krw": pl.Int64, "log_return": pl.Float64}
_FILLS_SCHEMA = {
    "session": pl.Date,
    "decision_session": pl.Date,
    "instrument_id": pl.String,
    "side": pl.String,
    "quantity": pl.Int64,
    "price": pl.Int64,
}
_REJECTS_SCHEMA = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "side": pl.String,
    "quantity": pl.Int64,
    "reason": pl.String,
}


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _empty_frame(schema: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame({name: pl.Series(name, [], dtype=dtype) for name, dtype in schema.items()})


def append_shadow(shadow_dir: Path, run: ShadowRun, *, variant: str) -> int:
    """Persist sessions not yet recorded; returns how many were appended."""
    if not isinstance(variant, str) or not variant or "/" in variant or variant in (".", ".."):
        raise ValueError(f"variant must be a plain directory name, got {variant!r}")
    root = Path(shadow_dir) / variant
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".append.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _append_shadow(root, run, variant=variant)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _append_shadow(root: Path, run: ShadowRun, *, variant: str) -> int:
    nav_path = root / "nav.parquet"
    fills_path = root / "fills.parquet"
    rejects_path = root / "rejects.parquet"
    runs_path = root / "runs.jsonl"
    sessions = list(run.sessions)
    raw_nav = np.asarray(run.nav_krw, dtype=np.float64)
    if not np.all(np.isfinite(raw_nav) & (raw_nav == np.floor(raw_nav))):
        raise ValueError("shadow NAV must be finite integer KRW")
    nav_ints = [int(value) for value in raw_nav.tolist()]
    logs = [float(value) for value in np.asarray(run.log_returns, dtype=np.float64).tolist()]
    if len(sessions) != len(nav_ints) or len(sessions) != len(logs):
        raise ValueError("shadow run sessions, nav_krw and log_returns differ in length")
    if any(a >= b for a, b in pairwise(sessions)):
        raise ValueError("shadow sessions must be strictly increasing")
    recorded: dict[date, int] = {}
    if nav_path.is_file():
        prior = pl.read_parquet(nav_path)
        for day, nav in zip(prior["session"].to_list(), prior["nav_krw"].to_list(), strict=True):
            recorded[day] = int(nav)
    fresh = [(day, nav) for day, nav in zip(sessions, nav_ints, strict=True) if day not in recorded]
    for day, nav in zip(sessions, nav_ints, strict=True):
        if day in recorded and recorded[day] != nav:
            last = sessions[-1].isoformat()
            payload = {
                "last_session": last,
                "new_nav_krw": nav,
                "recorded_nav_krw": recorded[day],
                "session": day.isoformat(),
                "variant": variant,
            }
            target = root / f"divergence-{last}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(target.parent),
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                try:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                    os.replace(temporary, target)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        temporary.unlink()
            raise ShadowHistoryChanged(
                f"shadow session {day.isoformat()} no longer reproduces: "
                f"recorded={recorded[day]} new={nav}"
            )
    if not fresh:
        return 0
    fresh_days = {day for day, _ in fresh}
    if recorded:
        prior_nav = pl.read_parquet(nav_path)
        fresh_nav = pl.DataFrame(
            {
                "session": [day for day, _ in fresh],
                "nav_krw": [nav for _, nav in fresh],
                "log_return": [log for day, log in zip(sessions, logs, strict=True) if day in fresh_days],
            },
            schema=_NAV_SCHEMA,
        )
        combined_nav = pl.concat([prior_nav, fresh_nav], how="vertical").sort("session")
    else:
        combined_nav = pl.DataFrame(
            {"session": sessions, "nav_krw": nav_ints, "log_return": logs}, schema=_NAV_SCHEMA
        ).sort("session")
    fresh_fills = [
        fill for fill in run.fills if fill.session in fresh_days
    ]
    fresh_rejects = [reject for reject in run.rejects if reject.session in fresh_days]
    if fills_path.is_file():
        prior_fills = pl.read_parquet(fills_path)
        new_fills = (
            pl.DataFrame(
                {
                    "session": [fill.session for fill in fresh_fills],
                    "decision_session": [fill.decision_session for fill in fresh_fills],
                    "instrument_id": [fill.instrument_id for fill in fresh_fills],
                    "side": [fill.side for fill in fresh_fills],
                    "quantity": [fill.quantity for fill in fresh_fills],
                    "price": [fill.price for fill in fresh_fills],
                },
                schema=_FILLS_SCHEMA,
            )
            if fresh_fills
            else _empty_frame(_FILLS_SCHEMA)
        )
        combined_fills = pl.concat([prior_fills, new_fills], how="vertical").sort(["session", "instrument_id", "side"])
    else:
        combined_fills = (
            pl.DataFrame(
                {
                    "session": [fill.session for fill in fresh_fills],
                    "decision_session": [fill.decision_session for fill in fresh_fills],
                    "instrument_id": [fill.instrument_id for fill in fresh_fills],
                    "side": [fill.side for fill in fresh_fills],
                    "quantity": [fill.quantity for fill in fresh_fills],
                    "price": [fill.price for fill in fresh_fills],
                },
                schema=_FILLS_SCHEMA,
            )
            if fresh_fills
            else _empty_frame(_FILLS_SCHEMA)
        ).sort(["session", "instrument_id", "side"])
    if rejects_path.is_file():
        prior_rejects = pl.read_parquet(rejects_path)
        new_rejects = (
            pl.DataFrame(
                {
                    "session": [reject.session for reject in fresh_rejects],
                    "instrument_id": [reject.instrument_id for reject in fresh_rejects],
                    "side": [reject.side for reject in fresh_rejects],
                    "quantity": [reject.quantity for reject in fresh_rejects],
                    "reason": [reject.reason for reject in fresh_rejects],
                },
                schema=_REJECTS_SCHEMA,
            )
            if fresh_rejects
            else _empty_frame(_REJECTS_SCHEMA)
        )
        combined_rejects = pl.concat([prior_rejects, new_rejects], how="vertical").sort(["session", "instrument_id", "side"])
    else:
        combined_rejects = (
            pl.DataFrame(
                {
                    "session": [reject.session for reject in fresh_rejects],
                    "instrument_id": [reject.instrument_id for reject in fresh_rejects],
                    "side": [reject.side for reject in fresh_rejects],
                    "quantity": [reject.quantity for reject in fresh_rejects],
                    "reason": [reject.reason for reject in fresh_rejects],
                },
                schema=_REJECTS_SCHEMA,
            )
            if fresh_rejects
            else _empty_frame(_REJECTS_SCHEMA)
        ).sort(["session", "instrument_id", "side"])
    _write_parquet(nav_path, combined_nav)
    _write_parquet(fills_path, combined_fills)
    _write_parquet(rejects_path, combined_rejects)
    line = json.dumps(
        {
            "dataset_ids": dict(run.dataset_ids),
            "last_session": sessions[-1].isoformat(),
            "ledger_hash": run.ledger_hash,
            "spec_hash": run.spec_hash,
        },
        sort_keys=True,
    )
    history = runs_path.read_text(encoding="utf-8") if runs_path.is_file() else ""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=root, prefix=".runs.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(history + line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, runs_path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    return len(fresh)
