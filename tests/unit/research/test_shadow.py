"""Shadow ledger over sealed decisions: targets, appends and conservation."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.backtest.engine import DelistPolicy
from src.core.pit import PITDataError
from src.core.time import KRX_TZ

INSTRS = ["KRX:000001", "KRX:000002"]
CAPITAL = 10_000_000


def _sessions(n: int = 12) -> list[date]:
    from tests.fixtures.synthetic_panel import synthetic_sessions

    return synthetic_sessions(n, start=date(2019, 1, 5))


def _spec() -> Any:
    from src.research.book import BookSpec
    from src.research.hedge import HedgeSpec
    from src.research.pipeline import StrategySpec
    from src.research.policy import TrendCashPolicy
    from src.research.model import ScorerConfig

    return StrategySpec(
        policy=TrendCashPolicy(rebalance_every_sessions=1, min_units_per_slot=0, n=2),
        scorer=ScorerConfig(first_test_year=2019, min_train_rows=1, min_cross_section=1, num_boost_round=2, min_data_in_leaf=2, num_threads=1),
        book=BookSpec(sleeves=1, stock_capital_fraction=0.75),
        hedge=HedgeSpec(
            hedge_ratio=0.0, beta_window_sessions=10, beta_min_sessions=2, beta_cap=2.0,
            rebalance_every_sessions=1, use_futures=False, contract_multiplier_krw=10000,
            initial_margin_rate=0.2175, margin_buffer_rate=0.10, margin_topup_trigger_fraction=0.75,
            futures_cost_rate=0.0003, inverse_cost_rate=0.0007, resize_sell_cost_rate=0.0025,
            resize_buy_cost_rate=0.0005, futures_tax_rate=0.11, futures_annual_deduction_krw=2500000,
            inverse_tax_rate=0.154,
        ),
    )


def _pipe(tmp_path: Path, sessions: list[date], **over: Any) -> Any:
    from pathlib import Path as _Path

    from src.core.market_rules import load_krx_market_rules
    from src.data.research_protocol import WindowGuard, load_research_protocol
    from src.data.research_scope import load_research_scope
    from src.research.hedge import HedgeInputs
    from src.research.ledger_bridge import run_ledger
    from src.research.pipeline import Pipeline, PipelineContext
    from src.research.registry import RunRegistry
    from tests.fixtures.synthetic_panel import synthetic_cube, synthetic_panel_row, write_synthetic_panel

    protocol = load_research_protocol(
        _Path("config/research/protocol.toml"), load_research_scope(_Path("config/research/kr_swing_2019_v1.toml"))
    )
    cube = over.get("cube") or synthetic_cube(sessions, INSTRS)
    rows = over.get("rows") or [synthetic_panel_row(day, inst) for day in sessions for inst in INSTRS]
    panel = write_synthetic_panel(tmp_path / "gold", "market_panel_test", rows)
    hedge_inputs = HedgeInputs(
        sessions=tuple(sessions),
        index_level=np.full(len(sessions), 1500.0),
        inverse_close=np.full(len(sessions), 8000.0),
    )
    ctx = PipelineContext(
        protocol=protocol,
        cube=cube,
        registry=RunRegistry(tmp_path / "runs"),
        guard=WindowGuard(protocol=protocol, last_session=sessions[-1]),
        panel_dir=panel,
        dividends=pl.DataFrame(),
        hedge_inputs=hedge_inputs,
        rules=load_krx_market_rules(_Path("config/market/krx_market_rules.toml")),
        engine_config_path=_Path("config/backtest/default_engine.toml"),
        market_cache_root=tmp_path / "mcache",
        reports_root=tmp_path / "reports",
        scores_root=tmp_path / "scores",
        ledger_runner=over.get("ledger_runner") or run_ledger,
        now=lambda: datetime(2026, 9, 30, tzinfo=KRX_TZ),
        dataset_ids={
            "market_panel": "market_panel_aaaaaaaaaaaaaaa",
            "dividend_events": "dividend_events_bbbbbbbbbbbbbbb",
            "hedge_series": "hedge_series_cccccccccccccccc",
            "cash_series": "cash_series_dddddddddddddddd",
        },
        cash_returns=np.zeros(len(sessions), dtype=np.float64),
    )
    return Pipeline(ctx)


def _sealed(data: date, execution: date, targets: dict[str, float], sealed_at: datetime) -> Any:
    from src.research.operate import SealedDecision

    return SealedDecision(
        data_session=data,
        execution_session=execution,
        sealed_at=sealed_at,
        spec_hash="spec",
        models_id="models",
        inputs={},
        targets=dict(targets),
        reference={},
        overlay={},
        state_digest="digest",
        prev_hash="",
        hash="",
    )


def _open_morning(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 8, 0, tzinfo=KRX_TZ)


def _chain(sessions: list[date], rows: list[int], weights: list[dict[str, float]]) -> tuple[Any, ...]:
    return tuple(
        _sealed(sessions[row], sessions[row + 1], mapping, _open_morning(sessions[row + 1]))
        for row, mapping in zip(rows, weights, strict=True)
    )


def test_sealed_targets_reproduce_research_ledger(tmp_path: Path) -> None:
    from src.research.hedge import BetaNeutralOverlay, derivative_config
    from src.research.ledger_bridge import overlay_market_from_inputs
    from src.research.shadow import run_shadow

    sessions = _sessions()
    pipe = _pipe(tmp_path, sessions)
    spec = _spec()
    rows = [2, 3, 4, 5]
    maps = [{INSTRS[0]: 0.5, INSTRS[1]: 0.5}, {INSTRS[0]: 1.0}, {INSTRS[1]: 1.0}, {INSTRS[0]: 0.25, INSTRS[1]: 0.25}]
    chain = _chain(sessions, rows, maps)
    run = run_shadow(pipe, spec, chain, capital_krw=CAPITAL, on_time_only=True)
    research = {row: np.array([m.get(inst, 0.0) for inst in INSTRS]) for row, m in zip(rows, maps, strict=True)}
    ctx = pipe._ctx
    auth = ctx.guard.authorize(start=sessions[3], end=sessions[-1])
    expected = ctx.ledger_runner(
        cube=ctx.cube, targets=research, panel_dir=Path(ctx.panel_dir), dividends=ctx.dividends,
        engine_config_path=Path(ctx.engine_config_path), rules=ctx.rules,
        market_cache_root=Path(ctx.market_cache_root), capital_krw=CAPITAL,
        halted_exit_policy=DelistPolicy.ZERO, start=sessions[3], end=sessions[-1],
        authorization=auth, cash_returns=ctx.cash_returns,
        overlay=BetaNeutralOverlay(spec.hedge, rebalance_offset=0),
        overlay_market=overlay_market_from_inputs(ctx.hedge_inputs),
        derivatives=derivative_config(spec.hedge), leg_derivatives=None,
        extra_slippage=0.0, auction_slippage_ticks=None,
        sessions_per_year=int(ctx.protocol.sessions_per_year),
        rebalance_band=float(spec.book.rebalance_band), record_orders=True,
    )
    assert run.sessions == expected.sessions
    assert np.array_equal(np.asarray(run.log_returns), np.asarray(expected.log_returns))


def test_late_decision_carried_forward_official_only() -> None:
    from src.research.shadow import executable_targets
    from tests.fixtures.synthetic_panel import synthetic_cube

    sessions = _sessions()
    cube = synthetic_cube(sessions, INSTRS)
    first = {INSTRS[0]: 0.5}
    second = {INSTRS[1]: 0.5}
    third = {INSTRS[0]: 1.0}
    late_at = datetime(2019, 1, 5 + 5, 9, 30, tzinfo=KRX_TZ)
    chain = (
        _sealed(sessions[2], sessions[3], first, _open_morning(sessions[3])),
        _sealed(sessions[3], sessions[4], second, _open_morning(sessions[4])),
        _sealed(sessions[4], sessions[5], third, late_at),
    )
    official, late = executable_targets(chain, cube, on_time_only=True)
    assert late == (sessions[4],)
    assert official[4].tolist() == [0.0, 0.5]
    ideal, _ = executable_targets(chain, cube, on_time_only=False)
    assert ideal[4].tolist() == [1.0, 0.0]


def test_unknown_instrument_fails_closed() -> None:
    from src.research.shadow import executable_targets
    from tests.fixtures.synthetic_panel import synthetic_cube

    sessions = _sessions()
    cube = synthetic_cube(sessions, INSTRS)
    chain = _chain(sessions, [2], [{"KRX:999999": 0.5}])
    with pytest.raises(PITDataError):
        executable_targets(chain, cube, on_time_only=True)


def test_append_is_idempotent(tmp_path: Path) -> None:
    from src.research.shadow import append_shadow, run_shadow

    sessions = _sessions()
    pipe = _pipe(tmp_path, sessions)
    run = run_shadow(pipe, _spec(), _chain(sessions, [2, 3], [{INSTRS[0]: 0.5}, {INSTRS[1]: 0.5}]), capital_krw=CAPITAL)
    assert append_shadow(tmp_path / "shadow", run, variant="official") > 0
    before = {
        path.name: path.read_bytes()
        for path in sorted((tmp_path / "shadow" / "official").iterdir())
    }
    assert append_shadow(tmp_path / "shadow", run, variant="official") == 0
    after = {
        path.name: path.read_bytes()
        for path in sorted((tmp_path / "shadow" / "official").iterdir())
    }
    assert before == after


def test_history_change_detected_not_overwritten(tmp_path: Path) -> None:
    from src.research.shadow import ShadowHistoryChanged, append_shadow, run_shadow
    from tests.fixtures.synthetic_panel import synthetic_panel_row, write_synthetic_panel

    sessions = _sessions()
    pipe = _pipe(tmp_path, sessions)
    chain = _chain(sessions, [2, 3], [{INSTRS[0]: 0.5}, {INSTRS[1]: 0.5}])
    run = run_shadow(pipe, _spec(), chain, capital_krw=CAPITAL)
    append_shadow(tmp_path / "shadow", run, variant="official")
    nav_before = (tmp_path / "shadow" / "official" / "nav.parquet").read_bytes()
    drift = sessions[5]
    rows = [synthetic_panel_row(day, inst) for day in sessions for inst in INSTRS]
    for row in rows:
        if row["session"] == drift and row["instrument_id"] == INSTRS[1]:
            row["close"] = 20000
    write_synthetic_panel(tmp_path / "gold", "market_panel_test", rows)
    changed = run_shadow(pipe, _spec(), chain, capital_krw=CAPITAL)
    with pytest.raises(ShadowHistoryChanged):
        append_shadow(tmp_path / "shadow", changed, variant="official")
    assert (tmp_path / "shadow" / "official" / "nav.parquet").read_bytes() == nav_before
    divergences = list((tmp_path / "shadow" / "official").glob("divergence-*.json"))
    assert len(divergences) == 1
    import json as _json

    payload = _json.loads(divergences[0].read_text(encoding="utf-8"))
    assert payload["recorded_nav_krw"] != payload["new_nav_krw"]


def test_conservation_exact_krw(tmp_path: Path) -> None:
    from src.research.ledger_bridge import run_ledger
    from src.research.shadow import run_shadow

    sessions = _sessions()
    captured: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> Any:
        outcome = run_ledger(**kwargs)
        captured["outcome"] = outcome
        return outcome

    pipe = _pipe(tmp_path, sessions, ledger_runner=_spy)
    chain = _chain(sessions, [2, 3, 4], [{INSTRS[0]: 0.5}, {INSTRS[1]: 0.5}, {}])
    run = run_shadow(pipe, _spec(), chain, capital_krw=CAPITAL)
    journal = captured["outcome"].journal_totals_krw
    assert int(run.nav_krw[-1]) == CAPITAL + sum(int(value) for value in journal.values())


def test_executable_targets_validation(tmp_path: Path) -> None:
    from src.research.shadow import executable_targets, run_shadow
    from tests.fixtures.synthetic_panel import synthetic_cube

    sessions = _sessions()
    cube = synthetic_cube(sessions, INSTRS)
    (outside,) = _chain([sessions[-1] + timedelta(days=30), sessions[-1] + timedelta(days=31)], [0], [{INSTRS[0]: 0.5}])
    with pytest.raises(PITDataError):
        executable_targets((outside,), cube, on_time_only=True)
    dup = _chain(sessions, [2, 2], [{INSTRS[0]: 0.5}, {INSTRS[1]: 0.5}])
    with pytest.raises(PITDataError):
        executable_targets(dup, cube, on_time_only=True)
    nan_weight = _chain(sessions, [2], [{INSTRS[0]: float("nan")}])
    with pytest.raises(PITDataError):
        executable_targets(nan_weight, cube, on_time_only=True)
    pipe = _pipe(tmp_path, sessions)
    with pytest.raises(ValueError, match="capital_krw"):
        run_shadow(pipe, _spec(), _chain(sessions, [2], [{INSTRS[0]: 0.5}]), capital_krw=0)
    with pytest.raises(PITDataError, match="no sealed decisions"):
        run_shadow(pipe, _spec(), (), capital_krw=CAPITAL)
    far = _sealed(sessions[-1], sessions[-1] + timedelta(days=30), {INSTRS[0]: 0.5}, _open_morning(sessions[-1]))
    with pytest.raises(PITDataError, match="shadow start"):
        run_shadow(pipe, _spec(), (far,), capital_krw=CAPITAL)


def test_run_shadow_with_trend_overlay(tmp_path: Path) -> None:
    from src.research.hedge import HedgeInputs
    from src.research.shadow import run_shadow
    from src.research.trend_overlay import TrendOverlaySpec

    sessions = _sessions(20)
    pipe = _pipe(tmp_path, sessions)
    levels = 1500.0 * np.exp(np.cumsum(np.full(len(sessions), 0.0004)))
    object.__setattr__(
        pipe._ctx, "trend_inputs",
        HedgeInputs(sessions=tuple(sessions), index_level=np.ascontiguousarray(levels), inverse_close=np.ascontiguousarray(np.full(len(sessions), 8000.0))),
    )
    object.__setattr__(pipe._ctx, "dataset_ids", dict(pipe._ctx.dataset_ids, trend_series="trend_series_eeeeeeeeeeeeeeee"))
    spec = _spec().model_copy(update={"trend_overlay": TrendOverlaySpec(
        ma_sessions=5, long_fraction=0.5, short_fraction=0.5, rebalance_every_sessions=1,
        contract_multiplier_krw=10000, initial_margin_rate=0.2, margin_buffer_rate=0.1,
        margin_topup_trigger_fraction=0.75, futures_cost_rate=0.0003, futures_tax_rate=0.1,
        futures_annual_deduction_krw=0)})
    chain = _chain(sessions, [5, 6], [{INSTRS[0]: 0.5}, {INSTRS[1]: 0.5}])
    run = run_shadow(pipe, spec, chain, capital_krw=CAPITAL)
    assert len(run.sessions) > 0
    assert run.late_sessions == ()


def test_append_incremental_and_guards(tmp_path: Path) -> None:
    import polars as pl
    import pytest

    from src.research.shadow import ShadowRun, append_shadow

    days = _sessions(8)
    first = ShadowRun(
        sessions=tuple(days[:5]),
        nav_krw=np.array([10_000_000.0, 10_000_100.0, 10_000_200.0, 10_000_300.0, 10_000_400.0]),
        log_returns=np.zeros(5),
        fills=(), rejects=(), late_sessions=(), ledger_hash="a" * 64,
    )
    assert append_shadow(tmp_path / "shadow", first, variant="official") == 5
    second = ShadowRun(
        sessions=tuple(days[:7]),
        nav_krw=np.array([10_000_000.0, 10_000_100.0, 10_000_200.0, 10_000_300.0, 10_000_400.0, 10_000_500.0, 10_000_600.0]),
        log_returns=np.zeros(7),
        fills=(), rejects=(), late_sessions=(), ledger_hash="b" * 64,
    )
    assert append_shadow(tmp_path / "shadow", second, variant="official") == 2
    frame = pl.read_parquet(tmp_path / "shadow" / "official" / "nav.parquet")
    assert frame.height == 7
    assert frame["nav_krw"].to_list() == [10_000_000, 10_000_100, 10_000_200, 10_000_300, 10_000_400, 10_000_500, 10_000_600]
    with pytest.raises(ValueError, match="variant"):
        append_shadow(tmp_path / "shadow", first, variant="../evil")
    broken = ShadowRun(
        sessions=tuple(days[:3]), nav_krw=np.array([1.0, 2.0]), log_returns=np.zeros(3),
        fills=(), rejects=(), late_sessions=(), ledger_hash="c" * 64,
    )
    with pytest.raises(ValueError, match="differ in length"):
        append_shadow(tmp_path / "shadow", broken, variant="official")


def test_run_shadow_delay_fallback(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from src.research.hedge import HedgeInputs
    from src.research.shadow import run_shadow
    from src.research.trend_overlay import TrendOverlaySpec

    sessions = _sessions(20)
    pipe = _pipe(tmp_path, sessions)
    object.__setattr__(
        pipe._ctx, "trend_inputs",
        HedgeInputs(sessions=tuple(sessions), index_level=np.full(len(sessions), 1500.0), inverse_close=np.full(len(sessions), 8000.0)),
    )
    object.__setattr__(pipe._ctx, "protocol", SimpleNamespace(sessions_per_year=252))
    spec = _spec().model_copy(update={"trend_overlay": TrendOverlaySpec(
        ma_sessions=5, long_fraction=0.5, short_fraction=0.5, rebalance_every_sessions=1,
        contract_multiplier_krw=10000, initial_margin_rate=0.2, margin_buffer_rate=0.1,
        margin_topup_trigger_fraction=0.75, futures_cost_rate=0.0003, futures_tax_rate=0.1,
        futures_annual_deduction_krw=0)})
    run = run_shadow(pipe, spec, _chain(sessions, [5], [{INSTRS[0]: 0.5}]), capital_krw=CAPITAL)
    assert len(run.sessions) > 0


def test_consecutive_late_decisions_use_latest_available_target() -> None:
    from dataclasses import replace

    from src.research.shadow import executable_targets
    from tests.fixtures.synthetic_panel import synthetic_cube

    sessions = _sessions()
    chain = list(_chain(sessions, [2, 3, 4], [{INSTRS[0]: 0.5}, {INSTRS[1]: 0.5}, {INSTRS[0]: 1.0}]))
    chain[1] = replace(chain[1], sealed_at=_open_morning(sessions[4]) + timedelta(hours=2))
    chain[2] = replace(chain[2], sealed_at=_open_morning(sessions[5]) + timedelta(hours=2))
    targets, _ = executable_targets(chain, synthetic_cube(sessions, INSTRS), on_time_only=True)
    assert targets[3].tolist() == [0.5, 0.0]
    assert targets[4].tolist() == [0.0, 0.5]
    assert targets[5].tolist() == [1.0, 0.0]


def test_shadow_tail_and_pending_last_decision(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.research.shadow import run_shadow
    from tests.fixtures.synthetic_panel import synthetic_cube

    days = _sessions()
    pipe = _pipe(tmp_path, days)
    tail = days[2:]
    object.__setattr__(pipe._ctx, "cube", synthetic_cube(tail, INSTRS))
    object.__setattr__(pipe._ctx, "cash_returns", pipe._ctx.cash_returns[2:])
    hedge = pipe._ctx.hedge_inputs
    object.__setattr__(pipe._ctx, "hedge_inputs", replace(
        hedge, sessions=tuple(tail), index_level=hedge.index_level[2:], inverse_close=hedge.inverse_close[2:]
    ))
    chain = list(_chain(days, [3, 10], [{INSTRS[0]: 0.5}, {INSTRS[1]: 0.5}]))
    chain.append(replace(chain[-1], data_session=days[-1], execution_session=days[-1] + timedelta(days=1)))
    run = run_shadow(pipe, _spec(), chain, capital_krw=CAPITAL)
    assert run.sessions[0] == days[4]
    assert run.sessions[-1] == days[-1]


@pytest.mark.parametrize("weight", [-0.5, float("inf"), 1.5])
def test_invalid_sealed_weights_rejected(weight: float) -> None:
    from src.research.shadow import executable_targets
    from tests.fixtures.synthetic_panel import synthetic_cube

    days = _sessions()
    with pytest.raises(PITDataError):
        executable_targets(_chain(days, [2], [{INSTRS[0]: weight}]), synthetic_cube(days, INSTRS), on_time_only=True)


def test_naive_seal_rejected() -> None:
    from dataclasses import replace

    from src.research.shadow import executable_targets
    from tests.fixtures.synthetic_panel import synthetic_cube

    days = _sessions()
    decision = _chain(days, [2], [{INSTRS[0]: 0.5}])[0]
    decision = replace(decision, sealed_at=decision.sealed_at.replace(tzinfo=None))
    with pytest.raises(PITDataError, match="timezone-aware"):
        executable_targets((decision,), synthetic_cube(days, INSTRS), on_time_only=True)


def test_append_invalid_nav_and_timeline(tmp_path: Path) -> None:
    from src.research.shadow import ShadowRun, append_shadow

    day = date(2020, 1, 6)
    with pytest.raises(ValueError, match="strictly increasing"):
        append_shadow(tmp_path, ShadowRun((day, day), np.ones(2), np.zeros(2), (), (), (), "h"), variant="official")
    with pytest.raises(ValueError, match="integer KRW"):
        append_shadow(tmp_path, ShadowRun((day,), np.array([1.5]), np.zeros(1), (), (), (), "h"), variant="official")


@pytest.mark.parametrize("days", [
    _sessions(),
    [date(2020, 12, 28), date(2020, 12, 29), date(2020, 12, 30), date(2021, 1, 4)],
])
def test_taxed_shadow_history_is_stable_on_extension(tmp_path: Path, days: list[date]) -> None:
    from src.research.shadow import append_shadow, run_shadow

    chain = _chain(days, [0], [{}])
    short_pipe = _pipe(tmp_path / "short", days[:-1])
    full_pipe = _pipe(tmp_path / "full", days)
    object.__setattr__(short_pipe._ctx, "cash_returns", np.full(len(days) - 1, 0.001))
    object.__setattr__(full_pipe._ctx, "cash_returns", np.full(len(days), 0.001))
    short = run_shadow(short_pipe, _spec(), chain, capital_krw=CAPITAL)
    full = run_shadow(full_pipe, _spec(), chain, capital_krw=CAPITAL)
    np.testing.assert_array_equal(short.nav_krw, full.nav_krw[:-1])
    assert append_shadow(tmp_path / "shadow", short, variant="official") == len(short.sessions)
    assert append_shadow(tmp_path / "shadow", full, variant="official") == 1
