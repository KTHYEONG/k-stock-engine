"""Engine loop: timeline order, fills, exits, determinism, and benchmark parity."""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.backtest.costs import CostConfig, Side
from src.backtest.engine import BacktestResult, DelistPolicy, EngineConfig, run_backtest
from src.backtest.events import build_engine_events
from src.backtest.execution import ExecutionConfig, ExecutionScenario
from src.backtest.market import MarketArrays, load_market_arrays
from src.backtest.strategy import EqualWeightLiquid, PortfolioSnapshot, Targets
from src.backtest.view import PITView
from src.core.market_rules import KrxMarketRules, load_krx_market_rules
from tests.unit.backtest.test_events import _div_frame, _div_row, _exit_row, _write_exits
from tests.unit.backtest.test_market import _mrow, _write_panel

RULES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "market" / "krx_market_rules.toml"


def _rules() -> KrxMarketRules:
    return load_krx_market_rules(RULES_PATH)


def _sessions(n: int, start: date = date(2020, 1, 6)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _flat_row(session: date, instrument_id: str, close: int, **overrides: Any) -> dict[str, Any]:
    row = _mrow(
        session,
        instrument_id,
        open=close,
        high=close,
        low=close,
        close=close,
        base_price=close,
        upper_limit=close * 10,
        lower_limit=max(1, close // 10),
    )
    row.update(overrides)
    return row


def _arrays(tmp_path: Path, name: str, rows: list[dict[str, Any]]) -> MarketArrays:
    panel = _write_panel(tmp_path / "gold", name, rows)
    return load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")


def _panel_dir(tmp_path: Path, name: str) -> Path:
    return tmp_path / "gold" / name


def _costs(commission: str = "0.00015", impact_k: float = 0.5) -> CostConfig:
    return CostConfig(commission_rate=Decimal(commission), impact_k=impact_k)


def _execution(carry: bool = False, participation: float = 1.0) -> ExecutionConfig:
    return ExecutionConfig(
        scenario=ExecutionScenario.OPEN_AUCTION,
        max_participation=participation,
        carry_unfilled=carry,
    )


def _config(
    initial_cash: int = 10_000_000,
    carry: bool = False,
    participation: float = 1.0,
    commission: str = "0.00015",
    impact_k: float = 0.5,
    policy: DelistPolicy = DelistPolicy.LAST_CLOSE,
    cash_buffer: float = 0.0,
) -> EngineConfig:
    return EngineConfig(
        initial_cash=initial_cash,
        execution=_execution(carry, participation),
        costs=_costs(commission, impact_k),
        halted_exit_policy=policy,
        cash_buffer=cash_buffer,
    )


class _ScriptedStrategy:
    """Rebalance on scheduled sessions toward fixed weight maps."""

    def __init__(self, schedule: dict[int, dict[int, float]]) -> None:
        self._schedule = schedule
        self.name = "scripted"

    def params(self) -> dict[str, str | int | float | bool]:
        return {"sessions": ",".join(str(t) for t in sorted(self._schedule))}

    def is_rebalance(self, view: PITView) -> bool:
        return view.t in self._schedule

    def decide(self, view: PITView, portfolio: PortfolioSnapshot) -> Targets:
        return Targets(weights=dict(self._schedule[view.t]))


def _positions_at(journal: Any, t: int, *, strict: bool) -> dict[int, int]:
    pos: dict[int, int] = {}
    for entry in journal:
        if entry.session_idx < t or (not strict and entry.session_idx == t):
            if entry.instrument_idx is not None:
                pos[entry.instrument_idx] = pos.get(entry.instrument_idx, 0) + entry.quantity_delta
    return {n: q for n, q in pos.items() if q}


def _conservation_run(tmp_path: Path) -> tuple[MarketArrays, BacktestResult, list[date]]:
    sessions = _sessions(30)
    rows: list[dict[str, Any]] = []
    for t, day in enumerate(sessions):
        rows.append(_flat_row(day, "KRX:A", 100, share_factor=2.0 if t == 10 else 1.0))
        if t <= 20:
            rows.append(_flat_row(day, "KRX:B", 200))
        rows.append(_flat_row(day, "KRX:C", 300))
    panel = _write_panel(tmp_path / "gold", "market_panel_cons", rows)
    _write_exits(panel, [_exit_row("KRX:B", sessions[20], last_close=200, exit_kind="traded_exit")])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    dividends = _div_frame([_div_row("KRX:A", sessions[12], sessions[15], dps_krw=100)])
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=dividends)
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=EqualWeightLiquid(min_adtv20_krw=0, max_names=2),
        config=_config(),
        deposits={sessions[0]: 1_000_000, date(2020, 1, 1): 2_000_000, sessions[15]: 500_000},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    return arrays, result, sessions


def test_journal_conservation_every_session(tmp_path: Path) -> None:
    arrays, result, sessions = _conservation_run(tmp_path)
    initial = 10_000_000
    recv = 0
    for record in result.nav:
        t = record.session_idx
        if t == 12:
            recv = _positions_at(result.journal, t, strict=True).get(0, 0) * 100
        if t == 15:
            recv = 0
        cash = initial + sum(e.cash_delta for e in result.journal if e.session_idx <= t)
        assert record.cash == cash
        pos = _positions_at(result.journal, t, strict=False)
        market_value = sum(q * int(arrays.int_fields["close"][t, n]) for n, q in pos.items())
        assert record.market_value == market_value
        assert record.dividend_receivable == recv
        assert record.nav == record.cash + record.dividend_receivable + market_value
        expected_ext = 3_000_000 + (500_000 if t >= 15 else 0)
        assert record.external_flow == expected_ext
    assert len(result.nav) == len(sessions)
    kinds = {e.kind.value for e in result.journal}
    assert {"deposit", "buy", "commission", "cash_in_lieu", "dividend", "exit_proceeds"} <= kinds


def test_orders_execute_next_session(tmp_path: Path) -> None:
    sessions = _sessions(3)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    rows[1] = _flat_row(sessions[1], "KRX:A", 100, open=105)
    panel = _write_panel(tmp_path / "gold", "market_panel_next", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=_ScriptedStrategy({0: {0: 0.01}}),
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    buys = [e for e in result.journal if e.kind.value == "buy"]
    assert len(buys) == 1
    assert buys[0].session_idx == 1
    assert buys[0].cash_delta == -100 * 105
    assert buys[0].quantity_delta == 100


def test_sale_proceeds_fund_same_session_buys(tmp_path: Path) -> None:
    sessions = _sessions(5)
    rows = [_flat_row(day, "KRX:A", 100, sell_tax_rate=0.0) for day in sessions]
    rows += [_flat_row(day, "KRX:B", 200, sell_tax_rate=0.0) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_rotation", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=_ScriptedStrategy({0: {0: 1.0}, 2: {1: 1.0}}),
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    assert not [r for r in result.rejects if r.reason == "cash"]
    b_buys = [e for e in result.journal if e.kind.value == "buy" and e.instrument_idx == 1]
    a_sells = [e for e in result.journal if e.kind.value == "sell" and e.instrument_idx == 0]
    assert len(b_buys) == 1 and len(a_sells) == 1
    assert b_buys[0].session_idx == a_sells[0].session_idx == 3
    assert b_buys[0].quantity_delta == 5000


def test_halted_exit_scenarios_differ_only_by_exit_value(tmp_path: Path) -> None:
    sessions = _sessions(15)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    rows += [_flat_row(day, "KRX:C", 150) for day in sessions[:11]]
    panel = _write_panel(tmp_path / "gold", "market_panel_halt", rows)
    _write_exits(
        panel, [_exit_row("KRX:C", sessions[10], last_close=150, last_volume=0, exit_kind="halted_exit")]
    )
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    results = {}
    for policy in (DelistPolicy.LAST_CLOSE, DelistPolicy.ZERO):
        results[policy] = run_backtest(
            arrays=arrays,
            events=events,
            strategy=_ScriptedStrategy({0: {1: 1.0}}),
            config=_config(initial_cash=1_500_000, impact_k=0.0, commission="0", policy=policy),
            deposits={},
            asof_tables={},
            rules=_rules(),
            start=sessions[0],
            end=sessions[-1],
        )
    close_run, zero_run = results[DelistPolicy.LAST_CLOSE], results[DelistPolicy.ZERO]
    assert close_run.nav[:11] == zero_run.nav[:11]
    assert close_run.nav[-1].nav - zero_run.nav[-1].nav == 10_000 * 150


def test_deterministic_ledger_hash(tmp_path: Path) -> None:
    _, first, _ = _conservation_run(tmp_path)
    arrays = load_market_arrays(
        panel_dir=_panel_dir(tmp_path, "market_panel_cons"), cache_root=tmp_path / "cache"
    )
    panel = _panel_dir(tmp_path, "market_panel_cons")
    sessions = arrays.sessions
    dividends = _div_frame([_div_row("KRX:A", sessions[12], sessions[15], dps_krw=100)])
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=dividends)
    second = run_backtest(
        arrays=arrays,
        events=events,
        strategy=EqualWeightLiquid(min_adtv20_krw=0, max_names=2),
        config=_config(),
        deposits={sessions[0]: 1_000_000, date(2020, 1, 1): 2_000_000, sessions[15]: 500_000},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    assert first.ledger_hash == second.ledger_hash
    assert first == second


def test_lookahead_free_end_to_end(tmp_path: Path) -> None:
    sessions = _sessions(12)
    rows: list[dict[str, Any]] = []
    for t, day in enumerate(sessions):
        rows.append(_flat_row(day, "KRX:A", 100 + t))
        rows.append(_flat_row(day, "KRX:B", 200 - t))
    panel = _write_panel(tmp_path / "gold", "market_panel_base", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)

    rng = random.Random(0)
    perturbed: list[dict[str, Any]] = []
    for t, day in enumerate(sessions):
        for inst, base in (("KRX:A", 100 + t), ("KRX:B", 200 - t)):
            close = base if t <= 8 else int(base * 1.05) + 7
            perturbed.append(_flat_row(day, inst, close))
    panel2 = _write_panel(tmp_path / "gold", "market_panel_perturbed", perturbed)
    _write_exits(panel2, [])
    arrays2 = load_market_arrays(panel_dir=panel2, cache_root=tmp_path / "cache")
    events2 = build_engine_events(arrays=arrays2, panel_dir=panel2, dividends=None)

    kwargs: dict[str, Any] = {
        "deposits": {},
        "asof_tables": {},
        "rules": _rules(),
        "start": sessions[0],
        "end": sessions[-1],
    }
    base = run_backtest(
        arrays=arrays, events=events, strategy=_ScriptedStrategy({t: {0: 0.5, 1: 0.5} for t in range(12)}),
        config=_config(impact_k=0.0, commission="0"), **kwargs,
    )
    other = run_backtest(
        arrays=arrays2, events=events2, strategy=_ScriptedStrategy({t: {0: 0.5, 1: 0.5} for t in range(12)}),
        config=_config(impact_k=0.0, commission="0"), **kwargs,
    )
    assert [f for f in base.fills if f.order.decision_session_idx < 8] == [
        f for f in other.fills if f.order.decision_session_idx < 8
    ]
    assert [r for r in base.nav if r.session_idx <= 8] == [r for r in other.nav if r.session_idx <= 8]


def test_delisted_orders_rejected(tmp_path: Path) -> None:
    sessions = _sessions(3)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    rows.append(_flat_row(sessions[0], "KRX:B", 100))
    panel = _write_panel(tmp_path / "gold", "market_panel_delist", rows)
    _write_exits(panel, [_exit_row("KRX:B", sessions[0], last_close=100, exit_kind="traded_exit")])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=_ScriptedStrategy({0: {1: 1.0}}),
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    assert result.fills == ()
    assert any(r.reason == "delisted" for r in result.rejects)
    assert all(e.kind.value != "exit_proceeds" for e in result.journal)


def test_carried_halted_order_executes_next_session(tmp_path: Path) -> None:
    sessions = _sessions(4)
    rows = [_flat_row(day, "KRX:A", 100, volume=0 if day == sessions[1] else 1000) for day in sessions]
    results = {}
    for carry in (True, False):
        panel = _write_panel(tmp_path / "gold", f"market_panel_carry_{int(carry)}", rows)
        _write_exits(panel, [])
        arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / f"cache_{int(carry)}")
        events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
        results[carry] = run_backtest(
            arrays=arrays,
            events=events,
            strategy=_ScriptedStrategy({0: {0: 1.0}}),
            config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0", carry=carry),
            deposits={},
            asof_tables={},
            rules=_rules(),
            start=sessions[0],
            end=sessions[-1],
        )
    carried, expired = results[True], results[False]
    assert any(r.reason == "halted" for r in carried.rejects)
    assert len(carried.fills) == 1
    assert carried.fills[0].order.decision_session_idx == 1
    buys = [e for e in carried.journal if e.kind.value == "buy"]
    assert [b.session_idx for b in buys] == [2]
    assert expired.fills == ()


def test_reverse_split_shortfall_rejected_as_cash(tmp_path: Path) -> None:
    sessions = _sessions(4)
    rows = [_flat_row(day, "KRX:A", 100, share_factor=0.5 if day == sessions[2] else 1.0) for day in sessions]
    rows += [_flat_row(day, "KRX:B", 100) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_split_sell", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=_ScriptedStrategy({0: {0: 1.0}, 1: {1: 1.0}}),
        config=_config(initial_cash=1_000, impact_k=0.0, commission="0"),
        deposits={},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    sells = [f for f in result.fills if f.order.side == Side.SELL]
    assert [f.quantity for f in sells] == [5]
    assert any(r.reason == "cash" for r in result.rejects)


def test_buy_shortfall_uses_commission_aware_reduction(tmp_path: Path) -> None:
    sessions = _sessions(3)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    rows[1] = _flat_row(sessions[1], "KRX:A", 300, open=300, upper_limit=1000)
    panel = _write_panel(tmp_path / "gold", "market_panel_gap", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=_ScriptedStrategy({0: {0: 1.0}}),
        config=_config(initial_cash=905, impact_k=0.0, commission="0.01"),
        deposits={},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    assert [(f.quantity, f.price) for f in result.fills] == [(2, 300)]
    assert any(r.reason == "cash" for r in result.rejects)


def test_unaffordable_buy_rejected_as_cash(tmp_path: Path) -> None:
    sessions = _sessions(3)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    rows[1] = _flat_row(sessions[1], "KRX:A", 300, open=300, upper_limit=1000)
    panel = _write_panel(tmp_path / "gold", "market_panel_broke", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=_ScriptedStrategy({0: {0: 1.0}}),
        config=_config(initial_cash=150, impact_k=0.0, commission="0"),
        deposits={},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    assert result.fills == ()
    assert any(r.reason == "cash" for r in result.rejects)


def test_zero_close_target_skipped(tmp_path: Path) -> None:
    sessions = _sessions(2)
    rows = [_flat_row(day, "KRX:A", 0, open=0, high=0, low=0, base_price=1) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_zero", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=_ScriptedStrategy({0: {0: 1.0}}),
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    assert result.fills == ()
    assert result.rejects == ()


def test_bad_window_and_deposits_rejected(tmp_path: Path) -> None:
    sessions = _sessions(3)
    panel = _write_panel(
        tmp_path / "gold", "market_panel_bad", [_flat_row(day, "KRX:A", 100) for day in sessions]
    )
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    strategy = _ScriptedStrategy({})
    config = _config()
    with pytest.raises(ValueError):
        run_backtest(
            arrays=arrays, events=events, strategy=strategy, config=config, deposits={},
            asof_tables={}, rules=_rules(), start=date(2020, 1, 5), end=sessions[-1],
        )
    with pytest.raises(ValueError):
        run_backtest(
            arrays=arrays, events=events, strategy=strategy, config=config,
            deposits={sessions[-1] + timedelta(days=1): 100}, asof_tables={}, rules=_rules(),
            start=sessions[0], end=sessions[-1],
        )
    with pytest.raises(ValueError):
        EngineConfig(
            initial_cash=1, execution=_execution(), costs=_costs(),
            halted_exit_policy=DelistPolicy.LAST_CLOSE, cash_buffer=1.0,
        )


@pytest.mark.slow
def test_reference_benchmark_parity() -> None:
    """Test-only frictionless close-fill model reproduces eligible_ew_pr."""
    root = Path("data/gold/kr_swing_2019_v1")
    panel = root / "market_panel_1ea26065dc4dd5a9"
    bench = root / "reference_benchmarks_43e897f6e9afb0ed" / "benchmarks.parquet"
    parts = sorted(
        str(p) for p in panel.rglob("*.parquet") if p.name != "instrument_exits.parquet"
    )
    frame = (
        pl.scan_parquet(parts)
        .filter(pl.col("session").dt.year().is_in([2020, 2021]))
        .select("session", "instrument_id", "eligible", "price_state", "ret_price", "close", "share_factor")
        .collect()
    )
    stored = {
        row["session"]: row["ret"]
        for row in pl.read_parquet(bench)
        .filter(pl.col("benchmark_id") == "eligible_ew_pr")
        .select("session", "ret")
        .to_dicts()
    }
    by_session = frame.partition_by("session", as_dict=True)
    sessions_all = sorted(s for (s,) in by_session)
    diffs: list[float] = []
    for current, nxt in zip(sessions_all[:-1], sessions_all[1:]):
        if current.year != 2020:
            continue
        prev_rows = by_session[(current,)].filter(
            pl.col("eligible") & (pl.col("price_state") == "tradable") & (pl.col("close") > 0)
        )
        curr_rows = by_session[(nxt,)].select(
            "instrument_id",
            pl.col("close").alias("close_next"),
            pl.col("share_factor").alias("factor_next"),
            pl.col("ret_price").alias("ret_next"),
        )
        kept = prev_rows.join(curr_rows, on="instrument_id", how="inner").filter(
            pl.col("ret_next").is_not_null() & (pl.col("close_next") > 0)
        )
        growth = float(
            kept.select(
                (pl.col("factor_next").fill_null(1.0) * pl.col("close_next") / pl.col("close")).alias("g")
            )["g"].mean()
        )
        diffs.append(abs(growth - 1.0 - stored[nxt]))
    assert len(diffs) >= 200
    assert max(diffs) <= 1e-9
