"""Replay engine: timeline order, fills, exits, determinism, and benchmark parity."""

from __future__ import annotations

import itertools
import math

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.backtest.costs import CostConfig, Side
from src.backtest.engine import BacktestResult, DelistPolicy, EngineConfig, run_backtest
from src.backtest.events import build_engine_events
from src.backtest.execution import ExecutionConfig, ExecutionScenario
from src.backtest.market import MarketArrays, load_market_arrays
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


def _costs(commission: str = "0.00015", impact_k: float = 0.5, withholding: str = "0") -> CostConfig:
    return CostConfig(
        commission_rate=Decimal(commission),
        impact_k=impact_k,
        dividend_withholding_rate=Decimal(withholding),
    )


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


def _positions_at(journal: Any, t: int, *, strict: bool) -> dict[int, int]:
    pos: dict[int, int] = {}
    for entry in journal:
        if (entry.session_idx < t or (not strict and entry.session_idx == t)) and entry.instrument_idx is not None:
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
        targets={0: {0: 1.0 / 3.0, 1: 1.0 / 3.0, 2: 1.0 / 3.0}},
        config=_config(),
        deposits={sessions[0]: 1_000_000, date(2020, 1, 1): 2_000_000, sessions[15]: 500_000},
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
        targets={0: {0: 0.01}},
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    buys = [e for e in result.journal if e.kind.value == "buy"]
    assert len(buys) == 1
    assert buys[0].session_idx == 1
    assert buys[0].cash_delta == -100 * 105
    assert buys[0].quantity_delta == 100


def test_decision_before_window_executes_first_session(tmp_path: Path) -> None:
    sessions = _sessions(3)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_pre", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    result = run_backtest(
        arrays=arrays,
        events=events,
        targets={0: {0: 0.5}},
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={},
        rules=_rules(),
        start=sessions[1],
        end=sessions[-1],
    )
    buys = [e for e in result.journal if e.kind.value == "buy"]
    assert len(buys) == 1
    assert buys[0].session_idx == 1
    assert [r.session_idx for r in result.nav] == [1, 2]


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
        targets={0: {0: 1.0}, 2: {1: 1.0}},
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    assert not [r for r in result.rejects if r.reason == "cash"]
    b_buys = [e for e in result.journal if e.kind.value == "buy" and e.instrument_idx == 1]
    a_sells = [e for e in result.journal if e.kind.value == "sell" and e.instrument_idx == 0]
    assert len(b_buys) == 1
    assert len(a_sells) == 1
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
            targets={0: {1: 1.0}},
            config=_config(initial_cash=1_500_000, impact_k=0.0, commission="0", policy=policy),
            deposits={},
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
        targets={0: {0: 1.0 / 3.0, 1: 1.0 / 3.0, 2: 1.0 / 3.0}},
        config=_config(),
        deposits={sessions[0]: 1_000_000, date(2020, 1, 1): 2_000_000, sessions[15]: 500_000},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    assert first.ledger_hash == second.ledger_hash
    assert first.nav == second.nav
    assert first.fills == second.fills
    assert first.rejects == second.rejects
    assert first.journal == second.journal
    assert np.array_equal(first.stock_book_returns, second.stock_book_returns, equal_nan=True)


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

    perturbed: list[dict[str, Any]] = []
    for t, day in enumerate(sessions):
        for inst, base in (("KRX:A", 100 + t), ("KRX:B", 200 - t)):
            close = base if t <= 8 else int(base * 1.05) + 7
            perturbed.append(_flat_row(day, inst, close))
    panel2 = _write_panel(tmp_path / "gold", "market_panel_perturbed", perturbed)
    _write_exits(panel2, [])
    arrays2 = load_market_arrays(panel_dir=panel2, cache_root=tmp_path / "cache")
    events2 = build_engine_events(arrays=arrays2, panel_dir=panel2, dividends=None)

    schedule = {t: {0: 0.5, 1: 0.5} for t in range(11)}
    kwargs: dict[str, Any] = {
        "deposits": {},
        "rules": _rules(),
        "start": sessions[0],
        "end": sessions[-1],
    }
    base = run_backtest(
        arrays=arrays, events=events, targets=schedule,
        config=_config(impact_k=0.0, commission="0"), **kwargs,
    )
    other = run_backtest(
        arrays=arrays2, events=events2, targets=schedule,
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
        targets={0: {1: 1.0}},
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={},
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
            targets={0: {0: 1.0}},
            config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0", carry=carry),
            deposits={},
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
        targets={0: {0: 1.0}, 1: {1: 1.0}},
        config=_config(initial_cash=1_000, impact_k=0.0, commission="0"),
        deposits={},
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
        targets={0: {0: 1.0}},
        config=_config(initial_cash=905, impact_k=0.0, commission="0.01"),
        deposits={},
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
        targets={0: {0: 1.0}},
        config=_config(initial_cash=150, impact_k=0.0, commission="0"),
        deposits={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    assert result.fills == ()
    assert any(r.reason == "cash" for r in result.rejects)


def test_sell_shortfall_carries_a_retry_order(tmp_path: Path) -> None:
    """A reverse split shrinks the holding below the sell order; the unsold remainder is re-ordered."""
    sessions = _sessions(5)
    rows = [
        _flat_row(day, "KRX:A", 100, share_factor=0.5 if day == sessions[2] else 1.0)
        for day in sessions
    ]
    rows += [_flat_row(day, "KRX:B", 100) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_carry_sell", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)

    result = run_backtest(
        arrays=arrays,
        events=events,
        targets={0: {0: 1.0}, 1: {1: 1.0}},
        config=_config(initial_cash=1_000, impact_k=0.0, commission="0", carry=True),
        deposits={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )

    sells = [e for e in result.journal if e.kind.value == "sell"]
    assert [(e.session_idx, e.quantity_delta) for e in sells] == [(2, -5)]
    unsold = [
        (r.order.decision_session_idx, r.order.quantity)
        for r in result.rejects
        if r.reason == "cash" and r.order.instrument_idx == 0
    ]
    assert unsold == [(1, 10), (2, 5), (3, 5)]


def test_cash_shortfall_carries_a_retry_order(tmp_path: Path) -> None:
    """A cash-truncated buy is journaled as a reject and, with carry, re-ordered for the next session."""
    sessions = _sessions(5)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    rows[1] = _flat_row(sessions[1], "KRX:A", 300, open=300, upper_limit=1000)
    panel = _write_panel(tmp_path / "gold", "market_panel_carry_cash", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)

    result = run_backtest(
        arrays=arrays,
        events=events,
        targets={0: {0: 1.0}},
        config=_config(initial_cash=905, impact_k=0.0, commission="0.01", carry=True),
        deposits={sessions[2]: 1_000},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )

    buys = [e for e in result.journal if e.kind.value == "buy"]
    assert [(e.session_idx, e.quantity_delta) for e in buys] == [(1, 2), (2, 7)]


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
        targets={0: {0: 1.0}},
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={},
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
    config = _config()
    with pytest.raises(ValueError, match="not within the panel sessions"):
        run_backtest(
            arrays=arrays, events=events, targets={}, config=config, deposits={},
            rules=_rules(), start=date(2020, 1, 5), end=sessions[-1],
        )
    with pytest.raises(ValueError, match="after the run end"):
        run_backtest(
            arrays=arrays, events=events, targets={}, config=config,
            deposits={sessions[-1] + timedelta(days=1): 100}, rules=_rules(),
            start=sessions[0], end=sessions[-1],
        )
    with pytest.raises(ValueError, match="cash_buffer"):
        EngineConfig(
            initial_cash=1, execution=_execution(), costs=_costs(),
            halted_exit_policy=DelistPolicy.LAST_CLOSE, cash_buffer=1.0,
        )


def test_invalid_target_schedules_fail_fast(tmp_path: Path) -> None:
    sessions = _sessions(4)
    panel = _write_panel(
        tmp_path / "gold", "market_panel_targets", [_flat_row(day, "KRX:A", 100) for day in sessions]
    )
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    config = _config()
    base: dict[str, Any] = {
        "arrays": arrays, "events": events, "config": config, "deposits": {},
        "rules": _rules(), "start": sessions[0], "end": sessions[-1],
    }
    with pytest.raises(ValueError, match="non-negative"):
        run_backtest(targets={0: {0: -0.5}}, **base)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        run_backtest(targets={0: {0: math.inf}}, **base)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="above 1"):
        run_backtest(targets={0: {0: 1.5}}, **base)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outside the run window"):
        run_backtest(targets={3: {0: 0.5}}, **base)
    with pytest.raises(ValueError, match="outside the run window"):
        run_backtest(targets={-1: {0: 0.5}}, **base)
    with pytest.raises(ValueError, match="outside the panel"):
        run_backtest(targets={0: {9: 0.5}}, **base)
    with pytest.raises(ValueError, match="must be an int"):
        run_backtest(targets={True: {0: 0.5}}, **base)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="must be a mapping"):
        run_backtest(targets={0: [0.5]}, **base)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="must be an int"):
        run_backtest(targets={0: {True: 0.5}}, **base)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite number"):
        run_backtest(targets={0: {0: True}}, **base)  # type: ignore[dict-item]


def _sweep_arrays(tmp_path: Path, sessions: list[date]) -> MarketArrays:
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_sweep", rows)
    _write_exits(panel, [])
    return load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")


def test_cash_yield_credited_and_taxed_at_year_end(tmp_path: Path) -> None:
    import numpy as _np

    from src.backtest.ledger import JournalKind as _Kind

    sessions = [date(2020, 12, 30), date(2020, 12, 31), date(2021, 1, 4)]
    arrays = _sweep_arrays(tmp_path, sessions)
    events = build_engine_events(arrays=arrays, panel_dir=_panel_dir(tmp_path, "market_panel_sweep"), dividends=None)
    cash = _np.zeros(len(sessions))
    cash[1] = 0.0001
    result = run_backtest(
        arrays=arrays, events=events, targets={}, config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1], cash_returns=cash,
    )
    yields = [e for e in result.journal if e.kind is _Kind.CASH_YIELD]
    assert [(e.session_idx, e.cash_delta) for e in yields] == [(1, 100)]
    taxes = [e for e in result.journal if e.kind is _Kind.CASH_YIELD_TAX]
    assert len(taxes) == 1
    assert taxes[0].session_idx == 1
    assert taxes[0].cash_delta == -15
    assert result.nav[-1].cash == 1_000_000 + 100 - 15


def test_missing_cash_return_fails_closed(tmp_path: Path) -> None:
    import numpy as _np

    from src.core.pit import PITDataError as _PIT

    sessions = _sessions(3)
    arrays = _sweep_arrays(tmp_path, sessions)
    events = build_engine_events(arrays=arrays, panel_dir=_panel_dir(tmp_path, "market_panel_sweep"), dividends=None)
    cash = _np.zeros(len(sessions))
    cash[1] = _np.nan
    with pytest.raises(_PIT, match="cash return missing"):
        run_backtest(
            arrays=arrays, events=events, targets={}, config=_config(impact_k=0.0, commission="0"),
            deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1], cash_returns=cash,
        )
    with pytest.raises(ValueError, match="cash_returns must have shape"):
        run_backtest(
            arrays=arrays, events=events, targets={}, config=_config(),
            deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1],
            cash_returns=_np.zeros(len(sessions) + 1),
        )


def test_sweep_disabled_is_noop(tmp_path: Path) -> None:
    import numpy as _np

    from src.backtest.ledger import JournalKind as _Kind

    sessions = _sessions(3)
    arrays = _sweep_arrays(tmp_path, sessions)
    events = build_engine_events(arrays=arrays, panel_dir=_panel_dir(tmp_path, "market_panel_sweep"), dividends=None)
    config = _config(impact_k=0.0, commission="0")
    base = run_backtest(
        arrays=arrays, events=events, targets={}, config=config,
        deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1],
    )
    zeros = run_backtest(
        arrays=arrays, events=events, targets={}, config=config,
        deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1],
        cash_returns=_np.zeros(len(sessions)),
    )
    assert [r.nav for r in base.nav] == [r.nav for r in zeros.nav]
    assert not [e for e in zeros.journal if e.kind is _Kind.CASH_YIELD_TAX]
    assert base.ledger_hash == zeros.ledger_hash


def test_final_record_includes_year_end_tax(tmp_path: Path) -> None:
    """The year-end assessment runs before the close mark, so the last record already carries the tax."""
    import numpy as _np

    from src.backtest.ledger import JournalKind as _Kind, LedgerAccount as _Account

    sessions = [date(2020, 12, 29), date(2020, 12, 30)]
    arrays = _sweep_arrays(tmp_path, sessions)
    events = build_engine_events(arrays=arrays, panel_dir=_panel_dir(tmp_path, "market_panel_sweep"), dividends=None)
    result = run_backtest(
        arrays=arrays, events=events, targets={},
        config=_config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1],
        cash_returns=_np.array([0.0, 0.01]),
    )
    assert [(e.session_idx, e.cash_delta) for e in result.journal if e.kind is _Kind.CASH_YIELD_TAX] == [
        (1, -1_540)
    ]
    assert result.nav[-1].cash == 1_008_460
    assert result.nav[-1].nav == 1_008_460
    cash_deltas = sum(e.cash_delta for e in result.journal if e.account is _Account.CASH)
    assert cash_deltas == 8_460
    assert result.nav[-1].cash == 1_000_000 + cash_deltas


def test_year_end_tax_beyond_cash_is_funded_by_next_session_sales(tmp_path: Path) -> None:
    import numpy as _np

    from src.backtest.ledger import JournalKind as _Kind, LedgerAccount as _Account

    sessions = [date(2020, 12, 29), date(2020, 12, 30), date(2020, 12, 31), date(2021, 1, 4)]
    rows = [_flat_row(day, "KRX:A", 100, sell_tax_rate=0.0) for day in sessions]
    rows += [_flat_row(day, "KRX:B", 200, sell_tax_rate=0.0) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_payable", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    result = run_backtest(
        arrays=arrays, events=events, targets={0: {0: 1.0}, 2: {0: 0.5, 1: 0.05}},
        config=_config(initial_cash=10_000_000, impact_k=0.0, commission="0"),
        deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1],
        cash_returns=_np.array([5.0, 0.0, 0.0, 0.0]),
    )

    assessed = result.nav[2]
    assert (assessed.cash, assessed.tax_payable, assessed.market_value) == (0, 7_700_000, 60_000_000)
    assert assessed.nav == 60_000_000 - 7_700_000
    settled = result.nav[3]
    assert (settled.tax_payable, settled.cash, settled.nav) == (0, 23_535_000, 52_300_000)

    order = [(entry.kind, entry.account) for entry in result.journal if entry.session_idx == 3]
    assert order.index((_Kind.SELL, _Account.CASH)) < order.index(
        (_Kind.TAX_PAYMENT, _Account.CASH)
    ) < order.index((_Kind.BUY, _Account.CASH))
    unpaid = sum(
        entry.cash_delta for entry in result.journal if entry.account is _Account.PAYABLE
    )
    assert unpaid == 0
    assert settled.cash == 10_000_000 + sum(
        entry.cash_delta for entry in result.journal if entry.account is _Account.CASH
    )


def test_long_target_executes_and_conserves_krw(tmp_path: Path) -> None:
    from src.backtest.ledger import JournalKind as _Kind, LedgerAccount
    from src.backtest.overlay import DerivativeConfig, OverlayMarket, OverlayState, OverlayTarget

    sessions = _sessions(4)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_long", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)

    class _Long:
        def target(self, state: OverlayState) -> OverlayTarget:
            return OverlayTarget(contracts=-1, inverse_value_krw=0)

    derivatives = DerivativeConfig(
        contract_multiplier_krw=10_000, initial_margin_rate=0.1, margin_buffer_rate=0.05,
        margin_topup_trigger_fraction=1.0, futures_cost_rate=0.0, inverse_cost_rate=0.0,
        futures_tax_rate=Decimal("0.11"), futures_annual_deduction_krw=2_500_000,
        inverse_tax_rate=Decimal("0.154"),
    )
    market = OverlayMarket(
        index_level=np.asarray([1000.0, 1000.0, 1030.0, 1060.0], dtype=np.float64),
        inverse_close=np.asarray([10_000.0] * 4, dtype=np.float64),
    )
    result = run_backtest(
        arrays=arrays, events=events, targets={0: {0: 0.5}}, config=_config(initial_cash=10_000_000, commission="0", impact_k=0.0),
        deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1],
        overlay=_Long(), overlay_market=market, derivatives=derivatives,
    )
    for record in result.nav:
        journal = [e for e in result.journal if e.session_idx <= record.session_idx]
        assert record.cash == 10_000_000 + sum(e.cash_delta for e in journal if e.account is LedgerAccount.CASH)
        assert record.margin == sum(e.cash_delta for e in journal if e.account is LedgerAccount.MARGIN)
        assert record.nav == (
            record.cash + record.dividend_receivable + record.market_value
            + record.margin + record.inverse_value - record.tax_payable
        )
    variations = [e.cash_delta for e in result.journal if e.kind is _Kind.VARIATION_MARGIN]
    assert sum(variations) == 600_000
    assert result.nav[0].margin == 0
    assert result.nav[1].market_value == 4_250_000
    assert [(e.session_idx, e.quantity_delta) for e in result.journal if e.kind is _Kind.FUTURES_TRADE] == [(1, -1)]


def test_unaffordable_long_shrinks_never_flips(tmp_path: Path) -> None:
    from src.backtest.ledger import JournalKind as _Kind
    from src.backtest.overlay import DerivativeConfig, OverlayMarket, OverlayState, OverlayTarget

    sessions = _sessions(4)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_long_thin", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)

    class _BigLong:
        def target(self, state: OverlayState) -> OverlayTarget:
            return OverlayTarget(contracts=-5, inverse_value_krw=0)

    derivatives = DerivativeConfig(
        contract_multiplier_krw=10_000, initial_margin_rate=0.1, margin_buffer_rate=0.05,
        margin_topup_trigger_fraction=1.0, futures_cost_rate=0.0, inverse_cost_rate=0.0,
        futures_tax_rate=Decimal("0.11"), futures_annual_deduction_krw=2_500_000,
        inverse_tax_rate=Decimal("0.154"),
    )
    market = OverlayMarket(
        index_level=np.asarray([1000.0] * 4, dtype=np.float64),
        inverse_close=np.asarray([10_000.0] * 4, dtype=np.float64),
    )
    result = run_backtest(
        arrays=arrays, events=events, targets={}, config=_config(initial_cash=2_000_000),
        deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1],
        overlay=_BigLong(), overlay_market=market, derivatives=derivatives,
    )
    held = 0
    for entry in result.journal:
        if entry.kind is _Kind.FUTURES_TRADE:
            held += entry.quantity_delta
            assert held <= 0
    assert held == -1


def test_bounded_fundable_search_negative() -> None:
    import time

    from src.backtest.engine import _largest_fundable_contracts
    from src.backtest.overlay import DerivativeConfig, required_reserve_krw
    from src.backtest.engine import _futures_commission

    config = DerivativeConfig(
        contract_multiplier_krw=10_000, initial_margin_rate=0.1, margin_buffer_rate=0.05,
        margin_topup_trigger_fraction=1.0, futures_cost_rate=0.0005, inverse_cost_rate=0.0,
        futures_tax_rate=Decimal("0.11"), futures_annual_deduction_krw=2_500_000,
        inverse_tax_rate=Decimal("0.154"),
    )
    level, mult = 1000.0, 10_000

    def _reference(target: int, held: int, available: int) -> int:
        candidate = target
        while candidate <= 0:
            reserve = required_reserve_krw(contracts=candidate, level=level, config=config)
            commission = _futures_commission(
                float(config.futures_cost_rate), delta=candidate - held,
                multiplier=mult, level=level,
            )
            if reserve + commission <= available:
                return candidate
            candidate += 1
        return 0

    for target, held, available in [(-5, 0, 10_000_000), (-5, -2, 10_000_000), (-7, 1, 3_000_000)]:
        assert _largest_fundable_contracts(
            target_contracts=target, held=held, available=available, level=level,
            config=config, multiplier=mult,
        ) == _reference(target, held, available)

    started = time.perf_counter()
    got = _largest_fundable_contracts(
        target_contracts=-(10**8), held=0, available=10_000_000, level=level,
        config=config, multiplier=mult,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05
    assert got == _reference(-10, 0, 10_000_000) == -6


@pytest.mark.slow
def test_reference_benchmark_parity() -> None:
    """Test-only frictionless close-fill model reproduces eligible_ew_pr."""
    import pytest

    from src.data.dataset_registry import DatasetRegistry

    scope = "kr_swing_2019_v1"
    state_dir = Path("data/state") / scope
    if not (state_dir / "datasets.json").exists():
        pytest.skip("real-data registry absent")
    registry = DatasetRegistry(state_dir)
    root = Path("data/gold") / scope
    panel = root / registry.require("market_panel")
    bench = next((root / registry.require("reference_benchmarks")).glob("*.parquet"))
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
    for current, nxt in itertools.pairwise(sessions_all):
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


def _band_arrays() -> Any:
    import numpy as _np

    from types import SimpleNamespace as _NS

    close = _np.array([[1000, 1000]], dtype=_np.int64)
    present = _np.array([[True, True]])
    return _NS(int_fields={"close": close}, bool_fields={"present": present})


def test_band_skips_small_drift_trades_large_drift() -> None:
    from src.backtest.costs import Side as _Side
    from src.backtest.engine import _target_orders as _orders

    arrays = _band_arrays()
    weight = {0: 0.1}
    # nav 1M, close 1000 -> target 100 shares.
    assert _orders(
        weights=weight, arrays=arrays, t=0, holdings={0: 120}, nav=1_000_000,
        buffer_scale=1.0, rebalance_band=0.5,
    ) == []
    got = _orders(
        weights=weight, arrays=arrays, t=0, holdings={0: 160}, nav=1_000_000,
        buffer_scale=1.0, rebalance_band=0.5,
    )
    assert len(got) == 1
    assert got[0].side is _Side.SELL
    assert got[0].quantity == 60
    boundary = _orders(
        weights=weight, arrays=arrays, t=0, holdings={0: 150}, nav=1_000_000,
        buffer_scale=1.0, rebalance_band=0.5,
    )
    assert len(boundary) == 1
    assert boundary[0].quantity == 50
    assert _orders(
        weights=weight, arrays=arrays, t=0, holdings={0: 120}, nav=1_000_000, buffer_scale=1.0,
    ) != []


def test_band_never_suppresses_exits_or_entries() -> None:
    from src.backtest.costs import Side as _Side
    from src.backtest.engine import _target_orders as _orders

    arrays = _band_arrays()
    exited = _orders(
        weights={}, arrays=arrays, t=0, holdings={0: 120}, nav=1_000_000,
        buffer_scale=1.0, rebalance_band=0.9,
    )
    assert len(exited) == 1
    assert exited[0].side is _Side.SELL
    assert exited[0].quantity == 120
    for weights in ({0: 0.0}, {0: 0.0001}):
        zero_target = _orders(
            weights=weights, arrays=arrays, t=0, holdings={0: 120}, nav=1_000_000,
            buffer_scale=1.0, rebalance_band=0.9,
        )
        assert len(zero_target) == 1
        assert zero_target[0].quantity == 120
    entered = _orders(
        weights={1: 0.1}, arrays=arrays, t=0, holdings={}, nav=1_000_000,
        buffer_scale=1.0, rebalance_band=0.9,
    )
    assert len(entered) == 1
    assert entered[0].side is _Side.BUY
    assert entered[0].quantity == 100


def test_zero_band_is_identity_and_range_validated(tmp_path: Path) -> None:
    sessions = _sessions(3)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_band", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    kwargs: dict[str, Any] = {
        "arrays": arrays, "events": events, "targets": {0: {0: 0.5}},
        "config": _config(initial_cash=1_000_000, impact_k=0.0, commission="0"),
        "deposits": {}, "rules": _rules(), "start": sessions[0], "end": sessions[-1],
    }
    default = run_backtest(**kwargs)
    explicit = run_backtest(**kwargs, rebalance_band=0.0)
    assert default.ledger_hash == explicit.ledger_hash
    for bad in (-0.1, 1.0, math.inf, math.nan, True):
        with pytest.raises(ValueError, match="rebalance_band"):
            run_backtest(**kwargs, rebalance_band=bad)
