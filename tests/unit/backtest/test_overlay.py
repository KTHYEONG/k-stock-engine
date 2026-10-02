"""Account overlay invariants: futures margin, inverse ETF, and the engine close phase."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.backtest.engine import BacktestResult, EngineConfig, _sync_futures, run_backtest
from src.backtest.costs import CostConfig
from src.backtest.events import EngineEvents, build_engine_events
from src.backtest.ledger import JournalKind, Ledger, LedgerAccount
from src.backtest.market import MarketArrays, load_market_arrays
from src.backtest.overlay import (
    DerivativeConfig,
    OverlayMarket,
    OverlayState,
    OverlayTarget,
    futures_expiry_rows,
    required_reserve_krw,
)
from src.core.pit import PITDataError
from tests.unit.backtest.test_engine import _config, _flat_row, _rules
from tests.unit.backtest.test_events import _div_frame, _div_row, _write_exits
from tests.unit.backtest.test_market import _write_panel


def _derivatives(**over: Any) -> DerivativeConfig:
    base: dict[str, Any] = {
        "contract_multiplier_krw": 10_000,
        "initial_margin_rate": 0.1,
        "margin_buffer_rate": 0.05,
        "margin_topup_trigger_fraction": 1.0,
        "futures_cost_rate": 0.0,
        "inverse_cost_rate": 0.0,
        "futures_tax_rate": Decimal("0.11"),
        "futures_annual_deduction_krw": 2_500_000,
        "inverse_tax_rate": Decimal("0.154"),
    }
    base.update(over)
    return DerivativeConfig(**base)


def _sessions(n: int, start: date = date(2020, 1, 6)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _span(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


class _Policy:
    """Overlay policy driven by a callable, recording every state it is handed."""

    def __init__(self, decide: Callable[[OverlayState], OverlayTarget | None]) -> None:
        self._decide = decide
        self.states: list[OverlayState] = []

    def target(self, state: OverlayState) -> OverlayTarget | None:
        self.states.append(state)
        return self._decide(state)


def _hold(by_row: Mapping[int, tuple[int, int]]) -> _Policy:
    """Hedge by panel row: ``_hold({0: (2, 20_000)})`` shorts 2 contracts plus a 20,000 KRW inverse ETF."""
    table = {row: OverlayTarget(contracts, value) for row, (contracts, value) in by_row.items()}

    def decide(state: OverlayState) -> OverlayTarget | None:
        return table.get(state.session_idx)

    return _Policy(decide)


def _flat_policy() -> _Policy:
    return _Policy(lambda _state: None)


def _arrays(
    tmp_path: Path,
    name: str,
    sessions: Sequence[date],
    *,
    closes: Sequence[int] | int = 100,
    opens: Sequence[int] | None = None,
) -> tuple[MarketArrays, EngineEvents]:
    series = [closes] * len(sessions) if isinstance(closes, int) else list(closes)
    open_series = series if opens is None else list(opens)
    rows = [
        _flat_row(day, "KRX:A", series[idx], open=open_series[idx])
        for idx, day in enumerate(sessions)
    ]
    panel = _write_panel(tmp_path / "gold", name, rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    return arrays, build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)


def _market(
    sessions: Sequence[date],
    levels: Sequence[float] | float,
    *,
    inverse: Sequence[float] | float = 10_000.0,
) -> OverlayMarket:
    n = len(sessions)
    level_values = [float(levels)] * n if isinstance(levels, (int, float)) else list(levels)
    inverse_values = [float(inverse)] * n if isinstance(inverse, (int, float)) else list(inverse)
    return OverlayMarket(
        index_level=np.asarray(level_values, dtype=np.float64),
        inverse_close=np.asarray(inverse_values, dtype=np.float64),
    )


def _run(
    tmp_path: Path,
    *,
    name: str,
    sessions: Sequence[date],
    market: OverlayMarket,
    derivatives: DerivativeConfig,
    policy: _Policy,
    config: EngineConfig,
    targets: Mapping[int, Mapping[int, float]] | None = None,
    deposits: Mapping[date, int] | None = None,
    closes: Sequence[int] | int = 100,
    start: date | None = None,
    end: date | None = None,
    cash_returns: np.ndarray | None = None,
) -> BacktestResult:
    arrays, events = _arrays(tmp_path, name, sessions, closes=closes)
    return run_backtest(
        arrays=arrays,
        events=events,
        targets=dict(targets or {}),
        config=config,
        deposits=dict(deposits or {}),
        rules=_rules(),
        start=start or sessions[0],
        end=end or sessions[-1],
        cash_returns=cash_returns,
        overlay=policy,
        overlay_market=market,
        derivatives=derivatives,
    )


def _kinds(result: BacktestResult, kind: JournalKind) -> list[Any]:
    return [entry for entry in result.journal if entry.kind is kind]


def _account_sum(result: BacktestResult, account: LedgerAccount) -> int:
    return sum(entry.cash_delta for entry in result.journal if entry.account is account)


def _assert_reconciles(result: BacktestResult, initial_cash: int) -> None:
    """Every account equals its journal sum, the closing NAV equals the closing balances, and cash never went negative."""
    last = result.nav[-1]
    assert last.cash == initial_cash + _account_sum(result, LedgerAccount.CASH)
    assert last.margin == _account_sum(result, LedgerAccount.MARGIN)
    assert last.tax_payable == -_account_sum(result, LedgerAccount.PAYABLE)
    assert last.nav == (
        last.cash
        + last.dividend_receivable
        + last.market_value
        + last.margin
        + last.inverse_value
        - last.tax_payable
    )
    assert all(record.cash >= 0 for record in result.nav)


def test_expiry_rows_are_the_second_thursdays() -> None:
    sessions = _span(date(2025, 1, 1), date(2025, 12, 31))

    rows = futures_expiry_rows(sessions)

    assert len(rows) == 4
    for expiry in (
        date(2025, 3, 13),
        date(2025, 6, 12),
        date(2025, 9, 11),
        date(2025, 12, 11),
    ):
        assert sessions.index(max(day for day in sessions if day <= expiry)) in rows
    assert sessions.index(date(2025, 3, 13)) in rows


def test_expiry_rows_fall_back_to_the_prior_session() -> None:
    sessions = [
        day for day in _span(date(2025, 3, 10), date(2025, 3, 14)) if day != date(2025, 3, 13)
    ]

    rows = futures_expiry_rows(sessions)

    assert len(rows) == 1
    assert sessions[next(iter(rows))] == date(2025, 3, 12)
    assert futures_expiry_rows([]) == frozenset()


def test_roll_is_charged_once_on_the_margin_account(tmp_path: Path) -> None:
    sessions = _span(date(2025, 3, 10), date(2025, 3, 14))
    config = _config()
    expiry_row = sessions.index(date(2025, 3, 13))

    result = _run(
        tmp_path,
        name="ov_roll_hold",
        sessions=sessions,
        market=_market(sessions, 1000.0),
        derivatives=_derivatives(futures_cost_rate=0.001),
        policy=_Policy(lambda _state: OverlayTarget(contracts=1, inverse_value_krw=0)),
        config=config,
    )

    rolls = _kinds(result, JournalKind.FUTURES_ROLL)
    assert [(entry.session_idx, entry.cash_delta, entry.account) for entry in rolls] == [
        (expiry_row, -math.ceil(2 * 0.001 * 10_000 * 1000.0), LedgerAccount.MARGIN)
    ]
    # The 20,000 roll cost stays inside the [M, R] tolerance band: no top-up from cash.
    assert result.nav[expiry_row].margin == result.nav[expiry_row - 1].margin - 20_000
    assert result.nav[expiry_row].cash == result.nav[expiry_row - 1].cash
    assert not [
        entry
        for entry in result.journal
        if entry.session_idx == expiry_row and entry.kind is JournalKind.MARGIN_TRANSFER
    ]
    _assert_reconciles(result, config.initial_cash)


def test_roll_is_skipped_without_a_position(tmp_path: Path) -> None:
    sessions = _span(date(2025, 3, 10), date(2025, 3, 14))
    config = _config()

    result = _run(
        tmp_path,
        name="ov_roll_flat",
        sessions=sessions,
        market=_market(sessions, 1000.0),
        derivatives=_derivatives(futures_cost_rate=0.01),
        policy=_flat_policy(),
        config=config,
    )

    assert _kinds(result, JournalKind.FUTURES_ROLL) == []
    assert all(record.margin == 0 for record in result.nav)
    _assert_reconciles(result, config.initial_cash)


def test_excess_margin_is_released_at_every_close(tmp_path: Path) -> None:
    """An unchanged 1-contract target still returns the short's gain: margin back to the 1,050,000 reserve."""
    sessions = _sessions(4)
    config = _config(initial_cash=10_000_000, commission="0", impact_k=0.0)
    derivatives = _derivatives()

    result = _run(
        tmp_path,
        name="ov_release_excess",
        sessions=sessions,
        market=_market(sessions, [1000.0, 1000.0, 700.0, 700.0]),
        derivatives=derivatives,
        policy=_Policy(lambda _state: OverlayTarget(contracts=1, inverse_value_krw=0)),
        config=config,
        targets={2: {0: 1.0}},
        closes=100,
    )

    assert [record.margin for record in result.nav[:3]] == [0, 1_500_000, 1_050_000]
    assert result.nav[2].cash == 11_950_000
    assert [
        entry.cash_delta
        for entry in _kinds(result, JournalKind.MARGIN_TRANSFER)
        if entry.account is LedgerAccount.CASH
    ] == [-1_500_000, 3_450_000]
    assert not [reject for reject in result.rejects if reject.reason == "cash"]
    assert sum(fill.quantity * fill.price for fill in result.fills) == 11_950_000
    # The last session is a year end: the annual futures tax is booked after the close's margin sync.
    assert [entry.cash_delta for entry in _kinds(result, JournalKind.FUTURES_TAX)] == [-55_000]
    assert result.nav[3].margin == 995_000
    _assert_reconciles(result, config.initial_cash)


def test_maintenance_tops_margin_up_from_cash(tmp_path: Path) -> None:
    """A +5% index day costs 500,000 of margin; maintenance restores the full 1,575,000 reserve from cash."""
    sessions = _sessions(4)
    config = _config()
    derivatives = _derivatives()

    result = _run(
        tmp_path,
        name="ov_topup",
        sessions=sessions,
        market=_market(sessions, [1000.0, 1000.0, 1050.0, 1050.0]),
        derivatives=derivatives,
        policy=_hold({0: (1, 0)}),
        config=config,
    )

    assert required_reserve_krw(contracts=1, level=1050.0, config=derivatives) == 1_575_000
    assert [record.margin for record in result.nav] == [0, 1_500_000, 1_575_000, 1_575_000]
    assert result.nav[2].cash == result.nav[1].cash - 575_000
    transfers = [
        entry.cash_delta
        for entry in _kinds(result, JournalKind.MARGIN_TRANSFER)
        if entry.account is LedgerAccount.CASH
    ]
    assert transfers == [-1_500_000, -575_000]
    assert [(entry.session_idx, entry.quantity_delta) for entry in _kinds(result, JournalKind.FUTURES_TRADE)] == [(1, 1)]
    _assert_reconciles(result, config.initial_cash)


def test_forced_reduction_instead_of_borrowing(tmp_path: Path) -> None:
    """A 1,620,000 reserve the account cannot fund after +8% forces the hedge flat rather than a loan."""
    sessions = _sessions(4)
    config = _config(initial_cash=1_600_000)

    result = _run(
        tmp_path,
        name="ov_force",
        sessions=sessions,
        market=_market(sessions, [1000.0, 1000.0, 1080.0, 1080.0]),
        derivatives=_derivatives(futures_cost_rate=0.0005),
        policy=_hold({0: (1, 0)}),
        config=config,
    )

    assert [(entry.session_idx, entry.quantity_delta) for entry in _kinds(result, JournalKind.FUTURES_TRADE)] == [
        (1, 1),
        (2, -1),
    ]
    assert result.nav[2].margin == 0
    assert result.nav[2].cash == 789_600
    assert [entry.cash_delta for entry in _kinds(result, JournalKind.FUTURES_COMMISSION)] == [-5_000, -5_400]
    _assert_reconciles(result, config.initial_cash)


def test_costs_never_crash_a_thin_cash_account(tmp_path: Path) -> None:
    """99 KRW of free cash, an expiry row and a year-end tax: both charges come off margin and the run finishes."""
    sessions = _span(date(2025, 12, 28), date(2026, 3, 14))
    year_end_row = sessions.index(date(2025, 12, 31))
    expiry_row = sessions.index(date(2026, 3, 12))
    config = _config(initial_cash=1_600_099)

    result = _run(
        tmp_path,
        name="ov_thin",
        sessions=sessions,
        market=_market(sessions, [100.0, 100.0] + [50.0] * (len(sessions) - 2)),
        derivatives=_derivatives(contract_multiplier_krw=100_000, futures_cost_rate=0.01),
        policy=_Policy(lambda _state: OverlayTarget(contracts=1, inverse_value_krw=0)),
        config=config,
    )

    assert result.nav[1].cash == 99
    assert [(e.session_idx, e.cash_delta, e.account) for e in _kinds(result, JournalKind.FUTURES_TAX)] == [
        (year_end_row, -275_000, LedgerAccount.MARGIN)
    ]
    assert [(e.session_idx, e.cash_delta, e.account) for e in _kinds(result, JournalKind.FUTURES_ROLL)] == [
        (expiry_row, -100_000, LedgerAccount.MARGIN)
    ]
    assert result.nav[year_end_row].tax_payable == 0
    assert len(result.nav) == len(sessions)
    _assert_reconciles(result, config.initial_cash)


def test_negative_margin_is_repaid_before_any_inverse_purchase(tmp_path: Path) -> None:
    """A -1,399,900 margin left by a forced unwind is repaid from a later deposit at the same close."""
    sessions = _sessions(6)
    config = _config(initial_cash=1_600_100, commission="0", impact_k=0.0)

    result = _run(
        tmp_path,
        name="ov_negative_margin",
        sessions=sessions,
        market=_market(sessions, [1000.0, 1000.0, 1300.0, 1300.0, 1300.0, 1300.0]),
        derivatives=_derivatives(),
        policy=_hold({0: (1, 0), 1: (1, 0), 2: (0, 0), 3: (0, 20_000)}),
        config=config,
        deposits={sessions[4]: 5_000_000},
    )

    assert [(record.margin, record.cash) for record in result.nav[:4]] == [
        (0, 1_600_100),
        (1_500_000, 100_100),
        (-1_399_900, 0),
        (-1_399_900, 0),
    ]
    assert (result.nav[4].margin, result.nav[4].cash) == (0, 3_580_100)
    assert result.nav[4].inverse_value == 20_000
    close_kinds = [
        (entry.kind, entry.account)
        for entry in result.journal
        if entry.session_idx == 4
        and entry.kind in (JournalKind.MARGIN_TRANSFER, JournalKind.INVERSE_BUY)
    ]
    assert close_kinds == [
        (JournalKind.MARGIN_TRANSFER, LedgerAccount.CASH),
        (JournalKind.MARGIN_TRANSFER, LedgerAccount.MARGIN),
        (JournalKind.INVERSE_BUY, LedgerAccount.CASH),
    ]
    _assert_reconciles(result, config.initial_cash)


def test_every_session_reconciles_with_an_overlay(tmp_path: Path) -> None:
    """One long run: every NavRecord equals the balances rebuilt from the journal up to that session."""
    sessions = _span(date(2025, 12, 26), date(2026, 3, 20))
    config = _config(initial_cash=20_000_000, commission="0.00015", impact_k=0.0)
    rows = [
        _flat_row(day, "KRX:A", 1_000 + 5 * idx, share_factor=1.5 if idx == 5 else 1.0)
        for idx, day in enumerate(sessions)
    ]
    panel = _write_panel(tmp_path / "gold", "ov_reconcile", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    dividends = _div_frame([_div_row("KRX:A", sessions[2], sessions[4], dps_krw=100)])
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=dividends)

    result = run_backtest(
        arrays=arrays,
        events=events,
        targets={0: {0: 0.9}, 3: {0: 0.5}},
        config=config,
        deposits={sessions[0]: 5_000_000},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
        overlay=_Policy(lambda _state: OverlayTarget(contracts=2, inverse_value_krw=30_000)),
        overlay_market=_market(
            sessions,
            [1200.0 - 5.0 * idx for idx in range(len(sessions))],
            inverse=[12_000.0 + 40.0 * idx for idx in range(len(sessions))],
        ),
        derivatives=_derivatives(futures_cost_rate=0.0005, inverse_cost_rate=0.001, futures_annual_deduction_krw=0),
    )

    kinds = {entry.kind.value for entry in result.journal}
    assert {
        "deposit", "buy", "commission", "sell", "sell_tax", "dividend", "dividend_tax",
        "cash_in_lieu", "margin_transfer", "variation_margin", "futures_trade",
        "futures_commission", "futures_roll", "futures_tax", "inverse_buy", "inverse_sell",
        "inverse_commission", "inverse_tax",
    } <= kinds

    cursor = 0
    cash = config.initial_cash
    margin = 0
    payable = 0
    positions: dict[int, int] = {}
    for record in result.nav:
        while cursor < len(result.journal) and result.journal[cursor].session_idx <= record.session_idx:
            entry = result.journal[cursor]
            cursor += 1
            if entry.account is LedgerAccount.CASH:
                cash += entry.cash_delta
            elif entry.account is LedgerAccount.MARGIN:
                margin += entry.cash_delta
            else:
                payable += entry.cash_delta
            if entry.instrument_idx is not None:
                positions[entry.instrument_idx] = positions.get(entry.instrument_idx, 0) + entry.quantity_delta
        held = {n: q for n, q in positions.items() if q}
        close = arrays.int_fields["close"][record.session_idx]
        assert record.cash == cash
        assert record.margin == margin
        assert record.tax_payable == -payable
        assert record.market_value == sum(q * int(close[n]) for n, q in held.items())
        assert record.nav == (
            record.cash
            + record.dividend_receivable
            + record.market_value
            + record.margin
            + record.inverse_value
            - record.tax_payable
        )
        assert record.cash >= 0
    assert cursor == len(result.journal)


def test_close_phase_release_cannot_fund_the_same_session_buys(tmp_path: Path) -> None:
    """The 3,000,000 released at session t's close is unavailable to session t's open-auction buys."""
    sessions = _sessions(4)
    config = _config(initial_cash=10_000_000, commission="0.00015", impact_k=0.0, cash_buffer=0.1)
    arrays, events = _arrays(tmp_path, "ov_release", sessions, closes=1000)

    result = run_backtest(
        arrays=arrays,
        events=events,
        targets={0: {0: 1.0}, 1: {0: 1.0}, 2: {0: 1.0}},
        config=config,
        deposits={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
        overlay=_hold({0: (2, 0), 1: (0, 0)}),
        overlay_market=_market(sessions, 1000.0),
        derivatives=_derivatives(),
    )

    cash_before_release = config.initial_cash + sum(
        entry.cash_delta
        for entry in result.journal
        if entry.session_idx < 2 and entry.account is LedgerAccount.CASH
    )
    buys = [
        fill
        for fill in result.fills
        if fill.order.decision_session_idx == 1 and fill.order.side.value == "buy"
    ]
    assert result.nav[1].margin == 3_000_000
    assert cash_before_release == 699_055
    assert sum(fill.quantity * fill.price for fill in buys) == 698_000
    assert sum(fill.quantity * fill.price for fill in buys) < cash_before_release
    assert any(reject.reason == "cash" for reject in result.rejects)
    assert result.nav[2].margin == 0
    assert result.nav[2].cash == 3_000_951 > cash_before_release
    _assert_reconciles(result, config.initial_cash)


def test_final_record_reconciles_after_the_year_end_futures_tax(tmp_path: Path) -> None:
    sessions = _span(date(2025, 12, 30), date(2026, 1, 2))
    config = _config(initial_cash=1_600_099)

    result = _run(
        tmp_path,
        name="ov_final",
        sessions=sessions,
        market=_market(sessions, [100.0, 100.0, 100.0, 50.0]),
        derivatives=_derivatives(contract_multiplier_krw=100_000),
        policy=_Policy(lambda _state: OverlayTarget(contracts=1, inverse_value_krw=0)),
        config=config,
    )

    taxes = _kinds(result, JournalKind.FUTURES_TAX)
    assert [(entry.session_idx, entry.cash_delta, entry.account) for entry in taxes] == [
        (3, -275_000, LedgerAccount.MARGIN)
    ]
    assert result.nav[-1].cash == 5_850_099
    assert result.nav[-1].margin == 475_000
    _assert_reconciles(result, config.initial_cash)
    assert math.isnan(result.stock_book_returns[0])
    assert result.stock_book_returns[-1] == 0.0


def test_inverse_purchase_is_capped_by_free_cash(tmp_path: Path) -> None:
    """A hedge worth more than free cash buys only the affordable lots, plus their commission."""
    sessions = _sessions(3)
    config = _config(initial_cash=1_000_000, commission="0", impact_k=0.0)

    result = _run(
        tmp_path,
        name="ov_inverse_cap",
        sessions=sessions,
        market=_market(sessions, 1000.0, inverse=100_000.0),
        derivatives=_derivatives(inverse_cost_rate=0.01),
        policy=_hold({0: (0, 5_000_000)}),
        config=config,
    )

    buys = _kinds(result, JournalKind.INVERSE_BUY)
    assert [entry.quantity_delta for entry in buys] == [9]
    assert buys[0].cash_delta == -900_000
    assert [entry.cash_delta for entry in _kinds(result, JournalKind.INVERSE_COMMISSION)] == [-9_000]
    assert result.nav[1].inverse_value == 900_000
    assert result.nav[1].cash == 91_000
    _assert_reconciles(result, config.initial_cash)


def test_inverse_lots_are_tracked_and_taxed_per_disposal(tmp_path: Path) -> None:
    sessions = _sessions(4)
    config = _config(initial_cash=10_000_000, commission="0", impact_k=0.0)

    result = _run(
        tmp_path,
        name="ov_inverse",
        sessions=sessions,
        market=_market(sessions, 1000.0, inverse=[10_000.0, 10_000.0, 12_000.0, 12_000.0]),
        derivatives=_derivatives(inverse_cost_rate=0.001),
        policy=_hold({0: (1, 20_000), 1: (1, 20_000), 2: (0, 0)}),
        config=config,
    )

    assert [entry.quantity_delta for entry in _kinds(result, JournalKind.INVERSE_BUY)] == [2]
    assert [entry.cash_delta for entry in _kinds(result, JournalKind.INVERSE_BUY)] == [-20_000]
    assert [entry.quantity_delta for entry in _kinds(result, JournalKind.INVERSE_SELL)] == [-1, -1]
    assert [entry.cash_delta for entry in _kinds(result, JournalKind.INVERSE_SELL)] == [12_000, 12_000]
    assert [entry.cash_delta for entry in _kinds(result, JournalKind.INVERSE_COMMISSION)] == [-20, -12, -12]
    assert [entry.cash_delta for entry in _kinds(result, JournalKind.INVERSE_TAX)] == [-308, -308]
    assert [record.inverse_value for record in result.nav] == [0, 20_000, 12_000, 0]
    assert [entry.quantity_delta for entry in _kinds(result, JournalKind.FUTURES_TRADE)] == [1, -1]
    _assert_reconciles(result, config.initial_cash)


def test_missing_inverse_close_fails_closed(tmp_path: Path) -> None:
    sessions = _sessions(3)
    arrays, events = _arrays(tmp_path, "ov_inv_nan", sessions)
    market = _market(sessions, 1000.0, inverse=[10_000.0, 10_000.0, float("nan")])
    derivatives = _derivatives()
    base: dict[str, Any] = {
        "arrays": arrays, "events": events, "targets": {}, "config": _config(),
        "deposits": {}, "rules": _rules(), "start": sessions[0], "end": sessions[-1],
        "overlay_market": market, "derivatives": derivatives,
    }

    with pytest.raises(PITDataError, match="inverse close missing"):
        run_backtest(**base, overlay=_hold({0: (0, 20_000)}))

    with pytest.raises(PITDataError, match="inverse close missing"):
        run_backtest(**base, overlay=_hold({0: (0, 20_000), 1: (0, 20_000)}))


def test_missing_index_level_fails_closed(tmp_path: Path) -> None:
    sessions = _sessions(4)
    arrays, events = _arrays(tmp_path, "ov_level_nan", sessions)
    base: dict[str, Any] = {
        "arrays": arrays, "events": events, "targets": {}, "config": _config(),
        "deposits": {}, "rules": _rules(), "end": sessions[-1],
        "overlay": _flat_policy(), "derivatives": _derivatives(),
    }

    with pytest.raises(PITDataError, match="index level missing"):
        run_backtest(**base, start=sessions[0], overlay_market=_market(sessions, [1000.0, 1000.0, float("nan"), 1000.0]))
    with pytest.raises(PITDataError, match="index level missing"):
        run_backtest(**base, start=sessions[1], overlay_market=_market(sessions, [float("nan"), 1000.0, 1000.0, 1000.0]))


def test_overlay_arguments_are_all_or_none(tmp_path: Path) -> None:
    sessions = _sessions(3)
    arrays, events = _arrays(tmp_path, "ov_all_or_none", sessions)
    market = _market(sessions, 1000.0)
    derivatives = _derivatives()
    policy = _flat_policy()
    base: dict[str, Any] = {
        "arrays": arrays, "events": events, "targets": {}, "config": _config(),
        "deposits": {}, "rules": _rules(), "start": sessions[0], "end": sessions[-1],
    }

    for partial in (
        {"overlay": policy},
        {"overlay": policy, "derivatives": derivatives},
        {"overlay_market": market, "derivatives": derivatives},
        {"overlay_market": market},
    ):
        with pytest.raises(ValueError, match="together"):
            run_backtest(**base, **partial)  # type: ignore[arg-type]
    assert run_backtest(**base, overlay=policy, overlay_market=market, derivatives=derivatives).ledger_hash


def test_overlay_market_must_align_to_sessions(tmp_path: Path) -> None:
    sessions = _sessions(3)
    arrays, events = _arrays(tmp_path, "ov_align", sessions)

    with pytest.raises(ValueError, match="align"):
        run_backtest(
            arrays=arrays, events=events, targets={}, config=_config(), deposits={}, rules=_rules(),
            start=sessions[0], end=sessions[-1],
            overlay=_flat_policy(),
            overlay_market=OverlayMarket(index_level=np.ones(2), inverse_close=np.ones(2) * 10_000.0),
            derivatives=_derivatives(),
        )


def test_stock_orders_are_sized_on_the_funded_stock_book(tmp_path: Path) -> None:
    """The 3-contract reserve is withheld from the stock book before the open-auction orders are sized."""
    sessions = _sessions(3)
    arrays, events = _arrays(tmp_path, "ov_sizing", sessions, closes=1000)
    derivatives = DerivativeConfig(
        contract_multiplier_krw=10_000, initial_margin_rate=0.25, margin_buffer_rate=0.0675,
        margin_topup_trigger_fraction=1.0, futures_cost_rate=0.0, inverse_cost_rate=0.0,
        futures_tax_rate=Decimal("0.11"), futures_annual_deduction_krw=2_500_000,
        inverse_tax_rate=Decimal("0.154"),
    )
    reserve = required_reserve_krw(contracts=3, level=1500.0, config=derivatives)
    assert reserve == 14_287_500

    result = run_backtest(
        arrays=arrays, events=events, targets={0: {0: 1.0}},
        config=_config(initial_cash=100_000_000, commission="0", impact_k=0.0, cash_buffer=0.1),
        deposits={}, rules=_rules(), start=sessions[0], end=sessions[-1],
        overlay=_hold({0: (3, 0)}), overlay_market=_market(sessions, 1500.0), derivatives=derivatives,
    )

    notional = sum(fill.quantity * fill.price for fill in result.fills)
    assert notional == math.floor((100_000_000 - reserve) * 0.9 / 1000) * 1000
    assert notional <= (100_000_000 - reserve) * (1 - 0.1)
    assert result.nav[0].margin == 0
    assert result.nav[1].margin == reserve
    _assert_reconciles(result, 100_000_000)


def test_overlay_financing_is_excluded_from_stock_book_returns(tmp_path: Path) -> None:
    """Margin transfers and inverse purchases are financing: the stock-book return stays exactly zero."""
    sessions = _sessions(4)
    config = _config(initial_cash=10_000_000, commission="0", impact_k=0.0)

    result = _run(
        tmp_path,
        name="ov_sb_returns",
        sessions=sessions,
        market=_market(sessions, 1000.0),
        derivatives=_derivatives(),
        policy=_hold({0: (1, 20_000)}),
        config=config,
    )

    assert result.stock_book_returns.shape == (len(sessions),)
    assert np.isnan(result.stock_book_returns[0])
    assert result.stock_book_returns[1:].tolist() == [0.0, 0.0, 0.0]


def test_stock_book_return_keeps_trading_and_drops_deposits(tmp_path: Path) -> None:
    """The deposit that funds a purchase is not a return; the later price gain is."""
    sessions = _sessions(3)
    arrays, events = _arrays(tmp_path, "ov_sb_flow", sessions, closes=[100, 100, 110])
    config = _config(initial_cash=10_000_000, commission="0", impact_k=0.0)

    result = run_backtest(
        arrays=arrays, events=events, targets={0: {0: 1.0}}, config=config,
        deposits={sessions[1]: 5_000_000}, rules=_rules(), start=sessions[0], end=sessions[-1],
    )

    assert result.nav[1].market_value == 10_000_000
    assert math.isnan(result.stock_book_returns[0])
    assert result.stock_book_returns[1] == pytest.approx(0.0)
    assert result.stock_book_returns[2] == pytest.approx(16_000_000 / 15_000_000 - 1)


def test_a_flat_overlay_is_a_no_op(tmp_path: Path) -> None:
    sessions = _span(date(2020, 1, 6), date(2020, 1, 20))
    arrays, events = _arrays(
        tmp_path, "ov_noop", sessions, closes=[100 + idx for idx in range(len(sessions))]
    )
    base: dict[str, Any] = {
        "arrays": arrays, "events": events, "targets": {0: {0: 0.5}, 3: {0: 1.0}},
        "config": _config(initial_cash=10_000_000, commission="0", impact_k=0.0),
        "deposits": {sessions[2]: 1_000_000}, "rules": _rules(),
        "start": sessions[0], "end": sessions[-1],
    }

    plain = run_backtest(**base)
    overlay = run_backtest(
        **base,
        overlay=_Policy(lambda _state: OverlayTarget(contracts=0, inverse_value_krw=0)),
        overlay_market=_market(sessions, 1000.0),
        derivatives=_derivatives(),
    )

    assert overlay.ledger_hash == plain.ledger_hash
    assert overlay.journal == plain.journal
    assert overlay.nav == plain.nav
    assert np.array_equal(overlay.stock_book_returns, plain.stock_book_returns, equal_nan=True)
    assert all(record.margin == 0 and record.inverse_value == 0 for record in plain.nav)


def test_overlay_state_is_causal(tmp_path: Path) -> None:
    sessions = _sessions(6)
    config = _config(initial_cash=10_000_000, commission="0", impact_k=0.0)

    policy = _flat_policy()
    baseline = _run(
        tmp_path, name="ov_causal_a", sessions=sessions, market=_market(sessions, 1000.0),
        derivatives=_derivatives(), policy=policy, config=config, start=sessions[1],
    )
    assert [state.session_idx for state in policy.states] == [0, 1, 2, 3, 4, 5]
    assert baseline.nav[0].session_idx == 1
    for state in policy.states:
        row = state.session_idx
        assert state.stock_book_returns.shape == (row,)
        assert state.index_returns.shape == (row,)
        if row == 0:
            assert state.nav == config.initial_cash
            continue
        assert np.isnan(state.stock_book_returns[0])
        assert np.isnan(state.index_returns[0])
        assert np.all(np.isfinite(state.stock_book_returns[1:]))
        assert np.all(np.isfinite(state.index_returns[1:]))

    corrupted_policy = _flat_policy()
    corrupted = _run(
        tmp_path, name="ov_causal_b", sessions=sessions,
        market=_market(
            sessions, [1000.0, 1000.0, 1000.0, 777.0, 555.0, 333.0],
            inverse=[10_000.0, 10_000.0, 10_000.0, 13_000.0, 8_000.0, 6_000.0],
        ),
        derivatives=_derivatives(), policy=corrupted_policy, config=config,
        start=sessions[1], closes=[100, 100, 100, 250, 40, 40],
    )

    assert baseline.nav[:2] == corrupted.nav[:2]
    for clean, dirty in zip(policy.states[:3], corrupted_policy.states[:3], strict=True):
        assert clean.session_idx == dirty.session_idx
        assert (clean.nav, clean.index_level, clean.contracts) == (
            dirty.nav, dirty.index_level, dirty.contracts
        )
        assert np.array_equal(clean.stock_book_returns, dirty.stock_book_returns, equal_nan=True)
        assert np.array_equal(clean.index_returns, dirty.index_returns, equal_nan=True)


def test_decision_before_the_window_executes_in_the_first_session(tmp_path: Path) -> None:
    sessions = _sessions(4)
    config = _config(initial_cash=10_000_000, commission="0", impact_k=0.0)
    arrays, events = _arrays(tmp_path, "ov_prior_row", sessions)
    policy = _hold({0: (1, 0)})

    result = run_backtest(
        arrays=arrays, events=events, targets={}, config=config, deposits={}, rules=_rules(),
        start=sessions[1], end=sessions[-1],
        overlay=policy, overlay_market=_market(sessions, 1000.0), derivatives=_derivatives(),
    )

    assert policy.states[0].session_idx == 0
    assert policy.states[0].stock_book_returns.shape == (0,)
    assert _kinds(result, JournalKind.FUTURES_TRADE)[0].session_idx == 1
    assert _kinds(result, JournalKind.MARGIN_TRANSFER)[0].session_idx == 1


def test_policy_must_return_a_target_or_none(tmp_path: Path) -> None:
    sessions = _sessions(3)
    arrays, events = _arrays(tmp_path, "ov_bad_policy", sessions)

    class _Bad:
        def target(self, state: OverlayState) -> OverlayTarget | None:
            return "three contracts"  # type: ignore[return-value]

    with pytest.raises(ValueError, match="OverlayTarget"):
        run_backtest(
            arrays=arrays, events=events, targets={}, config=_config(), deposits={}, rules=_rules(),
            start=sessions[0], end=sessions[-1],
            overlay=_Bad(), overlay_market=_market(sessions, 1000.0), derivatives=_derivatives(),
        )


def test_cash_sweep_and_year_end_tax_run_with_the_overlay(tmp_path: Path) -> None:
    sessions = _span(date(2024, 12, 30), date(2025, 1, 3))
    year_end_row = sessions.index(date(2024, 12, 31))
    config = _config(initial_cash=10_000_000, commission="0", impact_k=0.0)

    result = _run(
        tmp_path, name="ov_sweep", sessions=sessions, market=_market(sessions, 1000.0),
        derivatives=_derivatives(), policy=_flat_policy(), config=config,
        cash_returns=np.full(len(sessions), 0.0001),
    )

    taxes = _kinds(result, JournalKind.CASH_YIELD_TAX)
    assert [entry.session_idx for entry in taxes] == [year_end_row, len(sessions) - 1]
    assert taxes[0].cash_delta == -math.floor(2_000 * 0.154)
    assert result.nav[year_end_row].cash == 10_002_000 - math.floor(2_000 * 0.154)
    _assert_reconciles(result, config.initial_cash)


def test_config_and_target_validation() -> None:
    config = _derivatives()
    assert required_reserve_krw(contracts=0, level=1000.0, config=config) == 0
    assert required_reserve_krw(contracts=2, level=1500.0, config=config) == math.ceil(
        0.15 * 2 * 10_000 * 1500.0
    )
    for kwargs in (
        {"contract_multiplier_krw": 0},
        {"contract_multiplier_krw": True},
        {"initial_margin_rate": 0.0},
        {"initial_margin_rate": 1.0},
        {"margin_buffer_rate": -0.01},
        {"margin_topup_trigger_fraction": 0.0},
        {"margin_topup_trigger_fraction": 1.5},
        {"futures_cost_rate": 1.0},
        {"inverse_cost_rate": -0.1},
        {"futures_tax_rate": 0.11},
        {"futures_tax_rate": Decimal("1.0")},
        {"inverse_tax_rate": Decimal("-0.1")},
        {"futures_annual_deduction_krw": -1},
    ):
        with pytest.raises(ValueError, match="must"):
            _derivatives(**kwargs)
    assert _derivatives(margin_topup_trigger_fraction=1.0).margin_topup_trigger_fraction == 1.0

    with pytest.raises(ValueError, match="must"):
        OverlayTarget(contracts=-1, inverse_value_krw=0)
    with pytest.raises(ValueError, match="must"):
        OverlayTarget(contracts=True, inverse_value_krw=0)
    with pytest.raises(ValueError, match="must"):
        OverlayTarget(contracts=1, inverse_value_krw=-1)

    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="contracts must be"):
            required_reserve_krw(contracts=bad, level=1000.0, config=config)  # type: ignore[arg-type]
    for bad in (float("nan"), -1.0, 0.0):
        with pytest.raises(PITDataError, match="level must be finite"):
            required_reserve_krw(contracts=1, level=bad, config=config)
    with pytest.raises(PITDataError, match="level must be finite"):
        required_reserve_krw(contracts=1, level=None, config=config)  # type: ignore[arg-type]

def _band_derivatives(**over: Any) -> DerivativeConfig:
    """Spec maintenance-scenario terms: 10,000 multiplier, 10% + 5% reserve, 0.75 trigger."""
    return _derivatives(margin_topup_trigger_fraction=0.75, **over)


def test_band_holds_the_hedge_without_cash(tmp_path: Path) -> None:
    """1 contract, cash 0, +1%: margin 1.40M sits in [M 757,500, R 1,515,000] — no trade, no transfer."""
    sessions = _sessions(3)
    config = _config(initial_cash=1_500_000, commission="0", impact_k=0.0)
    derivatives = _band_derivatives()

    result = _run(
        tmp_path,
        name="ov_band_hold",
        sessions=sessions,
        market=_market(sessions, [1000.0, 1000.0, 1010.0]),
        derivatives=derivatives,
        policy=_hold({0: (1, 0)}),
        config=config,
    )

    assert result.nav[1].margin == 1_500_000
    assert result.nav[2].margin == 1_400_000
    assert result.nav[2].cash == 0
    assert [(e.session_idx, e.quantity_delta) for e in _kinds(result, JournalKind.FUTURES_TRADE)] == [(1, 1)]
    assert not [e for e in result.journal if e.session_idx == 2 and e.kind is JournalKind.MARGIN_TRANSFER]
    _assert_reconciles(result, config.initial_cash)


def test_band_never_churns(tmp_path: Path) -> None:
    """20 sessions within ±3% of entry with an unchanged target: no trade and no cash-to-margin top-up."""
    sessions = _sessions(20)
    levels = [1000.0, 1000.0] + [990.0 if idx % 2 == 0 else 1010.0 for idx in range(18)]
    config = _config(initial_cash=10_000_000, commission="0", impact_k=0.0)

    result = _run(
        tmp_path,
        name="ov_band_calm",
        sessions=sessions,
        market=_market(sessions, levels),
        derivatives=_band_derivatives(),
        policy=_hold({0: (1, 0)}),
        config=config,
    )

    assert [(e.session_idx, e.quantity_delta) for e in _kinds(result, JournalKind.FUTURES_TRADE)] == [(1, 1)]
    assert not [
        e for e in _kinds(result, JournalKind.MARGIN_TRANSFER)
        if e.session_idx > 1 and e.account is LedgerAccount.CASH and e.cash_delta < 0
    ]
    _assert_reconciles(result, config.initial_cash)


def test_maintenance_topup_restores_the_full_reserve(tmp_path: Path) -> None:
    """+8% costs 800,000 of margin (700,000 < M 810,000): the call restores R 1,620,000 from cash."""
    sessions = _sessions(4)
    config = _config(initial_cash=3_500_000, commission="0", impact_k=0.0)
    derivatives = _band_derivatives()

    result = _run(
        tmp_path,
        name="ov_band_topup",
        sessions=sessions,
        market=_market(sessions, [1000.0, 1000.0, 1080.0, 1080.0]),
        derivatives=derivatives,
        policy=_hold({0: (1, 0)}),
        config=config,
    )

    assert (result.nav[1].margin, result.nav[1].cash) == (1_500_000, 2_000_000)
    assert (result.nav[2].margin, result.nav[2].cash) == (1_620_000, 1_080_000)
    assert [(e.session_idx, e.quantity_delta) for e in _kinds(result, JournalKind.FUTURES_TRADE)] == [(1, 1)]
    _assert_reconciles(result, config.initial_cash)


def test_topup_never_spends_a_tax_liability() -> None:
    """The maintenance step sees available = margin + cash - tax_payable: 1.20M < R 1.62M flattens the hedge."""
    derivatives = _band_derivatives()
    ledger = Ledger(initial_cash=3_500_000)
    ledger.transfer_margin(session_idx=0, amount=1_500_000)
    ledger.trade_futures(session_idx=0, contracts=1, level=1000.0, multiplier=10_000, cost_rate=0.0)
    ledger.settle_variation(session_idx=1, prev_level=1000.0, level=1080.0, multiplier=10_000)
    assert ledger.margin == 700_000
    ledger.accrue_cash_yield(session_idx=1, gross_return=1.0)
    ledger.buy(session_idx=1, instrument_idx=0, quantity=4_000_000, price=1, commission=0)
    ledger.settle_cash_yield_tax(
        session_idx=1,
        config=CostConfig(
            commission_rate=Decimal("0"), impact_k=0.0, dividend_withholding_rate=Decimal("0"),
            cash_yield_tax_rate=Decimal("0.75"),
        ),
    )
    assert (ledger.cash, ledger.tax_payable) == (0, 1_500_000)
    ledger.sell(session_idx=1, instrument_idx=0, quantity=2_000_000, price=1, commission=0, sell_tax=0)

    _sync_futures(
        ledger, session_idx=1, target_contracts=1, level=1080.0, config=derivatives, multiplier=10_000,
    )

    assert ledger.contracts == 0
    assert ledger.margin == 0
    assert (ledger.cash, ledger.tax_payable) == (2_700_000, 1_500_000)


def test_forced_partial_reduction_instead_of_borrowing() -> None:
    """3 contracts, +8% (margin 2.10M < M 2.43M): only k=1 is fundable, margin back to 1.62M, cash 469,200."""
    derivatives = _band_derivatives(futures_cost_rate=0.0005)
    ledger = Ledger(initial_cash=4_515_000)
    ledger.transfer_margin(session_idx=0, amount=4_515_000)
    ledger.trade_futures(session_idx=0, contracts=3, level=1000.0, multiplier=10_000, cost_rate=0.0005)
    assert ledger.margin == 4_500_000
    ledger.settle_variation(session_idx=1, prev_level=1000.0, level=1080.0, multiplier=10_000)
    assert ledger.margin == 2_100_000

    _sync_futures(
        ledger, session_idx=1, target_contracts=3, level=1080.0, config=derivatives, multiplier=10_000,
    )

    assert ledger.contracts == 1
    assert ledger.margin == 1_620_000
    assert ledger.cash == 469_200


def test_rebalance_syncs_to_the_reserve(tmp_path: Path) -> None:
    """1 contract inside the band with a pending target of 2 and 2M cash: contracts 2, margin R 3,030,000."""
    sessions = _sessions(4)
    config = _config(initial_cash=3_500_000, commission="0", impact_k=0.0)

    result = _run(
        tmp_path,
        name="ov_band_rebalance",
        sessions=sessions,
        market=_market(sessions, [1000.0, 1000.0, 1010.0, 1010.0]),
        derivatives=_band_derivatives(),
        policy=_hold({0: (1, 0), 1: (2, 0)}),
        config=config,
    )

    assert [(e.session_idx, e.quantity_delta) for e in _kinds(result, JournalKind.FUTURES_TRADE)] == [
        (1, 1),
        (2, 1),
    ]
    assert (result.nav[2].margin, result.nav[2].cash) == (3_030_000, 370_000)
    _assert_reconciles(result, config.initial_cash)
