"""Performance summary: TWR neutrality, log growth, MWR, and cost attribution."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.backtest.costs import Side
from src.backtest.engine import BacktestResult
from src.backtest.execution import Fill, Order, Reject
from src.backtest.ledger import JournalEntry, JournalKind, NavRecord
from src.backtest.metrics import summarize
from tests.unit.backtest.test_market import _mrow, _write_panel

from src.backtest.market import load_market_arrays


def _panel(tmp_path: Path, name: str, n: int) -> Any:
    sessions = [date(2020, 1, 6) + timedelta(days=i) for i in range(n)]
    panel = _write_panel(
        tmp_path / "gold", name, [_mrow(day, "KRX:A", close=100) for day in sessions]
    )
    return load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")


def _record(session_idx: int, nav: int, ext: int) -> NavRecord:
    return NavRecord(
        session_idx=session_idx,
        cash=nav,
        dividend_receivable=0,
        market_value=0,
        nav=nav,
        external_flow=ext,
    )


def _result(records: list[NavRecord], **overrides: Any) -> BacktestResult:
    params: dict[str, Any] = {
        "nav": tuple(records),
        "fills": (),
        "rejects": (),
        "journal": (),
        "dividends_integrated": True,
        "ledger_hash": "test",
    }
    params.update(overrides)
    return BacktestResult(**params)


def test_deposit_alone_is_zero_return(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, "market_panel_dep", 3)
    result = _result([_record(0, 0, 0), _record(1, 100, 100), _record(2, 100, 100)])
    summary = summarize(result=result, arrays=arrays, sessions_per_year=252)
    assert summary.twr_total == 0
    assert summary.final_nav == 100
    assert summary.total_external_flow == 100


def test_log_growth_matches_closed_form(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, "market_panel_growth", 253)
    records = [
        _record(t, round(1_000_000_000 * 2.0 ** (t / 252)), 0) for t in range(253)
    ]
    summary = summarize(result=_result(records), arrays=arrays, sessions_per_year=252)
    assert summary.log_growth_annualized == pytest.approx(math.log(2), abs=1e-6)
    assert summary.log_growth_by_year
    assert summary.mwr_annualized is None
    assert summary.price_return_only is False


def test_price_return_flag_propagates(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, "market_panel_flag", 1)
    result = _result([_record(0, 50, 50)], dividends_integrated=False)
    assert summarize(result=result, arrays=arrays, sessions_per_year=252).price_return_only is True


def test_mwr_solves_big_gain(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, "market_panel_mwr", 2)
    result = _result([_record(0, 1_000, 1_000), _record(1, 1_100, 1_000)])
    summary = summarize(result=result, arrays=arrays, sessions_per_year=252)
    assert summary.mwr_annualized is not None
    assert summary.mwr_annualized > 1.0


def test_turnover_cost_rejects_participation(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, "market_panel_attr", 3)
    buy = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=0)
    sell = Order(instrument_idx=0, side=Side.SELL, quantity=10, decision_session_idx=1)
    journal = (
        JournalEntry(1, JournalKind.BUY, 0, -1000, 10),
        JournalEntry(1, JournalKind.COMMISSION, 0, -10, 0),
        JournalEntry(2, JournalKind.SELL, 0, 1100, -10),
        JournalEntry(2, JournalKind.SELL_TAX, 0, -5, 0),
        JournalEntry(2, JournalKind.COMMISSION, 0, -10, 0),
    )
    result = _result(
        [_record(0, 10_000, 0), _record(1, 10_000, 0), _record(2, 10_100, 0)],
        fills=(Fill(buy, 10, 100), Fill(sell, 10, 110)),
        rejects=(Reject(sell, "capacity"),),
        journal=journal,
    )
    summary = summarize(result=result, arrays=arrays, sessions_per_year=3)
    assert summary.turnover_annualized == pytest.approx((0.1 + 0.11) / 2 * 3)
    assert summary.cost_drag_annualized == pytest.approx((0.001 + 0.0015) / 2 * 3)
    assert summary.reject_counts == {"capacity": 1}
    assert summary.max_participation == pytest.approx(1100 / 1_000_000_000.0)


def test_sessions_per_year_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, "market_panel_spy", 1)
    result = _result([_record(0, 10, 0)])
    with pytest.raises(ValueError):
        summarize(result=result, arrays=arrays, sessions_per_year=0)
