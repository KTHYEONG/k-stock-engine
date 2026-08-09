"""PLAN-05-ETF-PARITY-REGRESSION: new etfs matches legacy signal/fill/metrics.

The new ``etfs`` subsystem is a lift-and-shift of the legacy ETF logic. This
regression proves fixture parity for signals, fills, and metric inputs so the
legacy engine can later be retired.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.legacy.etf_v1.backtester import ETFBacktester
from src.legacy.etf_v1.strategy_engine import ETFStrategyEngine
from src.etfs.backtesting.engine import EtfBacktester, IndexSwitchParams
from src.etfs.domain.universe import KOSPI_ETF_UNIVERSE
from src.etfs.strategies.index_switch_v1 import IndexSwitchV1
from tests.fixtures.etfs.helpers import make_etf_fixture, preprocess_index

LEGACY_PARAMS = {
    "MACRO_EMA_PERIOD": 50,
    "FAST_EMA_PERIOD": 10,
    "ROC_N": 1,
    "ROC_LOWER": -0.01,
    "IBS_ENTRY": 0.20,
    "IBS_EXIT": 0.75,
    "MAX_HOLD_DAYS": 5,
    "STOP_LOSS_PCT": 0.08,
}

NEW_PARAMS = IndexSwitchParams(
    macro_ema_period=50,
    fast_ema_period=10,
    roc_n=1,
    roc_lower=-0.01,
    ibs_entry=0.20,
    ibs_exit=0.75,
    max_hold_days=5,
    stop_loss_pct=0.08,
)


def test_plan_05_etf_signal_parity_with_legacy_engine() -> None:
    index_df, _ = make_etf_fixture()
    preprocessed = preprocess_index(index_df)
    legacy = ETFStrategyEngine(LEGACY_PARAMS).generate_signal(preprocessed)
    new = IndexSwitchV1(NEW_PARAMS).generate_signal(preprocessed)

    assert new["signal_trigger"].to_list() == legacy["signal_trigger"].to_list()
    assert new["ibs"].to_list() == legacy["ibs"].to_list()


def test_plan_05_etf_fill_and_metric_parity() -> None:
    index_df, etf_df = make_etf_fixture(n_days=120)
    legacy_bt = ETFBacktester(index_df, etf_df)
    legacy_results = legacy_bt.run(LEGACY_PARAMS, target_market="KOSPI")

    new_bt = EtfBacktester(index_df, etf_df, params=NEW_PARAMS)
    new_results = new_bt.run(KOSPI_ETF_UNIVERSE, target_market="KOSPI")

    assert len(legacy_results) == 1
    assert len(new_results) == 1
    legacy = legacy_results[0]
    new = new_results[0]

    assert new.total_trades == legacy["total_trades"]
    assert new.win_rate == pytest_approx(legacy["win_rate"])
    assert new.profit_factor == pytest_approx(legacy["profit_factor"])
    assert new.total_return_pct == pytest_approx(legacy["total_return_pct"])
    assert new.final_balance == pytest_approx(legacy["final_balance"])
    assert np.allclose(new.equity_curve, legacy["equity_curve"])

    legacy_trades = legacy["trades_df"]
    assert len(new.trades) == len(legacy_trades)
    for new_trade, legacy_trade in zip(new.trades, legacy_trades.to_dict(orient="records"), strict=True):
        assert int(new_trade["entry_idx"]) == int(legacy_trade["entry_idx"])
        assert int(new_trade["exit_idx"]) == int(legacy_trade["exit_idx"])
        assert pytest_approx(new_trade["entry_price"]) == legacy_trade["entry_price"]
        assert pytest_approx(new_trade["exit_price"]) == legacy_trade["exit_price"]
        assert pytest_approx(new_trade["pnl"]) == legacy_trade["pnl"]


def test_plan_05_etf_fixture_is_deterministic() -> None:
    idx_a, etf_a = make_etf_fixture()
    idx_b, etf_b = make_etf_fixture()
    assert idx_a.equals(idx_b)
    assert etf_a.equals(etf_b)


def test_plan_05_etf_requires_explicit_market_and_universe() -> None:
    assert KOSPI_ETF_UNIVERSE.bull_1x == "069500"
    assert KOSPI_ETF_UNIVERSE.bear_1x == "114800"


def pytest_approx(value: float):
    return pytest.approx(value, rel=1e-9)
