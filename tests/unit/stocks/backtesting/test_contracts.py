"""Direct contract tests for the decomposed backtesting module."""
from __future__ import annotations

import src.stocks.backtesting.engine as engine_module
from src.stocks.backtesting.contracts import (
    ArtifactSchedule,
    BacktestAttribution,
    BacktestResult,
    BacktestValidationError,
)
from src.stocks.backtesting.execution import PreparedExecutionState
from src.stocks.backtesting.market import PreparedReplayMarket


def test_backtest_attribution_cost_components_are_immutable() -> None:
    attribution = BacktestAttribution(base_commission=1.0, base_total=1.0)

    assert attribution.base_commission == 1.0
    assert attribution.base_total == 1.0


def test_backtest_result_has_empty_safe_defaults() -> None:
    """The canonical result contract carries safe empty collections."""
    result = BacktestResult(
        ledger=(), trades=(), final_value=0.0, total_return=0.0, metrics={}
    )

    assert result.ledger == ()
    assert result.trades == ()
    assert result.metrics == {}
    assert result.stress_ledger == ()


def test_engine_reexports_canonical_contracts() -> None:
    """engine.py is a compatibility facade over the canonical contracts."""
    from src.stocks.backtesting.contracts import (
        BacktestRequest,
        BacktestTrade,
        BacktestLedgerRow,
    )

    assert engine_module.BacktestResult is BacktestResult
    assert engine_module.BacktestRequest is BacktestRequest
    assert engine_module.BacktestTrade is BacktestTrade
    assert engine_module.BacktestLedgerRow is BacktestLedgerRow
    assert engine_module.PreparedReplayMarket is PreparedReplayMarket


def test_empty_artifact_schedule_is_invalid() -> None:
    import pytest

    with pytest.raises(BacktestValidationError):
        ArtifactSchedule(slots=())


def test_prepared_execution_state_starts_from_initial_portfolio() -> None:
    from datetime import UTC, datetime
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position

    instrument = Instrument("KRX:00001", AssetKind.STOCK, "KRX", "00001", "KRW")
    snapshot = PortfolioSnapshot(
        account_snapshot_id="acc",
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=1_000.0,
        unsettled_cash=50.0,
        positions=(Position(instrument=instrument, quantity=3, average_cost=10.0),),
    )
    state = PreparedExecutionState.from_initial(snapshot)
    assert state.settled_cash == 1_000.0
    assert state.unsettled_cash == 50.0
    assert state.positions == {"KRX:00001": 3}
    assert not state.settlements
    assert not state.pending_orders
    state.reset_for_segment()
    assert not state.trades
    assert not state.ledger
    assert not state.last_close
