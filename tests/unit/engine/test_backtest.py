"""Backtest tests."""
from __future__ import annotations


def test_backtester_separates_close_decision_from_next_open_fill() -> None:
    from datetime import UTC, datetime, timedelta

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    day_one = datetime(2024, 1, 2, tzinfo=UTC)
    day_two = day_one + timedelta(days=1)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.1, hard_participation_cap=0.2)

    class BuyOnce:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time.date() != day_one.date():
                return ()
            return (TradeIntent("intent-1", AssetKind.STOCK, instrument.instrument_id, 100.0, context.decision_time, day_two, "champion-v1", "fixture", "key-1", context.portfolio.account_snapshot_id),)

    sessions = (BacktestSession(day_one, day_one + timedelta(hours=6, minutes=30), (HistoricalBar(day_one, instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object()), BacktestSession(day_two, day_two + timedelta(hours=6, minutes=30), (HistoricalBar(day_two, instrument.instrument_id, 11.0, 11.0, 1_000_000.0, 0.02),), (), object()))
    calendar = SessionCalendar((day_one, day_two, day_two + timedelta(days=1), day_two + timedelta(days=2)))
    result = EventBacktester(BacktestConfig("bt-1", 1_000.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, calendar, model)).run(sessions, BuyOnce())

    assert len(result.fills) == 1
    assert result.fills[0].trade_time == day_two


def test_backtester_requires_explicit_execution_dependencies() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.engine.backtest import BacktestConfig, BacktestIntegrityError, BacktestSession, EventBacktester
    from src.engine.fill_model import ExecutionScenario

    opened = datetime(2024, 1, 2, tzinfo=UTC)

    class NoTrade:
        def decide(self, context: object) -> tuple[object, ...]:
            del context
            return ()

    session = BacktestSession(opened, opened + timedelta(hours=6, minutes=30), (), (), object())
    config = BacktestConfig("missing-dependencies", 100.0, {}, ExecutionScenario.IDEAL)

    with pytest.raises(BacktestIntegrityError, match="cost_schedule"):
        EventBacktester(config).run((session,), NoTrade())


def test_backtester_rejects_fill_model_scenario_mismatch() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestIntegrityError, BacktestSession, EventBacktester
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel

    opened = datetime(2024, 1, 2, tzinfo=UTC)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 0),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    base_model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.BASE, target_participation_cap=0.01, hard_participation_cap=0.02)
    session = BacktestSession(opened, opened + timedelta(hours=6), (HistoricalBar(opened, instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object())
    config = BacktestConfig("scenario-mismatch", 100.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, SessionCalendar((opened,)), base_model)

    with pytest.raises(BacktestIntegrityError, match="scenario"):
        EventBacktester(config).run((session,), lambda context: ())


def test_backtester_rejects_duplicate_or_misaligned_session_bars() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestIntegrityError, BacktestSession, EventBacktester
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel

    opened = datetime(2024, 1, 2, tzinfo=UTC)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.01, hard_participation_cap=0.02)

    class NoTrade:
        def decide(self, context: object) -> tuple[object, ...]:
            del context
            return ()

    duplicate = HistoricalBar(opened, instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02)
    session = BacktestSession(opened, opened + timedelta(hours=6), (duplicate, duplicate), (), object())
    config = BacktestConfig("duplicate-bars", 100.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, SessionCalendar((opened,)), model)

    with pytest.raises(BacktestIntegrityError, match="duplicate bar"):
        EventBacktester(config).run((session,), NoTrade())


def test_backtester_records_partial_remainder_as_reject() -> None:
    from datetime import UTC, datetime, timedelta

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    first = datetime(2024, 1, 2, tzinfo=UTC)
    second = first + timedelta(days=1)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 0),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.01, hard_participation_cap=0.02)

    class BuyOnce:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time != first + timedelta(hours=6):
                return ()
            return (TradeIntent("buy", AssetKind.STOCK, instrument.instrument_id, 150.0, context.decision_time, second, "champion-v1", "fixture", "buy-key", context.portfolio.account_snapshot_id, target_quantity=150),)

    sessions = (
        BacktestSession(first, first + timedelta(hours=6), (HistoricalBar(first, instrument.instrument_id, 1.0, 1.0, 10_000.0, 0.02),), (), object()),
        BacktestSession(second, second + timedelta(hours=6), (HistoricalBar(second, instrument.instrument_id, 1.0, 1.0, 10_000.0, 0.02),), (), object()),
    )
    config = BacktestConfig("partial", 1_000.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, SessionCalendar((first, second)), model)

    result = EventBacktester(config).run(sessions, BuyOnce())

    assert result.fills[0].quantity == 100
    assert result.rejects[0].rejected_quantity == 50
    assert result.capacity_diagnostics[0].requested_quantity == 150
    assert result.capacity_diagnostics[0].filled_quantity == 100


def test_backtester_rejects_intent_not_created_at_current_decision() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestIntegrityError, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    first = datetime(2024, 1, 2, tzinfo=UTC)
    second = first + timedelta(days=1)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 0),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.01, hard_participation_cap=0.02)

    class StaleIntent:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            return (TradeIntent("stale", AssetKind.STOCK, instrument.instrument_id, 10.0, context.decision_time - timedelta(days=1), second, "champion-v1", "fixture", "stale-key", context.portfolio.account_snapshot_id),)

    sessions = (
        BacktestSession(first, first + timedelta(hours=6), (HistoricalBar(first, instrument.instrument_id, 1.0, 1.0, 10_000.0, 0.02),), (), object()),
        BacktestSession(second, second + timedelta(hours=6), (HistoricalBar(second, instrument.instrument_id, 1.0, 1.0, 10_000.0, 0.02),), (), object()),
    )
    config = BacktestConfig("stale", 100.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, SessionCalendar((first, second)), model)

    with pytest.raises(BacktestIntegrityError, match="decision_time"):
        EventBacktester(config).run(sessions, StaleIntent())


def test_backtester_cancels_order_when_execution_bar_is_missing() -> None:
    from datetime import UTC, datetime, timedelta

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    # Given: a target decided at T whose instrument is halted (no bar) at T+1.
    day_one = datetime(2024, 1, 2, tzinfo=UTC)
    day_two = day_one + timedelta(days=1)
    halted = Instrument("KRX:053950", AssetKind.STOCK, "KRX", "053950", "KRW")
    other = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.1, hard_participation_cap=0.2)

    class BuyHalted:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time.date() != day_one.date():
                return ()
            return (TradeIntent("intent-1", AssetKind.STOCK, halted.instrument_id, 100.0, context.decision_time, day_two, "compounding-v2", "fixture", "key-1", context.portfolio.account_snapshot_id),)

    sessions = (
        BacktestSession(day_one, day_one + timedelta(hours=6, minutes=30), (HistoricalBar(day_one, halted.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02), HistoricalBar(day_one, other.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02)), (), object()),
        BacktestSession(day_two, day_two + timedelta(hours=6, minutes=30), (HistoricalBar(day_two, other.instrument_id, 11.0, 11.0, 1_000_000.0, 0.02),), (), object()),
    )
    calendar = SessionCalendar((day_one, day_two, day_two + timedelta(days=1), day_two + timedelta(days=2)))
    instruments = {halted.instrument_id: halted, other.instrument_id: other}

    # When
    result = EventBacktester(BacktestConfig("bt-missing-bar", 1_000.0, instruments, ExecutionScenario.IDEAL, costs, calendar, model)).run(sessions, BuyHalted())

    # Then: the run completes, the order is cancelled with a receipt, nothing fills.
    assert result.fills == ()
    assert [reject.reason for reject in result.rejects] == ["missing execution bar"]
    assert result.rejects[0].rejected_quantity == 10
    assert result.rejects[0].event_time == day_two


def test_backtester_marks_halted_position_at_last_known_close() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    # Given: a position bought at T+1 whose instrument stops printing bars at T+2.
    days = tuple(datetime(2024, 1, 2, tzinfo=UTC) + timedelta(days=index) for index in range(3))
    instrument = Instrument("KRX:053950", AssetKind.STOCK, "KRX", "053950", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.1, hard_participation_cap=0.2)

    class BuyOnce:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time.date() != days[0].date():
                return ()
            return (TradeIntent("intent-1", AssetKind.STOCK, instrument.instrument_id, 100.0, context.decision_time, days[1], "compounding-v2", "fixture", "key-1", context.portfolio.account_snapshot_id),)

    sessions = (
        BacktestSession(days[0], days[0] + timedelta(hours=6, minutes=30), (HistoricalBar(days[0], instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object()),
        BacktestSession(days[1], days[1] + timedelta(hours=6, minutes=30), (HistoricalBar(days[1], instrument.instrument_id, 10.0, 12.0, 1_000_000.0, 0.02),), (), object()),
        BacktestSession(days[2], days[2] + timedelta(hours=6, minutes=30), (), (), object()),
    )
    calendar = SessionCalendar((*days, days[2] + timedelta(days=1), days[2] + timedelta(days=2)))

    # When
    result = EventBacktester(BacktestConfig("bt-halt-mark", 1_000.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, calendar, model)).run(sessions, BuyOnce())

    # Then: the halted session marks at the carried-forward close, not an invented price.
    assert len(result.fills) == 1
    assert result.daily_nav[-1].nav == pytest.approx(result.daily_nav[-2].nav, abs=1e-9)


def test_backtester_fails_closed_when_successor_position_never_had_a_close() -> None:
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.ledger import LedgerActionType, LedgerCorporateAction, LedgerSuccessorAllocation
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestIntegrityError, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    # Given: a merger delivers a successor instrument that never printed a bar,
    # so no prior close exists to carry forward.
    days = tuple(datetime(2024, 1, 2, tzinfo=UTC) + timedelta(days=index) for index in range(4))
    parent = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    successor = Instrument("KRX:999999", AssetKind.STOCK, "KRX", "999999", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.1, hard_participation_cap=0.2)
    allocations = (LedgerSuccessorAllocation(successor.instrument_id, Decimal("1"), Decimal("1")),)

    class BuyOnce:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time.date() != days[0].date():
                return ()
            return (TradeIntent("intent-1", AssetKind.STOCK, parent.instrument_id, 100.0, context.decision_time, days[1], "compounding-v2", "fixture", "key-1", context.portfolio.account_snapshot_id),)

    def _bar(day: datetime) -> HistoricalBar:
        return HistoricalBar(day, parent.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02)

    sessions = (
        BacktestSession(days[0], days[0] + timedelta(hours=6, minutes=30), (_bar(days[0]),), (), object()),
        BacktestSession(days[1], days[1] + timedelta(hours=6, minutes=30), (_bar(days[1]),), (), object()),
        BacktestSession(
            days[2],
            days[2] + timedelta(hours=6, minutes=30),
            (_bar(days[2]),),
            (LedgerCorporateAction("act-entitle", parent.instrument_id, LedgerActionType.EXCHANGE_ENTITLEMENT, days[2], 1.0, 0.0, allocations, "evt-1"),),
            object(),
        ),
        BacktestSession(
            days[3],
            days[3] + timedelta(hours=6, minutes=30),
            (_bar(days[3]),),
            (LedgerCorporateAction("act-deliver", parent.instrument_id, LedgerActionType.SUCCESSOR_DELIVERY, days[3], 1.0, 0.0, allocations, "evt-1"),),
            object(),
        ),
    )
    calendar = SessionCalendar((*days, days[3] + timedelta(days=1), days[3] + timedelta(days=2)))
    instruments = {parent.instrument_id: parent, successor.instrument_id: successor}

    # When/Then: carry-forward has nothing to carry, so the run fails closed.
    with pytest.raises(BacktestIntegrityError, match="missing raw close"):
        EventBacktester(BacktestConfig("bt-successor", 1_000.0, instruments, ExecutionScenario.IDEAL, costs, calendar, model)).run(sessions, BuyOnce())


def test_backtester_skips_intent_when_decision_bar_is_missing() -> None:
    from datetime import UTC, datetime, timedelta

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    # Given: a target_value intent for an instrument that is halted on the very
    # decision session, so no reference price exists to size the order.
    day_one = datetime(2024, 1, 2, tzinfo=UTC)
    day_two = day_one + timedelta(days=1)
    halted = Instrument("KRX:053950", AssetKind.STOCK, "KRX", "053950", "KRW")
    traded = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.1, hard_participation_cap=0.2)

    class TargetBoth:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time.date() != day_one.date():
                return ()
            account = context.portfolio.account_snapshot_id
            return (
                TradeIntent("intent-halted", AssetKind.STOCK, halted.instrument_id, 100.0, context.decision_time, day_two, "compounding-v2", "fixture", "key-halted", account),
                TradeIntent("intent-traded", AssetKind.STOCK, traded.instrument_id, 100.0, context.decision_time, day_two, "compounding-v2", "fixture", "key-traded", account),
            )

    sessions = (
        BacktestSession(day_one, day_one + timedelta(hours=6, minutes=30), (HistoricalBar(day_one, traded.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object()),
        BacktestSession(day_two, day_two + timedelta(hours=6, minutes=30), (HistoricalBar(day_two, traded.instrument_id, 11.0, 11.0, 1_000_000.0, 0.02),), (), object()),
    )
    calendar = SessionCalendar((day_one, day_two, day_two + timedelta(days=1), day_two + timedelta(days=2)))
    instruments = {halted.instrument_id: halted, traded.instrument_id: traded}

    # When
    result = EventBacktester(BacktestConfig("bt-decision-bar", 1_000.0, instruments, ExecutionScenario.IDEAL, costs, calendar, model)).run(sessions, TargetBoth())

    # Then: the halted intent is skipped with a receipt while the tradable one fills.
    assert [reject.reason for reject in result.rejects] == ["missing decision bar for sizing"]
    assert len(result.fills) == 1
    assert result.fills[0].instrument_id == traded.instrument_id


def test_backtester_carries_stale_mark_into_decision_snapshot_for_halted_holding() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    # Given: a position bought at T+1 whose instrument stops trading at T+2, and a
    # strategy that values its whole book on every decision.
    days = tuple(datetime(2024, 1, 2, tzinfo=UTC) + timedelta(days=index) for index in range(3))
    instrument = Instrument("KRX:053950", AssetKind.STOCK, "KRX", "053950", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.1, hard_participation_cap=0.2)
    observed_equity: list[float] = []

    class ValueBook:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            observed_equity.append(context.portfolio.equity(context.market_snapshot["mark_prices"]))
            if context.decision_time.date() != days[0].date():
                return ()
            return (TradeIntent("intent-1", AssetKind.STOCK, instrument.instrument_id, 100.0, context.decision_time, days[1], "compounding-v2", "fixture", "key-1", context.portfolio.account_snapshot_id),)

    def _snapshot(bars: tuple[HistoricalBar, ...]) -> dict[str, object]:
        return {"mark_prices": {bar.instrument_id: float(bar.raw_close) for bar in bars}}

    bars_day0 = (HistoricalBar(days[0], instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),)
    bars_day1 = (HistoricalBar(days[1], instrument.instrument_id, 10.0, 12.0, 1_000_000.0, 0.02),)
    sessions = (
        BacktestSession(days[0], days[0] + timedelta(hours=6, minutes=30), bars_day0, (), _snapshot(bars_day0)),
        BacktestSession(days[1], days[1] + timedelta(hours=6, minutes=30), bars_day1, (), _snapshot(bars_day1)),
        BacktestSession(days[2], days[2] + timedelta(hours=6, minutes=30), (), (), _snapshot(())),
    )
    calendar = SessionCalendar((*days, days[2] + timedelta(days=1), days[2] + timedelta(days=2)))

    # When
    result = EventBacktester(BacktestConfig("bt-stale-mark", 1_000.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, calendar, model)).run(sessions, ValueBook())

    # Then: the halted session values the book at the carried-forward close, not a crash.
    assert len(result.fills) == 1
    assert len(observed_equity) == 3
    assert observed_equity[2] == pytest.approx(observed_equity[1], abs=1e-9)
