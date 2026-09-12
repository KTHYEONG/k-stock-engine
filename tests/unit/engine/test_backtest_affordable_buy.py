def test_constrain_buy_to_settled_cash_scales_lot_and_reports_remainder() -> None:
    from datetime import datetime
    from src.core.costs import LiquiditySlippageModel, default_base_schedule, default_krx_tick_schedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.ledger import Ledger
    from src.core.time import KRX_TZ
    from src.engine.backtest import constrain_buy_to_settled_cash
    from src.engine.fill_model import BacktestOrder, ExecutionScenario, FillOutcome, HistoricalBar, HistoricalFillModel
    from src.execution.domain.orders import OrderSide

    session = datetime(2024, 6, 3, 9, tzinfo=KRX_TZ)
    instrument = Instrument('KRX:000001', AssetKind.STOCK, 'KRX', '000001', 'KRW')
    model = HistoricalFillModel(default_base_schedule(), LiquiditySlippageModel(0.1, default_krx_tick_schedule()), ExecutionScenario.BASE, target_participation_cap=0.10, hard_participation_cap=0.20)
    order = BacktestOrder('order-1', 'intent-1', instrument, OrderSide.BUY, 10, session, session)
    bar = HistoricalBar(session, instrument.instrument_id, 100.0, 100.0, 1_000_000.0, 0.20)
    result, cash_remainder = constrain_buy_to_settled_cash(ledger=Ledger('ledger', 350.0, session), fill_model=model, order=order, bar=bar)
    assert isinstance(result, FillOutcome)
    assert 0 < result.fill.quantity < 10
    assert cash_remainder == 10 - result.fill.quantity


def test_constrain_buy_to_settled_cash_rejects_when_no_lot_affordable() -> None:
    from datetime import datetime
    from src.core.costs import LiquiditySlippageModel, default_base_schedule, default_krx_tick_schedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.ledger import Ledger
    from src.core.time import KRX_TZ
    from src.engine.backtest import constrain_buy_to_settled_cash
    from src.engine.fill_model import BacktestOrder, BacktestReject, ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.orders import OrderSide

    session = datetime(2024, 6, 3, 9, tzinfo=KRX_TZ)
    instrument = Instrument('KRX:000001', AssetKind.STOCK, 'KRX', '000001', 'KRW')
    model = HistoricalFillModel(default_base_schedule(), LiquiditySlippageModel(0.1, default_krx_tick_schedule()), ExecutionScenario.BASE, target_participation_cap=0.10, hard_participation_cap=0.20)
    order = BacktestOrder('order-9', 'intent-9', instrument, OrderSide.BUY, 10, session, session)
    bar = HistoricalBar(session, instrument.instrument_id, 100.0, 100.0, 1_000_000.0, 0.20)
    result, remainder = constrain_buy_to_settled_cash(ledger=Ledger('ledger', 0.0, session), fill_model=model, order=order, bar=bar)
    assert isinstance(result, BacktestReject)
    assert result.reason == 'insufficient settled cash'
    assert remainder == 10


def test_backtester_scales_buy_to_settled_cash_and_records_remainder() -> None:
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
    instrument = Instrument('KRX:005930', AssetKind.STOCK, 'KRX', '005930', 'KRW')
    costs = CostSchedule('fixture', (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule('all', datetime(2020, 1, 1, tzinfo=UTC), 0.0, float('inf'), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.1, hard_participation_cap=0.2)

    class BuyOnce:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time.date() != day_one.date():
                return ()
            return (TradeIntent('intent-1', AssetKind.STOCK, instrument.instrument_id, 100.0, context.decision_time, day_two, 'champion-v1', 'fixture', 'key-1', context.portfolio.account_snapshot_id),)

    sessions = (BacktestSession(day_one, day_one + timedelta(hours=6, minutes=30), (HistoricalBar(day_one, instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object()), BacktestSession(day_two, day_two + timedelta(hours=6, minutes=30), (HistoricalBar(day_two, instrument.instrument_id, 11.0, 11.0, 1_000_000.0, 0.02),), (), object()))
    calendar = SessionCalendar((day_one, day_two, day_two + timedelta(days=1), day_two + timedelta(days=2)))
    result = EventBacktester(BacktestConfig('bt-cash', 15.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, calendar, model)).run(sessions, BuyOnce())
    assert len(result.fills) == 1
    assert 1 <= result.fills[0].quantity < 10
    assert any(reject.reason == 'insufficient settled cash remainder' for reject in result.rejects)
