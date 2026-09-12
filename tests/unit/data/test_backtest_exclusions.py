def test_resolve_backtest_exclusion_plan_unions_corporate_action_and_terminal_close() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_exclusions import resolve_backtest_exclusion_plan
    from src.data.backtest_sessions import BacktestMarketInputsPolicy

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(4))
    # KRX:A 는 전 구간 보유, KRX:B 는 마지막 세션 직전에 소멸 -> terminal close 결측
    rows = [(d, "KRX:A") for d in days] + [(d, "KRX:B") for d in days[:2]]
    daily_market = pl.DataFrame({
        "session": [r[0] for r in rows],
        "instrument_id": [r[1] for r in rows],
        "open": [100.0] * len(rows),
        "close": [100.0] * len(rows),
        "volume": [1000.0] * len(rows),
        "trading_value": [100000.0] * len(rows),
        "market_cap": [1e12] * len(rows),
        "available_at": [r[0].replace(hour=15, minute=30) for r in rows],
    })
    corporate_actions = pl.DataFrame(schema={
        "instrument_id": pl.String,
        "effective_date": pl.Datetime(time_zone="Asia/Seoul"),
        "type": pl.String,
        "factor": pl.Float64,
        "cash_amount": pl.Float64,
        "available_at": pl.Datetime(time_zone="Asia/Seoul"),
        "evidence_status": pl.String,
        "evidence_reason": pl.String,
    })

    plan = resolve_backtest_exclusion_plan(
        daily_market=daily_market,
        corporate_actions=corporate_actions,
        calendar=SessionCalendar(days),
        policy=BacktestMarketInputsPolicy(),
        terminal_session=days[-1],
    )

    assert plan.missing_terminal_close_instruments == frozenset({"KRX:B"})
    assert plan.corporate_action_instruments == frozenset()
    assert plan.excluded_instruments == frozenset({"KRX:B"})
    assert plan.eligible_daily_market["instrument_id"].unique().to_list() == ["KRX:A"]


def test_resolve_backtest_exclusion_plan_quarantines_unexplained_price_jump() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_exclusions import resolve_backtest_exclusion_plan
    from src.data.backtest_sessions import BacktestMarketInputsPolicy

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(4))
    closes_a = [100.0, 100.0, 100.0, 100.0]
    closes_b = [100.0, 100.0, 20.0, 20.0]  # 검증되지 않은 80% 하락 = 미설명 불연속
    rows = [(d, "KRX:A", c) for d, c in zip(days, closes_a, strict=True)]
    rows += [(d, "KRX:B", c) for d, c in zip(days, closes_b, strict=True)]
    daily_market = pl.DataFrame({
        "session": [r[0] for r in rows],
        "instrument_id": [r[1] for r in rows],
        "open": [r[2] for r in rows],
        "close": [r[2] for r in rows],
        "volume": [1000.0] * len(rows),
        "trading_value": [100000.0] * len(rows),
        "market_cap": [1e12] * len(rows),
        "available_at": [r[0].replace(hour=15, minute=30) for r in rows],
    })
    corporate_actions = pl.DataFrame(schema={
        "instrument_id": pl.String,
        "effective_date": pl.Datetime(time_zone="Asia/Seoul"),
        "type": pl.String,
        "factor": pl.Float64,
        "cash_amount": pl.Float64,
        "available_at": pl.Datetime(time_zone="Asia/Seoul"),
        "evidence_status": pl.String,
        "evidence_reason": pl.String,
    })

    plan = resolve_backtest_exclusion_plan(
        daily_market=daily_market,
        corporate_actions=corporate_actions,
        calendar=SessionCalendar(days),
        policy=BacktestMarketInputsPolicy(),
        terminal_session=days[-1],
    )

    assert "KRX:B" in plan.corporate_action_instruments
    assert plan.reasons["KRX:B"] == ("unexplained_price_discontinuity",)
    assert "KRX:B" not in plan.eligible_daily_market["instrument_id"].to_list()
    assert plan.quarantine_sessions_by_instrument["KRX:B"]


def test_resolve_backtest_exclusion_plan_orders_corporate_action_before_terminal_close() -> None:
    """CA 제외 종목은 terminal-close 집합에 중복 계상되지 않아야 한다 (현행 순서 등가)."""
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_exclusions import resolve_backtest_exclusion_plan
    from src.data.backtest_sessions import BacktestMarketInputsPolicy

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(4))
    # KRX:B 는 미설명 불연속 + 마지막 세션 결측을 동시에 가진다
    rows = [(d, "KRX:A", 100.0) for d in days]
    rows += [(days[0], "KRX:B", 100.0), (days[1], "KRX:B", 100.0), (days[2], "KRX:B", 20.0)]
    daily_market = pl.DataFrame({
        "session": [r[0] for r in rows],
        "instrument_id": [r[1] for r in rows],
        "open": [r[2] for r in rows],
        "close": [r[2] for r in rows],
        "volume": [1000.0] * len(rows),
        "trading_value": [100000.0] * len(rows),
        "market_cap": [1e12] * len(rows),
        "available_at": [r[0].replace(hour=15, minute=30) for r in rows],
    })
    corporate_actions = pl.DataFrame(schema={
        "instrument_id": pl.String,
        "effective_date": pl.Datetime(time_zone="Asia/Seoul"),
        "type": pl.String,
        "factor": pl.Float64,
        "cash_amount": pl.Float64,
        "available_at": pl.Datetime(time_zone="Asia/Seoul"),
        "evidence_status": pl.String,
        "evidence_reason": pl.String,
    })

    plan = resolve_backtest_exclusion_plan(
        daily_market=daily_market,
        corporate_actions=corporate_actions,
        calendar=SessionCalendar(days),
        policy=BacktestMarketInputsPolicy(),
        terminal_session=days[-1],
    )

    assert "KRX:B" in plan.corporate_action_instruments
    assert "KRX:B" not in plan.missing_terminal_close_instruments
    assert plan.excluded_instruments == frozenset({"KRX:B"})


def test_resolve_backtest_exclusion_plan_rejects_empty_daily_market() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_exclusions import resolve_backtest_exclusion_plan
    from src.data.backtest_sessions import BacktestMarketInputsPolicy
    from src.data.schemas import PITDataError

    kst = ZoneInfo("Asia/Seoul")
    day = dt.datetime(2024, 1, 2, 9, tzinfo=kst)
    daily_market = pl.DataFrame(schema={
        "session": pl.Datetime(time_zone="Asia/Seoul"),
        "instrument_id": pl.String,
        "open": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "trading_value": pl.Float64,
        "market_cap": pl.Float64,
        "available_at": pl.Datetime(time_zone="Asia/Seoul"),
    })

    with pytest.raises(PITDataError, match="daily market"):
        resolve_backtest_exclusion_plan(
            daily_market=daily_market,
            corporate_actions=pl.DataFrame(),
            calendar=SessionCalendar((day,)),
            policy=BacktestMarketInputsPolicy(),
            terminal_session=day,
        )


def test_resolve_backtest_exclusion_plan_rejects_terminal_session_outside_calendar() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_exclusions import resolve_backtest_exclusion_plan
    from src.data.backtest_sessions import BacktestMarketInputsPolicy
    from src.data.schemas import PITDataError

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(2))
    daily_market = pl.DataFrame({
        "session": list(days),
        "instrument_id": ["KRX:A", "KRX:A"],
        "open": [100.0, 100.0],
        "close": [100.0, 100.0],
        "volume": [1000.0, 1000.0],
        "trading_value": [100000.0, 100000.0],
        "market_cap": [1e12, 1e12],
        "available_at": [d.replace(hour=15, minute=30) for d in days],
    })

    with pytest.raises(PITDataError, match="terminal session"):
        resolve_backtest_exclusion_plan(
            daily_market=daily_market,
            corporate_actions=pl.DataFrame(),
            calendar=SessionCalendar(days),
            policy=BacktestMarketInputsPolicy(),
            terminal_session=days[-1] + timedelta(days=30),
        )


def test_resolve_backtest_exclusion_plan_is_deterministic_for_identical_inputs() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_exclusions import resolve_backtest_exclusion_plan
    from src.data.backtest_sessions import BacktestMarketInputsPolicy

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(4))
    rows = [(d, iid) for d in days for iid in ("KRX:A", "KRX:B")]
    daily_market = pl.DataFrame({
        "session": [r[0] for r in rows],
        "instrument_id": [r[1] for r in rows],
        "open": [100.0] * len(rows),
        "close": [100.0] * len(rows),
        "volume": [1000.0] * len(rows),
        "trading_value": [100000.0] * len(rows),
        "market_cap": [1e12] * len(rows),
        "available_at": [r[0].replace(hour=15, minute=30) for r in rows],
    })
    args = {
        "daily_market": daily_market,
        "corporate_actions": pl.DataFrame(),
        "calendar": SessionCalendar(days),
        "policy": BacktestMarketInputsPolicy(),
        "terminal_session": days[-1],
    }

    first = resolve_backtest_exclusion_plan(**args)
    second = resolve_backtest_exclusion_plan(**args)

    assert first.excluded_instruments == second.excluded_instruments
    assert dict(first.reasons) == dict(second.reasons)
    assert first.eligible_daily_market.equals(second.eligible_daily_market)


def test_backtest_exclusion_plan_reason_counts_aggregates_by_label() -> None:
    import polars as pl

    from src.data.backtest_exclusions import BacktestExclusionPlan

    plan = BacktestExclusionPlan(
        eligible_daily_market=pl.DataFrame(),
        corporate_action_instruments=frozenset({"KRX:A", "KRX:B"}),
        missing_terminal_close_instruments=frozenset({"KRX:C"}),
        reasons={
            "KRX:A": ("unexplained_price_discontinuity",),
            "KRX:B": ("unexplained_price_discontinuity",),
            "KRX:C": ("missing_terminal_close",),
        },
        quarantine_sessions_by_instrument={},
    )

    assert plan.excluded_instruments == frozenset({"KRX:A", "KRX:B", "KRX:C"})
    assert plan.reason_counts() == {"missing_terminal_close": 1, "unexplained_price_discontinuity": 2}
