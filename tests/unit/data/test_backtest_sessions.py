def test_bound_calendar_to_market_window_trims_to_market_span() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    from src.data.backtest_sessions import bound_calendar_to_market_window

    kst = ZoneInfo("Asia/Seoul")
    calendar = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(10))
    market = {calendar[3], calendar[5], calendar[6]}

    bounded = bound_calendar_to_market_window(calendar, market)

    assert bounded == calendar[3:7]

def test_bound_calendar_to_market_window_keeps_interior_sessions_absent_from_market() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    from src.data.backtest_sessions import bound_calendar_to_market_window

    kst = ZoneInfo("Asia/Seoul")
    calendar = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(5))
    market = {calendar[0], calendar[4]}

    bounded = bound_calendar_to_market_window(calendar, market)

    assert bounded == calendar
    assert calendar[2] in bounded

def test_bound_calendar_to_market_window_rejects_empty_market_sessions() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import pytest

    from src.data.backtest_sessions import bound_calendar_to_market_window
    from src.data.schemas import PITDataError

    kst = ZoneInfo("Asia/Seoul")
    calendar = (dt.datetime(2024, 1, 2, 9, tzinfo=kst),)

    with pytest.raises(PITDataError, match="at least one market session"):
        bound_calendar_to_market_window(calendar, set())

def test_bound_calendar_to_market_window_rejects_empty_calendar() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import pytest

    from src.data.backtest_sessions import bound_calendar_to_market_window
    from src.data.schemas import PITDataError

    kst = ZoneInfo("Asia/Seoul")

    with pytest.raises(PITDataError, match="calendar must be non-empty"):
        bound_calendar_to_market_window((), {dt.datetime(2024, 1, 2, 9, tzinfo=kst)})

def test_rolling_inputs_result_is_invariant_to_calendar_padding() -> None:
    """캘린더 앞뒤 패딩 세션은 롤링 산출물을 바꾸지 않아야 한다 (윈도우 바운딩 등가성)."""
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.backtest_sessions import BacktestMarketInputsPolicy, _rolling_inputs

    kst = ZoneInfo("Asia/Seoul")
    core = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(80))
    padding_before = tuple(core[0] - timedelta(days=i) for i in range(40, 0, -1))
    padding_after = tuple(core[-1] + timedelta(days=i) for i in range(1, 41))
    frame = pl.DataFrame(
        {
            "session": list(core),
            "instrument_id": ["KRX:A"] * len(core),
            "close": [10000.0 + i * 7.0 for i in range(len(core))],
            "trading_value": [1.0e9 + i for i in range(len(core))],
            "market_cap": [1.0e12] * len(core),
        }
    )
    policy = BacktestMarketInputsPolicy()

    narrow = _rolling_inputs(frame, policy, core, None)
    wide = _rolling_inputs(frame, policy, padding_before + core + padding_after, None)

    assert narrow[0] == wide[0]
    assert narrow[1] == wide[1]
    assert narrow[2] == wide[2]
    assert narrow[0]

