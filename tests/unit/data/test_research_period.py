def test_earliest_evaluable_decision_date_derives_fy2015_lookback_floor() -> None:
    from datetime import date

    from src.data.dart_backfill import (
        QVEF_FUNDAMENTAL_LOOKBACK_QUARTERS,
        _latest_available_quarter,
        _quarters_back,
    )
    from src.data.research_period import OPENDART_FIRST_FISCAL_YEAR, earliest_evaluable_decision_date

    # Given: OpenDART FY2015 floor and the TTM + YoY lookback (5 quarters).
    assert OPENDART_FIRST_FISCAL_YEAR == 2015
    assert QVEF_FUNDAMENTAL_LOOKBACK_QUARTERS == 5

    # When
    floor = earliest_evaluable_decision_date()

    # Then: 2016Q1 filings (cutoff 2016-05-15) are the first with a full lookback.
    assert floor == date(2016, 5, 16)
    assert _quarters_back(_latest_available_quarter(floor), QVEF_FUNDAMENTAL_LOOKBACK_QUARTERS)[0] == "2015Q1"
    assert _quarters_back(_latest_available_quarter(date(2016, 5, 15)), QVEF_FUNDAMENTAL_LOOKBACK_QUARTERS)[0] == "2014Q4"

    # And: the derivation is parametric in the lookback.
    assert earliest_evaluable_decision_date(first_fiscal_year=2015, lookback_quarters=8) == date(2017, 3, 31)
    assert earliest_evaluable_decision_date(first_fiscal_year=2015, lookback_quarters=1) == date(2015, 5, 16)


def test_earliest_evaluable_decision_date_rejects_invalid_inputs() -> None:
    import pytest

    from src.data.research_period import earliest_evaluable_decision_date

    for bad in ({"first_fiscal_year": 1999}, {"first_fiscal_year": True}):
        with pytest.raises(ValueError, match="first_fiscal_year"):
            earliest_evaluable_decision_date(**bad)
    for bad in ({"lookback_quarters": 0}, {"lookback_quarters": True}):
        with pytest.raises(ValueError, match="lookback_quarters"):
            earliest_evaluable_decision_date(**bad)


def test_research_period_policy_defaults_and_fail_closed_bounds() -> None:
    from datetime import date

    import pytest

    from src.data.research_period import ResearchPeriodPolicy

    # Given/When
    policy = ResearchPeriodPolicy()

    # Then: approved research convention.
    assert policy.first_fiscal_year == 2015
    assert policy.evaluation_start == date(2017, 4, 1)
    assert policy.development_end == date(2023, 12, 31)
    assert policy.holdout_start == date(2024, 1, 1)

    # And: the floor itself is accepted, one day earlier is not.
    assert ResearchPeriodPolicy(evaluation_start=date(2016, 5, 16)).evaluation_start == date(2016, 5, 16)
    with pytest.raises(ValueError, match="evaluation_start"):
        ResearchPeriodPolicy(evaluation_start=date(2016, 5, 15))
    with pytest.raises(ValueError, match="first_fiscal_year"):
        ResearchPeriodPolicy(first_fiscal_year=2014)
    with pytest.raises(ValueError, match="development_end"):
        ResearchPeriodPolicy(development_end=date(2017, 3, 31), holdout_start=date(2017, 4, 1))
    with pytest.raises(ValueError, match="holdout_start"):
        ResearchPeriodPolicy(holdout_start=date(2024, 1, 2))


def test_summarize_research_segments_splits_development_and_holdout() -> None:
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    import pytest

    from src.core.ledger import LedgerNav
    from src.data.research_period import ResearchPeriodPolicy, summarize_research_segments

    def _nav(day, nav):
        return LedgerNav(
            mark_id=f"m-{day.isoformat()}",
            as_of=datetime(day.year, day.month, day.day, 15, 30, tzinfo=ZoneInfo("Asia/Seoul")),
            nav=nav, settled_cash=nav, unsettled_cash=0.0, marked_value=0.0,
        )

    # Given: a warmup mark before 2017-04-01, two development marks, three holdout marks.
    navs = (
        _nav(date(2017, 3, 31), 100.0),
        _nav(date(2017, 4, 3), 110.0),
        _nav(date(2023, 12, 28), 121.0),
        _nav(date(2024, 1, 2), 133.1),
        _nav(date(2024, 12, 30), 119.79),
        _nav(date(2025, 1, 2), 131.769),
    )

    # When
    out = summarize_research_segments(navs, policy=ResearchPeriodPolicy())

    # Then
    assert set(out) == {"development", "holdout"}
    dev, hold = out["development"], out["holdout"]
    assert (dev["start"], dev["end"], dev["sessions"]) == ("2017-04-03", "2023-12-28", 2)
    assert dev["total_return"] == pytest.approx(0.21)
    assert dev["yearly_returns"] == {"2017": pytest.approx(0.1), "2023": pytest.approx(0.1)}
    assert dev["max_drawdown"] == pytest.approx(0.0)
    assert (hold["start"], hold["end"], hold["sessions"]) == ("2024-01-02", "2025-01-02", 3)
    assert hold["total_return"] == pytest.approx(131.769 / 121.0 - 1.0)
    assert hold["yearly_returns"] == {"2024": pytest.approx(119.79 / 121.0 - 1.0), "2025": pytest.approx(0.1)}
    assert hold["max_drawdown"] == pytest.approx(0.1)
    assert set(hold) == {"start", "end", "sessions", "total_return", "cagr", "max_drawdown", "annualized_volatility", "yearly_returns"}


def test_summarize_research_segments_omits_uncovered_segments() -> None:
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    from src.core.ledger import LedgerNav
    from src.data.research_period import summarize_research_segments

    def _nav(day, nav):
        return LedgerNav(
            mark_id=f"m-{day.isoformat()}",
            as_of=datetime(day.year, day.month, day.day, 15, 30, tzinfo=ZoneInfo("Asia/Seoul")),
            nav=nav, settled_cash=nav, unsettled_cash=0.0, marked_value=0.0,
        )

    # Given/When/Then: a 2016 window lies entirely before evaluation_start.
    assert summarize_research_segments((_nav(date(2016, 1, 4), 100.0), _nav(date(2016, 12, 29), 101.0))) == {}

    # And: a single in-segment mark with no prior base cannot form a return.
    assert summarize_research_segments((_nav(date(2024, 1, 2), 100.0),)) == {}

    # And: a holdout-only run reports only the holdout.
    only_holdout = summarize_research_segments((_nav(date(2024, 1, 2), 100.0), _nav(date(2024, 1, 3), 99.0)))
    assert set(only_holdout) == {"holdout"}
    assert only_holdout["holdout"]["sessions"] == 2

