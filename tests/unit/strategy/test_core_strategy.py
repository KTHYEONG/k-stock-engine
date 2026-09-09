from datetime import UTC, datetime, timedelta

from src.core.time import SessionCalendar
from src.strategy.core_strategy import CoreStrategyPolicy, is_core_rebalance_session, score_core_candidates

def test_core_score_deterministic() -> None:
    decision_time = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    scores = score_core_candidates(candidate_ids=('KRX:B', 'KRX:A', 'KRX:C'), decision_time=decision_time, market_caps={'KRX:A': 100.0, 'KRX:B': 100.0, 'KRX:C': 50.0}, volatilities={'KRX:A': 0.2, 'KRX:B': 0.2, 'KRX:C': 0.1}, policy=CoreStrategyPolicy())
    assert [row.instrument_id for row in scores if row.eligible] == ['KRX:A', 'KRX:C', 'KRX:B']
    assert [row.rank for row in scores if row.eligible] == [1, 2, 3]


def test_core_policy_constants_are_immutable() -> None:
    import pytest

    from src.strategy.core_strategy import CoreStrategyPolicy

    with pytest.raises(ValueError, match='immutable'):
        CoreStrategyPolicy(rebalance_sessions=5)


def test_core_calendar_cadence_and_next_open() -> None:
    from src.core.time import KRX_TZ

    sessions = tuple(datetime(2024, 1, 2, 9, tzinfo=KRX_TZ) + timedelta(days=index) for index in range(41))
    calendar = SessionCalendar(sessions)
    policy = CoreStrategyPolicy()
    assert is_core_rebalance_session(calendar=calendar, decision_time=sessions[0].replace(hour=15, minute=30), policy=policy)
    assert not is_core_rebalance_session(calendar=calendar, decision_time=sessions[1].replace(hour=15, minute=30), policy=policy)
    assert is_core_rebalance_session(calendar=calendar, decision_time=sessions[20].replace(hour=15, minute=30), policy=policy)
    assert calendar.advance(sessions[20].replace(hour=15, minute=30), 1) == sessions[21]


def test_core_score_rejects_bad_inputs() -> None:
    from datetime import UTC, datetime

    import pytest

    from src.strategy.core_strategy import CoreStrategyPolicy, score_core_candidates

    decision_time = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    policy = CoreStrategyPolicy()
    good_caps = {'KRX:A': 100.0}
    good_vols = {'KRX:A': 0.2}
    with pytest.raises(ValueError, match='timezone-aware'):
        score_core_candidates(candidate_ids=('KRX:A',), decision_time=decision_time.replace(tzinfo=None), market_caps=good_caps, volatilities=good_vols, policy=policy)
    with pytest.raises(ValueError, match='non-empty'):
        score_core_candidates(candidate_ids=(), decision_time=decision_time, market_caps={}, volatilities={}, policy=policy)
    with pytest.raises(ValueError, match='duplicate'):
        score_core_candidates(candidate_ids=('KRX:A', 'KRX:A'), decision_time=decision_time, market_caps=good_caps, volatilities=good_vols, policy=policy)
    with pytest.raises(ValueError, match='non-empty'):
        score_core_candidates(candidate_ids=('  ',), decision_time=decision_time, market_caps={'  ': 1.0}, volatilities={'  ': 0.1}, policy=policy)
    with pytest.raises(ValueError, match='missing snapshot'):
        score_core_candidates(candidate_ids=('KRX:A', 'KRX:B'), decision_time=decision_time, market_caps=good_caps, volatilities=good_vols, policy=policy)
    with pytest.raises(ValueError, match='non-positive'):
        score_core_candidates(candidate_ids=('KRX:A',), decision_time=decision_time, market_caps={'KRX:A': 0.0}, volatilities=good_vols, policy=policy)
    with pytest.raises(ValueError, match='non-positive'):
        score_core_candidates(candidate_ids=('KRX:A',), decision_time=decision_time, market_caps=good_caps, volatilities={'KRX:A': -1.0}, policy=policy)


def test_core_rebalance_unknown_session_raises() -> None:
    from datetime import datetime, timedelta

    import pytest

    from src.core.time import KRX_TZ, SessionCalendar
    from src.strategy.core_strategy import CoreStrategyPolicy, is_core_rebalance_session

    sessions = tuple(datetime(2024, 1, 2, 9, tzinfo=KRX_TZ) + timedelta(days=index) for index in range(3))
    calendar = SessionCalendar(sessions)
    policy = CoreStrategyPolicy()
    with pytest.raises(ValueError, match='timezone-aware'):
        is_core_rebalance_session(calendar=calendar, decision_time=sessions[0].replace(tzinfo=None), policy=policy)
    with pytest.raises(ValueError, match='not on calendar'):
        is_core_rebalance_session(calendar=calendar, decision_time=datetime(2023, 1, 1, 15, 30, tzinfo=KRX_TZ), policy=policy)


def test_core_decide_rebalance_hold_and_missing() -> None:
    from datetime import datetime, timedelta

    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.core.time import KRX_TZ, SessionCalendar
    from src.engine.decision import DecisionContext
    from src.strategy.core_strategy import CoreStrategy

    sessions = tuple(
        datetime(2024, 1, 2, 9, tzinfo=KRX_TZ) + timedelta(days=index) for index in range(22)
    )
    calendar = SessionCalendar(sessions)
    eligible = {sessions[0].date(): ('KRX:A', 'KRX:B'), sessions[1].date(): ('KRX:A',)}
    strategy = CoreStrategy(eligible_by_session=eligible, calendar=calendar)
    instruments = {
        iid: Instrument(iid, AssetKind.STOCK, 'KRX', iid.split(':')[-1], 'KRW') for iid in ('KRX:A', 'KRX:B')
    }
    snapshot = {
        'mark_prices': {'KRX:A': 10000.0, 'KRX:B': 20000.0},
        'market_caps': {'KRX:A': 1e12, 'KRX:B': 5e11},
        'adtv20': {'KRX:A': 1e12, 'KRX:B': 1e12},
        'volatilities': {'KRX:A': 0.2, 'KRX:B': 0.25},
        'sectors': {'KRX:A': 'Technology', 'KRX:B': 'Healthcare'},
        'instruments': instruments,
        'market_volatility': 0.15,
    }
    rebalance_time = sessions[0].replace(hour=15, minute=30)
    portfolio = PortfolioSnapshot('account-1', rebalance_time, 1_000_000_000.0, 0.0, ())
    intents = strategy.decide(DecisionContext(decision_time=rebalance_time, portfolio=portfolio, market_snapshot=snapshot))
    assert len(intents) == 2
    assert intents[0].execution_time == sessions[1]
    assert intents[0].strategy_id == 'core-v1'
    off_time = sessions[1].replace(hour=15, minute=30)
    off_portfolio = PortfolioSnapshot('account-1', off_time, 1_000_000_000.0, 0.0, ())
    assert strategy.decide(DecisionContext(decision_time=off_time, portfolio=portfolio, market_snapshot=snapshot)) == ()
    gap_time = sessions[20].replace(hour=15, minute=30)
    gap_portfolio = PortfolioSnapshot('account-1', gap_time, 1_000_000_000.0, 0.0, ())
    assert strategy.decide(DecisionContext(decision_time=gap_time, portfolio=gap_portfolio, market_snapshot=snapshot)) == ()
    from src.strategy.portfolio import ChampionPortfolioPolicy

    explicit = CoreStrategy(
        eligible_by_session=eligible,
        calendar=calendar,
        portfolio_policy=ChampionPortfolioPolicy(required_selection_policy_version='korean-core-v1-selection-v1'),
    )
    assert explicit.decide(DecisionContext(decision_time=off_time, portfolio=off_portfolio, market_snapshot=snapshot)) == ()
