def test_compounding_v2_policy_rejects_mutation() -> None:
    import pytest
    from src.strategy.compounding_v2_strategy import CompoundingV2Policy

    policy = CompoundingV2Policy()
    assert (policy.max_positions, policy.entry_rank, policy.retention_rank) == (12, 12, 24)
    assert (policy.selection_rebalance_sessions, policy.risk_rebalance_sessions) == (20, 10)
    with pytest.raises(ValueError, match='immutable'):
        CompoundingV2Policy(max_positions=13)
    with pytest.raises(ValueError, match='immutable'):
        CompoundingV2Policy(target_market_volatility=0.20)


def test_select_compounding_v2_weights_honors_rank_hysteresis_and_cap() -> None:
    from datetime import datetime
    import pytest
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import KRX_TZ
    from src.strategy.compounding_v2_strategy import CompoundingV2Policy, select_compounding_v2_weights
    from src.strategy.scoring import ChampionScoreRow

    decision = datetime(2024, 6, 3, 15, 30, tzinfo=KRX_TZ)
    ids = tuple(f'KRX:{index:06d}' for index in range(1, 14))
    scores = tuple(ChampionScoreRow(decision, iid, True, 100.0 - index, index, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1') for index, iid in enumerate(ids, start=1))
    instruments = {iid: Instrument(iid, AssetKind.STOCK, 'KRX', iid[4:], 'KRW') for iid in ids}
    weights = select_compounding_v2_weights(scores=scores, held_ids=(ids[12],), marks=dict.fromkeys(ids, 10000.0), volatilities=dict.fromkeys(ids, 0.2), instruments=instruments, decision_time=decision, policy=CompoundingV2Policy())
    assert tuple(weights) == (*ids[:11], ids[12])
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert max(weights.values()) <= 0.10 + 1e-12


def test_compounding_v2_defers_entries_until_sale_proceeds_settle() -> None:
    from datetime import datetime, timedelta
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position
    from src.core.time import KRX_TZ, SessionCalendar
    from src.engine.decision import DecisionContext
    from src.strategy.compounding_v2_strategy import CompoundingV2Strategy
    from src.strategy.scoring import ChampionScoreRow

    sessions = tuple(datetime(2024, 1, 2, 9, tzinfo=KRX_TZ) + timedelta(days=index) for index in range(22))
    ids = tuple(f'KRX:{index:06d}' for index in range(1, 14))
    instruments = {iid: Instrument(iid, AssetKind.STOCK, 'KRX', iid[4:], 'KRW') for iid in ids}
    decision = sessions[20].replace(hour=15, minute=30)
    scores = tuple(ChampionScoreRow(decision, iid, True, 100.0 - index, index, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1') for index, iid in enumerate(ids, start=1))
    strategy = CompoundingV2Strategy(scores_by_session={decision.date(): scores}, calendar=SessionCalendar(sessions))
    snapshot = {'mark_prices': dict.fromkeys(ids, 10000.0), 'volatilities': dict.fromkeys(ids, 0.2), 'instruments': instruments, 'market_index_level': 2.0, 'market_index_sma100': 1.0, 'market_index_sma200': 1.0, 'market_volatility': 0.15}
    held = Position(instruments[ids[12]], 10.0, 10000.0)
    exits = strategy.decide(DecisionContext(decision, PortfolioSnapshot('acct', decision, 1_000_000.0, 0.0, (held,)), snapshot))
    assert {intent.instrument_id for intent in exits} == {ids[12]}
    assert exits[0].target_value == 0.0
    later = sessions[21].replace(hour=15, minute=30)
    deferred = strategy.decide(DecisionContext(later, PortfolioSnapshot('acct', later, 1_000_000.0, 100000.0, ()), snapshot))
    assert deferred == ()


def test_select_compounding_v2_weights_rejects_malformed_scores() -> None:
    from datetime import datetime
    import pytest
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import KRX_TZ
    from src.strategy.compounding_v2_strategy import CompoundingV2Policy, select_compounding_v2_weights
    from src.strategy.scoring import ChampionScoreRow

    decision = datetime(2024, 6, 3, 15, 30, tzinfo=KRX_TZ)
    policy = CompoundingV2Policy()
    instruments = {'KRX:000001': Instrument('KRX:000001', AssetKind.STOCK, 'KRX', '000001', 'KRW')}
    marks = {'KRX:000001': 10000.0}
    vols = {'KRX:000001': 0.20}
    good = ChampionScoreRow(decision, 'KRX:000001', True, 90.0, 1, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')
    with pytest.raises(ValueError, match='non-empty'):
        select_compounding_v2_weights(scores=(), held_ids=(), marks=marks, volatilities=vols, instruments=instruments, decision_time=decision, policy=policy)
    future = ChampionScoreRow(datetime(2024, 6, 4, 15, 30, tzinfo=KRX_TZ), 'KRX:000001', True, 90.0, 1, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')
    with pytest.raises(ValueError, match='after decision_time'):
        select_compounding_v2_weights(scores=(future,), held_ids=(), marks=marks, volatilities=vols, instruments=instruments, decision_time=decision, policy=policy)
    with pytest.raises(ValueError, match='duplicate'):
        select_compounding_v2_weights(scores=(good, good), held_ids=(), marks=marks, volatilities=vols, instruments=instruments, decision_time=decision, policy=policy)
    bad_rank = ChampionScoreRow(decision, 'KRX:000001', True, 90.0, None, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')
    with pytest.raises(ValueError, match='score/rank'):
        select_compounding_v2_weights(scores=(bad_rank,), held_ids=(), marks=marks, volatilities=vols, instruments=instruments, decision_time=decision, policy=policy)
    assert select_compounding_v2_weights(scores=(good,), held_ids=(), marks=marks, volatilities=vols, instruments=instruments, decision_time=decision, policy=policy) == {}


def test_compounding_v2_emits_target_allocations_when_settled() -> None:
    from datetime import datetime, timedelta
    import pytest
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.core.time import KRX_TZ, SessionCalendar
    from src.engine.decision import DecisionContext
    from src.strategy.compounding_v2_strategy import CompoundingV2Strategy
    from src.strategy.scoring import ChampionScoreRow

    sessions = tuple(datetime(2024, 1, 2, 9, tzinfo=KRX_TZ) + timedelta(days=index) for index in range(22))
    ids = tuple(f'KRX:{index:06d}' for index in range(1, 14))
    instruments = {iid: Instrument(iid, AssetKind.STOCK, 'KRX', iid[4:], 'KRW') for iid in ids}
    score_time = sessions[19].replace(hour=15, minute=30)
    scores = tuple(ChampionScoreRow(score_time, iid, True, 100.0 - index, index, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1') for index, iid in enumerate(ids, start=1))
    strategy = CompoundingV2Strategy(scores_by_session={score_time.date(): scores}, calendar=SessionCalendar(sessions))
    decision = sessions[20].replace(hour=15, minute=30)
    snapshot = {'mark_prices': dict.fromkeys(ids, 10000.0), 'volatilities': dict.fromkeys(ids, 0.20), 'instruments': instruments, 'market_index_level': 2.0, 'market_index_sma100': 1.0, 'market_index_sma200': 1.0, 'market_volatility': 0.15}
    intents = strategy.decide(DecisionContext(decision, PortfolioSnapshot('acct', decision, 1_000_000.0, 0.0, ()), snapshot))
    assert len(intents) == 12
    assert {intent.instrument_id for intent in intents} == set(ids[:12])
    assert sum(intent.target_value for intent in intents) == pytest.approx(1_000_000.0, abs=1e-6)
    assert max(intent.target_value for intent in intents) <= 100_000.0 + 1e-6
    assert all(intent.execution_time == sessions[21] for intent in intents)
    assert all(intent.strategy_id == 'compounding-v2' for intent in intents)


def test_compounding_v2_defers_new_entries_while_unsettled_cash_remains() -> None:
    from datetime import datetime, timedelta
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position
    from src.core.time import KRX_TZ, SessionCalendar
    from src.engine.decision import DecisionContext
    from src.strategy.compounding_v2_strategy import CompoundingV2Strategy
    from src.strategy.scoring import ChampionScoreRow

    sessions = tuple(datetime(2024, 1, 2, 9, tzinfo=KRX_TZ) + timedelta(days=index) for index in range(22))
    ids = tuple(f'KRX:{index:06d}' for index in range(1, 14))
    instruments = {iid: Instrument(iid, AssetKind.STOCK, 'KRX', iid[4:], 'KRW') for iid in ids}
    score_time = sessions[19].replace(hour=15, minute=30)
    scores = tuple(ChampionScoreRow(score_time, iid, True, 100.0 - index, index, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1') for index, iid in enumerate(ids, start=1))
    strategy = CompoundingV2Strategy(scores_by_session={score_time.date(): scores}, calendar=SessionCalendar(sessions))
    decision = sessions[20].replace(hour=15, minute=30)
    snapshot = {'mark_prices': dict.fromkeys(ids, 10000.0), 'volatilities': dict.fromkeys(ids, 0.20), 'instruments': instruments, 'market_index_level': 2.0, 'market_index_sma100': 1.0, 'market_index_sma200': 1.0, 'market_volatility': 0.15}
    held = Position(instruments[ids[0]], 10.0, 10000.0)
    intents = strategy.decide(DecisionContext(decision, PortfolioSnapshot('acct', decision, 900_000.0, 100_000.0, (held,)), snapshot))
    assert {intent.instrument_id for intent in intents} == {ids[0]}
    assert intents[0].target_value == 100_000.0
