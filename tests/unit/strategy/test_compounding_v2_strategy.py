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


def test_select_compounding_v2_weights_fills_band_when_gold_ranks_have_holes() -> None:
    from datetime import datetime

    import pytest

    from src.core.instruments import AssetKind, Instrument
    from src.core.time import KRX_TZ
    from src.strategy.compounding_v2_strategy import CompoundingV2Policy, select_compounding_v2_weights
    from src.strategy.scoring import ChampionScoreRow

    # Given: 20 scored names, but only every rank in `tradable_ranks` survives the
    # backtest-side mark/volatility/instrument filters.
    decision = datetime(2024, 6, 3, 15, 30, tzinfo=KRX_TZ)
    policy = CompoundingV2Policy()
    tradable_ranks = (1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 17, 20)
    all_ids = {rank: f'KRX:{rank:06d}' for rank in range(1, 21)}
    scores = tuple(
        ChampionScoreRow(decision, all_ids[rank], True, 100.0 - rank, rank, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')
        for rank in range(1, 21)
    )
    tradable_ids = [all_ids[rank] for rank in tradable_ranks]
    marks = dict.fromkeys(tradable_ids, 10000.0)
    vols = dict.fromkeys(tradable_ids, 0.2)
    instruments = {iid: Instrument(iid, AssetKind.STOCK, 'KRX', iid[4:], 'KRW') for iid in tradable_ids}
    assert len([rank for rank in tradable_ranks if rank <= policy.entry_rank]) == 9

    # When
    weights = select_compounding_v2_weights(
        scores=scores,
        held_ids=(),
        marks=marks,
        volatilities=vols,
        instruments=instruments,
        decision_time=decision,
        policy=policy,
    )

    # Then: dense ranks 1..12 over the tradable set fill the whole book.
    assert len(weights) == policy.max_positions
    assert set(weights) == set(tradable_ids)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert max(weights.values()) <= policy.security_weight_cap + 1e-12


def test_select_compounding_v2_weights_allows_partial_book_capped_at_security_weight() -> None:
    from datetime import datetime

    import pytest

    from src.core.instruments import AssetKind, Instrument
    from src.core.time import KRX_TZ
    from src.strategy.compounding_v2_strategy import CompoundingV2Policy, select_compounding_v2_weights
    from src.strategy.scoring import ChampionScoreRow

    decision = datetime(2024, 6, 3, 15, 30, tzinfo=KRX_TZ)
    policy = CompoundingV2Policy()

    def _build(count: int) -> dict[str, float]:
        ids = tuple(f'KRX:{index:06d}' for index in range(1, count + 1))
        scores = tuple(
            ChampionScoreRow(decision, iid, True, 100.0 - index, index, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')
            for index, iid in enumerate(ids, start=1)
        )
        instruments = {iid: Instrument(iid, AssetKind.STOCK, 'KRX', iid[4:], 'KRW') for iid in ids}
        return select_compounding_v2_weights(
            scores=scores,
            held_ids=(),
            marks=dict.fromkeys(ids, 10000.0),
            volatilities=dict.fromkeys(ids, 0.2),
            instruments=instruments,
            decision_time=decision,
            policy=policy,
        )

    # When/Then: 9 names is at or above the floor -> invested at the cap, residual in cash.
    partial = _build(9)
    assert len(partial) == 9
    assert all(value == pytest.approx(policy.security_weight_cap, abs=1e-12) for value in partial.values())
    assert sum(partial.values()) == pytest.approx(0.9, abs=1e-12)

    # When/Then: below the floor -> fail closed with an empty mapping.
    assert _build(policy.min_positions - 1) == {}


def test_dense_decision_ranks_and_tradable_score_rows_drop_untradable_names() -> None:
    from datetime import datetime
    from math import nan

    from src.core.instruments import AssetKind, Instrument
    from src.core.time import KRX_TZ
    from src.strategy.compounding_v2_strategy import dense_decision_ranks, tradable_score_rows
    from src.strategy.scoring import ChampionScoreRow

    # Given: one survivor per failure mode plus three healthy rows.
    decision = datetime(2024, 6, 3, 15, 30, tzinfo=KRX_TZ)
    good = ('KRX:000001', 'KRX:000004', 'KRX:000007')
    rows = (
        ChampionScoreRow(decision, 'KRX:000001', True, 99.0, 1, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1'),
        ChampionScoreRow(decision, 'KRX:000002', False, None, None, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1'),
        ChampionScoreRow(decision, 'KRX:000003', True, 97.0, 3, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1'),
        ChampionScoreRow(decision, 'KRX:000004', True, 96.0, 4, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1'),
        ChampionScoreRow(decision, 'KRX:000005', True, 95.0, 5, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1'),
        ChampionScoreRow(decision, 'KRX:000006', True, 94.0, 6, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1'),
        ChampionScoreRow(decision, 'KRX:000007', True, 93.0, 7, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1'),
    )
    marks = {'KRX:000001': 10000.0, 'KRX:000003': 0.0, 'KRX:000004': 10000.0, 'KRX:000005': 10000.0, 'KRX:000006': 10000.0, 'KRX:000007': 10000.0}
    vols = {'KRX:000001': 0.2, 'KRX:000003': 0.2, 'KRX:000004': 0.2, 'KRX:000005': nan, 'KRX:000006': 0.2, 'KRX:000007': 0.2}
    instruments = {
        'KRX:000001': Instrument('KRX:000001', AssetKind.STOCK, 'KRX', '000001', 'KRW'),
        'KRX:000003': Instrument('KRX:000003', AssetKind.STOCK, 'KRX', '000003', 'KRW'),
        'KRX:000004': Instrument('KRX:000004', AssetKind.STOCK, 'KRX', '000004', 'KRW'),
        'KRX:000005': Instrument('KRX:000005', AssetKind.STOCK, 'KRX', '000005', 'KRW'),
        'KRX:000007': Instrument('KRX:000007', AssetKind.STOCK, 'KRX', '000007', 'KRW'),
    }

    # When
    survivors = tradable_score_rows(scores=rows, marks=marks, volatilities=vols, instruments=instruments)
    ranks = dense_decision_ranks(survivors)

    # Then
    assert tuple(row.instrument_id for row in survivors) == good
    assert ranks == {'KRX:000001': 1, 'KRX:000004': 2, 'KRX:000007': 3}


def test_compounding_v2_records_selection_diagnostics_for_every_selection_session() -> None:
    from datetime import datetime, timedelta

    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.core.time import KRX_TZ, SessionCalendar
    from src.engine.decision import DecisionContext
    from src.strategy.compounding_v2_strategy import (
        CompoundingV2Strategy,
        summarize_compounding_v2_selection_shortfalls,
    )
    from src.strategy.scoring import ChampionScoreRow

    # Given: only 3 tradable names on a selection-cadence session (index 20).
    sessions = tuple(datetime(2024, 1, 2, 9, tzinfo=KRX_TZ) + timedelta(days=index) for index in range(22))
    ids = tuple(f'KRX:{index:06d}' for index in range(1, 4))
    instruments = {iid: Instrument(iid, AssetKind.STOCK, 'KRX', iid[4:], 'KRW') for iid in ids}
    score_time = sessions[19].replace(hour=15, minute=30)
    scores = tuple(
        ChampionScoreRow(score_time, iid, True, 100.0 - index, index, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')
        for index, iid in enumerate(ids, start=1)
    )
    strategy = CompoundingV2Strategy(scores_by_session={score_time.date(): scores}, calendar=SessionCalendar(sessions))
    decision = sessions[20].replace(hour=15, minute=30)
    snapshot = {
        'mark_prices': dict.fromkeys(ids, 10000.0),
        'volatilities': dict.fromkeys(ids, 0.20),
        'instruments': instruments,
        'market_index_level': 2.0,
        'market_index_sma100': 1.0,
        'market_index_sma200': 1.0,
        'market_volatility': 0.15,
    }

    # When
    intents = strategy.decide(DecisionContext(decision, PortfolioSnapshot('acct', decision, 1_000_000.0, 0.0, ()), snapshot))

    # Then: no trades, but the cause is recorded rather than silent.
    assert intents == ()
    diagnostics = strategy.selection_diagnostics
    assert len(diagnostics) == 1
    assert diagnostics[0].decision_session == decision.date()
    assert diagnostics[0].eligible_rows == 3
    assert diagnostics[0].tradable_rows == 3
    assert diagnostics[0].selected_positions == 0
    assert diagnostics[0].shortfall_reason == 'below_min_positions'
    assert summarize_compounding_v2_selection_shortfalls(diagnostics) == {
        'selection_sessions': 1,
        'invested_sessions': 0,
        'no_score_rows': 0,
        'below_min_positions': 1,
    }


def test_summarize_compounding_v2_selection_shortfalls_counts_every_reason() -> None:
    from datetime import date

    from src.strategy.compounding_v2_strategy import (
        CompoundingV2SelectionDiagnostic,
        summarize_compounding_v2_selection_shortfalls,
    )

    # Given
    diagnostics = (
        CompoundingV2SelectionDiagnostic(date(2024, 1, 31), 0, 0, 0, 0, 'no_score_rows'),
        CompoundingV2SelectionDiagnostic(date(2024, 3, 4), 500, 5, 5, 0, 'below_min_positions'),
        CompoundingV2SelectionDiagnostic(date(2024, 4, 2), 500, 30, 22, 12, None),
    )

    # When
    summary = summarize_compounding_v2_selection_shortfalls(diagnostics)

    # Then
    assert summary == {
        'selection_sessions': 3,
        'invested_sessions': 1,
        'no_score_rows': 1,
        'below_min_positions': 1,
    }
    assert summarize_compounding_v2_selection_shortfalls(()) == {
        'selection_sessions': 0,
        'invested_sessions': 0,
        'no_score_rows': 0,
        'below_min_positions': 0,
    }


def test_compounding_v2_policy_freezes_selection_floor_constants() -> None:
    import pytest

    from src.strategy.compounding_v2_strategy import CompoundingV2Policy

    # Given/When
    policy = CompoundingV2Policy()

    # Then: floor is domain-derived from the concentration cap.
    assert policy.min_positions == 8
    assert policy.selection_policy_version == 'compounding-v2-selection-v2'
    assert policy.min_positions * policy.security_weight_cap >= 0.80 - 1e-12
    with pytest.raises(ValueError, match='immutable'):
        CompoundingV2Policy(min_positions=4)
    with pytest.raises(ValueError, match='immutable'):
        CompoundingV2Policy(selection_policy_version='compounding-v2-selection-v1')
