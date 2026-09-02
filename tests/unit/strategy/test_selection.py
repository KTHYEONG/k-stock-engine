def test_select_champion_targets_applies_entry_and_retention_boundaries() -> None:
    from datetime import UTC, datetime

    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position
    from src.strategy.scoring import ChampionScoreRow
    from src.strategy.selection import SelectionReason, select_champion_targets

    now = datetime(2024, 1, 3, tzinfo=UTC)
    def score(instrument_id: str, rank: int) -> ChampionScoreRow:
        return ChampionScoreRow(now, instrument_id, True, 100.0 - rank, rank, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')

    held_40 = Instrument('KRX:HELD40', AssetKind.STOCK, 'KRX', 'HELD40', 'KRW')
    held_41 = Instrument('KRX:HELD41', AssetKind.STOCK, 'KRX', 'HELD41', 'KRW')
    portfolio = PortfolioSnapshot('snapshot-1', now, 0.0, 0.0, (Position(held_41, 1, 1.0), Position(held_40, 1, 1.0)))

    result = select_champion_targets((score('KRX:ENTRY20', 20), score('KRX:ENTRY21', 21), score('KRX:HELD40', 40), score('KRX:HELD41', 41)), portfolio, decision_time=now)

    assert result.selected_instrument_ids == ('KRX:ENTRY20', 'KRX:HELD40')
    assert [(d.instrument_id, d.reason) for d in result.decisions if not d.selected] == [('KRX:HELD41', SelectionReason.EXIT_RETENTION_RANK)]
    assert result.unfilled_slots == 18


def test_select_champion_targets_prioritizes_survivors_over_new_entries() -> None:
    from datetime import UTC, datetime

    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position
    from src.strategy.scoring import ChampionScoreRow
    from src.strategy.selection import SelectionReason, select_champion_targets

    now = datetime(2024, 1, 3, tzinfo=UTC)
    def score(instrument_id: str, rank: int) -> ChampionScoreRow:
        return ChampionScoreRow(now, instrument_id, True, 100.0 - rank, rank, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')

    held_21 = Instrument('KRX:HELD21', AssetKind.STOCK, 'KRX', 'HELD21', 'KRW')
    held_40 = Instrument('KRX:HELD40', AssetKind.STOCK, 'KRX', 'HELD40', 'KRW')
    portfolio = PortfolioSnapshot('snapshot-2', now, 0.0, 0.0, (Position(held_40, 1, 1.0), Position(held_21, 1, 1.0)))

    result = select_champion_targets((score('KRX:NEW01', 1), score('KRX:HELD21', 21), score('KRX:HELD40', 40)), portfolio, decision_time=now)

    assert result.selected_instrument_ids == ('KRX:NEW01', 'KRX:HELD21', 'KRX:HELD40')
    assert {d.instrument_id: d.reason for d in result.decisions if d.selected} == {'KRX:NEW01': SelectionReason.NEW_ENTRY, 'KRX:HELD21': SelectionReason.SURVIVOR, 'KRX:HELD40': SelectionReason.SURVIVOR}


def test_select_champion_targets_forces_ineligible_and_missing_held_scores_to_exit() -> None:
    from datetime import UTC, datetime

    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position
    from src.strategy.scoring import ChampionScoreReason, ChampionScoreRow
    from src.strategy.selection import SelectionReason, select_champion_targets

    now = datetime(2024, 1, 3, tzinfo=UTC)
    blocked = Instrument('KRX:BLOCKED', AssetKind.STOCK, 'KRX', 'BLOCKED', 'KRW')
    absent = Instrument('KRX:ABSENT', AssetKind.STOCK, 'KRX', 'ABSENT', 'KRW')
    portfolio = PortfolioSnapshot('snapshot-3', now, 0.0, 0.0, (Position(blocked, 1, 1.0), Position(absent, 1, 1.0)))
    score = ChampionScoreRow(now, 'KRX:BLOCKED', False, None, None, (ChampionScoreReason.MISSING_VALUE,), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')

    result = select_champion_targets((score,), portfolio, decision_time=now)

    assert result.selected_instrument_ids == ()
    assert [(d.instrument_id, d.reason) for d in result.decisions] == [('KRX:ABSENT', SelectionReason.EXIT_MISSING_SCORE), ('KRX:BLOCKED', SelectionReason.EXIT_INELIGIBLE)]
    assert result.unfilled_slots == 20


def test_select_champion_targets_leaves_insufficient_coverage_as_cash_slots() -> None:
    from datetime import UTC, datetime

    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.scoring import ChampionScoreRow
    from src.strategy.selection import select_champion_targets

    now = datetime(2024, 1, 3, tzinfo=UTC)
    scores = (
        ChampionScoreRow(now, 'KRX:ONE', True, 1.0, 1, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1'),
        ChampionScoreRow(now, 'KRX:TWO', True, 0.5, 2, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1'),
    )
    portfolio = PortfolioSnapshot('snapshot-4', now, 0.0, 0.0, ())

    result = select_champion_targets(scores, portfolio, decision_time=now)

    assert result.selected_instrument_ids == ('KRX:ONE', 'KRX:TWO')
    assert result.unfilled_slots == 18
    assert len(result.decisions) == 2


def test_select_champion_targets_is_input_order_invariant_for_ranked_ties() -> None:
    from datetime import UTC, datetime

    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.scoring import ChampionScoreRow
    from src.strategy.selection import select_champion_targets

    now = datetime(2024, 1, 3, tzinfo=UTC)
    alpha = ChampionScoreRow(now, 'KRX:ALPHA', True, 0.5, 1, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')
    beta = ChampionScoreRow(now, 'KRX:BETA', True, 0.5, 2, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')
    portfolio = PortfolioSnapshot('snapshot-5', now, 0.0, 0.0, ())

    first = select_champion_targets((beta, alpha), portfolio, decision_time=now)
    second = select_champion_targets((alpha, beta), portfolio, decision_time=now)

    assert first == second
    assert first.selected_instrument_ids == ('KRX:ALPHA', 'KRX:BETA')


def test_select_champion_targets_exits_lowest_ranked_survivor_over_capacity() -> None:
    from datetime import UTC, datetime

    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position
    from src.strategy.scoring import ChampionScoreRow
    from src.strategy.selection import SelectionReason, select_champion_targets

    now = datetime(2024, 1, 3, tzinfo=UTC)
    scores = tuple(ChampionScoreRow(now, f'KRX:{rank:02d}', True, 100.0 - rank, rank, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1') for rank in range(1, 22))
    positions = tuple(Position(Instrument(f'KRX:{rank:02d}', AssetKind.STOCK, 'KRX', f'{rank:02d}', 'KRW'), 1, 1.0) for rank in reversed(range(1, 22)))
    portfolio = PortfolioSnapshot('snapshot-capacity', now, 0.0, 0.0, positions)

    result = select_champion_targets(scores, portfolio, decision_time=now)

    assert len(result.selected_instrument_ids) == 20
    assert result.selected_instrument_ids[0] == 'KRX:01'
    assert result.selected_instrument_ids[-1] == 'KRX:20'
    assert [(d.instrument_id, d.reason) for d in result.decisions if not d.selected] == [('KRX:21', SelectionReason.EXIT_CAPACITY)]


def test_select_champion_targets_rejects_wrong_policy_and_future_score_rows() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.scoring import ChampionScoreRow
    from src.strategy.selection import select_champion_targets

    now = datetime(2024, 1, 3, tzinfo=UTC)
    portfolio = PortfolioSnapshot('snapshot-6', now, 0.0, 0.0, ())
    wrong_policy = ChampionScoreRow(now, 'KRX:ONE', True, 1.0, 1, (), 'champion-v1-qvef-v1', 'other-v1')
    future = ChampionScoreRow(now + timedelta(seconds=1), 'KRX:ONE', True, 1.0, 1, (), 'champion-v1-qvef-v1', 'champion-v1-scoring-v1')

    with pytest.raises(ValueError, match='policy'):
        select_champion_targets((wrong_policy,), portfolio, decision_time=now)
    with pytest.raises(ValueError, match='decision_session'):
        select_champion_targets((future,), portfolio, decision_time=now)
