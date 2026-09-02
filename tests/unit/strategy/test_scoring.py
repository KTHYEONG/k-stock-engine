def test_score_champion_rows_uses_exact_equal_weights_and_ranks() -> None:
    from datetime import UTC, datetime

    from src.features.contracts import QvefFeatureRow
    from src.strategy.scoring import score_champion_rows

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    common = dict(decision_session=decision, sector='Technology', gross_profitability=None, roe=None, cfo_to_assets=None, book_to_price=None, earnings_to_price=None, operating_income_change=None, sales_growth=None, operating_margin_change=None, foreign_flow_5=None, foreign_flow_20=None, component_presence=('all_components_present',), source_available_at=(('financial_facts', decision),), policy_version='champion-v1-qvef-v1')  # noqa: C408
    rows = (
        QvefFeatureRow(instrument_id='KRX:LOW', quality_score=0.0, value_score=0.0, earnings_score=0.0, foreign_flow_score=0.0, **common),
        QvefFeatureRow(instrument_id='KRX:HIGH', quality_score=1.0, value_score=0.5, earnings_score=0.0, foreign_flow_score=-0.5, **common),
    )

    scores = {row.instrument_id: row for row in score_champion_rows(rows, decision_time=decision)}

    assert scores['KRX:HIGH'].champion_score == 0.25
    assert scores['KRX:HIGH'].rank == 1
    assert scores['KRX:LOW'].champion_score == 0.0
    assert scores['KRX:LOW'].rank == 2


def test_score_champion_rows_excludes_each_missing_factor() -> None:
    from dataclasses import replace
    from datetime import UTC, datetime

    from src.features.contracts import QvefFeatureRow
    from src.strategy.scoring import ChampionScoreReason, score_champion_rows

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    base = QvefFeatureRow(decision, 'KRX:BASE', 'Technology', None, None, None, None, None, None, None, None, None, None, 0.1, 0.2, 0.3, 0.4, ('all_components_present',), (('financial_facts', decision),), 'champion-v1-qvef-v1')
    cases = (
        ('quality_score', ChampionScoreReason.MISSING_QUALITY),
        ('value_score', ChampionScoreReason.MISSING_VALUE),
        ('earnings_score', ChampionScoreReason.MISSING_EARNINGS),
        ('foreign_flow_score', ChampionScoreReason.MISSING_FOREIGN_FLOW),
    )
    rows = tuple(replace(base, instrument_id=f'KRX:{index}', **{field: None}) for index, (field, _reason) in enumerate(cases))

    scores = {row.instrument_id: row for row in score_champion_rows(rows, decision_time=decision)}

    for index, (_field, reason) in enumerate(cases):
        row = scores[f'KRX:{index}']
        assert row.eligible is False
        assert row.champion_score is None
        assert row.rank is None
        assert row.exclusion_reasons == (reason,)


def test_score_champion_rows_orders_ties_by_canonical_instrument_id_independent_of_input_order() -> None:
    from datetime import UTC, datetime

    from src.features.contracts import QvefFeatureRow
    from src.strategy.scoring import score_champion_rows

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    common = dict(decision_session=decision, sector='Technology', gross_profitability=None, roe=None, cfo_to_assets=None, book_to_price=None, earnings_to_price=None, operating_income_change=None, sales_growth=None, operating_margin_change=None, foreign_flow_5=None, foreign_flow_20=None, quality_score=0.5, value_score=0.5, earnings_score=0.5, foreign_flow_score=0.5, component_presence=('all_components_present',), source_available_at=(('financial_facts', decision),), policy_version='champion-v1-qvef-v1')  # noqa: C408
    alpha = QvefFeatureRow(instrument_id='KRX:ALPHA', **common)
    beta = QvefFeatureRow(instrument_id='KRX:BETA', **common)

    first = score_champion_rows((beta, alpha), decision_time=decision)
    second = score_champion_rows((alpha, beta), decision_time=decision)

    assert first == second
    assert tuple((row.instrument_id, row.rank) for row in first) == (('KRX:ALPHA', 1), ('KRX:BETA', 2))


def test_score_champion_rows_rejects_future_or_wrong_version_feature_rows() -> None:
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.features.contracts import QvefFeatureRow
    from src.strategy.scoring import score_champion_rows

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    row = QvefFeatureRow(decision, 'KRX:SAFE', 'Technology', None, None, None, None, None, None, None, None, None, None, 0.1, 0.2, 0.3, 0.4, ('all_components_present',), (('financial_facts', decision),), 'champion-v1-qvef-v1')

    with pytest.raises(ValueError, match='available'):
        score_champion_rows((replace(row, source_available_at=(('financial_facts', decision + timedelta(seconds=1)),)),), decision_time=decision)
    with pytest.raises(ValueError, match='policy'):
        score_champion_rows((replace(row, policy_version='other-strategy-v1'),), decision_time=decision)
