def test_materialize_champion_scores_writes_immutable_reason_coded_gold_dataset(tmp_path) -> None:
    from datetime import UTC, datetime
    import json

    import pytest

    from src.core.datasets import DatasetCertification
    from src.features.contracts import QvefFeatureRow
    from src.strategy.scoring import ChampionScorePolicy, materialize_champion_scores, score_champion_rows

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    feature = QvefFeatureRow(decision, 'KRX:ONE', 'Technology', None, None, None, None, None, None, None, None, None, None, 0.1, 0.2, 0.3, 0.4, ('all_components_present',), (('financial_facts', decision),), 'champion-v1-qvef-v1')
    scores = score_champion_rows((feature,), decision_time=decision)
    kwargs = dict(root=tmp_path, dataset_id='champion-scores-v1', decision_time=decision, policy=ChampionScorePolicy(), provider_version='fixture', calendar_hash='calendar', master_hash='master', quality_report_hash='quality', certification=DatasetCertification.RESEARCH)  # noqa: C408

    path = materialize_champion_scores(scores, **kwargs)
    manifest = json.loads((path / 'dataset_manifest.json').read_text(encoding='utf-8'))

    assert manifest['feature_set'] == 'stock_champion_scores_v1'
    assert manifest['content_hash']
    assert (path / 'content_manifest.json').exists()
    with pytest.raises(ValueError, match='already exists'):
        materialize_champion_scores(scores, **kwargs)


def test_materialize_champion_scores_rejects_inconsistent_eligible_ranks(tmp_path) -> None:
    from dataclasses import replace
    from datetime import UTC, datetime

    import pytest

    from src.core.datasets import DatasetCertification
    from src.features.contracts import QvefFeatureRow
    from src.strategy.scoring import ChampionScorePolicy, materialize_champion_scores, score_champion_rows

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    common = dict(decision_session=decision, sector='Technology', gross_profitability=None, roe=None, cfo_to_assets=None, book_to_price=None, earnings_to_price=None, operating_income_change=None, sales_growth=None, operating_margin_change=None, foreign_flow_5=None, foreign_flow_20=None, quality_score=0.5, value_score=0.5, earnings_score=0.5, foreign_flow_score=0.5, component_presence=('all_components_present',), source_available_at=(('financial_facts', decision),), policy_version='champion-v1-qvef-v1')  # noqa: C408
    features = (QvefFeatureRow(instrument_id='KRX:ALPHA', **common), QvefFeatureRow(instrument_id='KRX:BETA', **common))
    scores = score_champion_rows(features, decision_time=decision)
    invalid = (scores[0], replace(scores[1], rank=scores[0].rank))

    with pytest.raises(ValueError, match='unique and consecutive'):
        materialize_champion_scores(invalid, root=tmp_path, dataset_id='invalid-ranks', decision_time=decision, policy=ChampionScorePolicy(), provider_version='fixture', calendar_hash='calendar', master_hash='master', quality_report_hash='quality', certification=DatasetCertification.RESEARCH)
