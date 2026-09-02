def test_materialize_qvef_features_writes_immutable_gold_dataset(tmp_path) -> None:
    from datetime import UTC, datetime
    import json

    from src.core.datasets import DatasetCertification
    from src.features.contracts import QvefFeaturePolicy, QvefFeatureRow
    from src.features.materialize import materialize_qvef_features

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    policy = QvefFeaturePolicy()
    row = QvefFeatureRow(decision_session=decision, instrument_id='KRX:000001', sector='Technology', gross_profitability=0.4, roe=0.4, cfo_to_assets=0.16, book_to_price=0.5, earnings_to_price=0.1, operating_income_change=0.1, sales_growth=0.2, operating_margin_change=0.05, foreign_flow_5=0.01, foreign_flow_20=0.02, quality_score=0.0, value_score=0.0, earnings_score=0.0, foreign_flow_score=0.0, component_presence=('all_components_present',), source_available_at=(('financial_facts', decision), ('daily_market', decision), ('investor_flow', decision)), policy_version=policy.version)

    path = materialize_qvef_features((row,), root=tmp_path, dataset_id='qvef-v1', decision_time=decision, policy=policy, provider_version='fixture', calendar_hash='calendar', master_hash='master', quality_report_hash='quality', certification=DatasetCertification.RESEARCH)

    manifest = json.loads((path / 'dataset_manifest.json').read_text(encoding='utf-8'))
    assert manifest['feature_set'] == 'stock_champion_qvef_v1'
    assert manifest['content_hash']
    assert manifest['universe_policy_version'] == policy.version
    assert (path / 'content_manifest.json').exists()
