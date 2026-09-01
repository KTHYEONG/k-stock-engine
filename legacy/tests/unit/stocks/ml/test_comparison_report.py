from __future__ import annotations


def test_comparison_report_records_completed_run_without_artifact_path(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.core.instruments import AssetKind
    from legacy.stocks.ml.comparison_report import MlComparisonReport
    from legacy.stocks.ml.result_ledger import MlRunContext
    from legacy.stocks.research.artifacts import ModelArtifactRegistry
    from legacy.stocks.research.models import DeterministicBaseline, ModelManifest

    manifest = ModelManifest(artifact_id="run-001", asset_kind=AssetKind.STOCK, feature_set="stock_net_alpha_v1", feature_schema_hash="schema", universe_policy_hash="universe", label_definition="net_alpha_o2o", label_horizon_sessions=10, eligible_from="2024-01-01T00:00:00+00:00", eligible_to="2024-12-31T00:00:00+00:00", model_type="no_trade")
    registry = ModelArtifactRegistry.in_memory()
    registry.publish(DeterministicBaseline(manifest=manifest), manifest)
    registry.write_metrics("run-001", {"promoted": False, "promotion_reasons": ["base-lower-cagr"]})
    context = MlRunContext(artifact_id="run-001", snapshot_id="snapshot", started_at=datetime(2024, 1, 1, tzinfo=UTC), request=None, feature_rows=1, instrument_count=1, session_count=1, feature_column_count=1, feature_session_range=None, label_definition="net_alpha_o2o", label_horizon_sessions=10, feature_schema_hash="schema", universe_policy_hash="universe")

    MlComparisonReport(tmp_path).record_completed(context, manifest, registry)

    report = (tmp_path / "ml-cmp.md").read_text(encoding="utf-8")
    assert "run-001" in report
    assert "base-lower-cagr" in report
    assert "data/artifacts" not in report


def test_comparison_report_labels_observed_wealth_as_uncertified(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.core.instruments import AssetKind
    from legacy.stocks.ml.comparison_report import MlComparisonReport
    from legacy.stocks.ml.result_ledger import MlRunContext
    from legacy.stocks.research.artifacts import ModelArtifactRegistry
    from legacy.stocks.research.models import DeterministicBaseline, ModelManifest

    manifest = ModelManifest(artifact_id='wealth-run', asset_kind=AssetKind.STOCK, feature_set='stock_net_alpha_v1', feature_schema_hash='schema', universe_policy_hash='universe', label_definition='net_alpha_o2o', label_horizon_sessions=10, eligible_from='2024-01-01T00:00:00+00:00', eligible_to='2024-12-31T00:00:00+00:00', model_type='no_trade')
    registry = ModelArtifactRegistry.in_memory()
    registry.publish(DeterministicBaseline(manifest=manifest), manifest)
    registry.write_metrics('wealth-run', {'promoted': False, 'promotion_reasons': ['non-positive-base-lower-cagr'], 'growth_route': {'base_lower_cagr': -0.01, 'stress_lower_cagr': 0.01, 'mdd': 0.10, 'wealth_evidence': {'initial_cash_krw': 10_000_000.0, 'base_terminal_wealth_krw': 10_250_000.0, 'stress_terminal_wealth_krw': 9_900_000.0, 'base_observed_return': 0.025, 'stress_observed_return': -0.01, 'observed_base_growth_positive': True}}})
    context = MlRunContext(artifact_id='wealth-run', snapshot_id='snapshot', started_at=datetime(2024, 1, 1, tzinfo=UTC), request=None, feature_rows=1, instrument_count=1, session_count=1, feature_column_count=1, feature_session_range=None, label_definition='net_alpha_o2o', label_horizon_sessions=10, feature_schema_hash='schema', universe_policy_hash='universe')

    MlComparisonReport(tmp_path).record_completed(context, manifest, registry)

    report = (tmp_path / 'ml-cmp.md').read_text(encoding='utf-8')
    assert '10,250,000.000' in report
    assert '관측 성장(인증 아님)' in report
    assert 'non-positive-base-lower-cagr' in report
