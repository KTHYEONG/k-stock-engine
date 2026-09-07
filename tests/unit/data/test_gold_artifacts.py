from datetime import UTC, datetime

import pytest

from src.data.gold_artifacts import resolve_gold_artifact_bundle
from src.data.schemas import PITDataError


def test_resolve_gold_artifact_bundle_rejects_missing_selected_component(tmp_path) -> None:
    selected = 'selected-2016'
    (tmp_path / 'universe' / selected).mkdir(parents=True)
    (tmp_path / 'qvef' / selected).mkdir(parents=True)
    (tmp_path / 'champion_scores' / 'other-2017').mkdir(parents=True)

    with pytest.raises(PITDataError, match='champion_scores/selected-2016'):
        resolve_gold_artifact_bundle(
            gold_root=tmp_path,
            dataset_id=selected,
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
        )


def test_resolve_gold_artifact_bundle_rejects_path_traversal(tmp_path) -> None:
    with pytest.raises(PITDataError, match='dataset id'):
        resolve_gold_artifact_bundle(
            gold_root=tmp_path,
            dataset_id='../2017',
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
        )


def test_gold_artifact_bundle_loads_three_verified_selected_components(tmp_path) -> None:
    import polars as pl

    from src.core.datasets import DatasetCertification, HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    from src.data.gold_artifacts import load_gold_artifact_frames
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    dataset_id = 'gold-2016-verified'
    components = (
        ('universe', 'stock_historical_eligible_universe_v1', 1.0),
        ('qvef', 'stock_champion_qvef_v1', 2.0),
        ('champion_scores', 'stock_champion_scores_v1', 3.0),
    )
    for kind, feature_set, value in components:
        frame = pl.DataFrame({
            'decision_session': [decision_time],
            'instrument_id': ['KRX:005930'],
            'fixture_value': [value],
        })
        manifest = make_manifest(
            asset_kind=AssetKind.STOCK,
            columns=frame.columns,
            feature_set=feature_set,
            label_definition='none',
            label_horizon_sessions=1,
            time_start=decision_time,
            time_end=decision_time,
            provider_version='fixture',
            universe_policy_version='fixture-policy',
            row_count=frame.height,
            generated_time=decision_time,
            certification=DatasetCertification.RESEARCH,
            schema_version='v2',
            content_hash=canonical_content_hash(frame, frame.columns),
            storage_layout=HIVE_PARTITION_LAYOUT,
        )
        ParquetDatasetStore(tmp_path / kind).write_partitioned(
            frame,
            dataset_id=dataset_id,
            manifest=manifest,
            expected_feature_set=feature_set,
            decision_time=decision_time,
        )

    bundle = resolve_gold_artifact_bundle(
        gold_root=tmp_path,
        dataset_id=dataset_id,
        decision_time=decision_time,
    )
    universe, qvef, scores = load_gold_artifact_frames(
        bundle=bundle,
        decision_time=decision_time,
    )

    assert bundle.dataset_id == dataset_id
    assert (universe.item(0, 'fixture_value'), qvef.item(0, 'fixture_value'), scores.item(0, 'fixture_value')) == (1.0, 2.0, 3.0)
