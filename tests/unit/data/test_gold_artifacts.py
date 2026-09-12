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
        # scores/qvef carry the columns load_gold_artifact_frames certifies.
        certification_columns: dict[str, list[object]] = {}
        if kind == 'champion_scores':
            certification_columns = {'eligible': [True]}
        elif kind == 'qvef':
            certification_columns = {
                'quality_score': [0.5],
                'value_score': [0.5],
                'earnings_score': [0.5],
                'foreign_flow_score': [0.5],
            }
        frame = pl.DataFrame({
            'decision_session': [decision_time],
            'instrument_id': ['KRX:005930'],
            'fixture_value': [value],
            **certification_columns,
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


def test_load_gold_universe_and_scores_requires_exact_pair(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace
    import polars as pl
    import pytest
    import src.data.gold_artifacts as gold
    from src.data.schemas import PITDataError

    dataset_id = 'dataset'
    for kind in ('universe', 'champion_scores'):
        (tmp_path / kind / dataset_id).mkdir(parents=True)
    class FakeStore:
        def __init__(self, root): self.root = root
        def read_manifest(self, dataset_id): return SimpleNamespace(content_hash=str(self.root))
        def read(self, dataset_id, asset_kind, feature_set, decision_time): return pl.DataFrame({'kind': [str(self.root)]})
    monkeypatch.setattr(gold, 'ParquetDatasetStore', FakeStore)
    monkeypatch.setattr(gold, 'validate_dataset_manifest', lambda *args: None)
    bundle = gold.resolve_gold_artifact_bundle(gold_root=tmp_path, dataset_id=dataset_id, decision_time=datetime.now(UTC), required_kinds=('universe', 'champion_scores'))
    universe, scores = gold.load_gold_universe_and_scores(bundle=bundle, decision_time=datetime.now(UTC))
    assert universe.height == scores.height == 1
    with pytest.raises(PITDataError, match='missing'):
        gold.resolve_gold_artifact_bundle(gold_root=tmp_path, dataset_id='missing', decision_time=datetime.now(UTC), required_kinds=('universe', 'champion_scores'))


def test_load_gold_artifact_frames_certifies_score_policy(monkeypatch) -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    import src.data.gold_artifacts as gold_artifacts
    from src.strategy.scoring import ChampionScoreFactorIntegrityError

    # Given: a bundle whose store returns a stale eligible row.
    session = datetime(2024, 6, 3, tzinfo=UTC)
    scores = pl.DataFrame(
        {'decision_session': [session], 'instrument_id': ['KRX:000001'], 'eligible': [True]}
    )
    qvef = pl.DataFrame(
        {
            'decision_session': [session],
            'instrument_id': ['KRX:000001'],
            'quality_score': [0.5],
            'value_score': [0.5],
            'earnings_score': [None],
            'foreign_flow_score': [None],
        }
    )
    universe = pl.DataFrame({'decision_session': [session], 'instrument_id': ['KRX:000001']})
    frames = {
        'stock_historical_eligible_universe_v1': universe,
        'stock_champion_qvef_v1': qvef,
        'stock_champion_scores_v1': scores,
    }

    class _Store:
        def __init__(self, root: object) -> None:
            del root

        def read(self, dataset_id: str, asset_kind: object, feature_set: str, decision_time: object) -> pl.DataFrame:
            del dataset_id, asset_kind, decision_time
            return frames[feature_set]

    monkeypatch.setattr(gold_artifacts, 'ParquetDatasetStore', _Store)
    bundle = gold_artifacts.GoldArtifactBundle(
        dataset_id='ds1',
        universe_path=__import__('pathlib').Path('g/universe/ds1'),
        qvef_path=__import__('pathlib').Path('g/qvef/ds1'),
        champion_scores_path=__import__('pathlib').Path('g/champion_scores/ds1'),
        universe_manifest_hash='u',
        qvef_manifest_hash='q',
        champion_scores_manifest_hash='s',
    )

    # When/Then: the loader refuses the stale artifact.
    with pytest.raises(ChampionScoreFactorIntegrityError):
        gold_artifacts.load_gold_artifact_frames(bundle=bundle, decision_time=session)
