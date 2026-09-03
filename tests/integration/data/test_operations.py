def test_prepare_rebuild_migrates_only_six_seeds_before_collection(tmp_path) -> None:
    from datetime import UTC, date, datetime
    import pytest

    from src.data.operations import StockDataRebuildRequest, prepare_stock_data_rebuild

    request = StockDataRebuildRequest(data_root=tmp_path / 'data', bronze_root=tmp_path / 'data' / 'bronze', silver_root=tmp_path / 'data' / 'silver', gold_root=tmp_path / 'data' / 'gold', artifact_root=tmp_path / 'data' / 'artifacts', coverage_start=date(2024, 1, 2), coverage_end=date(2024, 1, 2), decision_time=datetime(2024, 1, 3, tzinfo=UTC))
    (request.data_root / 'canonical').mkdir(parents=True)

    with pytest.raises(ValueError, match="official KRX and DART collectors"):
        prepare_stock_data_rebuild(request, krx=None, dart=None)
    assert (request.data_root / 'canonical').exists()


def test_purge_refuses_before_silver_gold_and_backtest_proof(tmp_path) -> None:
    from datetime import UTC, date, datetime

    import pytest

    from src.data.operations import StockDataRebuildRequest, RebuildPreparation, execute_verified_legacy_purge
    from src.data.legacy_inventory import MigrationArtifact

    request = StockDataRebuildRequest(data_root=tmp_path / 'data', bronze_root=tmp_path / 'data' / 'bronze', silver_root=tmp_path / 'data' / 'silver', gold_root=tmp_path / 'data' / 'gold', artifact_root=tmp_path / 'data' / 'artifacts', coverage_start=date(2024, 1, 2), coverage_end=date(2024, 1, 2), decision_time=datetime(2024, 1, 3, tzinfo=UTC))
    (request.data_root / 'canonical').mkdir(parents=True)
    preparation = RebuildPreparation(migration=MigrationArtifact.empty_verified(request.data_root), silver_report=None, gold_artifact=None, backtest_artifact_path=None)

    with pytest.raises(ValueError, match='Silver'):
        execute_verified_legacy_purge(request, preparation, confirm_purge=True)
    assert (request.data_root / 'canonical').exists()
