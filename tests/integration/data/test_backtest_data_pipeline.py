def test_materialize_backtest_inputs_rejects_missing_champion_evidence(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.pipeline import materialize_backtest_inputs
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match=r'investor_flow.*financial_facts'):
        materialize_backtest_inputs(
            bronze_root=tmp_path / 'bronze',
            silver_root=tmp_path / 'silver',
            gold_root=tmp_path / 'gold',
            decision_time=datetime(2024, 1, 3, tzinfo=UTC),
        )

    assert not (tmp_path / 'gold').exists()


def test_materialize_rejects_missing_evidence_without_fixture_gold(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.core.datasets import DatasetCertification
    from src.data.pipeline import materialize_backtest_inputs
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match=r'investor_flow.*financial_facts'):
        materialize_backtest_inputs(bronze_root=tmp_path / 'bronze', silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold', artifact_root=tmp_path / 'artifacts', decision_time=datetime(2024, 1, 3, tzinfo=UTC), certification=DatasetCertification.RESEARCH)
    assert not (tmp_path / 'gold').exists()
