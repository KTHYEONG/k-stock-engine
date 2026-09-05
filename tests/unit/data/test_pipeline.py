from src.data.pipeline import _require_certified_inputs
from src.data.schemas import PITDataError
import pytest


def test_pipeline_requires_silver_inputs(tmp_path) -> None:
    with pytest.raises(PITDataError, match="investor_flow"):
        _require_certified_inputs(tmp_path / "silver", tmp_path / "bronze")


def test_materialization_builds_all_available_sessions() -> None:
    from src.data.pipeline import materialize_backtest_inputs

    assert callable(materialize_backtest_inputs)


def test_load_silver_tables_selects_latest_manifest_per_table(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import polars as pl

    import src.data.pipeline as pipeline
    from src.data.schemas import SilverTable

    seen: list[tuple[object, str, object]] = []
    decision_time = datetime(2016, 12, 30, tzinfo=UTC)

    def fake_load(*, root, table, decision_time):
        seen.append((root, table.value, decision_time))
        return pl.DataFrame({"t": [table.value]})

    monkeypatch.setattr(pipeline, "load_latest_silver_table", fake_load)

    tables = pipeline._load_silver_tables(tmp_path, decision_time)

    assert {t.value for t in tables} == {t.value for t in SilverTable}
    assert len(seen) == len(SilverTable)
    assert {name for _, name, _ in seen} == {t.value for t in SilverTable}
    assert all(root == tmp_path and at == decision_time for root, _, at in seen)
