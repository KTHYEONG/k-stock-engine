from src.data.pipeline import _require_certified_inputs
from src.data.schemas import PITDataError
import pytest


def test_pipeline_requires_silver_inputs(tmp_path) -> None:
    with pytest.raises(PITDataError, match="investor_flow"):
        _require_certified_inputs(tmp_path / "silver", tmp_path / "bronze")


def test_materialization_builds_all_available_sessions() -> None:
    from src.data.pipeline import materialize_backtest_inputs

    assert callable(materialize_backtest_inputs)
