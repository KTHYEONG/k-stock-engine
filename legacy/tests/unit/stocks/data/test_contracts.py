"""Shim for contracts."""
from legacy.stocks.data.contracts import DatasetFrame
def test_dataset_frame_exists() -> None:
    assert DatasetFrame is not None
