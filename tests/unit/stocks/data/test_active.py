"""Active data wiring test."""
from src.stocks.data.active import ActiveResearchDataSelection
from src.stocks.ml.contracts import ExecutableOverlayData
from src.stocks.data.hedge import load_executable_overlay_data

def test_active_wiring() -> None:
    assert ActiveResearchDataSelection is not None
    assert ExecutableOverlayData is not None
    assert load_executable_overlay_data is not None
