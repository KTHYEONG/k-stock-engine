"""Active data wiring test."""
from legacy.stocks.data.active import ActiveResearchDataSelection
from legacy.stocks.ml.contracts import ExecutableOverlayData
from legacy.stocks.data.hedge import load_executable_overlay_data

def test_active_wiring() -> None:
    assert ActiveResearchDataSelection is not None
    assert ExecutableOverlayData is not None
    assert load_executable_overlay_data is not None
