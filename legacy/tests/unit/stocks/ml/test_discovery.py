from __future__ import annotations

from legacy.stocks.ml.discovery import HorizonDiscovery


def test_horizon_discovery_defaults_are_bounded() -> None:
    discovery = HorizonDiscovery((), (), {})

    assert discovery.path_evaluation_count == 0
    assert discovery.path_evaluation_bound == 0
