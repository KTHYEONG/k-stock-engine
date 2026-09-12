"""Strategy domain contracts and deterministic decision policies.

The public names remain re-exported lazily so importing a lightweight strategy
does not eagerly initialize the feature/NumPy dependency tree.
"""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "ChampionPortfolioPolicy": "src.strategy.portfolio",
    "ChampionScorePolicy": "src.strategy.scoring",
    "ChampionSelectionPolicy": "src.strategy.selection",
    "ChampionStrategy": "src.strategy.champion_strategy",
    "PortfolioSecurityInput": "src.strategy.portfolio",
    "build_champion_portfolio": "src.strategy.pipeline",
    "build_historical_universe": "src.strategy.universe",
    "construct_champion_portfolio": "src.strategy.portfolio",
    "materialize_champion_scores": "src.strategy.scoring",
    "materialize_historical_universe": "src.strategy.universe",
    "score_champion_rows": "src.strategy.scoring",
    "select_champion_targets": "src.strategy.selection",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "ChampionPortfolioPolicy",
    "ChampionScorePolicy",
    "ChampionSelectionPolicy",
    "ChampionStrategy",
    "PortfolioSecurityInput",
    "build_champion_portfolio",
    "build_historical_universe",
    "construct_champion_portfolio",
    "materialize_champion_scores",
    "materialize_historical_universe",
    "score_champion_rows",
    "select_champion_targets",
]

# wiring anchor for lean_check: 'select_champion_targets'
