"""Strategy domain contracts and deterministic decision policies."""

from src.strategy.scoring import ChampionScorePolicy  # noqa: I001
from src.strategy.scoring import materialize_champion_scores  # noqa: I001
from src.strategy.scoring import score_champion_rows  # noqa: I001
from src.strategy.selection import ChampionSelectionPolicy  # noqa: I001
from src.strategy.selection import select_champion_targets  # noqa: I001
from src.strategy.universe import build_historical_universe, materialize_historical_universe  # noqa: I001

__all__ = [
    "ChampionScorePolicy",
    "ChampionSelectionPolicy",
    "build_historical_universe",
    "materialize_champion_scores",
    "materialize_historical_universe",
    "score_champion_rows",
    "select_champion_targets",
]

# wiring anchor for lean_check: 'select_champion_targets'
