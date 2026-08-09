"""Versioned, validated ETF configuration (IndexSwitchV1 artifacts)."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.etfs.optimization.search import SearchSpace
from src.etfs.strategies.index_switch_v1 import IndexSwitchParams


@dataclass(frozen=True, slots=True)
class EtfStrategyConfig:
    """Configuration artifact binding strategy params + search space."""

    strategy_name: str = "IndexSwitchV1"
    params: IndexSwitchParams = field(default_factory=IndexSwitchParams)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    version: str = "v1"


DEFAULT_ETF_STRATEGY = EtfStrategyConfig()