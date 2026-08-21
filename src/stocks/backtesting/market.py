"""Prepared replay market extracted from engine.py.

``PreparedReplayMarket`` is the immutable, array-backed market state
for efficient backtesting replay.
"""
from __future__ import annotations

from src.stocks.backtesting.engine import PreparedReplayMarket as _PreparedReplayMarket


class PreparedReplayMarket(_PreparedReplayMarket):
    """Stable market contract name during decomposition."""


__all__ = ["PreparedReplayMarket"]
