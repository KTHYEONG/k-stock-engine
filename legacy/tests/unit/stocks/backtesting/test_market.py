from __future__ import annotations

from legacy.stocks.backtesting.engine import PreparedReplayMarket as EngineMarket
from legacy.stocks.backtesting.market import PreparedReplayMarket


def test_prepared_market_is_the_engine_market_type() -> None:
    assert issubclass(PreparedReplayMarket, EngineMarket)
