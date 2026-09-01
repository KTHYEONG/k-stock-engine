"""ETF universe and index-underlying relationships (AssetKind.ETF only)."""
from __future__ import annotations

from dataclasses import dataclass

from src.core.instruments import AssetKind, Instrument


@dataclass(frozen=True, slots=True)
class EtfUniverse:
    """Declared ETF/index universe per market (configuration artifact)."""

    market: str
    index_ticker: str
    bull_1x: str
    bear_1x: str
    bull_2x: str | None = None
    bear_2x: str | None = None

    def instruments(self, exchange: str = "KRX", currency: str = "KRW") -> list[Instrument]:
        symbols = [self.bull_1x, self.bear_1x, self.bull_2x, self.bear_2x]
        return [
            Instrument(
                instrument_id=f"{exchange}:{symbol}",
                asset_kind=AssetKind.ETF,
                exchange=exchange,
                symbol=symbol,
                currency=currency,
            )
            for symbol in symbols
            if symbol
        ]


KOSPI_ETF_UNIVERSE = EtfUniverse(
    market="KOSPI",
    index_ticker="코스피 200",
    bull_1x="069500",
    bull_2x="122630",
    bear_1x="114800",
    bear_2x="252670",
)

KOSDAQ_ETF_UNIVERSE = EtfUniverse(
    market="KOSDAQ",
    index_ticker="코스닥 150",
    bull_1x="229200",
    bull_2x="233740",
    bear_1x="251340",
    bear_2x="252710",
)
