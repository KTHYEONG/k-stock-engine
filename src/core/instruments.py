"""Canonical instrument identity contracts.

`AssetKind` is an explicit, required value on every instrument. It is never
inferred from a code, filename, or data location.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetKind(Enum):
    """The explicit asset family. Never inferred from tickers or paths."""

    STOCK = "STOCK"
    ETF = "ETF"


@dataclass(frozen=True, slots=True)
class Instrument:
    """Canonical instrument identity.

    ``instrument_id`` is canonical and unique across the system; ``symbol`` is
    the canonical exchange symbol and must never be used to infer ``asset_kind``.
    """

    instrument_id: str
    asset_kind: AssetKind
    exchange: str
    symbol: str
    currency: str

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")


@dataclass(frozen=True, slots=True)
class ProviderSymbol:
    """A provider-issued symbol that resolves into a canonical Instrument."""

    provider: str
    symbol: str


class InstrumentResolver:
    """Resolves provider symbols into canonical instruments.

    A provider map resolves provider symbols into canonical identity before
    storage. Unknown or mixed-kind mappings are rejected rather than guessed.
    """

    def __init__(self, provider_map: dict[tuple[str, str], Instrument]):
        self._provider_map = dict(provider_map)

    def resolve(self, provider: str, symbol: str) -> Instrument:
        try:
            return self._provider_map[(provider, symbol)]
        except KeyError as exc:
            raise ValueError(f"Unknown provider symbol {provider}:{symbol}") from exc
