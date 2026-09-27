"""KIS investor-flow collector returning one raw FHPTJ04160001 answer."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.integrations.errors import ProviderError
from src.integrations.kis.client import KisClient, KisCredentials
from src.integrations.responses import RawResponse

__all__ = ["KisInvestorFlowCollector"]


class KisInvestorFlowCollector:
    """Fetch one raw KIS investor-trade page without mapping or storage."""

    def __init__(self, symbols: tuple[str, ...] | None = None, *, client: Any | None = None) -> None:
        if symbols is not None:
            cleaned = tuple(dict.fromkeys(symbol.strip() for symbol in symbols if symbol.strip()))
            if not cleaned:
                raise ValueError("KIS investor flow requires at least one symbol")
            self._symbols: tuple[str, ...] | None = cleaned
        else:
            self._symbols = None
        self._client = client or KisClient(KisCredentials.from_env())

    def health_check(self) -> None:
        """Confirm collection credentials are live without touching Bronze."""
        health = getattr(self._client, "health_check", None)
        if callable(health):
            health()

    def fetch(self, symbol: str, anchor: date) -> RawResponse:
        """Return one raw ``FHPTJ04160001`` answer anchored at ``anchor``.

        An answer with no rows returns ``rows == ()`` and is never an
        exception.

        Raises:
            ProviderError: the transport failed.
        """
        if not symbol.strip():
            raise ValueError("KIS investor flow requires a symbol")
        if not isinstance(anchor, date):
            raise ValueError("anchor must be a date")
        try:
            raw_rows = self._client.inquire_investor_trade_by_stock_daily(symbol.strip(), anchor)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"KIS investor flow transport failed for {symbol.strip()}", provider="KIS"
            ) from exc
        return RawResponse(
            query={"symbol": symbol.strip(), "anchor": anchor.isoformat()},
            rows=tuple(dict(row) for row in raw_rows if isinstance(row, dict)),
        )
