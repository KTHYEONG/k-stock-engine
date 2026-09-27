"""LS OpenAPI investor-flow collector returning one raw t1702 answer."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.integrations.errors import ProviderError
from src.integrations.ls.client import LsClient, LsCredentials
from src.integrations.responses import RawResponse

__all__ = ["LsInvestorFlowCollector"]

_FLOW_FIELDS = tuple(f"tjj{i:04d}" for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18))


def _is_zero_flow_placeholder(row: dict[str, Any]) -> bool:
    """True when every investor-flow field of a raw t1702 row is zero."""
    return all(row.get(field) in (0, "0", None, "") for field in _FLOW_FIELDS)


def _has_placeholder_date(row: dict[str, Any]) -> bool:
    raw = str(row.get("date") or row.get("session") or "").strip()
    if not raw:
        return False
    text = raw.replace("/", "-")
    if len(text) == 8 and text.isdigit():
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return True
    return False


class LsInvestorFlowCollector:
    """Fetch one raw LS t1702 answer without mapping, filtering or storage."""

    def __init__(self, symbols: tuple[str, ...] | None = None, *, client: Any | None = None) -> None:
        if symbols is not None:
            cleaned = tuple(dict.fromkeys(symbol.strip() for symbol in symbols if symbol.strip()))
            if not cleaned:
                raise ValueError("LS investor flow requires at least one symbol")
            self._symbols: tuple[str, ...] | None = cleaned
        else:
            self._symbols = None
        self._client = client or LsClient(LsCredentials.from_env())

    def health_check(self) -> None:
        """Confirm collection credentials are live without touching Bronze."""
        health = getattr(self._client, "health_check", None)
        if callable(health):
            health()

    def fetch(self, symbol: str, start: date, end: date) -> RawResponse:
        """Return one raw t1702 answer for ``symbol`` over ``[start, end]``.

        Only all-zero rows with a placeholder date are dropped. An answer with
        no rows returns ``rows == ()`` and is never an exception.

        Raises:
            ProviderError: the transport failed.
        """
        if not symbol.strip():
            raise ValueError("LS investor flow requires a symbol")
        if start > end:
            raise ValueError("coverage_start must not be after coverage_end")
        try:
            raw_rows = self._client.inquire_investor_trend(symbol.strip(), start, end)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"LS t1702 transport failed for {symbol.strip()}", provider="LS") from exc
        rows = tuple(
            dict(row)
            for row in raw_rows
            if isinstance(row, dict) and not (_has_placeholder_date(row) and _is_zero_flow_placeholder(row))
        )
        return RawResponse(
            query={"symbol": symbol.strip(), "start": start.isoformat(), "end": end.isoformat()},
            rows=rows,
        )
