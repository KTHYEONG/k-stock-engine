"""KIS industry-classification evidence mapped without inferred values."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.pit import PITDataError
from src.integrations.errors import ProviderError
from src.integrations.kis.client import KisClient, KisCredentials


class KisIndustryCollector:
    """Collect per-ticker current industry classifications via KIS inquire-price.

    The mapped record's ``industry_name`` is copied verbatim from the raw
    ``bstp_kor_isnm`` field — never normalized, translated, or bucketed into
    a code taxonomy at this layer. Transport-only: pages are returned and
    nothing is persisted here; the data layer owns Bronze writes.
    """

    def __init__(self, symbols: tuple[str, ...], *, client: Any | None = None) -> None:
        cleaned = tuple(dict.fromkeys(symbol.strip() for symbol in symbols if symbol.strip()))
        if not cleaned:
            raise ValueError("KIS industry classification requires at least one symbol")
        self._symbols = cleaned
        self._client = client or KisClient(KisCredentials.from_env())

    @staticmethod
    def _map_output(symbol: str, output: dict[str, Any]) -> dict[str, object]:
        industry = output.get("bstp_kor_isnm")
        if not isinstance(industry, str) or not industry.strip():
            raise PITDataError(f"KIS industry classification missing bstp_kor_isnm for {symbol}")
        return {
            "ticker": symbol,
            "industry_name": industry,
            "market_name": str(output.get("rprs_mrkt_kor_name") or ""),
        }

    def fetch_industry_classification(
        self, *, bronze_root: Path | str | None = None, retrieved_at: datetime | None = None
    ) -> Iterable[dict[str, object]]:
        """Fetch one current snapshot per symbol and return pages without persisting."""
        moment = retrieved_at if retrieved_at is not None and retrieved_at.tzinfo is not None else datetime.now(UTC)
        for symbol in self._symbols:
            try:
                output = self._client.inquire_price(symbol)
            except ProviderError:
                raise
            except Exception as exc:
                # reason: adapter boundary — third-party client failures carry no status to classify.
                raise PITDataError(f"KIS industry classification collection failed for {symbol}") from exc
            record = self._map_output(symbol, output)
            yield {
                "provider": "KIS",
                "endpoint": "inquire-price",
                "symbol": symbol,
                "collected_at": moment.isoformat(),
                "output": dict(output),
                "records": [record],
            }


class KisStockClassificationCollector:
    """Collect per-ticker KSIC classification and listing-abolition date via KIS.

    Transport-only: pages are returned and nothing is persisted here; the data
    layer owns Bronze writes.
    """

    def __init__(self, symbols: tuple[str, ...], *, client: Any | None = None) -> None:
        cleaned = tuple(dict.fromkeys(symbol.strip() for symbol in symbols if symbol.strip()))
        if not cleaned:
            raise ValueError("KIS industry classification requires at least one symbol")
        self._symbols = cleaned
        self._client = client or KisClient(KisCredentials.from_env())

    @staticmethod
    def _map_output(symbol: str, output: dict[str, Any]) -> dict[str, object]:
        code = str(output.get("std_idst_clsf_cd") or "").strip()
        if len(code) != 6 or not code.isascii() or not code.isdigit():
            raise PITDataError(f"KIS stock classification missing valid std_idst_clsf_cd for {symbol}")
        name = str(output.get("std_idst_clsf_cd_name") or "").strip()
        abol = str(output.get("lstg_abol_dt") or "").strip()
        if not abol:
            delisted_on = ""
        elif len(abol) == 8 and abol.isdigit():
            try:
                delisted_on = datetime.strptime(abol, "%Y%m%d").date().isoformat()
            except ValueError as exc:
                raise PITDataError(f"KIS stock classification invalid lstg_abol_dt for {symbol}") from exc
        else:
            raise PITDataError(f"KIS stock classification invalid lstg_abol_dt for {symbol}")
        return {"ticker": symbol, "ksic_code": code, "ksic_name": name, "delisted_on": delisted_on}

    def fetch_stock_classification(
        self, *, bronze_root: Path | str | None = None, retrieved_at: datetime | None = None
    ) -> Iterable[dict[str, object]]:
        """Fetch one current KSIC snapshot per symbol and return pages without persisting."""
        moment = retrieved_at if retrieved_at is not None and retrieved_at.tzinfo is not None else datetime.now(UTC)
        for symbol in self._symbols:
            try:
                output = self._client.search_stock_info(symbol)
            except ProviderError:
                raise
            except Exception as exc:
                # reason: adapter boundary — third-party client failures carry no status to classify.
                raise PITDataError(f"KIS stock classification collection failed for {symbol}") from exc
            record = self._map_output(symbol, output)
            yield {
                "provider": "KIS",
                "endpoint": "search-stock-info",
                "symbol": symbol,
                "collected_at": moment.isoformat(),
                "output": dict(output),
                "records": [record],
            }
