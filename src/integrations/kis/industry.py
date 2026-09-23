"""KIS industry-classification evidence mapped without inferred values."""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.pit import BronzeReceipt, PITDataError
from src.integrations.kis.client import KisClient, KisCredentials


class KisIndustryCollector:
    """Collect per-ticker current industry classifications via KIS inquire-price.

    The mapped record's ``industry_name`` is copied verbatim from the raw
    ``bstp_kor_isnm`` field — never normalized, translated, or bucketed into
    a code taxonomy at this layer (that belongs in the Silver builder or
    later features work, not the collector).
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
        self, *, bronze_root: Path | str, retrieved_at: datetime | None = None
    ) -> Iterable[dict[str, object]]:
        """Fetch one current snapshot per symbol and persist each to Bronze.

        Raises:
            PITDataError: ``inquire_price`` fails transiently, or its output
                is missing ``bstp_kor_isnm`` or holds an empty string for it
                (fail closed — never substitute an ``"UNKNOWN"`` placeholder
                silently). No Bronze receipt is written for a symbol whose
                output fails this check.
        """
        from src.core.pit import EvidenceKind
        from src.data.bronze import BronzeStore

        moment = retrieved_at if retrieved_at is not None and retrieved_at.tzinfo is not None else datetime.now(UTC)
        collected_date = moment.date()
        store = BronzeStore(Path(bronze_root))
        for symbol in self._symbols:
            try:
                output = self._client.inquire_price(symbol)
            except Exception as exc:
                raise PITDataError(f"KIS industry classification collection failed for {symbol}") from exc
            record = self._map_output(symbol, output)
            payload = {
                "provider": "KIS",
                "endpoint": "inquire-price",
                "symbol": symbol,
                "collected_at": moment.isoformat(),
                "output": dict(output),
                "records": [record],
            }
            receipt: BronzeReceipt = store.import_bytes(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
                kind=EvidenceKind.INDUSTRY,
                retrieved_at=moment,
                source_label=f"KIS:inquire-price:{symbol}:{collected_date.isoformat()}",
            )
            yield {
                "provider": "KIS",
                "endpoint": "inquire-price",
                "symbol": symbol,
                "collected_at": moment.isoformat(),
                "records": [record],
                "bronze_receipt_hash": receipt.content_hash,
                "bronze_receipt": receipt,
            }
