"""LS OpenAPI investor-flow collector supporting multi-year queries via t1702."""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.core.pit import BronzeReceipt, PITDataError
from src.integrations.ls.client import LsClient, LsCredentials


class LsInvestorFlowCollector:
    """Collect per-ticker LS t1702 net share quantities by investor group in multi-year batches."""

    def __init__(self, symbols: tuple[str, ...], *, client: Any | None = None) -> None:
        cleaned = tuple(dict.fromkeys(symbol.strip() for symbol in symbols if symbol.strip()))
        if not cleaned:
            raise ValueError("LS investor flow requires at least one symbol")
        self._symbols = cleaned
        self._client = client or LsClient(LsCredentials.from_env())

    def close(self) -> None:
        """Release the retained LS client after a bounded collection run."""
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> LsInvestorFlowCollector:
        """Enter a sequential LS collection lifetime without issuing a request."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the owned LS client regardless of collection outcome."""
        self.close()

    @staticmethod
    def _session(value: object) -> date:
        text = str(value).strip().replace("/", "-")
        if len(text) == 8 and text.isdigit():
            text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise PITDataError("LS investor flow has invalid session") from exc

    @staticmethod
    def _shares(row: dict[str, Any], field: str) -> int:
        raw = row.get(field)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            raise PITDataError(f"LS investor flow missing {field}")
        if isinstance(raw, bool):
            raise PITDataError(f"LS investor flow has invalid {field}")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            if not math.isfinite(raw) or not raw.is_integer():
                raise PITDataError(f"LS investor flow has non-integral {field}")
            return int(raw)
        try:
            parsed = Decimal(str(raw).replace(",", "").strip())
        except InvalidOperation as exc:
            raise PITDataError(f"LS investor flow has invalid {field}") from exc
        if parsed != parsed.to_integral_value():
            raise PITDataError(f"LS investor flow has non-integral {field}")
        return int(parsed)

    def _map_rows(self, symbol: str, rows: tuple[dict[str, Any], ...]) -> list[dict[str, object]]:
        """Map raw t1702 rows to share-denominated net records.

        Every record carries ``unit="shares"`` and the four aggregate groups
        whose identities are verified on raw rows. Net quantities are integers
        of shares; no KRW value is derived here because the provider does not
        supply per-group amounts in this mode.

        Raises:
            PITDataError: missing/non-numeric group field, non-integral quantity,
                or a violated aggregate identity.
        """
        mapped: list[dict[str, object]] = []
        for row in rows:
            raw_sess = str(row.get("date") or row.get("session") or "").strip()
            if not raw_sess:
                continue
            sess = self._session(raw_sess).isoformat()
            individual = self._shares(row, "tjj0008")
            foreign_sub_a = self._shares(row, "tjj0009")
            foreign_sub_b = self._shares(row, "tjj0010")
            foreign = self._shares(row, "tjj0016")
            other_sub_a = self._shares(row, "tjj0007")
            other_sub_b = self._shares(row, "tjj0011")
            other = self._shares(row, "tjj0017")
            inst_parts = [self._shares(row, f"tjj000{i}") for i in range(7)]
            institution = self._shares(row, "tjj0018")
            if institution != sum(inst_parts):
                raise PITDataError("LS investor flow violates institution aggregate identity")
            if foreign != foreign_sub_a + foreign_sub_b:
                raise PITDataError("LS investor flow violates foreign aggregate identity")
            if other != other_sub_a + other_sub_b:
                raise PITDataError("LS investor flow violates other aggregate identity")
            if individual + foreign + other + institution != 0:
                raise PITDataError("LS investor flow violates zero-sum identity")
            mapped.append(
                {
                    "session": sess,
                    "ticker": symbol,
                    "_source_provider": "LS",
                    "unit": "shares",
                    "individual_net_shares": individual,
                    "foreign_net_shares": foreign,
                    "institution_net_shares": institution,
                    "other_net_shares": other,
                }
            )
        return mapped

    def _persist_raw_page(
        self,
        symbol: str,
        start: date,
        end: date,
        raw_rows: tuple[dict[str, Any], ...],
        *,
        bronze_root: Path | str,
        retrieved_at: datetime | None,
    ) -> BronzeReceipt:
        from src.core.pit import EvidenceKind
        from src.data.bronze import BronzeStore

        payload = {
            "provider": "LS",
            "endpoint": "frgr-itt",
            "symbol": symbol,
            "anchor": end.isoformat(),
            "query": {"symbol": symbol, "start": start.isoformat(), "end": end.isoformat()},
            "rows": [dict(row) for row in raw_rows],
            "records": self._map_rows(symbol, raw_rows),
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        moment = retrieved_at if retrieved_at is not None and retrieved_at.tzinfo is not None else datetime.now(UTC)
        store = BronzeStore(Path(bronze_root))
        return store.import_bytes(
            text.encode("utf-8"),
            kind=EvidenceKind.INVESTOR_FLOW,
            retrieved_at=moment,
            source_label=f"LS:frgr-itt:{symbol}:{end.isoformat()}",
        )

    def probe(self, symbol: str, session: date) -> dict[str, object]:
        rows = self._map_rows(symbol, self._client.inquire_investor_trend(symbol, session, session))
        if not any(row["session"] == session.isoformat() for row in rows):
            raise PITDataError("LS investor flow missing requested session")
        return {"provider": "LS", "endpoint": "frgr-itt", "records": rows}

    def fetch_investor_flow(
        self,
        start: date,
        end: date,
        *,
        bronze_root: Path | str | None = None,
        retrieved_at: datetime | None = None,
        symbols: tuple[str, ...] | None = None,
    ) -> Iterable[dict[str, object]]:
        if start > end:
            raise PITDataError("coverage_start must not be after coverage_end")
        requested_symbols = self._symbols if symbols is None else tuple(symbols)
        if not requested_symbols or any(symbol not in self._symbols for symbol in requested_symbols):
            raise PITDataError("LS investor flow requested symbol is outside collector universe")

        for symbol in requested_symbols:
            raw_rows = self._client.inquire_investor_trend(symbol, start, end)
            if not raw_rows:
                continue
            mapped = self._map_rows(symbol, raw_rows)
            filtered = [r for r in mapped if start.isoformat() <= str(r["session"]) <= end.isoformat()]
            if not filtered:
                continue
            if bronze_root is not None:
                self._persist_raw_page(symbol, start, end, raw_rows, bronze_root=bronze_root, retrieved_at=retrieved_at)
            yield {
                "provider": "LS",
                "endpoint": "frgr-itt",
                "symbol": symbol,
                "anchor": end.isoformat(),
                "records": filtered,
            }
