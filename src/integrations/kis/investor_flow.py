"""KIS investor-flow evidence mapped without inferred values."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.data.schemas import PITDataError
from src.integrations.kis.client import KisClient, KisCredentials


class KisInvestorFlowCollector:
    """Collect per-ticker KIS transaction-value investor flows in bounded pages."""

    def __init__(self, symbols: tuple[str, ...], *, client: Any | None = None) -> None:
        cleaned = tuple(dict.fromkeys(symbol.strip() for symbol in symbols if symbol.strip()))
        if not cleaned:
            raise ValueError("KIS investor flow requires at least one symbol")
        self._symbols = cleaned
        self._client = client or KisClient(KisCredentials.from_env())

    @staticmethod
    def _session(value: object) -> date:
        text = str(value).strip().replace("/", "-")
        if len(text) == 8 and text.isdigit():
            text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise PITDataError("KIS investor flow has invalid session") from exc

    @staticmethod
    def _value(row: dict[str, Any], field: str) -> float:
        raw = row.get(field)
        if raw is None or str(raw).strip() == "":
            raise PITDataError(f"KIS investor flow missing {field}")
        try:
            return float(str(raw).replace(",", ""))
        except ValueError as exc:
            raise PITDataError(f"KIS investor flow has invalid {field}") from exc

    def _map_rows(self, symbol: str, rows: tuple[dict[str, Any], ...]) -> list[dict[str, object]]:
        mapped = [
            {
                "session": self._session(row.get("stck_bsop_date")).isoformat(),
                "ticker": symbol,
                "foreign_buy_value": self._value(row, "frgn_shnu_tr_pbmn"),
                "foreign_sell_value": self._value(row, "frgn_seln_tr_pbmn"),
                "foreign_net_value": self._value(row, "frgn_ntby_tr_pbmn"),
                "institution_net_value": self._value(row, "orgn_ntby_tr_pbmn"),
                "retail_net_value": self._value(row, "prsn_ntby_tr_pbmn"),
            }
            for row in rows
        ]
        if not mapped:
            raise PITDataError("KIS investor flow response is empty")
        return mapped

    def _persist_raw_page(
        self,
        symbol: str,
        anchor: date,
        raw_rows: tuple[dict[str, Any], ...],
        *,
        bronze_root: Path | str,
        retrieved_at: datetime | None,
    ) -> Path:
        from src.data.bronze import BronzeStore
        from src.data.schemas import EvidenceKind

        payload = {
            "provider": "KIS",
            "endpoint": "investor-trade-by-stock-daily",
            "symbol": symbol,
            "anchor": anchor.isoformat(),
            "query": {"symbol": symbol, "anchor": anchor.isoformat()},
            "rows": [dict(row) for row in raw_rows],
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        moment = retrieved_at if retrieved_at is not None and retrieved_at.tzinfo is not None else datetime.now(UTC)
        store = BronzeStore(Path(bronze_root))
        import contextlib
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        try:
            stored = store.import_json(tmp_path, kind=EvidenceKind.INVESTOR_FLOW, retrieved_at=moment)
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
        if stored.content_hash != digest:
            raise PITDataError("KIS Bronze receipt digest mismatch")
        return stored.payload_path

    def probe(self, symbol: str, session: date) -> dict[str, object]:
        rows = self._map_rows(symbol, self._client.inquire_investor_trade_by_stock_daily(symbol, session))
        if not any(row["session"] == session.isoformat() for row in rows):
            raise PITDataError("KIS investor flow missing requested session")
        return {"provider": "KIS", "endpoint": "investor-trade-by-stock-daily", "records": rows}

    def fetch_investor_flow(
        self,
        start: date,
        end: date,
        *,
        bronze_root: Path | str | None = None,
        retrieved_at: datetime | None = None,
    ) -> Iterable[dict[str, object]]:
        if start > end:
            raise PITDataError("coverage_start must not be after coverage_end")
        for symbol in self._symbols:
            anchor = end
            seen: set[str] = set()
            while anchor >= start:
                try:
                    raw_rows = self._client.inquire_investor_trade_by_stock_daily(symbol, anchor)
                except Exception as exc:
                    raise PITDataError("KIS investor flow collection failed") from exc
                if bronze_root is not None:
                    self._persist_raw_page(symbol, anchor, raw_rows, bronze_root=bronze_root, retrieved_at=retrieved_at)
                rows = self._map_rows(symbol, raw_rows)
                selected = [row for row in rows if start.isoformat() <= str(row["session"]) <= end.isoformat()]
                if not selected:
                    raise PITDataError(f"KIS investor flow missing requested session range for {symbol}")
                unique = [row for row in selected if str(row["session"]) not in seen]
                seen.update(str(row["session"]) for row in unique)
                if unique:
                    yield {
                        "provider": "KIS",
                        "endpoint": "investor-trade-by-stock-daily",
                        "symbol": symbol,
                        "anchor": anchor.isoformat(),
                        "records": unique,
                    }
                earliest = min(date.fromisoformat(str(row["session"])) for row in rows)
                if earliest <= start:
                    break
                anchor = earliest - timedelta(days=1)
