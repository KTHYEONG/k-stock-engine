"""Kiwoom REST API investor-flow collector supporting 100-session batches via ka10059."""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.data.schemas import BronzeReceipt, PITDataError
from src.integrations.kiwoom.client import KiwoomClient, KiwoomCredentials


class KiwoomInvestorFlowCollector:
    """Collect per-ticker Kiwoom transaction-value investor flows in 100-session batches."""

    def __init__(self, symbols: tuple[str, ...], *, client: Any | None = None) -> None:
        cleaned = tuple(dict.fromkeys(symbol.strip() for symbol in symbols if symbol.strip()))
        if not cleaned:
            raise ValueError("Kiwoom investor flow requires at least one symbol")
        self._symbols = cleaned
        self._client = client or KiwoomClient(KiwoomCredentials.from_env())

    @staticmethod
    def _session(value: object) -> date:
        text = str(value).strip().replace("/", "-")
        if len(text) == 8 and text.isdigit():
            text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise PITDataError("Kiwoom investor flow has invalid session") from exc

    @staticmethod
    def _value(row: dict[str, Any], field: str) -> float:
        raw = row.get(field)
        if raw is None or str(raw).strip() == "":
            raise PITDataError(f"Kiwoom investor flow missing {field}")
        try:
            return float(str(raw).replace(",", ""))
        except ValueError as exc:
            raise PITDataError(f"Kiwoom investor flow has invalid {field}") from exc

    def _map_rows(self, symbol: str, rows: tuple[dict[str, Any], ...]) -> list[dict[str, object]]:
        mapped: list[dict[str, object]] = []
        for row in rows:
            raw_sess = str(row.get("dt") or row.get("session") or "").strip()
            if not raw_sess:
                continue
            sess = self._session(raw_sess).isoformat()
            # ind_invsr: retail, frgnr_invsr: foreign, orgn: institution (in million KRW)
            retail_m = self._value(row, "ind_invsr")
            foreign_m = self._value(row, "frgnr_invsr")
            inst_m = self._value(row, "orgn")
            # convert to KRW float
            retail_net = retail_m * 1_000_000.0
            foreign_net = foreign_m * 1_000_000.0
            inst_net = inst_m * 1_000_000.0
            mapped.append(
                {
                    "session": sess,
                    "ticker": symbol,
                    "foreign_buy_value": max(foreign_net, 0.0),
                    "foreign_sell_value": max(-foreign_net, 0.0),
                    "foreign_net_value": foreign_net,
                    "institution_net_value": inst_net,
                    "retail_net_value": retail_net,
                }
            )
        return mapped

    def _persist_raw_page(
        self,
        symbol: str,
        anchor: date,
        raw_rows: tuple[dict[str, Any], ...],
        *,
        bronze_root: Path | str,
        retrieved_at: datetime | None,
    ) -> BronzeReceipt:
        from src.data.bronze import BronzeStore
        from src.data.schemas import EvidenceKind

        payload = {
            "provider": "KIWOOM",
            "endpoint": "ka10059",
            "symbol": symbol,
            "anchor": anchor.isoformat(),
            "query": {"symbol": symbol, "anchor": anchor.isoformat()},
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
            source_label=f"KIWOOM:ka10059:{symbol}:{anchor.isoformat()}",
        )

    def probe(self, symbol: str, session: date) -> dict[str, object]:
        rows = self._map_rows(symbol, self._client.inquire_investor_trend(symbol, session))
        if not any(row["session"] == session.isoformat() for row in rows):
            raise PITDataError("Kiwoom investor flow missing requested session")
        return {"provider": "KIWOOM", "endpoint": "ka10059", "records": rows}

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
            raise PITDataError("Kiwoom investor flow requested symbol is outside collector universe")

        for symbol in requested_symbols:
            cursor = end
            seen: set[str] = set()
            while cursor >= start:
                raw_rows = self._client.inquire_investor_trend(symbol, cursor)
                if not raw_rows:
                    break
                mapped = self._map_rows(symbol, raw_rows)
                selected = [r for r in mapped if start.isoformat() <= str(r["session"]) <= end.isoformat()]
                unique = [r for r in selected if str(r["session"]) not in seen]
                seen.update(str(r["session"]) for r in unique)
                if bronze_root is not None:
                    self._persist_raw_page(symbol, cursor, raw_rows, bronze_root=bronze_root, retrieved_at=retrieved_at)
                if unique:
                    yield {
                        "provider": "KIWOOM",
                        "endpoint": "ka10059",
                        "symbol": symbol,
                        "anchor": cursor.isoformat(),
                        "records": unique,
                    }
                earliest_session = min(self._session(r.get("dt") or r.get("session")) for r in raw_rows)
                if earliest_session >= cursor:
                    break
                from datetime import timedelta
                cursor = earliest_session - timedelta(days=1)
