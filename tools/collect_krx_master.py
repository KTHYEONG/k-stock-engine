"""Resumable KRX historical master collection for PIT universe preparation."""
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

from src.data.bronze import BronzeStore
from src.data.schemas import EvidenceKind
from src.integrations.krx.client import KrxApiClient


def main() -> int:
    root = Path("data/bronze/stocks")
    calendar_path = next(iter((root / "calendar").glob("*/payload.json")), None)
    if calendar_path is None:
        raise SystemExit("calendar Bronze receipt is required")
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
    sessions = sorted({date.fromisoformat(str(value)[:10]) for value in calendar["sessions"]})
    sessions = [value for value in sessions if value >= date(2016, 1, 4)]
    existing: set[date] = set()
    for path in (root / "security_master").glob("*/payload.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload.get("as_of") if isinstance(payload, dict) else None
            if value:
                existing.add(date.fromisoformat(str(value)[:10]))
        except (OSError, ValueError, TypeError):
            continue
    store = BronzeStore(root)
    client = KrxApiClient()
    started = time.monotonic()
    completed = 0
    for session in sessions:
        if session in existing:
            continue
        records = client.fetch_master_records(session)
        payload = {"provider": "KRX", "endpoint": "sto/*_isu_base_info", "as_of": session.isoformat(), "records": records}
        store.import_bytes(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
            kind=EvidenceKind.SECURITY_MASTER,
            retrieved_at=datetime.now(UTC),
            source_label=f"KRX:historical-master:{session.isoformat()}",
        )
        completed += 1
        if completed % 25 == 0:
            sys.stdout.write(json.dumps({"new": completed, "existing": len(existing), "total": len(sessions), "elapsed_seconds": round(time.monotonic() - started, 1)}) + "\n")
            sys.stdout.flush()
    sys.stdout.write(json.dumps({"new": completed, "existing": len(existing), "total": len(sessions), "elapsed_seconds": round(time.monotonic() - started, 1)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
