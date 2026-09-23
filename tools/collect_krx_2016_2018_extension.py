"""One-time KRX daily-market and security-master backfill for 2016-01-04..2018-12-31.

Session list comes from the ``exchange_calendars`` XKRX calendar, empirically
verified against the scope's own certified 2019-2025 calendar (1719/1719
exact match, zero discrepancy) before being trusted here — this repo has no
KRX "holiday list" endpoint, and deriving sessions by trial-and-error against
the live API would waste quota on every holiday.

Conservative pacing (default 1.5s/request, versus the 1.0s client default)
leaves headroom for a sibling project sharing the same KRX/DART API keys on
this host, per explicit user instruction.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from src.core.pit import EvidenceKind
from src.data.bronze import BronzeStore
from src.data.collection import collect_daily_market_sessions
from src.integrations.krx.client import KrxApiClient
from src.integrations.krx.historical import KrxHistoricalCollector
from src.integrations.quota import ProviderQuotaStateStore

SCOPE_ROOT = Path("data")
BRONZE_ROOT = SCOPE_ROOT / "bronze" / "kr_swing_2019_v1"
STATE_ROOT = SCOPE_ROOT / "state" / "kr_swing_2019_v1"


def _xkrx_sessions(start: str, end: str) -> tuple:
    import exchange_calendars as ecals
    from datetime import date

    cal = ecals.get_calendar("XKRX")
    return tuple(sorted({s.date() for s in cal.sessions_in_range(start, end)}))


def _collect_security_master(sessions: tuple, *, min_interval: float) -> dict[str, int]:
    root = BRONZE_ROOT
    existing: set = set()
    for path in (root / "security_master").glob("*/payload.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = payload.get("as_of") if isinstance(payload, dict) else None
        if value:
            from datetime import date

            existing.add(date.fromisoformat(str(value)[:10]))
    to_fetch = [day for day in sessions if day not in existing]
    store = BronzeStore(root)
    quota_store = ProviderQuotaStateStore(STATE_ROOT / "quota")
    client = KrxApiClient(quota_store=quota_store, min_interval=min_interval)
    started = time.monotonic()
    for index, session in enumerate(to_fetch, start=1):
        records = client.fetch_master_records(session)
        payload = {"provider": "KRX", "endpoint": "sto/*_isu_base_info", "as_of": session.isoformat(), "records": records}
        store.import_bytes(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
            kind=EvidenceKind.SECURITY_MASTER,
            retrieved_at=datetime.now(UTC),
            source_label=f"KRX:historical-master:{session.isoformat()}",
        )
        if index % 25 == 0 or index == len(to_fetch):
            sys.stdout.write(json.dumps({"stage": "security_master", "done": index, "total": len(to_fetch), "elapsed_s": round(time.monotonic() - started, 1)}) + "\n")
            sys.stdout.flush()
    return {"new": len(to_fetch), "existing": len(existing)}


def _collect_daily_market(sessions: tuple, *, min_interval: float) -> dict[str, int]:
    quota_store = ProviderQuotaStateStore(STATE_ROOT / "quota")
    krx = KrxHistoricalCollector(quota_store=quota_store, min_interval=min_interval)
    artifact = collect_daily_market_sessions(
        sessions=sessions, krx=krx, bronze_root=BRONZE_ROOT, retrieved_at=datetime.now(UTC)
    )
    return {"content_hash": artifact.content_hash, "sessions": len(sessions)}


def main() -> int:
    sessions = _xkrx_sessions("2016-01-04", "2018-12-31")
    sys.stdout.write(json.dumps({"stage": "plan", "sessions": len(sessions), "first": str(sessions[0]), "last": str(sessions[-1])}) + "\n")
    sys.stdout.flush()
    master_result = _collect_security_master(sessions, min_interval=1.5)
    sys.stdout.write(json.dumps({"stage": "security_master_done", **master_result}) + "\n")
    sys.stdout.flush()
    market_result = _collect_daily_market(sessions, min_interval=1.5)
    sys.stdout.write(json.dumps({"stage": "daily_market_done", **market_result}) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
