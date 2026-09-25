"""Resumable OpenDART cash-dividend decision collection (disclosures then archives).

Runs only after the 2016-2018 fact backfill completes and shares the DART key,
so every chunk re-reads ``scoped_dart_request_headroom`` and stops on a
``blocked`` result. Pilot first with ``--max-filings 50`` across 2016-2025 to
validate the parser before the full run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from src.data.cli import load_data_runtime
from src.data.dart_backfill import build_scoped_dart_collector, scoped_dart_request_headroom
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.research_scope import PRIMARY_DART_KEY_ENV
from src.data.schemas import EvidenceKind
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload
from src.integrations.dart.client import DartCorpCodeRecord
from src.integrations.dart.dividend_decision import is_dividend_decision_title
from src.integrations.quota import ProviderQuotaStateStore

DIVIDEND_DECISION_SOURCE = "opendart:dividend_decision"
COVERAGE_START = date(2016, 1, 1)
COVERAGE_END = date(2025, 12, 31)
_WINDOW_MONTHS = 3


def _emit(**fields: object) -> None:
    sys.stdout.write(json.dumps(fields, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _windows(start: date, end: date) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    year, month = start.year, start.month
    while True:
        w_start = date(year, month, 1)
        m2 = month + _WINDOW_MONTHS - 1
        y2 = year + (m2 - 1) // 12
        m2 = (m2 - 1) % 12 + 1
        import calendar as _cal

        w_end = date(y2, m2, _cal.monthrange(y2, m2)[1])
        if w_end > end:
            w_end = end
        if w_start <= end and w_end >= start:
            out.append((max(w_start, start), w_end))
        if w_end >= end:
            break
        month = m2 + 1
        year = y2
        if month > 12:
            month = 1
            year += 1
    return out


def _eligible_corp_codes(bronze_root: Path) -> dict[str, str]:
    paths = sorted((bronze_root / "dart_corp_codes").glob("*/payload.json"))
    if not paths:
        raise SystemExit("dart_corp_codes Bronze evidence is missing")
    mapping: dict[str, str] = {}
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    for row in payload:
        if not isinstance(row, dict):
            continue
        rec = DartCorpCodeRecord(
            ticker=str(row.get("ticker") or ""),
            corp_code=str(row.get("corp_code") or ""),
            corp_name=str(row.get("corp_name") or ""),
        )
        if not rec.corp_code:
            continue
        mapping[rec.corp_code] = rec.ticker
    return mapping


def _envelope_payload(*, rcept_no: str, corp_code: str, received_on: str, report_nm: str, archive: bytes) -> bytes:
    envelope = {
        "rcept_no": rcept_no,
        "corp_code": corp_code,
        "received_on": received_on,
        "report_nm": report_nm,
        "archive_b64": base64.b64encode(archive).decode("ascii"),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="resolve and estimate only; makes no API calls")
    parser.add_argument("--max-filings", type=int, default=None, help="stop after N filings (pilot, e.g. 50)")
    parser.add_argument("--key-env", default=PRIMARY_DART_KEY_ENV)
    args = parser.parse_args(argv)
    if not os.environ.get(args.key_env):
        raise SystemExit(f"{args.key_env} is not set")

    runtime = load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"), data_root=Path("/home/kth/k-stock-engine/data")
    )
    bronze_root = runtime.workspace.bronze_root
    mapping = _eligible_corp_codes(bronze_root)
    catalog = ReceiptCatalog(bronze_root / "catalog")
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    headroom = scoped_dart_request_headroom(runtime=runtime, quota_store=quota_store, key_env=args.key_env)
    _emit(stage="plan", corp_codes=len(mapping), headroom_now=headroom, coverage=[COVERAGE_START.isoformat(), COVERAGE_END.isoformat()])

    dart = build_scoped_dart_collector(runtime=runtime, quota_store=quota_store, key_env=args.key_env)
    try:
        dart.health_check()
    except Exception as exc:  # noqa: BLE001 - any failure means this host cannot collect right now
        _emit(stage="done", status="provider_unreachable", error=str(exc)[:200])
        return 3

    answered = catalog.latest(source=DIVIDEND_DECISION_SOURCE, natural_keys=set())
    _ = answered
    matched: list[dict[str, str]] = []
    status = "complete"
    for w_start, w_end in _windows(COVERAGE_START, COVERAGE_END):
        headroom = scoped_dart_request_headroom(runtime=runtime, quota_store=quota_store, key_env=args.key_env)
        if headroom < 1:
            status = "budget_exhausted"
            break
        try:
            disclosures = dart.list_disclosures(w_start, w_end)
        except Exception as exc:  # noqa: BLE001 - transport/quota failures stop the run without losing chunks
            message = str(exc).lower()
            if "blocked" in message or "quota" in message:
                status = "quota_blocked"
                break
            status = "provider_unstable"
            break
        for item in disclosures:
            if item.get("corp_code") not in mapping:
                continue
            if not is_dividend_decision_title(str(item.get("report_nm") or "")):
                continue
            matched.append(item)
            if args.max_filings is not None and len(matched) >= args.max_filings:
                break
        if args.max_filings is not None and len(matched) >= args.max_filings:
            status = "chunk_limit"
            break
    _emit(stage="disclosures", matched=len(matched))
    if args.dry_run or not matched:
        _emit(stage="done", status="dry_run" if args.dry_run else "complete", matched=len(matched))
        return 0

    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    done = 0
    chunk: list[ScopedRawPayload] = []
    for item in matched:
        headroom = scoped_dart_request_headroom(runtime=runtime, quota_store=quota_store, key_env=args.key_env)
        if headroom < 1:
            status = "budget_exhausted"
            break
        rcept_no = str(item["rcept_no"])
        try:
            archive = dart.fetch_document_archive(rcept_no)
        except Exception as exc:  # noqa: BLE001 - keep the run resumable; failed filings stay pending
            message = str(exc).lower()
            if "blocked" in message or "quota" in message:
                status = "quota_blocked"
                break
            continue
        received_on = str(item.get("rcept_dt") or "").strip()
        try:
            as_of = date.fromisoformat(received_on[:10]) if received_on else datetime.now(UTC).date()
        except ValueError:
            continue
        chunk.append(
            ScopedRawPayload(
                kind=EvidenceKind.CORPORATE_ACTIONS,
                source=DIVIDEND_DECISION_SOURCE,
                natural_key=rcept_no,
                as_of=as_of,
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                payload=_envelope_payload(
                    rcept_no=rcept_no,
                    corp_code=str(item.get("corp_code") or ""),
                    received_on=as_of.isoformat(),
                    report_nm=str(item.get("report_nm") or ""),
                    archive=archive,
                ),
                retrieved_at=datetime.now(UTC),
                source_label=f"{DIVIDEND_DECISION_SOURCE}:{rcept_no}",
            )
        )
        done += 1
        if len(chunk) >= 20:
            writer.persist_many(tuple(chunk))
            _emit(stage="chunk", done=done, matched=len(matched))
            chunk = []
    if chunk:
        writer.persist_many(tuple(chunk))
        _emit(stage="chunk", done=done, matched=len(matched))
    _emit(stage="done", status=status, done=done, matched=len(matched))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
