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
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from src.data.cli import load_data_runtime
from src.data.dart_backfill import build_scoped_dart_collector, scoped_dart_request_headroom
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.research_scope import PRIMARY_DART_KEY_ENV
from src.data.schemas import EvidenceKind
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload
from src.integrations.dart.dividend_decision import is_dividend_decision_title
from src.integrations.quota import ProviderQuotaStateStore

DIVIDEND_DECISION_SOURCE = "opendart:dividend_decision"
COVERAGE_START = date(2016, 1, 1)
COVERAGE_END = date(2025, 12, 31)
# Per-corp disclosure lists already in Bronze cover the scope up to this date; only later windows need the API.
BRONZE_LIST_END = date(2019, 6, 30)
_DIVIDEND_DETAIL_TYPE = "I001"
_CHUNK = 40
_FAILURE_THRESHOLD = 3
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


def _eligible_corp_codes(bronze_root: Path, silver_root: Path) -> dict[str, str]:
    """Corp code -> ticker for the ordinary-share universe (the same scope as the fact backfill)."""
    from collect_dart_2016_2018_extension import _eligible_tickers, _ticker_by_corp_code

    return _ticker_by_corp_code(bronze_root, _eligible_tickers(silver_root))


def _envelope_payload(*, rcept_no: str, corp_code: str, received_on: str, report_nm: str, archive: bytes | None) -> bytes:
    if archive is None:
        return json.dumps(
            {"rcept_no": rcept_no, "corp_code": corp_code, "received_on": received_on, "report_nm": report_nm, "document_status": "unavailable"},
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")
    envelope = {
        "rcept_no": rcept_no,
        "corp_code": corp_code,
        "received_on": received_on,
        "report_nm": report_nm,
        "archive_b64": base64.b64encode(archive).decode("ascii"),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _bronze_list_matches(bronze_root: Path, eligible: dict[str, str]) -> dict[str, dict[str, str]]:
    matches: dict[str, dict[str, str]] = {}
    for path in (bronze_root / "disclosures").glob("*/payload.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, dict) else None
        for item in records or ():
            if (
                item.get("corp_code") in eligible
                and COVERAGE_START.isoformat().replace("-", "") <= str(item.get("rcept_dt"))
                and is_dividend_decision_title(str(item.get("report_nm") or ""))
            ):
                matches[str(item["rcept_no"])] = item
    return matches


def _api_list_matches(dart, runtime, quota_store, key_env: str, eligible: dict[str, str], cache_path: Path) -> tuple[dict[str, dict[str, str]], str]:  # type: ignore[no-untyped-def]
    """List dividend decisions after the Bronze list horizon; closed windows are cached so reruns cost nothing."""
    cache: dict[str, list[dict[str, str]]] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    matches: dict[str, dict[str, str]] = {}
    status = "complete"
    for w_start, w_end in _windows(BRONZE_LIST_END + timedelta(days=1), COVERAGE_END):
        key = f"{w_start.isoformat()}..{w_end.isoformat()}"
        if key not in cache:
            if scoped_dart_request_headroom(runtime=runtime, quota_store=quota_store, key_env=key_env) < 1:
                return matches, "budget_exhausted"
            try:
                rows = dart.list_disclosures(w_start, w_end, detail_type=_DIVIDEND_DETAIL_TYPE)
            except Exception as exc:  # noqa: BLE001 - transport/quota failures stop the run without losing cached windows
                message = str(exc).lower()
                return matches, "quota_blocked" if ("blocked" in message or "quota" in message) else "provider_unstable"
            cache[key] = [r for r in rows if is_dividend_decision_title(str(r.get("report_nm") or ""))]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        for item in cache[key]:
            if item.get("corp_code") in eligible:
                matches[str(item["rcept_no"])] = item
    return matches, status


def _fetch(dart, rcept_no: str) -> tuple[bytes | None, str | None]:  # type: ignore[no-untyped-def]
    try:
        return dart.fetch_document_archive(rcept_no), None
    except Exception as exc:  # noqa: BLE001 - keep the run resumable; failed filings stay pending
        return None, str(exc) or type(exc).__name__


def _payload(item: dict[str, str], archive: bytes | None) -> ScopedRawPayload:
    assert archive is not None
    rcept_no = str(item["rcept_no"])
    as_of = date.fromisoformat(f"{item['rcept_dt'][:4]}-{item['rcept_dt'][4:6]}-{item['rcept_dt'][6:8]}")
    is_zip = archive[:2] == b"PK"
    return ScopedRawPayload(
        kind=EvidenceKind.CORPORATE_ACTIONS,
        source=DIVIDEND_DECISION_SOURCE,
        natural_key=rcept_no,
        as_of=as_of,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS if is_zip else EvidenceStatus.EMPTY,
        # DART answers "file does not exist" (014) for some corrections; record the absence
        # without an archive so the filing is not refetched and the materializer skips it.
        payload=_envelope_payload(
            rcept_no=rcept_no,
            corp_code=str(item.get("corp_code") or ""),
            received_on=as_of.isoformat(),
            report_nm=str(item.get("report_nm") or ""),
            archive=archive if is_zip else None,
        ),
        retrieved_at=datetime.now(UTC),
        source_label=f"{DIVIDEND_DECISION_SOURCE}:{rcept_no}",
    )


def _spread(items: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Pick ``limit`` filings evenly across the date-sorted list so a pilot covers every year."""
    if limit >= len(items):
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="resolve and estimate only; makes no document requests")
    parser.add_argument("--max-filings", type=int, default=None, help="pilot: fetch N filings spread evenly over 2016-2025")
    parser.add_argument("--key-env", default=PRIMARY_DART_KEY_ENV)
    args = parser.parse_args(argv)
    if not os.environ.get(args.key_env):
        raise SystemExit(f"{args.key_env} is not set")

    runtime = load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"), data_root=Path("/home/kth/k-stock-engine/data")
    )
    bronze_root = runtime.workspace.bronze_root
    eligible = _eligible_corp_codes(bronze_root, runtime.workspace.silver_root)
    catalog = ReceiptCatalog(bronze_root / "catalog")
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    _emit(stage="plan", corp_codes=len(eligible), headroom_now=scoped_dart_request_headroom(runtime=runtime, quota_store=quota_store, key_env=args.key_env))

    dart = build_scoped_dart_collector(runtime=runtime, quota_store=quota_store, key_env=args.key_env)
    try:
        dart.health_check()
    except Exception as exc:  # noqa: BLE001 - any failure means this host cannot collect right now
        _emit(stage="done", status="provider_unreachable", error=str(exc)[:200])
        return 3

    listed = _bronze_list_matches(bronze_root, eligible)
    api_listed, status = _api_list_matches(
        dart, runtime, quota_store, args.key_env, eligible, runtime.workspace.state_root / "dividend_decision_lists.json"
    )
    listed.update(api_listed)
    answered = catalog.latest(source=DIVIDEND_DECISION_SOURCE, natural_keys=set(listed))
    pending = sorted(
        (item for key, item in listed.items() if key not in answered),
        key=lambda item: (str(item["rcept_dt"]), str(item["rcept_no"])),
    )
    _emit(stage="disclosures", listed=len(listed), answered=len(answered), pending=len(pending), list_status=status)
    if args.max_filings is not None:
        pending = _spread(pending, args.max_filings)
    if args.dry_run or not pending:
        _emit(stage="done", status="dry_run" if args.dry_run else status, pending=len(pending))
        return 0

    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    workers = runtime.scope.collection.dart_key_policy(args.key_env).max_workers
    done = 0
    consecutive_failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(pending), _CHUNK):
            batch = pending[start : start + _CHUNK]
            if scoped_dart_request_headroom(runtime=runtime, quota_store=quota_store, key_env=args.key_env) < len(batch):
                status = "budget_exhausted"
                break
            results = list(pool.map(lambda item: _fetch(dart, str(item["rcept_no"])), batch))
            payloads: list[ScopedRawPayload] = []
            for item, (archive, error) in zip(batch, results, strict=True):
                if error is not None:
                    message = error.lower()
                    if "blocked" in message or "quota" in message:
                        status = "quota_blocked"
                    consecutive_failures += 1
                    continue
                consecutive_failures = 0
                payloads.append(_payload(item, archive))
            if payloads:
                writer.persist_many(tuple(payloads))
                done += len(payloads)
            _emit(stage="chunk", done=done, pending=len(pending))
            if status == "quota_blocked":
                break
            if consecutive_failures >= _FAILURE_THRESHOLD:
                # 연속 전송 실패는 차단 신호일 수 있어 즉시 멈추고 다음 실행에서 이어 받는다.
                status = "provider_unstable"
                break
    _emit(stage="done", status=status, done=done, pending=len(pending))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
