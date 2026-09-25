"""Resumable OpenDART fact backfill for fiscal periods 2016Q1..2018Q4.

Run repeatedly (once per KST day) until it reports ``status == "complete"``.
``--dry-run`` resolves the plan and estimates the remaining work with zero API
calls.

Why this script drives lower-level pieces instead of
``run_dart_historical_backfill_batch``: that helper derives ``required_periods``
as a fixed trailing lookback from one ``validation_start``, which cannot express
"collect this whole historical range", and its security-master loader targets a
nested table layout this scope does not use.

Safety properties:
- Identities already answered in the receipt catalog (success, empty, or
  extraction_failed) are skipped; provider-unavailable/blocked ones are retried.
- Each chunk is persisted (Bronze + one catalog revision) before the next
  request, so a failure loses at most one chunk instead of the whole run.
- Headroom comes from ``scoped_dart_request_headroom`` and is re-read before
  every chunk; a chunk is sized for the worst case of three requests per
  identity (CFS, OFS, document archive).
- The first ``blocked`` page (DART quota exhausted) stops the run immediately.
- A health check runs before collection, and three consecutive transport failures
  (connection reset, client cooldown) abort the chunk without persisting the failed
  identities; failed identities stay pending and are retried by the next run.
- Catalog revisions are full snapshots; after a run, reclaim old revisions with
  ``compact-storage-generations --apply``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.data.cli import load_data_runtime
from src.data.collection import collect_dart_financial_facts
from src.data.dart_backfill import build_scoped_dart_collector, scoped_dart_request_headroom
from src.data.research_scope import PRIMARY_DART_KEY_ENV
from src.integrations.dart.xbrl import DartCircuitOpenError
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.scoped_ingestion import FACT_SOURCE, ScopedBronzeWriter, dart_fact_natural_key
from src.integrations.dart.client import DartCorpCodeRecord
from src.integrations.dart.xbrl import DartXbrlCollector
from src.integrations.quota import ProviderQuotaStateStore

REQUIRED_PERIODS = frozenset(f"{year}Q{q}" for year in (2016, 2017, 2018) for q in (1, 2, 3, 4))
WINDOW_START = date(2015, 1, 1)
WINDOW_END = date(2019, 6, 30)
ELIGIBILITY_START = date(2016, 1, 4)
ELIGIBILITY_END = date(2018, 12, 31)
_ANSWERED = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY, EvidenceStatus.EXTRACTION_FAILED})
_WORST_CASE_REQUESTS_PER_IDENTITY = 3


def _emit(**fields: object) -> None:
    sys.stdout.write(json.dumps(fields, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _eligible_tickers(silver_root: Path) -> frozenset[str]:
    datasets = sorted(p for p in silver_root.glob("ordinary_universe_*") if p.is_dir() and not p.name.startswith("."))
    if len(datasets) != 1:
        raise SystemExit(f"expected exactly one ordinary_universe dataset, found {len(datasets)}")
    files = sorted(datasets[0].rglob("*.parquet"))
    tickers = (
        pl.scan_parquet([str(p) for p in files])
        .filter(pl.col("eligible") & pl.col("session").is_between(ELIGIBILITY_START, ELIGIBILITY_END))
        .select(pl.col("ticker").cast(pl.String))
        .unique()
        .collect()["ticker"]
        .to_list()
    )
    return frozenset(str(t) for t in tickers)


def _ticker_by_corp_code(bronze_root: Path, eligible: frozenset[str]) -> dict[str, str]:
    paths = sorted((bronze_root / "dart_corp_codes").glob("*/payload.json"))
    if not paths:
        raise SystemExit("dart_corp_codes Bronze evidence is missing")
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        rec = DartCorpCodeRecord(
            ticker=str(row.get("ticker") or ""), corp_code=str(row.get("corp_code") or ""), corp_name=str(row.get("corp_name") or "")
        )
        if rec.ticker not in eligible or not rec.corp_code:
            continue
        prev = mapping.get(rec.corp_code)
        if prev is not None and prev != rec.ticker:
            raise SystemExit(f"corp code {rec.corp_code} maps to multiple tickers: {prev}, {rec.ticker}")
        mapping[rec.corp_code] = rec.ticker
    return mapping


def _corp_codes_without_disclosures(bronze_root: Path, corp_codes: frozenset[str]) -> frozenset[str]:
    seen: set[str] = set()
    for path in (bronze_root / "disclosures").glob("*/payload.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            code = str(payload.get("corp_code") or "").strip()
            if code:
                seen.add(code)
    return corp_codes - seen


def _pending_identities(bronze_root: Path, catalog: ReceiptCatalog, mapping: dict[str, str]) -> tuple[int, int, list[dict[str, str]]]:
    identities = DartXbrlCollector.filing_identities_from_bronze(
        bronze_root, start=WINDOW_START, end=WINDOW_END,
        ticker_by_corp_code=mapping, required_periods=REQUIRED_PERIODS, corp_codes=frozenset(mapping),
    )
    latest: dict[str, dict[str, str]] = {}
    for item in identities:
        key = dart_fact_natural_key(corp_code=item["corp_code"], biz_year=item["biz_year"], reprt_code=item["reprt_code"])
        current = latest.get(key)
        if current is None or (item["published_at"], item["filing_id"]) > (current["published_at"], current["filing_id"]):
            latest[key] = item
    answered = catalog.latest(source=FACT_SOURCE, natural_keys=set(latest))
    pending = [
        item for key, item in latest.items()
        if key not in answered or answered[key].status not in _ANSWERED
    ]
    pending.sort(key=lambda item: (item["published_at"], item["filing_id"]))
    return len(identities), len(latest), pending


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="resolve and estimate only; makes no API calls")
    parser.add_argument("--max-chunks", type=int, default=None, help="stop after N chunks (pilot)")
    parser.add_argument(
        "--key-env", default=PRIMARY_DART_KEY_ENV,
        help="environment variable of the OpenDART key; budget, reserve, and pacing come from the scope policy of that key",
    )
    args = parser.parse_args(argv)
    if not os.environ.get(args.key_env):
        raise SystemExit(f"{args.key_env} is not set")

    runtime = load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"), data_root=Path("/home/kth/k-stock-engine/data")
    )
    bronze_root = runtime.workspace.bronze_root
    chunk_size = runtime.scope.collection.dart_batch_identities
    mapping = _ticker_by_corp_code(bronze_root, _eligible_tickers(runtime.workspace.silver_root))
    missing_disclosures = _corp_codes_without_disclosures(bronze_root, frozenset(mapping))
    if missing_disclosures:
        _emit(stage="abort", reason="disclosures_missing", corp_codes=len(missing_disclosures))
        return 2

    catalog = ReceiptCatalog(bronze_root / "catalog")
    raw_count, key_count, pending = _pending_identities(bronze_root, catalog, mapping)
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    headroom = scoped_dart_request_headroom(runtime=runtime, quota_store=quota_store, key_env=args.key_env)
    policy = runtime.scope.collection.dart_key_policy(args.key_env)
    daily = policy.daily_budget - policy.daily_reserve
    _emit(
        stage="plan", key_env=args.key_env, corp_codes=len(mapping), raw_identities=raw_count, unique_keys=key_count,
        pending=len(pending), chunk_size=chunk_size, headroom_now=headroom,
        est_requests_min=len(pending), est_requests_max=len(pending) * _WORST_CASE_REQUESTS_PER_IDENTITY,
        est_days_min=round(len(pending) / daily, 2), est_days_max=round(len(pending) * _WORST_CASE_REQUESTS_PER_IDENTITY / daily, 2),
    )
    if args.dry_run or not pending:
        _emit(stage="done", status="dry_run" if args.dry_run else "complete", pending=len(pending))
        return 0

    dart = build_scoped_dart_collector(runtime=runtime, quota_store=quota_store, key_env=args.key_env)
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    try:
        dart.health_check()
    except Exception as exc:  # noqa: BLE001 - any failure means this host cannot collect right now
        _emit(stage="done", status="provider_unreachable", error=str(exc)[:200], pending_left=len(pending))
        return 3
    totals = {"standardized": 0, "legacy_document": 0, "unavailable": 0, "extraction_failed": 0}
    done = requests_used = chunks = 0
    status = "complete"
    while done < len(pending):
        if args.max_chunks is not None and chunks >= args.max_chunks:
            status = "chunk_limit"
            break
        headroom = scoped_dart_request_headroom(runtime=runtime, quota_store=quota_store, key_env=args.key_env)
        allowance = min(chunk_size, len(pending) - done, headroom // _WORST_CASE_REQUESTS_PER_IDENTITY)
        if allowance < 1:
            status = "budget_exhausted"
            break
        chunk = tuple(pending[done : done + allowance])
        before = headroom
        try:
            artifact = collect_dart_financial_facts(
                dart=dart, identities=chunk, bronze_root=bronze_root, retrieved_at=datetime.now(UTC), scoped_writer=writer
            )
        except DartCircuitOpenError:
            status = "provider_unstable"
            break
        report = json.loads(Path(artifact.report_path).read_text(encoding="utf-8"))
        for name in totals:
            totals[name] += int(report.get(name, 0))
        used = before - scoped_dart_request_headroom(runtime=runtime, quota_store=quota_store, key_env=args.key_env)
        requests_used += used
        done += len(report["filing_ids"])
        chunks += 1
        avg = requests_used / done
        _emit(
            stage="chunk", chunk=chunks, done=done, pending=len(pending), requests_used=requests_used,
            avg_requests_per_identity=round(avg, 3), remaining_est_days=round((len(pending) - done) * avg / daily, 2), **totals,
        )
        if int(report.get("blocked", 0)) > 0:
            status = "quota_blocked"
            break
        if dart.aborted:
            # 연속 전송 실패는 제공자 제한의 신호이므로 더 밀어붙이지 않는다.
            status = "provider_unstable"
            break
    _emit(stage="done", status=status, done=done, pending_left=len(pending) - done, requests_used=requests_used, **totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
