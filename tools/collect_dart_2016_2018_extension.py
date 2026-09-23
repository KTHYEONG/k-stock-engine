"""One-time OpenDART fact backfill for fiscal periods 2016Q1..2018Q4.

The existing ``build_dart_historical_backfill_plan``/``run_dart_historical_backfill_batch``
helpers compute ``required_periods`` as a fixed trailing lookback from one
``validation_start`` point (the scope's live QVEF fundamental-lookback
semantics), which cannot express "collect this entire historical range."
This script drives the lower-level, range-agnostic building blocks directly
instead: an explicit ``required_periods`` set, ticker<->corp_code resolution
from this scope's own certified ordinary universe (not the legacy nested
security_master loader, which globs unrelated flat-hash Silver datasets on
this scope's layout), and the scope's existing quota-headroom reservation
(``scoped_dart_request_headroom``) so today's run never exceeds the budget
left after reserving room for a sibling project's own DART usage.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.data.cli import load_data_runtime
from src.data.collection import collect_dart_disclosures, collect_dart_financial_facts
from src.data.dart_backfill import build_scoped_dart_collector, scoped_dart_request_headroom
from src.data.receipt_catalog import ReceiptCatalog
from src.data.scoped_ingestion import ScopedBronzeWriter
from src.integrations.dart.client import DartCorpCodeRecord

REQUIRED_PERIODS = frozenset(f"{year}Q{q}" for year in (2016, 2017, 2018) for q in (1, 2, 3, 4))


def _eligible_tickers(silver_root: Path, *, start: date, end: date) -> frozenset[str]:
    datasets = sorted(p for p in silver_root.glob("ordinary_universe_*") if p.is_dir() and not p.name.startswith("."))
    if len(datasets) != 1:
        raise SystemExit(f"expected exactly one ordinary_universe dataset, found {len(datasets)}")
    files = sorted(datasets[0].rglob("*.parquet"))
    tickers = (
        pl.scan_parquet([str(p) for p in files])
        .filter(pl.col("eligible") & pl.col("session").is_between(start, end))
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
    records = tuple(
        DartCorpCodeRecord(ticker=str(row.get("ticker") or ""), corp_code=str(row.get("corp_code") or ""), corp_name=str(row.get("corp_name") or ""))
        for row in payload
        if isinstance(row, dict)
    )
    mapping: dict[str, str] = {}
    for rec in records:
        if rec.ticker not in eligible or not rec.corp_code:
            continue
        prev = mapping.get(rec.corp_code)
        if prev is not None and prev != rec.ticker:
            raise SystemExit(f"corp code {rec.corp_code} maps to multiple tickers: {prev}, {rec.ticker}")
        mapping[rec.corp_code] = rec.ticker
    return {code: mapping[code] for code in mapping}


def main() -> int:
    runtime = load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"),
        data_root=Path("/home/kth/k-stock-engine/data"),
    )
    bronze_root = runtime.workspace.bronze_root
    silver_root = runtime.workspace.silver_root
    eligible = _eligible_tickers(silver_root, start=date(2016, 1, 4), end=date(2018, 12, 31))
    ticker_by_corp_code = _ticker_by_corp_code(bronze_root, eligible)
    sys.stdout.write(json.dumps({"stage": "resolved", "eligible_tickers": len(eligible), "corp_codes": len(ticker_by_corp_code)}) + "\n")
    sys.stdout.flush()

    headroom = scoped_dart_request_headroom(runtime=runtime)
    sys.stdout.write(json.dumps({"stage": "headroom", "available": headroom}) + "\n")
    sys.stdout.flush()
    if headroom <= 0:
        sys.stdout.write(json.dumps({"stage": "done", "reason": "no headroom left today"}) + "\n")
        return 0

    dart = build_scoped_dart_collector(runtime=runtime)
    retrieved_at = datetime.now(UTC)
    corp_codes = tuple(sorted(ticker_by_corp_code))
    collect_dart_disclosures(
        dart=dart, start=date(2015, 1, 1), end=date(2019, 6, 30), bronze_root=bronze_root, retrieved_at=retrieved_at, corp_codes=corp_codes
    )
    sys.stdout.write(json.dumps({"stage": "disclosures_done"}) + "\n")
    sys.stdout.flush()

    identities = dart.filing_identities_from_bronze(
        bronze_root, start=date(2015, 1, 1), end=date(2019, 6, 30),
        ticker_by_corp_code=ticker_by_corp_code, required_periods=REQUIRED_PERIODS, corp_codes=frozenset(corp_codes),
    )
    sys.stdout.write(json.dumps({"stage": "identities_resolved", "count": len(identities)}) + "\n")
    sys.stdout.flush()

    catalog = ReceiptCatalog(bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    batch = identities[:headroom]
    if batch:
        collect_dart_financial_facts(dart=dart, identities=batch, bronze_root=bronze_root, retrieved_at=retrieved_at, scoped_writer=writer)
    sys.stdout.write(json.dumps({"stage": "facts_done", "collected": len(batch), "remaining": len(identities) - len(batch)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
