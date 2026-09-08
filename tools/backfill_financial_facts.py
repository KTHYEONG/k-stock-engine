"""Fast multi-account OpenDART financial facts collector (2016-2025)."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import polars as pl

from src.integrations.dart.client import DartApiClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("financial_backfill")

FACT_MAP = {
    "ifrs-full_Revenue": "sales",
    "dart_Revenue": "sales",
    "ifrs-full_GrossProfit": "gross_profit",
    "dart_OperatingIncomeLoss": "operating_profit",
    "ifrs-full_ProfitLoss": "net_income",
    "ifrs-full_Assets": "assets",
    "ifrs-full_Liabilities": "debt",
    "ifrs-full_Equity": "equity",
}

FACT_NM_MAP = {
    "매출액": "sales",
    "수익(매출액)": "sales",
    "영업수익": "sales",
    "매출총이익": "gross_profit",
    "영업이익": "operating_profit",
    "영업이익(손실)": "operating_profit",
    "당기순이익": "net_income",
    "당기순이익(손실)": "net_income",
    "자산총계": "assets",
    "부채총계": "debt",
    "자본총계": "equity",
}

REPORT_CODES = [
    ("11013", "Q1"),
    ("11012", "Q2"),
    ("11014", "Q3"),
    ("11011", "Q4"),
]


def load_corp_code_map(bronze_root: Path) -> dict[str, str]:
    existing = sorted((bronze_root / "dart_corp_codes").glob("*/payload.json"))
    if not existing:
        raise RuntimeError("No cached dart_corp_codes found")
    payload = json.loads(existing[-1].read_text(encoding="utf-8"))
    ticker_by_corp: dict[str, str] = {}
    for row in payload:
        if isinstance(row, dict):
            c = str(row.get("corp_code") or "").strip()
            t = str(row.get("ticker") or "").strip()
            if c and t and len(t) == 6 and t.isdigit():
                ticker_by_corp[c] = t
    return ticker_by_corp


def run_financial_backfill(
    years: list[int] | None = None,
    force: bool = False,
    bronze_root: Path = Path("data/bronze/stocks"),
    silver_root: Path = Path("data/silver/stocks"),
) -> None:
    api_key = os.getenv("OPENDART_API_KEY")
    if not api_key:
        raise ValueError("OPENDART_API_KEY not configured")

    client = DartApiClient(api_key=api_key)
    ticker_by_corp = load_corp_code_map(bronze_root)
    corp_codes = sorted(ticker_by_corp.keys())
    logger.info("Found %d listed corporations in corp_code map", len(corp_codes))

    all_target_years = list(range(2016, 2026)) if years is None else years
    silver_dir = silver_root / "financial_facts"
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Filter already collected years unless force=True
    pending_years: list[int] = []
    for y in all_target_years:
        target_parquet = silver_dir / f"year={y}" / f"facts_{y}.parquet"
        if target_parquet.exists() and not force:
            try:
                cnt = pl.read_parquet(target_parquet).height
                if cnt > 50000:
                    logger.info("Year %d already has %d facts in %s (skipping)", y, cnt, target_parquet)
                    continue
            except Exception:
                pass
        pending_years.append(y)

    logger.info("Pending years to fetch: %s", pending_years)
    if not pending_years:
        logger.info("All requested years are already collected!")
        return

    chunk_size = 100
    chunks = [corp_codes[i : i + chunk_size] for i in range(0, len(corp_codes), chunk_size)]

    fact_bronze_dir = bronze_root / "financial_facts"
    fact_bronze_dir.mkdir(parents=True, exist_ok=True)

    total_calls = 0
    start_time = time.time()

    for year in pending_years:
        year_rows: list[dict[str, Any]] = []
        logger.info("====== Starting Year: %d ======", year)

        for reprt_code, quarter in REPORT_CODES:
            fiscal_period = f"{year}{quarter}"
            period_rows = 0

            for chunk_idx, chunk in enumerate(chunks):
                total_calls += 1
                try:
                    records = client.fetch_multi_accounts(tuple(chunk), biz_year=str(year), reprt_code=reprt_code)
                except Exception as exc:
                    logger.warning("Error fetching chunk %d for %s: %s", chunk_idx, fiscal_period, exc)
                    time.sleep(0.5)
                    continue

                if not records:
                    time.sleep(0.1)
                    continue

                digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
                page_dir = fact_bronze_dir / digest
                page_dir.mkdir(parents=True, exist_ok=True)
                (page_dir / "payload.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

                for rec in records:
                    fs_div = str(rec.get("fs_div") or "").strip()
                    corp_code = str(rec.get("corp_code") or "").strip().zfill(8)
                    ticker = ticker_by_corp.get(corp_code, "")
                    if not ticker:
                        continue

                    acc_id = str(rec.get("account_id") or "").strip()
                    acc_nm = str(rec.get("account_nm") or "").strip()

                    fact_name = FACT_MAP.get(acc_id) or FACT_NM_MAP.get(acc_nm)
                    if not fact_name:
                        continue

                    raw_val = str(rec.get("thstrm_amount") or "").strip().replace(",", "")
                    if not raw_val or raw_val == "-":
                        raw_val = str(rec.get("thstrm_add_amount") or "").strip().replace(",", "")
                    if not raw_val or raw_val == "-":
                        continue

                    try:
                        val = float(raw_val)
                    except ValueError:
                        continue

                    rcept_no = str(rec.get("rcept_no") or "").strip()
                    pub_dt = str(rec.get("rcept_dt") or "").strip()
                    if len(pub_dt) == 8:
                        published_at = datetime(int(pub_dt[:4]), int(pub_dt[4:6]), int(pub_dt[6:]), 18, 0, tzinfo=UTC)
                    elif len(rcept_no) >= 8 and rcept_no[:8].isdigit():
                        published_at = datetime(int(rcept_no[:4]), int(rcept_no[4:6]), int(rcept_no[6:8]), 18, 0, tzinfo=UTC)
                    else:
                        published_at = datetime(year, 12, 31, 18, 0, tzinfo=UTC)

                    year_rows.append({
                        "company_id": ticker,
                        "dart_corp_code": corp_code,
                        "ticker": ticker,
                        "fiscal_period": fiscal_period,
                        "filing_id": rcept_no or f"{corp_code}-{fiscal_period}",
                        "fact": fact_name,
                        "published_at": published_at,
                        "available_at": published_at,
                        "value": val,
                        "unit": "KRW",
                        "consolidated": (fs_div == "CFS"),
                        "restatement_id": "r0",
                        "source_hash": digest,
                        "source_kind": "opendart_multi_account",
                        "mapping_version": "dart-multi-v1",
                        "raw_document_hash": digest,
                    })
                    period_rows += 1

                time.sleep(0.12)  # safe pace (~8 req/sec)

            logger.info("Period %s completed: +%d fact rows", fiscal_period, period_rows)

        # Immediately save this year to Silver
        if year_rows:
            year_df = pl.DataFrame(year_rows).unique(subset=["ticker", "fiscal_period", "fact", "consolidated"])
            out_path = silver_dir / f"year={year}" / f"facts_{year}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            year_df.write_parquet(out_path, compression="zstd")
            logger.info(">>> Successfully saved year %d: %d rows -> %s (elapsed: %.1fs, total calls: %d)",
                        year, year_df.height, out_path, time.time() - start_time, total_calls)

    logger.info("=== All requested financial facts backfill successfully finished! Total calls: %d ===", total_calls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()
    run_financial_backfill(years=args.years, force=args.force)
