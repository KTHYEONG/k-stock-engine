"""Fast LS OpenAPI investor flow collector (2016-2026) with resumability."""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import polars as pl

from src.integrations.ls.investor_flow import LsInvestorFlowCollector
from src.integrations.ls.client import LsClient, LsCredentials

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("investor_flow_backfill")

DATE_RANGES = [
    (date(2016, 1, 1), date(2018, 12, 31)),
    (date(2019, 1, 1), date(2021, 12, 31)),
    (date(2022, 1, 1), date(2024, 12, 31)),
    (date(2025, 1, 1), date(2026, 3, 10)),
]


def load_universe_tickers(silver_root: Path) -> list[str]:
    pfiles = list((silver_root / "security_master").glob("**/*.parquet"))
    if not pfiles:
        raise RuntimeError("No security_master parquet found")
    all_tickers: list[str] = []
    for p in pfiles:
        try:
            t_col = pl.read_parquet(p, columns=["ticker"])["ticker"].drop_nulls().to_list()
            all_tickers.extend(t_col)
        except Exception:
            continue
    clean = sorted(dict.fromkeys(t.strip() for t in all_tickers if len(t.strip()) == 6 and t.strip().isdigit()))
    return clean


def load_already_collected_tickers(silver_root: Path) -> set[str]:
    flow_dir = silver_root / "investor_flow"
    if not flow_dir.exists():
        return set()
    # Check 2024 flow file as a representative indicator
    pfiles = list(flow_dir.glob("**/flow_2024.parquet"))
    collected: set[str] = set()
    for p in pfiles:
        try:
            df = pl.read_parquet(p, columns=["instrument_id"])
            for inst in df["instrument_id"].drop_nulls().unique().to_list():
                raw = str(inst).replace("KRX:", "").strip()
                if raw:
                    collected.add(raw)
        except Exception:
            continue
    return collected


def run_investor_flow_backfill(
    max_symbols: int | None = None,
    force: bool = False,
    silver_root: Path = Path("data/silver/stocks"),
    bronze_root: Path = Path("data/bronze/stocks"),
) -> None:
    client = LsClient(LsCredentials.from_env())
    all_tickers = load_universe_tickers(silver_root)
    logger.info("Total universe tickers: %d", len(all_tickers))

    already_done = set() if force else load_already_collected_tickers(silver_root)
    logger.info("Already completed tickers: %d", len(already_done))

    pending_tickers = [t for t in all_tickers if t not in already_done]
    target_tickers = pending_tickers[:max_symbols] if max_symbols else pending_tickers
    logger.info("Pending tickers to collect: %d", len(target_tickers))

    if not target_tickers:
        logger.info("All universe tickers are already collected!")
        return

    collector = LsInvestorFlowCollector(tuple(target_tickers), client=client)
    flow_silver_dir = silver_root / "investor_flow"
    flow_silver_dir.mkdir(parents=True, exist_ok=True)

    all_silver_rows: list[dict[str, Any]] = []
    total_calls = 0
    start_time = time.time()
    total_tickers = len(target_tickers)

    for idx, ticker in enumerate(target_tickers):
        ticker_start = time.time()
        ticker_rows = 0

        for s_dt, e_dt in DATE_RANGES:
            total_calls += 1
            try:
                pages = list(collector.fetch_investor_flow(s_dt, e_dt, bronze_root=bronze_root, symbols=(ticker,)))
            except Exception as exc:
                logger.warning("Error fetching %s (%s~%s): %s", ticker, s_dt, e_dt, exc)
                time.sleep(1.05)
                continue

            for page in pages:
                records = page.get("records") or []
                for r in records:
                    sess_str = str(r["session"])
                    dt_obj = datetime.fromisoformat(sess_str).replace(tzinfo=UTC)
                    all_silver_rows.append({
                        "session": dt_obj,
                        "instrument_id": f"KRX:{ticker}",
                        "foreign_buy_value": float(r["foreign_buy_value"]),
                        "foreign_sell_value": float(r["foreign_sell_value"]),
                        "foreign_net_value": float(r["foreign_net_value"]),
                        "institution_net_value": float(r["institution_net_value"]),
                        "retail_net_value": float(r["retail_net_value"]),
                        "available_at": dt_obj,
                        "source_hash": hashlib.sha256(f"{ticker}:{sess_str}".encode()).hexdigest(),
                    })
                    ticker_rows += 1

            time.sleep(1.05)  # LS rate limiter (1 req/sec)

        elapsed = time.time() - start_time
        sec_per_ticker = elapsed / max(1, idx + 1)
        remaining_sec = (total_tickers - (idx + 1)) * sec_per_ticker

        logger.info(
            "[%d/%d] Ticker %s completed: +%d rows (elapsed: %.1fs, pace: %.1fs/tkr, est. left: %.1fm)",
            idx + 1, total_tickers, ticker, ticker_rows, elapsed, sec_per_ticker, remaining_sec / 60.0
        )

        # Batch save to Silver every 25 tickers or at end
        if (idx + 1) % 25 == 0 or (idx + 1) == total_tickers:
            if all_silver_rows:
                df = pl.DataFrame(all_silver_rows)
                years = df["session"].dt.year().unique().to_list()
                for y in years:
                    sub = df.filter(df["session"].dt.year() == y)
                    out_path = flow_silver_dir / f"year={y}" / f"flow_{y}.parquet"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if out_path.exists():
                        old = pl.read_parquet(out_path)
                        merged = pl.concat([old, sub]).unique(subset=["session", "instrument_id"])
                        merged.write_parquet(out_path, compression="zstd")
                    else:
                        sub.unique(subset=["session", "instrument_id"]).write_parquet(out_path, compression="zstd")
                logger.info(">>> Checkpoint saved to Silver up to ticker %d (flushed %d rows)", idx + 1, len(all_silver_rows))
                all_silver_rows.clear()

    logger.info("=== All investor flow backfill completed! Total calls: %d, Elapsed: %.1fs ===", total_calls, time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()
    run_investor_flow_backfill(max_symbols=args.max_symbols, force=args.force)
