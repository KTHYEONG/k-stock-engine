"""High-speed dual-broker (Kiwoom + LS) parallel investor-flow backfill engine (2016-2026).

Architecture:
- Shared Priority Queue: Tickers ordered by 2026 market-cap/turnover descending.
- Worker 1 (Kiwoom ka10059): Fast ~5 req/s (0.2s interval), ~3-5s per ticker.
- Worker 2 (LS t1702): Bulk 750 sessions/req (1.05s interval), ~16-18s per ticker.
- Writer Thread: Batches Silver parquet updates every 20 tickers or 30 seconds with zstd.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import queue
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv

from src.integrations.kiwoom.client import KiwoomClient, KiwoomCredentials
from src.integrations.ls.client import LsClient, LsCredentials

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
logger = logging.getLogger("investor_flow_parallel")

MIN_DATE = date(2016, 1, 1)
MAX_DATE = date(2026, 3, 10)

LS_DATE_RANGES = [
    (date(2016, 1, 1), date(2018, 12, 31)),
    (date(2019, 1, 1), date(2021, 12, 31)),
    (date(2022, 1, 1), date(2024, 12, 31)),
    (date(2025, 1, 1), date(2026, 3, 10)),
]


def load_universe_ranked(silver_root: Path) -> list[str]:
    """Load all universe tickers ranked by recent trading turnover and market cap."""
    pfiles = list((silver_root / "security_master").glob("**/*.parquet"))
    if not pfiles:
        raise RuntimeError("No security_master parquet found")
    all_tickers_set: set[str] = set()
    for p in pfiles:
        try:
            t_col = pl.read_parquet(p, columns=["ticker"])["ticker"].drop_nulls().to_list()
            for t in t_col:
                s = str(t).strip()
                if len(s) == 6 and s.isdigit():
                    all_tickers_set.add(s)
        except Exception as exc:
            logger.debug("Failed reading security master file %s: %s", p, exc)
            continue

    ranking: dict[str, float] = {}
    dm_files = sorted((silver_root / "daily_market").glob("**/year=2026/**/*.parquet"))
    for p in dm_files:
        try:
            df = pl.read_parquet(p, columns=["instrument_id", "trading_value", "market_cap"])
            agg = df.group_by("instrument_id").agg([
                pl.col("trading_value").sum().alias("tv_sum"),
                pl.col("market_cap").max().alias("mc_max"),
            ])
            for r in agg.iter_rows(named=True):
                inst = str(r["instrument_id"]).replace("KRX:", "").strip()
                score = float(r.get("tv_sum") or 0.0) + float(r.get("mc_max") or 0.0)
                ranking[inst] = max(ranking.get(inst, 0.0), score)
        except Exception as exc:
            logger.debug("Failed reading daily market file %s: %s", p, exc)
            continue

    ranked_tickers = sorted(all_tickers_set, key=lambda t: ranking.get(t, 0.0), reverse=True)
    return ranked_tickers


def load_already_collected_tickers(silver_root: Path) -> set[str]:
    flow_dir = silver_root / "investor_flow"
    if not flow_dir.exists():
        return set()
    pfiles = list(flow_dir.glob("**/flow_2024.parquet"))
    collected: set[str] = set()
    for p in pfiles:
        try:
            df = pl.read_parquet(p, columns=["instrument_id"])
            for inst in df["instrument_id"].drop_nulls().unique().to_list():
                raw = str(inst).replace("KRX:", "").strip()
                if raw:
                    collected.add(raw)
        except Exception as exc:
            logger.debug("Failed reading flow file %s: %s", p, exc)
            continue
    return collected


def collect_ticker_kiwoom(client: KiwoomClient, ticker: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur_dt = MAX_DATE

    for _ in range(35):
        try:
            raw_batch = client.inquire_investor_trend(ticker, cur_dt)
        except Exception as exc:
            logger.warning("Kiwoom network error for %s on %s: %s", ticker, cur_dt, exc)
            time.sleep(0.5)
            break

        if not raw_batch:
            break

        for r in raw_batch:
            dt_str = str(r.get("dt", "")).strip()
            if not dt_str:
                continue
            try:
                d_obj = datetime.strptime(dt_str, "%Y%m%d").replace(tzinfo=UTC)
            except Exception as exc:
                logger.debug("Failed parsing date %s: %s", dt_str, exc)
                continue
            if d_obj.date() < MIN_DATE:
                continue

            f_net = float(str(r.get("frgnr_invsr", 0)).replace(",", "")) * 1_000_000.0
            inst_net = float(str(r.get("orgn", 0)).replace(",", "")) * 1_000_000.0
            ret_net = float(str(r.get("ind_invsr", 0)).replace(",", "")) * 1_000_000.0
            rows.append({
                "session": d_obj,
                "instrument_id": f"KRX:{ticker}",
                "foreign_buy_value": max(f_net, 0.0),
                "foreign_sell_value": max(-f_net, 0.0),
                "foreign_net_value": f_net,
                "institution_net_value": inst_net,
                "retail_net_value": ret_net,
                "available_at": d_obj,
                "source_hash": hashlib.sha256(f"{ticker}:{dt_str}".encode()).hexdigest(),
            })

        oldest_dt_str = str(raw_batch[-1].get("dt", "")).strip()
        if not oldest_dt_str:
            break
        try:
            oldest_dt = datetime.strptime(oldest_dt_str, "%Y%m%d").date()
        except Exception as exc:
            logger.debug("Failed parsing oldest date %s: %s", oldest_dt_str, exc)
            break

        if oldest_dt <= MIN_DATE or oldest_dt >= cur_dt:
            break
        cur_dt = oldest_dt - timedelta(days=1)
        time.sleep(0.20)

    return rows


def collect_ticker_ls(client: LsClient, ticker: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for s_dt, e_dt in LS_DATE_RANGES:
        try:
            raw_batch = client.inquire_investor_trend(ticker, s_dt, e_dt)
        except Exception as exc:
            logger.warning("LS network error for %s (%s~%s): %s", ticker, s_dt, e_dt, exc)
            time.sleep(1.05)
            continue

        for r in raw_batch:
            raw_sess = str(r.get("date", "")).strip()
            if not raw_sess:
                continue
            try:
                dt_obj = datetime.strptime(raw_sess, "%Y%m%d").replace(tzinfo=UTC)
            except Exception as exc:
                logger.debug("Failed parsing LS date %s: %s", raw_sess, exc)
                continue
            if not (MIN_DATE <= dt_obj.date() <= MAX_DATE):
                continue

            try:
                f_buy = float(str(r.get("for_buy", 0)).replace(",", ""))
                f_sell = float(str(r.get("for_sell", 0)).replace(",", ""))
                f_net = float(str(r.get("for_net", 0)).replace(",", ""))
                inst_net = float(str(r.get("org_net", 0)).replace(",", ""))
                ret_net = float(str(r.get("ind_net", 0)).replace(",", ""))
            except Exception as exc:
                logger.debug("Failed parsing values in row %s: %s", r, exc)
                continue

            rows.append({
                "session": dt_obj,
                "instrument_id": f"KRX:{ticker}",
                "foreign_buy_value": f_buy,
                "foreign_sell_value": f_sell,
                "foreign_net_value": f_net,
                "institution_net_value": inst_net,
                "retail_net_value": ret_net,
                "available_at": dt_obj,
                "source_hash": hashlib.sha256(f"{ticker}:{raw_sess}".encode()).hexdigest(),
            })

        time.sleep(1.05)

    return rows


def writer_thread_func(
    result_queue: queue.Queue[tuple[str, list[dict[str, Any]]] | None],
    silver_root: Path,
    total_count: int,
    stop_event: threading.Event,
) -> None:
    flow_silver_dir = silver_root / "investor_flow"
    flow_silver_dir.mkdir(parents=True, exist_ok=True)

    buffer_rows: list[dict[str, Any]] = []
    completed_tickers = 0
    last_flush_time = time.time()
    start_time = time.time()

    def flush() -> None:
        nonlocal buffer_rows
        if not buffer_rows:
            return
        df = pl.DataFrame(buffer_rows)
        years = df["session"].dt.year().unique().to_list()
        for y in years:
            sub = df.filter(df["session"].dt.year() == y)
            out_path = flow_silver_dir / f"year={y}" / f"flow_{y}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                try:
                    old = pl.read_parquet(out_path)
                    merged = pl.concat([old, sub]).unique(subset=["session", "instrument_id"])
                except Exception as exc:
                    logger.warning("Failed merging with existing parquet %s: %s", out_path, exc)
                    merged = sub.unique(subset=["session", "instrument_id"])
            else:
                merged = sub.unique(subset=["session", "instrument_id"])
            merged.write_parquet(out_path, compression="zstd")
        flushed_count = len(buffer_rows)
        buffer_rows = []
        elapsed = time.time() - start_time
        rate = completed_tickers / max(1.0, elapsed) * 60.0
        rem_sec = (total_count - completed_tickers) / max(0.01, completed_tickers / max(1.0, elapsed))
        logger.info(
            ">>> [CHECKPOINT] Flushed %d rows to Silver. Completed %d/%d (%.1f%%, %.1f tkr/min, est. left: %.1f min)",
            flushed_count, completed_tickers, total_count, (completed_tickers / max(1, total_count)) * 100, rate, rem_sec / 60.0
        )

    while not stop_event.is_set() or not result_queue.empty():
        try:
            item = result_queue.get(timeout=1.0)
        except queue.Empty:
            if time.time() - last_flush_time >= 30.0 and buffer_rows:
                flush()
                last_flush_time = time.time()
            continue

        if item is None:
            break

        _ticker, rows = item
        completed_tickers += 1
        buffer_rows.extend(rows)

        if completed_tickers % 20 == 0 or (time.time() - last_flush_time >= 30.0):
            flush()
            last_flush_time = time.time()

    flush()
    logger.info(">>> Writer thread gracefully finished all flushes.")


def kiwoom_worker_func(
    work_queue: queue.Queue[str],
    result_queue: queue.Queue[tuple[str, list[dict[str, Any]]] | None],
    stop_event: threading.Event,
) -> None:
    try:
        client = KiwoomClient(KiwoomCredentials.from_env())
        client.ensure_token()
        logger.info("Kiwoom worker initialized successfully.")
    except Exception as exc:
        logger.error("Kiwoom worker init failed: %s", exc)
        return

    while not stop_event.is_set():
        try:
            ticker = work_queue.get_nowait()
        except queue.Empty:
            break

        t0 = time.time()
        rows = collect_ticker_kiwoom(client, ticker)
        dur = time.time() - t0
        logger.info("[Kiwoom] %s finished: %d rows (%.2fs)", ticker, len(rows), dur)
        result_queue.put((ticker, rows))
        work_queue.task_done()


def ls_worker_func(
    work_queue: queue.Queue[str],
    result_queue: queue.Queue[tuple[str, list[dict[str, Any]]] | None],
    stop_event: threading.Event,
) -> None:
    try:
        client = LsClient(LsCredentials.from_env())
        client.ensure_token()
        logger.info("LS worker initialized successfully.")
    except Exception as exc:
        logger.error("LS worker init failed: %s", exc)
        return

    while not stop_event.is_set():
        try:
            ticker = work_queue.get_nowait()
        except queue.Empty:
            break

        t0 = time.time()
        rows = collect_ticker_ls(client, ticker)
        dur = time.time() - t0
        logger.info("[LS] %s finished: %d rows (%.2fs)", ticker, len(rows), dur)
        result_queue.put((ticker, rows))
        work_queue.task_done()


def run_parallel_backfill(
    max_symbols: int | None = None,
    silver_root: Path = Path("data/silver/stocks"),
) -> None:
    all_ranked = load_universe_ranked(silver_root)
    already_done = load_already_collected_tickers(silver_root)
    logger.info("Total universe tickers: %d", len(all_ranked))
    logger.info("Already completed tickers: %d", len(already_done))

    pending = [t for t in all_ranked if t not in already_done]
    target = pending[:max_symbols] if max_symbols else pending
    logger.info("Pending tickers to collect: %d", len(target))

    if not target:
        logger.info("All target tickers are already collected!")
        return

    work_queue: queue.Queue[str] = queue.Queue()
    for t in target:
        work_queue.put(t)

    result_queue: queue.Queue[tuple[str, list[dict[str, Any]]] | None] = queue.Queue()
    stop_event = threading.Event()

    writer_thread = threading.Thread(
        target=writer_thread_func,
        args=(result_queue, silver_root, len(target), stop_event),
        name="Thread-Writer",
        daemon=True,
    )
    writer_thread.start()

    workers = [
        threading.Thread(target=kiwoom_worker_func, args=(work_queue, result_queue, stop_event), name="Thread-Kiwoom"),
        threading.Thread(target=ls_worker_func, args=(work_queue, result_queue, stop_event), name="Thread-LS"),
    ]

    for w in workers:
        w.start()

    try:
        work_queue.join()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, stopping gracefully...")
        stop_event.set()

    stop_event.set()
    for w in workers:
        w.join()

    result_queue.put(None)
    writer_thread.join()
    logger.info("=== All investor-flow backfill tasks completed successfully! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Broker Parallel Investor Flow Backfill")
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit number of symbols to collect")
    args = parser.parse_args()

    run_parallel_backfill(max_symbols=args.max_symbols)
