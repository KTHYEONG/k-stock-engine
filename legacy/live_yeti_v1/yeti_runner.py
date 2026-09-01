import argparse
import signal
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.integrations.kis.client import KisClient, KisCredentials
from legacy.live_yeti_v1.yeti_strategy import YetiLiveStrategy
from legacy.live_yeti_v1.yeti_trader import ExecutionConfig, YetiLiveTrader
from legacy.live_yeti_v1.yeti_state import StateManager
from legacy.stock_yetirank_v1.utils.logger import setup_logger

logger = setup_logger("execution.run_yetirank_live_bot")
_RUNNING = True

def graceful_shutdown(signum, frame):
    global _RUNNING
    logger.info("Received terminate signal... gracefully shutting down.")
    _RUNNING = False

def run_data_pipeline(low_spec: bool = False):
    logger.info("🚀 Starting automated data pipeline...")
    try:
        # Collect & engineer data up to today (skipped if already exists)
        # Using a 30-day buffer to ensure recent dates are fully populated
        today_str = datetime.now().strftime("%Y%m%d")
        start_str = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        
        logger.info(f"Running collect_data.py for {start_str} ~ {today_str}...")
        collect_cmd = [sys.executable, "-m", "legacy.stock_yetirank_v1.data.collectors.collect_data", "--start", start_str, "--end", today_str]
        if low_spec:
            collect_cmd.append("--low-spec")
            logger.info("Using low-spec (serial) collection mode.")
        
        subprocess.run(collect_cmd, check=True)
        
        logger.info(f"Running feature_engineer.py for {start_str} ~ {today_str}...")
        subprocess.run(
            [sys.executable, "-m", "legacy.stock_yetirank_v1.data.feature_engineer", "--start", start_str, "--end", today_str],
            check=True
        )
        logger.info("✅ Automated data pipeline completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Data pipeline failed: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error in data pipeline: {e}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YetiRank live auto-trading bot (KIS API)")
    
    import dateutil.relativedelta
    one_year_ago = (datetime.now() - dateutil.relativedelta.relativedelta(years=1)).strftime("%Y%m%d")
    
    parser.add_argument("--start", type=str, default=one_year_ago, help="Signal data start date (YYYYMMDD). Must be at least 6 months ago for indicators.")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y%m%d"), help="Signal data end date (YYYYMMDD)")
    parser.add_argument("--model-id", type=str, default="latest", help="Model ID or 'latest'")
    parser.add_argument("--env", type=str, default="real", choices=["real", "demo"])

    parser.add_argument("--once", action="store_true", help="Run only once immediately")
    parser.add_argument("--signal-date", type=str, default=None, help="Force signal date (YYYYMMDD)")

    parser.add_argument("--data-time", type=str, default="07:00", help="KST run time for data collection")
    parser.add_argument("--run-time", type=str, default="08:50", help="KST run time in loop mode")
    parser.add_argument("--interval-sec", type=int, default=30, help="Loop polling interval seconds")
    parser.add_argument("--low-spec", action="store_true", help="Run data collection sequentially to save memory on 1GB instances")

    parser.add_argument("--live", action="store_true", help="Send real orders. Default is dry-run")
    return parser.parse_args()


def parse_hhmm(hhmm: str) -> tuple[int, int]:
    hh, mm = hhmm.split(":", 1)
    return int(hh), int(mm)


def main() -> None:
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    load_dotenv()
    args = parse_args()

    creds = KisCredentials.from_env(env=args.env)
    broker = KisClient(creds)

    strategy = YetiLiveStrategy(
        start_date=args.start,
        end_date=args.end,
        model_id=args.model_id,
    )
    state = StateManager()
    exec_cfg = ExecutionConfig.from_env(dry_run_override=(not args.live))

    trader = YetiLiveTrader(
        broker=broker,
        strategy=strategy,
        state=state,
        config=exec_cfg,
    )

    if args.once:
        run_data_pipeline(low_spec=args.low_spec)
        plan = trader.run_once(signal_date=args.signal_date)
        logger.info(
            "Single run complete. signal_date=%s dry_run=%s targets=%d",
            plan.signal_date,
            exec_cfg.dry_run,
            len(plan.target_weights),
        )
        return

    data_hh, data_mm = parse_hhmm(args.data_time)
    run_hh, run_mm = parse_hhmm(args.run_time)
    kst = ZoneInfo("Asia/Seoul")
    
    last_data_date = None
    last_run_date = None

    logger.info(
        "Loop started. env=%s dry_run=%s data_time=%s run_time=%s interval=%ss low_spec=%s",
        args.env,
        exec_cfg.dry_run,
        args.data_time,
        args.run_time,
        args.interval_sec,
        args.low_spec,
    )

    while _RUNNING:
        now = datetime.now(tz=kst)
        today = now.date()
        is_weekday = now.weekday() < 5
        
        # 1. Execute Data Pipeline
        data_ready = (now.hour, now.minute) >= (data_hh, data_mm)
        if is_weekday and data_ready and last_data_date != today:
            logger.info("Time to update data. Starting pipeline...")
            run_data_pipeline(low_spec=args.low_spec)
            last_data_date = today
            
            # If we just updated data, force backtester to clear cache for the fresh run
            strategy.backtester._cached_predictions = None

        # 2. Execute Trading
        trade_ready = (now.hour, now.minute) >= (run_hh, run_mm)
        if is_weekday and trade_ready and last_run_date != today:
            try:
                # Re-initialize strategy/cache to pick up newly downloaded data if memory is stale
                if strategy.backtester._cached_predictions is None or getattr(strategy.backtester, "_cached_predictions", None) is None:
                    strategy = YetiLiveStrategy(start_date=args.start, end_date=args.end, model_id=args.model_id)
                    trader.strategy = strategy
                
                plan = trader.run_once(signal_date=args.signal_date)
                last_run_date = today
                logger.info(
                    "Daily run complete. signal_date=%s exposure=%.2f targets=%d",
                    plan.signal_date,
                    plan.target_exposure,
                    len(plan.target_weights),
                )
            except Exception as exc:
                logger.exception("Daily run failed: %s", exc)

        # Non-blocking sleep for graceful shutdown
        for _ in range(max(args.interval_sec, 5)):
            if not _RUNNING: 
                break
            time.sleep(1)

    logger.info("Bot stopped cleanly.")

if __name__ == "__main__":
    main()
