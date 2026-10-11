"""Execution CLI: pre-open auction capture for the shadow ledger."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

__all__ = ["main"]

_LOG = logging.getLogger(__name__)


def _today() -> date:
    from src.core.time import KRX_TZ

    return datetime.now(KRX_TZ).date()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.execution")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("auction-open")
    capture.add_argument("--decisions-dir", type=Path, required=True)
    capture.add_argument("--shadow-dir", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--limit", type=int, required=False, default=80)
    capture.add_argument("--events-root", type=Path, default=None)
    return parser


def _emit(payload: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _session_open(session: object) -> datetime | None:
    from src.core.krx_calendar import xkrx_calendar_through
    from src.core.time import KRX_TZ

    try:
        calendar = xkrx_calendar_through(session)  # type: ignore[arg-type]
    except ValueError:
        return None
    for instant in calendar.sessions:
        local = instant.astimezone(KRX_TZ)
        if local.date() == session:
            if (local.hour, local.minute) != (9, 0):
                return None
            return local
    return None


def _net_holdings(shadow_dir: Path) -> tuple[str, ...]:
    import polars as pl

    fills_path = Path(shadow_dir) / "official" / "fills.parquet"
    if not fills_path.is_file():
        return ()
    frame = pl.read_parquet(fills_path)
    if frame.height == 0:
        return ()
    signed = frame.with_columns(
        pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(-pl.col("quantity")).alias("_signed")
    )
    net = signed.group_by("instrument_id").agg(pl.col("_signed").sum().alias("net"))
    return tuple(sorted(row for row in net.filter(pl.col("net") != 0)["instrument_id"].to_list()))


def _run_auction_open(args: argparse.Namespace) -> Mapping[str, object]:
    from datetime import datetime as _datetime

    from src.core.time import KRX_TZ
    from src.execution.auction_capture import capture_open_auction, capture_symbols, write_auction_rows
    from src.research.operate import read_chain

    limit = int(getattr(args, "limit", 80))
    chain = read_chain(Path(args.decisions_dir))
    if not chain:
        return {"symbols": 0, "expected_rows": 0, "open_resolved": 0, "unresolved": []}
    session = _today()
    open_at = _session_open(session)
    if open_at is None:
        _LOG.info("[EXEC] auction capture skipped: %s is not a regular 09:00 session", session.isoformat())
        return {"symbols": 0, "expected_rows": 0, "open_resolved": 0, "unresolved": []}
    holdings = _net_holdings(Path(args.shadow_dir))
    symbols = capture_symbols(chain, holdings, session=session, limit=limit)
    if not symbols:
        return {"symbols": 0, "expected_rows": 0, "open_resolved": 0, "unresolved": []}
    from src.config.providers import load_provider_policy
    from src.config.runtime import load_runtime_config
    from src.integrations.kis import build_pool_client
    from src.integrations.kis.admission import AdmissionClass

    policy = load_provider_policy(load_runtime_config()).kis
    client = build_pool_client(policy, admission_class=AdmissionClass.STANDARD)
    rows = capture_open_auction(
        client,
        symbols,
        session=session,
        open_at=open_at,
        now=lambda: _datetime.now(KRX_TZ),
        sleep=time.sleep,
    )
    if rows:
        write_auction_rows(Path(args.out), rows)
    opened = {row.symbol for row in rows if row.kind == "open"}
    expected = sum(1 for row in rows if row.kind == "expected")
    return {
        "symbols": len(symbols),
        "expected_rows": expected,
        "open_resolved": len(opened),
        "unresolved": sorted(set(symbols) - opened),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one execution command with a JSON-line envelope."""
    from src.core.pit import PITDataError

    args = _build_parser().parse_args(argv)
    if args.events_root is not None:
        return _scheduled_capture(args)
    try:
        if args.command == "auction-open":
            _emit(_run_auction_open(args))
    except (PITDataError, ValueError, OSError) as exc:
        _LOG.error("[EXEC] command=%s status=failed error=%s", args.command, exc)
        _emit({"error": str(exc)})
        return 1
    return 0


def _scheduled_capture(args: argparse.Namespace) -> int:
    from src.core.time import KRX_TZ
    from src.ops.guards import trading_day
    from src.ops.run_outcome import record_run_outcome
    from src.research.operate import read_chain

    today = _today()

    def record(outcome: str, reason: str = "", metrics: Mapping[str, object] | None = None) -> None:
        record_run_outcome(args.events_root, "auction_open", outcome, run_date=today,
                           now=datetime.now(KRX_TZ), reason=reason, metrics=metrics)

    if not trading_day(today):
        record("SKIPPED", "non_trading_day")
        return 0
    try:
        chain = read_chain(args.decisions_dir)
        if not any(item.execution_session == today for item in chain):
            record("NO_DECISION", "missing_execution_decision")
            return 1
        if _session_open(today) is None:
            record("SKIPPED", "non_regular_open")
            return 0
        result = _run_auction_open(args)
        if result.get("unresolved"):
            record("DEGRADED", "unresolved_auction", result)
            return 1
        record("OK", metrics=result)
        _emit(result)
        return 0
    except Exception as exc:
        record("DEGRADED", type(exc).__name__)
        return 1
