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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.integrations.kis.client import KisCredentials

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
    rehearse = sub.add_parser("rehearse")
    rehearse.add_argument("--decisions-dir", type=Path, required=True)
    rehearse.add_argument("--store-path", type=Path, required=True)
    rehearse.add_argument("--evidence-path", type=Path, required=True)
    rehearse.add_argument("--nav-krw", type=int, required=True)
    rehearse.add_argument("--cash-buffer", type=float, default=0.02)
    rehearse.add_argument("--rebalance-band", type=float, default=0.0)
    rehearse.add_argument("--session", type=str, default=None)
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
    if getattr(args, "events_root", None) is not None:
        return _scheduled_capture(args)
    try:
        if args.command == "auction-open":
            _emit(_run_auction_open(args))
        elif args.command == "rehearse":
            _emit(_run_rehearse(args))
    except (PITDataError, ValueError, OSError) as exc:
        _LOG.error("[EXEC] command=%s status=failed error=%s", args.command, exc)
        _emit({"error": str(exc)})
        return 1
    return 0


def _mock_credentials() -> KisCredentials:
    from src.config.secrets import read_secret
    from src.integrations.kis.client import KisCredentials

    app_key = read_secret("KIS_MOCK_APP_KEY", "").strip()
    app_secret = read_secret("KIS_MOCK_APP_SECRET", "").strip()
    account_raw = read_secret("KIS_MOCK_ACCOUNT_NO", "").strip()
    env = read_secret("KIS_ENV", "vps").strip() or "vps"
    if not app_key or not app_secret or not account_raw:
        raise ValueError("mock keys are required: KIS_MOCK_APP_KEY/SECRET/ACCOUNT_NO")
    product = "01"
    account_no = account_raw
    if "-" in account_raw:
        account_no, product = account_raw.split("-", 1)
    return KisCredentials(
        app_key=app_key,
        app_secret=app_secret,
        account_no=account_no,
        account_product_code=product or "01",
        env=env,
    )


def _previous_session(session: date) -> date | None:
    from src.core.krx_calendar import xkrx_calendar_through
    from src.core.time import KRX_TZ

    earlier = [
        day
        for day in (item.astimezone(KRX_TZ).date() for item in xkrx_calendar_through(session).sessions)
        if day < session
    ]
    return max(earlier) if earlier else None


def _run_rehearse(args: argparse.Namespace) -> Mapping[str, object]:
    from datetime import datetime as _datetime

    from src.config.runtime import load_runtime_config
    from src.core.market_rules import KrxMarket, load_krx_market_rules
    from src.core.time import KRX_TZ
    from src.execution.adapters.kis_broker import KisMockBroker
    from src.execution.adapters.sqlite_state_store import SqliteStateStore
    from src.execution.application.rehearsal import (
        RehearsalReport,
        rehearse_open,
        update_readiness,
        verify_terminal_states,
    )
    from src.integrations.kis.client import KisClient
    from src.research.operate import read_chain

    chain = read_chain(Path(args.decisions_dir))
    if not chain:
        raise ValueError("no sealed decisions")
    if getattr(args, "session", None):
        session = date.fromisoformat(str(args.session))
    else:
        session = _today()
    matches = [item for item in chain if item.execution_session == session]
    if not matches:
        raise ValueError(f"no decision for execution session {session.isoformat()}")
    decision = max(matches, key=lambda item: item.sealed_at)
    rules = load_krx_market_rules(load_runtime_config().market_rules)
    names = set(decision.targets) | set(decision.reference)
    markets = dict.fromkeys(names, KrxMarket.KOSPI)
    client = KisClient(_mock_credentials())
    broker = KisMockBroker(client, rules=rules, markets=markets, session=session)
    store = SqliteStateStore(Path(args.store_path))
    try:
        report = rehearse_open(
            decision,
            broker,
            store,
            nav_krw=int(args.nav_krw),
            cash_buffer=float(args.cash_buffer),
            rebalance_band=float(args.rebalance_band),
            now=_datetime.now(KRX_TZ),
        )
        previous = _previous_session(session)
        previous_terminal = previous is not None and verify_terminal_states(broker, store, previous)
    finally:
        store.close()
    evidence_path = Path(args.evidence_path)
    prior: list[RehearsalReport] = [report]
    try:
        import json as _json

        if evidence_path.is_file():
            payload = _json.loads(evidence_path.read_text(encoding="utf-8"))
            for row in payload.get("reports", []):
                if str(row.get("session")) == report.session.isoformat():
                    continue
                row_session = date.fromisoformat(str(row["session"]))
                prior.append(
                    RehearsalReport(
                        session=row_session,
                        intents=int(row["intents"]),
                        submitted=int(row["submitted"]),
                        rejected=int(row["rejected"]),
                        duplicates_blocked=int(row["duplicates_blocked"]),
                        reconciled=bool(row["reconciled"]),
                        position_mismatches=tuple(row.get("position_mismatches", [])),
                        terminal_states_verified=bool(row.get("terminal_states_verified", False))
                        or (row_session == previous and previous_terminal),
                        duplicate_orders_absent=bool(row.get("duplicate_orders_absent", False)),
                        position_book_verified=bool(row.get("position_book_verified", False)),
                    )
                )
    except (OSError, ValueError, KeyError):
        prior = [report]
    update_readiness(evidence_path, prior)
    return {
        "session": report.session.isoformat(),
        "intents": report.intents,
        "submitted": report.submitted,
        "rejected": report.rejected,
        "duplicates_blocked": report.duplicates_blocked,
        "reconciled": report.reconciled,
        "position_mismatches": list(report.position_mismatches),
    }


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
