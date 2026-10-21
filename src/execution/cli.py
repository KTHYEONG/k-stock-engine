"""Execution CLI: pre-open auction capture for the shadow ledger."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypedDict

if TYPE_CHECKING:
    from src.execution.auction_capture import AuctionRow, CaptureBatch
    from src.integrations.kis.client import KisCredentials

__all__ = ["main"]

_LOG = logging.getLogger(__name__)

_DEFAULT_SLOT_MARGIN_S: Final = 0.0
# Share of captured symbols with no open print beyond which the open counts as delayed, not a few untraded names.
_UNRESOLVED_MAX_RATIO: Final = 0.2


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
    capture.add_argument("--slot-margin-s", type=float, required=False, default=_DEFAULT_SLOT_MARGIN_S)
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


class _AuctionMetrics(TypedDict):
    symbols: int
    expected_rows: int
    open_resolved: int
    unresolved: list[str]
    failed: dict[str, str]
    slots_expected: int
    slots_captured: int
    persist_error: str | None


class _AuctionPersister(Protocol):
    persist_error: str | None

    def on_batch(self, batch: CaptureBatch) -> None: ...

    def finalize(self, rows: Sequence[AuctionRow]) -> None: ...


def _make_persister(out: Path) -> _AuctionPersister:
    from src.execution.auction_capture import write_auction_rows

    class _Persister:
        def __init__(self) -> None:
            self.persist_error: str | None = None

        def on_batch(self, batch: CaptureBatch) -> None:
            self.finalize(batch.rows)

        def finalize(self, rows: Sequence[AuctionRow]) -> None:
            if not rows:
                return
            try:
                write_auction_rows(out, rows)
            except Exception as exc:
                if self.persist_error is None:
                    self.persist_error = type(exc).__name__
                _LOG.error("[EXEC] stage=auction_persist status=failed error=%s", type(exc).__name__, exc_info=True)

    return _Persister()


def _empty_auction_result() -> _AuctionMetrics:
    return {
        "symbols": 0,
        "expected_rows": 0,
        "open_resolved": 0,
        "unresolved": [],
        "failed": {},
        "slots_expected": 0,
        "slots_captured": 0,
        "persist_error": None,
    }


def _run_auction_open(args: argparse.Namespace) -> _AuctionMetrics:
    from datetime import datetime as _datetime

    from src.core.time import KRX_TZ
    from src.execution.auction_capture import (
        AuctionCapturePolicy,
        capture_open_auction,
        capture_symbols,
    )
    from src.research.operate import read_chain

    limit = int(getattr(args, "limit", 80))
    margin = float(getattr(args, "slot_margin_s", _DEFAULT_SLOT_MARGIN_S))
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("slot_margin_s must be finite and >= 0")
    chain = read_chain(Path(args.decisions_dir))
    if not chain:
        return _empty_auction_result()
    session = _today()
    open_at = _session_open(session)
    if open_at is None:
        _LOG.info("[EXEC] auction capture skipped: %s is not a regular 09:00 session", session.isoformat())
        return _empty_auction_result()
    holdings = _net_holdings(Path(args.shadow_dir))
    symbols = capture_symbols(chain, holdings, session=session, limit=limit)
    if not symbols:
        return _empty_auction_result()
    from src.config.providers import load_provider_policy
    from src.config.runtime import load_runtime_config
    from src.integrations.kis import build_pool_client
    from src.integrations.kis.admission import AdmissionClass

    policy = load_provider_policy(load_runtime_config()).kis
    capture_policy = AuctionCapturePolicy(
        call_estimate_s=float(policy.min_interval_seconds),
        slot_margin_s=margin,
    )
    client = build_pool_client(policy, admission_class=AdmissionClass.STANDARD)
    persist = _make_persister(Path(args.out))
    result = capture_open_auction(
        client,
        symbols,
        session=session,
        open_at=open_at,
        now=lambda: _datetime.now(KRX_TZ),
        sleep=time.sleep,
        policy=capture_policy,
        on_batch=persist.on_batch,
    )
    persist.finalize(result.rows)
    opened = {row.symbol for row in result.rows if row.kind == "open"}
    expected = sum(1 for row in result.rows if row.kind == "expected")
    return {
        "symbols": len(symbols),
        "expected_rows": expected,
        "open_resolved": len(opened),
        "unresolved": sorted(set(symbols) - opened),
        "failed": dict(sorted(result.failed.items())),
        "slots_expected": result.slots_expected,
        "slots_captured": result.slots_captured,
        "persist_error": persist.persist_error,
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


_DECISION_POLL_S = 15.0


def _now() -> datetime:
    from src.core.time import KRX_TZ

    return datetime.now(KRX_TZ)


def _await_execution_decision(decisions_dir: Path, session: date, deadline: datetime) -> bool:
    """Poll the sealed chain until it holds a decision for ``session`` or ``deadline`` passes."""
    from src.research.operate import read_chain

    while True:
        if any(item.execution_session == session for item in read_chain(decisions_dir)):
            return True
        if _now() >= deadline:
            return False
        time.sleep(_DECISION_POLL_S)


def _scheduled_capture(args: argparse.Namespace) -> int:
    from src.core.time import KRX_TZ, OPEN_SNAPSHOT_OFFSETS_MIN
    from src.ops.guards import trading_day
    from src.ops.run_outcome import record_run_outcome

    today = _today()

    def record(outcome: str, reason: str = "", metrics: Mapping[str, object] | None = None) -> None:
        record_run_outcome(args.events_root, "auction_open", outcome, run_date=today,
                           now=datetime.now(KRX_TZ), reason=reason, metrics=metrics)

    if not trading_day(today):
        record("SKIPPED", "non_trading_day")
        return 0
    try:
        open_at = _session_open(today)
        if open_at is None:
            record("SKIPPED", "non_regular_open")
            return 0
        last_snapshot = open_at + timedelta(minutes=max(OPEN_SNAPSHOT_OFFSETS_MIN))
        if not _await_execution_decision(args.decisions_dir, today, last_snapshot):
            record("NO_DECISION", "missing_execution_decision")
            return 1
        result = _run_auction_open(args)
        reasons: list[str] = []
        if result["slots_captured"] < result["slots_expected"]:
            reasons.append("slots_incomplete")
        if result.get("failed"):
            reasons.append("symbol_failed")
        if result.get("persist_error") is not None:
            reasons.append("persist_failed")
        unexplained = set(result.get("unresolved", ())) - set(result.get("failed", {}))
        if len(unexplained) > _UNRESOLVED_MAX_RATIO * result.get("symbols", 0):
            reasons.append("unresolved_auction")
        if reasons:
            record("DEGRADED", ",".join(reasons), result)
            return 1
        record("OK", metrics=result)
        _emit(result)
        return 0
    except Exception as exc:
        record("DEGRADED", type(exc).__name__)
        return 1
