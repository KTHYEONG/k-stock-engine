"""Read-only pre-seal check: the morning seal's input gates, run at 07:30 so a failure is visible before 08:32."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src.core.time import KRX_TZ

if TYPE_CHECKING:
    from src.config.providers import KindPolicy
    from src.data.runtime import DataRuntime

__all__ = ["CheckResult", "check_flow_freshness", "check_kind_consistency", "check_kind_relay_age", "preseal_check"]

_LOG = logging.getLogger(__name__)

PASS: Final = "PASS"  # noqa: S105 - verdict token, not a credential
WARN: Final = "WARN"
FAIL: Final = "FAIL"
SKIPPED: Final = "SKIPPED"

SEAL_TIME: Final = time(8, 32)
FLOW_SEAL_TIME: Final = time(8, 50)
_CONTEXT: Final = "PRESEAL_CTX"
_KIND: Final = "KIND_CONSISTENCY"
_FLOW: Final = "FLOW_FRESHNESS"
_RELAY: Final = "KIND_RELAY_AGE"
_EVENT_JOB: Final = "preseal"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One verdict line: ``<ID> PASS|WARN|FAIL key=value ...``."""

    check: str
    verdict: str
    fields: tuple[tuple[str, str], ...] = ()

    def line(self) -> str:
        return " ".join([self.check, self.verdict, *(f"{key}={value}" for key, value in self.fields)])


def _token(text: str) -> str:
    """Collapse whitespace so a free-text detail stays one ``key=value`` token."""
    return "_".join(text.split())[:200]


def _at(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, tzinfo=KRX_TZ)


def _kst_sessions_through(day: date) -> list[date]:
    from src.core.krx_calendar import xkrx_calendar_through

    return sorted({moment.astimezone(KRX_TZ).date() for moment in xkrx_calendar_through(day).sessions})


def check_kind_consistency(bronze_root: Path) -> CheckResult:
    """Stream the KIND catalog through the seal's own iterator; any contradiction is a FAIL."""
    from src.core.pit import PITDataError
    from src.data.kind_notices import iter_kind_notices
    from src.data.receipt_catalog import ReceiptCatalog

    rows = 0
    try:
        for _row in iter_kind_notices(ReceiptCatalog(Path(bronze_root) / "catalog")):
            rows += 1
    except PITDataError as exc:
        return CheckResult(_KIND, FAIL, (("error", "PITDataError"), ("detail", _token(str(exc)))))
    return CheckResult(_KIND, PASS, (("rows", str(rows)),))


def check_flow_freshness(
    runtime: DataRuntime, *, sessions: Sequence[date], last_session: date, today: date
) -> CheckResult:
    """Investor-flow gate for the source sessions the next seal requires, judged on what Silver holds.

    ``sessions`` is the session calendar through the data session. Sources that postdate the newest
    Silver session are not built yet (normal at 07:30) and only WARN.
    """
    import polars as pl

    from src.config.live_gate import load_live_gate_policy
    from src.config.runtime import load_runtime_config
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import read_dataset
    from src.data.investor_flow_freshness import StaleInvestorFlowError, check_investor_flow_freshness
    from src.research.operate import required_source_sessions

    if last_session not in sessions:
        return CheckResult(_FLOW, FAIL, (("reason", "state_outside_calendar"),))
    sources = list(required_source_sessions(sessions, sessions.index(last_session) + 1))
    if not sources:
        return CheckResult(_FLOW, PASS, (("required", "0"),))
    flow_dir = runtime.workspace.silver_root / DatasetRegistry(runtime.workspace.state_root).require("investor_flow")
    try:
        newest = read_dataset(flow_dir, columns=("session",)).select(pl.col("session").max()).collect().item()
    except Exception as exc:
        return CheckResult(_FLOW, FAIL, (("reason", "flow_dataset_unreadable"), ("error", type(exc).__name__)))
    judged = [session for session in sources if newest is not None and session <= newest]
    unbuilt = [session for session in sources if session not in judged]
    if judged:
        try:
            check_investor_flow_freshness(
                runtime=runtime,
                flow_dataset_dir=flow_dir,
                source_sessions=judged,
                seal_time=_at(today, FLOW_SEAL_TIME),
                policy=load_live_gate_policy(load_runtime_config().live_gate),
            )
        except StaleInvestorFlowError as exc:
            missing = ",".join(
                f"{item.source_session.isoformat()}:{item.missing_fraction:.6f}"
                for item in exc.freshness
                if item.missing_cells > 0
            )
            return CheckResult(_FLOW, FAIL, (("missing", missing), ("required", str(len(sources)))))
    if unbuilt:
        return CheckResult(
            _FLOW, WARN,
            (("reason", "silver_not_built"), ("unbuilt", str(len(unbuilt))), ("judged", str(len(judged)))),
        )
    return CheckResult(_FLOW, PASS, (("judged", str(len(judged))),))


def check_kind_relay_age(state_root: Path, *, policy: KindPolicy, now: datetime, seal_at: datetime) -> CheckResult:
    """Relay bundle age against the budget now (FAIL) and at the seal (WARN when it will cross before then)."""
    from src.core.pit import PITDataError
    from src.data.kind_relay import check_relay_freshness, decode_relay_bundle
    from src.data.kind_relay_import import relay_inbox_path

    try:
        raw = relay_inbox_path(Path(state_root)).read_bytes()
    except OSError:
        return CheckResult(_RELAY, FAIL, (("reason", "bundle_missing"),))
    try:
        bundle = decode_relay_bundle(raw)
    except PITDataError as exc:
        return CheckResult(_RELAY, FAIL, (("reason", _token(str(exc))),))
    budget = timedelta(hours=policy.relay_max_age_hours)
    age_h = f"{(now - bundle.fetched_at).total_seconds() / 3600:.2f}"
    try:
        check_relay_freshness(bundle, now=now, policy=policy, last_accepted_at=None)
    except PITDataError as exc:
        return CheckResult(_RELAY, FAIL, (("reason", _token(str(exc))), ("age_h", age_h)))
    if seal_at - bundle.fetched_at > budget:
        return CheckResult(_RELAY, WARN, (("reason", "crosses_budget_before_seal"), ("age_h", age_h)))
    return CheckResult(_RELAY, PASS, (("age_h", age_h), ("budget_h", str(policy.relay_max_age_hours))))


def _guarded(check: str, run: Callable[[], CheckResult]) -> CheckResult:
    try:
        return run()
    except Exception as exc:  # an unreadable input is a failed gate, never a silent pass
        return CheckResult(check, FAIL, (("error", type(exc).__name__),))


def _record(root: Path, outcome: str, today: date, moment: datetime, reason: str, metrics: dict[str, str]) -> None:
    from src.ops.run_outcome import record_run_outcome

    record_run_outcome(
        root / "ops" / "events", _EVENT_JOB, outcome, run_date=today, now=moment, reason=reason, metrics=metrics
    )


def preseal_check(data_root: Path, *, today: date, now: datetime) -> int:
    """Run the pre-seal gates for ``today``'s seal; 0 when no check FAILs, 1 otherwise. Never writes data."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.runtime import resolve_data_runtime
    from src.ops.guards import trading_day
    from src.research.operate import load_state

    root = Path(data_root)
    moment = now if now.tzinfo is not None else now.replace(tzinfo=KRX_TZ)
    if not trading_day(today):
        _LOG.info("[DATA] stage=preseal %s", CheckResult(_CONTEXT, SKIPPED, (("day_kind", "non_trading"),)).line())
        _record(root, SKIPPED, today, moment, "non_trading_day", {})
        return 0
    sessions = _kst_sessions_through(today)
    through_data = sessions[: sessions.index(today)]
    data_session = through_data[-1]
    seal_at = _at(today, SEAL_TIME)
    runtime = resolve_data_runtime(data_root=root)
    state_root = runtime.workspace.state_root

    def _flow() -> CheckResult:
        try:
            last_session = load_state(state_root / "operate" / "state.json").last_session
        except (OSError, ValueError) as exc:
            return CheckResult(_FLOW, FAIL, (("reason", "state_unreadable"), ("error", type(exc).__name__)))
        return check_flow_freshness(runtime, sessions=through_data, last_session=last_session, today=today)

    def _relay() -> CheckResult:
        policy = load_provider_policy(load_runtime_config()).kind
        return check_kind_relay_age(state_root, policy=policy, now=moment, seal_at=seal_at)

    results = [
        CheckResult(_CONTEXT, PASS, (("day_kind", "trading"), ("data_session", data_session.isoformat()),
                                     ("seal_at", SEAL_TIME.strftime("%H:%M")))),
        _guarded(_KIND, lambda: check_kind_consistency(runtime.workspace.bronze_root)),
        _guarded(_FLOW, _flow),
        _guarded(_RELAY, _relay),
    ]
    for result in results:
        _LOG.info("[DATA] stage=preseal %s", result.line())
    failed = [result for result in results if result.verdict == FAIL]
    _record(
        root, "DEGRADED" if failed else "OK", today, moment,
        failed[0].check if failed else "", {result.check: result.verdict for result in results},
    )
    return 1 if failed else 0
