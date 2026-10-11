"""Mock-domain open rehearsal and readiness evidence."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from src.core.time import KRX_TZ
from src.execution.application.plan_orders import plan_intents
from src.execution.application.readiness import ReadinessEvidence
from src.execution.application.submit_intents import ExecutionContext, submit_intents
from src.execution.domain.decision import TargetDecision
from src.execution.domain.intents import TradeIntent
from src.execution.domain.orders import OrderState
from src.execution.settings import ExecutionSettings


@dataclass(frozen=True, slots=True)
class RehearsalReport:
    session: date
    intents: int
    submitted: int
    rejected: int
    duplicates_blocked: int
    reconciled: bool
    position_mismatches: tuple[str, ...]
    terminal_states_verified: bool = False
    duplicate_orders_absent: bool = False
    position_book_verified: bool = False


class _ReferencePrices:
    def __init__(self, prices: Mapping[str, float]) -> None:
        self._prices = dict(prices)

    def price_of(self, instrument_id: str) -> float:
        if instrument_id not in self._prices:
            raise ValueError(f"missing reference price for {instrument_id!r}")
        price = float(self._prices[instrument_id])
        if price <= 0:
            raise ValueError(f"invalid reference price for {instrument_id!r}")
        return price


def _in_entry_window(now: datetime, session: date) -> bool:
    if now.utcoffset() is None:
        return False
    local = now.astimezone(KRX_TZ)
    if local.date() != session:
        return False
    minutes = local.hour * 60 + local.minute
    return 8 * 60 + 30 <= minutes <= 8 * 60 + 58


def rehearse_open(
    decision: TargetDecision,
    broker: object,
    store: object,
    *,
    nav_krw: int,
    cash_buffer: float,
    rebalance_band: float,
    now: datetime,
) -> RehearsalReport:
    """Plan intents from the sealed decision against the mock account, submit them once, and reconcile."""
    from src.execution.application.readiness import SubmissionGate

    session = decision.execution_session
    if not _in_entry_window(now, session):
        raise ValueError("rehearsal runs only between 08:30 and 08:58 Asia/Seoul on the execution session")
    snapshot_before = broker.snapshot(as_of=now)  # type: ignore[attr-defined]
    reference_floats = {key: float(value) for key, value in dict(decision.reference).items()}
    reference_ints = {key: int(float(value)) for key, value in reference_floats.items()}
    intents = plan_intents(
        decision,
        snapshot_before,
        nav_krw=nav_krw,
        reference_prices=reference_ints,
        cash_buffer=cash_buffer,
        rebalance_band=rebalance_band,
        strategy_id=decision.models_id,
        planning_time=now,
    )
    blocked = 0
    fresh: list[TradeIntent] = []
    for intent in intents:
        if store.is_duplicate_intent(intent.idempotency_key):  # type: ignore[attr-defined]
            blocked += 1
        else:
            fresh.append(intent)
    submitted_count = 0
    rejected_count = 0
    records: list[object] = []
    if fresh:
        context = ExecutionContext(
            settings=ExecutionSettings(default_mode="paper"),
            gate=SubmissionGate(),
            broker=broker,  # type: ignore[arg-type]
            state_store=store,  # type: ignore[arg-type]
            price_provider=_ReferencePrices(reference_floats),
            account=getattr(broker, "account", "kis-mock"),
            now=now,
            positions=snapshot_before,
        )
        submitted_records = submit_intents(fresh, context)
        records = list(submitted_records)
        for record in records:
            state = getattr(record, "state", None)
            if state is OrderState.REJECTED:
                rejected_count += 1
            else:
                submitted_count += 1
    try:
        broker.reconcile(getattr(broker, "account", "kis-mock"))  # type: ignore[attr-defined]
        reconciled = True
    except NotImplementedError:
        reconciled = False
    snapshot_after = broker.snapshot(as_of=now)  # type: ignore[attr-defined]
    before_qty = {p.instrument.instrument_id: int(p.quantity) for p in snapshot_before.positions}
    after_qty = {p.instrument.instrument_id: int(p.quantity) for p in snapshot_after.positions}
    implied = dict(before_qty)
    for record in records:
        instrument_id = str(getattr(record, "instrument_id", ""))
        qty = int(float(getattr(record, "submitted_quantity", 0.0)))
        if getattr(record, "state", None) is OrderState.REJECTED:
            continue
        side = str(getattr(getattr(record, "side", ""), "value", getattr(record, "side", "")))
        if side.upper() == "BUY":
            implied[instrument_id] = implied.get(instrument_id, 0) + qty
        else:
            implied[instrument_id] = implied.get(instrument_id, 0) - qty
    mismatches = sorted(
        name
        for name in set(after_qty) | set(implied)
        if int(after_qty.get(name, 0)) != int(implied.get(name, 0))
    )
    return RehearsalReport(
        session=session,
        intents=len(intents),
        submitted=submitted_count,
        rejected=rejected_count,
        duplicates_blocked=blocked,
        reconciled=reconciled,
        position_mismatches=tuple(mismatches),
    )


_TERMINAL_STATES = frozenset({OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED})


def verify_terminal_states(broker: object, store: object, session: date) -> bool:
    """True when every order this project submitted for the session is terminal at the broker."""
    ours = [
        record
        for record in store.records(session=session)  # type: ignore[attr-defined]
        if record.state is not OrderState.REJECTED
    ]
    if not ours:
        return False
    broker_states = {
        record.order_id: record.state
        for record in broker.order_states(session)  # type: ignore[attr-defined]
    }
    return all(broker_states.get(record.reason) in _TERMINAL_STATES for record in ours)


def _report_to_payload(report: RehearsalReport) -> dict[str, object]:
    return {
        "session": report.session.isoformat(),
        "intents": int(report.intents),
        "submitted": int(report.submitted),
        "rejected": int(report.rejected),
        "duplicates_blocked": int(report.duplicates_blocked),
        "reconciled": bool(report.reconciled),
        "position_mismatches": list(report.position_mismatches),
        "terminal_states_verified": bool(report.terminal_states_verified),
        "duplicate_orders_absent": bool(report.duplicate_orders_absent),
        "position_book_verified": bool(report.position_book_verified),
    }


def _clean(report: RehearsalReport) -> bool:
    return report.rejected == 0 and not report.position_mismatches


def update_readiness(
    evidence_path: Path,
    reports: Sequence[RehearsalReport],
    *,
    required_days: int = 20,
) -> ReadinessEvidence:
    """Evidence derived from consecutive rehearsal reports; persisted as JSON."""
    if isinstance(required_days, bool) or not isinstance(required_days, int) or required_days <= 0:
        raise ValueError("required_days must be a positive int")
    ordered = sorted(reports, key=lambda item: item.session)
    if len({item.session for item in ordered}) != len(ordered):
        raise ValueError("readiness requires one aggregate report per session")
    from src.core.krx_calendar import xkrx_calendar_through

    session_indices = {
        instant.astimezone(KRX_TZ).date(): index
        for index, instant in enumerate(xkrx_calendar_through(ordered[-1].session).sessions)
    } if ordered else {}
    trailing: list[RehearsalReport] = []
    for report in ordered:
        if report.session not in session_indices:
            trailing = []
            continue
        if trailing and session_indices[report.session] != session_indices[trailing[-1].session] + 1:
            trailing = []
        if _clean(report) and report.intents > 0:
            trailing.append(report)
        else:
            trailing = []
    consecutive = len(trailing)
    window = trailing[-required_days:] if consecutive >= required_days else []
    paper_acceptance = len(window) == required_days and all(item.rejected == 0 for item in window)
    broker_reconciliation = len(window) == required_days and all(not item.position_mismatches for item in window)
    broker_reconciliation = broker_reconciliation and all(item.position_book_verified for item in window)
    order_transitions = len(window) == required_days and all(
        item.reconciled and item.terminal_states_verified for item in window
    )
    idempotency_verified = any(
        item.duplicates_blocked > 0 and item.duplicate_orders_absent for item in ordered
    )
    evidence = ReadinessEvidence(
        broker_reconciliation=bool(broker_reconciliation),
        order_state_transitions=bool(order_transitions),
        paper_acceptance=bool(paper_acceptance),
        idempotency_verified=bool(idempotency_verified),
        notes=(f"consecutive_clean={consecutive}", f"required_days={required_days}"),
    )
    path = Path(evidence_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "broker_reconciliation": evidence.broker_reconciliation,
        "order_state_transitions": evidence.order_state_transitions,
        "paper_acceptance": evidence.paper_acceptance,
        "idempotency_verified": evidence.idempotency_verified,
        "notes": list(evidence.notes),
        "consecutive_clean": consecutive,
        "required_days": required_days,
        "reports": [_report_to_payload(item) for item in ordered],
    }
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return evidence
