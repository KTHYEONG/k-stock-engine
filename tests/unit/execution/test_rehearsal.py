"""Rehearsal invariant tests."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from src.core.instruments import AssetKind, Instrument
from src.core.time import KRX_TZ
from src.execution.adapters.sqlite_state_store import SqliteStateStore
from src.execution.application.readiness import (
    LiveExecutionNotReadyError,
    ReadinessEvidence,
    SubmissionGate,
    SubmissionRequest,
)
from src.execution.application.rehearsal import RehearsalReport, rehearse_open, update_readiness
from src.execution.domain.orders import OrderSide, OrderState, OrderStateRecord
from src.execution.domain.portfolio import PortfolioSnapshot, Position
from src.research.operate import SealedDecision


def _decision(targets: dict[str, float], reference: dict[str, float], session: date) -> SealedDecision:
    return SealedDecision(
        data_session=date(2026, 10, 1),
        execution_session=session,
        sealed_at=datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ),
        spec_hash="s",
        models_id="model-a",
        inputs={},
        targets=dict(targets),
        reference=dict(reference),
        overlay={},
        state_digest="d",
        prev_hash="",
        hash="a" * 32,
    )


class FakeBroker:
    def __init__(self, positions: dict[str, int] | None = None) -> None:
        self._positions = dict(positions or {})
        self._order: list[str] = []
        self.account = "12345678"

    def snapshot(self, *, as_of: datetime) -> PortfolioSnapshot:
        items = tuple(
            Position(
                instrument=Instrument(iid, AssetKind.STOCK, "KRX", iid.split(":")[-1], "KRW"),
                quantity=float(qty),
                average_cost=1000.0,
            )
            for iid, qty in sorted(self._positions.items())
        )
        return PortfolioSnapshot(
            account_snapshot_id=f"fake:{as_of.isoformat()}",
            as_of=datetime(2026, 10, 1, 19, tzinfo=KRX_TZ),
            settled_cash=1_000_000_000.0,
            unsettled_cash=0.0,
            positions=items,
        )

    def submit(self, request) -> OrderStateRecord:  # type: ignore[no-untyped-def]
        self._order.append(request.instrument_id)
        qty = int(float(request.quantity))
        return OrderStateRecord(
            order_id=request.order_id,
            intent_id=request.intent_id,
            instrument_id=request.instrument_id,
            asset_kind=request.asset_kind,
            side=request.side,
            state=OrderState.SUBMITTED,
            submitted_quantity=float(qty),
            reason="fake",
        )

    def reconcile(self, account: str) -> list[OrderStateRecord]:
        raise NotImplementedError("unconfirmed")


def timedelta_min():  # type: ignore[no-untyped-def]
    from datetime import timedelta

    return timedelta(minutes=1)


def _at(session: date, hour: int, minute: int) -> datetime:
    return datetime(session.year, session.month, session.day, hour, minute, tzinfo=KRX_TZ)


def _store(tmp_path: Path) -> SqliteStateStore:
    return SqliteStateStore(tmp_path / "state.db")


def test_sells_sent_before_buys(tmp_path: Path) -> None:
    session = date(2026, 10, 2)
    targets = {"KRX:S1": 0.0, "KRX:S2": 0.0, "KRX:B1": 0.4, "KRX:B2": 0.3, "KRX:B3": 0.2}
    reference = dict.fromkeys(targets, 100.0)
    decision = _decision(targets, reference, session)
    broker = FakeBroker({"KRX:S1": 50, "KRX:S2": 40})
    store = _store(tmp_path)
    try:
        report = rehearse_open(
            decision, broker, store, nav_krw=10_000_000,
            cash_buffer=0.0, rebalance_band=0.0, now=_at(session, 8, 40),
        )
        assert report.intents == 5
        assert report.submitted == 5
        assert broker._order[:2] == ["KRX:S1", "KRX:S2"]
        assert len(store.records(session=session)) == 5
    finally:
        store.close()


def test_second_run_is_noop(tmp_path: Path) -> None:
    session = date(2026, 10, 2)
    decision = _decision({"KRX:005930": 0.5}, {"KRX:005930": 50000.0}, session)
    broker = FakeBroker({})
    store = _store(tmp_path)
    try:
        first = rehearse_open(
            decision, broker, store, nav_krw=10_000_000,
            cash_buffer=0.0, rebalance_band=0.0, now=_at(session, 8, 40),
        )
        assert first.submitted >= 1
        second = rehearse_open(
            decision, broker, store, nav_krw=10_000_000,
            cash_buffer=0.0, rebalance_band=0.0, now=_at(session, 8, 45),
        )
        assert second.submitted == 0
        assert second.duplicates_blocked == second.intents == first.intents
    finally:
        store.close()


def test_outside_entry_window_sends_nothing(tmp_path: Path) -> None:
    session = date(2026, 10, 2)
    decision = _decision({"KRX:005930": 0.5}, {"KRX:005930": 50000.0}, session)
    broker = FakeBroker({})
    store = _store(tmp_path)
    try:
        with pytest.raises(ValueError, match="08:30"):
            rehearse_open(
                decision, broker, store, nav_krw=10_000_000,
                cash_buffer=0.0, rebalance_band=0.0, now=_at(session, 9, 1),
            )
        assert store.records(session=session) == ()
        assert broker._order == []
    finally:
        store.close()


def test_secrets_never_reach_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from src.core.market_rules import KrxMarket, load_krx_market_rules
    from src.execution.adapters.kis_broker import KisMockBroker
    from src.integrations.kis.client import KisClient, KisCredentials

    session = date(2026, 10, 2)
    decision = _decision({"KRX:005930": 0.1}, {"KRX:005930": 50000.0}, session)
    app_key = "sekret-app-key-123"
    app_secret = "sekret-app-secret-456"
    account_no = "98765432"
    creds = KisCredentials(
        app_key=app_key, app_secret=app_secret,
        account_no=account_no, account_product_code="01", env="vps",
    )
    client = KisClient(creds)

    def fake_call(method: str, path: str, tr_id: str | None = None, **kwargs: object) -> dict[str, object]:
        if path.endswith("inquire-balance"):
            return {"rt_cd": "0", "output1": [], "output2": [{"dnca_tot_amt": "10000000"}]}
        return {"rt_cd": "0", "msg_cd": "000", "msg1": "ok", "output": {"ODNO": "111"}}

    client._call = fake_call  # type: ignore[attr-defined]
    client.ensure_token = lambda: "tok-abc-999"  # type: ignore[method-assign]
    try:
        rules = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))
    except Exception:
        from src.config.runtime import load_runtime_config

        rules = load_krx_market_rules(load_runtime_config().market_rules)
    broker = KisMockBroker(
        client, rules=rules, markets={"KRX:005930": KrxMarket.KOSPI}, session=session
    )
    store = _store(tmp_path)
    try:
        import logging

        with caplog.at_level(logging.INFO, logger="execution.kis_broker"):
            rehearse_open(
                decision, broker, store, nav_krw=10_000_000,
                cash_buffer=0.0, rebalance_band=0.0, now=_at(session, 8, 40),
            )
        blob = "\n".join(record.getMessage() for record in caplog.records)
        assert app_key not in blob
        assert app_secret not in blob
        assert "tok-abc-999" not in blob
        assert account_no not in blob
    finally:
        store.close()


def _clean_report(session: date, blocked: int = 0) -> RehearsalReport:
    return RehearsalReport(
        session=session, intents=2, submitted=2, rejected=0,
        duplicates_blocked=blocked, reconciled=False, position_mismatches=(),
    )


def test_evidence_is_conservative(tmp_path: Path) -> None:
    from src.core.krx_calendar import xkrx_calendar_through

    start = date(2026, 9, 1)
    sessions = [instant.astimezone(KRX_TZ).date() for instant in xkrx_calendar_through(date(2026, 11, 1)).sessions
                if instant.astimezone(KRX_TZ).date() >= start]
    reports_19 = [_clean_report(day) for day in sessions[:19]]
    evidence_path = tmp_path / "evidence.json"
    evidence = update_readiness(evidence_path, reports_19)
    assert evidence.paper_acceptance is False
    reports_20 = [_clean_report(day) for day in sessions[:20]]
    evidence = update_readiness(evidence_path, reports_20)
    assert evidence.paper_acceptance is True
    bad = _clean_report(sessions[20])
    bad = RehearsalReport(
        session=bad.session, intents=bad.intents, submitted=bad.submitted,
        rejected=bad.rejected, duplicates_blocked=bad.duplicates_blocked,
        reconciled=bad.reconciled, position_mismatches=("KRX:005930",),
    )
    evidence = update_readiness(evidence_path, [*reports_20, bad])
    assert evidence.broker_reconciliation is False
    assert evidence.paper_acceptance is False


def test_reconciliation_and_duplicate_counts_are_not_sufficient_proof(tmp_path: Path) -> None:
    report = RehearsalReport(date(2026, 10, 2), 1, 0, 0, 1, True, ())
    evidence = update_readiness(tmp_path / "evidence.json", [report], required_days=1)
    assert not evidence.order_state_transitions
    assert not evidence.idempotency_verified
    assert not evidence.broker_reconciliation


def test_gate_stays_closed_for_live() -> None:
    from src.execution.domain.intents import TradeIntent

    intent = TradeIntent(
        intent_id="i",
        asset_kind=AssetKind.STOCK,
        instrument_id="KRX:005930",
        target_value=1000.0,
        decision_time=datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ),
        execution_time=datetime(2026, 10, 2, 9, 0, tzinfo=KRX_TZ),
        strategy_id="s",
        reason="r",
        idempotency_key="k-live",
        account_snapshot_id="snap",
    )
    gate = SubmissionGate(ReadinessEvidence())
    with pytest.raises(LiveExecutionNotReadyError):
        gate.authorize(SubmissionRequest(intent=intent, mode="live"))


def _stored(order_id: str, odno: str, state: OrderState = OrderState.SUBMITTED) -> OrderStateRecord:
    return OrderStateRecord(
        order_id=order_id, intent_id=order_id, instrument_id="KRX:005930", asset_kind=AssetKind.STOCK,
        side=OrderSide.BUY, state=state, submitted_quantity=10.0, reason=odno,
    )


class _StatesBroker:
    def __init__(self, states: dict[str, OrderState]) -> None:
        self._states = states

    def order_states(self, session: date) -> list[OrderStateRecord]:
        return [_stored(odno, odno, state) for odno, state in self._states.items()]


def test_terminal_states_require_every_submitted_order_terminal(tmp_path: Path) -> None:
    from src.execution.application.rehearsal import verify_terminal_states

    session = date(2026, 10, 2)
    store = _store(tmp_path)
    store.save(_stored(f"{session.isoformat()}:a", "11"))
    store.save(_stored(f"{session.isoformat()}:b", "12"))
    store.save(_stored(f"{session.isoformat()}:c", "", OrderState.REJECTED))
    done = _StatesBroker({"11": OrderState.FILLED, "12": OrderState.CANCELED})
    assert verify_terminal_states(done, store, session) is True
    open_order = _StatesBroker({"11": OrderState.FILLED, "12": OrderState.SUBMITTED})
    assert verify_terminal_states(open_order, store, session) is False
    assert verify_terminal_states(_StatesBroker({"11": OrderState.FILLED}), store, session) is False
    assert verify_terminal_states(done, store, date(2026, 10, 5)) is False
    store.close()
