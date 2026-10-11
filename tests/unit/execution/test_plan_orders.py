"""plan_intents invariant tests."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from src.core.instruments import AssetKind, Instrument
from src.core.time import KRX_TZ
from src.execution.application.plan_orders import plan_intents
from src.execution.domain.portfolio import PortfolioSnapshot, Position
from src.research.operate import SealedDecision


def _decision(**overrides: object) -> SealedDecision:
    base: dict[str, object] = {
        "data_session": date(2026, 10, 1),
        "execution_session": date(2026, 10, 2),
        "sealed_at": datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ),
        "spec_hash": "spec",
        "models_id": "model-a",
        "inputs": {},
        "targets": {"KRX:005930": 0.5, "KRX:000660": 0.3},
        "reference": {"KRX:005930": 50000.0, "KRX:000660": 100000.0},
        "overlay": {},
        "state_digest": "d",
        "prev_hash": "",
        "hash": "abcdef0123456789abcdef0123456789",
    }
    base.update(overrides)
    return SealedDecision(**base)  # type: ignore[arg-type]


def _snapshot(quantities: dict[str, int], when: datetime | None = None) -> PortfolioSnapshot:
    as_of = when or datetime(2026, 10, 1, 19, 0, tzinfo=KRX_TZ)
    positions = tuple(
        Position(
            instrument=Instrument(iid, AssetKind.STOCK, "KRX", iid.split(":")[-1], "KRW"),
            quantity=float(qty),
            average_cost=1000.0,
        )
        for iid, qty in sorted(quantities.items())
    )
    return PortfolioSnapshot(
        account_snapshot_id="snap-1",
        as_of=as_of,
        settled_cash=0.0,
        unsettled_cash=0.0,
        positions=positions,
    )


def test_matches_engine_order_sizing() -> None:
    import math

    import numpy as np

    from src.backtest.engine import _target_orders
    from src.backtest.market import MarketArrays

    nav = 10_000_000
    buffer = 0.02
    closes = {"KRX:005930": 50000, "KRX:000660": 100000}
    weights = {"KRX:005930": 0.5, "KRX:000660": 0.3}
    decision = _decision(targets=dict(weights), reference={k: float(v) for k, v in closes.items()})
    snapshot = _snapshot({"KRX:005930": 10})
    intents = plan_intents(
        decision,
        snapshot,
        nav_krw=nav,
        reference_prices=dict(closes),
        cash_buffer=buffer,
        rebalance_band=0.0,
        strategy_id="s",
    )
    plan_delta = {i.instrument_id: int(i.target_quantity or 0) - snapshot.quantity_of(i.instrument_id) for i in intents}
    ids = ("KRX:005930", "KRX:000660")
    arrays = MarketArrays(
        dataset_id="test",
        sessions=(date(2026, 10, 1),),
        instrument_ids=ids,
        int_fields={"close": np.array([[50000, 100000]], dtype=np.int64)},
        float_fields={},
        bool_fields={"present": np.array([[True, True]])},
        market=np.array([[1, 1]], dtype=np.int8),
    )
    engine_weights = {0: 0.5, 1: 0.3}
    orders = _target_orders(
        weights=engine_weights,
        arrays=arrays,
        t=0,
        holdings={0: 10},
        nav=nav,
        buffer_scale=1.0 - buffer,
        rebalance_band=0.0,
    )
    engine_delta = {0: 0, 1: 0}
    for order in orders:
        sign = 1 if order.side == "buy" else -1
        engine_delta[order.instrument_idx] += sign * int(order.quantity)
        _ = math.floor(1)
    assert plan_delta.get("KRX:005930", 0) == engine_delta[0]
    assert plan_delta.get("KRX:000660", 0) == engine_delta[1]


def test_band_suppresses_small_resizes_only() -> None:
    decision = _decision(
        targets={"KRX:A": 0.5, "KRX:B": 0.5},
        reference={"KRX:A": 100.0, "KRX:B": 100.0},
        hash="h" * 32,
    )
    nav = 10_000
    snapshot = _snapshot({"KRX:A": 40, "KRX:B": 10})
    intents = plan_intents(
        decision,
        snapshot,
        nav_krw=nav,
        reference_prices={"KRX:A": 100, "KRX:B": 100},
        cash_buffer=0.0,
        rebalance_band=0.5,
        strategy_id="s",
    )
    names = {i.instrument_id for i in intents}
    assert "KRX:A" not in names
    assert "KRX:B" in names


def test_dropped_name_is_sold_in_full() -> None:
    decision = _decision(targets={"KRX:005930": 0.5}, reference={"KRX:005930": 50000.0, "KRX:000660": 100000.0})
    snapshot = _snapshot({"KRX:005930": 5, "KRX:000660": 7})
    intents = plan_intents(
        decision,
        snapshot,
        nav_krw=10_000_000,
        reference_prices={"KRX:005930": 50000, "KRX:000660": 100000},
        cash_buffer=0.0,
        rebalance_band=0.0,
        strategy_id="s",
    )
    dropped = [i for i in intents if i.instrument_id == "KRX:000660"]
    assert len(dropped) == 1
    assert dropped[0].target_quantity == 0


def test_missing_reference_price_fails_closed() -> None:
    decision = _decision(targets={"KRX:005930": 0.5}, reference={"KRX:005930": 50000.0})
    snapshot = _snapshot({"KRX:005930": 0})
    with pytest.raises(ValueError, match="reference"):
        plan_intents(
            decision,
            snapshot,
            nav_krw=10_000_000,
            reference_prices={},
            cash_buffer=0.0,
            rebalance_band=0.0,
            strategy_id="s",
        )


def test_deterministic_ids() -> None:
    decision = _decision()
    snapshot = _snapshot({})
    first = plan_intents(
        decision, snapshot, nav_krw=10_000_000,
        reference_prices={"KRX:005930": 50000, "KRX:000660": 100000},
        cash_buffer=0.0, rebalance_band=0.0, strategy_id="s",
    )
    second = plan_intents(
        decision, snapshot, nav_krw=10_000_000,
        reference_prices={"KRX:005930": 50000, "KRX:000660": 100000},
        cash_buffer=0.0, rebalance_band=0.0, strategy_id="s",
    )
    assert [(i.intent_id, i.idempotency_key) for i in first] == [(i.intent_id, i.idempotency_key) for i in second]
    assert first[0].intent_id.startswith("2026-10-02:")
    assert first[0].idempotency_key.startswith("abcdef0123456789"[:16] + ":")


def test_execution_planning_preserves_snapshot_time_and_sealed_inputs() -> None:
    decision = _decision()
    now = datetime(2026, 10, 2, 8, 40, tzinfo=KRX_TZ)
    snapshot = _snapshot({"KRX:005930": 10}, when=now)
    kwargs = {"nav_krw": 10_000_000, "reference_prices": {"KRX:005930": 50000, "KRX:000660": 100000},
              "cash_buffer": 0.0, "rebalance_band": 0.0, "strategy_id": "s"}
    intents = plan_intents(decision, snapshot, planning_time=now, **kwargs)
    assert snapshot.as_of == now
    assert all(intent.decision_time == decision.sealed_at for intent in intents)
    assert next(intent for intent in intents if intent.instrument_id == "KRX:005930").target_quantity == 100
    with pytest.raises(ValueError, match="newer"):
        plan_intents(decision, snapshot, **kwargs)
    with pytest.raises(ValueError, match="newer"):
        plan_intents(decision, snapshot, planning_time=datetime(2026, 10, 2, 8, 39, tzinfo=KRX_TZ), **kwargs)
    with pytest.raises(ValueError, match="precede"):
        plan_intents(decision, snapshot, planning_time=datetime(2026, 10, 1, 19, tzinfo=KRX_TZ), **kwargs)
    with pytest.raises(ValueError, match="follow"):
        plan_intents(decision, snapshot, planning_time=datetime(2026, 10, 2, 9, 1, tzinfo=KRX_TZ), **kwargs)
    with pytest.raises(ValueError, match="timezone-aware"):
        plan_intents(decision, snapshot, planning_time=datetime(2026, 10, 2, 8, 40), **kwargs)
