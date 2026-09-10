"""Deterministic long-only cash-equity ledger."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class LedgerSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class LedgerActionType(StrEnum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND = "dividend"
    DELISTING_CASH_OUT = "delisting_cash_out"
    DELISTING_UNSETTLED = "delisting_unsettled"
    EXCHANGE_ENTITLEMENT = "exchange_entitlement"
    SUCCESSOR_DELIVERY = "successor_delivery"
    CASH_IN_LIEU_SETTLEMENT = "cash_in_lieu_settlement"


_SUCCESSOR_ACTION_TYPES = frozenset(
    {
        LedgerActionType.EXCHANGE_ENTITLEMENT,
        LedgerActionType.SUCCESSOR_DELIVERY,
        LedgerActionType.CASH_IN_LIEU_SETTLEMENT,
    }
)


_DELISTING_ACTION_TYPES = frozenset(
    {
        LedgerActionType.DELISTING_CASH_OUT,
        LedgerActionType.DELISTING_UNSETTLED,
    }
)


@dataclass(frozen=True, slots=True)
class LedgerFill:
    fill_id: str
    instrument_id: str
    side: LedgerSide
    quantity: int
    price: float
    commission: float
    tax: float
    slippage_cost: float
    trade_time: datetime
    settlement_time: datetime

    def __post_init__(self) -> None:
        if not self.fill_id:
            raise ValueError("fill_id must be non-empty")
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        if not isinstance(self.side, LedgerSide):
            raise ValueError("side must be BUY or SELL")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise ValueError("quantity must be positive integer")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if isinstance(self.price, bool) or not isinstance(self.price, (int, float)):
            raise ValueError("price must be positive")
        if not math.isfinite(float(self.price)) or float(self.price) <= 0:
            raise ValueError("price must be positive finite")
        for name, val in (
            ("commission", self.commission),
            ("tax", self.tax),
            ("slippage_cost", self.slippage_cost),
        ):
            if isinstance(val, bool):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite")
            if float(val) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.side is LedgerSide.BUY and float(self.tax) != 0.0:
            raise ValueError("tax must be zero for BUY")
        if self.trade_time.tzinfo is None or self.settlement_time.tzinfo is None:
            raise ValueError("trade_time and settlement_time must be aware")
        if self.settlement_time < self.trade_time:
            raise ValueError("settlement_time must not be before trade_time")


@dataclass(frozen=True, slots=True)
class LedgerPosition:
    instrument_id: str
    quantity: int
    average_cost: float


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    ledger_id: str
    as_of: datetime
    settled_cash: float
    unsettled_cash: float
    positions: tuple[LedgerPosition, ...]
    commission: float
    tax: float
    slippage_cost: float


@dataclass(frozen=True, slots=True)
class LedgerSuccessorAllocation:
    successor_instrument_id: str
    ratio: Decimal
    cost_basis_weight: Decimal

    def __post_init__(self) -> None:
        if not self.successor_instrument_id:
            raise ValueError("successor_instrument_id must be non-empty")
        for name, value in (("ratio", self.ratio), ("cost_basis_weight", self.cost_basis_weight)):
            if isinstance(value, bool) or not isinstance(value, Decimal):
                raise ValueError(f"{name} must be Decimal")
            if not value.is_finite() or value <= 0:
                raise ValueError(f"{name} must be positive finite")


@dataclass(frozen=True, slots=True)
class LedgerCorporateAction:
    action_id: str
    instrument_id: str
    action_type: LedgerActionType
    effective_time: datetime
    factor: float
    cash_amount: float
    successor_allocations: tuple[LedgerSuccessorAllocation, ...] = ()
    lifecycle_event_id: str | None = None
    settlement_instrument_id: str | None = None
    cash_settlement_per_entitlement_unit: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        if not isinstance(self.action_type, LedgerActionType):
            raise ValueError("action_type must be LedgerActionType")
        if self.effective_time.tzinfo is None:
            raise ValueError("effective_time must be aware")
        if not math.isfinite(float(self.factor)) or float(self.factor) <= 0:
            raise ValueError("factor must be positive finite")
        if not math.isfinite(float(self.cash_amount)) or float(self.cash_amount) < 0:
            raise ValueError("cash_amount must be non-negative finite")


@dataclass(frozen=True, slots=True)
class LedgerMark:
    mark_id: str
    as_of: datetime
    prices: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.mark_id:
            raise ValueError("mark_id must be non-empty")
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be aware")
        # prices validation done in record_mark


@dataclass(frozen=True, slots=True)
class LedgerNav:
    mark_id: str
    as_of: datetime
    nav: float
    settled_cash: float
    unsettled_cash: float
    marked_value: float


@dataclass(frozen=True, slots=True)
class LedgerJournalEntry:
    event_id: str
    event_type: str
    event_time: datetime
    payload: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not self.event_type:
            raise ValueError("event_type must be non-empty")
        if self.event_time.tzinfo is None:
            raise ValueError("event_time must be aware")


@dataclass(slots=True)
class _PendingProceed:
    settlement_time: datetime
    amount: float
    settled: bool = False


class Ledger:
    def __init__(self, ledger_id: str, initial_cash: float, opened_at: datetime) -> None:
        if not ledger_id:
            raise ValueError("ledger_id must be non-empty")
        if opened_at.tzinfo is None:
            raise ValueError("opened_at must be aware")
        if isinstance(initial_cash, bool) or not isinstance(initial_cash, (int, float)):
            raise ValueError("initial_cash must be finite")
        if not math.isfinite(float(initial_cash)) or float(initial_cash) < 0:
            raise ValueError("initial_cash must be non-negative finite")
        self._ledger_id = ledger_id
        self._settled_cash: float = float(initial_cash)
        self._unsettled_cash: float = 0.0
        self._positions: dict[str, tuple[int, float]] = {}
        self._commission: float = 0.0
        self._tax: float = 0.0
        self._slippage_cost: float = 0.0
        self._fill_ids: set[str] = set()
        self._pendings: list[_PendingProceed] = []
        self._latest_time: datetime = opened_at
        self._journal: list[LedgerJournalEntry] = []
        self._journal_ids: set[str] = set()
        self._action_ids: set[str] = set()
        self._mark_ids: set[str] = set()
        self._entitlements: dict[tuple[str, str], Decimal] = {}
        self._entitlement_costs: dict[tuple[str, str], Decimal] = {}
        self._entitlement_allocations: dict[str, tuple[LedgerSuccessorAllocation, ...]] = {}
        self._lifecycle_successor_ids: set[str] = set()
        self._lifecycle_delisting_ids: set[str] = set()

    def _settle_due(self, as_of: datetime) -> None:
        for p in self._pendings:
            if not p.settled and p.settlement_time <= as_of:
                p.settled = True
                self._settled_cash += p.amount
                self._unsettled_cash -= p.amount

    def quantity_of(self, instrument_id: str) -> int:
        return self._positions.get(instrument_id, (0, 0.0))[0]

    def validate_fill(self, fill: LedgerFill) -> None:
        """Validate a fill without changing Ledger state."""
        if fill.fill_id in self._fill_ids:
            raise ValueError(f"duplicate fill_id {fill.fill_id!r}")
        if fill.fill_id in self._journal_ids:
            raise ValueError(f"duplicate fill_id {fill.fill_id!r}")
        if fill.trade_time < self._latest_time:
            raise ValueError("trade_time must be nondecreasing")
        current_qty = self._positions.get(fill.instrument_id, (0, 0.0))[0]
        if fill.side is LedgerSide.SELL and fill.quantity > current_qty:
            raise ValueError("holdings insufficient for sell")
        due_amount = sum(
            p.amount for p in self._pendings
            if not p.settled and p.settlement_time <= fill.trade_time
        )
        prospective_settled = self._settled_cash + due_amount
        if fill.side is LedgerSide.BUY:
            total = fill.quantity * float(fill.price) + float(fill.commission)
            if prospective_settled - total < -1e-9:
                raise ValueError("settled cash would be negative")
        else:
            proceeds = fill.quantity * float(fill.price) - float(fill.commission) - float(fill.tax)
            if proceeds < 0:
                raise ValueError("sell proceeds would be negative")

    def record_fill(self, fill: LedgerFill) -> None:
        # 사전검증을 먼저 끝내 실패 시 내부 상태가 변하지 않도록 한다.
        self.validate_fill(fill)
        # compute due amount without mutating
        due_amount = 0.0
        due_indices: list[int] = []
        for idx, p in enumerate(self._pendings):
            if not p.settled and p.settlement_time <= fill.trade_time:
                due_amount += p.amount
                due_indices.append(idx)
        prospective_settled = self._settled_cash + due_amount
        prospective_unsettled = self._unsettled_cash - due_amount
        # validate and compute fill effects on prospective
        if fill.side is LedgerSide.BUY:
            notional = fill.quantity * float(fill.price)
            total = notional + float(fill.commission)
            if prospective_settled - total < -1e-9:
                raise ValueError("settled cash would be negative")
            # update prospective positions
            old_qty, old_avg = self._positions.get(fill.instrument_id, (0, 0.0))
            new_qty = old_qty + fill.quantity
            new_avg = (
                old_qty * old_avg
                + fill.quantity * float(fill.price)
                + float(fill.commission)
            ) / new_qty
            # commit: settle due
            for idx in due_indices:
                self._pendings[idx].settled = True
                # journal for settlement
                pend = self._pendings[idx]
                # create settlement journal entry (one-time)
                settle_id = f"settle:{fill.fill_id}:{idx}"
                if settle_id not in self._journal_ids:
                    entry = LedgerJournalEntry(
                        event_id=settle_id,
                        event_type="settlement",
                        event_time=fill.trade_time,
                        payload=(("amount", pend.amount), ("settlement_time", pend.settlement_time.isoformat())),
                    )
                    self._journal.append(entry)
                    self._journal_ids.add(settle_id)
            self._settled_cash = prospective_settled - total
            self._unsettled_cash = prospective_unsettled
            self._positions[fill.instrument_id] = (new_qty, new_avg)
            self._commission += float(fill.commission)
            self._slippage_cost += float(fill.slippage_cost)
            # tax is zero for buy, already validated
        else:
            # SELL
            old_qty, old_avg = self._positions.get(fill.instrument_id, (0, 0.0))
            new_qty = old_qty - fill.quantity
            proceeds = fill.quantity * float(fill.price) - float(fill.commission) - float(fill.tax)
            # settle due first
            for idx in due_indices:
                self._pendings[idx].settled = True
                pend = self._pendings[idx]
                settle_id = f"settle:{fill.fill_id}:{idx}"
                if settle_id not in self._journal_ids:
                    entry = LedgerJournalEntry(
                        event_id=settle_id,
                        event_type="settlement",
                        event_time=fill.trade_time,
                        payload=(("amount", pend.amount), ("settlement_time", pend.settlement_time.isoformat())),
                    )
                    self._journal.append(entry)
                    self._journal_ids.add(settle_id)
            self._settled_cash = prospective_settled
            self._unsettled_cash = prospective_unsettled + proceeds
            if new_qty == 0:
                self._positions.pop(fill.instrument_id, None)
            else:
                self._positions[fill.instrument_id] = (new_qty, old_avg)
            self._commission += float(fill.commission)
            self._tax += float(fill.tax)
            self._slippage_cost += float(fill.slippage_cost)
            self._pendings.append(_PendingProceed(fill.settlement_time, proceeds, False))
        self._fill_ids.add(fill.fill_id)
        # journal for fill
        payload = (
            ("fill_id", fill.fill_id),
            ("instrument_id", fill.instrument_id),
            ("side", fill.side.value),
            ("quantity", fill.quantity),
            ("price", float(fill.price)),
        )
        entry = LedgerJournalEntry(
            event_id=fill.fill_id,
            event_type="fill",
            event_time=fill.trade_time,
            payload=payload,
        )
        if entry.event_id in self._journal_ids:
            raise ValueError(f"duplicate journal event_id {entry.event_id!r}")
        self._journal.append(entry)
        self._journal_ids.add(entry.event_id)
        if fill.trade_time > self._latest_time:
            self._latest_time = fill.trade_time

    def apply_fill(self, fill: LedgerFill) -> None:
        self.record_fill(fill)

    def settle(self, as_of: datetime) -> None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be aware")
        if as_of < self._latest_time:
            raise ValueError("as_of must be at or after latest event")
        # create settlement journal for pendings that will settle
        for idx, p in enumerate(self._pendings):
            if not p.settled and p.settlement_time <= as_of:
                settle_id = f"settle:{as_of.isoformat()}:{idx}"
                # ensure uniqueness
                suffix = 0
                base = settle_id
                while settle_id in self._journal_ids:
                    suffix += 1
                    settle_id = f"{base}:{suffix}"
                entry = LedgerJournalEntry(
                    event_id=settle_id,
                    event_type="settlement",
                    event_time=as_of,
                    payload=(("amount", p.amount), ("settlement_time", p.settlement_time.isoformat())),
                )
                self._journal.append(entry)
                self._journal_ids.add(settle_id)
        self._settle_due(as_of)
        if as_of > self._latest_time:
            self._latest_time = as_of

    def snapshot(self, as_of: datetime) -> LedgerSnapshot:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be aware")
        if as_of < self._latest_time:
            raise ValueError("as_of must be at or after latest event")
        self._settle_due(as_of)
        if as_of > self._latest_time:
            self._latest_time = as_of
        positions = tuple(
            LedgerPosition(instrument_id=instr, quantity=qty, average_cost=avg)
            for instr, (qty, avg) in sorted(self._positions.items())
        )
        return LedgerSnapshot(
            ledger_id=self._ledger_id,
            as_of=as_of,
            settled_cash=float(self._settled_cash),
            unsettled_cash=float(self._unsettled_cash),
            positions=positions,
            commission=float(self._commission),
            tax=float(self._tax),
            slippage_cost=float(self._slippage_cost),
        )

    def apply_corporate_actions(
        self,
        actions: tuple[LedgerCorporateAction, ...],
        *,
        session_open: datetime,
        cash_in_lieu_prices: Mapping[str, float],
    ) -> tuple[LedgerJournalEntry, ...]:
        if session_open.tzinfo is None:
            raise ValueError("session_open must be aware")
        if session_open < self._latest_time:
            raise ValueError("session_open must be nondecreasing")
        # pre-validation fail closed before any mutation
        seen_in_batch: set[str] = set()
        for act in actions:
            if not act.action_id:
                raise ValueError("action_id must be non-empty")
            if act.action_id in self._action_ids:
                raise ValueError(f"duplicate action_id {act.action_id!r}")
            if act.action_id in seen_in_batch:
                raise ValueError(f"duplicate action_id {act.action_id!r}")
            seen_in_batch.add(act.action_id)
            if act.action_id in self._journal_ids:
                raise ValueError(f"duplicate action_id {act.action_id!r}")
            if not act.instrument_id:
                raise ValueError("instrument_id must be non-empty")
            if not isinstance(act.action_type, LedgerActionType):
                raise ValueError(f"unknown action_type {act.action_type!r}")
            if act.effective_time != session_open:
                raise ValueError("effective_time must equal session_open")
            if not math.isfinite(float(act.factor)) or float(act.factor) <= 0:
                raise ValueError("factor must be positive finite")
            if not math.isfinite(float(act.cash_amount)) or float(act.cash_amount) < 0:
                raise ValueError("cash_amount must be non-negative finite")
            if act.action_type in _SUCCESSOR_ACTION_TYPES:
                from src.core.pit import PITDataError as _PITDataError

                if not act.lifecycle_event_id:
                    raise _PITDataError(f"successor action requires lifecycle_event_id for {act.instrument_id!r}")
                if float(act.factor) != 1.0:
                    raise ValueError("factor must be 1.0 for successor action")
                if float(act.cash_amount) != 0.0:
                    raise ValueError("cash_amount must be zero for successor action")
                if act.action_type in (
                    LedgerActionType.EXCHANGE_ENTITLEMENT,
                    LedgerActionType.SUCCESSOR_DELIVERY,
                ):
                    if act.settlement_instrument_id is not None:
                        raise _PITDataError(f"successor exchange must not carry settlement fields for {act.instrument_id!r}")
                    if act.cash_settlement_per_entitlement_unit is not None:
                        raise _PITDataError(f"successor exchange must not carry settlement fields for {act.instrument_id!r}")
                    if not act.successor_allocations:
                        raise _PITDataError(f"successor exchange requires allocations for {act.instrument_id!r}")
                    weight_total = sum((alloc.cost_basis_weight for alloc in act.successor_allocations), Decimal("0"))
                    if weight_total != Decimal("1"):
                        raise _PITDataError(f"successor cost_basis_weight must sum to 1 for {act.instrument_id!r}")
                else:
                    if not act.settlement_instrument_id:
                        raise _PITDataError(f"cash-in-lieu requires settlement_instrument_id for {act.instrument_id!r}")
                    per_unit = act.cash_settlement_per_entitlement_unit
                    if (
                        isinstance(per_unit, bool)
                        or not isinstance(per_unit, Decimal)
                        or not per_unit.is_finite()
                        or per_unit <= 0
                    ):
                        raise _PITDataError(f"cash-in-lieu requires disclosed per-unit cash for {act.instrument_id!r}")
            elif act.action_type == LedgerActionType.DIVIDEND:
                if float(act.factor) != 1.0:
                    raise ValueError("invalid factor for dividend")
            elif act.action_type == LedgerActionType.DELISTING_CASH_OUT:
                _ = float(act.cash_amount)
            elif act.action_type == LedgerActionType.DELISTING_UNSETTLED:
                if float(act.factor) != 1.0:  # pragma: no cover
                    raise ValueError("invalid factor for delisting_unsettled")
                if float(act.cash_amount) != 0.0:  # pragma: no cover
                    raise ValueError("cash_amount must be zero for delisting_unsettled")
            else:
                if float(act.cash_amount) != 0.0:
                    raise ValueError("cash_amount must be zero for split")
                if float(act.factor) == 1.0:
                    raise ValueError("invalid factor for split")
                price = cash_in_lieu_prices.get(act.instrument_id)
                if price is None:
                    raise ValueError(f"missing raw open for {act.instrument_id!r}")
                if isinstance(price, bool) or not isinstance(price, (int, float)):
                    raise ValueError("cash_in_lieu price must be finite")
                if not math.isfinite(float(price)) or float(price) <= 0:
                    raise ValueError("cash_in_lieu price must be positive finite")
            # unsupported types already handled via enum
        # snapshot opening quantities before any action
        opening_snapshot: dict[str, tuple[int, float]] = dict(self._positions)
        prospective_settled = float(self._settled_cash)
        prospective_positions: dict[str, tuple[int, float]] = dict(self._positions)
        prospective_entitlements: dict[tuple[str, str], Decimal] = dict(self._entitlements)
        prospective_costs: dict[tuple[str, str], Decimal] = dict(self._entitlement_costs)
        prospective_allocs: dict[str, tuple[LedgerSuccessorAllocation, ...]] = dict(self._entitlement_allocations)
        prospective_successor_ids: set[str] = set(self._lifecycle_successor_ids)
        prospective_delisting_ids: set[str] = set(self._lifecycle_delisting_ids)
        new_entries: list[LedgerJournalEntry] = []
        # Successor lifecycle consistency is validated before any mutation.
        from src.core.pit import PITDataError as _SuccessorPITDataError

        batch_delisting: set[str] = set()
        batch_successor: set[str] = set()
        for act in actions:
            if act.lifecycle_event_id:
                if act.action_type in _SUCCESSOR_ACTION_TYPES:
                    batch_successor.add(act.lifecycle_event_id)
                elif act.action_type in _DELISTING_ACTION_TYPES:
                    batch_delisting.add(act.lifecycle_event_id)
        for lifecycle_id in sorted(batch_successor & (batch_delisting | self._lifecycle_delisting_ids)):
            raise _SuccessorPITDataError(f"conflicting lifecycle {lifecycle_id!r} mixes delisting and successor actions")
        for lifecycle_id in sorted(batch_delisting & self._lifecycle_successor_ids):
            raise _SuccessorPITDataError(f"conflicting lifecycle {lifecycle_id!r} mixes delisting and successor actions")
        for act in actions:
            if act.action_type == LedgerActionType.EXCHANGE_ENTITLEMENT:
                assert act.lifecycle_event_id is not None
                opening_qty, _ = opening_snapshot.get(act.instrument_id, (0, 0.0))
                if opening_qty <= 0:
                    raise _SuccessorPITDataError(f"exchange entitlement requires opening position for {act.instrument_id!r}")
                for alloc in act.successor_allocations:
                    if (act.lifecycle_event_id, alloc.successor_instrument_id) in self._entitlements:
                        raise ValueError(f"duplicate entitlement for {act.lifecycle_event_id!r}")
            elif act.action_type == LedgerActionType.SUCCESSOR_DELIVERY:
                assert act.lifecycle_event_id is not None
                expected = self._entitlement_allocations.get(act.lifecycle_event_id)
                if expected is None:
                    raise _SuccessorPITDataError(f"successor delivery without entitlement for {act.lifecycle_event_id!r}")
                if tuple(expected) != tuple(act.successor_allocations):
                    raise _SuccessorPITDataError(f"successor delivery allocations mismatch for {act.lifecycle_event_id!r}")
            elif act.action_type == LedgerActionType.CASH_IN_LIEU_SETTLEMENT:
                assert act.lifecycle_event_id is not None
                assert act.settlement_instrument_id is not None
                residual_key = (act.lifecycle_event_id, act.settlement_instrument_id)
                residual = self._entitlements.get(residual_key)
                if residual is None:
                    raise _SuccessorPITDataError(f"cash-in-lieu without residual entitlement for {act.lifecycle_event_id!r}")
                if residual <= 0 or residual == residual.to_integral_value():
                    raise _SuccessorPITDataError("cash-in-lieu requires a fractional residual entitlement")
        # handle dividend and splits from opening snapshot
        for act in actions:
            if act.action_type == LedgerActionType.EXCHANGE_ENTITLEMENT:
                assert act.lifecycle_event_id is not None
                ent_qty, ent_avg = opening_snapshot.get(act.instrument_id, (0, 0.0))
                ent_total = Decimal(str(ent_avg)) * Decimal(ent_qty)
                prospective_positions.pop(act.instrument_id, None)
                for alloc in act.successor_allocations:
                    ent_key = (act.lifecycle_event_id, alloc.successor_instrument_id)
                    prospective_entitlements[ent_key] = Decimal(ent_qty) * alloc.ratio
                    prospective_costs[ent_key] = ent_total * alloc.cost_basis_weight
                prospective_allocs[act.lifecycle_event_id] = act.successor_allocations
                prospective_successor_ids.add(act.lifecycle_event_id)
                payload_entitlement: tuple[tuple[str, object], ...] = (
                    ("action_type", act.action_type.value),
                    ("instrument_id", act.instrument_id),
                    ("lifecycle_event_id", act.lifecycle_event_id),
                    ("quantity", ent_qty),
                )
                new_entries.append(
                    LedgerJournalEntry(
                        event_id=act.action_id,
                        event_type=act.action_type.value,
                        event_time=session_open,
                        payload=payload_entitlement,
                    )
                )
            elif act.action_type == LedgerActionType.SUCCESSOR_DELIVERY:
                assert act.lifecycle_event_id is not None
                for alloc in act.successor_allocations:
                    delivery_key = (act.lifecycle_event_id, alloc.successor_instrument_id)
                    entitled_qty = prospective_entitlements.get(delivery_key, Decimal("0"))
                    carried_cost = prospective_costs.get(delivery_key, Decimal("0"))
                    integral_qty = int(entitled_qty)
                    residual_qty = entitled_qty - Decimal(integral_qty)
                    if integral_qty > 0 and entitled_qty > 0:
                        unit_cost = carried_cost / entitled_qty
                        prev_qty, prev_avg = prospective_positions.get(alloc.successor_instrument_id, (0, 0.0))
                        combined_qty = prev_qty + integral_qty
                        combined_cost = Decimal(str(prev_avg)) * Decimal(prev_qty) + unit_cost * Decimal(integral_qty)
                        prospective_positions[alloc.successor_instrument_id] = (combined_qty, float(combined_cost / Decimal(combined_qty)))
                    prospective_entitlements[delivery_key] = residual_qty
                    if entitled_qty > 0:
                        prospective_costs[delivery_key] = carried_cost / entitled_qty * residual_qty
                    else:  # pragma: no cover - entitlement prevalidation requires a positive ratio
                        prospective_costs[delivery_key] = Decimal("0")
                prospective_successor_ids.add(act.lifecycle_event_id)
                payload_delivery: tuple[tuple[str, object], ...] = (
                    ("action_type", act.action_type.value),
                    ("instrument_id", act.instrument_id),
                    ("lifecycle_event_id", act.lifecycle_event_id),
                )
                new_entries.append(
                    LedgerJournalEntry(
                        event_id=act.action_id,
                        event_type=act.action_type.value,
                        event_time=session_open,
                        payload=payload_delivery,
                    )
                )
            elif act.action_type == LedgerActionType.CASH_IN_LIEU_SETTLEMENT:
                assert act.lifecycle_event_id is not None
                assert act.settlement_instrument_id is not None
                per_unit = act.cash_settlement_per_entitlement_unit
                assert isinstance(per_unit, Decimal)
                residual_key = (act.lifecycle_event_id, act.settlement_instrument_id)
                residual_qty = prospective_entitlements.get(residual_key, Decimal("0"))
                cash_credit = float(residual_qty * per_unit)
                prospective_settled += cash_credit
                prospective_entitlements.pop(residual_key, None)
                prospective_costs.pop(residual_key, None)
                prospective_successor_ids.add(act.lifecycle_event_id)
                payload_settlement: tuple[tuple[str, object], ...] = (
                    ("action_type", act.action_type.value),
                    ("instrument_id", act.instrument_id),
                    ("lifecycle_event_id", act.lifecycle_event_id),
                    ("settlement_instrument_id", act.settlement_instrument_id),
                    ("cash", float(cash_credit)),
                )
                new_entries.append(
                    LedgerJournalEntry(
                        event_id=act.action_id,
                        event_type=act.action_type.value,
                        event_time=session_open,
                        payload=payload_settlement,
                    )
                )
            elif act.action_type == LedgerActionType.DELISTING_UNSETTLED:
                from src.core.pit import PITDataError as _PITDataError

                qty, _ = opening_snapshot.get(act.instrument_id, (0, 0.0))
                if qty != 0:
                    raise _PITDataError(f"unsettled delisting with open position for {act.instrument_id!r}")
                payload_unsettled: tuple[tuple[str, object], ...] = (
                    ("action_type", act.action_type.value),
                    ("instrument_id", act.instrument_id),
                    ("cash_amount", 0.0),
                    ("quantity", qty),
                )
                entry_unsettled = LedgerJournalEntry(
                    event_id=act.action_id,
                    event_type=act.action_type.value,
                    event_time=session_open,
                    payload=payload_unsettled,
                )
                if entry_unsettled.event_id in self._journal_ids:
                    raise ValueError(f"duplicate journal event_id {entry_unsettled.event_id!r}")  # pragma: no cover
                new_entries.append(entry_unsettled)
                prospective_positions.pop(act.instrument_id, None)
                if act.lifecycle_event_id:
                    prospective_delisting_ids.add(act.lifecycle_event_id)
            elif act.action_type == LedgerActionType.DELISTING_CASH_OUT:
                qty, _ = opening_snapshot.get(act.instrument_id, (0, 0.0))
                credit = qty * float(act.cash_amount)
                prospective_settled += credit
                valuation_source = "disclosed_settlement" if float(act.cash_amount) > 0 else "final_close"
                payload_delist: tuple[tuple[str, object], ...] = (
                    ("action_type", act.action_type.value),
                    ("instrument_id", act.instrument_id),
                    ("cash_amount", float(act.cash_amount)),
                    ("quantity", qty),
                    ("valuation_source", valuation_source),
                )
                entry_delist = LedgerJournalEntry(
                    event_id=act.action_id,
                    event_type=act.action_type.value,
                    event_time=session_open,
                    payload=payload_delist,
                )
                if entry_delist.event_id in self._journal_ids:
                    raise ValueError(f"duplicate journal event_id {entry_delist.event_id!r}")  # pragma: no cover
                new_entries.append(entry_delist)
                prospective_positions.pop(act.instrument_id, None)
                if act.lifecycle_event_id:
                    prospective_delisting_ids.add(act.lifecycle_event_id)
            elif act.action_type == LedgerActionType.DIVIDEND:
                qty, _ = opening_snapshot.get(act.instrument_id, (0, 0.0))
                if qty > 0 and float(act.cash_amount) > 0:
                    credit = qty * float(act.cash_amount)
                    prospective_settled += credit
                payload: tuple[tuple[str, object], ...] = (
                    ("action_type", act.action_type.value),
                    ("instrument_id", act.instrument_id),
                    ("cash_amount", float(act.cash_amount)),
                    ("quantity", qty),
                )
                entry = LedgerJournalEntry(
                    event_id=act.action_id,
                    event_type="dividend",
                    event_time=session_open,
                    payload=payload,
                )
                if entry.event_id in self._journal_ids:
                    raise ValueError(f"duplicate journal event_id {entry.event_id!r}")
                new_entries.append(entry)
            else:
                qty, avg = opening_snapshot.get(act.instrument_id, (0, 0.0))
                if qty == 0:
                    payload = (
                        ("action_type", act.action_type.value),
                        ("instrument_id", act.instrument_id),
                        ("factor", float(act.factor)),
                        ("old_quantity", 0),
                        ("new_quantity", 0),
                    )
                    entry = LedgerJournalEntry(
                        event_id=act.action_id,
                        event_type=act.action_type.value,
                        event_time=session_open,
                        payload=payload,
                    )
                    new_entries.append(entry)
                    continue
                raw_new = qty * float(act.factor)
                retained = math.floor(raw_new + 1e-9)
                fractional = raw_new - retained
                if fractional < 1e-9:
                    fractional = 0.0
                # handle floating point near integer
                if abs(fractional) < 1e-9:
                    fractional = 0.0
                price = float(cash_in_lieu_prices[act.instrument_id])
                cash_lieu = fractional * price
                prospective_settled += cash_lieu
                new_avg = float(avg) / float(act.factor) if float(act.factor) != 0 else float(avg)
                if retained == 0:
                    prospective_positions.pop(act.instrument_id, None)
                else:
                    prospective_positions[act.instrument_id] = (int(retained), new_avg)
                # ensure retained is valid lot (positive multiple) - lot 1 assumed
                if int(retained) % 1 != 0:
                    raise ValueError("invalid lot remainder")
                payload_action: tuple[tuple[str, object], ...] = (
                    ("action_type", act.action_type.value),
                    ("instrument_id", act.instrument_id),
                    ("factor", float(act.factor)),
                    ("old_quantity", qty),
                    ("new_quantity", int(retained)),
                    ("fractional", float(fractional)),
                )
                entry_action = LedgerJournalEntry(
                    event_id=act.action_id,
                    event_type=act.action_type.value,
                    event_time=session_open,
                    payload=payload_action,
                )
                new_entries.append(entry_action)
                if fractional > 1e-12:
                    lieu_id = f"{act.action_id}:cash_in_lieu"
                    if lieu_id in self._journal_ids:
                        raise ValueError(f"duplicate journal event_id {lieu_id!r}")
                    payload_lieu: tuple[tuple[str, object], ...] = (
                        ("instrument_id", act.instrument_id),
                        ("fractional", float(fractional)),
                        ("price", price),
                        ("cash", float(cash_lieu)),
                    )
                    entry_lieu = LedgerJournalEntry(
                        event_id=lieu_id,
                        event_type="cash_in_lieu",
                        event_time=session_open,
                        payload=payload_lieu,
                    )
                    if lieu_id in seen_in_batch:
                        raise ValueError(f"duplicate journal event_id {lieu_id!r}")
                    new_entries.append(entry_lieu)
        # commit
        self._settled_cash = prospective_settled
        self._positions = prospective_positions
        self._entitlements = prospective_entitlements
        self._entitlement_costs = prospective_costs
        self._entitlement_allocations = prospective_allocs
        self._lifecycle_successor_ids = prospective_successor_ids
        self._lifecycle_delisting_ids = prospective_delisting_ids
        for e in new_entries:
            if e.event_id in self._journal_ids:
                raise ValueError(f"duplicate journal event_id {e.event_id!r}")
            self._journal.append(e)
            self._journal_ids.add(e.event_id)
        for act in actions:
            self._action_ids.add(act.action_id)
        # also mark lieu ids as seen for future duplicate check
        for e in new_entries:
            if e.event_type == "cash_in_lieu":
                self._action_ids.add(e.event_id)
        if session_open > self._latest_time:
            self._latest_time = session_open
        return tuple(new_entries)

    def record_mark(self, mark: LedgerMark) -> LedgerNav:
        if not mark.mark_id:
            raise ValueError("mark_id must be non-empty")
        if mark.mark_id in self._journal_ids or mark.mark_id in self._mark_ids:
            raise ValueError(f"duplicate mark_id {mark.mark_id!r}")
        if mark.as_of.tzinfo is None:
            raise ValueError("as_of must be aware")
        if mark.as_of < self._latest_time:
            raise ValueError("mark as_of must be nondecreasing")
        price_map: dict[str, float] = {}
        for instr, price in mark.prices:
            if not instr:
                raise ValueError("instrument_id must be non-empty")
            if instr in price_map:
                raise ValueError(f"duplicate price for {instr!r}")
            if isinstance(price, bool) or not isinstance(price, (int, float)):
                raise ValueError("price must be finite")
            if not math.isfinite(float(price)) or float(price) <= 0:
                raise ValueError("price must be positive finite")
            price_map[instr] = float(price)
        position_instrs = set(self._positions.keys())
        price_instrs = set(price_map.keys())
        if position_instrs != price_instrs:
            raise ValueError(f"mark must cover every open position exactly once: {position_instrs!r} vs {price_instrs!r}")
        marked_value = 0.0
        for instr, (qty, _) in self._positions.items():
            marked_value += qty * price_map[instr]
        nav_val = float(self._settled_cash) + float(self._unsettled_cash) + marked_value
        nav = LedgerNav(
            mark_id=mark.mark_id,
            as_of=mark.as_of,
            nav=nav_val,
            settled_cash=float(self._settled_cash),
            unsettled_cash=float(self._unsettled_cash),
            marked_value=marked_value,
        )
        payload: tuple[tuple[str, object], ...] = (
            ("mark_id", mark.mark_id),
            ("as_of", mark.as_of.isoformat()),
            ("nav", nav_val),
        )
        entry = LedgerJournalEntry(
            event_id=mark.mark_id,
            event_type="mark",
            event_time=mark.as_of,
            payload=payload,
        )
        if entry.event_id in self._journal_ids:
            raise ValueError(f"duplicate journal event_id {entry.event_id!r}")
        self._journal.append(entry)
        self._journal_ids.add(entry.event_id)
        self._mark_ids.add(mark.mark_id)
        if mark.as_of > self._latest_time:
            self._latest_time = mark.as_of
        return nav

    def journal(self) -> tuple[LedgerJournalEntry, ...]:
        return tuple(self._journal)
