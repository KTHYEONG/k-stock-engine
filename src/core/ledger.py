"""Deterministic long-only cash-equity ledger."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class LedgerSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class LedgerActionType(StrEnum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND = "dividend"


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
class LedgerCorporateAction:
    action_id: str
    instrument_id: str
    action_type: LedgerActionType
    effective_time: datetime
    factor: float
    cash_amount: float

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
            if act.action_type == LedgerActionType.DIVIDEND:
                if float(act.factor) != 1.0:
                    raise ValueError("invalid factor for dividend")
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
        new_entries: list[LedgerJournalEntry] = []
        # handle dividend and splits from opening snapshot
        for act in actions:
            if act.action_type == LedgerActionType.DIVIDEND:
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
