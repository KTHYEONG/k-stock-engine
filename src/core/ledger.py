"""Deterministic long-only cash-equity ledger."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class LedgerSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


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
        if fill.trade_time > self._latest_time:
            self._latest_time = fill.trade_time

    def settle(self, as_of: datetime) -> None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be aware")
        if as_of < self._latest_time:
            raise ValueError("as_of must be at or after latest event")
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
