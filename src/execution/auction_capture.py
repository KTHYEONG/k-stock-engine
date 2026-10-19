"""Pre-open expected-price snapshots and confirmed open prints for shadow fills."""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from src.core.time import OPEN_SNAPSHOT_OFFSETS_MIN

__all__ = [
    "OPEN_CONFIRM_FROM_S",
    "OPEN_CONFIRM_UNTIL_S",
    "OPEN_SNAPSHOT_OFFSETS_MIN",
    "SLOT_DEADLINE_ERROR",
    "AuctionCapturePolicy",
    "AuctionRow",
    "CaptureBatch",
    "CaptureResult",
    "capture_open_auction",
    "capture_symbols",
    "write_auction_rows",
]

OPEN_CONFIRM_FROM_S: Final = 30
OPEN_CONFIRM_UNTIL_S: Final = 180

_OPEN_CONFIRM_MAX_ATTEMPTS: Final = 6
_OPEN_RETRY_PAUSE_S: Final = 1.0

SLOT_DEADLINE_ERROR: Final = "SlotDeadline"


@dataclass(frozen=True, slots=True)
class AuctionRow:
    session: date
    symbol: str
    kind: str
    scheduled_at: datetime
    received_at: datetime
    output1: str
    output2: str


@dataclass(frozen=True, slots=True)
class AuctionCapturePolicy:
    call_estimate_s: float
    slot_margin_s: float

    def __post_init__(self) -> None:
        for name, value in (
            ("call_estimate_s", self.call_estimate_s),
            ("slot_margin_s", self.slot_margin_s),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if not float(self.call_estimate_s) > 0:
            raise ValueError(f"call_estimate_s must be > 0, got {self.call_estimate_s!r}")
        if not float(self.slot_margin_s) >= 0:
            raise ValueError(f"slot_margin_s must be >= 0, got {self.slot_margin_s!r}")


@dataclass(frozen=True, slots=True)
class CaptureBatch:
    kind: str
    scheduled_at: datetime
    rows: tuple[AuctionRow, ...]


@dataclass(frozen=True, slots=True)
class CaptureResult:
    rows: tuple[AuctionRow, ...]
    failed: Mapping[str, str]
    slots_expected: int
    slots_captured: int


def _ticker(name: str) -> str:
    text = str(name).strip()
    return text.split(":")[-1] if ":" in text else text


def capture_symbols(
    chain: Sequence[Any], holdings: Collection[str], *, session: date, limit: int = 80
) -> tuple[str, ...]:
    """Tickers to capture: names whose sealed weight changes into this session, then current holdings, capped."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"limit must be an int >= 1, got {limit!r}")
    ordered = sorted(chain, key=lambda item: item.data_session)
    current: Any | None = None
    for decision in ordered:
        if decision.execution_session == session and (
            current is None or decision.data_session > current.data_session
        ):
            current = decision
    held = sorted({_ticker(name) for name in holdings})
    if current is None:
        candidates = [item for item in ordered if item.execution_session <= session]
        if not candidates:
            return ()
        latest = max(candidates, key=lambda item: item.data_session)
        weights = { _ticker(key): float(value) for key, value in dict(latest.targets).items()}
        return tuple(
            ticker
            for ticker in sorted(set(held) | set(weights), key=lambda name: (-weights.get(name, 0.0), name))
        )[: int(limit)]
    previous: Any | None = None
    for decision in ordered:
        if decision.data_session < current.data_session and (
            previous is None or decision.data_session > previous.data_session
        ):
            previous = decision
    old = {str(key): float(value) for key, value in dict(previous.targets).items()} if previous is not None else {}
    new = {str(key): float(value) for key, value in dict(current.targets).items()}
    deltas: dict[str, float] = {}
    for key in set(old) | set(new):
        move = abs(float(new.get(key, 0.0)) - float(old.get(key, 0.0)))
        if move != 0.0:
            deltas[_ticker(key)] = move
    ranked = sorted(deltas, key=lambda name: (-deltas[name], name))
    picked = list(ranked[: int(limit)])
    taken = set(picked)
    fresh_weights = {_ticker(key): float(value) for key, value in new.items()}
    rest = sorted(
        (name for name in held if name not in taken),
        key=lambda name: (-fresh_weights.get(name, 0.0), name),
    )
    picked.extend(rest[: max(0, int(limit) - len(picked))])
    return tuple(picked)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def capture_open_auction(
    client: Any,
    symbols: Sequence[str],
    *,
    session: date,
    open_at: datetime,
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
    policy: AuctionCapturePolicy,
    on_batch: Callable[[CaptureBatch], None] | None = None,
) -> CaptureResult:
    """Capture expected-price slots then the open print; emits one batch per finished slot and one for the open phase."""
    names = tuple(str(name) for name in symbols)
    if not names:
        return CaptureResult(rows=(), failed={}, slots_expected=0, slots_captured=0)
    if len(set(names)) != len(names):
        raise ValueError("symbols must be unique")
    if open_at.tzinfo is None:
        raise ValueError("open_at must be timezone-aware")
    rows: list[AuctionRow] = []
    failed: dict[str, str] = {}
    offsets = sorted(OPEN_SNAPSHOT_OFFSETS_MIN)
    scheduled_list = [open_at + timedelta(minutes=int(offset)) for offset in offsets]
    started = now()
    expected_idx = [i for i, scheduled in enumerate(scheduled_list) if scheduled >= started]
    slots_expected = len(expected_idx)
    slots_captured = 0
    duration_total = 0.0
    calls_completed = 0

    for idx in expected_idx:
        scheduled = scheduled_list[idx]
        deadline = scheduled_list[idx + 1] if idx + 1 < len(scheduled_list) else open_at
        wait = (scheduled - now()).total_seconds()
        if wait > 0:
            sleep(wait)
        slot_rows: list[AuctionRow] = []
        for order, symbol in enumerate(names):
            observed_mean = duration_total / calls_completed if calls_completed else 0.0
            estimate = max(policy.call_estimate_s, observed_mean)
            stamp = now()
            if estimate + policy.slot_margin_s > (deadline - stamp).total_seconds():
                for tail in names[order:]:
                    failed.setdefault(tail, SLOT_DEADLINE_ERROR)
                break
            call_start = stamp
            try:
                try:
                    payload = client.inquire_asking_price_exp_ccn(symbol)
                finally:
                    received = now()
                    duration_total += max(0.0, (received - call_start).total_seconds())
                    calls_completed += 1
                if not isinstance(payload, Mapping):
                    raise ValueError("expected auction payload must be a mapping")
                row = AuctionRow(
                    session=session,
                    symbol=symbol,
                    kind="expected",
                    scheduled_at=scheduled,
                    received_at=received,
                    output1=json.dumps(payload.get("output1", {}), sort_keys=True),
                    output2=json.dumps(payload.get("output2", []), sort_keys=True),
                )
            except Exception as exc:
                failed.setdefault(symbol, type(exc).__name__)
                continue
            slot_rows.append(row)
            rows.append(row)
        if slot_rows:
            slots_captured += 1
            if on_batch is not None:
                on_batch(CaptureBatch(kind="expected", scheduled_at=scheduled, rows=tuple(slot_rows)))
    confirm_from = open_at + timedelta(seconds=int(OPEN_CONFIRM_FROM_S))
    deadline = open_at + timedelta(seconds=int(OPEN_CONFIRM_UNTIL_S))
    wait = (confirm_from - now()).total_seconds()
    if wait > 0:
        sleep(wait)
    attempts = dict.fromkeys(names, 0)
    first_open_failure: dict[str, str] = {}
    unresolved = list(names)
    while unresolved:
        if now() >= deadline:
            break
        for symbol in list(unresolved):
            if now() >= deadline:
                break
            attempts[symbol] += 1
            try:
                output = client.inquire_price(symbol)
                if not isinstance(output, Mapping):
                    raise ValueError("open print payload must be a mapping")
                received = now()
                if _positive_int(output.get("stck_oprc")) is None:
                    continue
                row = AuctionRow(
                    session=session,
                    symbol=symbol,
                    kind="open",
                    scheduled_at=open_at,
                    received_at=received,
                    output1=json.dumps(dict(output), sort_keys=True),
                    output2=json.dumps(output.get("output2", {}), sort_keys=True),
                )
            except Exception as exc:
                first_open_failure.setdefault(symbol, type(exc).__name__)
                continue
            unresolved.remove(symbol)
            rows.append(row)
        if unresolved and now() < deadline and any(attempts[symbol] < _OPEN_CONFIRM_MAX_ATTEMPTS for symbol in unresolved):
            sleep(_OPEN_RETRY_PAUSE_S)
        elif unresolved and now() < deadline:
            sleep(max(0.0, (deadline - now()).total_seconds()))
            break
    for symbol in unresolved:
        if symbol in first_open_failure:
            failed.setdefault(symbol, first_open_failure[symbol])
    open_rows = [row for row in rows if row.kind == "open" and row.scheduled_at == open_at]
    if open_rows and on_batch is not None:
        on_batch(CaptureBatch(kind="open", scheduled_at=open_at, rows=tuple(open_rows)))
    return CaptureResult(rows=tuple(rows), failed=dict(failed), slots_expected=slots_expected, slots_captured=slots_captured)


_AUCTION_SCHEMA = {
    "session": "date",
    "symbol": "str",
    "kind": "str",
    "scheduled_at": "dt",
    "received_at": "dt",
    "output1": "str",
    "output2": "str",
}


def write_auction_rows(root: Path, rows: Sequence[AuctionRow]) -> Path:
    """Merge rows into root/<YYYY-MM>/<session>.parquet atomically, deduplicated on (symbol, kind, scheduled_at); returns the path."""
    import polars as pl

    items = list(rows)
    if not items:
        raise ValueError("no auction rows to write")
    sessions = {item.session for item in items}
    if len(sessions) != 1:
        raise ValueError("auction rows must share one session")
    session = items[0].session
    target = Path(root) / session.strftime("%Y-%m") / f"{session.isoformat()}.parquet"
    frame = pl.DataFrame(
        {
            "session": [item.session for item in items],
            "symbol": [item.symbol for item in items],
            "kind": [item.kind for item in items],
            "scheduled_at": [item.scheduled_at for item in items],
            "received_at": [item.received_at for item in items],
            "output1": [item.output1 for item in items],
            "output2": [item.output2 for item in items],
        },
        schema={
            "session": pl.Date,
            "symbol": pl.String,
            "kind": pl.String,
            "scheduled_at": pl.Datetime("us", "Asia/Seoul"),
            "received_at": pl.Datetime("us", "Asia/Seoul"),
            "output1": pl.String,
            "output2": pl.String,
        },
    )
    if target.is_file():
        combined = pl.concat([pl.read_parquet(target), frame], how="vertical")
    else:
        combined = frame
    combined = combined.unique(subset=["symbol", "kind", "scheduled_at"], keep="first").sort(
        ["scheduled_at", "symbol", "kind"]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        combined.write_parquet(temporary)
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return target
