"""Pre-open expected-price snapshots and confirmed open prints for shadow fills."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from src.core.time import OPEN_SNAPSHOT_OFFSETS_MIN

__all__ = [
    "OPEN_CONFIRM_FROM_S",
    "OPEN_CONFIRM_UNTIL_S",
    "OPEN_SNAPSHOT_OFFSETS_MIN",
    "AuctionRow",
    "capture_open_auction",
    "capture_symbols",
    "write_auction_rows",
]

OPEN_CONFIRM_FROM_S: Final = 30
OPEN_CONFIRM_UNTIL_S: Final = 180

_OPEN_CONFIRM_MAX_ATTEMPTS: Final = 6
_OPEN_RETRY_PAUSE_S: Final = 1.0


@dataclass(frozen=True, slots=True)
class AuctionRow:
    session: date
    symbol: str
    kind: str
    scheduled_at: datetime
    received_at: datetime
    output1: str
    output2: str


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
) -> tuple[AuctionRow, ...]:
    """Poll expected-price snapshots before the open and the confirmed open print after it."""
    names = tuple(str(name) for name in symbols)
    if not names:
        return ()
    if len(set(names)) != len(names):
        raise ValueError("symbols must be unique")
    if open_at.tzinfo is None:
        raise ValueError("open_at must be timezone-aware")
    rows: list[AuctionRow] = []
    started = now()
    for offset in sorted(OPEN_SNAPSHOT_OFFSETS_MIN):
        scheduled = open_at + timedelta(minutes=int(offset))
        if scheduled < started:
            continue
        wait = (scheduled - now()).total_seconds()
        if wait > 0:
            sleep(wait)
        for symbol in names:
            payload = client.inquire_asking_price_exp_ccn(symbol)
            if not isinstance(payload, dict):
                raise ValueError(f"expected auction payload for {symbol!r} must be a mapping")
            rows.append(
                AuctionRow(
                    session=session,
                    symbol=symbol,
                    kind="expected",
                    scheduled_at=scheduled,
                    received_at=now(),
                    output1=json.dumps(payload.get("output1", {}), sort_keys=True),
                    output2=json.dumps(payload.get("output2", []), sort_keys=True),
                )
            )
    confirm_from = open_at + timedelta(seconds=int(OPEN_CONFIRM_FROM_S))
    deadline = open_at + timedelta(seconds=int(OPEN_CONFIRM_UNTIL_S))
    wait = (confirm_from - now()).total_seconds()
    if wait > 0:
        sleep(wait)
    attempts = dict.fromkeys(names, 0)
    unresolved = list(names)
    while unresolved:
        if now() >= deadline:
            break
        for symbol in list(unresolved):
            if now() >= deadline:
                break
            attempts[symbol] += 1
            output = client.inquire_price(symbol)
            if not isinstance(output, dict):
                raise ValueError(f"open print payload for {symbol!r} must be a mapping")
            received = now()
            if _positive_int(output.get("stck_oprc")) is None:
                continue
            unresolved.remove(symbol)
            rows.append(
                AuctionRow(
                    session=session,
                    symbol=symbol,
                    kind="open",
                    scheduled_at=open_at,
                    received_at=received,
                    output1=json.dumps(output, sort_keys=True),
                    output2=json.dumps(output.get("output2", {}), sort_keys=True),
                )
            )
        if unresolved and now() < deadline and any(attempts[symbol] < _OPEN_CONFIRM_MAX_ATTEMPTS for symbol in unresolved):
            sleep(_OPEN_RETRY_PAUSE_S)
        elif unresolved and now() < deadline:
            sleep(max(0.0, (deadline - now()).total_seconds()))
            break
    return tuple(rows)


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
    """Append rows to root/<YYYY-MM>/<session>.parquet atomically; returns the file path."""
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
        combined = pl.concat([pl.read_parquet(target), frame], how="vertical").sort(
            ["scheduled_at", "symbol", "kind"]
        )
    else:
        combined = frame.sort(["scheduled_at", "symbol", "kind"])
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
