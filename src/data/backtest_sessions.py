"""PIT backtest session builder from certified Silver snapshots."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import polars as pl

from src.core.time import SessionCalendar
from src.data.schemas import PITDataError, PITSnapshotRequest, SilverTable
from src.data.snapshot import PITSnapshotRepository
from src.engine.backtest import BacktestSession
from src.engine.fill_model import HistoricalBar


def _frame_for(repository: PITSnapshotRepository) -> pl.DataFrame | None:
    frames = getattr(repository, "_frames", {})
    frame = frames.get(SilverTable.DAILY_MARKET)
    return frame


def build_backtest_sessions(
    *,
    snapshot_repository: PITSnapshotRepository,
    calendar: SessionCalendar,
    start: datetime,
    end: datetime,
    decision_time_of: Callable[[datetime], datetime],
) -> tuple[BacktestSession, ...]:
    if start.tzinfo is None or end.tzinfo is None:
        raise PITDataError("start and end must be timezone-aware")
    if start > end:
        raise PITDataError("start must not be after end")
    ordered = tuple(sorted(calendar.sessions))
    decisions = tuple(s for s in ordered if start <= s <= end)
    if not decisions:
        raise PITDataError("no sessions in requested range")
    full = _frame_for(snapshot_repository)
    if full is None or full.height == 0:
        raise PITDataError("missing daily market bars for requested coverage")
    required_cols = {"session", "instrument_id", "open", "close", "available_at"}
    missing_cols = [c for c in required_cols if c not in full.columns]
    if missing_cols:
        raise PITDataError(f"daily market missing columns: {missing_cols}")
    grouped: dict[tuple[datetime, str], list[dict[str, Any]]] = {}
    for row in full.to_dicts():
        sess = row.get("session")
        iid = row.get("instrument_id")
        if sess is None or iid is None:
            raise PITDataError("unknown instrument in daily market")
        if not isinstance(iid, str) or not iid.strip():
            raise PITDataError("unknown instrument in daily market")
        key = (sess, iid)
        grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        if len(rows) > 1:
            raise PITDataError(f"duplicate bar for {key[1]!r} at {key[0]}")
    sessions: list[BacktestSession] = []
    for session_open in decisions:
        decision_time = decision_time_of(session_open)
        if decision_time.tzinfo is None:
            raise PITDataError("decision_time must be timezone-aware")
        request = PITSnapshotRequest(decision_time=decision_time, required_tables=frozenset({SilverTable.DAILY_MARKET}))
        try:
            snap = snapshot_repository.snapshot(request)
        except PITDataError as exc:
            raise PITDataError(str(exc)) from exc
        frame = snap.get(SilverTable.DAILY_MARKET)
        if frame is None or frame.height == 0:
            raise PITDataError(f"missing bar for decision session {session_open}")
        for row in frame.to_dicts():
            avail = row.get("available_at")
            if avail is not None and getattr(avail, "tzinfo", None) is not None and avail > decision_time:
                raise PITDataError(f"available_at after decision time for {row.get('instrument_id')}")
        idx = ordered.index(session_open)
        if idx + 1 >= len(ordered):
            raise PITDataError(f"missing next session bar after {session_open}")
        next_open = ordered[idx + 1]
        next_rows = [r for (s, _), r in grouped.items() if s == next_open]
        if not next_rows:
            raise PITDataError(f"missing next session bar after {session_open}")
        bars: list[HistoricalBar] = []
        for row in frame.to_dicts():
            if row.get("session") != session_open:
                continue
            iid = str(row["instrument_id"])
            raw_open = float(row["open"])
            raw_close = float(row["close"])
            tv = row.get("trading_value", 1_000_000.0)
            try:
                adtv = float(tv) if tv is not None else 1_000_000.0
            except (TypeError, ValueError):
                adtv = 1_000_000.0
            if adtv <= 0:
                raise PITDataError(f"missing/non-positive adtv for {iid}")
            if raw_open <= 0 or raw_close <= 0:
                raise PITDataError(f"missing/non-positive raw open/close for {iid}")
            bars.append(HistoricalBar(session_open, iid, raw_open, raw_close, adtv, 0.02))
        if not bars:
            raise PITDataError(f"missing bar for decision session {session_open}")
        bars = sorted(bars, key=lambda b: b.instrument_id)
        sessions.append(
            BacktestSession(
                session_open=session_open,
                decision_time=decision_time,
                bars=tuple(bars),
                actions=(),
                market_snapshot=object(),
            )
        )
    return tuple(sessions)
