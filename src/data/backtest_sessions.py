"""PIT backtest session builder from certified Silver snapshots."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import polars as pl

from src.core.instruments import AssetKind, Instrument
from src.core.time import SessionCalendar
from src.data.schemas import PITDataError, SilverTable
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
    required_cols = {"session", "instrument_id", "open", "close", "volume", "trading_value", "available_at"}
    missing_cols = [c for c in required_cols if c not in full.columns]
    if missing_cols:
        raise PITDataError(f"daily market missing columns: {missing_cols}")
    if full.select(["session", "instrument_id"]).is_duplicated().any():
        raise PITDataError("duplicate bar for (session, instrument_id) pair")
    bad_mask = (
        pl.col("open").is_null()
        | (pl.col("open") <= 0)
        | pl.col("close").is_null()
        | (pl.col("close") <= 0)
    )
    if full.select(bad_mask.any().alias("_bad")).item(0, 0):
        raise PITDataError("non-positive price bar detected (open or close <= 0)")
    distinct_sessions = full.select("session").unique()["session"].to_list()
    decision_times = [decision_time_of(s) for s in distinct_sessions]
    decision_frame = pl.DataFrame({"session": distinct_sessions, "decision_time": decision_times})
    joined = full.join(decision_frame, on="session", how="left")
    if joined.select((pl.col("available_at") > pl.col("decision_time")).any().alias("_leak")).item(0, 0):
        raise PITDataError("available_at after decision time detected")
    partitioned = full.partition_by("session", as_dict=True)
    by_session = {k[0] if isinstance(k, tuple) else k: v for k, v in partitioned.items()}
    sessions: list[BacktestSession] = []
    for session_open in decisions:
        decision_time = decision_time_of(session_open)
        if decision_time.tzinfo is None:
            raise PITDataError("decision_time must be timezone-aware")
        idx = ordered.index(session_open)
        next_open = ordered[idx + 1] if idx + 1 < len(ordered) else None
        if next_open is None or next_open not in by_session:
            raise PITDataError(f"missing next session bar after {session_open}")
        part = by_session[session_open]
        instrument_ids = part.get_column("instrument_id").to_list()
        opens = part.get_column("open").cast(pl.Float64).to_list()
        closes = part.get_column("close").cast(pl.Float64).to_list()
        trading_values = part.get_column("trading_value").cast(pl.Float64).to_list()
        order = sorted(range(len(instrument_ids)), key=lambda i: str(instrument_ids[i]))
        bars = tuple(
            HistoricalBar(
                session_open,
                str(instrument_ids[i]),
                float(opens[i]),
                float(closes[i]),
                float(trading_values[i]),
                0.02,
            )
            for i in order
        )
        mark_prices = {str(instrument_ids[i]): float(closes[i]) for i in range(len(instrument_ids))}
        adtv20 = {str(instrument_ids[i]): float(trading_values[i]) for i in range(len(instrument_ids))}
        volatilities = dict.fromkeys(mark_prices, 0.02)
        sectors = dict.fromkeys(mark_prices, "KRX")
        instruments: dict[str, Any] = {
            iid: Instrument(iid, AssetKind.STOCK, "KRX", iid.split(":")[-1], "KRW") for iid in mark_prices
        }
        market_snapshot: dict[str, Any] = {
            "mark_prices": mark_prices,
            "adtv20": adtv20,
            "volatilities": volatilities,
            "sectors": sectors,
            "instruments": instruments,
            "market_volatility": 0.02,
        }
        sessions.append(
            BacktestSession(
                session_open=session_open,
                decision_time=decision_time,
                bars=bars,
                actions=(),
                market_snapshot=market_snapshot,
            )
        )
    return tuple(sessions)
