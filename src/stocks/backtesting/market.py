"""Immutable, array-backed prepared replay market.

``PreparedReplayMarket`` is the canonical sparse encoded market/index shared by
every replay candidate: one canonical ``(session, instrument_id)`` row order,
aligned ``float64`` execution arrays, causal rolling ADTV/volatility,
per-session row ranges, and an ``O(1)`` ``(instrument_id, session)`` key
lookup. It owns no subclass relationship to the engine; the engine imports it
from here.

``cache_bytes`` accounts the deduplicated live state actually retained after
the build (NumPy buffers, session/index mappings, Python key/row objects);
transient build-time frames are excluded because they are released.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import numpy as np
import polars as pl

from src.core.instruments import Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.contracts import (
    REQUIRED_BACKTEST_COLUMNS,
    ArtifactSchedule,
    BacktestValidationError,
)

# Conservative per-entry estimate for one ``(str, datetime) -> _PreparedRow``
# dictionary entry: hash-slot pointer + tuple header + datetime payload +
# row-instance slot. Shared instrument strings are accounted once below.
_ROW_KEY_ENTRY_BYTES = 208


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise BacktestValidationError(f"non-datetime session value: {value!r}")


@dataclass(frozen=True, slots=True)
class _PreparedRow:
    """Array-backed replay row: ``get``/``[]`` resolve from the market."""

    market: PreparedReplayMarket
    index: int

    def get(self, key: str, default: object = None) -> object:
        return self.market.value_at(self.index, key, default)

    def __getitem__(self, key: str) -> object:
        return self.market.value_at(self.index, key)


@dataclass(frozen=True, slots=True)
class PreparedReplayMarket:
    """Immutable, array-backed replay market shared by every candidate.

    Built once per training snapshot, the market owns the canonical sorted
    ``session``/``instrument_id`` row index and aligned ``float64`` arrays for
    execution fields, close returns, trading value, rolling ADTV, and rolling
    volatility, so a candidate contributes only an aligned score overlay instead
    of re-partitioning frames and recomputing market-wide statistics per replay.
    ``session_ranges`` maps each session index to its contiguous row range;
    ``rows_by_key`` resolves ``(instrument_id, session)`` rows in ``O(1)``.
    """

    sessions: tuple[datetime, ...]
    session_ranges: Mapping[int, tuple[int, int]]
    session_max_available_time: Mapping[int, datetime]
    rows_by_key: Mapping[tuple[str, datetime], _PreparedRow]
    instrument_ids: np.ndarray
    row_session_of: np.ndarray
    close: np.ndarray
    open_: np.ndarray
    volume: np.ndarray
    trading_value: np.ndarray
    adtv: np.ndarray
    volatility: np.ndarray
    has_volatility: bool
    available_time: np.ndarray
    limit_locked: np.ndarray | None
    action_interval_covered: np.ndarray | None
    close_returns: np.ndarray
    instruments: Mapping[str, Instrument]
    artifacts: ArtifactSchedule | None
    initial_portfolio: PortfolioSnapshot | None
    cache_bytes: int

    #: Actual ``build`` invocations observed process-wide (test/budget hook).
    build_call_count: ClassVar[int] = 0

    @classmethod
    def reset_build_call_count(cls) -> None:
        cls.build_call_count = 0

    @property
    def row_count(self) -> int:
        return int(self.instrument_ids.size)

    def value_at(self, index: int, key: str, default: object = None) -> object:
        """Return the aligned column value for one market row."""
        if key == "instrument_id":
            return self.instrument_ids[index]
        if key == "session":
            return self.sessions[int(self.row_session_of[index])]
        if key == "open":
            return self.open_[index]
        if key == "close":
            return self.close[index]
        if key == "volume":
            return self.volume[index]
        if key == "trading_value":
            return self.trading_value[index]
        if key == "adtv":
            return self.adtv[index]
        if key == "feature__volatility_20d":
            if not self.has_volatility:
                return default
            return self.volatility[index]
        if key == "available_time":
            return self.available_time[index]
        if key == "limit_locked":
            if self.limit_locked is None:
                return default
            return bool(self.limit_locked[index])
        if key == "action_interval_covered":
            if self.action_interval_covered is None:
                return default
            return self.action_interval_covered[index]
        return default

    @classmethod
    def build(
        cls,
        frame: pl.DataFrame,
        adtv_window: int,
        *,
        instruments: Mapping[str, Instrument] | None = None,
        artifacts: ArtifactSchedule | None = None,
        initial_portfolio: PortfolioSnapshot | None = None,
    ) -> PreparedReplayMarket:
        """Build the immutable market once from a validated replay frame.

        The frame must carry ``REQUIRED_BACKTEST_COLUMNS``; the causal
        ``adtv_window``-session rolling ADTV and per-instrument close-return
        series are computed once and aligned to the canonical
        ``(session, instrument_id)`` row order. Raises
        ``BacktestValidationError`` for missing columns or non-finite execution
        values.
        """
        cls.build_call_count += 1
        missing = [c for c in REQUIRED_BACKTEST_COLUMNS if c not in frame.columns]
        if missing:
            raise BacktestValidationError(f"panel must carry {', '.join(missing)}")
        ordered = frame.sort(["session", "instrument_id"])
        if ordered.is_empty():
            raise BacktestValidationError("panel has no rows")
        with_adtv = ordered.with_columns(
            pl.col("trading_value")
            .rolling_mean(adtv_window, min_samples=1)
            .over("instrument_id")
            .alias("adtv")
        )
        return_series = (
            ordered.sort("session")
            .with_columns(
                (pl.col("close").log().diff().over("instrument_id")).alias("__logret")
            )["__logret"]
            .fill_null(0.0)
        )
        sessions = tuple(
            _as_datetime(s) for s in ordered["session"].unique().sort().to_list()
        )
        session_index_of = {
            session: i for i, session in enumerate(sessions)
        }
        row_sessions = [session_index_of[_as_datetime(s)] for s in ordered["session"].to_list()]
        ranges: dict[int, tuple[int, int]] = {}
        current = -1
        start = 0
        for i, session_idx in enumerate(row_sessions):
            if session_idx != current:
                if current != -1:
                    ranges[current] = (start, i)
                current = session_idx
                start = i
        ranges[current] = (start, len(row_sessions))

        instrument_ids = np.asarray(
            [str(i) for i in ordered["instrument_id"].to_list()], dtype=object
        )
        available_time = np.asarray(
            [
                _as_datetime(v)
                if v is not None
                else None
                for v in ordered.get_column("available_time").to_list()
            ],
            dtype=object,
        )
        session_max_available_time: dict[int, datetime] = {}
        for session_idx, (range_start, range_stop) in ranges.items():
            values = [
                value
                for value in available_time[range_start:range_stop]
                if isinstance(value, datetime) and value.tzinfo is not None
            ]
            if not values:
                raise BacktestValidationError("no available_time at prepared session")
            session_max_available_time[session_idx] = max(values)
        limit_locked = (
            ordered["limit_locked"].to_numpy().astype(bool)
            if "limit_locked" in ordered.columns
            else None
        )
        action_interval_covered = (
            np.asarray(ordered["action_interval_covered"].to_list(), dtype=object)
            if "action_interval_covered" in ordered.columns
            else None
        )
        rows_by_key: dict[tuple[str, datetime], _PreparedRow] = {}
        close = ordered["close"].to_numpy().astype(np.float64)
        open_ = ordered["open"].to_numpy().astype(np.float64)
        volume = ordered["volume"].to_numpy().astype(np.float64)
        trading_value = ordered["trading_value"].to_numpy().astype(np.float64)
        adtv = with_adtv["adtv"].to_numpy().astype(np.float64)
        volatility = (
            ordered["feature__volatility_20d"].to_numpy().astype(np.float64)
            if "feature__volatility_20d" in ordered.columns
            else np.zeros(ordered.height, dtype=np.float64)
        )
        close_returns = return_series.to_numpy().astype(np.float64)
        row_session_array = np.asarray(row_sessions, dtype=np.int64)
        market = cls(
            sessions=sessions,
            session_ranges=ranges,
            session_max_available_time=session_max_available_time,
            rows_by_key=rows_by_key,
            instrument_ids=instrument_ids,
            row_session_of=row_session_array,
            close=close,
            open_=open_,
            volume=volume,
            trading_value=trading_value,
            adtv=adtv,
            volatility=volatility,
            has_volatility="feature__volatility_20d" in ordered.columns,
            available_time=available_time,
            limit_locked=limit_locked,
            action_interval_covered=action_interval_covered,
            close_returns=close_returns,
            instruments=instruments or {},
            artifacts=artifacts,
            initial_portfolio=initial_portfolio,
            cache_bytes=0,
        )
        for i in range(ordered.height):
            rows_by_key[
                (str(ordered["instrument_id"][i]), sessions[int(row_sessions[i])])
            ] = _PreparedRow(market, i)
        # Deduplicated live-state accounting: NumPy buffers, session/index
        # mappings, key/row Python objects, and shared strings. The transient
        # Polars build frame is excluded (released when this method returns).
        object.__setattr__(
            market,
            "cache_bytes",
            _live_state_bytes(
                arrays=(row_session_array, close, open_, volume, trading_value,
                        adtv, volatility, close_returns),
                object_arrays=(instrument_ids, available_time),
                optional_arrays=(limit_locked, action_interval_covered),
                sessions=sessions,
                mappings=(ranges, session_max_available_time),
                rows_by_key=rows_by_key,
                unique_instruments=len(set(instrument_ids.tolist())),
            ),
        )
        return market


def _live_state_bytes(
    *,
    arrays: tuple[np.ndarray, ...],
    object_arrays: tuple[np.ndarray, ...],
    optional_arrays: tuple[np.ndarray | None, ...],
    sessions: tuple[datetime, ...],
    mappings: tuple[Mapping[int, object], ...],
    rows_by_key: Mapping[tuple[str, datetime], _PreparedRow],
    unique_instruments: int,
) -> int:
    """Estimate deduplicated live bytes of one prepared market's state."""
    total = sum(int(value.nbytes) for value in arrays if isinstance(value, np.ndarray))
    for value in object_arrays:
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    for optional_value in optional_arrays:
        if isinstance(optional_value, np.ndarray):
            total += int(optional_value.nbytes)
    total += sys.getsizeof(sessions) + len(sessions) * 48
    for mapping in mappings:
        if isinstance(mapping, dict):
            total += sys.getsizeof(mapping) + len(mapping) * 96
    if isinstance(rows_by_key, dict):
        total += sys.getsizeof(rows_by_key)
        total += len(rows_by_key) * _ROW_KEY_ENTRY_BYTES
    total += max(0, int(unique_instruments)) * 80
    return total
