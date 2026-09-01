# mypy: ignore-errors
"""PreparedAllocationMarket construction, causal arrays and reference-market validation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

_SESSION_COLUMN = "session"

@dataclass(frozen=True, slots=True)
class PreparedAllocationMarket:
    sessions: tuple[datetime, ...]
    session_ranges: Mapping[int, tuple[int, int]]
    instrument_ids: np.ndarray
    row_session_of: np.ndarray
    row_sessions: np.ndarray
    close: np.ndarray
    adtv: np.ndarray
    sector: np.ndarray
    returns: np.ndarray
    volatility_lookback_sessions: int
    vol_series: np.ndarray
    dense: bool
    n_instruments: int
    sorted_instruments: np.ndarray
    instrument_position_of: np.ndarray
    instrument_position_lookup: Mapping[str, int]
    returns_matrix: np.ndarray
    rows_by_key: Mapping[tuple[str, datetime], int]
    cache_bytes: int
    expected_active_alpha: np.ndarray
    expected_net_alpha: np.ndarray
    alpha_lower_bound: np.ndarray
    net_alpha_lower_bound: np.ndarray
    exit_cost_rate: np.ndarray
    @property
    def row_count(self) -> int:
        return int(self.instrument_ids.size)
    @classmethod
    def build(cls, frame: pl.DataFrame, volatility_lookback_sessions: int = 20) -> PreparedAllocationMarket:
        if volatility_lookback_sessions < 1:
            raise ValueError("volatility lookback sessions must be positive")
        missing = [c for c in ("instrument_id", _SESSION_COLUMN, "sector", "adtv", "close") if c not in frame.columns]
        if missing:
            raise ValueError(f"prepared market frame must carry {', '.join(missing)}")
        ordered = frame.sort([_SESSION_COLUMN, "instrument_id"])
        if ordered.is_empty():
            raise ValueError("prepared market frame has no rows")
        sessions = tuple(datetime.fromisoformat(str(s)) if not isinstance(s, datetime) else s for s in ordered[_SESSION_COLUMN].unique().sort().to_list())
        session_index_of = {session: i for i, session in enumerate(sessions)}
        row_sessions_list = [session_index_of[datetime.fromisoformat(str(s)) if not isinstance(s, datetime) else s] for s in ordered[_SESSION_COLUMN].to_list()]
        ranges: dict[int, tuple[int, int]] = {}
        current = -1
        start = 0
        for i, session_idx in enumerate(row_sessions_list):
            if session_idx != current:
                if current != -1:
                    ranges[current] = (start, i)
                current = session_idx
                start = i
        ranges[current] = (start, len(row_sessions_list))
        row_sessions = np.asarray([sessions[i] for i in row_sessions_list], dtype=object)
        instrument_ids = np.asarray([str(i) for i in ordered["instrument_id"].to_list()], dtype=object)
        unique_ids = [str(i) for i in ordered["instrument_id"].unique().sort().to_list()]
        n_instruments = len(unique_ids)
        sorted_instruments = np.asarray(unique_ids, dtype=object)
        position_map = {instrument: position for position, instrument in enumerate(unique_ids)}
        instrument_position_of = np.asarray([position_map[str(i)] for i in ordered["instrument_id"].to_list()], dtype=np.int64)
        logret = pl.col("close").log() - pl.col("close").log().shift(1).over("instrument_id")
        with_ret = ordered.with_columns(logret.alias("__ret"), logret.rolling_std(window_size=volatility_lookback_sessions, min_samples=2).over("instrument_id").alias("__vol"))
        returns = with_ret["__ret"].to_numpy().astype(np.float64)
        vol_series = with_ret["__vol"].to_numpy().astype(np.float64)
        n_sessions = len(sessions)
        dense = ordered.height == n_sessions * n_instruments
        if dense:
            first_session_ids = [str(i) for i in ordered["instrument_id"][ranges[0][0]:ranges[0][1]].to_list()]
            dense = first_session_ids == unique_ids
        if dense:
            returns_matrix = returns.reshape(n_sessions, n_instruments)
        else:
            returns_matrix = np.full((n_sessions, n_instruments), np.nan, dtype=np.float64)
            for session_index in range(n_sessions):
                lo, hi = ranges[session_index]
                returns_matrix[session_index, instrument_position_of[lo:hi]] = returns[lo:hi]
        rows_by_key: dict[tuple[str, datetime], int] = {}
        def economic_column(name: str) -> np.ndarray:
            if name not in ordered.columns:
                return np.full(ordered.height, np.nan, dtype=np.float64)
            return ordered[name].to_numpy().astype(np.float64)
        market = cls(sessions=sessions, session_ranges=ranges, instrument_ids=instrument_ids, row_session_of=np.asarray(row_sessions_list, dtype=np.int64), row_sessions=row_sessions, close=ordered["close"].to_numpy().astype(np.float64), adtv=ordered["adtv"].to_numpy().astype(np.float64), sector=np.asarray(ordered["sector"].to_list(), dtype=object), returns=returns, volatility_lookback_sessions=volatility_lookback_sessions, vol_series=vol_series, dense=dense, n_instruments=n_instruments, sorted_instruments=sorted_instruments, instrument_position_of=instrument_position_of, instrument_position_lookup=position_map, returns_matrix=returns_matrix, rows_by_key=rows_by_key, cache_bytes=int(ordered.estimated_size()), expected_active_alpha=economic_column("expected_active_alpha"), expected_net_alpha=economic_column("expected_net_alpha"), alpha_lower_bound=economic_column("alpha_lower_bound"), net_alpha_lower_bound=economic_column("net_alpha_lower_bound"), exit_cost_rate=economic_column("exit_cost_rate"))
        for i in range(ordered.height):
            rows_by_key[(str(ordered["instrument_id"][i]), sessions[int(row_sessions_list[i])])] = i
        return market
