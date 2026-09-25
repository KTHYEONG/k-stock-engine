"""Session-indexed pre-open events for the engine timeline."""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from src.backtest.market import MarketArrays
from src.core.pit import PITDataError
from src.data.datasets import dataset_partition_paths

_EXIT_TABLE = "instrument_exits.parquet"


class ExitKind(StrEnum):
    TRADED = "traded_exit"
    HALTED = "halted_exit"


@dataclass(frozen=True, slots=True)
class ExitEvent:
    instrument_idx: int
    last_session_idx: int
    last_close: int
    kind: ExitKind


@dataclass(frozen=True, slots=True)
class DividendEvent:
    """Cash dividend entitlement: holders at the close before ``ex_session_idx`` receive ``dps_krw`` per share on ``pay_session_idx``."""

    instrument_idx: int
    ex_session_idx: int
    pay_session_idx: int
    dps_krw: int


@dataclass(frozen=True, slots=True)
class EngineEvents:
    """Session-indexed pre-open events for the engine timeline.

    Corporate actions come from the panel ``share_factor`` (KRX base-price
    adjustment); cash dividends are a separate input because KRX base prices
    do not adjust for them. ``dividends_integrated`` is False when no dividend
    table was supplied, and results must then be labeled price-return only.
    """

    share_factor_by_session: Mapping[int, tuple[tuple[int, float, int], ...]]
    exits_by_session: Mapping[int, tuple[ExitEvent, ...]]
    dividends_by_ex_session: Mapping[int, tuple[DividendEvent, ...]]
    dividends_integrated: bool


def _as_date(value: Any, *, what: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise PITDataError(f"invalid {what}: {value!r}")


def _share_factor_events(arrays: MarketArrays) -> dict[int, tuple[tuple[int, float, int], ...]]:
    factors = arrays.float_fields["share_factor"]
    base = arrays.int_fields["base_price"]
    hits = np.argwhere(~np.isnan(factors) & (factors != 1.0))
    grouped: dict[int, list[tuple[int, float, int]]] = {}
    for hit in hits.tolist():
        session_idx, instrument_idx = int(hit[0]), int(hit[1])
        factor = float(factors[session_idx, instrument_idx])
        if not np.isfinite(factor) or factor <= 0.0:
            raise PITDataError(f"invalid share factor at session {session_idx}: {factor!r}")
        grouped.setdefault(session_idx, []).append(
            (instrument_idx, factor, int(base[session_idx, instrument_idx]))
        )
    return {key: tuple(sorted(items)) for key, items in sorted(grouped.items())}


def _exit_events(
    panel_dir: Path,
    instrument_index: dict[str, int],
    sessions: list[date],
    session_index: dict[date, int],
) -> dict[int, tuple[ExitEvent, ...]]:
    try:
        verified_paths = dataset_partition_paths(panel_dir)
    except PITDataError as exc:
        raise PITDataError(f"exit table verification failed: {panel_dir}") from exc
    exits_path = next((path for path in verified_paths if path.name == _EXIT_TABLE), None)
    if exits_path is None:
        # Pre-v2 manifests kept the exit companion outside ``partitions``.
        legacy_path = panel_dir / _EXIT_TABLE
        if not legacy_path.is_file():
            return {}
        exits_path = legacy_path
    try:
        frame = pl.read_parquet(exits_path)
    except Exception as exc:
        raise PITDataError(f"exit table is unreadable: {exits_path}") from exc
    required = {"instrument_id", "last_session", "last_close", "exit_kind"}
    if not required <= set(frame.columns):
        raise PITDataError(f"exit table is missing columns: {sorted(required)}")
    grouped: dict[int, list[ExitEvent]] = {}
    for row in frame.to_dicts():
        instrument_id = row["instrument_id"]
        if not isinstance(instrument_id, str) or instrument_id not in instrument_index:
            raise PITDataError(f"exit references unknown instrument: {instrument_id!r}")
        last_session = _as_date(row["last_session"], what="exit last_session")
        if last_session not in session_index:
            raise PITDataError(f"exit references unknown session: {last_session!r}")
        last_idx = session_index[last_session]
        if last_idx >= len(sessions) - 1:
            continue
        try:
            kind = ExitKind(str(row["exit_kind"]))
        except ValueError as exc:
            raise PITDataError(f"invalid exit kind: {row['exit_kind']!r}") from exc
        try:
            last_close = int(cast("Any", row["last_close"]))
        except (TypeError, ValueError) as exc:
            raise PITDataError(f"invalid exit last_close: {row['last_close']!r}") from exc
        key = last_idx + 1
        grouped.setdefault(key, []).append(
            ExitEvent(
                instrument_idx=instrument_index[instrument_id],
                last_session_idx=last_idx,
                last_close=last_close,
                kind=kind,
            )
        )
    return {
        key: tuple(sorted(items, key=lambda event: event.instrument_idx))
        for key, items in sorted(grouped.items())
    }


def _dividend_events(
    frame: pl.DataFrame,
    instrument_index: dict[str, int],
    sessions: list[date],
    session_index: dict[date, int],
) -> dict[int, tuple[DividendEvent, ...]]:
    required = {"instrument_id", "ex_session", "pay_session", "dps_krw"}
    if not required <= set(frame.columns):
        raise PITDataError(f"dividend table is missing columns: {sorted(required)}")
    grouped: dict[int, list[DividendEvent]] = {}
    for row in frame.to_dicts():
        instrument_id = row["instrument_id"]
        if not isinstance(instrument_id, str) or instrument_id not in instrument_index:
            raise PITDataError(f"dividend references unknown instrument: {instrument_id!r}")
        ex_session = _as_date(row["ex_session"], what="dividend ex_session")
        if ex_session not in session_index:
            raise PITDataError(f"dividend references unknown session: {ex_session!r}")
        pay_date = _as_date(row["pay_session"], what="dividend pay_session")
        if pay_date < ex_session:
            raise PITDataError("dividend pays before its ex-date")
        if pay_date in session_index:
            pay_idx = session_index[pay_date]
        else:
            pay_idx = bisect.bisect_right(sessions, pay_date)
            if pay_idx >= len(sessions):
                raise PITDataError(f"dividend references unknown session: {pay_date!r}")
        dps = row["dps_krw"]
        if isinstance(dps, bool) or not isinstance(dps, (int, np.integer)) or int(dps) <= 0:
            raise PITDataError(f"invalid dps_krw: {dps!r}")
        ex_idx = session_index[ex_session]
        grouped.setdefault(ex_idx, []).append(
            DividendEvent(
                instrument_idx=instrument_index[instrument_id],
                ex_session_idx=ex_idx,
                pay_session_idx=pay_idx,
                dps_krw=int(dps),
            )
        )
    return {
        key: tuple(sorted(items, key=lambda event: event.instrument_idx))
        for key, items in sorted(grouped.items())
    }


def build_engine_events(
    *, arrays: MarketArrays, panel_dir: Path, dividends: pl.DataFrame | None
) -> EngineEvents:
    """Index corporate actions, exits, and dividends by the session they take effect.

    Raises:
        PITDataError: an exit or dividend references an unknown instrument or
            session, a dividend pays before its ex-date, ``dps_krw`` is not a
            positive integer, or a share factor is non-finite or non-positive.
    """
    sessions = list(arrays.sessions)
    session_index = {session: idx for idx, session in enumerate(sessions)}
    instrument_index = {iid: idx for idx, iid in enumerate(arrays.instrument_ids)}
    share_events = _share_factor_events(arrays)
    exit_events = _exit_events(Path(panel_dir), instrument_index, sessions, session_index)
    if dividends is None:
        return EngineEvents(
            share_factor_by_session=share_events,
            exits_by_session=exit_events,
            dividends_by_ex_session={},
            dividends_integrated=False,
        )
    dividend_events = _dividend_events(dividends, instrument_index, sessions, session_index)
    return EngineEvents(
        share_factor_by_session=share_events,
        exits_by_session=exit_events,
        dividends_by_ex_session=dividend_events,
        dividends_integrated=True,
    )
