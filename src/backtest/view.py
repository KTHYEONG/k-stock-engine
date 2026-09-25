"""Decision-time point-in-time view over dense market arrays and as-of tables."""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from datetime import date, datetime, time
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.backtest.market import MarketArrays
from src.core.pit import PITDataError
from src.core.time import KRX_TZ

_AVAIL_AT = time(18, 0)


class AsOfTable:
    """A frame sorted by ``available_at`` supporting O(log n) as-of cuts."""

    __slots__ = ("_frame", "_keys")

    def __init__(self, frame: pl.DataFrame) -> None:
        if "available_at" not in frame.columns:
            raise PITDataError("as-of table requires an available_at column")
        dtype = frame.schema["available_at"]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
            raise PITDataError("as-of table requires a timezone-aware available_at column")
        self._frame = frame.sort("available_at")
        self._keys: list[datetime] = self._frame["available_at"].to_list()

    def upto(self, decision_time: datetime) -> pl.DataFrame:
        if decision_time.tzinfo is None or decision_time.utcoffset() is None:
            raise PITDataError("decision_time must be timezone-aware")
        stop = bisect.bisect_right(self._keys, decision_time)
        return self._frame.slice(0, stop)


class PITView:
    """Decision-time view: only data observable at ``decision_time`` is reachable.

    Market arrays are exposed as zero-copy slices ``[: t + 1]`` (panel rows are
    available 18:00 KST on their session, before the decision). As-of tables
    (financial facts, investor flow, industry) are pre-sorted by
    ``available_at`` and cut with a binary search, so lookahead is structurally
    impossible rather than a convention strategies must follow.
    """

    __slots__ = ("_arrays", "_asof_tables", "_decision_time", "_t")

    def __init__(
        self,
        *,
        arrays: MarketArrays,
        t: int,
        decision_time: datetime,
        asof_tables: Mapping[str, AsOfTable],
    ) -> None:
        if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < len(arrays.sessions):
            raise PITDataError(f"session index out of range: {t!r}")
        if decision_time.tzinfo is None or decision_time.utcoffset() is None:
            raise PITDataError("decision_time must be timezone-aware")
        available_at = datetime.combine(arrays.sessions[t], _AVAIL_AT, tzinfo=KRX_TZ)
        if decision_time < available_at:
            raise PITDataError("decision_time precedes the session availability instant")
        self._arrays = arrays
        self._t = t
        self._decision_time = decision_time
        self._asof_tables = dict(asof_tables)

    @property
    def t(self) -> int:
        return self._t

    @property
    def session_date(self) -> date:
        """Session date at ``t`` (calendar metadata, not market data)."""
        return self._arrays.sessions[self._t]

    def field(self, name: str) -> NDArray[Any]:
        if name in self._arrays.int_fields:
            source = self._arrays.int_fields[name]
        elif name in self._arrays.float_fields:
            source = self._arrays.float_fields[name]
        elif name in self._arrays.bool_fields:
            source = self._arrays.bool_fields[name]
        elif name == "market":
            source = self._arrays.market
        else:
            raise KeyError(name)
        out = source[: self._t + 1]
        out.flags.writeable = False
        return out

    def table(self, name: str) -> pl.DataFrame:
        return self._asof_tables[name].upto(self._decision_time)
