# mypy: ignore-errors
"""Causal screen-cache preparation, labels, feature screening, candidate evidence."""
# ruff: noqa: SIM108
from __future__ import annotations

import numpy as np
import polars as pl

from src.stocks.ml.labels import SESSION_COLUMN

_ID_COLUMN = "instrument_id"
_SESSION_IDX = "session_index"

def sample_labeled_screen_rows(frame: pl.DataFrame, max_rows: int, *, minimum_names_per_session: int = 2) -> np.ndarray:
    if max_rows <= 0 or frame.is_empty():
        return np.array([], dtype=np.int64)
    if not isinstance(minimum_names_per_session, int) or minimum_names_per_session < 1:
        raise ValueError("minimum_names_per_session must be positive int")
    session_col = SESSION_COLUMN if SESSION_COLUMN in frame.columns else (_SESSION_IDX if _SESSION_IDX in frame.columns else None)
    has_adtv = "adtv_20d" in frame.columns
    has_instrument = _ID_COLUMN in frame.columns
    indexed = frame.with_row_index("__row_idx_tmp_sample")
    if session_col is None:
        sort_by: list[str] = []
        descending: list[bool] = []
        if has_adtv:
            sort_by.append("adtv_20d")
            descending.append(True)
        if has_instrument:
            sort_by.append(_ID_COLUMN)
            descending.append(False)
        if sort_by:
            sorted_frame = indexed.sort(by=sort_by, descending=descending)
        else:
            sorted_frame = indexed
        take = min(int(max_rows), int(sorted_frame.height))
        return sorted_frame.head(take)["__row_idx_tmp_sample"].to_numpy().astype(np.int64, copy=False)
    try:
        sessions_sorted = sorted(indexed[session_col].unique().to_list())
    except Exception:
        sessions_sorted = indexed[session_col].unique().sort().to_list()
    per_session: dict[object, list[int]] = {}
    for s in sessions_sorted:
        sub = indexed.filter(pl.col(session_col) == s)
        sort_by2: list[str] = []
        descending2: list[bool] = []
        if has_adtv:
            sort_by2.append("adtv_20d")
            descending2.append(True)
        if has_instrument:
            sort_by2.append(_ID_COLUMN)
            descending2.append(False)
        if sort_by2:
            sub_sorted = sub.sort(by=sort_by2, descending=descending2)
        else:
            sub_sorted = sub
        per_session[s] = sub_sorted["__row_idx_tmp_sample"].to_numpy().astype(np.int64, copy=False).tolist()
    result: list[int] = []
    max_per_session = max((len(v) for v in per_session.values()), default=0)
    for round_idx in range(max_per_session):
        for s in sessions_sorted:
            lst = per_session[s]
            if round_idx < len(lst) and len(result) < int(max_rows):
                result.append(int(lst[round_idx]))
            if len(result) >= int(max_rows):
                break
        if len(result) >= int(max_rows):
            break
    return np.array(result, dtype=np.int64)

def prepare_screening_fold_cache(
    frame: pl.DataFrame,
    *,
    max_rows: int,
    minimum_names_per_session: int = 2,
) -> dict[str, object]:
    """Prepare the deterministic row index used by every screening fold.

    The cache deliberately contains row indices rather than copied feature data;
    callers can materialize the selected rows after applying their causal cut.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars DataFrame")
    if not isinstance(max_rows, int) or max_rows < 0:
        raise ValueError("max_rows must be a non-negative int")
    if not isinstance(minimum_names_per_session, int) or minimum_names_per_session < 1:
        raise ValueError("minimum_names_per_session must be positive int")
    session_col = SESSION_COLUMN if SESSION_COLUMN in frame.columns else _SESSION_IDX
    if session_col not in frame.columns:
        raise ValueError("screening frame must carry session or session_index")
    counts = frame.group_by(session_col).len().sort(session_col)
    eligible = counts.filter(pl.col("len") >= minimum_names_per_session)
    eligible_rows = frame.filter(pl.col(session_col).is_in(eligible[session_col]))
    row_indices = sample_labeled_screen_rows(
        eligible_rows, max_rows, minimum_names_per_session=minimum_names_per_session
    )
    return {
        "row_indices": row_indices,
        "session_column": session_col,
        "eligible_sessions": tuple(eligible[session_col].to_list()),
        "dropped_sessions": int(counts.height - eligible.height),
    }
