# mypy: ignore-errors
"""Session indexing, locked holdout, temporal data windows."""
from __future__ import annotations

import polars as pl

_SESSION_IDX = "session_index"

def _index_sessions(frame: pl.DataFrame) -> pl.DataFrame:
    if _SESSION_IDX not in frame.columns:
        frame = frame.with_columns(pl.col("session").rank("dense").cast(pl.Int64).alias(_SESSION_IDX))
    return frame.with_columns(pl.col(_SESSION_IDX).rank("dense").cast(pl.Int64).alias(_SESSION_IDX))

def _locked_holdout(panel: pl.DataFrame, request) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    holdout_sessions = request.forward_holdout_sessions
    if holdout_sessions <= 0:
        holdout_sessions = max(1, panel["session"].n_unique() // 5)
    sessions = sorted(panel["session"].unique().to_list())
    if len(sessions) <= holdout_sessions:
        return pl.DataFrame(), pl.DataFrame(), "insufficient-holdout-history"
    holdout_set = set(sessions[-holdout_sessions:])
    pre_holdout = panel.filter(~pl.col("session").is_in(list(holdout_set)))
    holdout = panel.filter(pl.col("session").is_in(list(holdout_set)))
    if pre_holdout.is_empty() or holdout.is_empty():
        return pl.DataFrame(), pl.DataFrame(), "insufficient-holdout-history"
    return pre_holdout, holdout, ""
