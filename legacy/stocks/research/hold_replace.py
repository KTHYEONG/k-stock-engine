"""Costed hold-or-replace action labels for the v5 action-value model.

The second-stage ``HoldReplaceValueModel`` learns the incremental value of
retaining an incumbent versus paying to replace it, rather than raw
cross-sectional rank. This module builds the causal, costed action labels from
a training-period portfolio trace produced by an out-of-fold alpha seed: for
each decision state and feasible name the label is the log-return-space
replacement value

``replace_value(i, j, t) = net_hold_to_horizon(i, t)
                           - net_hold_to_horizon(j, t)
                           - entry_cost(i, t) - exit_cost(j, t)
                           - capacity_penalty(i, t)``

where ``j`` is an incumbent or cash and every cost is resolved with the
effective-dated fill-cost model. A retained incumbent has zero switch cost; an
entry carries both a buy and an expected exit cost. Labels are emitted only
after their existing label-availability timestamp and only for actions
feasible under the frozen constraints. Missing or non-finite economic inputs
raise ``ValueError``; unavailable actions are excluded rather than imputed.
All transforms are vectorized Polars/NumPy; no row-wise ``apply`` is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import numpy as np
import polars as pl

from src.core.costs import CostSchedule

ID_COLUMN = "instrument_id"
SESSION_COLUMN = "session"
SESSION_INDEX_COLUMN = "session_index"
LABEL_AVAILABLE_COLUMN = "label_available_time"
ADTV_COLUMN = "adtv"
CLOSE_COLUMN = "close"

CASH_INCUMBENT = "__cash__"


@dataclass(frozen=True, slots=True)
class PortfolioDecisionState:
    """One canonical decision state in a training-period portfolio trace.

    ``incumbents`` and ``incumbent_weights`` are the frozen holdings at the
    decision; both tuples are position-aligned and sorted by instrument id.
    """

    session_index: int
    decision_time: datetime
    incumbents: tuple[str, ...]
    incumbent_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PortfolioStateTrace:
    """Deterministic training-period portfolio states from an OOF alpha seed.

    ``decisions`` must be strictly ascending by ``session_index`` and must not
    repeat a key; the seed never scores its own fitting rows and is not fit
    with validation or holdout data.
    """

    decisions: tuple[PortfolioDecisionState, ...]

    def __post_init__(self) -> None:
        indices = [state.session_index for state in self.decisions]
        if len(set(indices)) != len(indices):
            raise ValueError("PortfolioStateTrace decisions must be unique")
        if indices != sorted(indices):
            raise ValueError("PortfolioStateTrace decisions must be ascending")


def _forward_log_return(
    panel: pl.DataFrame,
    *,
    label_column: str,
    holding_horizon_sessions: int,
) -> pl.DataFrame:
    """Decision-time-available forward log return over the holding horizon.

    The forward return is ``log(close[T + horizon] / close[T])`` in
    chronological per-instrument order; rows whose forward window is incomplete
    carry a null label so they are excluded rather than guessed.
    """
    ordered = panel.sort([ID_COLUMN, SESSION_COLUMN])
    close_now = pl.col(CLOSE_COLUMN)
    close_forward = pl.col(CLOSE_COLUMN).shift(-holding_horizon_sessions)
    forward = (
        pl.when(close_now.is_not_null() & close_forward.is_not_null())
        .then(close_forward.log() - close_now.log())
        .otherwise(None)
        .over(ID_COLUMN)
        .alias("__net_hold_to_horizon")
    )
    return ordered.with_columns(forward)


def _cost_rates(cost_schedule: CostSchedule, decision_time: datetime) -> tuple[float, float]:
    """Resolve the entry (buy) and exit (sell) cost rates at a decision."""
    point = cost_schedule.cost_for(decision_time)
    slippage = point.slippage_bps / 10_000.0
    entry_rate = point.commission_rate + slippage
    exit_rate = point.commission_rate + point.tax_rate + slippage
    return entry_rate, exit_rate


def build_hold_replace_labels(
    panel: pl.DataFrame,
    seed_trace: PortfolioStateTrace,
    *,
    label_column: str,
    label_available_column: str = LABEL_AVAILABLE_COLUMN,
    cost_schedule: CostSchedule,
    holding_horizon_sessions: int,
) -> pl.DataFrame:
    """Emit completed, decision-time-available costed hold/replace actions.

    For every decision in ``seed_trace`` the feasible challenger set is the
    decision-time cross-section of names with a finite close, finite ADTV, and
    a completed forward horizon. Each challenger ``i`` is paired with every
    incumbent ``j`` (plus cash) to emit one row keyed by ``(session_index,
    decision_time, instrument_id, incumbent_id)`` carrying the log-return
    replacement value and the decision-time-available action features. Raises
    ``ValueError`` for duplicate keys, non-monotonic or future-known labels,
    missing identity/time/economic columns, or non-finite inputs.
    """
    if holding_horizon_sessions < 1:
        raise ValueError("holding_horizon_sessions must be positive")
    required = (
        SESSION_INDEX_COLUMN,
        SESSION_COLUMN,
        ID_COLUMN,
        CLOSE_COLUMN,
        ADTV_COLUMN,
        label_available_column,
    )
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"hold/replace panel must carry {', '.join(missing)}")
    if not seed_trace.decisions:
        return pl.DataFrame(
            schema={
                SESSION_INDEX_COLUMN: pl.Int64,
                "decision_time": pl.Datetime("us", "UTC"),
                ID_COLUMN: pl.Utf8,
                "incumbent_id": pl.Utf8,
                "incumbent_weight": pl.Float64,
                "entry_cost": pl.Float64,
                "exit_cost": pl.Float64,
                "capacity_penalty": pl.Float64,
                label_column: pl.Float64,
                label_available_column: pl.Datetime("us", "UTC"),
            }
        )

    with_forward = _forward_log_return(
        panel, label_column=label_column,
        holding_horizon_sessions=holding_horizon_sessions,
    )
    if not np.all(np.isfinite(with_forward[ADTV_COLUMN].to_numpy())):
        raise ValueError("hold/replace panel carries non-finite ADTV")
    if not np.all(np.isfinite(with_forward[CLOSE_COLUMN].to_numpy())):
        raise ValueError("hold/replace panel carries non-finite close")

    output_rows: list[dict[str, object]] = []
    keys: set[tuple[object, object, object, object]] = set()
    for state in seed_trace.decisions:
        decision_time = state.decision_time
        entry_rate, exit_rate = _cost_rates(cost_schedule, decision_time)
        cross = with_forward.filter(
            (pl.col(SESSION_INDEX_COLUMN) == state.session_index)
            & pl.col("__net_hold_to_horizon").is_not_null()
            & pl.col(label_available_column).is_not_null()
        ).select(
            ID_COLUMN,
            "__net_hold_to_horizon",
            ADTV_COLUMN,
            pl.col(label_available_column),
        )
        if cross.is_empty():
            continue
        median_adtv_value = cross[ADTV_COLUMN].median()
        median_adtv = (
            float(cast("float", median_adtv_value))
            if median_adtv_value is not None
            else 0.0
        )
        incumbent_rows = {
            incumbent: weight
            for incumbent, weight in zip(
                state.incumbents, state.incumbent_weights, strict=True
            )
            if isinstance(incumbent, str) and incumbent
        }
        challenger_map = {
            str(row[ID_COLUMN]): row
            for row in cross.iter_rows(named=True)
            if math.isfinite(float(row["__net_hold_to_horizon"]))
            and math.isfinite(float(row[ADTV_COLUMN]))
        }
        incumbents = tuple(sorted(incumbent_rows))
        for challenger, row in challenger_map.items():
            challenger_hold = float(row["__net_hold_to_horizon"])
            challenger_adtv = float(row[ADTV_COLUMN])
            label_available_time = row[label_available_column]
            capacity_penalty = (
                0.0
                if median_adtv <= 0 or challenger_adtv >= median_adtv
                else -math.log(max(challenger_adtv, 1e-12) / median_adtv)
            )
            for incumbent in incumbents:
                key = (
                    state.session_index,
                    _as_utc(decision_time),
                    challenger,
                    incumbent,
                )
                if key in keys:
                    raise ValueError("duplicate hold/replace action key")
                keys.add(key)
                incumbent_hold = (
                    challenger_hold
                    if incumbent == challenger
                    else float(
                        challenger_map[incumbent]["__net_hold_to_horizon"]
                        if incumbent in challenger_map
                        else 0.0
                    )
                )
                entry_cost = 0.0 if incumbent == challenger else entry_rate
                exit_cost = 0.0 if incumbent == challenger else exit_rate
                replace_value = (
                    challenger_hold
                    - incumbent_hold
                    - entry_cost
                    - exit_cost
                    - capacity_penalty
                )
                output_rows.append(
                    {
                        SESSION_INDEX_COLUMN: int(state.session_index),
                        "decision_time": _as_utc(decision_time),
                        ID_COLUMN: challenger,
                        "incumbent_id": incumbent,
                        "incumbent_weight": float(incumbent_rows[incumbent]),
                        "entry_cost": entry_cost,
                        "exit_cost": exit_cost,
                        "capacity_penalty": capacity_penalty,
                        label_column: float(replace_value),
                        label_available_column: _as_utc(label_available_time),
                    }
                )
            cash_key = (state.session_index, _as_utc(decision_time), challenger, CASH_INCUMBENT)
            if cash_key in keys:
                raise ValueError("duplicate hold/replace action key")
            keys.add(cash_key)
            replace_value_cash = (
                challenger_hold - entry_rate - exit_rate - capacity_penalty
            )
            output_rows.append(
                {
                    SESSION_INDEX_COLUMN: int(state.session_index),
                    "decision_time": _as_utc(decision_time),
                    ID_COLUMN: challenger,
                    "incumbent_id": CASH_INCUMBENT,
                    "incumbent_weight": 0.0,
                    "entry_cost": entry_rate,
                    "exit_cost": exit_rate,
                    "capacity_penalty": capacity_penalty,
                    label_column: float(replace_value_cash),
                    label_available_column: _as_utc(label_available_time),
                }
            )

    frame = pl.DataFrame(
        output_rows,
        schema={
            SESSION_INDEX_COLUMN: pl.Int64,
            "decision_time": pl.Datetime("us", "UTC"),
            ID_COLUMN: pl.Utf8,
            "incumbent_id": pl.Utf8,
            "incumbent_weight": pl.Float64,
            "entry_cost": pl.Float64,
            "exit_cost": pl.Float64,
            "capacity_penalty": pl.Float64,
            label_column: pl.Float64,
            label_available_column: pl.Datetime("us", "UTC"),
        },
    )
    if frame.is_empty():
        return frame
    if not np.all(np.isfinite(frame[label_column].to_numpy())):
        raise ValueError("hold/replace labels must be finite")
    return frame.sort([SESSION_INDEX_COLUMN, "decision_time", ID_COLUMN, "incumbent_id"])


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
