"""Entry/exit-semantic stock labels with declared horizon and cost treatment."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import polars as pl

ID_COLUMN = "instrument_id"
SESSION_COLUMN = "session"

RESIDUAL_O2O_LABEL = "residual_o2o_5d"
RELEVANCE_COLUMN = "relevance"
LABEL_AVAILABLE_COLUMN = "label_available_time"
LAMBDARANK_GAIN = (0, 1, 3, 7, 15)
MIN_LAMBDARANK_GROUP = 20
_MARKET_LOG_RETURN_COLUMN = "market_log_return"


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """Declares entry/exit fields, holding horizon, and cost assumption.

    The canonical phase-1 label is ``log(close[T+horizon] / open[T+1])``: a
    decision after the close of session ``T`` enters at the next eligible open
    and liquidates at the close of ``T + horizon``. Labels are built after the
    point-in-time universe and never recomputed inside a trainer.
    """

    name: str
    entry_field: str
    exit_field: str
    horizon_sessions: int
    corporate_action_treatment: str = "none"
    cost_assumption: str = "no-cost"

    def __post_init__(self) -> None:
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be positive")

    @property
    def fingerprint(self) -> str:
        return sha256(
            f"{self.name}:{self.entry_field}:{self.exit_field}:"
            f"{self.horizon_sessions}:{self.corporate_action_treatment}:{self.cost_assumption}".encode()
        ).hexdigest()

    def apply(
        self,
        frame: pl.DataFrame,
        id_column: str = ID_COLUMN,
        session_column: str = SESSION_COLUMN,
    ) -> pl.DataFrame:
        """Sort by instrument then session and compute next-open forward-close labels.

        Labels are emitted only when both the next-session entry price and the
        forward exit price exist; terminal rows carry a null label so they can be
        excluded from training rather than guessed.
        """
        required = (id_column, session_column, self.entry_field, self.exit_field)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"label {self.name} requires {', '.join(missing)}")
        ordered = frame.sort([id_column, session_column])
        entry = pl.col(self.entry_field).shift(-1)
        exit_price = pl.col(self.exit_field).shift(-self.horizon_sessions)
        label = (
            pl.when(entry.is_not_null() & exit_price.is_not_null())
            .then(exit_price.log() - entry.log())
            .otherwise(None)
            .over(id_column)
            .alias(self.name)
        )
        return ordered.with_columns(label)


def residual_open_to_open_label(
    frame: pl.DataFrame,
    horizon_sessions: int = 5,
    id_column: str = ID_COLUMN,
    session_column: str = SESSION_COLUMN,
) -> pl.DataFrame:
    """Compute the residual open-to-open label and LambdaRank relevance.

    The label is ``log(open[T+1+horizon] / open[T+1])`` minus the matched
    market open-to-open return over the same window, mirroring a decision after
    ``T`` close, entry at ``T+1`` open, and exit at ``T+horizon+1`` open. When
    the frame carries ``market_log_return`` it is used as the matched market;
    otherwise the per-session equal-weight mean of instrument open-to-open
    returns is the matched-market proxy.

    Residuals are winsorized at 1%/99% within each session and mapped to
    LambdaRank relevance ``0..4``. Sessions with fewer than 20 eligible names
    are excluded, and every emitted row carries a ``label_available_time`` at or
    after the terminal horizon open.
    """
    if horizon_sessions <= 0:
        raise ValueError("horizon_sessions must be positive")
    required = (id_column, session_column, "open")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"residual label requires {', '.join(missing)}")

    ordered = frame.sort([id_column, session_column])
    entry = pl.col("open").shift(-1).over(id_column)
    exit_price = pl.col("open").shift(-(horizon_sessions + 1)).over(id_column)
    o2o_expr = (
        pl.when(entry.is_not_null() & exit_price.is_not_null())
        .then(exit_price.log() - entry.log())
        .otherwise(None)
        .alias("__o2o")
    )
    exit_session_expr = (
        pl.col(session_column).shift(-(horizon_sessions + 1)).over(id_column)
    )
    base = ordered.with_columns(o2o_expr, exit_session_expr.alias("__exit_session"))

    if _MARKET_LOG_RETURN_COLUMN in base.columns:
        market_daily = (
            base.select(session_column, pl.col(_MARKET_LOG_RETURN_COLUMN))
            .sort(session_column)
            .group_by(session_column)
            .agg(pl.col(_MARKET_LOG_RETURN_COLUMN).mean().alias("__mkt"))
            .with_columns(
                pl.col("__mkt")
                .rolling_sum(window_size=horizon_sessions)
                .shift(-1)
                .alias("__matched")
            )
        )
        base = base.join(market_daily, on=session_column, how="left")
    else:
        market_daily = (
            base.select(session_column, pl.col("__o2o"))
            .sort(session_column)
            .group_by(session_column)
            .agg(pl.col("__o2o").mean().alias("__matched"))
        )
        base = base.join(market_daily, on=session_column, how="left")

    with_residual = base.with_columns(
        (pl.col("__o2o") - pl.col("__matched")).alias(RESIDUAL_O2O_LABEL)
    )

    non_finite = with_residual.filter(
        pl.col(RESIDUAL_O2O_LABEL).is_not_null()
        & ~pl.col(RESIDUAL_O2O_LABEL).is_finite()
    )
    if not non_finite.is_empty():
        raise ValueError("non-finite residual label values")

    winsor = (
        with_residual.filter(pl.col(RESIDUAL_O2O_LABEL).is_not_null())
        .group_by(session_column)
        .agg(
            pl.col(RESIDUAL_O2O_LABEL).quantile(0.01).alias("__lo"),
            pl.col(RESIDUAL_O2O_LABEL).quantile(0.99).alias("__hi"),
        )
    )
    clipped = (
        with_residual.join(winsor, on=session_column, how="left")
        .with_columns(
            pl.col(RESIDUAL_O2O_LABEL).clip(pl.col("__lo"), pl.col("__hi")).alias("__clipped")
        )
    )
    count = pl.col(RESIDUAL_O2O_LABEL).count().over(session_column)
    pct_rank = (
        (pl.col("__clipped").rank("average").over(session_column) - 1.0) / (count - 1.0)
    ).fill_null(0.5)
    relevance = (
        pl.when(pct_rank.is_not_null())
        .then((pct_rank * 5.0).floor().cast(pl.Int8).clip(0, 4))
        .otherwise(None)
    )

    available = (
        pl.col("__exit_session")
        .cast(pl.Datetime("us", "UTC"))
        .alias(LABEL_AVAILABLE_COLUMN)
    )
    out = clipped.with_columns(
        relevance.alias(RELEVANCE_COLUMN),
        available,
    ).select(
        id_column,
        session_column,
        RESIDUAL_O2O_LABEL,
        RELEVANCE_COLUMN,
        LABEL_AVAILABLE_COLUMN,
    )

    eligible_sessions = (
        out.filter(pl.col(RESIDUAL_O2O_LABEL).is_not_null())
        .group_by(session_column)
        .len()
        .filter(pl.col("len") >= MIN_LAMBDARANK_GROUP)[session_column]
    )
    if eligible_sessions.is_empty():
        return out.with_columns(pl.lit(None, dtype=pl.Float64).alias(RESIDUAL_O2O_LABEL))
    return (
        out.filter(
            pl.col(session_column).is_in(eligible_sessions)
            | pl.col(RESIDUAL_O2O_LABEL).is_null()
        ).sort([id_column, session_column])
    )
