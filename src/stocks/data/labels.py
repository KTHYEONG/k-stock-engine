"""Calendar-aware forward-label builder and immutable label publication.

Labels are never predictors and never reside in a feature panel. A label
dataset is built per ``(base_panel_hash, calendar_hash, label_definition,
horizon, price convention)`` and keyed by ``(instrument_id, session)``.

Timing is strictly KRX-session based: the decision session's entry is the next
KRX session open and its exit is the close of ``horizon`` sessions later. Rows
whose future horizon is incomplete (missing calendar session or missing price)
are absent, never guessed, and every emitted label carries a terminal
``label_available_time`` at or after the terminal horizon session (spec
acceptance criterion 3).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.core.costs import CostSchedule, LiquiditySlippageModel
from src.core.datasets import (
    HIVE_PARTITION_LAYOUT,
    DatasetCertification,
    DatasetManifest,
    make_manifest,
)
from src.core.instruments import AssetKind
from src.stocks.data.outcome_evidence import merge_open_bar_evidence, resolve_policy_outcome
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.domain.execution_policy import (
    SCHEDULED_OPEN_V1,
    ExecutionOutcomePolicy,
)
from src.stocks.research.labels import RELEVANCE_COLUMN, LabelDefinition
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

logger = logging.getLogger("stocks.data.labels")

ID_COLUMN = "instrument_id"
SESSION_COLUMN = "session"
LABEL_AVAILABLE_COLUMN = "label_available_time"
LABEL_FEATURE_SET = "labels"
_KRX_AVAILABLE_TIME = time(15, 31)
_KRX_TZ = ZoneInfo("Asia/Seoul")

RESIDUAL_O2O_PREFIX = "residual_o2o_"
MIN_RESIDUAL_GROUP = 20
RESIDUAL_ALGORITHM_VERSION = "calendar-residual-o2o-v1"
SUPPORTED_RESIDUAL_HORIZONS = (5, 10, 15)
MULTI_HORIZON_RESIDUAL_DEFINITION = "residual_o2o_multi_5_10_15d"

COST_AWARE_O2O_PREFIX = "net_residual_o2o_"
COST_AWARE_GROSS_PREFIX = "gross_o2o_"
COST_AWARE_RISK_FITTED_PREFIX = "risk_fitted_"
COST_AWARE_RISK_RESIDUAL_PREFIX = "risk_residual_"
COST_AWARE_REFERENCE_COST_PREFIX = "reference_cost_"
COST_AWARE_RESIDUAL_ALGORITHM_VERSION = "calendar-cost-aware-residual-o2o-v1"
MULTI_HORIZON_COST_AWARE_DEFINITION = "net_residual_o2o_multi_5_10_15d"
_COST_AWARE_CONTROL_COLUMNS = ("market_cap", "beta", "volatility")
_MIN_COST_AWARE_ROWS_PER_SESSION = 30
_REFERENCE_PARTICIPATION = 0.01

NET_ALPHA_TARGET_PREFIX = "net_alpha_"
NET_ALPHA_CONTINUOUS_SUFFIX = "_target"
NET_ALPHA_DEFINITION = "net_alpha_o2o"
NET_ALPHA_ALGORITHM_VERSION = "calendar-net-alpha-o2o-v1"
_MIN_NET_ALPHA_ROWS_PER_SESSION = 30


def build_label_dataset(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    definition: LabelDefinition,
) -> pl.DataFrame:
    """Build calendar-aware forward labels keyed by ``(instrument_id, session)``.

    Only rows with a complete ``horizon``-session future horizon are emitted;
    each such row has a terminal ``label_available_time``. The base panel must
    carry ``instrument_id``, ``session``, and the definition's entry/exit
    fields (``open`` / ``close`` by convention).

    Raises:
        ValueError: if the base panel lacks identity or price columns, or if a
            base-panel session is not a KRX calendar session.
    """
    missing = [c for c in (ID_COLUMN, SESSION_COLUMN) if c not in base_panel.columns]
    if missing:
        raise ValueError(f"base panel missing {', '.join(missing)}")
    price_missing = [c for c in (definition.entry_field, definition.exit_field) if c not in base_panel.columns]
    if price_missing:
        raise ValueError(f"label {definition.name} requires price columns {price_missing}")

    sessions = list(calendar.sessions)
    by_date = {session: index for index, session in enumerate(sessions)}
    if len(by_date) != len(sessions):
        raise ValueError("calendar contains duplicate sessions")

    panel = base_panel.with_columns(pl.col(SESSION_COLUMN).cast(pl.Date).alias("_session_date"))
    calendar_frame = pl.DataFrame(
        {
            "_session_date": sessions,
            "_cal_pos": list(range(len(sessions))),
            "_entry_date": [sessions[p + 1] if p + 1 < len(sessions) else None for p in range(len(sessions))],
            "_exit_date": [
                sessions[p + definition.horizon_sessions]
                if p + definition.horizon_sessions < len(sessions)
                else None
                for p in range(len(sessions))
            ],
        }
    )
    panel = panel.join(calendar_frame, on="_session_date", how="left")
    unknown = panel.filter(
        pl.col("_cal_pos").is_null() | pl.col("_session_date").is_null()
    )
    if not unknown.is_empty():
        raise ValueError("base panel contains non-calendar sessions")

    prices = panel.select(
        ID_COLUMN,
        pl.col("_session_date").alias("_price_date"),
        pl.col(definition.entry_field),
        pl.col(definition.exit_field),
    )
    entries = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_entry_date"),
        pl.col(definition.entry_field).alias("_entry_price"),
    )
    exits = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_exit_date"),
        pl.col(definition.exit_field).alias("_exit_price"),
    )
    panel = (
        panel.join(entries, on=[ID_COLUMN, "_entry_date"], how="left")
        .join(exits, on=[ID_COLUMN, "_exit_date"], how="left")
        .filter(pl.col("_entry_price").is_not_null() & pl.col("_exit_price").is_not_null())
    )

    label = pl.col("_exit_price").log() - pl.col("_entry_price").log()
    label_available = (
        pl.col("_exit_date")
        .dt.combine(pl.lit(_KRX_AVAILABLE_TIME))
        .dt.replace_time_zone("Asia/Seoul")
        .dt.convert_time_zone("UTC")
    )
    out = panel.with_columns(
        pl.col("_session_date").alias(SESSION_COLUMN),
        label.alias(definition.name),
        label_available.alias(LABEL_AVAILABLE_COLUMN),
    ).select(
        ID_COLUMN,
        SESSION_COLUMN,
        definition.name,
        LABEL_AVAILABLE_COLUMN,
    ).sort([ID_COLUMN, SESSION_COLUMN])

    incomplete = out.filter(pl.col(LABEL_AVAILABLE_COLUMN).is_null() | pl.col(definition.name).is_null())
    if not incomplete.is_empty():
        raise ValueError("label builder emitted an incomplete horizon row")
    logger.info(
        "built label %s: %s rows over %s sessions",
        definition.name,
        out.height,
        len(sessions),
    )
    return out


def build_residual_o2o_label_dataset(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    *,
    horizon_sessions: int = 5,
) -> pl.DataFrame:
    """Build calendar-correct residual open-to-open labels with LambdaRank relevance.

    For a decision session at calendar position ``p``, a label is computed only
    when the same instrument has a valid open at both exact calendar sessions:

    ``entry = calendar[p + 1]`` and ``exit = calendar[p + 1 + horizon]``.
    The gross return is ``log(open(exit) / open(entry))`` and the residual is
    the gross minus the equal-weight mean gross within the decision session.
    The per-session 1st/99th percentile-clipped residual rank maps to
    ``relevance = floor(percentile * 5)`` clipped to integer ``0..4``.

    Sessions with fewer than ``MIN_RESIDUAL_GROUP`` finite residuals are dropped
    completely and no fabricated label is ever retained. The result schema is:

    ``instrument_id, session, residual_o2o_<horizon>d, relevance, label_available_time``

    ``label_available_time`` is set to the exact exit-session open timestamp in
    UTC (never the decision session), so a decision row is only usable after the
    terminal horizon open is observable.
    """
    if horizon_sessions <= 0:
        raise ValueError("horizon_sessions must be positive")
    missing = [c for c in (ID_COLUMN, SESSION_COLUMN, "open") if c not in base_panel.columns]
    if missing:
        raise ValueError(f"residual label requires base panel columns {missing}")

    label_column = f"{RESIDUAL_O2O_PREFIX}{horizon_sessions}d"
    sessions = list(calendar.sessions)
    by_date = {session: index for index, session in enumerate(sessions)}
    if len(by_date) != len(sessions):
        raise ValueError("calendar contains duplicate sessions")

    panel = base_panel.with_columns(pl.col(SESSION_COLUMN).cast(pl.Date).alias("_session_date"))
    calendar_frame = pl.DataFrame(
        {
            "_session_date": sessions,
            "_cal_pos": list(range(len(sessions))),
            "_entry_date": [
                sessions[p + 1] if p + 1 < len(sessions) else None for p in range(len(sessions))
            ],
            "_exit_date": [
                sessions[p + 1 + horizon_sessions]
                if p + 1 + horizon_sessions < len(sessions)
                else None
                for p in range(len(sessions))
            ],
        }
    )
    panel = panel.join(calendar_frame, on="_session_date", how="left")
    unknown = panel.filter(pl.col("_cal_pos").is_null() | pl.col("_session_date").is_null())
    if not unknown.is_empty():
        raise ValueError("base panel contains non-calendar sessions")

    prices = panel.select(
        ID_COLUMN,
        pl.col("_session_date").alias("_price_date"),
        pl.col("open"),
    )
    entries = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_entry_date"),
        pl.col("open").alias("_entry_open"),
    )
    exits = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_exit_date"),
        pl.col("open").alias("_exit_open"),
    )
    panel = (
        panel.join(entries, on=[ID_COLUMN, "_entry_date"], how="left")
        .join(exits, on=[ID_COLUMN, "_exit_date"], how="left")
        .filter(pl.col("_entry_open").is_not_null() & pl.col("_exit_open").is_not_null())
    )

    gross = pl.col("_exit_open").log() - pl.col("_entry_open").log()
    residual = gross - gross.mean().over(SESSION_COLUMN)
    label_available = (
        pl.col("_exit_date")
        .dt.combine(pl.lit(_KRX_AVAILABLE_TIME))
        .dt.replace_time_zone("Asia/Seoul")
        .dt.convert_time_zone("UTC")
    )
    session_utc = (
        pl.col("_session_date")
        .cast(pl.Datetime)
        .dt.replace_time_zone("UTC")
    )

    winsor = (
        panel.with_columns(residual.alias("_residual"))
        .group_by(SESSION_COLUMN)
        .agg(
            pl.col("_residual").quantile(0.01).alias("__lo"),
            pl.col("_residual").quantile(0.99).alias("__hi"),
        )
    )
    clipped = (
        panel.with_columns(residual.alias("_residual"))
        .join(winsor, on=SESSION_COLUMN, how="left")
        .with_columns(
            pl.col("_residual").clip(pl.col("__lo"), pl.col("__hi")).alias("__clipped")
        )
    )
    count = pl.col("_residual").count().over(SESSION_COLUMN)
    pct_rank = (
        (pl.col("__clipped").rank("average").over(SESSION_COLUMN) - 1.0) / (count - 1.0)
    ).fill_null(0.5)
    relevance = (
        pl.when(pct_rank.is_not_null())
        .then((pct_rank * 5.0).floor().cast(pl.Int8).clip(0, 4))
        .otherwise(None)
    )

    out = clipped.with_columns(
        session_utc.alias(SESSION_COLUMN),
        pl.col("_residual").alias(label_column),
        relevance.alias(RELEVANCE_COLUMN),
        label_available.alias(LABEL_AVAILABLE_COLUMN),
    ).select(
        ID_COLUMN,
        SESSION_COLUMN,
        label_column,
        RELEVANCE_COLUMN,
        LABEL_AVAILABLE_COLUMN,
    )

    eligible_sessions = (
        out.filter(pl.col(label_column).is_not_null())
        .group_by(SESSION_COLUMN)
        .len()
        .filter(pl.col("len") >= MIN_RESIDUAL_GROUP)[SESSION_COLUMN]
    )
    if eligible_sessions.is_empty():
        logger.info(
            "residual label %s: no session has at least %s finite residuals",
            label_column,
            MIN_RESIDUAL_GROUP,
        )
        return out.head(0)
    out = out.filter(
        pl.col(SESSION_COLUMN).is_in(eligible_sessions.to_list())
    ).sort([ID_COLUMN, SESSION_COLUMN])
    incomplete = out.filter(
        pl.col(LABEL_AVAILABLE_COLUMN).is_null() | pl.col(label_column).is_null()
    )
    if not incomplete.is_empty():
        raise ValueError("residual label builder emitted an incomplete row")
    logger.info(
        "built residual label %s: %s rows over %s eligible sessions",
        label_column,
        out.height,
        len(eligible_sessions),
    )
    return out

def build_multi_horizon_residual_label_dataset(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    *,
    horizons: tuple[int, ...] = (5, 10, 15),
) -> pl.DataFrame:
    """Build a key-aligned multi-horizon residual open-to-open label panel.

    Each horizon ``h`` is computed independently with the same calendar-correct
    open-to-open residual and relevance semantics as the single-horizon builder,
    then the panels are inner-joined on ``(instrument_id, session)`` so every
    emitted key carries all horizons and all selected routes share an identical
    universe. A decision-session key is emitted only when every horizon has a
    finite label, its own ``relevance``, and at least ``MIN_RESIDUAL_GROUP``
    finite same-session constituents for each relevance calculation. The result
    schema is exactly:

    ``instrument_id, session, residual_o2o_<h0>d, relevance_<h0>d,
    label_available_time_<h0>d, residual_o2o_<h1>d, ...``

    with each ``label_available_time_<h>d`` bound to that horizon's terminal
    exit-session open in UTC.

    Raises:
        ValueError: for an empty, unsupported, duplicated, or non-ascending
            horizon set, or when any emitted key has a non-finite horizon label.
    """
    if not horizons:
        raise ValueError("horizons must be non-empty")
    if tuple(horizons) != tuple(sorted(set(horizons))):
        raise ValueError("horizons must be strictly ascending and unique")
    unsupported = [h for h in horizons if h not in SUPPORTED_RESIDUAL_HORIZONS]
    if unsupported:
        raise ValueError(
            f"unsupported residual horizons {unsupported}; "
            f"supported {SUPPORTED_RESIDUAL_HORIZONS}"
        )
    horizon_frames = [
        build_residual_o2o_label_dataset(base_panel, calendar, horizon_sessions=h)
        for h in horizons
    ]
    renamed: list[pl.DataFrame] = []
    for h, frame in zip(horizons, horizon_frames, strict=True):
        renamed.append(
            frame.rename(
                {
                    RELEVANCE_COLUMN: f"relevance_{h}d",
                    LABEL_AVAILABLE_COLUMN: f"label_available_time_{h}d",
                }
            )
        )
    out = renamed[0]
    for frame in renamed[1:]:
        out = out.join(frame, on=[ID_COLUMN, SESSION_COLUMN], how="inner")
    out = out.sort([ID_COLUMN, SESSION_COLUMN])
    finite_columns = [
        c for c in out.columns if c.startswith(RESIDUAL_O2O_PREFIX)
    ]
    incomplete = out.filter(
        pl.any_horizontal(
            pl.col(c).is_null() | ~pl.col(c).is_finite() for c in finite_columns
        )
    )
    if not incomplete.is_empty():
        raise ValueError("multi-horizon label builder emitted an incomplete row")
    logger.info(
        "built multi-horizon residual label panel %s: %s rows",
        tuple(horizons),
        out.height,
    )
    return out


def label_available_time(exit_session: date) -> datetime:
    """Availability timestamp for a terminal horizon session (after close)."""
    return datetime.combine(exit_session, _KRX_AVAILABLE_TIME, tzinfo=_KRX_TZ).astimezone(UTC)


def _round_trip_cost_rate(
    cost_schedule: CostSchedule,
    liquidity_model: LiquiditySlippageModel,
    *,
    decision_time: datetime,
    adtv: float,
    volatility: float,
    reference_price: float,
    participation: float,
) -> float:
    """One round trip cost rate (decimal) for the reference notional.

    The reference notional is ``participation * decision_time_ADTV``; slippage
    uses the point-in-time liquidity model at that notional, and the round trip
    adds buy/sell commission once each plus the sell-side tax. Fails closed on
    non-positive inputs or a missing cost-coverage decision time.
    """
    if participation <= 0:
        raise ValueError("reference participation must be positive")
    notional = participation * adtv
    if notional <= 0:
        raise ValueError("reference notional must be positive")
    point = cost_schedule.cost_for(decision_time)
    slippage_bps = liquidity_model.slippage_bps(
        notional=notional,
        adtv_20d=adtv,
        daily_volatility=volatility,
        reference_price=reference_price,
        effective_time=decision_time,
    )
    return (
        2.0 * point.commission_rate + point.tax_rate + 2.0 * slippage_bps / 10_000.0
    )


def _project_risk_return(
    design: np.ndarray,
    gross: np.ndarray,
) -> np.ndarray:
    """Equal-weight cross-sectional risk projection via deterministic SVD.

    Solves ``min ||gross - design @ beta||`` with ``np.linalg.lstsq``
    (deterministic SVD path) and returns the fitted values. The caller rejects
    rank-deficient or non-finite sessions; no row is ever fabricated here.
    """
    coefficients, _, rank, _ = np.linalg.lstsq(design, gross, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("risk projection is rank-deficient")
    fitted = np.asarray(design @ coefficients, dtype=np.float64)
    if not np.all(np.isfinite(fitted)):
        raise ValueError("risk projection produced non-finite fitted returns")
    return fitted


def _sector_design_matrix(sector_labels: np.ndarray, log_sizes: np.ndarray, betas: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """Deterministic equal-weight design: intercept, sector dummies, controls.

    Sector dummies omit the lexicographically first sector (reference category)
    to keep the design full rank; the three decision-time controls are
    standardized cross-sectionally (z-scores) after log-size conversion.
    """
    sectors = np.asarray([str(s) for s in sector_labels], dtype=object)
    unique = np.sort(np.unique(sectors))
    dummy_cols = [
        (sectors == sector).astype(np.float64) for sector in unique[1:]
    ]
    controls = [log_sizes, betas, vols]
    for values in controls:
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        # A constant control contributes no explanatory dimension and would
        # make the least-squares design rank deficient. Omit it explicitly;
        # this is important for source vintages without point-in-time beta.
        if std > 0:
            dummy_cols.append(
                np.where(np.isnan(values), 0.0, (values - mean) / std)
            )
    return np.column_stack([np.ones(sectors.shape[0], dtype=np.float64), *dummy_cols])


def build_single_horizon_cost_aware_residual_labels(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    cost_schedule: CostSchedule,
    liquidity_model: LiquiditySlippageModel,
    *,
    horizon_sessions: int,
    reference_participation: float,
) -> pl.DataFrame:
    """Build one cost-aware, risk-residualized open-to-open label horizon.

    For each decision session with at least ``_MIN_COST_AWARE_ROWS_PER_SESSION``
    finite constituents, the gross open-to-open return is projected on an
    equal-weight cross-sectional design (intercept, sector dummies omitting the
    lexicographically first sector, decision-time standardized log-size, beta,
    and volatility) with deterministic SVD. The risk residual is
    ``gross - fitted`` and the net residual subtracts the point-in-time round
    trip cost at ``reference_participation * decision_time_ADTV``. Sessions that
    are rank-deficient, undersized, or non-finite are rejected outright.

    The result schema is ``instrument_id, session, gross_o2o_<h>d,
    risk_fitted_<h>d, risk_residual_<h>d, reference_cost_<h>d,
    net_residual_<h>d, relevance_<h>d, label_available_time_<h>d`` where
    ``relevance`` is the target-only cross-sectional quintile of the net
    residual and ``label_available_time`` is the terminal exit-open timestamp.
    """
    if horizon_sessions <= 0:
        raise ValueError("horizon_sessions must be positive")
    if reference_participation <= 0:
        raise ValueError("reference_participation must be positive")
    required = (
        ID_COLUMN,
        SESSION_COLUMN,
        "open",
        "sector",
        "adtv",
        *_COST_AWARE_CONTROL_COLUMNS,
    )
    missing = [c for c in required if c not in base_panel.columns]
    if missing:
        raise ValueError(f"cost-aware residual label requires base panel columns {missing}")

    sessions = list(calendar.sessions)
    by_date = {session: index for index, session in enumerate(sessions)}
    if len(by_date) != len(sessions):
        raise ValueError("calendar contains duplicate sessions")

    panel = base_panel.with_columns(pl.col(SESSION_COLUMN).cast(pl.Date).alias("_session_date"))
    calendar_frame = pl.DataFrame(
        {
            "_session_date": sessions,
            "_cal_pos": list(range(len(sessions))),
            "_entry_date": [sessions[p + 1] if p + 1 < len(sessions) else None for p in range(len(sessions))],
            "_exit_date": [
                sessions[p + 1 + horizon_sessions]
                if p + 1 + horizon_sessions < len(sessions)
                else None
                for p in range(len(sessions))
            ],
        }
    )
    panel = panel.join(calendar_frame, on="_session_date", how="left")
    unknown = panel.filter(pl.col("_cal_pos").is_null() | pl.col("_session_date").is_null())
    if not unknown.is_empty():
        raise ValueError("base panel contains non-calendar sessions")

    prices = panel.select(
        ID_COLUMN,
        pl.col("_session_date").alias("_price_date"),
        pl.col("open"),
    )
    entries = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_entry_date"),
        pl.col("open").alias("_entry_open"),
    )
    exits = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_exit_date"),
        pl.col("open").alias("_exit_open"),
    )
    panel = (
        panel.join(entries, on=[ID_COLUMN, "_entry_date"], how="left")
        .join(exits, on=[ID_COLUMN, "_exit_date"], how="left")
        .filter(pl.col("_entry_open").is_not_null() & pl.col("_exit_open").is_not_null())
    )
    gross = pl.col("_exit_open").log() - pl.col("_entry_open").log()
    panel = panel.with_columns(gross.alias("_gross"))
    label_available = (
        pl.col("_exit_date")
        .dt.combine(pl.lit(_KRX_AVAILABLE_TIME))
        .dt.replace_time_zone("Asia/Seoul")
        .dt.convert_time_zone("UTC")
    )
    panel = panel.with_columns(label_available.alias("_label_available"))

    suffix = f"{horizon_sessions}d"
    gross_column = f"{COST_AWARE_GROSS_PREFIX}{suffix}"
    fitted_column = f"{COST_AWARE_RISK_FITTED_PREFIX}{suffix}"
    residual_column = f"{COST_AWARE_RISK_RESIDUAL_PREFIX}{suffix}"
    cost_column = f"{COST_AWARE_REFERENCE_COST_PREFIX}{suffix}"
    net_column = f"{COST_AWARE_O2O_PREFIX}{suffix}"
    relevance_column = f"relevance_{suffix}"
    available_column = f"label_available_time_{suffix}"

    log_size = panel.select(
        ID_COLUMN, SESSION_COLUMN, pl.col("market_cap").log().alias("_log_size")
    )
    panel = panel.join(log_size, on=[ID_COLUMN, SESSION_COLUMN], how="left")

    rows: list[dict[str, object]] = []
    for session_date in panel["_session_date"].unique().sort():
        session = panel.filter(pl.col("_session_date") == session_date)
        finite = session.filter(
            pl.col("_gross").is_not_null()
            & pl.col("_gross").is_finite()
            & pl.col("_log_size").is_not_null()
            & pl.col("beta").is_not_null()
            & pl.col("volatility").is_not_null()
            & pl.col("adtv").is_not_null()
            & (pl.col("adtv") > 0)
            & pl.col("sector").is_not_null()
            & pl.col("open").is_not_null()
            & (pl.col("open") > 0)
        )
        if finite.height < _MIN_COST_AWARE_ROWS_PER_SESSION:
            continue
        design = _sector_design_matrix(
            np.asarray(finite["sector"].to_list(), dtype=object),
            np.asarray(finite["_log_size"].to_list(), dtype=np.float64),
            np.asarray(finite["beta"].to_list(), dtype=np.float64),
            np.asarray(finite["volatility"].to_list(), dtype=np.float64),
        )
        if design.shape[0] <= design.shape[1]:
            continue
        gross_values = np.asarray(finite["_gross"].to_list(), dtype=np.float64)
        if not np.all(np.isfinite(design)):
            continue
        try:
            fitted = _project_risk_return(design, gross_values)
        except ValueError:
            continue
        residual = gross_values - fitted
        adtvs = np.asarray(finite["adtv"].to_list(), dtype=np.float64)
        vols = np.asarray(finite["volatility"].to_list(), dtype=np.float64)
        reference_prices = np.asarray(finite["open"].to_list(), dtype=np.float64)
        decision_times = [_as_utc_datetime(value) for value in finite[SESSION_COLUMN].to_list()]
        costs = [
            _round_trip_cost_rate(
                cost_schedule,
                liquidity_model,
                decision_time=decision_time,
                adtv=float(adtv),
                volatility=float(vol),
                reference_price=float(price),
                participation=reference_participation,
            )
            for decision_time, adtv, vol, price in zip(
                decision_times, adtvs, vols, reference_prices, strict=True
            )
        ]
        cost_array = np.asarray(costs, dtype=np.float64)
        net = residual - cost_array
        order = np.argsort(net, kind="stable")
        n = net.shape[0]
        quintile = np.zeros(n, dtype=np.int8)
        for bucket in range(5):
            lo = int(np.ceil(bucket * n / 5.0))
            hi = int(np.floor((bucket + 1) * n / 5.0))
            quintile[order[lo:hi]] = bucket
        available_times = finite["_label_available"].to_list()
        for index, row in enumerate(finite.iter_rows(named=True)):
            rows.append(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    SESSION_COLUMN: _as_utc_datetime(row[SESSION_COLUMN]),
                    gross_column: float(gross_values[index]),
                    fitted_column: float(fitted[index]),
                    residual_column: float(residual[index]),
                    cost_column: float(cost_array[index]),
                    net_column: float(net[index]),
                    relevance_column: int(quintile[index]),
                    available_column: available_times[index],
                }
            )
    if not rows:
        logger.info(
            "cost-aware residual label %s: no session cleared risk/cost gates",
            net_column,
        )
        return pl.DataFrame(
            schema={
                ID_COLUMN: pl.Utf8,
                SESSION_COLUMN: pl.Datetime("us", "UTC"),
                gross_column: pl.Float64,
                fitted_column: pl.Float64,
                residual_column: pl.Float64,
                cost_column: pl.Float64,
                net_column: pl.Float64,
                relevance_column: pl.Int8,
                available_column: pl.Datetime("us", "UTC"),
            }
        )
    out = pl.DataFrame(rows)
    logger.info(
        "built cost-aware residual label %s: %s rows",
        net_column,
        out.height,
    )
    return out


def build_multi_horizon_cost_aware_residual_label_dataset(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    cost_schedule: CostSchedule,
    liquidity_model: LiquiditySlippageModel,
    *,
    horizons: tuple[int, ...] = (5, 10, 15),
    reference_participation: float,
) -> pl.DataFrame:
    """Build a key-aligned multi-horizon cost-aware residual label panel.

    Each horizon is computed with the calendar-correct open-to-open gross
    return, the cross-sectional risk projection, and the point-in-time
    reference cost exactly as the single-horizon builder, then the panels are
    inner-joined on ``(instrument_id, session)`` so all routes share an
    identical universe. ``reference_participation`` is the participation rate
    used to size the reference round-trip cost notional.

    The result schema is ``instrument_id, session, gross_o2o_<h0>d,
    risk_fitted_<h0>d, risk_residual_<h0>d, reference_cost_<h0>d,
    net_residual_<h0>d, relevance_<h0>d, label_available_time_<h0>d,
    gross_o2o_<h1>d, ...``.

    Raises:
        ValueError: for an empty, unsupported, duplicated, or non-ascending
            horizon set, a non-positive participation rate, missing risk/cost
            inputs, or when any emitted key has a non-finite net residual.
    """
    if not horizons:
        raise ValueError("horizons must be non-empty")
    if tuple(horizons) != tuple(sorted(set(horizons))):
        raise ValueError("horizons must be strictly ascending and unique")
    unsupported = [h for h in horizons if h not in SUPPORTED_RESIDUAL_HORIZONS]
    if unsupported:
        raise ValueError(
            f"unsupported residual horizons {unsupported}; "
            f"supported {SUPPORTED_RESIDUAL_HORIZONS}"
        )
    if reference_participation <= 0:
        raise ValueError("reference_participation must be positive")
    horizon_frames = [
        build_single_horizon_cost_aware_residual_labels(
            base_panel,
            calendar,
            cost_schedule,
            liquidity_model,
            horizon_sessions=h,
            reference_participation=reference_participation,
        )
        for h in horizons
    ]
    out = horizon_frames[0]
    for frame in horizon_frames[1:]:
        out = out.join(frame, on=[ID_COLUMN, SESSION_COLUMN], how="inner")
    out = out.sort([ID_COLUMN, SESSION_COLUMN])
    incomplete = out.filter(
        pl.any_horizontal(
            pl.col(c).is_null() | ~pl.col(c).is_finite()
            for c in out.columns
            if c not in (ID_COLUMN, SESSION_COLUMN)
            and not c.startswith("label_available_time_")
        )
    )
    if not incomplete.is_empty():
        raise ValueError("multi-horizon cost-aware label builder emitted an incomplete row")
    logger.info(
        "built multi-horizon cost-aware residual label panel %s: %s rows",
        tuple(horizons),
        out.height,
    )
    return out


def build_net_alpha_label_dataset(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    cost_schedule: CostSchedule,
    liquidity_model: LiquiditySlippageModel,
    *,
    horizon_sessions: int,
    reference_notional: float,
) -> pl.DataFrame:
    """Build a continuous, session-robust net-alpha label horizon.

    Replaces the quintile ``relevance`` target with the cost/risk-aware net
    residual open-to-open label normalized to a continuous session-robust
    scale: within each decision session the net residual is median-centered and
    divided by its MAD (median absolute deviation), so the cross-sectional
    target is comparable across sessions and preserves return magnitude. A
    session whose MAD is zero is unlearnable and is excluded with its reason
    recorded.

    ``reference_notional`` is the reference notional for the round-trip cost
    (derived from the replay minimum order unit and portfolio value, never an
    arbitrary participation constant). ``label_available_time`` is the exact
    exit-session open timestamp in UTC.

    The result schema is ``instrument_id, session, gross_o2o_<h>d,
    risk_fitted_<h>d, risk_residual_<h>d, reference_cost_<h>d,
    net_residual_o2o_<h>d, net_alpha_<h>d_target, label_available_time_<h>d``
    with no relevance column.

    This is the realised-label projection of
    :func:`build_net_alpha_label_dataset_with_status`; the two-return variant
    also emits the outcome-status sidecar that classifies every rejected
    decision key.
    """
    labels, _status = build_net_alpha_label_dataset_with_status(
        base_panel,
        calendar,
        cost_schedule,
        liquidity_model,
        horizon_sessions=horizon_sessions,
        reference_notional=reference_notional,
    )
    return labels


def build_net_alpha_label_dataset_with_status(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    cost_schedule: CostSchedule,
    liquidity_model: LiquiditySlippageModel,
    *,
    horizon_sessions: int,
    reference_notional: float,
    policy: ExecutionOutcomePolicy | None = None,
    bar_evidence: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a net-alpha label horizon and its typed outcome-status sidecar.

    The realised label frame is exactly the current :func:`build_net_alpha_label_dataset`
    output. The status sidecar additionally emits exactly one row per decision
    key ``(instrument_id, session)`` with a typed state from the fixed
    vocabulary: ``REALIZED``, ``PARTIAL_TAIL`` (required exit beyond the
    calendar tail), ``MISSING_ENTRY_PRICE``/``MISSING_EXIT_PRICE``,
    ``MISSING_DECISION_INPUT`` (control unavailable), and the label
    construction failures ``UNDERSIZED_CROSS_SECTION``, ``RISK_PROJECTION_FAILED``,
    and ``ZERO_MAD``. A key is never silently dropped without a typed state, so
    the outcome evidence remains economically evaluable.

    Entry/exit sessions are resolved under ``policy`` (default
    ``scheduled_open_v1``): the scheduled policy requires the exact scheduled
    opens and a deferred policy may use the first verified open no later than
    its explicit delay bounds. A missing open is never zero-filled,
    forward-filled, or replaced by another OHLC field.
    """
    if horizon_sessions <= 0:
        raise ValueError("horizon_sessions must be positive")
    if reference_notional <= 0:
        raise ValueError("reference_notional must be positive")
    policy = policy or SCHEDULED_OPEN_V1
    required = (
        ID_COLUMN,
        SESSION_COLUMN,
        "open",
        "sector",
        "adtv",
        *_COST_AWARE_CONTROL_COLUMNS,
    )
    missing = [c for c in required if c not in base_panel.columns]
    if missing:
        raise ValueError(f"net-alpha label requires base panel columns {missing}")

    panel = base_panel.with_columns(pl.col(SESSION_COLUMN).cast(pl.Date).alias("_session_date"))
    evidence = resolve_policy_outcome(
        panel,
        calendar,
        horizon_sessions=horizon_sessions,
        policy=policy,
        bar_evidence=bar_evidence,
    )
    prices = merge_open_bar_evidence(panel, bar_evidence).select(
        ID_COLUMN,
        pl.col("price_date").alias("_price_date"),
        pl.col("open"),
    )
    controls = panel.select(
        ID_COLUMN,
        pl.col("_session_date").alias("_decision_date"),
        "sector",
        "adtv",
        "market_cap",
        "beta",
        "volatility",
        "open",
    )
    entry_prices = prices.rename({"_price_date": "_actual_entry_date", "open": "_entry_open"})
    exit_prices = prices.rename({"_price_date": "_actual_exit_date", "open": "_exit_open"})
    joined = (
        evidence.join(
            controls,
            left_on=[ID_COLUMN, "session"],
            right_on=[ID_COLUMN, "_decision_date"],
            how="left",
        )
        .join(
            entry_prices,
            left_on=[ID_COLUMN, "actual_entry_session"],
            right_on=[ID_COLUMN, "_actual_entry_date"],
            how="left",
        )
        .join(
            exit_prices,
            left_on=[ID_COLUMN, "actual_exit_session"],
            right_on=[ID_COLUMN, "_actual_exit_date"],
            how="left",
        )
    )
    complete = joined.filter(pl.col("outcome_status") == "REALIZED")
    incomplete = evidence.filter(pl.col("outcome_status") != "REALIZED")
    gross = pl.col("_exit_open").log() - pl.col("_entry_open").log()
    complete = complete.with_columns(
        gross.alias("_gross"),
        pl.col("label_available_time").alias("_label_available"),
        pl.col("session").alias("_session_date"),
    )
    panel = complete

    suffix = f"{horizon_sessions}d"
    gross_column = f"{COST_AWARE_GROSS_PREFIX}{suffix}"
    fitted_column = f"{COST_AWARE_RISK_FITTED_PREFIX}{suffix}"
    residual_column = f"{COST_AWARE_RISK_RESIDUAL_PREFIX}{suffix}"
    cost_column = f"{COST_AWARE_REFERENCE_COST_PREFIX}{suffix}"
    net_column = f"{COST_AWARE_O2O_PREFIX}{suffix}"
    target_column = f"{NET_ALPHA_TARGET_PREFIX}{suffix}{NET_ALPHA_CONTINUOUS_SUFFIX}"
    available_column = f"label_available_time_{suffix}"

    log_size = panel.select(
        ID_COLUMN, SESSION_COLUMN, pl.col("market_cap").log().alias("_log_size")
    )
    panel = panel.join(log_size, on=[ID_COLUMN, SESSION_COLUMN], how="left")

    rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    rejected_sessions: list[tuple[object, str]] = []
    for session_date in panel["_session_date"].unique().sort():
        session = panel.filter(pl.col("_session_date") == session_date)
        finite = session.filter(
            pl.col("_gross").is_not_null()
            & pl.col("_gross").is_finite()
            & pl.col("_log_size").is_not_null()
            & pl.col("beta").is_not_null()
            & pl.col("volatility").is_not_null()
            & pl.col("adtv").is_not_null()
            & (pl.col("adtv") > 0)
            & pl.col("sector").is_not_null()
            & pl.col("open").is_not_null()
            & (pl.col("open") > 0)
        )
        session_status = session.with_columns(
            pl.col("_gross")
            .is_not_null()
            .and_(pl.col("_gross").is_finite())
            .and_(pl.col("_log_size").is_not_null())
            .and_(pl.col("beta").is_not_null())
            .and_(pl.col("volatility").is_not_null())
            .and_(pl.col("adtv").is_not_null())
            .and_(pl.col("adtv") > 0)
            .and_(pl.col("sector").is_not_null())
            .and_(pl.col("open").is_not_null())
            .and_(pl.col("open") > 0)
            .alias("_finite")
        )
        missing_input = session_status.filter(~pl.col("_finite"))
        if not missing_input.is_empty():
            status_rows.extend(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    SESSION_COLUMN: _as_utc_datetime(row[SESSION_COLUMN]),
                    "outcome_status": "MISSING_DECISION_INPUT",
                }
                for row in missing_input.iter_rows(named=True)
            )
        if finite.height < _MIN_NET_ALPHA_ROWS_PER_SESSION:
            rejected_sessions.append((session_date, "undersized"))
            status_rows.extend(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    SESSION_COLUMN: _as_utc_datetime(row[SESSION_COLUMN]),
                    "outcome_status": "UNDERSIZED_CROSS_SECTION",
                }
                for row in finite.iter_rows(named=True)
            )
            continue
        design = _sector_design_matrix(
            np.asarray(finite["sector"].to_list(), dtype=object),
            np.asarray(finite["_log_size"].to_list(), dtype=np.float64),
            np.asarray(finite["beta"].to_list(), dtype=np.float64),
            np.asarray(finite["volatility"].to_list(), dtype=np.float64),
        )
        if design.shape[0] <= design.shape[1]:
            rejected_sessions.append((session_date, "rank-deficient"))
            status_rows.extend(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    SESSION_COLUMN: _as_utc_datetime(row[SESSION_COLUMN]),
                    "outcome_status": "RISK_PROJECTION_FAILED",
                }
                for row in finite.iter_rows(named=True)
            )
            continue
        gross_values = np.asarray(finite["_gross"].to_list(), dtype=np.float64)
        if not np.all(np.isfinite(design)):
            rejected_sessions.append((session_date, "non-finite-design"))
            status_rows.extend(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    SESSION_COLUMN: _as_utc_datetime(row[SESSION_COLUMN]),
                    "outcome_status": "RISK_PROJECTION_FAILED",
                }
                for row in finite.iter_rows(named=True)
            )
            continue
        try:
            fitted = _project_risk_return(design, gross_values)
        except ValueError:
            rejected_sessions.append((session_date, "risk-projection-failed"))
            status_rows.extend(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    SESSION_COLUMN: _as_utc_datetime(row[SESSION_COLUMN]),
                    "outcome_status": "RISK_PROJECTION_FAILED",
                }
                for row in finite.iter_rows(named=True)
            )
            continue
        residual = gross_values - fitted
        adtvs = np.asarray(finite["adtv"].to_list(), dtype=np.float64)
        vols = np.asarray(finite["volatility"].to_list(), dtype=np.float64)
        reference_prices = np.asarray(finite["open"].to_list(), dtype=np.float64)
        decision_times = [_as_utc_datetime(value) for value in finite[SESSION_COLUMN].to_list()]
        costs = np.asarray(
            [
                _round_trip_cost_rate(
                    cost_schedule,
                    liquidity_model,
                    decision_time=decision_time,
                    adtv=float(adtv),
                    volatility=float(vol),
                    reference_price=float(price),
                    participation=reference_notional / float(adtv),
                )
                for decision_time, adtv, vol, price in zip(
                    decision_times, adtvs, vols, reference_prices, strict=True
                )
            ],
            dtype=np.float64,
        )
        net = residual - costs
        median = float(np.median(net))
        mad = float(np.median(np.abs(net - median)))
        if not np.isfinite(mad) or mad <= 0.0:
            rejected_sessions.append((session_date, "zero-mad"))
            status_rows.extend(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    SESSION_COLUMN: _as_utc_datetime(row[SESSION_COLUMN]),
                    "outcome_status": "ZERO_MAD",
                }
                for row in finite.iter_rows(named=True)
            )
            continue
        target = (net - median) / mad
        available_times = finite["_label_available"].to_list()
        session_objects = finite[SESSION_COLUMN].to_list()
        instrument_ids = finite[ID_COLUMN].to_list()
        rows.extend(
            {
                ID_COLUMN: instrument_ids[index],
                SESSION_COLUMN: _as_utc_datetime(session_objects[index]),
                gross_column: float(gross_values[index]),
                fitted_column: float(fitted[index]),
                residual_column: float(residual[index]),
                cost_column: float(costs[index]),
                net_column: float(net[index]),
                target_column: float(target[index]),
                available_column: available_times[index],
            }
            for index in range(finite.height)
        )
        status_rows.extend(
            {
                ID_COLUMN: row[ID_COLUMN],
                SESSION_COLUMN: _as_utc_datetime(row[SESSION_COLUMN]),
                "outcome_status": "REALIZED",
            }
            for row in finite.iter_rows(named=True)
        )

    if not incomplete.is_empty():
        status_rows.extend(
            {
                ID_COLUMN: row[ID_COLUMN],
                SESSION_COLUMN: _as_utc_datetime(row[SESSION_COLUMN]),
                "outcome_status": row["outcome_status"],
            }
            for row in incomplete.iter_rows(named=True)
        )
    if not rows:
        logger.info(
            "net-alpha label %s: no session cleared risk/cost/MAD gates",
            target_column,
        )
        labels = pl.DataFrame(
            {
                ID_COLUMN: pl.Series(dtype=pl.Utf8),
                SESSION_COLUMN: pl.Series(dtype=pl.Datetime("us", "UTC")),
                gross_column: pl.Series(dtype=pl.Float64),
                fitted_column: pl.Series(dtype=pl.Float64),
                residual_column: pl.Series(dtype=pl.Float64),
                cost_column: pl.Series(dtype=pl.Float64),
                net_column: pl.Series(dtype=pl.Float64),
                target_column: pl.Series(dtype=pl.Float64),
                available_column: pl.Series(dtype=pl.Datetime("us", "UTC")),
            }
        )
    else:
        labels = pl.DataFrame(rows)
    status_frame = pl.DataFrame(
        status_rows,
        schema={
            ID_COLUMN: pl.Utf8,
            SESSION_COLUMN: pl.Datetime("us", "UTC"),
            "outcome_status": pl.Utf8,
        },
    ).sort([ID_COLUMN, SESSION_COLUMN])
    unknown = status_frame.filter(
        ~pl.col("outcome_status").is_in(
            ("REALIZED", "PARTIAL_TAIL", "MISSING_ENTRY_PRICE",
             "MISSING_EXIT_PRICE", "MISSING_DECISION_INPUT",
             "UNDERSIZED_CROSS_SECTION", "RISK_PROJECTION_FAILED",
             "ZERO_MAD", "UNSUPPORTED_CORPORATE_ACTION")
        )
    )
    if not unknown.is_empty():
        raise ValueError("net-alpha status sidecar emitted an unknown state")
    logger.info(
        "built net-alpha label %s: %s rows, rejected sessions %s",
        target_column,
        labels.height,
        rejected_sessions,
    )
    return labels, status_frame


@dataclass(frozen=True, slots=True)
class LabelDatasetResult:
    """Immutable outcome of a label-dataset publication."""

    dataset_id: str
    manifest: DatasetManifest
    partition_paths: tuple[Path, ...]
    row_count: int
    base_panel_hash: str


def publish_label_dataset(
    labels_frame: pl.DataFrame,
    *,
    destination_root: Path,
    dataset_id: str,
    base_panel_hash: str,
    calendar_hash: str,
    definition: LabelDefinition,
    provider_version: str = "base-panel-labels",
    universe_policy_version: str = "provisional-legacy",
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    generated_time: datetime | None = None,
    algorithm_version: str = "",
) -> LabelDatasetResult:
    """Publish an immutable, session-keyed label dataset.

    The label dataset never lives in a feature panel; it is a separate
    ``canonical/stocks/labels/<label-id>`` version keyed by
    ``(instrument_id, session)`` and bound to the base-panel and calendar
    hashes it was built from.
    """
    required = (ID_COLUMN, SESSION_COLUMN, definition.name, LABEL_AVAILABLE_COLUMN)
    missing = [c for c in required if c not in labels_frame.columns]
    if missing:
        raise ValueError(f"label dataset missing columns {missing}")
    if labels_frame.is_empty():
        raise ValueError("cannot publish an empty label dataset")
    if labels_frame.filter(pl.col(LABEL_AVAILABLE_COLUMN).is_null()).height:
        raise ValueError("label dataset contains rows without label_available_time")

    generated_time = generated_time or datetime.now(UTC)
    ordered_columns = list(labels_frame.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set=LABEL_FEATURE_SET,
        label_definition=definition.name,
        label_horizon_sessions=definition.horizon_sessions,
        time_start=_as_utc_datetime(labels_frame[SESSION_COLUMN].min()),
        time_end=_as_utc_datetime(labels_frame[SESSION_COLUMN].max()),
        provider_version=provider_version,
        universe_policy_version=universe_policy_version,
        row_count=labels_frame.height,
        generated_time=generated_time,
        certification=certification,
        calendar_hash=calendar_hash,
        schema_version="v2",
        content_hash=canonical_content_hash(labels_frame, ordered_columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    content_manifest: dict[str, object] = {
        "base_panel_hash": base_panel_hash,
        "calendar_hash": calendar_hash,
        "label_definition": definition.name,
        "label_horizon_sessions": definition.horizon_sessions,
        "entry_field": definition.entry_field,
        "exit_field": definition.exit_field,
        "generated_time": generated_time.isoformat(),
    }
    if algorithm_version:
        content_manifest["label_algorithm_version"] = algorithm_version
    store = ParquetDatasetStore(Path(destination_root))
    dataset_dir = store.write_partitioned(
        labels_frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=LABEL_FEATURE_SET,
        decision_time=generated_time,
        content_manifest=content_manifest,
    )
    partition_paths = tuple(sorted((dataset_dir / "partitions").rglob("*.parquet")))
    logger.info(
        "published label dataset %s: %s rows, horizon %s",
        dataset_id,
        labels_frame.height,
        definition.horizon_sessions,
    )
    return LabelDatasetResult(
        dataset_id=dataset_id,
        manifest=manifest,
        partition_paths=partition_paths,
        row_count=labels_frame.height,
        base_panel_hash=base_panel_hash,
    )


def publish_residual_o2o_label_dataset(
    labels_frame: pl.DataFrame,
    *,
    destination_root: Path,
    dataset_id: str,
    base_panel_hash: str,
    calendar_hash: str,
    horizon_sessions: int = 5,
    provider_version: str = "base-panel-labels",
    universe_policy_version: str = "provisional-legacy",
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    generated_time: datetime | None = None,
) -> LabelDatasetResult:
    """Publish a calendar-correct residual open-to-open label dataset.

    A dedicated publisher so the generic ``fwd_ret_*`` path stays unchanged for
    v1 consumers. The label column is ``residual_o2o_<horizon>d`` and the
    content manifest records the residual-label algorithm version.
    """
    label_column = f"{RESIDUAL_O2O_PREFIX}{horizon_sessions}d"
    definition = LabelDefinition(
        name=label_column,
        entry_field="open",
        exit_field="open",
        horizon_sessions=horizon_sessions,
    )
    return publish_label_dataset(
        labels_frame,
        destination_root=destination_root,
        dataset_id=dataset_id,
        base_panel_hash=base_panel_hash,
        calendar_hash=calendar_hash,
        definition=definition,
        provider_version=provider_version,
        universe_policy_version=universe_policy_version,
        certification=certification,
        generated_time=generated_time,
        algorithm_version=RESIDUAL_ALGORITHM_VERSION,
    )

def publish_multi_horizon_residual_label_dataset(
    labels_frame: pl.DataFrame,
    *,
    destination_root: Path,
    dataset_id: str,
    base_panel_hash: str,
    calendar_hash: str,
    horizons: tuple[int, ...] = (5, 10, 15),
    provider_version: str = "base-panel-labels",
    universe_policy_version: str = "provisional-legacy",
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    generated_time: datetime | None = None,
) -> LabelDatasetResult:
    """Publish the immutable multi-horizon residual label panel.

    The manifest declares ``label_definition="residual_o2o_multi_5_10_15d"``
    and ``label_horizon_sessions`` equal to the control horizon (the first
    element of ``horizons``), so v2 consumers keyed on the five-day primary
    label keep a stable contract. The content manifest records the ordered
    horizon set and the residual-label algorithm version; the schema/content
    hashes bind the exact column order and values.
    """
    if not horizons:
        raise ValueError("horizons must be non-empty")
    expected_columns: list[str] = [ID_COLUMN, SESSION_COLUMN]
    for h in horizons:
        expected_columns += [
            f"{RESIDUAL_O2O_PREFIX}{h}d",
            f"relevance_{h}d",
            f"label_available_time_{h}d",
        ]
    if labels_frame.columns != expected_columns:
        raise ValueError(
            "multi-horizon label dataset must carry exactly "
            f"{expected_columns}, got {labels_frame.columns}"
        )
    if labels_frame.is_empty():
        raise ValueError("cannot publish an empty multi-horizon label dataset")
    control_available = f"label_available_time_{horizons[0]}d"
    if labels_frame.filter(pl.col(control_available).is_null()).height:
        raise ValueError(
            "multi-horizon label dataset contains rows without the control "
            "label_available_time"
        )
    generated_time = generated_time or datetime.now(UTC)
    ordered_columns = list(labels_frame.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set=LABEL_FEATURE_SET,
        label_definition=MULTI_HORIZON_RESIDUAL_DEFINITION,
        label_horizon_sessions=horizons[0],
        time_start=_as_utc_datetime(labels_frame[SESSION_COLUMN].min()),
        time_end=_as_utc_datetime(labels_frame[SESSION_COLUMN].max()),
        provider_version=provider_version,
        universe_policy_version=universe_policy_version,
        row_count=labels_frame.height,
        generated_time=generated_time,
        certification=certification,
        calendar_hash=calendar_hash,
        schema_version="v2",
        content_hash=canonical_content_hash(labels_frame, ordered_columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    content_manifest: dict[str, object] = {
        "base_panel_hash": base_panel_hash,
        "calendar_hash": calendar_hash,
        "label_definition": MULTI_HORIZON_RESIDUAL_DEFINITION,
        "label_horizon_sessions": horizons[0],
        "horizons": list(horizons),
        "entry_field": "open",
        "exit_field": "open",
        "generated_time": generated_time.isoformat(),
        "label_algorithm_version": RESIDUAL_ALGORITHM_VERSION,
    }
    store = ParquetDatasetStore(Path(destination_root))
    dataset_dir = store.write_partitioned(
        labels_frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=LABEL_FEATURE_SET,
        decision_time=generated_time,
        content_manifest=content_manifest,
    )
    partition_paths = tuple(sorted((dataset_dir / "partitions").rglob("*.parquet")))
    logger.info(
        "published multi-horizon label dataset %s: %s rows, horizons %s",
        dataset_id,
        labels_frame.height,
        list(horizons),
    )
    return LabelDatasetResult(
        dataset_id=dataset_id,
        manifest=manifest,
        partition_paths=partition_paths,
        row_count=labels_frame.height,
        base_panel_hash=base_panel_hash,
    )


def publish_net_alpha_label_dataset(
    labels_frame: pl.DataFrame,
    *,
    destination_root: Path,
    dataset_id: str,
    base_panel_hash: str,
    calendar_hash: str,
    horizon_sessions: int,
    provider_version: str = "base-panel-labels",
    universe_policy_version: str = "provisional-legacy",
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    generated_time: datetime | None = None,
) -> LabelDatasetResult:
    """Publish a continuous net-alpha label horizon.

    The manifest declares ``label_definition="net_alpha_o2o"`` with the exact
    horizon; the content manifest records the net-alpha algorithm version and
    the reference notional semantics. Each horizon is published independently
    (no common-universe inner join), so the training side records per-horizon
    label universes instead of shrinking the sample.
    """
    if horizon_sessions <= 0:
        raise ValueError("horizon_sessions must be positive")
    suffix = f"{horizon_sessions}d"
    definition = LabelDefinition(
        name=f"{COST_AWARE_O2O_PREFIX}{suffix}",
        entry_field="open",
        exit_field="open",
        horizon_sessions=horizon_sessions,
    )
    required = (ID_COLUMN, SESSION_COLUMN, definition.name, LABEL_AVAILABLE_COLUMN)
    missing = [c for c in required if c not in labels_frame.columns]
    if missing:
        raise ValueError(f"net-alpha label dataset missing columns {missing}")
    if labels_frame.is_empty():
        raise ValueError("cannot publish an empty net-alpha label dataset")
    generated_time = generated_time or datetime.now(UTC)
    ordered_columns = list(labels_frame.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set=LABEL_FEATURE_SET,
        label_definition=NET_ALPHA_DEFINITION,
        label_horizon_sessions=horizon_sessions,
        time_start=_as_utc_datetime(labels_frame[SESSION_COLUMN].min()),
        time_end=_as_utc_datetime(labels_frame[SESSION_COLUMN].max()),
        provider_version=provider_version,
        universe_policy_version=universe_policy_version,
        row_count=labels_frame.height,
        generated_time=generated_time,
        certification=certification,
        calendar_hash=calendar_hash,
        schema_version="v2",
        content_hash=canonical_content_hash(labels_frame, ordered_columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    content_manifest: dict[str, object] = {
        "base_panel_hash": base_panel_hash,
        "calendar_hash": calendar_hash,
        "label_definition": NET_ALPHA_DEFINITION,
        "label_horizon_sessions": horizon_sessions,
        "entry_field": "open",
        "exit_field": "open",
        "generated_time": generated_time.isoformat(),
        "label_algorithm_version": NET_ALPHA_ALGORITHM_VERSION,
    }
    store = ParquetDatasetStore(Path(destination_root))
    dataset_dir = store.write_partitioned(
        labels_frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=LABEL_FEATURE_SET,
        decision_time=generated_time,
        content_manifest=content_manifest,
    )
    partition_paths = tuple(sorted((dataset_dir / "partitions").rglob("*.parquet")))
    logger.info(
        "published net-alpha label dataset %s: %s rows, horizon %s",
        dataset_id,
        labels_frame.height,
        horizon_sessions,
    )
    return LabelDatasetResult(
        dataset_id=dataset_id,
        manifest=manifest,
        partition_paths=partition_paths,
        row_count=labels_frame.height,
        base_panel_hash=base_panel_hash,
    )


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    raise ValueError(f"expected a date or datetime timestamp, got {value!r}")
