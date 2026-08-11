"""Versioned, dependency-declared stock feature definitions.

All renderers operate on a panel already sorted by ``instrument_id`` and
``session`` and scope every lag or rolling window over ``instrument_id`` so a
multi-stock panel can never mix instruments. Raw values keep raw names;
cross-sectional ranks (computed later by the composite model) use distinct
names.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import polars as pl

ID_COLUMN = "instrument_id"
SESSION_COLUMN = "session"
_PRICE_COLUMNS = ("open", "high", "low", "close")
_TARGET_PREFIXES = ("target_", "label_")


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """A single versioned feature with declared input dependencies."""

    name: str
    version: int
    inputs: tuple[str, ...] = ()
    description: str = ""

    def render(self, frame: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError(f"{self.name} has no renderer; subclass it")

    @property
    def fingerprint(self) -> str:
        return sha256(
            f"{self.name}@{self.version}:{','.join(self.inputs)}".encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class MomentumFeature(FeatureDefinition):
    """(close - close_shift_n) / close_shift_n momentum feature."""

    lookback: int = 5

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

    def render(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._require_inputs(frame)
        shifted = pl.col("close").shift(self.lookback).over(ID_COLUMN)
        return frame.with_columns(((pl.col("close") - shifted) / shifted).alias(self.name))

    def _require_inputs(self, frame: pl.DataFrame) -> None:
        missing = [c for c in self.inputs if c not in frame.columns]
        if missing:
            raise ValueError(f"feature {self.name} missing inputs {missing}")


@dataclass(frozen=True, slots=True)
class ReversalFeature(FeatureDefinition):
    """Five-session reversal: raw five-session log return.

    The composite model owns orientation; the raw factor keeps its raw units.
    """

    lookback: int = 5

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

    def render(self, frame: pl.DataFrame) -> pl.DataFrame:
        if "close" not in frame.columns:
            raise ValueError("missing close column")
        shifted = pl.col("close").shift(self.lookback).over(ID_COLUMN)
        return frame.with_columns((pl.col("close") / shifted).log().alias(self.name))


@dataclass(frozen=True, slots=True)
class TrendFeature(FeatureDefinition):
    """20-to-120-session trend: close[T-short] / close[T-long] - 1."""

    short_lookback: int = 20
    long_lookback: int = 120

    def __post_init__(self) -> None:
        if self.short_lookback <= 0 or self.long_lookback <= 0:
            raise ValueError("lookbacks must be positive")
        if self.short_lookback >= self.long_lookback:
            raise ValueError("short_lookback must be less than long_lookback")

    def render(self, frame: pl.DataFrame) -> pl.DataFrame:
        if "close" not in frame.columns:
            raise ValueError("missing close column")
        short = pl.col("close").shift(self.short_lookback).over(ID_COLUMN)
        long = pl.col("close").shift(self.long_lookback).over(ID_COLUMN)
        return frame.with_columns((short / long - 1.0).alias(self.name))


@dataclass(frozen=True, slots=True)
class VolatilityFeature(FeatureDefinition):
    """Realized volatility of daily log returns over a session window."""

    lookback: int = 20

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")

    def render(self, frame: pl.DataFrame) -> pl.DataFrame:
        if "close" not in frame.columns:
            raise ValueError("missing close column")
        log_return = (pl.col("close").log() - pl.col("close").log().shift(1)).over(ID_COLUMN)
        return frame.with_columns(
            log_return.rolling_std(window_size=self.lookback, min_samples=2).alias(self.name)
        )


@dataclass(frozen=True, slots=True)
class CloseLocationFeature(FeatureDefinition):
    """Close-location strength: mean (close - low) / (high - low) over a window."""

    lookback: int = 20

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

    def render(self, frame: pl.DataFrame) -> pl.DataFrame:
        for col in ("high", "low", "close"):
            if col not in frame.columns:
                raise ValueError(f"missing {col} column")
        price_range = pl.col("high") - pl.col("low")
        location = (
            pl.when(price_range != 0)
            .then((pl.col("close") - pl.col("low")) / price_range)
            .otherwise(pl.lit(0.5))
        ).over(ID_COLUMN)
        return frame.with_columns(
            location.rolling_mean(window_size=self.lookback, min_samples=1).alias(self.name)
        )


@dataclass(frozen=True, slots=True)
class LogMarketCapFeature(FeatureDefinition):
    """Log market capitalization: a capacity/risk control, not a free alpha claim."""

    def render(self, frame: pl.DataFrame) -> pl.DataFrame:
        if "market_cap" not in frame.columns:
            raise ValueError("missing market_cap column")
        return frame.with_columns(pl.col("market_cap").log().alias(self.name))


def feature_set_fingerprint(features: list[FeatureDefinition]) -> str:
    """Deterministic fingerprint of an ordered feature set definition."""
    return sha256(
        "\n".join(f.fingerprint for f in features).encode("utf-8")
    ).hexdigest()


STOCK_ALPHA_V2_ALLOWLIST: tuple[str, ...] = (
    "adtv_20d",
    "amihud_20d",
    "bp_ratio",
    "close_high_ratio_10d",
    "disparity_120d",
    "ep_ratio",
    "flow_consensus",
    "flow_intensity_20d",
    "fluc_rate",
    "foreign_net_buy",
    "individual_net_buy",
    "info_ratio_20d",
    "institution_net_buy",
    "intraday_ret",
    "mcap_rank",
    "min_vol_5d",
    "net_purchase_total",
    "overnight_ret",
    "pbr",
    "per",
    "relative_trend_score",
    "ret_21_60d",
    "ret_2_5d",
    "ret_6_20d",
    "sector_ret_5d",
    "trend_120d_rank",
    "turnover_ratio",
    "vol_20d_rank",
    "vol_asymmetry_20d",
    "vol_regime",
    "volatility_20d",
    "volatility_60d",
    "volume_shock",
    "vpt_20d",
)

STOCK_ALPHA_V2_FEATURE_SET = "stock_alpha_v2"


def stock_alpha_v2_allowlist() -> tuple[str, ...]:
    """Frozen v2 source allowlist: 34 empirically ready columns in manifest order.

    ``vix_zscore_20d`` is excluded: a US-close value keyed to the same KRX
    calendar date may not have existed at the KRX decision time. Statement
    totals and profits are excluded until complete DART facts are normalized.
    """
    return STOCK_ALPHA_V2_ALLOWLIST


def v2_feature_columns(
    frame: pl.DataFrame,
    *,
    source_prefix: str = "feature__",
    allowlist: tuple[str, ...] = STOCK_ALPHA_V2_ALLOWLIST,
) -> tuple[str, ...]:
    """Ordered ``source_prefix``-qualified v2 columns present in ``frame``.

    The result mirrors the frozen allowlist order so LightGBM always consumes
    an identical manifest-ordered input contract regardless of panel layout.
    """
    return tuple(
        f"{source_prefix}{name}"
        for name in allowlist
        if f"{source_prefix}{name}" in frame.columns
    )


def fit_v2_winsor_quantiles(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    low: float = 0.01,
    high: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Fit per-feature 1%/99% clip thresholds on training rows only.

    A single vectorized Polars aggregation computes all train-only quantiles;
    the linear interpolation matches the historical ``np.quantile`` contract.
    Null-only columns still return ``(0.0, 0.0)``; non-finite values are
    rejected fail-closed by the calling transform validation.
    """
    quantiles: dict[str, tuple[float, float]] = {}
    if not feature_columns:
        return quantiles
    exprs: list[pl.Expr] = []
    for index, column in enumerate(feature_columns):
        source = pl.col(column).cast(pl.Float64)
        exprs.append(source.quantile(low, interpolation="linear").alias(f"__qlo_{index}"))
        exprs.append(source.quantile(high, interpolation="linear").alias(f"__qhi_{index}"))
    values = frame.select(exprs).row(0)
    for index, column in enumerate(feature_columns):
        lo = values[2 * index]
        hi = values[2 * index + 1]
        quantiles[column] = (
            (0.0, 0.0) if lo is None and hi is None else (float(lo), float(hi))
        )
    return quantiles


def apply_v2_transforms(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    winsor_quantiles: dict[str, tuple[float, float]] | None = None,
    session_column: str = SESSION_COLUMN,
    sector_column: str = "sector",
    low: float = 0.01,
    high: float = 0.99,
) -> pl.DataFrame:
    """Build the v2 predictor frame: raw, clipped rank, sector-demeaned rank.

    Each allowlisted feature contributes three float32 predictors: the raw
    value (null retained for LightGBM), its per-session clipped percentile
    rank, and its per-session percentile rank demeaned by the session-sector
    mean. NaN/Inf is rejected; null values are retained for LightGBM.
    """
    _reject_target_columns(frame, feature_columns)
    quantiles = winsor_quantiles or fit_v2_winsor_quantiles(
        frame, feature_columns, low=low, high=high
    )
    missing_quantile = [c for c in feature_columns if c not in quantiles]
    if missing_quantile:
        raise ValueError(f"missing winsor quantiles for {missing_quantile}")
    if sector_column not in frame.columns:
        raise ValueError(f"frame must carry {sector_column!r} for sector-demeaned rank")
    for column in feature_columns:
        non_finite = frame.filter(
            pl.col(column).is_not_null() & ~pl.col(column).is_finite()
        )
        if not non_finite.is_empty():
            raise ValueError(f"non-finite value in feature column {column}")

    rank_exprs: list[pl.Expr] = []
    for column in feature_columns:
        lo, hi = quantiles[column]
        clipped = pl.col(column).clip(lo, hi)
        within = pl.col(column).count().over(session_column)
        rank_exprs.append(

                ((clipped.rank("average").over(session_column) - 1.0) / (within - 1.0))
                .fill_null(0.5)
                .cast(pl.Float32)
                .alias(f"__rank_{column}")

        )
    ranked = frame.with_columns(rank_exprs)

    sector_exprs: list[pl.Expr] = []
    for column in feature_columns:
        sector_mean = (
            pl.col(f"__rank_{column}").mean().over([session_column, sector_column])
        )
        sector_exprs.append(
            ((pl.col(f"__rank_{column}") - sector_mean).cast(pl.Float32)).alias(
                f"__sector_rank_{column}"
            )
        )
    expanded = ranked.with_columns(sector_exprs)

    out_exprs: list[pl.Expr] = []
    for column in expanded.columns:
        if column.startswith(("__rank_", "__sector_rank_")):
            continue
        if column in feature_columns:
            continue
        out_exprs.append(pl.col(column))
    for column in feature_columns:
        out_exprs.extend(
            [
                pl.col(column).cast(pl.Float32).alias(column),
                pl.col(f"__rank_{column}").alias(f"{column}__rank"),
                pl.col(f"__sector_rank_{column}").alias(f"{column}__sector_rank"),
            ]
        )
    return expanded.select(out_exprs)


def v2_missing_rates(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    coverage_threshold: float = 0.75,
    missing_flag_threshold: float = 0.01,
) -> dict[str, float]:
    """Per-feature training-window null rates and coverage exclusions.

    A non-null ratio below ``coverage_threshold`` excludes the feature;
    a missing rate above ``missing_flag_threshold`` sets its missing flag.
    """
    rates: dict[str, float] = {}
    for column in feature_columns:
        non_null = frame[column].is_not_null().sum()
        rates[column] = 1.0 - (float(non_null) / frame.height if frame.height else 1.0)
    return rates


def _reject_target_columns(frame: pl.DataFrame, feature_columns: tuple[str, ...]) -> None:
    del frame
    offending = [c for c in feature_columns if c.startswith(_TARGET_PREFIXES)]
    if offending:
        raise ValueError(f"v2 predictor rejects target/label columns: {offending}")


def phase1_allowlist() -> list[FeatureDefinition]:
    """Phase-1 auditable predictor allowlist shared by training and scoring."""
    return [
        ReversalFeature(name="rev_5d", version=1, inputs=("close",)),
        TrendFeature(name="trend_20_120", version=1, inputs=("close",)),
        VolatilityFeature(name="vol_20d", version=1, inputs=("close",)),
        CloseLocationFeature(name="closeloc_20d", version=1, inputs=("high", "low", "close")),
        LogMarketCapFeature(name="ln_mktcap", version=1, inputs=("market_cap",)),
    ]


def build_features(
    frame: pl.DataFrame,
    features: list[FeatureDefinition],
    id_column: str = ID_COLUMN,
    session_column: str = SESSION_COLUMN,
) -> pl.DataFrame:
    """Sort the panel by instrument and session, then render every feature.

    Determinism is invariant to input row order: the shuffled and sorted panels
    must produce identical feature columns after sorting.
    """
    if id_column not in frame.columns or session_column not in frame.columns:
        raise ValueError(f"frame must carry {id_column!r} and {session_column!r}")
    for feature in features:
        missing = [c for c in feature.inputs if c not in frame.columns]
        if missing:
            raise ValueError(f"feature {feature.name} missing declared inputs {missing}")
    for price_column in _PRICE_COLUMNS:
        if price_column in frame.columns and frame.filter(
            pl.col(price_column) <= 0
        ).height > 0:
            raise ValueError(f"{price_column} must be positive")
    out = frame.sort([id_column, session_column])
    for feature in features:
        out = feature.render(out)
    return out
