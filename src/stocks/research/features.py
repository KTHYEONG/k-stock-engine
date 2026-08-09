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
        location = (
            (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))
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
