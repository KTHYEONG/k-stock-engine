"""Versioned feature definition and build pipeline tests."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from src.stocks.research.features import (
    LogMarketCapFeature,
    ReversalFeature,
    TrendFeature,
    build_features,
    phase1_allowlist,
)


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["KRX:A"] * 6 + ["KRX:B"] * 6,
            "session": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)] * 2,
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0] * 2,
            "high": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0] * 2,
            "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0] * 2,
            "close": [10.0, 12.0, 11.0, 14.0, 13.0, 16.0] * 2,
            "market_cap": [1e12] * 12,
        }
    )


def test_phase1_allowlist_has_economically_distinct_factors() -> None:
    names = {f.name for f in phase1_allowlist()}
    assert {
        "rev_5d",
        "trend_20_120",
        "vol_20d",
        "closeloc_20d",
        "ln_mktcap",
    }.issubset(names)


def test_allowlist_features_render_deterministically() -> None:
    features = phase1_allowlist()
    out = build_features(_panel(), features)
    names = [f.name for f in features]
    shuffled = _panel().sample(fraction=1.0, seed=7, shuffle=True)
    assert out.sort(["instrument_id", "session"]).select(names).equals(
        build_features(shuffled, features).sort(["instrument_id", "session"]).select(names)
    )


def test_feature_raises_on_non_positive_price() -> None:
    frame = _panel().with_columns(pl.lit(0.0).alias("close"))
    with pytest.raises(ValueError, match="close"):
        build_features(frame, phase1_allowlist())


def test_close_location_uses_neutral_value_for_zero_range_sessions() -> None:
    frame = _panel().with_columns(
        pl.col("low").alias("high"),
        pl.col("low").alias("close"),
    )
    out = build_features(frame, phase1_allowlist())
    assert out["closeloc_20d"].to_list() == [0.5] * frame.height


def test_reversal_preserves_raw_units_under_raw_name() -> None:
    frame = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"] * 6,
            "session": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)],
            "close": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        }
    )
    out = build_features(frame, [ReversalFeature(name="rev_5d", version=1, inputs=("close",))])
    assert out["rev_5d"].to_list()[5] == pytest.approx(math.log(150.0 / 100.0))


def test_trend_requires_declared_long_lookback() -> None:
    frame = _panel().select("instrument_id", "session")
    with pytest.raises(ValueError, match="missing declared inputs"):
        build_features(frame, [TrendFeature(name="t", version=1, inputs=("close",))])


def test_log_market_cap_requires_market_cap() -> None:
    frame = _panel().drop("market_cap")
    with pytest.raises(ValueError, match="market_cap"):
        build_features(frame, [LogMarketCapFeature(name="ln_mktcap", version=1, inputs=("market_cap",))])
