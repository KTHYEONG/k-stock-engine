"""Versioned feature definition and build pipeline tests."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from legacy.stocks.research.features import (
    LogMarketCapFeature,
    ReversalFeature,
    TrendFeature,
    build_features,
    fit_v2_winsor_quantiles,
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


def test_vectorized_winsor_quantiles_match_numpy_contract() -> None:
    import numpy as np

    frame = pl.DataFrame(
        {
            "feature__a": [1.0, 2.0, 3.0, 4.0, None, 100.0, 0.5, None, 7.0, 8.0],
            "feature__b": [None] * 10,
            "feature__c": [10.0, 20.0, None, 40.0, 50.0, 60.0, 70.0, 80.0, None, 100.0],
        }
    )
    columns = ("feature__a", "feature__b", "feature__c")
    quantiles = fit_v2_winsor_quantiles(frame, columns)

    expected_a = (
        float(np.quantile(frame["feature__a"].drop_nulls().to_numpy(), 0.01)),
        float(np.quantile(frame["feature__a"].drop_nulls().to_numpy(), 0.99)),
    )
    assert quantiles["feature__a"][0] == pytest.approx(expected_a[0])
    assert quantiles["feature__a"][1] == pytest.approx(expected_a[1])
    assert quantiles["feature__b"] == (0.0, 0.0)
    expected_c = (
        float(np.quantile(frame["feature__c"].drop_nulls().to_numpy(), 0.01)),
        float(np.quantile(frame["feature__c"].drop_nulls().to_numpy(), 0.99)),
    )
    assert quantiles["feature__c"][0] == pytest.approx(expected_c[0])
    assert quantiles["feature__c"][1] == pytest.approx(expected_c[1])
    assert tuple(quantiles) == columns


def test_apply_v2_transforms_emits_rank_sector_rank_and_missing_indicators() -> None:
    from legacy.stocks.research.features import apply_v2_transforms

    frame = pl.DataFrame(
        {
            "session": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "sector": ["S1", "S1", "S2", "S1", "S1", "S2"],
            "feature__x": [1.0, 2.0, None, 3.0, None, 4.0],
            "feature__y": [0.5, None, 0.7, 0.9, 1.1, None],
        }
    )
    transformed = apply_v2_transforms(
        frame,
        ("feature__x", "feature__y"),
        winsor_quantiles={
            "feature__x": (0.0, 4.0),
            "feature__y": (0.0, 1.0),
        },
    )
    for name in (
        "feature__x__rank",
        "feature__x__sector_rank",
        "feature__x__missing",
        "feature__y__rank",
        "feature__y__sector_rank",
        "feature__y__missing",
    ):
        assert name in transformed.columns
    assert transformed["feature__x__missing"].to_list() == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert transformed["feature__y__missing"].to_list() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert transformed["feature__x__missing"].dtype == pl.Float32
    assert transformed["feature__x"].to_list() == [1.0, 2.0, None, 3.0, None, 4.0]
    rank_x = transformed["feature__x__rank"].to_list()
    assert rank_x[2] == pytest.approx(0.5)
    assert rank_x[4] == pytest.approx(0.5)
    assert transformed["feature__x__rank"].dtype == pl.Float32


def test_apply_v3_transforms_excludes_non_alpha_and_emits_rank_sector_rank() -> None:
    from legacy.stocks.research.features import apply_v3_transforms

    frame = pl.DataFrame(
        {
            "session": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "sector": ["S1", "S1", "S2", "S1", "S1", "S2"],
            "momentum": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "volatility": [0.2, 0.3, 0.1, 0.4, 0.5, 0.3],
            "adtv": [1.0e9, 2.0e9, 3.0e9, 4.0e9, 5.0e9, 6.0e9],
        }
    )
    roles = {"momentum": "ALPHA", "volatility": "RISK", "adtv": "LIQUIDITY"}
    transformed, learner_columns = apply_v3_transforms(frame, roles)
    # RISK/LIQUIDITY sources never enter the learner.
    assert "momentum__rank" in learner_columns
    assert "momentum__sector_rank" in learner_columns
    assert not any("volatility" in c or "adtv" in c for c in learner_columns)
    assert not any(c.startswith(("volatility", "adtv")) for c in transformed.columns)


def test_apply_v3_transforms_rejects_invalid_roles() -> None:
    from legacy.stocks.research.features import apply_v3_transforms

    frame = pl.DataFrame(
        {
            "session": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
            "sector": ["S1"] * 3,
            "momentum": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="one of"):
        apply_v3_transforms(frame, {"momentum": "SIGNAL"})


def test_apply_v3_transforms_missing_flag_only_for_mixed_sources() -> None:
    from legacy.stocks.research.features import apply_v3_transforms

    frame = pl.DataFrame(
        {
            "session": [datetime(2024, 1, 1, tzinfo=UTC)] * 6,
            "sector": ["S1"] * 6,
            "mixed": [1.0, None, 3.0, 0.5, 2.0, None],
            "complete": [3.0, 2.0, 1.0, 4.0, 0.5, 2.5],
        }
    )
    transformed, learner_columns = apply_v3_transforms(
        frame, {"mixed": "ALPHA", "complete": "ALPHA"}
    )
    assert "mixed__missing" in learner_columns
    assert "complete__missing" not in learner_columns
    assert transformed["mixed__missing"].to_list() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_apply_v3_transforms_clusters_exact_rank_equivalent_sources() -> None:
    from legacy.stocks.research.features import apply_v3_transforms

    frame = pl.DataFrame(
        {
            "session": [datetime(2024, 1, 1, tzinfo=UTC)] * 8
            + [datetime(2024, 1, 2, tzinfo=UTC)] * 8,
            "sector": ["S1"] * 16,
            "disparity": [float(i % 8) for i in range(16)],
            "trend_rank": [float(i % 8) for i in range(16)],
            "independent": [float((i * 7 + 3) % 11) for i in range(16)],
        }
    )
    transformed, learner_columns = apply_v3_transforms(
        frame,
        {"trend_rank": "ALPHA", "disparity": "ALPHA", "independent": "ALPHA"},
    )
    # disparity == trend_rank exactly, so only the lexicographically first
    # canonical source survives with its rank/sector-rank predictors.
    assert "disparity__rank" in learner_columns
    assert "trend_rank__rank" not in learner_columns
    assert "independent__rank" in learner_columns


def test_holdout_mutation_never_changes_schema_or_pre_holdout_transforms() -> None:
    """Holdout value/null mutations must not change schema, learner columns, or
    the pre-holdout transform outputs."""
    from legacy.stocks.ml.features import (
        apply_model_feature_schema,
        fit_model_feature_schema,
    )

    def _frame() -> pl.DataFrame:
        sessions = [datetime(2024, 1, 1, tzinfo=UTC)] * 6 + [
            datetime(2024, 1, 2, tzinfo=UTC)
        ] * 6
        sector = ["S1"] * 12
        momentum = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2
        missing_flow = [1.0, None, 3.0, 0.5, 2.0, None] * 2
        return pl.DataFrame(
            {
                "session": sessions,
                "sector": sector,
                "instrument_id": [f"KRX:{i}" for i in range(12)],
                "momentum": momentum,
                "flow": missing_flow,
            }
        )

    roles = {"momentum": "ALPHA", "flow": "ALPHA"}
    frame = _frame()
    pre_holdout = frame.filter(pl.col("session") < datetime(2024, 1, 2, tzinfo=UTC))
    holdout = frame.filter(pl.col("session") >= datetime(2024, 1, 2, tzinfo=UTC))
    assert not pre_holdout.is_empty()
    assert not holdout.is_empty()

    schema = fit_model_feature_schema(pre_holdout, roles)
    baseline_transformed = apply_model_feature_schema(pre_holdout, schema)

    # Mutate every holdout value and null pattern; schema and pre-holdout
    # transforms must be byte-identical.
    mutated_holdout = holdout.with_columns(
        pl.lit(123.0).alias("momentum"),
        pl.when(pl.col("flow").is_not_null()).then(None).otherwise(1.0).alias("flow"),
    )
    mutated_schema = fit_model_feature_schema(mutated_holdout, roles)
    assert mutated_schema.fingerprint == schema.fingerprint
    assert mutated_schema.learner_columns == schema.learner_columns
    assert mutated_schema.representative_sources == schema.representative_sources
    assert mutated_schema.missing_sources == schema.missing_sources

    mutated_transformed = apply_model_feature_schema(pre_holdout, mutated_schema)
    assert mutated_transformed.equals(baseline_transformed)
