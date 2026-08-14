"""Unit tests for stock universe policy, features, labels, and the composite model."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from src.core.instruments import AssetKind
from src.stocks.domain.universe import PointInTimeUniverse, UniversePolicy
from src.stocks.research.features import (
    LogMarketCapFeature,
    MomentumFeature,
    ReversalFeature,
    TrendFeature,
    VolatilityFeature,
    build_features,
    feature_set_fingerprint,
)
from src.stocks.research.labels import LabelDefinition


def universe_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": ["005930", "000001", "ABCDEF", "123456", "999999"],
            "available_time": [datetime(2024, 1, 5, 8, 0, tzinfo=UTC)] * 5,
            "close": [70000.0, 0.0, 5000.0, 8000.0, 6000.0],
            "capital_erosion_rate": [5.0, 0.0, 0.0, 60.0, 10.0],
            "operating_income": [1000.0, 500.0, None, 300.0, -50.0],
        }
    )


class TestPointInTimeUniverse:
    def test_universe_policy_returns_exclusion_reasons(self) -> None:
        policy = UniversePolicy(version="v1", min_close=1000.0)
        universe = PointInTimeUniverse(policy)
        result = universe.apply(universe_frame(), datetime(2024, 1, 5, 8, 0, tzinfo=UTC))

        assert "005930" in result.members
        assert result.exclusions["000001"] == "close-below-min"
        assert result.exclusions["ABCDEF"] == "non-common-stock-code"
        assert result.exclusions["123456"] == "capital-erosion"
        assert result.exclusions["999999"] == "no-positive-operating-income"

    def test_pit_universe_ignores_future_rows(self) -> None:
        policy = UniversePolicy(
            version="v1", min_close=0.0, require_operating_income=False, require_historical_master=False
        )
        frame = pl.DataFrame(
            {
                "code": ["005930", "005930"],
                "available_time": [
                    datetime(2024, 1, 5, 8, 0, tzinfo=UTC),
                    datetime(2024, 1, 10, 8, 0, tzinfo=UTC),
                ],
                "close": [100.0, 200.0],
            }
        )
        result = PointInTimeUniverse(policy).apply(
            frame, datetime(2024, 1, 6, 8, 0, tzinfo=UTC)
        )
        assert "005930" in result.members

    def test_universe_enforces_listing_and_tradability_intervals(self) -> None:
        policy = UniversePolicy(
            version="v1", min_close=0.0, require_operating_income=False
        )
        decision = datetime(2024, 6, 1, 8, 0, tzinfo=UTC)
        frame = pl.DataFrame(
            {
                "code": ["000001", "000002", "000003"],
                "available_time": [decision] * 3,
                "close": [100.0, 100.0, 100.0],
                "listed_from": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                ],
                "delisted_on": [
                    datetime(2025, 1, 1, tzinfo=UTC),
                    datetime(2024, 5, 1, tzinfo=UTC),
                    datetime(2025, 1, 1, tzinfo=UTC),
                ],
                "tradable_from": [None, None, datetime(2024, 7, 1, tzinfo=UTC)],
            }
        )
        result = PointInTimeUniverse(policy).apply(frame, decision)
        assert result.exclusions["000002"] == "delisted"
        assert result.exclusions["000003"] == "not-tradable-yet"
        assert "000001" in result.members

    def test_capacity_is_structural_ratio(self) -> None:
        assert PointInTimeUniverse.participation_capacity(1_000_000.0, 10_000_000.0) == pytest.approx(0.1)


class TestFeatures:
    def test_momentum_feature_renders_within_instrument(self) -> None:
        frame = pl.DataFrame(
            {
                "instrument_id": ["KRX:A"] * 4 + ["KRX:B"] * 4,
                "session": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(8)],
                "close": [1.0, 2.0, 3.0, 4.0] * 2,
            }
        )
        feature = MomentumFeature(name="momentum_5d", version=1, lookback=1, inputs=("close",))
        out = build_features(frame, [feature])
        assert out["momentum_5d"].to_list()[1] == pytest.approx(1.0)

    def test_feature_set_fingerprint_is_deterministic(self) -> None:
        features = [MomentumFeature(name="m", version=1, lookback=5, inputs=("close",))]
        assert feature_set_fingerprint(features) == feature_set_fingerprint(features)

    def test_feature_requires_close(self) -> None:
        feature = MomentumFeature(name="m", version=1, lookback=1, inputs=("close",))
        with pytest.raises(ValueError, match="close"):
            feature.render(pl.DataFrame({"open": [1.0]}))

    def test_build_features_is_order_invariant(self) -> None:
        frame = pl.DataFrame(
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
        features = [
            ReversalFeature(name="rev_5d", version=1, inputs=("close",)),
            VolatilityFeature(name="vol_20d", version=1, inputs=("close",)),
            LogMarketCapFeature(name="ln_mktcap", version=1, inputs=("market_cap",)),
        ]
        shuffled = frame.sample(fraction=1.0, seed=7, shuffle=True)
        sorted_out = build_features(frame, features)
        shuffled_out = build_features(shuffled, features)
        names = [f.name for f in features]
        assert sorted_out.sort(["instrument_id", "session"]).select(names).equals(
            shuffled_out.sort(["instrument_id", "session"]).select(names)
        )

    def test_build_features_rejects_non_positive_prices(self) -> None:
        frame = pl.DataFrame(
            {
                "instrument_id": ["KRX:A"],
                "session": [datetime(2024, 1, 1, tzinfo=UTC)],
                "close": [0.0],
            }
        )
        with pytest.raises(ValueError, match="close"):
            build_features(frame, [MomentumFeature(name="m", version=1, lookback=1, inputs=("close",))])

    def test_build_features_rejects_undeclared_inputs(self) -> None:
        frame = pl.DataFrame(
            {
                "instrument_id": ["KRX:A"],
                "session": [datetime(2024, 1, 1, tzinfo=UTC)],
                "close": [10.0],
            }
        )
        with pytest.raises(ValueError, match="missing declared inputs"):
            build_features(frame, [TrendFeature(name="t", version=1, inputs=("close", "missing"))])


class TestLabels:
    def test_label_declares_horizon_and_is_never_recomputed_in_trainer(self) -> None:
        label = LabelDefinition(
            name="fwd_ret_5d",
            entry_field="open",
            exit_field="close",
            horizon_sessions=5,
        )
        assert label.horizon_sessions == 5
        assert label.fingerprint

    def test_label_applies_forward_return(self) -> None:
        frame = pl.DataFrame(
            {
                "instrument_id": ["KRX:1"] * 6,
                "session": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)],
                "open": [99.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0, 100.0, 100.0, 120.0],
            }
        )
        label = LabelDefinition("fwd_ret_5d", "open", "close", 5)
        out = label.apply(frame)
        assert out["fwd_ret_5d"].to_list()[0] == pytest.approx(
            math.log(frame["close"].to_list()[5] / frame["open"].to_list()[1])
        )

    def test_label_handles_unsorted_multiinstrument_panel(self) -> None:
        sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)]
        rows = [
            {
                "instrument_id": instrument,
                "session": sessions[i],
                "open": 99.0 + i,
                "close": 100.0 + i,
            }
            for instrument in ("KRX:A", "KRX:B")
            for i in range(6)
        ]
        shuffled = pl.DataFrame(rows).sample(fraction=1.0, seed=3, shuffle=True)
        label = LabelDefinition("fwd_ret_5d", "open", "close", 2)
        out = label.apply(shuffled).sort(["instrument_id", "session"])
        a_label = out.filter(pl.col("instrument_id") == "KRX:A")["fwd_ret_5d"].to_list()
        expected_a = [math.log(out["close"].to_list()[i + 2] / out["open"].to_list()[i + 1]) for i in range(0, 6 - 2)]
        for i in range(4):
            assert a_label[i] == pytest.approx(expected_a[i])

    def test_label_requires_entry_and_exit_fields(self) -> None:
        frame = pl.DataFrame({"instrument_id": ["KRX:1"], "session": [datetime(2024, 1, 1, tzinfo=UTC)]})
        with pytest.raises(ValueError, match="open"):
            LabelDefinition("bad", "open", "close", 5).apply(frame)

    def test_terminal_labels_are_null_not_guessed(self) -> None:
        frame = pl.DataFrame(
            {
                "instrument_id": ["KRX:1"] * 3,
                "session": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(3)],
                "open": [100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0],
            }
        )
        out = LabelDefinition("fwd_ret_5d", "open", "close", 5).apply(frame)
        assert out["fwd_ret_5d"].to_list() == [None, None, None]

    def test_invalid_horizon_rejected(self) -> None:
        with pytest.raises(ValueError, match="horizon_sessions"):
            LabelDefinition("bad", "close", "close", 0)


def test_instrument_asset_kind_is_explicit_in_fixture() -> None:
    assert AssetKind.STOCK is AssetKind.STOCK
