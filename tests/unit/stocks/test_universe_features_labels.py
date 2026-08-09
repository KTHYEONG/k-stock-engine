"""Unit tests for stock universe policy, features, and labels."""
from __future__ import annotations

from datetime import datetime, UTC

import polars as pl
import pytest

from src.core.instruments import AssetKind
from src.stocks.domain.universe import PointInTimeUniverse, UniversePolicy
from src.stocks.features.definitions import MomentumFeature, build_features, feature_set_fingerprint
from src.stocks.labels.definitions import LabelDefinition


def universe_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": ["005930", "000001", "ABCDEF", "123456", "999999"],
            "date": [datetime(2024, 1, 5, tzinfo=UTC)] * 5,
            "close": [70000.0, 0.0, 5000.0, 8000.0, 6000.0],
            "capital_erosion_rate": [5.0, 0.0, 0.0, 60.0, 10.0],
            "operating_income": [1000.0, 500.0, None, 300.0, -50.0],
        }
    )


class TestPointInTimeUniverse:
    def test_universe_policy_returns_exclusion_reasons(self) -> None:
        policy = UniversePolicy(version="v1", min_close=1000.0)
        universe = PointInTimeUniverse(policy)
        result = universe.apply(universe_frame(), datetime(2024, 1, 5, tzinfo=UTC))

        assert "005930" in result.members
        assert result.exclusions["000001"] == "close-below-min"
        assert result.exclusions["ABCDEF"] == "non-common-stock-code"
        assert result.exclusions["123456"] == "capital-erosion"
        assert result.exclusions["999999"] == "no-positive-operating-income"

    def test_pit_universe_ignores_future_rows(self) -> None:
        policy = UniversePolicy(version="v1", min_close=0.0, require_operating_income=False)
        frame = pl.DataFrame(
            {
                "code": ["005930", "005930"],
                "date": [
                    datetime(2024, 1, 5, tzinfo=UTC),
                    datetime(2024, 1, 10, tzinfo=UTC),
                ],
                "close": [100.0, 200.0],
            }
        )
        result = PointInTimeUniverse(policy).apply(
            frame, datetime(2024, 1, 6, tzinfo=UTC)
        )
        assert "005930" in result.members


class TestFeatures:
    def test_momentum_feature_renders(self) -> None:
        frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
        feature = MomentumFeature(name="momentum_5d", version=1, lookback=1)
        out = build_features(frame, [feature])
        assert out["momentum_5d"].to_list()[1] == pytest.approx(1.0)

    def test_feature_set_fingerprint_is_deterministic(self) -> None:
        features = [MomentumFeature(name="m", version=1, lookback=5)]
        assert feature_set_fingerprint(features) == feature_set_fingerprint(features)

    def test_feature_requires_close(self) -> None:
        feature = MomentumFeature(name="m", version=1, lookback=1)
        with pytest.raises(ValueError, match="close"):
            feature.render(pl.DataFrame({"open": [1.0]}))


class TestLabels:
    def test_label_declares_horizon_and_is_never_recomputed_in_trainer(self) -> None:
        label = LabelDefinition(
            name="fwd_ret_5d",
            entry_field="close",
            exit_field="close",
            horizon_sessions=5,
        )
        assert label.horizon_sessions == 5
        assert label.fingerprint

    def test_label_applies_forward_return(self) -> None:
        frame = pl.DataFrame(
            {
                "instrument_id": ["KRX:1"] * 6,
                "close": [100.0, 100.0, 100.0, 100.0, 100.0, 120.0],
            }
        )
        label = LabelDefinition("fwd_ret_5d", "close", "close", 5)
        out = label.apply(frame)
        assert out["fwd_ret_5d"].to_list()[0] == pytest.approx(0.2)

    def test_invalid_horizon_rejected(self) -> None:
        with pytest.raises(ValueError, match="horizon_sessions"):
            LabelDefinition("bad", "close", "close", 0)


def test_instrument_asset_kind_is_explicit_in_fixture() -> None:
    assert AssetKind.STOCK is AssetKind.STOCK
