"""PLAN-02-POINT-IN-TIME-DATASET: Point-in-time dataset contract."""
from __future__ import annotations

from datetime import datetime, UTC

import polars as pl
import pytest

from src.core.instruments import AssetKind
from src.core.time import TemporalViolationError
from src.core.datasets import make_manifest, schema_hash, validate_dataset_manifest
from src.stocks.research.datasets import validate_stock_rows_available
from tests.fixtures.stocks.helpers import stock_instrument_df, stock_manifest


class TestDatasetContract:
    def test_valid_manifest_validation_passes(self) -> None:
        decision_time = datetime(2024, 3, 15, 8, 50, tzinfo=UTC)
        manifest = stock_manifest(decision_time=decision_time)
        assert (
            validate_dataset_manifest(
                manifest, AssetKind.STOCK, "stock_alpha_v1", decision_time
            )
            is None
        )

    def test_kind_mismatch_is_rejected(self) -> None:
        manifest = stock_manifest(asset_kind=AssetKind.ETF)
        with pytest.raises(ValueError, match="asset_kind"):
            validate_dataset_manifest(manifest, AssetKind.STOCK, "stock_alpha_v1", datetime.now(UTC))

    def test_feature_set_mismatch_is_rejected(self) -> None:
        manifest = stock_manifest(feature_set="other_v2")
        with pytest.raises(ValueError, match="feature_set"):
            validate_dataset_manifest(manifest, AssetKind.STOCK, "stock_alpha_v1", datetime.now(UTC))

    def test_unavailable_dataset_is_rejected(self) -> None:
        # dataset generated after decision_time -> not available yet
        decision_time = datetime(2024, 2, 1, 8, 0, tzinfo=UTC)
        manifest = stock_manifest(decision_time=datetime(2024, 3, 1, tzinfo=UTC))
        with pytest.raises(ValueError, match="not available"):
            validate_dataset_manifest(manifest, AssetKind.STOCK, "stock_alpha_v1", decision_time)

    def test_row_available_after_decision_is_rejected(self) -> None:
        df = stock_instrument_df()
        decision_time = datetime(2024, 1, 5, 8, 0, tzinfo=UTC)
        assert df["available_time"].max() > decision_time
        with pytest.raises(TemporalViolationError):
            validate_stock_rows_available(df, decision_time)

    def test_duplicate_instrument_session_is_rejected(self) -> None:
        df = stock_instrument_df(n_sessions=5, n_tickers=1)
        dup = pl.concat([df, df.filter(pl.col("session_index") == 2)])
        with pytest.raises(ValueError, match="duplicate"):
            validate_stock_rows_available(
                dup, datetime(2024, 1, 20, tzinfo=UTC)
            )

    def test_valid_panel_passes_validation(self) -> None:
        df = stock_instrument_df(n_sessions=5, n_tickers=2)
        assert (
            validate_stock_rows_available(df, df["available_time"].max())
            is None
        )

    def test_valid_stock_row_keeps_schema_and_universe_fingerprints(self) -> None:
        df = stock_instrument_df(n_sessions=5, n_tickers=2)
        columns = df.columns
        manifest = make_manifest(
            asset_kind=AssetKind.STOCK,
            columns=columns,
            feature_set="stock_alpha_v1",
            label_definition="fwd_ret_5d",
            label_horizon_sessions=5,
            time_start=datetime(2024, 1, 1, tzinfo=UTC),
            time_end=datetime(2024, 1, 5, tzinfo=UTC),
            provider_version="fixture",
            universe_policy_version="v1",
            row_count=df.height,
        )
        assert manifest.schema_hash == schema_hash(columns)
        assert manifest.asset_kind is AssetKind.STOCK
        assert manifest.universe_policy_hash  # non-empty fingerprint
        assert manifest.feature_set_hash  # non-empty fingerprint
