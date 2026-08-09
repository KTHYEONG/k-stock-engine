"""Dataset certification and production fail-closed manifest tests."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from src.core.datasets import (
    DatasetCertification,
    DatasetManifest,
    make_manifest,
    validate_production_manifest,
)
from src.core.instruments import AssetKind


def base_manifest(**overrides: object) -> DatasetManifest:
    values = {
        "asset_kind": AssetKind.STOCK,
        "columns": ["session", "instrument_id", "close"],
        "feature_set": "stock_alpha_v1",
        "label_definition": "fwd_ret_5d",
        "label_horizon_sessions": 5,
        "time_start": datetime(2024, 1, 1, tzinfo=UTC),
        "time_end": datetime(2024, 3, 1, tzinfo=UTC),
        "provider_version": "fixture",
        "universe_policy_version": "fixture",
        "row_count": 100,
        "generated_time": datetime(2024, 1, 1, tzinfo=UTC),
    }
    values.update(overrides)
    return make_manifest(**values)


class TestDatasetCertification:
    def test_certification_tiers_are_explicit(self) -> None:
        assert DatasetCertification.PRODUCTION.value == "production"
        assert DatasetCertification.RESEARCH.value == "research"
        assert DatasetCertification.PROVISIONAL.value == "provisional"

    def test_default_manifest_is_provisional(self) -> None:
        manifest = base_manifest()
        assert manifest.certification is DatasetCertification.PROVISIONAL

    def test_manifest_carries_production_hashes(self) -> None:
        manifest = base_manifest(
            certification=DatasetCertification.PRODUCTION,
            calendar_hash="cal-1",
            corporate_action_hash="ca-1",
            cost_source_hash="cost-1",
        )
        assert manifest.calendar_hash == "cal-1"
        assert manifest.corporate_action_hash == "ca-1"
        assert manifest.cost_source_hash == "cost-1"


class TestValidateProductionManifest:
    def test_accepts_complete_production_manifest(self) -> None:
        manifest = base_manifest(
            certification=DatasetCertification.PRODUCTION,
            calendar_hash="cal-1",
            corporate_action_hash="ca-1",
            cost_source_hash="cost-1",
        )
        validate_production_manifest(manifest)

    def test_rejects_non_production_certification(self) -> None:
        for tier in (DatasetCertification.PROVISIONAL, DatasetCertification.RESEARCH):
            manifest = base_manifest(certification=tier)
            with pytest.raises(ValueError, match="PRODUCTION"):
                validate_production_manifest(manifest)

    def test_rejects_missing_calendar_hash(self) -> None:
        manifest = base_manifest(
            certification=DatasetCertification.PRODUCTION,
            calendar_hash="",
            corporate_action_hash="ca-1",
            cost_source_hash="cost-1",
        )
        with pytest.raises(ValueError, match="calendar_hash"):
            validate_production_manifest(manifest)

    def test_rejects_missing_corporate_action_hash(self) -> None:
        manifest = base_manifest(
            certification=DatasetCertification.PRODUCTION,
            calendar_hash="cal-1",
            corporate_action_hash="",
            cost_source_hash="cost-1",
        )
        with pytest.raises(ValueError, match="corporate_action_hash"):
            validate_production_manifest(manifest)

    def test_rejects_missing_cost_source_hash(self) -> None:
        manifest = base_manifest(
            certification=DatasetCertification.PRODUCTION,
            calendar_hash="cal-1",
            corporate_action_hash="ca-1",
            cost_source_hash="",
        )
        with pytest.raises(ValueError, match="cost_source_hash"):
            validate_production_manifest(manifest)

    def test_replace_round_trips_certification(self) -> None:
        manifest = base_manifest()
        upgraded = replace(
            manifest,
            certification=DatasetCertification.PRODUCTION,
            calendar_hash="cal-1",
            corporate_action_hash="ca-1",
            cost_source_hash="cost-1",
        )
        validate_production_manifest(upgraded)
        assert upgraded.certification is DatasetCertification.PRODUCTION
