"""Snapshot economic provenance contract tests.

Scenarios: PROVENANCE_GATE_01.
"""
from __future__ import annotations

import polars as pl

from src.stocks.ml.contracts import NetAlphaResearchData
from src.stocks.ml.data import (
    assess_snapshot_outcome_readiness,
    snapshot_economic_provenance,
)
from src.stocks.ml.labels import ID_COLUMN


def _make_research_data(
    *,
    status_provenance: str = "pinned",
    has_evidence: bool = True,
    evidence_hash: str | None = "abc123",
) -> NetAlphaResearchData:
    """Build a minimal NetAlphaResearchData for provenance testing."""
    feature_frame = pl.DataFrame({
        ID_COLUMN: ["KRX:00001", "KRX:00002", "KRX:00003"],
        "session": [
            __import__("datetime").datetime(2024, 1, 2, tzinfo=__import__("datetime").UTC),
            __import__("datetime").datetime(2024, 1, 3, tzinfo=__import__("datetime").UTC),
            __import__("datetime").datetime(2024, 1, 4, tzinfo=__import__("datetime").UTC),
        ],
        "feature_a": [1.0, 2.0, 3.0],
    })
    label_frame = pl.DataFrame({
        ID_COLUMN: ["KRX:00001", "KRX:00002", "KRX:00003"],
        "session": [
            __import__("datetime").datetime(2024, 1, 2, tzinfo=__import__("datetime").UTC),
            __import__("datetime").datetime(2024, 1, 3, tzinfo=__import__("datetime").UTC),
            __import__("datetime").datetime(2024, 1, 4, tzinfo=__import__("datetime").UTC),
        ],
        "net_alpha_target": [0.01, 0.02, 0.03],
        "label_available_time": [
            __import__("datetime").datetime(2024, 1, 3, tzinfo=__import__("datetime").UTC),
            __import__("datetime").datetime(2024, 1, 4, tzinfo=__import__("datetime").UTC),
            __import__("datetime").datetime(2024, 1, 5, tzinfo=__import__("datetime").UTC),
        ],
    })
    status_frame = pl.DataFrame({
        ID_COLUMN: ["KRX:00001", "KRX:00002", "KRX:00003"],
        "session": [
            __import__("datetime").datetime(2024, 1, 2, tzinfo=__import__("datetime").UTC),
            __import__("datetime").datetime(2024, 1, 3, tzinfo=__import__("datetime").UTC),
            __import__("datetime").datetime(2024, 1, 4, tzinfo=__import__("datetime").UTC),
        ],
        "outcome_status": ["REALIZED", "REALIZED", "REALIZED"],
    })
    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind

    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="h",
        provider_version="p",
        universe_policy_version="u",
        universe_policy_hash="u",
        feature_set="stock_net_alpha_v1",
        feature_set_hash="f",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=3,
        time_start=__import__("datetime").datetime(2024, 1, 1, tzinfo=__import__("datetime").UTC),
        time_end=__import__("datetime").datetime(2024, 1, 5, tzinfo=__import__("datetime").UTC),
        generated_time=__import__("datetime").datetime(2024, 1, 5, tzinfo=__import__("datetime").UTC),
        row_count=3,
    )

    evidence_by_horizon: dict[int, pl.DataFrame] = {}
    if has_evidence:
        evidence_by_horizon[3] = pl.DataFrame({
            ID_COLUMN: ["KRX:00001", "KRX:00002", "KRX:00003"],
            "session": [
                __import__("datetime").datetime(2024, 1, 2, tzinfo=__import__("datetime").UTC),
                __import__("datetime").datetime(2024, 1, 3, tzinfo=__import__("datetime").UTC),
                __import__("datetime").datetime(2024, 1, 4, tzinfo=__import__("datetime").UTC),
            ],
            "horizon_sessions": [3, 3, 3],
            "policy_hash": ["policy_abc", "policy_abc", "policy_abc"],
            "resolution_kind": ["SCHEDULED_OPEN", "SCHEDULED_OPEN", "SCHEDULED_OPEN"],
            "outcome_status": ["REALIZED", "REALIZED", "REALIZED"],
            "scheduled_entry_session": [
                __import__("datetime").datetime(2024, 1, 2, tzinfo=__import__("datetime").UTC),
                __import__("datetime").datetime(2024, 1, 3, tzinfo=__import__("datetime").UTC),
                __import__("datetime").datetime(2024, 1, 4, tzinfo=__import__("datetime").UTC),
            ],
            "scheduled_exit_session": [
                __import__("datetime").datetime(2024, 1, 5, tzinfo=__import__("datetime").UTC),
                __import__("datetime").datetime(2024, 1, 6, tzinfo=__import__("datetime").UTC),
                __import__("datetime").datetime(2024, 1, 7, tzinfo=__import__("datetime").UTC),
            ],
            "entry_disposition": ["SCHEDULED_OPEN", "SCHEDULED_OPEN", "SCHEDULED_OPEN"],
            "exit_disposition": ["SCHEDULED_CLOSE", "SCHEDULED_CLOSE", "SCHEDULED_CLOSE"],
        })

    return NetAlphaResearchData(
        feature_frame=feature_frame,
        labels_by_horizon={3: label_frame},
        manifest=manifest,
        status_by_horizon={3: status_frame},
        evidence_by_horizon=evidence_by_horizon,
        status_provenance=status_provenance,
    )


class TestProvenanceGate:
    """PROVENANCE_GATE_01."""

    def test_status_only_returns_unpinned_before_fitting(self) -> None:
        """Status-only data returns outcome-evidence-unpinned."""
        data = _make_research_data(status_provenance="legacy-inferred", has_evidence=False)
        provenance = snapshot_economic_provenance(data, (3,))
        assert provenance.status_hash is None
        assert provenance.evidence_hash is None
        assert provenance.reason == "outcome-provenance-unpinned"

    def test_complete_matching_pair_returns_pinned(self) -> None:
        """Complete status/evidence pair returns pinned provenance."""
        data = _make_research_data(
            status_provenance="pinned", has_evidence=True, evidence_hash="abc123"
        )
        provenance = snapshot_economic_provenance(data, (3,))
        assert provenance.status_hash is not None
        assert provenance.evidence_hash is not None
        assert provenance.reason is None

    def test_readiness_passes_with_pinned_provenance(self) -> None:
        """A pinned provenance passes the readiness gate."""
        data = _make_research_data(
            status_provenance="pinned", has_evidence=True
        )
        readiness = assess_snapshot_outcome_readiness(data, (3,))
        assert readiness.passed
