"""Canonical net-alpha materialization entry point.

This module owns the canonical NetAlphaMaterializationRequest/Result and
materialize_net_alpha_snapshot. The legacy research_v2 module re-exports these
identities for compatibility; no subclass adapter is permitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.core.datasets import DatasetCertification
from src.stocks.data.contracts import ResearchWindows
from src.stocks.domain.execution_policy import ExecutionOutcomePolicy
from src.stocks.ml.contracts import DEFAULT_CANDIDATE_HORIZON_SESSIONS


@dataclass(frozen=True, slots=True)
class NetAlphaMaterializationRequest:
    """Explicit, non-empty inputs for one net-alpha snapshot materialization."""

    source_snapshot_id: str
    feature_dataset_id: str
    label_dataset_id: str
    snapshot_id: str
    catalog_root: Path
    base_root: Path
    feature_root: Path
    label_root: Path
    generated_time: datetime
    windows: ResearchWindows
    certification: DatasetCertification = DatasetCertification.PROVISIONAL
    min_coverage: float = 0.75
    calendar_path: Path | None = None
    candidate_horizon_sessions: tuple[int, ...] = DEFAULT_CANDIDATE_HORIZON_SESSIONS
    reference_notional: float = 100_000_000.0
    policy: ExecutionOutcomePolicy | None = None
    raw_bar_dataset_id: str | None = None
    outcome_open_bar_dataset_id: str | None = None
    tradability_events_dataset_id: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "source_snapshot_id",
            "feature_dataset_id",
            "label_dataset_id",
            "snapshot_id",
        ):
            if not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be within (0, 1]")
        if not self.candidate_horizon_sessions:
            raise ValueError("candidate_horizon_sessions must be non-empty")
        if tuple(self.candidate_horizon_sessions) != tuple(
            sorted(set(self.candidate_horizon_sessions))
        ):
            raise ValueError(
                "candidate_horizon_sessions must be strictly ascending and unique"
            )
        if any(h < 1 for h in self.candidate_horizon_sessions):
            raise ValueError("candidate_horizon_sessions must be positive sessions")
        if self.reference_notional <= 0:
            raise ValueError("reference_notional must be positive")
        if self.raw_bar_dataset_id is not None and not self.raw_bar_dataset_id:
            raise ValueError("raw_bar_dataset_id must be non-empty when supplied")
        if self.outcome_open_bar_dataset_id is not None and not self.outcome_open_bar_dataset_id:
            raise ValueError("outcome_open_bar_dataset_id must be non-empty when supplied")
        if self.tradability_events_dataset_id is not None and not self.tradability_events_dataset_id:
            raise ValueError("tradability_events_dataset_id must be non-empty when supplied")
        if self.raw_bar_dataset_id is not None and self.outcome_open_bar_dataset_id is not None:
            raise ValueError("supply either raw_bar_dataset_id or outcome_open_bar_dataset_id")


@dataclass(frozen=True, slots=True)
class NetAlphaMaterializationResult:
    """Immutable outcome of one net-alpha snapshot materialization."""

    snapshot_id: str
    feature_dataset_id: str
    label_dataset_id: str
    feature_content_hash: str
    label_content_hash: str
    feature_row_count: int
    label_row_count: int
    min_coverage: float
    certification: DatasetCertification
    policy_id: str = "scheduled_open_v1"


def materialize_net_alpha_snapshot(
    request: NetAlphaMaterializationRequest,
) -> NetAlphaMaterializationResult:
    """Materialize the canonical net-alpha snapshot via the shared implementation."""
    # Delegate to the heavy implementation in research_v2 without creating a subclass.
    from src.stocks.data.research_v2 import _materialize_net_alpha_snapshot_impl

    return _materialize_net_alpha_snapshot_impl(request)  # type: ignore[arg-type,return-value]


__all__ = [
    "NetAlphaMaterializationRequest",
    "NetAlphaMaterializationResult",
    "materialize_net_alpha_snapshot",
]
