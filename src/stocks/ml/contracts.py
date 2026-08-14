"""Net-alpha ML contracts: request, research data, label specs, selection evidence.

Every contract is an immutable, typed input or evidence record. The request
carries only the fields the net-alpha mainline needs: candidate horizons are a
pre-registered discovery grid, the selected result has at most one primary and
one conditional secondary horizon, and there is no fixed 5/10/15 route.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from src.core.costs import CostSchedule
from src.core.datasets import DatasetManifest

DEFAULT_CANDIDATE_HORIZON_SESSIONS = (3, 5, 8, 10, 15, 20)
CANONICAL_FEATURE_SET = "stock_net_alpha_v1"


@dataclass(frozen=True, slots=True)
class PortfolioSettings:
    """Portfolio sizing and constraint settings shared by replay and scoring."""

    top_k: int = 20
    max_single_weight: float = 0.08
    max_exposure: float = 0.90
    participation_limit: float = 0.005
    portfolio_value: float = 100_000_000.0
    initial_cash: float = 100_000_000.0
    reference_notional: float = 100_000_000.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 < self.max_single_weight <= 1.0:
            raise ValueError("max_single_weight must be in (0, 1]")
        if not 0.0 < self.max_exposure <= 1.0:
            raise ValueError("max_exposure must be in (0, 1]")
        if self.participation_limit <= 0.0 or self.participation_limit >= 1.0:
            raise ValueError("participation_limit must be in (0, 1)")
        if self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.reference_notional <= 0:
            raise ValueError("reference_notional must be positive")


@dataclass(frozen=True, slots=True)
class RiskSettings:
    """Risk settings for the common policy replay."""

    calibration_bucket_count: int = 10
    min_calibration_sessions: int = 126
    risk_aversion: float = 2.0
    no_trade_band_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.calibration_bucket_count < 2:
            raise ValueError("calibration_bucket_count must be at least 2")
        if self.min_calibration_sessions < 1:
            raise ValueError("min_calibration_sessions must be positive")
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if self.no_trade_band_bps < 0:
            raise ValueError("no_trade_band_bps must be non-negative")


@dataclass(frozen=True, slots=True)
class NetAlphaTrainingRequest:
    """Input contract for the net-alpha training workflow.

    ``candidate_horizon_sessions`` is a pre-registered discovery grid, not an
    operating route: the trainer fits the baseline for every candidate and
    selects at most one primary and one conditional secondary horizon from OOF
    replay evidence. ``model_threads`` is the single thread budget for the
    challenger LightGBM (default 1); there is no Optuna trial, resume, or
    ``lgb_threads`` knob.
    """

    artifact_id: str
    candidate_horizon_sessions: tuple[int, ...] = DEFAULT_CANDIDATE_HORIZON_SESSIONS
    fold_count: int = 3
    embargo_sessions: int = 5
    forward_holdout_sessions: int = 0
    bootstrap_alpha: float = 0.05
    bootstrap_resamples: int = 200
    model_threads: int = 1
    max_rss_mib: int | None = None
    seed: int = 42
    portfolio: PortfolioSettings = field(default_factory=PortfolioSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    base_cost_schedule: CostSchedule | None = None
    stress_cost_schedule: CostSchedule | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        if not self.candidate_horizon_sessions:
            raise ValueError("candidate_horizon_sessions must be non-empty")
        if tuple(self.candidate_horizon_sessions) != tuple(
            sorted(set(self.candidate_horizon_sessions))
        ):
            raise ValueError("candidate_horizon_sessions must be strictly ascending and unique")
        if any(h < 1 for h in self.candidate_horizon_sessions):
            raise ValueError("candidate_horizon_sessions must be positive sessions")
        if self.fold_count < 1:
            raise ValueError("fold_count must be positive")
        if self.embargo_sessions < 0:
            raise ValueError("embargo_sessions must be non-negative")
        if self.forward_holdout_sessions < 0:
            raise ValueError("forward_holdout_sessions must be non-negative")
        if not 0.0 < self.bootstrap_alpha < 1.0:
            raise ValueError("bootstrap_alpha must be in (0, 1)")
        if self.bootstrap_resamples < 2:
            raise ValueError("bootstrap_resamples must be at least 2")
        if self.model_threads < 1:
            raise ValueError("model_threads must be positive")
        if self.max_rss_mib is not None and self.max_rss_mib <= 0:
            raise ValueError("max_rss_mib must be positive when supplied")


@dataclass(frozen=True, slots=True)
class HorizonLabelSpec:
    """One horizon's label contract: target and availability columns."""

    horizon_sessions: int
    target_column: str
    label_available_column: str

    def __post_init__(self) -> None:
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        if not self.target_column:
            raise ValueError("target_column must be non-empty")
        if not self.label_available_column:
            raise ValueError("label_available_column must be non-empty")


@dataclass(frozen=True, slots=True)
class HorizonJoinEvidence:
    """Retained/dropped row evidence for one horizon's point-in-time join."""

    horizon_sessions: int
    feature_rows: int
    label_rows: int
    joined_rows: int
    drop_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NetAlphaResearchData:
    """Canonical read model: feature frame plus per-horizon label frames.

    ``feature_frame`` carries one row per ``(instrument_id, decision_session)``
    and the declared ALPHA source columns. ``labels_by_horizon`` maps a
    candidate horizon to an independent label frame; horizons are never
    inner-joined into a common universe. ``join_evidence`` records retained and
    dropped row counts per horizon.
    """

    feature_frame: pl.DataFrame
    labels_by_horizon: dict[int, pl.DataFrame]
    manifest: DatasetManifest
    join_evidence: tuple[HorizonJoinEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.feature_frame.is_empty():
            raise ValueError("NetAlphaResearchData requires a non-empty feature frame")
        if not self.labels_by_horizon:
            raise ValueError("NetAlphaResearchData requires at least one horizon")
        for horizon, frame in self.labels_by_horizon.items():
            if frame.is_empty():
                raise ValueError(
                    f"NetAlphaResearchData horizon {horizon} label frame is empty"
                )


@dataclass(frozen=True, slots=True)
class ModelSelectionEvidence:
    """Immutable outcome of one horizon-discovery and model-family selection run."""

    primary_horizon_sessions: int | None
    secondary_horizon_sessions: int | None
    lower_bounds: dict[int, float]
    effective_horizon_count: float
    selection_reasons: tuple[str, ...]
    selected_model: str = "net_alpha_elastic_net"

    @property
    def selected_horizons(self) -> tuple[int, ...]:
        return tuple(
            h
            for h in (self.primary_horizon_sessions, self.secondary_horizon_sessions)
            if h is not None
        )

    def to_json(self) -> dict[str, object]:
        return {
            "primary_horizon_sessions": self.primary_horizon_sessions,
            "secondary_horizon_sessions": self.secondary_horizon_sessions,
            "lower_bounds": dict(self.lower_bounds),
            "effective_horizon_count": self.effective_horizon_count,
            "selection_reasons": list(self.selection_reasons),
            "selected_model": self.selected_model,
        }
