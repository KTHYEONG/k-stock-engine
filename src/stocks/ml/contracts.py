"""Net-alpha ML contracts: request, research data, label specs, selection evidence.

Every contract is an immutable, typed input or evidence record. The request
carries only the fields the net-alpha mainline needs: candidate horizons are a
pre-registered discovery grid, the selected result has at most one primary and
one conditional secondary horizon, and there is no fixed 5/10/15 route.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from src.core.costs import CostSchedule, LiquiditySlippageModel
from src.core.datasets import DatasetManifest

if TYPE_CHECKING:
    from src.stocks.domain.execution_policy import ExecutionOutcomePolicy
    from src.stocks.ml.data import HorizonOutcomeCoverage

DEFAULT_CANDIDATE_HORIZON_SESSIONS = (10, 20)
DEFAULT_CANDIDATE_REBALANCE_FREQUENCY_SESSIONS = (5, 10, 20)
DEFAULT_CANDIDATE_TOP_K = (12, 16, 20, 24)
CANONICAL_FEATURE_SET = "stock_net_alpha_v1"

ELASTIC_NET_FAMILY = "net_alpha_elastic_net"
TAIL_LAMBDARANK_FAMILY = "economic_tail_lambdarank"
RAWNET_LGBM_FAMILY = "economic_rawnet_lgbm"
DECLARED_ECONOMIC_FAMILIES = (
    ELASTIC_NET_FAMILY,
    TAIL_LAMBDARANK_FAMILY,
    RAWNET_LGBM_FAMILY,
)

OUTCOME_REALIZED = "REALIZED"
OUTCOME_PARTIAL_TAIL = "PARTIAL_TAIL"
OUTCOME_MISSING_ENTRY_PRICE = "MISSING_ENTRY_PRICE"
OUTCOME_MISSING_EXIT_PRICE = "MISSING_EXIT_PRICE"
OUTCOME_MISSING_DECISION_INPUT = "MISSING_DECISION_INPUT"
OUTCOME_UNDERSIZED_CROSS_SECTION = "UNDERSIZED_CROSS_SECTION"
OUTCOME_RISK_PROJECTION_FAILED = "RISK_PROJECTION_FAILED"
OUTCOME_ZERO_MAD = "ZERO_MAD"
OUTCOME_UNSUPPORTED_CORPORATE_ACTION = "UNSUPPORTED_CORPORATE_ACTION"
OUTCOME_UNEXECUTABLE_EXIT = "UNEXECUTABLE_EXIT"
OUTCOME_STATUS_VOCABULARY = (
    OUTCOME_REALIZED,
    OUTCOME_PARTIAL_TAIL,
    OUTCOME_MISSING_ENTRY_PRICE,
    OUTCOME_MISSING_EXIT_PRICE,
    OUTCOME_MISSING_DECISION_INPUT,
    OUTCOME_UNDERSIZED_CROSS_SECTION,
    OUTCOME_RISK_PROJECTION_FAILED,
    OUTCOME_ZERO_MAD,
    OUTCOME_UNSUPPORTED_CORPORATE_ACTION,
    OUTCOME_UNEXECUTABLE_EXIT,
)
RESOLVED_OUTCOME_STATUSES = (OUTCOME_REALIZED,)
TAIL_OUTCOME_STATUS = OUTCOME_PARTIAL_TAIL
OUTCOME_STATUS_COLUMN = "outcome_status"


def validate_outcome_status(value: object) -> str:
    """Validate a typed outcome status against the fixed vocabulary.

    Raises ``ValueError`` for an unknown, non-string, or empty status. The
    vocabulary is fixed and immutable; the producer may never invent a state.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"outcome status must be a non-empty string, got {value!r}")
    if value not in OUTCOME_STATUS_VOCABULARY:
        raise ValueError(
            f"unknown outcome status {value!r}; allowed "
            f"{OUTCOME_STATUS_VOCABULARY}"
        )
    return value

LEGACY_OVERLAY_PROFILE_ID = "legacy_overlay_5bps"
LOWER_BOUND_ONLY_PROFILE_ID = "lower_bound_only"
LOWER_BOUND_HALF_KELLY_PROFILE_ID = "lower_bound_half_kelly"
DEFAULT_POLICY_PROFILE_IDS = (LEGACY_OVERLAY_PROFILE_ID, LOWER_BOUND_ONLY_PROFILE_ID, LOWER_BOUND_HALF_KELLY_PROFILE_ID)
EXCESS_FULL_KELLY_PROFILE_ID = "excess_full_kelly"
GROWTH_FULL_UTILIZATION_PROFILE_ID = "growth_full_utilization"
ALLOWED_EXTRA_PROFILE_IDS = (EXCESS_FULL_KELLY_PROFILE_ID, GROWTH_FULL_UTILIZATION_PROFILE_ID)


@dataclass(frozen=True, slots=True)
class PolicyProfile:
    """Immutable pre-registered economic policy profile for the OOF frontier.

    A profile differs from another only in the decimal no-trade entry band
    applied on top of the calibrated ``net_alpha_lower_bound`` and the
    ``growth_risk_aversion`` scaling the Kelly exposure. It never changes the
    portfolio caps, cost schedules, liquidity model, purging, embargo, or
    bootstrap alpha unless an explicit override field is present.
    ``profile_id`` must be non-empty and unique within a frontier;
    ``no_trade_band_bps`` must be a finite non-negative value;
    ``growth_risk_aversion`` must be a finite strictly positive value.
    ``legacy_overlay_5bps`` reproduces the historical 5-bps entry filter;
    ``lower_bound_only`` keeps the lower-bound positivity gate without any
    extra overlay; ``lower_bound_half_kelly`` uses half-Kelly exposure
    (aversion=2). ``single_name_cap_override`` raises the single-name cap to
    at most the equal-weight basis ``1/top_k``; ``gross_utilization_target``
    normalizes deployed gross toward that fraction of equity; and
    ``vol_target_override`` replaces the canonical 12% annual volatility
    budget with a declared value (all three still clamped by the request
    portfolio caps).
    """

    profile_id: str
    no_trade_band_bps: float = 0.0
    growth_risk_aversion: float = 1.0
    execution_utility_mode: Literal["legacy_target_interpolation_v1", "delta_cost_aware_v1", "sparse_hold_replace_v2"] = "delta_cost_aware_v1"
    sizing_mode: Literal["alpha_vol_squared_v1", "risk_balanced_waterfill_v2", "confidence_mean_variance_v1"] = "alpha_vol_squared_v1"
    single_name_cap_override: float | None = None
    gross_utilization_target: float | None = None
    vol_target_override: float | None = None
    participation_limit_override: float | None = None
    turnover_budget_override: float | None = None

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if not np.isfinite(self.no_trade_band_bps) or self.no_trade_band_bps < 0.0:
            raise ValueError("no_trade_band_bps must be a finite non-negative value")
        if not np.isfinite(self.growth_risk_aversion) or self.growth_risk_aversion <= 0.0:
            raise ValueError("growth_risk_aversion must be a finite strictly positive value")
        for field_name in (
            "single_name_cap_override",
            "gross_utilization_target",
            "vol_target_override",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not np.isfinite(value) or not 0.0 < float(value) <= 1.0:
                raise ValueError(
                    f"{field_name} override must be None or finite in (0, 1], "
                    f"got {value!r}"
                )
        participation = self.participation_limit_override
        if participation is not None and (
            not np.isfinite(participation) or not 0.0 < float(participation) <= 1.0
        ):
            raise ValueError(
                "participation_limit_override must be None or finite in (0, 1], "
                f"got {participation!r}"
            )
        turnover = self.turnover_budget_override
        if turnover is not None and (
            not np.isfinite(turnover) or not 0.0 <= float(turnover) < 1.0
        ):
            raise ValueError(
                "turnover_budget_override must be None or finite in [0, 1), "
                f"got {turnover!r}"
            )
        if self.execution_utility_mode not in (
            "legacy_target_interpolation_v1",
            "delta_cost_aware_v1",
            "sparse_hold_replace_v2",
        ):
            raise ValueError(
                f"execution_utility_mode must be 'legacy_target_interpolation_v1', "
                f"'delta_cost_aware_v1', or 'sparse_hold_replace_v2', "
                f"got {self.execution_utility_mode!r}"
            )
        if self.sizing_mode not in (
            "alpha_vol_squared_v1",
            "risk_balanced_waterfill_v2",
            "confidence_mean_variance_v1",
        ):
            raise ValueError(
                f"sizing_mode must be 'alpha_vol_squared_v1', "
                f"'risk_balanced_waterfill_v2', or "
                f"'confidence_mean_variance_v1', got {self.sizing_mode!r}"
            )


@dataclass(frozen=True, slots=True)
class ExecutionFrontierSettings:
    """Pre-registered execution frontier: (H, C, K) candidate grid.

    ``candidate_horizon_sessions`` is the set of label horizons H,
    ``candidate_rebalance_frequency_sessions`` the set of review cadences C
    (must satisfy C <= H for every H), and ``candidate_top_k`` the set of
    maximum active-name counts K (must satisfy K >= ceil(gross_cap /
    single_name_cap)).  All valid (H, C, K) triples are evaluated without
    model refits; infeasible cells are filtered before replay.
    """

    candidate_horizon_sessions: tuple[int, ...] = DEFAULT_CANDIDATE_HORIZON_SESSIONS
    candidate_rebalance_frequency_sessions: tuple[int, ...] = DEFAULT_CANDIDATE_REBALANCE_FREQUENCY_SESSIONS
    candidate_top_k: tuple[int, ...] = DEFAULT_CANDIDATE_TOP_K

    def __post_init__(self) -> None:
        if not self.candidate_horizon_sessions:
            raise ValueError("candidate_horizon_sessions must be non-empty")
        if not self.candidate_rebalance_frequency_sessions:
            raise ValueError("candidate_rebalance_frequency_sessions must be non-empty")
        if not self.candidate_top_k:
            raise ValueError("candidate_top_k must be non-empty")
        if tuple(self.candidate_horizon_sessions) != tuple(
            sorted(set(self.candidate_horizon_sessions))
        ):
            raise ValueError("candidate_horizon_sessions must be strictly ascending and unique")
        if tuple(self.candidate_rebalance_frequency_sessions) != tuple(
            sorted(set(self.candidate_rebalance_frequency_sessions))
        ):
            raise ValueError(
                "candidate_rebalance_frequency_sessions must be strictly ascending and unique"
            )
        if tuple(self.candidate_top_k) != tuple(sorted(set(self.candidate_top_k))):
            raise ValueError("candidate_top_k must be strictly ascending and unique")
        if any(h < 1 for h in self.candidate_horizon_sessions):
            raise ValueError("candidate_horizon_sessions must be positive sessions")
        if any(c < 1 for c in self.candidate_rebalance_frequency_sessions):
            raise ValueError("candidate_rebalance_frequency_sessions must be positive sessions")
        if any(k < 1 for k in self.candidate_top_k):
            raise ValueError("candidate_top_k must be positive")

    def feasible_cells(
        self, gross_cap: float, single_name_cap: float
    ) -> tuple[tuple[int, int, int], ...]:
        """Return only (H, C, K) triples satisfying C <= H and K >= ceil(gross/single)."""
        from math import ceil

        min_k = ceil(gross_cap / single_name_cap) if single_name_cap > 0 else 1
        cells: list[tuple[int, int, int]] = []
        for h in self.candidate_horizon_sessions:
            for c in self.candidate_rebalance_frequency_sessions:
                if c > h:
                    continue
                cells.extend(
                    (h, c, k)
                    for k in self.candidate_top_k
                    if k >= min_k
                )
        return tuple(cells)

    def feasible_cells_for_profile(
        self,
        portfolio_gross_cap: float,
        portfolio_single_name_cap: float,
        *,
        single_name_cap_override: float | None = None,
        gross_utilization_target: float | None = None,
    ) -> tuple[tuple[int, int, int], ...]:
        """Profile-scoped feasible cells under effective per-name caps.

        The per-K single-name cap is ``min(single_name_cap_override, 1/K)``
        when an override is supplied, else ``portfolio_single_name_cap``;
        a cell (H, C, K) is feasible iff ``C <= H`` and
        ``K * cap >= eff_gross`` where
        ``eff_gross = min(gross_utilization_target or portfolio_gross_cap,
        portfolio_gross_cap)``. With no overrides this reduces exactly to
        :meth:`feasible_cells`, so profiles without cap overrides keep the
        legacy frontier bit-for-bit.
        """
        eff_gross = portfolio_gross_cap
        if gross_utilization_target is not None:
            eff_gross = min(gross_utilization_target, portfolio_gross_cap)
        cells: list[tuple[int, int, int]] = []
        for h in self.candidate_horizon_sessions:
            for c in self.candidate_rebalance_frequency_sessions:
                if c > h:
                    continue
                cells.extend(
                    (h, c, k)
                    for k in self.candidate_top_k
                    if k * self._effective_single_name_cap(
                        k, portfolio_single_name_cap, single_name_cap_override
                    )
                    >= eff_gross - 1e-12
                )
        return tuple(cells)

    @staticmethod
    def _effective_single_name_cap(
        top_k: int,
        base_cap: float,
        override: float | None,
    ) -> float:
        if override is None:
            return base_cap
        return min(override, 1.0 / top_k)

    def require_feasible_horizons(
        self, gross_cap: float, single_name_cap: float
    ) -> tuple[tuple[int, int, int], ...]:
        """Validate that every requested horizon has at least one feasible cell.

        Raises ``ValueError`` naming the infeasible horizon and its cadence set
        when any horizon owns zero feasible (H, C, K) cells.  Returns the
        feasible cells when all horizons are satisfiable.
        """
        from math import ceil

        min_k = ceil(gross_cap / single_name_cap) if single_name_cap > 0 else 1
        for h in self.candidate_horizon_sessions:
            has_feasible = any(
                c <= h and k >= min_k
                for c in self.candidate_rebalance_frequency_sessions
                for k in self.candidate_top_k
            )
            if not has_feasible:
                raise ValueError(
                    f"horizon H={h} has no feasible (H,C,K) cell with "
                    f"C in {tuple(self.candidate_rebalance_frequency_sessions)} "
                    f"and K >= {min_k}; every horizon must own at least one "
                    f"feasible cell before model fitting"
                )
        return self.feasible_cells(gross_cap, single_name_cap)


DEFAULT_POLICY_PROFILES = (
    PolicyProfile(
        profile_id=LEGACY_OVERLAY_PROFILE_ID,
        no_trade_band_bps=5.0,
        growth_risk_aversion=1.0,
        execution_utility_mode="sparse_hold_replace_v2",
        sizing_mode="risk_balanced_waterfill_v2",
    ),
    PolicyProfile(
        profile_id=LOWER_BOUND_ONLY_PROFILE_ID,
        no_trade_band_bps=0.0,
        growth_risk_aversion=1.0,
        execution_utility_mode="sparse_hold_replace_v2",
        sizing_mode="risk_balanced_waterfill_v2",
    ),
    PolicyProfile(
        profile_id=LOWER_BOUND_HALF_KELLY_PROFILE_ID,
        no_trade_band_bps=0.0,
        growth_risk_aversion=2.0,
        execution_utility_mode="sparse_hold_replace_v2",
        sizing_mode="confidence_mean_variance_v1",
    ),
)


def validate_policy_profiles(profiles: tuple[PolicyProfile, ...]) -> tuple[PolicyProfile, ...]:
    """Validate a policy frontier: the three defaults plus allowed extras only.

    Raises ``ValueError`` on an empty frontier, a duplicate profile id, a
    missing default profile, or an unexpected extra profile. The opt-in
    ``excess_full_kelly`` and ``growth_full_utilization`` rungs are the only
    permitted extras and must be appended after the defaults. The frontier is
    pre-registered: the discovery grid replays every cached OOF score under
    these exact policies and never refits a learner per profile.
    """
    if not profiles:
        raise ValueError("policy frontier requires at least one profile")
    ids = [profile.profile_id for profile in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("policy profile ids must be unique")
    missing = [default_id for default_id in DEFAULT_POLICY_PROFILE_IDS if default_id not in ids]
    if missing:
        raise ValueError(
            f"default policy profile {missing[0]!r} is missing from the frontier"
        )
    extras = [pid for pid in ids if pid not in DEFAULT_POLICY_PROFILE_IDS]
    disallowed = [pid for pid in extras if pid not in ALLOWED_EXTRA_PROFILE_IDS]
    if disallowed:
        raise ValueError(
            f"policy frontier extra profile {disallowed[0]!r} is not permitted; "
            f"allowed extras: {ALLOWED_EXTRA_PROFILE_IDS}"
        )
    expected = DEFAULT_POLICY_PROFILE_IDS + tuple(extras)
    if tuple(ids) != expected:
        raise ValueError(
            "policy frontier must keep the default order "
            f"{DEFAULT_POLICY_PROFILE_IDS} with any allowed extras appended; "
            f"got {tuple(ids)}"
        )
    return profiles


def policy_portfolio_fingerprint(
    top_k: int,
    max_single_weight: float,
    max_exposure: float,
    participation_limit: float,
) -> str:
    """Deterministic SHA-256 fingerprint of the policy's portfolio constraints.

    Binds the top-k, single-name cap, gross exposure cap, and participation
    limit that the training OOF, forward holdout, artifact score, and the
    independent portfolio backtester must share. Callers with divergent caps
    produce different fingerprints, so a backtester can never silently use a
    different policy than the one the artifact was selected under.
    """
    payload = json.dumps(
        {
            "top_k": int(top_k),
            "max_single_weight": float(max_single_weight),
            "max_exposure": float(max_exposure),
            "participation_limit": float(participation_limit),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
class CompoundingCertificationSettings:
    """Pre-registered research-governance policy for compound-growth evidence.

    This is a governance policy, never an optimization knob tuned against a
    holdout outcome. ``annualization_sessions`` defines one full annualization
    window and ``min_observed_sessions`` the minimum observed sessions a
    forward holdout must cover; ``min_active_cohort_fraction`` and
    ``max_drawdown`` are explicit validated risk-policy fields. The bootstrap
    is seeded and reuses the request's statistical confidence settings unless
    explicitly overridden. Values outside their finite domains raise
    ``ValueError``.
    """

    annualization_sessions: int = 252
    min_observed_sessions: int = 252
    min_active_cohort_fraction: float = 0.2
    max_drawdown: float = 0.5
    bootstrap_alpha: float = 0.05
    bootstrap_resamples: int = 200
    seed: int = 42
    allowed_tail_censoring_sessions: int = 0
    hedge_leverage_grid: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.annualization_sessions < 1:
            raise ValueError("annualization_sessions must be positive")
        if self.min_observed_sessions < 1:
            raise ValueError("min_observed_sessions must be positive")
        if (
            self.allowed_tail_censoring_sessions < 0
            or self.allowed_tail_censoring_sessions >= self.min_observed_sessions
        ):
            raise ValueError(
                "allowed_tail_censoring_sessions must be in "
                "[0, min_observed_sessions)"
            )
        if not 0.0 < self.min_active_cohort_fraction <= 1.0:
            raise ValueError("min_active_cohort_fraction must be in (0, 1]")
        if not 0.0 < self.max_drawdown < 1.0:
            raise ValueError("max_drawdown must be in (0, 1)")
        if not 0.0 < self.bootstrap_alpha < 1.0:
            raise ValueError("bootstrap_alpha must be in (0, 1)")
        if self.bootstrap_resamples < 2:
            raise ValueError("bootstrap_resamples must be at least 2")
        if self.hedge_leverage_grid is not None:
            grid = tuple(self.hedge_leverage_grid)
            if not grid:
                raise ValueError("hedge_leverage_grid must not be empty when set")
            if any(not math.isfinite(value) for value in grid):
                raise ValueError("hedge_leverage_grid values must be finite")
            if any(value < 1.0 or value > 5.0 for value in grid):
                raise ValueError(
                    "hedge_leverage_grid values must be in [1.0, 5.0]"
                )
            if list(grid) != sorted(set(grid)):
                raise ValueError(
                    "hedge_leverage_grid values must be strictly ascending "
                    "and unique"
                )


@dataclass(frozen=True, slots=True)
class NetAlphaTrainingRequest:
    """Input contract for the net-alpha training workflow.

    ``candidate_horizon_sessions`` is a pre-registered discovery grid, not an
    operating route: the trainer fits the baseline for every candidate and
    selects at most one primary and one conditional secondary horizon from OOF
    replay evidence. ``policy_profiles`` is the pre-registered two-policy
    frontier replayed over each candidate's cached OOF scores; the selected
    profile is frozen into the artifact and the operational backtester.
    ``compounding`` is the pre-registered research-governance
    certificate policy for the untouched forward holdout, never a post-hoc
    threshold. ``model_threads`` is the single thread budget for the
    challenger LightGBM (default 1); there is no Optuna trial, resume, or
    ``lgb_threads`` knob. ``memory_reserve_mib`` is the measured
    concurrent-workload reserve subtracted only from cgroup/system headroom,
    never from the request RSS budget. ``max_training_lookback_sessions``
    optionally bounds every fitting fold to the newest eligible sessions
    after purge and embargo (minimum one annualized year); ``None`` preserves
    expanding training windows.
    """

    artifact_id: str
    candidate_horizon_sessions: tuple[int, ...] = DEFAULT_CANDIDATE_HORIZON_SESSIONS
    execution_frontier: ExecutionFrontierSettings = field(
        default_factory=ExecutionFrontierSettings
    )
    policy_profiles: tuple[PolicyProfile, ...] = DEFAULT_POLICY_PROFILES
    fold_count: int = 3
    embargo_sessions: int = 5
    forward_holdout_sessions: int = 0
    bootstrap_alpha: float = 0.05
    bootstrap_resamples: int = 2000
    model_threads: int = 1
    max_rss_mib: int | None = None
    memory_reserve_mib: int = 0
    max_training_lookback_sessions: int | None = None
    seed: int = 42
    portfolio: PortfolioSettings = field(default_factory=PortfolioSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    compounding: CompoundingCertificationSettings = field(
        default_factory=CompoundingCertificationSettings
    )
    base_cost_schedule: CostSchedule | None = None
    stress_cost_schedule: CostSchedule | None = None
    liquidity_model: LiquiditySlippageModel | None = None
    stress_liquidity_model: LiquiditySlippageModel | None = None
    execution_policy: ExecutionOutcomePolicy | None = None
    enforce_snapshot_outcome_readiness: bool = True
    discovery_model_family: str = ELASTIC_NET_FAMILY
    enable_horizon_blend: bool = False
    holm_family_scope: Literal["frontier", "route_gatekeeping"] = "frontier"
    discovery_workers: int = 1
    enable_sparse_retained_rewaterfill: bool = False
    enable_excess_route: bool = False

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        if self.holm_family_scope not in ("frontier", "route_gatekeeping"):
            raise ValueError(
                "holm_family_scope must be 'frontier' or 'route_gatekeeping', "
                f"got {self.holm_family_scope!r}"
            )
        if self.discovery_workers < 1:
            raise ValueError("discovery_workers must be a positive integer")
        if not self.candidate_horizon_sessions:
            raise ValueError("candidate_horizon_sessions must be non-empty")
        if tuple(self.candidate_horizon_sessions) != tuple(
            sorted(set(self.candidate_horizon_sessions))
        ):
            raise ValueError("candidate_horizon_sessions must be strictly ascending and unique")
        if any(h < 1 for h in self.candidate_horizon_sessions):
            raise ValueError("candidate_horizon_sessions must be positive sessions")
        object.__setattr__(self, "policy_profiles", validate_policy_profiles(tuple(self.policy_profiles)))
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
        if self.memory_reserve_mib < 0:
            raise ValueError("memory_reserve_mib must be non-negative")
        if self.max_training_lookback_sessions is not None:
            lookback = self.max_training_lookback_sessions
            if lookback <= 0:
                raise ValueError("max_training_lookback_sessions must be positive")
            if lookback < 252:
                raise ValueError(
                    "max_training_lookback_sessions must be at least 252 "
                    f"sessions (one annualized certificate year), got {lookback}"
                )
        if self.discovery_model_family not in DECLARED_ECONOMIC_FAMILIES:
            raise ValueError(
                f"unknown discovery_model_family "
                f"{self.discovery_model_family!r}; declared families are "
                f"{DECLARED_ECONOMIC_FAMILIES}"
            )
        if self.enable_horizon_blend and len(self.candidate_horizon_sessions) < 2:
            raise ValueError(
                "enable_horizon_blend requires at least two candidate horizons; "
                f"got {tuple(self.candidate_horizon_sessions)}"
            )


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
class OutcomeStatusCounts:
    """Immutable bounded per-status outcome counts for one horizon/segment.

    ``counts`` is a sorted tuple of ``(status, count)`` pairs restricted to the
    fixed vocabulary; an unknown state or a negative count raises
    ``ValueError``. ``REALIZED`` is the only state that may enter realised
    return arithmetic; ``PARTIAL_TAIL`` is a chronological tail; every other
    state is an unresolved, non-certifiable outcome.
    """

    counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        pairs = []
        for status, count in self.counts:
            validate_outcome_status(status)
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"outcome status count must be a non-negative int, got {count!r}")
            pairs.append((status, count))
        if len({status for status, _ in pairs}) != len(pairs):
            raise ValueError("outcome status counts contain duplicate states")
        object.__setattr__(self, "counts", tuple(sorted(pairs)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, int]) -> OutcomeStatusCounts:
        """Build bounded counts from a status-to-count mapping.

        Unknown states raise ``ValueError``; zero-count states are dropped so
        the record stays bounded.
        """
        pairs = []
        for status, count in mapping.items():
            validate_outcome_status(status)
            if not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"outcome status count must be a non-negative int, got {count!r}"
                )
            if count > 0:
                pairs.append((status, count))
        return cls(tuple(pairs))

    def count(self, status: str) -> int:
        validate_outcome_status(status)
        for known, value in self.counts:
            if known == status:
                return value
        return 0

    @property
    def realized(self) -> int:
        return self.count(OUTCOME_REALIZED)

    @property
    def partial_tail(self) -> int:
        return self.count(OUTCOME_PARTIAL_TAIL)

    @property
    def unresolved(self) -> int:
        return sum(
            count
            for status, count in self.counts
            if status not in RESOLVED_OUTCOME_STATUSES and status != OUTCOME_PARTIAL_TAIL
        )

    def to_json(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass(frozen=True, slots=True)
class SegmentOutcomeCounts:
    """Bounded per-segment outcome counts for one horizon."""

    segment_id: int
    counts: OutcomeStatusCounts

    def __post_init__(self) -> None:
        if self.segment_id < 0:
            raise ValueError("segment_id must be non-negative")

    def to_json(self) -> dict[str, object]:
        return {"segment_id": int(self.segment_id), "counts": self.counts.to_json()}


@dataclass(frozen=True, slots=True)
class HorizonJoinEvidence:
    """Retained/dropped row evidence for one horizon's point-in-time join.

    ``decision_rows`` counts the unique decision feature keys preserved by the
    composition for the horizon, ``realized_rows`` the rows with a realised
    label, and ``status_counts`` the bounded per-status outcome counts over the
    decision universe. ``drop_reasons`` retains the deterministic admission
    reasons of keys that produced no realised label.
    """

    horizon_sessions: int
    feature_rows: int
    label_rows: int
    joined_rows: int
    drop_reasons: tuple[str, ...] = ()
    decision_rows: int = 0
    realized_rows: int = 0
    status_counts: OutcomeStatusCounts | None = None


@dataclass(frozen=True, slots=True)
class NetAlphaResearchData:
    """Canonical read model: feature frame plus per-horizon label frames.

    ``feature_frame`` carries one row per ``(instrument_id, decision_session)``
    and the declared ALPHA source columns. ``labels_by_horizon`` maps a
    candidate horizon to an independent label frame; horizons are never
    inner-joined into a common universe. ``status_by_horizon`` maps a candidate
    horizon to its typed outcome-status sidecar (one status row per decision
    key), and ``coverage_by_horizon`` the built vectorized outcome coverage.
    ``evidence_by_horizon`` maps a horizon to its hash-bound outcome-evidence
    projection carrying ``resolution_kind``/``policy_hash`` per key, which the
    data-provenance gate uses to distinguish confirmed no-bars from collection
    gaps. ``status_provenance`` is ``pinned`` when the composed frame carried a
    snapshot-pinned status/evidence spine and ``legacy-inferred`` otherwise.
    ``join_evidence`` records retained/dropped row counts, decision/realised
    rows, and per-status counts per horizon.
    """

    feature_frame: pl.DataFrame
    labels_by_horizon: dict[int, pl.DataFrame]
    manifest: DatasetManifest
    join_evidence: tuple[HorizonJoinEvidence, ...] = ()
    status_by_horizon: dict[int, pl.DataFrame] = field(default_factory=dict)
    coverage_by_horizon: dict[int, HorizonOutcomeCoverage] = field(
        default_factory=dict
    )
    evidence_by_horizon: dict[int, pl.DataFrame] = field(default_factory=dict)
    status_provenance: str = "legacy-inferred"

    def __post_init__(self) -> None:
        if self.feature_frame.is_empty():
            raise ValueError("NetAlphaResearchData requires a non-empty feature frame")
        if not self.labels_by_horizon:
            raise ValueError("NetAlphaResearchData requires at least one horizon")
        if self.status_provenance not in ("pinned", "legacy-inferred"):
            raise ValueError(
                "status_provenance must be 'pinned' or 'legacy-inferred'"
            )
        for horizon, frame in self.labels_by_horizon.items():
            if frame.is_empty():
                raise ValueError(
                    f"NetAlphaResearchData horizon {horizon} label frame is empty"
                )
        for horizon, frame in self.status_by_horizon.items():
            if frame.is_empty():
                raise ValueError(
                    f"NetAlphaResearchData horizon {horizon} status frame is empty"
                )
        for horizon, frame in self.evidence_by_horizon.items():
            if frame.is_empty():
                raise ValueError(
                    f"NetAlphaResearchData horizon {horizon} evidence frame is empty"
                )


#: Training-side alias for the canonical composed read model.
NetAlphaTrainingData = NetAlphaResearchData


@dataclass(frozen=True, slots=True)
class RegularizationGrid:
    """Pre-registered scale-invariant ElasticNet penalty fractions.

    The selector evaluates these fractions of the fold-local ``alpha_max``
    (``max(abs(X.T @ y)) / n`` on a centered target and standardized design)
    instead of a fixed absolute penalty, so the chosen strength is invariant to
    target units.
    """

    fractions: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30)

    def __post_init__(self) -> None:
        if not self.fractions:
            raise ValueError("fractions must be non-empty")
        if any(not np.isfinite(f) or f <= 0.0 for f in self.fractions):
            raise ValueError("fractions must be finite and positive")
        if tuple(self.fractions) != tuple(sorted(set(self.fractions))):
            raise ValueError("fractions must be strictly ascending and unique")


@dataclass(frozen=True, slots=True)
class FoldScoreDiagnostic:
    """One purged fold's target-free prediction diagnostics.

    Carries the fold score standard deviation, finite/unique prediction counts,
    the fold-local regularization metadata selected by the nested ElasticNet
    selector, and a deterministic failure reason. The reason is the empty string
    when the fold produced usable non-constant predictions; expected invalid
    inputs are classified here (``fit-error:...``, ``constant-oof-score``, ...)
    so the ``ValueError`` is never silently swallowed.
    """

    fold_index: int
    score_std: float = 0.0
    finite_count: int = 0
    unique_count: int = 0
    rank_ic: float = 0.0
    alpha: float | None = None
    fraction: float | None = None
    alpha_max: float | None = None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if self.finite_count < 0 or self.unique_count < 0:
            raise ValueError("finite/unique counts must be non-negative")
        if not np.isfinite(self.score_std) or self.score_std < 0.0:
            raise ValueError("score_std must be a finite non-negative value")
        if not np.isfinite(self.rank_ic):
            raise ValueError("rank_ic must be finite")
        for name in ("alpha", "fraction", "alpha_max"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when supplied")

    def to_json(self) -> dict[str, object]:
        return {
            "fold_index": int(self.fold_index),
            "score_std": round(float(self.score_std), 12),
            "finite_count": int(self.finite_count),
            "unique_count": int(self.unique_count),
            "rank_ic": round(float(self.rank_ic), 12),
            "alpha": None if self.alpha is None else round(float(self.alpha), 12),
            "fraction": None if self.fraction is None else round(float(self.fraction), 12),
            "alpha_max": None if self.alpha_max is None else round(float(self.alpha_max), 12),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class HorizonOOFDiagnostic:
    """Aggregated per-horizon OOF diagnostics for one model family."""

    horizon_sessions: int
    model_family: str
    fold_diagnostics: tuple[FoldScoreDiagnostic, ...] = ()
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be a positive session count")
        if not self.model_family:
            raise ValueError("model_family must be non-empty")

    @property
    def fold_score_stds(self) -> tuple[float, ...]:
        return tuple(diag.score_std for diag in self.fold_diagnostics)

    @property
    def fold_finite_counts(self) -> tuple[int, ...]:
        return tuple(diag.finite_count for diag in self.fold_diagnostics)

    @property
    def fold_unique_counts(self) -> tuple[int, ...]:
        return tuple(diag.unique_count for diag in self.fold_diagnostics)

    @property
    def fold_rank_ics(self) -> tuple[float, ...]:
        return tuple(diag.rank_ic for diag in self.fold_diagnostics)

    @property
    def usable_fold_count(self) -> int:
        return sum(1 for diag in self.fold_diagnostics if not diag.failure_reason)

    def to_json(self) -> dict[str, object]:
        return {
            "horizon_sessions": int(self.horizon_sessions),
            "model_family": self.model_family,
            "fold_score_stds": [round(float(v), 12) for v in self.fold_score_stds],
            "fold_finite_counts": [int(v) for v in self.fold_finite_counts],
            "fold_unique_counts": [int(v) for v in self.fold_unique_counts],
            "fold_rank_ics": [round(float(v), 12) for v in self.fold_rank_ics],
            "usable_fold_count": int(self.usable_fold_count),
            "failure_reason": self.failure_reason,
            "folds": [diag.to_json() for diag in self.fold_diagnostics],
        }


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


@dataclass(frozen=True, slots=True)
class EconomicFamilyStudySettings:
    """Immutable pre-registered economic-family study configuration.

    ``candidate_lookback_sessions`` mirrors the temporal-window study: strictly
    ascending finite session caps with at most one trailing ``None`` for the
    expanding-window control. ``common_min_train_sessions`` must cover the
    maximum finite candidate so every window shares one first validation
    boundary on one common purged fold calendar. ``model_families`` is the
    pre-registered family order; the first entry wins ties deterministically.
    """

    candidate_lookback_sessions: tuple[int | None, ...] = (504, 756, 1260, None)
    common_min_train_sessions: int = 1260
    min_validation_segment_sessions: int = 126
    # Pre-registered study grid stays pinned to the legacy two-family split;
    # RAWNET_LGBM_FAMILY enters via explicit settings or mainline discovery.
    model_families: tuple[str, ...] = (ELASTIC_NET_FAMILY, TAIL_LAMBDARANK_FAMILY)

    def __post_init__(self) -> None:
        if not self.candidate_lookback_sessions:
            raise ValueError("candidate_lookback_sessions must be non-empty")
        finite = [v for v in self.candidate_lookback_sessions if v is not None]
        if any(v < 1 for v in finite):
            raise ValueError("finite candidate lookbacks must be positive sessions")
        if len(set(finite)) != len(finite) or list(finite) != sorted(finite):
            raise ValueError(
                "finite candidate lookbacks must be strictly ascending and unique"
            )
        if any(v is None for v in tuple(self.candidate_lookback_sessions)[:-1]):
            raise ValueError(
                "expanding (None) is only permitted in the final position"
            )
        if self.common_min_train_sessions < 1:
            raise ValueError("common_min_train_sessions must be positive")
        if self.min_validation_segment_sessions < 1:
            raise ValueError("min_validation_segment_sessions must be positive")
        if finite and self.common_min_train_sessions < max(finite):
            raise ValueError(
                "common_min_train_sessions must be at least the maximum "
                "finite candidate lookback"
            )
        if not self.model_families:
            raise ValueError("model_families must be non-empty")
        if len(set(self.model_families)) != len(self.model_families):
            raise ValueError("model_families must be unique")
        unknown = [
            family
            for family in self.model_families
            if family not in DECLARED_ECONOMIC_FAMILIES
        ]
        if unknown:
            raise ValueError(
                f"unknown model families {unknown}; declared families are "
                f"{DECLARED_ECONOMIC_FAMILIES}"
            )
