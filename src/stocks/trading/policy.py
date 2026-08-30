"""Stock trading policy types - canonical ownership.

``StockRiskPolicy``, ``CompoundingPolicyConfig``, ``ExecutionUtility``, and ``SizingMethod``
define the immutable risk profile and semantic enum types for trading decisions.
This module is the single owner; portfolio_constructor re-exports compatibility aliases.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


@dataclass(frozen=True, slots=True)
class CompoundingPolicyConfig:
    """Immutable lower-confidence compounding overlay configuration."""

    enabled: bool = True
    growth_risk_aversion: float = 1.0
    forecast_horizon_sessions: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.growth_risk_aversion) or self.growth_risk_aversion <= 0.0:
            raise ValueError(
                "growth_risk_aversion must be finite and strictly positive"
            )
        if self.forecast_horizon_sessions is not None and (
            not isinstance(self.forecast_horizon_sessions, int)
            or self.forecast_horizon_sessions < 1
        ):
            raise ValueError(
                "forecast_horizon_sessions must be a positive integer when provided"
            )


@dataclass(frozen=True, slots=True)
class LotSizingConfig:
    """Account-only causal lot sizing config."""

    enabled: bool = False
    entry_price_buffer_bps: float | None = None
    opening_gap_quantile: float = 0.95

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be bool")
        if self.entry_price_buffer_bps is not None:
            v = float(self.entry_price_buffer_bps)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError("entry_price_buffer_bps must be finite non-negative when provided")
        if self.enabled and self.entry_price_buffer_bps is None:
            raise ValueError("entry_price_buffer_bps required when lot sizing is enabled")
        if not math.isfinite(self.opening_gap_quantile) or not 0.0 < self.opening_gap_quantile < 1.0:
            raise ValueError("opening_gap_quantile must be finite in (0, 1)")


class ExecutionUtility(StrEnum):
    """Semantic execution utility modes."""

    LEGACY_TARGET_INTERPOLATION = "legacy_target_interpolation_v1"
    DELTA_COST_AWARE = "delta_cost_aware_v1"
    SPARSE_HOLD_REPLACE = "sparse_hold_replace_v2"


class SizingMethod(StrEnum):
    """Semantic sizing methods."""

    ALPHA_VOL_SQUARED = "alpha_vol_squared_v1"
    RISK_BALANCED_WATERFILL = "risk_balanced_waterfill_v2"
    CONFIDENCE_MEAN_VARIANCE = "confidence_mean_variance_v1"


@dataclass(frozen=True, slots=True)
class StockRiskPolicy:
    """Frozen, versioned risk profile for target construction."""

    top_k: int = 20
    target_count: int | None = None
    enter_rank: int = 15
    keep_rank: int = 30
    gross_cap: float = 0.90
    single_name_cap: float = 0.08
    sector_cap: float = 0.25
    participation_limit: float = 0.005
    no_trade_band_bps: float = 0.0
    target_annual_volatility: float = 0.12
    turnover_budget: float = 0.20
    volatility_lookback_sessions: int = 20
    covariance_lookback_sessions: int = 60
    rebalance_frequency_sessions: int = 5
    annualization_sessions: int = 252
    economic_hysteresis: bool = True
    compounding: CompoundingPolicyConfig = field(default_factory=CompoundingPolicyConfig)
    compounding_evidence: list[dict[str, object]] = field(
        default_factory=list, repr=False, compare=False
    )
    economic_ranking_mode: Literal["raw_score_v1", "economic_net_v1"] = "raw_score_v1"
    execution_utility_mode: Literal["legacy_target_interpolation_v1", "delta_cost_aware_v1", "sparse_hold_replace_v2"] = "legacy_target_interpolation_v1"
    sizing_mode: Literal[
        "alpha_vol_squared_v1", "risk_balanced_waterfill_v2", "confidence_mean_variance_v1", "continuous_uncertainty_v1"
    ] = "alpha_vol_squared_v1"
    economic_gate_mode: Literal["lower_bound_v1", "finite_mean_v1"] = "lower_bound_v1"
    retained_sizing_mode: Literal["freeze_v1", "band_limited_rewaterfill_v1"] = "freeze_v1"
    net_exposure_gate_mode: Literal["off_v1", "trend_vol_v1"] = "off_v1"
    gate_trend_lookback_sessions: int = 60
    gate_floor: float = 0.25
    lot_sizing: LotSizingConfig = field(default_factory=LotSizingConfig)

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.target_count is not None:
            if self.target_count <= 0:
                raise ValueError("target_count must be positive")
            if self.target_count != self.top_k or self.target_count != self.enter_rank:
                raise ValueError(
                    "target_count, top_k, and enter_rank must agree for v3 policy"
                )
        if not (0 < self.enter_rank <= self.keep_rank):
            raise ValueError("ranks must satisfy 0 < enter_rank <= keep_rank")
        if not (0.0 < self.single_name_cap <= self.sector_cap <= self.gross_cap <= 1.0):
            raise ValueError(
                "caps must satisfy 0 < single_name_cap <= sector_cap <= gross_cap <= 1"
            )
        if not (0.0 <= self.participation_limit <= 1.0):
            raise ValueError("participation_limit must be in [0, 1]")
        if not math.isfinite(self.no_trade_band_bps) or self.no_trade_band_bps < 0.0:
            raise ValueError("no_trade_band_bps must be a finite non-negative value")
        if self.target_annual_volatility <= 0:
            raise ValueError("target_annual_volatility must be positive")
        if self.turnover_budget < 0:
            raise ValueError("turnover_budget must be non-negative")
        if self.economic_gate_mode not in ("lower_bound_v1", "finite_mean_v1"):
            raise ValueError("economic_gate_mode must be 'lower_bound_v1' or 'finite_mean_v1'")
        if self.economic_gate_mode == "finite_mean_v1" and self.sizing_mode != "continuous_uncertainty_v1":
            raise ValueError("finite_mean_v1 requires continuous_uncertainty_v1 sizing")
        if (
            self.volatility_lookback_sessions <= 0
            or self.covariance_lookback_sessions <= 0
            or self.rebalance_frequency_sessions <= 0
            or self.annualization_sessions <= 0
        ):
            raise ValueError("lookbacks, frequency, and annualization must be positive")
        if self.economic_ranking_mode not in ("raw_score_v1", "economic_net_v1"):
            raise ValueError(
                "economic_ranking_mode must be 'raw_score_v1' or 'economic_net_v1'"
            )
        if self.execution_utility_mode not in (
            "legacy_target_interpolation_v1",
            "delta_cost_aware_v1",
            "sparse_hold_replace_v2",
        ):
            raise ValueError(
                "execution_utility_mode must be 'legacy_target_interpolation_v1', "
                "'delta_cost_aware_v1', or 'sparse_hold_replace_v2', "
                f"got {self.execution_utility_mode!r}"
            )
        if self.sizing_mode not in (
            "alpha_vol_squared_v1",
            "risk_balanced_waterfill_v2",
            "confidence_mean_variance_v1",
            "continuous_uncertainty_v1",
        ):
            raise ValueError(
                "sizing_mode must be 'alpha_vol_squared_v1', "
                f"'risk_balanced_waterfill_v2', 'confidence_mean_variance_v1', or 'continuous_uncertainty_v1', got {self.sizing_mode!r}"
            )
        if self.retained_sizing_mode not in ("freeze_v1", "band_limited_rewaterfill_v1"):
            raise ValueError(
                "retained_sizing_mode must be 'freeze_v1' or "
                f"'band_limited_rewaterfill_v1', got {self.retained_sizing_mode!r}"
            )
        if self.net_exposure_gate_mode not in ("off_v1", "trend_vol_v1"):
            raise ValueError(
                "net_exposure_gate_mode must be 'off_v1' or 'trend_vol_v1', "
                f"got {self.net_exposure_gate_mode!r}"
            )
        if self.gate_trend_lookback_sessions <= 0:
            raise ValueError("gate_trend_lookback_sessions must be positive")
        if not math.isfinite(self.gate_floor) or not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("gate_floor must be a finite fraction in [0, 1)")


def stock_risk_policy_fingerprint(policy: StockRiskPolicy) -> str:
    """Deterministic canonical SHA-256 fingerprint of a frozen risk policy."""
    payload = json.dumps(
        {
            "top_k": int(policy.top_k),
            "target_count": None if policy.target_count is None else int(policy.target_count),
            "enter_rank": int(policy.enter_rank),
            "keep_rank": int(policy.keep_rank),
            "gross_cap": float(policy.gross_cap),
            "single_name_cap": float(policy.single_name_cap),
            "sector_cap": float(policy.sector_cap),
            "participation_limit": float(policy.participation_limit),
            "no_trade_band_bps": float(policy.no_trade_band_bps),
            "target_annual_volatility": float(policy.target_annual_volatility),
            "turnover_budget": float(policy.turnover_budget),
            "volatility_lookback_sessions": int(policy.volatility_lookback_sessions),
            "covariance_lookback_sessions": int(policy.covariance_lookback_sessions),
            "rebalance_frequency_sessions": int(policy.rebalance_frequency_sessions),
            "annualization_sessions": int(policy.annualization_sessions),
            "economic_hysteresis": bool(policy.economic_hysteresis),
            "compounding": {
                "enabled": bool(policy.compounding.enabled),
                "growth_risk_aversion": float(policy.compounding.growth_risk_aversion),
                "forecast_horizon_sessions": (
                    int(policy.compounding.forecast_horizon_sessions)
                    if policy.compounding.forecast_horizon_sessions is not None
                    else None
                ),
            },
            "economic_ranking_mode": str(policy.economic_ranking_mode),
            "execution_utility_mode": str(policy.execution_utility_mode),
            "sizing_mode": str(policy.sizing_mode),
            "economic_gate_mode": str(policy.economic_gate_mode),
            "retained_sizing_mode": str(policy.retained_sizing_mode),
            **(
                {}
                if str(policy.net_exposure_gate_mode) == "off_v1"
                else {
                    "net_exposure_gate_mode": str(policy.net_exposure_gate_mode),
                    "gate_floor": float(policy.gate_floor),
                    "gate_trend_lookback_sessions": int(
                        policy.gate_trend_lookback_sessions
                    ),
                }
            ),
            **(
                {}
                if not getattr(policy.lot_sizing, "enabled", False)
                else {
                    "lot_sizing_enabled": True,
                    "lot_sizing_buffer_bps": float(policy.lot_sizing.entry_price_buffer_bps or 0.0),
                    "lot_sizing_opening_gap_quantile": float(policy.lot_sizing.opening_gap_quantile),
                    "lot_sizing_candidate_mode": "causal_lot_waterfill_v1",
                }
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CompoundingPolicyConfig",
    "ExecutionUtility",
    "SizingMethod",
    "StockRiskPolicy",
    "stock_risk_policy_fingerprint",
]
