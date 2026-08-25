"""Canonical research configuration and parameter provenance.

``CanonicalResearchProfile`` is the single owner of default statistical and
portfolio values.  CLI defaults are projected from it; they are not copied as
numeric literals.  Every resolved field records one source: ``profile``,
``cli``, ``artifact``, ``dataset``, or ``environment``.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from src.stocks.compatibility import (
    ArtifactContractIdentity,
    parse_execution_utility,
    parse_sizing_method,
)
from src.stocks.ml.contracts import (
    DEFAULT_POLICY_PROFILES,
    EXCESS_FULL_KELLY_PROFILE_ID,
    GROWTH_FULL_UTILIZATION_PROFILE_ID,
    PolicyProfile,
)

__all__ = [
    "EXCESS_FULL_KELLY_PROFILE_ID",
    "GROWTH_FULL_UTILIZATION_PROFILE_ID",
    "CanonicalResearchProfile",
]


class ParameterSource(StrEnum):
    PROFILE = "profile"
    CLI = "cli"
    ARTIFACT = "artifact"
    DATASET = "dataset"
    ENVIRONMENT = "environment"


# Canonical constant lives in contracts; the import above re-exports it for
# config callers.

# Ceiling resolved against the equal-weight basis min(ceiling, 1/top_k) at
# request build time; 0.16 admits the K=8 basis fully while staying finite.
_EXCESS_FULL_KELLY_SINGLE_NAME_CEILING = 0.16
_EXCESS_FULL_KELLY_GROSS_UTILIZATION = 0.92

# Growth-utilization rung: declares its own annual volatility budget so the
# canonical 12% default no longer silently dilutes the utilization target.
_GROWTH_FULL_UTILIZATION_GROSS_UTILIZATION = 0.95
_GROWTH_FULL_UTILIZATION_VOL_TARGET = 0.20

# Declared execution-conversion limits: the canonical 0.5% participation cap
# and 0.20 turnover budget saturate every decision and collapse target
# differences before fills; the rung declares looser pre-registered limits
# while the liquidity slippage model still prices the impact.
_GROWTH_FULL_UTILIZATION_PARTICIPATION = 0.02
_GROWTH_FULL_UTILIZATION_TURNOVER_BUDGET = 0.40


def policy_profiles_with_excess_full_kelly() -> tuple[PolicyProfile, ...]:
    """Opt-in profile ladder adding the excess-full-Kelly rung.

    The default ``CanonicalResearchProfile`` never carries this profile so
    flag-off runs stay byte-identical; requests opt in by replacing their
    ``policy_profiles`` with this tuple. The rung widens its single-name cap
    toward the equal-weight basis, normalizes deployed gross toward 92%, and
    runs the same v5 sparse execution/sizing pair as the canonical defaults so
    cadence scheduling and dense-shadow gating stay uniform across the ladder.
    """
    return (
        *DEFAULT_POLICY_PROFILES,
        PolicyProfile(
            profile_id=EXCESS_FULL_KELLY_PROFILE_ID,
            no_trade_band_bps=0.0,
            growth_risk_aversion=1.0,
            execution_utility_mode="sparse_hold_replace_v2",
            sizing_mode="risk_balanced_waterfill_v2",
            single_name_cap_override=_EXCESS_FULL_KELLY_SINGLE_NAME_CEILING,
            gross_utilization_target=_EXCESS_FULL_KELLY_GROSS_UTILIZATION,
        ),
    )


def policy_profiles_with_growth_rungs() -> tuple[PolicyProfile, ...]:
    """Opt-in ladder adding the excess-full-Kelly and growth-utilization rungs.

    The ``growth_full_utilization`` rung extends the excess-full-Kelly rung
    with a declared 20% annual volatility budget and a 95% gross utilization
    target, so deployed exposure is governed by the declared risk policy
    rather than the canonical 12% default. Flag-off runs stay byte-identical;
    cadence scheduling and dense-shadow gating stay uniform across the ladder.
    """
    return (
        *policy_profiles_with_excess_full_kelly(),
        PolicyProfile(
            profile_id=GROWTH_FULL_UTILIZATION_PROFILE_ID,
            no_trade_band_bps=0.0,
            growth_risk_aversion=1.0,
            execution_utility_mode="sparse_hold_replace_v2",
            sizing_mode="risk_balanced_waterfill_v2",
            single_name_cap_override=_EXCESS_FULL_KELLY_SINGLE_NAME_CEILING,
            gross_utilization_target=_GROWTH_FULL_UTILIZATION_GROSS_UTILIZATION,
            vol_target_override=_GROWTH_FULL_UTILIZATION_VOL_TARGET,
            participation_limit_override=_GROWTH_FULL_UTILIZATION_PARTICIPATION,
            turnover_budget_override=_GROWTH_FULL_UTILIZATION_TURNOVER_BUDGET,
        ),
    )


@dataclass(frozen=True, slots=True)
class CanonicalResearchProfile:
    """Single owner of default statistical and portfolio values.

    This profile replaces the scattered defaults in ``StockAlphaSettings``,
    ``NetAlphaTrainingRequest``, ``SimulationRequest``, and
    ``StockRiskPolicy``.
    """

    top_k: int = 20
    max_single_weight: float = 0.08
    max_exposure: float = 0.90
    participation_limit: float = 0.005
    n_folds: int = 3
    embargo_sessions: int = 5
    candidate_horizon_sessions: tuple[int, ...] = (10, 20)
    # Canonical defaults are the contracts-owned v5 profile ladder so every
    # entry point (flag on/off) shares one mode identity per profile id.
    policy_profiles: tuple[PolicyProfile, ...] = DEFAULT_POLICY_PROFILES

    def fingerprint(self) -> str:
        canonical = json.dumps(
            {
                "top_k": self.top_k,
                "max_single_weight": self.max_single_weight,
                "max_exposure": self.max_exposure,
                "participation_limit": self.participation_limit,
                "n_folds": self.n_folds,
                "embargo_sessions": self.embargo_sessions,
                "candidate_horizon_sessions": list(
                    self.candidate_horizon_sessions
                ),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedField:
    value: object
    source: ParameterSource


@dataclass(frozen=True, slots=True)
class ResolvedResearchConfig:
    """Resolved configuration with provenance for every field."""

    top_k: ResolvedField
    max_single_weight: ResolvedField
    max_exposure: ResolvedField
    participation_limit: ResolvedField
    n_folds: ResolvedField
    embargo_sessions: ResolvedField
    canonical_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "top_k": self.top_k.value,
            "max_single_weight": self.max_single_weight.value,
            "max_exposure": self.max_exposure.value,
            "participation_limit": self.participation_limit.value,
            "n_folds": self.n_folds.value,
            "embargo_sessions": self.embargo_sessions.value,
            "canonical_fingerprint": self.canonical_fingerprint,
        }


def resolve_training_request(
    artifact_id: str,
    *,
    profile: CanonicalResearchProfile | None = None,
    overrides: Mapping[str, object] | None = None,
    costs: Mapping[str, object] | None = None,
) -> ResolvedResearchConfig:
    """Resolve training configuration from profile + CLI overrides.

    Parameters
    ----------
    artifact_id:
        Artifact identifier for this training run.
    profile:
        Canonical profile to use as defaults.  Falls back to
        ``CanonicalResearchProfile()``.
    overrides:
        Explicit CLI overrides.  Each key must match a profile field or
        ``ValueError`` is raised.
    costs:
        Cost schedule parameters (reserved for future use).

    Returns
    -------
    ResolvedResearchConfig
        Fully resolved configuration with provenance.
    """
    if profile is None:
        profile = CanonicalResearchProfile()
    if overrides is None:
        overrides = {}

    _profile_fields = {
        "top_k",
        "max_single_weight",
        "max_exposure",
        "participation_limit",
        "n_folds",
        "embargo_sessions",
    }

    unknown = set(overrides.keys()) - _profile_fields
    if unknown:
        raise ValueError(
            f"Unknown override fields: {unknown}; "
            f"allowed: {_profile_fields}"
        )

    def _resolve(
        field_name: str, default: object, profile_val: object
    ) -> ResolvedField:
        if field_name in overrides:
            return ResolvedField(
                value=overrides[field_name], source=ParameterSource.CLI
            )
        return ResolvedField(value=profile_val, source=ParameterSource.PROFILE)

    resolved = ResolvedResearchConfig(
        top_k=_resolve("top_k", profile.top_k, profile.top_k),
        max_single_weight=_resolve(
            "max_single_weight",
            profile.max_single_weight,
            profile.max_single_weight,
        ),
        max_exposure=_resolve(
            "max_exposure", profile.max_exposure, profile.max_exposure
        ),
        participation_limit=_resolve(
            "participation_limit",
            profile.participation_limit,
            profile.participation_limit,
        ),
        n_folds=_resolve("n_folds", profile.n_folds, profile.n_folds),
        embargo_sessions=_resolve(
            "embargo_sessions",
            profile.embargo_sessions,
            profile.embargo_sessions,
        ),
        canonical_fingerprint=profile.fingerprint(),
    )
    return resolved


def resolve_simulation_request(
    artifact: Mapping[str, object] | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve simulation configuration from artifact identity + overrides.

    Parameters
    ----------
    artifact:
        Model manifest mapping.  Simulation policy fields default to ``None``
        and resolve from the artifact.
    overrides:
        Explicit CLI overrides.  Each must match the artifact fingerprint or
        ``ValueError`` is raised.

    Returns
    -------
    dict
        Resolved simulation configuration.
    """
    if artifact is None:
        artifact = {}
    if overrides is None:
        overrides = {}

    result: dict[str, object] = {
        "top_k": artifact.get("top_k"),
        "max_single_weight": artifact.get("max_single_weight"),
        "max_exposure": artifact.get("max_exposure"),
        "participation_limit": artifact.get("participation_limit"),
    }

    if {"contract_id", "schema_revision", "fingerprint"} <= artifact.keys():
        ArtifactContractIdentity(
            contract_id=str(artifact["contract_id"]),
            schema_revision=cast(int, artifact["schema_revision"]),
            fingerprint=str(artifact["fingerprint"]),
        )
    if "execution_utility" in artifact:
        parse_execution_utility(str(artifact["execution_utility"]))
    if "sizing_method" in artifact:
        parse_sizing_method(str(artifact["sizing_method"]))

    for key, val in overrides.items():
        if key in result:
            result[key] = val

    return result
