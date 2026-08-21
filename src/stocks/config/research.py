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
from src.stocks.ml.contracts import PolicyProfile


class ParameterSource(StrEnum):
    PROFILE = "profile"
    CLI = "cli"
    ARTIFACT = "artifact"
    DATASET = "dataset"
    ENVIRONMENT = "environment"


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
    policy_profiles: tuple[PolicyProfile, ...] = (
        PolicyProfile(
            profile_id="legacy_overlay_5bps",
            no_trade_band_bps=5.0,
            growth_risk_aversion=1.0,
        ),
        PolicyProfile(
            profile_id="lower_bound_only",
            no_trade_band_bps=0.0,
            growth_risk_aversion=1.0,
        ),
        PolicyProfile(
            profile_id="lower_bound_half_kelly",
            no_trade_band_bps=0.0,
            growth_risk_aversion=2.0,
        ),
    )

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
