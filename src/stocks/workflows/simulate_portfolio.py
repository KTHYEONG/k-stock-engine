"""Stock portfolio-simulation workflow: artifact -> cycle -> replay ledger.

The simulation workflow replays the *same* pure planner used by paper and live
paths through ``StockBacktester``, so a historical replay step and a paper
planning cycle produce identical target allocations for identical inputs.

Policy equivalence: the trained artifact's selected ``policy_profile`` (id,
no-trade band, and portfolio fingerprint) is the single source of truth. The
backtester is always constructed from that profile, and a user request that
explicitly diverges from it (different profile id, different band, or a
different top-k/single-name/gross/participation fingerprint) raises
``ValueError`` so the independent backtest can never silently use a different
top-k/exposure/band than the OOF that selected the policy.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Literal, cast

import polars as pl

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.engine import (
    ArtifactSchedule,
    ArtifactSlot,
    BacktestRequest,
    BacktestResult,
    StockBacktester,
)
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.data.costs import CostEvidence
from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1
from src.stocks.ml.contracts import policy_portfolio_fingerprint
from src.stocks.ml.features import feature_transform_schema_from_manifest
from src.stocks.observability.contracts import RunDiagnostics
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.models import ModelManifest
from src.stocks.trading.portfolio_constructor import (
    CompoundingPolicyConfig,
    StockRiskPolicy,
    stock_risk_policy_fingerprint,
)
from src.stocks.trading.rebalance_schedule import rebalance_session_indices
from src.stocks.workflows.contracts import SimulationRequest


def simulate_portfolio(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    request: SimulationRequest,
    cost_evidence: CostEvidence | None = None,
    diagnostics: RunDiagnostics | None = None,
) -> BacktestResult:
    """Replay the trading cycle over the snapshot and return the ledger result.

    The ``StockRiskPolicy`` is always constructed from the artifact's selected
    ``policy_profile`` when the artifact carries one. A divergent explicit
    request (profile id, no-trade band, or portfolio fingerprint) raises
    ``ValueError``. ``cost_evidence`` is the hash-bound cost artifact resolved
    from the research snapshot; when supplied the replay uses the dynamic
    liquidity slippage model and statutory sell taxes instead of the static
    base/stress schedules.
    """
    manifest = snapshot.manifest
    artifact_manifest = registry.read_manifest(request.artifact_id)
    eligible_from = datetime.fromisoformat(artifact_manifest.eligible_from)
    eligible_to = datetime.fromisoformat(artifact_manifest.eligible_to)

    policy = _policy_from_artifact(artifact_manifest, request)
    _validate_artifact_input_lineage(snapshot, artifact_manifest)

    frame = snapshot.frame
    if "adtv" not in frame.columns and "adtv_20d" in frame.columns:
        frame = frame.with_columns(pl.col("adtv_20d").alias("adtv"))
    sessions = sorted(frame["session"].unique().to_list())
    instruments = _instruments_from_frame(frame)
    base = request.cost_schedule or default_base_schedule()
    stress = request.stress_cost_schedule or default_stress_schedule()
    profile_payload = _parse_policy_profile(artifact_manifest)
    is_legacy_daily = not (
        profile_payload is not None
        and profile_payload.get("execution_evidence_version")
        == "prepared-equity-v5-sparse-growth"
    )
    decision_indices = rebalance_session_indices(
        sessions,
        eligible_from,
        eligible_to,
        policy.rebalance_frequency_sessions,
        legacy_daily=is_legacy_daily,
    )

    initial_portfolio = PortfolioSnapshot(
        account_snapshot_id="backtest",
        as_of=datetime(2000, 1, 1, tzinfo=UTC),
        settled_cash=request.initial_cash,
        unsettled_cash=0.0,
        positions=(),
    )
    backtest_request = BacktestRequest(
        strategy_id="stock_alpha_v1",
        start_time=eligible_from,
        end_time=eligible_to,
        decision_session_indices=decision_indices,
        cost_schedule=base,
        stress_cost_schedule=stress,
        risk_policy=policy,
    )
    artifacts = ArtifactSchedule(
        slots=(
            ArtifactSlot(
                eligible_from=eligible_from,
                eligible_to=eligible_to,
                artifact_id=request.artifact_id,
            ),
        )
    )
    backtester = StockBacktester(diagnostics=diagnostics,
        registry=registry,
        instruments=instruments,
        manifest=manifest,
        cost_schedule=base,
        stress_cost_schedule=stress,
        cost_evidence=cost_evidence,
    )
    return backtester.run(
        frame, artifacts, initial_portfolio, backtest_request
    )


def _validate_artifact_input_lineage(
    snapshot: DatasetSnapshot,
    artifact_manifest: ModelManifest,
) -> None:
    """Fail closed when a v6 artifact's exact input lineage does not match.

    A v6 artifact (``holm_gate_version == "v6"``) records the raw feature schema
    hash, the input feature content hash, and the recomputed transform
    fingerprint. The independent historical replay must bind to the identical
    feature dataset; a divergent feature schema, content hash, or transform
    fingerprint raises ``ValueError`` with a diagnostic naming the differing
    hashes so an unrelated snapshot can never silently replay a v6 artifact.
    Legacy (pre-v6) artifacts skip this gate and replay under best-effort
    compatibility.
    """
    params = artifact_manifest.params or {}
    if params.get("holm_gate_version") != "v6":
        return
    snapshot_manifest = snapshot.manifest
    raw_schema_hash = params.get("raw_feature_schema_hash")
    feature_content_hash = params.get("feature_content_hash")
    stored_fingerprint = params.get("feature_transform_fingerprint")
    mismatches: list[str] = []
    if (
        raw_schema_hash is not None
        and snapshot_manifest.schema_hash != raw_schema_hash
    ):
        mismatches.append(
            f"feature_schema_hash snapshot={snapshot_manifest.schema_hash!r} "
            f"artifact={raw_schema_hash!r}"
        )
    if (
        feature_content_hash is not None
        and snapshot_manifest.content_hash
        and feature_content_hash != snapshot_manifest.content_hash
    ):
        mismatches.append(
            f"feature_content_hash snapshot={snapshot_manifest.content_hash!r} "
            f"artifact={feature_content_hash!r}"
        )
    if "feature_transform_schema" in params and stored_fingerprint is not None:
        try:
            schema = feature_transform_schema_from_manifest(artifact_manifest)
        except ValueError as exc:
            raise ValueError(
                f"v6 artifact {artifact_manifest.artifact_id!r} transform "
                f"schema is invalid: {exc}"
            ) from exc
        if schema.fingerprint != stored_fingerprint:
            mismatches.append(
                f"transform_fingerprint recomputed={schema.fingerprint!r} "
                f"artifact={stored_fingerprint!r}"
            )
    if mismatches:
        raise ValueError(
            "v6 independent replay input lineage mismatch for "
            f"{artifact_manifest.artifact_id!r}: " + "; ".join(mismatches)
        )


def artifact_policy_profile(
    registry: ModelArtifactRegistry, artifact_id: str
) -> dict[str, object] | None:
    """Return the artifact's selected ``policy_profile`` params, or ``None``.

    ``None`` covers ``NO_TRADE`` and legacy artifacts that never recorded a
    selected profile. Callers (e.g. the simulate CLI) use the returned profile
    to build an exactly matching ``SimulationRequest``.
    """
    return _parse_policy_profile(registry.read_manifest(artifact_id))


def _parse_policy_profile(
    artifact_manifest: ModelManifest,
) -> dict[str, object] | None:
    """Parse the immutable ``policy_profile`` params, or ``None`` when absent."""
    raw = artifact_manifest.params or {}
    payload = raw.get("policy_profile")
    if not payload:
        return None
    if not isinstance(payload, str):
        raise ValueError("manifest policy_profile must be a JSON string")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest policy_profile is malformed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("manifest policy_profile must be a JSON object")
    return parsed


def _profile_growth_risk_aversion(profile: dict[str, object]) -> float:
    """Extract growth_risk_aversion from a policy profile, defaulting to 1.0.

    Existing payloads lacking growth_risk_aversion default to 1.0 for
    backward-compatible replay; they are never rewritten.
    """
    raw = profile.get("growth_risk_aversion")
    if raw is None:
        return 1.0
    if not isinstance(raw, (int, float)):
        raise ValueError(
            "growth_risk_aversion must be a finite strictly positive number, "
            f"got {type(raw).__name__}"
        )
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "growth_risk_aversion must be finite and strictly positive, "
            f"got {value!r}"
        )
    return value


def _profile_economic_ranking_mode(
    profile: dict[str, object],
) -> Literal["raw_score_v1", "economic_net_v1"]:
    """Extract economic_ranking_mode from a policy profile, defaulting to raw_score_v1.

    Existing payloads lacking economic_ranking_mode default to raw_score_v1
    for backward-compatible replay of legacy v1 artifacts.
    """
    raw = profile.get("economic_ranking_mode")
    if raw is None:
        if profile.get("execution_evidence_version") == "prepared-equity-v2-economic-rank":
            raise ValueError(
                "economic_ranking_mode is required for prepared-equity-v2-economic-rank"
            )
        return "raw_score_v1"
    if not isinstance(raw, str):
        raise ValueError(
            "economic_ranking_mode must be a string, "
            f"got {type(raw).__name__}"
        )
    if raw not in ("raw_score_v1", "economic_net_v1"):
        raise ValueError(
            f"economic_ranking_mode must be 'raw_score_v1' or 'economic_net_v1', "
            f"got {raw!r}"
        )
    return cast(Literal["raw_score_v1", "economic_net_v1"], raw)


def _profile_forecast_horizon_sessions(profile: dict[str, object]) -> int | None:
    """Extract forecast_horizon_sessions from a policy profile.

    Existing payloads lacking forecast_horizon_sessions return ``None`` for
    backward-compatible v2 replay. A v3 artifact with the field present must
    carry a positive integer; non-integer or non-positive values raise
    ``ValueError``.
    """
    raw = profile.get("forecast_horizon_sessions")
    if raw is None:
        return None
    if not isinstance(raw, int) or raw < 1:
        raise ValueError(
            "forecast_horizon_sessions must be a positive integer, "
            f"got {raw!r}"
        )
    return raw


def _profile_sizing_mode(
    profile: dict[str, object],
) -> Literal[
    "alpha_vol_squared_v1",
    "risk_balanced_waterfill_v2",
    "confidence_mean_variance_v1",
]:
    """Extract sizing_mode from a policy profile, defaulting to alpha_vol_squared_v1.

    Existing payloads lacking sizing_mode default to alpha_vol_squared_v1
    for backward-compatible replay of legacy v1-v4 artifacts.
    """
    raw = profile.get("sizing_mode")
    if raw is None:
        return "alpha_vol_squared_v1"
    if not isinstance(raw, str):
        raise ValueError(
            "sizing_mode must be a string, "
            f"got {type(raw).__name__}"
        )
    if raw not in (
        "alpha_vol_squared_v1",
        "risk_balanced_waterfill_v2",
        "confidence_mean_variance_v1",
    ):
        raise ValueError(
            f"sizing_mode must be 'alpha_vol_squared_v1', "
            f"'risk_balanced_waterfill_v2', or 'confidence_mean_variance_v1', got {raw!r}"
        )
    return cast(
        Literal[
            "alpha_vol_squared_v1",
            "risk_balanced_waterfill_v2",
            "confidence_mean_variance_v1",
        ],
        raw,
    )


def _profile_rebalance_frequency_sessions(profile: dict[str, object]) -> int:
    """Extract rebalance_frequency_sessions from a policy profile, defaulting to 5.

    v5 artifacts persist the cadence as ``rebalance_frequency_sessions``.
    Missing values default to 5 for legacy artifacts. Non-integral or
    non-positive values raise ``ValueError``.
    """
    raw = profile.get("rebalance_frequency_sessions")
    if raw is None:
        return 5
    if not isinstance(raw, int) or raw < 1:
        raise ValueError(
            "rebalance_frequency_sessions must be a positive integer, "
            f"got {raw!r}"
        )
    return raw


def _profile_execution_utility_mode(
    profile: dict[str, object],
) -> Literal["legacy_target_interpolation_v1", "delta_cost_aware_v1", "sparse_hold_replace_v2"]:
    """Extract execution_utility_mode from a policy profile, defaulting to legacy.

    Existing payloads lacking execution_utility_mode default to
    legacy_target_interpolation_v1 for backward-compatible replay.
    v4 artifacts with an invalid mode raise ``ValueError``.
    """
    raw = profile.get("execution_utility_mode")
    if raw is None:
        evidence_version = profile.get("execution_evidence_version", "")
        if evidence_version == "prepared-equity-v4-delta-cost-aware":
            raise ValueError(
                "execution_utility_mode is required for "
                "prepared-equity-v4-delta-cost-aware"
            )
        return "legacy_target_interpolation_v1"
    if not isinstance(raw, str):
        raise ValueError(
            "execution_utility_mode must be a string, "
            f"got {type(raw).__name__}"
        )
    if raw not in ("legacy_target_interpolation_v1", "delta_cost_aware_v1", "sparse_hold_replace_v2"):
        raise ValueError(
            "execution_utility_mode must be 'legacy_target_interpolation_v1', "
            "'delta_cost_aware_v1', or 'sparse_hold_replace_v2', "
            f"got {raw!r}"
        )
    return cast(
        Literal["legacy_target_interpolation_v1", "delta_cost_aware_v1", "sparse_hold_replace_v2"], raw
    )


def _profile_retained_sizing_mode(
    profile: dict[str, object],
) -> Literal["freeze_v1", "band_limited_rewaterfill_v1"]:
    """Extract retained_sizing_mode from a policy profile, defaulting to freeze.

    Payloads persisted before the re-waterfill opt-in lack the key and
    reconstruct as ``freeze_v1``. Invalid values raise ``ValueError``.
    """
    raw = profile.get("retained_sizing_mode")
    if raw is None:
        return "freeze_v1"
    if not isinstance(raw, str):
        raise ValueError(
            "retained_sizing_mode must be a string, "
            f"got {type(raw).__name__}"
        )
    if raw not in ("freeze_v1", "band_limited_rewaterfill_v1"):
        raise ValueError(
            "retained_sizing_mode must be 'freeze_v1' or "
            "'band_limited_rewaterfill_v1', "
            f"got {raw!r}"
        )
    return cast(
        Literal["freeze_v1", "band_limited_rewaterfill_v1"], raw
    )


def _policy_from_artifact(
    artifact_manifest: ModelManifest, request: SimulationRequest
) -> StockRiskPolicy:
    """Construct the operational policy from the artifact's selected profile.

    When the artifact carries no ``policy_profile`` (e.g. a ``NO_TRADE`` or
    legacy artifact), the request's own caps are used unchanged. When it does,
    the artifact profile is the single source of truth and any divergent
    explicit request raises ``ValueError``.
    """
    profile = _parse_policy_profile(artifact_manifest)
    if profile is None:
        return StockRiskPolicy(
            top_k=request.top_k,
            gross_cap=request.max_exposure,
            single_name_cap=request.max_single_weight,
            participation_limit=request.participation_limit,
            no_trade_band_bps=request.no_trade_band_bps or 0.0,
        )
    _validate_request_policy(request, profile)
    evidence_version = profile.get("execution_evidence_version")
    if evidence_version == "prepared-equity-v7-horizon-locked":
        return _reconstruct_v7_policy(profile, request)
    aversion = _profile_growth_risk_aversion(profile)
    ranking_mode = _profile_economic_ranking_mode(profile)
    horizon = _profile_forecast_horizon_sessions(profile)
    mode = _profile_execution_utility_mode(profile)
    sizing = _profile_sizing_mode(profile)
    stored_cadence = _profile_rebalance_frequency_sessions(profile)
    return StockRiskPolicy(
        top_k=cast(int, profile["top_k"]),
        gross_cap=cast(float, profile["max_exposure"]),
        single_name_cap=cast(float, profile["max_single_weight"]),
        participation_limit=cast(float, profile["participation_limit"]),
        no_trade_band_bps=cast(float, profile["no_trade_band_bps"]),
        rebalance_frequency_sessions=stored_cadence,
        compounding=CompoundingPolicyConfig(
            growth_risk_aversion=aversion,
            forecast_horizon_sessions=horizon,
        ),
        economic_ranking_mode=ranking_mode,
        execution_utility_mode=mode,
        sizing_mode=sizing,
        retained_sizing_mode=_profile_retained_sizing_mode(profile),
    )


def _reconstruct_v7_policy(
    profile: dict[str, object], request: SimulationRequest
) -> StockRiskPolicy:
    """Reconstruct the horizon-locked v7 operational policy from its profile.

    The v7 route pins the rebalance cadence to the forecast horizon, derives the
    effective active count and candidate pool from the gross/single-name caps,
    and reconstructs the same policy fingerprint the prepared replay and score
    workflow used. A divergent capped active count, candidate pool, horizon
    cadence, or stored v7 fingerprint fails closed before the backtester runs.
    """
    aversion = _profile_growth_risk_aversion(profile)
    ranking_mode = _profile_economic_ranking_mode(profile)
    horizon = _profile_forecast_horizon_sessions(profile)
    mode = _profile_execution_utility_mode(profile)
    sizing = _profile_sizing_mode(profile)
    if horizon is None:
        raise ValueError("v7 horizon-locked policy requires forecast_horizon_sessions")
    if profile.get("rebalance_frequency_sessions") != horizon:
        raise ValueError(
            "v7 rebalance_frequency_sessions must equal forecast_horizon_sessions"
        )
    policy = StockRiskPolicy(
        top_k=cast(int, profile["top_k"]),
        gross_cap=cast(float, profile["max_exposure"]),
        single_name_cap=cast(float, profile["max_single_weight"]),
        participation_limit=cast(float, profile["participation_limit"]),
        no_trade_band_bps=cast(float, profile["no_trade_band_bps"]),
        rebalance_frequency_sessions=horizon,
        compounding=CompoundingPolicyConfig(
            growth_risk_aversion=aversion,
            forecast_horizon_sessions=horizon,
        ),
        economic_ranking_mode=ranking_mode,
        execution_utility_mode=mode,
        sizing_mode=sizing,
        retained_sizing_mode=_profile_retained_sizing_mode(profile),
    )
    _validate_v7_policy_profile(profile, policy)
    return policy


def _as_int(value: object) -> int | None:
    """Return ``value`` as an int when it is integral, else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _validate_v7_policy_profile(
    profile: dict[str, object], policy: StockRiskPolicy
) -> None:
    """Fail closed when the v7 derived caps or fingerprint diverge."""
    effective = profile.get("effective_active_count")
    if effective is not None:
        expected_effective = math.ceil(
            policy.gross_cap / policy.single_name_cap
        )
        effective_int = _as_int(effective)
        if effective_int is None or effective_int != expected_effective:
            raise ValueError(
                "v7 effective_active_count "
                f"{effective} diverges from ceil(gross/single)={expected_effective}"
            )
        candidate_pool = profile.get("candidate_pool_count")
        candidate_int = _as_int(candidate_pool)
        if candidate_pool is not None and (
            candidate_int is None or candidate_int != 2 * effective_int
        ):
            raise ValueError(
                "v7 candidate_pool_count "
                f"{candidate_pool} must be 2x effective_active_count {effective}"
            )
    v7_fingerprint = profile.get("v7_risk_policy_fingerprint")
    if v7_fingerprint is not None and stock_risk_policy_fingerprint(policy) != v7_fingerprint:
        raise ValueError(
            "v7 independent replay risk-policy fingerprint diverges from the "
            "artifact policy; the independent backtester must replay under the "
            "same horizon-locked policy that certified the artifact"
        )


def _validate_request_policy(
    request: SimulationRequest, profile: dict[str, object]
) -> None:
    """Fail closed when an explicit request diverges from the artifact profile."""
    if (
        request.policy_profile_id is not None
        and request.policy_profile_id != profile["profile_id"]
    ):
        raise ValueError(
            "portfolio simulation policy_profile_id "
            f"{request.policy_profile_id!r} diverges from the artifact profile "
            f"{profile['profile_id']!r}"
        )
    artifact_band = cast(float, profile["no_trade_band_bps"])
    if (
        request.no_trade_band_bps is not None
        and abs(request.no_trade_band_bps - artifact_band) > 1e-12
    ):
        raise ValueError(
            "portfolio simulation no_trade_band_bps "
            f"{request.no_trade_band_bps} diverges from the artifact band "
            f"{artifact_band}"
        )
    evidence_version = profile.get("execution_evidence_version")
    if evidence_version == "prepared-equity-v7-horizon-locked":
        if _profile_forecast_horizon_sessions(profile) is None:
            raise ValueError(
                "v7 horizon-locked policy requires forecast_horizon_sessions"
            )
        if (
            profile.get("rebalance_frequency_sessions")
            != _profile_forecast_horizon_sessions(profile)
        ):
            raise ValueError(
                "v7 rebalance_frequency_sessions must equal "
                "forecast_horizon_sessions"
            )
        v7_fingerprint = profile.get("v7_risk_policy_fingerprint")
        if v7_fingerprint is not None:
            policy = _reconstruct_v7_policy(profile, request)
            if stock_risk_policy_fingerprint(policy) != v7_fingerprint:
                raise ValueError(
                    "v7 independent replay risk-policy fingerprint diverges"
                )
        return
    if evidence_version in (
        "prepared-equity-v1",
        "prepared-equity-v2-economic-rank",
    ):
        _validate_prepared_equity_policy(request, profile, artifact_band)
        return
    if evidence_version == "prepared-equity-v3-horizon-consistent":
        if _profile_forecast_horizon_sessions(profile) is None:
            raise ValueError(
                "v3 horizon-consistent policy requires forecast_horizon_sessions"
            )
        _validate_prepared_equity_policy(request, profile, artifact_band)
        return
    if evidence_version == "prepared-equity-v4-delta-cost-aware":
        if _profile_execution_utility_mode(profile) != "delta_cost_aware_v1":
            raise ValueError(
                "v4 delta-cost-aware policy requires execution_utility_mode "
                "'delta_cost_aware_v1'"
            )
        _validate_prepared_equity_policy(request, profile, artifact_band)
        return
    if evidence_version == "prepared-equity-v5-sparse-growth":
        mode = _profile_execution_utility_mode(profile)
        sizing = _profile_sizing_mode(profile)
        if mode != "sparse_hold_replace_v2":
            raise ValueError(
                "v5 sparse-growth policy requires execution_utility_mode "
                "'sparse_hold_replace_v2'"
            )
        if sizing != "risk_balanced_waterfill_v2":
            raise ValueError(
                "v5 sparse-growth policy requires sizing_mode "
                "'risk_balanced_waterfill_v2'"
            )
        _validate_prepared_equity_policy(request, profile, artifact_band)
        return
    request_fingerprint = policy_portfolio_fingerprint(
        request.top_k,
        request.max_single_weight,
        request.max_exposure,
        request.participation_limit,
    )
    if request_fingerprint != profile["portfolio_fingerprint"]:
        raise ValueError(
            "portfolio simulation portfolio caps diverge from the artifact "
            "policy profile; the independent backtester must use the same "
            "top-k/max_single_weight/max_exposure/participation_limit as the "
            "OOF that selected the policy"
        )


def _validate_prepared_equity_policy(
    request: SimulationRequest,
    profile: dict[str, object],
    artifact_band: float,
) -> None:
    """Fail closed when a prepared-equity artifact's stored fingerprints diverge.

    The request-reconstructed risk policy and the default execution policy must
    match the artifact's ``risk_policy_fingerprint`` and
    ``execution_policy_hash``; otherwise the independent backtester could
    silently replay under a different policy than the one that certified the
    artifact.  For v5 artifacts, the stored cadence is also reconstructed so
    that a missing/non-integral/mismatched cadence raises ``ValueError``
    before replay.
    """
    aversion = _profile_growth_risk_aversion(profile)
    ranking_mode = _profile_economic_ranking_mode(profile)
    horizon = _profile_forecast_horizon_sessions(profile)
    mode = _profile_execution_utility_mode(profile)
    sizing = _profile_sizing_mode(profile)
    stored_cadence = _profile_rebalance_frequency_sessions(profile)
    policy = StockRiskPolicy(
        top_k=request.top_k,
        gross_cap=request.max_exposure,
        single_name_cap=request.max_single_weight,
        participation_limit=request.participation_limit,
        no_trade_band_bps=artifact_band,
        rebalance_frequency_sessions=stored_cadence,
        compounding=CompoundingPolicyConfig(
            growth_risk_aversion=aversion,
            forecast_horizon_sessions=horizon,
        ),
        economic_ranking_mode=ranking_mode,
        execution_utility_mode=mode,
        sizing_mode=sizing,
        retained_sizing_mode=_profile_retained_sizing_mode(profile),
    )
    if stock_risk_policy_fingerprint(policy) != profile["risk_policy_fingerprint"]:
        raise ValueError(
            "portfolio simulation risk-policy fingerprint diverges from the "
            "artifact policy; the independent backtester must replay under the "
            "same risk policy that certified the artifact"
        )
    if SCHEDULED_OPEN_V1.canonical_hash != profile["execution_policy_hash"]:
        raise ValueError(
            "portfolio simulation execution-policy hash diverges from the "
            "artifact policy; the independent backtester must replay under the "
            "same execution policy that certified the artifact"
        )


def _decision_indices(
    sessions: list[object],
    eligible_from: datetime,
    eligible_to: datetime,
) -> tuple[int, ...]:
    indices: list[int] = []
    for index, session in enumerate(sessions):
        if eligible_from <= cast(datetime, session) <= eligible_to:
            indices.append(index)
    return tuple(indices)


def _instruments_from_frame(frame: pl.DataFrame) -> dict[str, Instrument]:
    instruments: dict[str, Instrument] = {}
    for row in frame.select("instrument_id").unique().iter_rows(named=True):
        instrument_id = str(row["instrument_id"])
        instruments[instrument_id] = Instrument(
            instrument_id=instrument_id,
            asset_kind=AssetKind.STOCK,
            exchange="KRX",
            symbol=instrument_id.split(":")[-1],
            currency="KRW",
        )
    return instruments
