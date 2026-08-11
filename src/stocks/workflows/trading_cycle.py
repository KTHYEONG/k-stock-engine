"""Pure main trading-cycle planner shared by backtest, paper, and live paths.

``run_trading_cycle`` is pure with respect to external systems: it consumes a
validated dataset snapshot, a promoted artifact registry, a canonical instrument
mapping, and a reconciled account snapshot, and returns an immutable
``TradingCycleResult``. No broker call, filesystem write, or network I/O occurs
here; the CLI/scheduler persists the result and may hand intents to the
execution readiness gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256

import polars as pl

from src.core.datasets import (
    DatasetCertification,
    DatasetManifest,
    validate_production_manifest,
)
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation, PortfolioSnapshot
from src.execution.domain.intents import TradeIntent
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry, PredictionRequest
from src.stocks.research.datasets import research_eligible_frame, validate_stock_rows_available
from src.stocks.research.features import build_features, phase1_allowlist
from src.stocks.trading.portfolio_constructor import (
    PortfolioConstraintError,
    StockRiskPolicy,
    construct_target_allocations,
)

_VALID_MODES = ("plan", "paper", "live")


class CycleStatus(StrEnum):
    """Deterministic outcome of one planning cycle."""

    PLANNED = "PLANNED"
    NO_TRADE = "NO_TRADE"
    DE_RISK = "DE_RISK"
    HALT = "HALT"


class TradingCycleNotReadyError(RuntimeError):
    """Raised when live-mode evidence is incomplete or an input is not promoted."""


@dataclass(frozen=True, slots=True)
class TradingCycleRequest:
    """Immutable input contract binding one planning cycle.

    ``mode`` controls certification/readiness validation only; the planner
    remains side-effect free in every mode. Live mode has no permissive
    defaults: missing production evidence raises ``TradingCycleNotReadyError``.
    """

    strategy_id: str
    artifact_id: str
    dataset_id: str
    decision_time: datetime
    execution_time: datetime
    risk_policy: StockRiskPolicy
    mode: str = "plan"

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must be non-empty")
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if self.decision_time.tzinfo is None or self.execution_time.tzinfo is None:
            raise ValueError("decision_time and execution_time must be timezone-aware")
        if self.decision_time >= self.execution_time:
            raise ValueError("decision_time must be before execution_time")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"invalid mode {self.mode!r}; expected one of {_VALID_MODES}")


@dataclass(frozen=True, slots=True)
class TradingCycleResult:
    """Immutable outcome of one planning cycle.

    ``allocations`` and ``intents`` are target positions (non-negative desired
    notionals), never signed order deltas. Buy/sell/exit sides are derived at
    execution from reconciled broker state.
    """

    status: CycleStatus
    cycle_id: str
    decision_time: datetime
    dataset_hash: str
    artifact_id: str
    account_snapshot_id: str
    allocations: tuple[Allocation, ...]
    intents: tuple[TradeIntent, ...]
    selected_instruments: tuple[str, ...]
    reasons: tuple[str, ...]
    universe_hash: str = ""
    feature_hash: str = ""
    label_hash: str = ""
    cost_hash: str = ""
    risk_policy_hash: str = ""

    @property
    def no_intents(self) -> bool:
        return not self.intents


def run_trading_cycle(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    instruments: Mapping[str, Instrument],
    portfolio: PortfolioSnapshot,
    request: TradingCycleRequest,
) -> TradingCycleResult:
    """Run one deterministic planning cycle in the fixed step order."""
    manifest = snapshot.manifest
    _validate_mode_evidence(manifest, request.mode)

    visible = snapshot.frame.filter(pl.col("available_time") <= request.decision_time)
    dataset_hash = _snapshot_hash(visible, manifest, request)
    if visible.is_empty():
        return _no_trade_result(
            request, manifest, portfolio, dataset_hash, "no-rows-available-at-decision-time"
        )
    validate_stock_rows_available(visible, request.decision_time)
    portfolio.validate_as_of(request.decision_time)

    prediction = PredictionRequest(
        asset_kind=AssetKind.STOCK,
        feature_set=manifest.feature_set,
        feature_schema_hash=manifest.schema_hash,
        decision_time=request.decision_time,
    )
    loaded = registry.load(request.artifact_id, prediction)
    if request.mode in ("paper", "live") and not registry.is_promoted(request.artifact_id):
        raise TradingCycleNotReadyError(
            f"artifact {request.artifact_id!r} has no promoted=true evidence"
        )

    universe_gate = _universe_gate(visible)
    if universe_gate.is_empty():
        return _no_trade_result(request, manifest, portfolio, dataset_hash, "empty-universe-after-gate")

    gated = _drop_label_columns(research_eligible_frame(universe_gate))
    feature_frame = (
        gated
        if manifest.feature_set == "stock_alpha_v2"
        else build_features(gated, phase1_allowlist())
    )
    scored = loaded.model.predict(feature_frame)
    if scored.is_empty():
        return _no_trade_result(request, manifest, portfolio, dataset_hash, "empty-scored-panel")

    if getattr(loaded.model, "no_trade", False):
        return _no_trade_result(request, manifest, portfolio, dataset_hash, "no-trade-artifact")

    latest = scored.select(pl.col("session").max()).to_series()[0]
    cross_section = scored.filter(pl.col("session") == latest)
    if cross_section.is_empty():
        return _no_trade_result(request, manifest, portfolio, dataset_hash, "empty-latest-cross-section")

    try:
        allocations = construct_target_allocations(
            scored, instruments, portfolio, request.risk_policy
        )
    except PortfolioConstraintError as exc:
        return _no_trade_result(request, manifest, portfolio, dataset_hash, f"constraint:{exc}")

    if not allocations:
        return _no_trade_result(request, manifest, portfolio, dataset_hash, "no-feasible-allocation")

    intents = _build_intents(allocations, portfolio, request)
    fingerprints = _fingerprints(manifest, request)
    cycle_id = _cycle_id(request, manifest, portfolio, dataset_hash)

    selected = tuple(
        sorted({a.instrument.instrument_id for a in allocations})
    )
    de_risk = any(a.reason == "de-risk-sell-only" for a in allocations)
    reasons = (
        f"certification={manifest.certification.value}",
        f"cross_section_session={latest.isoformat()}",
        f"coverage={len(cross_section)}",
        "constraints=de-risk" if de_risk else "constraints=ok",
    )
    return TradingCycleResult(
        status=CycleStatus.DE_RISK if de_risk else CycleStatus.PLANNED,
        cycle_id=cycle_id,
        decision_time=request.decision_time,
        dataset_hash=dataset_hash,
        artifact_id=request.artifact_id,
        account_snapshot_id=portfolio.account_snapshot_id,
        allocations=tuple(sorted(allocations, key=lambda a: a.instrument.instrument_id)),
        intents=tuple(sorted(intents, key=lambda i: i.instrument_id)),
        selected_instruments=selected,
        reasons=reasons,
        universe_hash=fingerprints["universe_hash"],
        feature_hash=fingerprints["feature_hash"],
        label_hash=fingerprints["label_hash"],
        cost_hash=fingerprints["cost_hash"],
        risk_policy_hash=fingerprints["risk_policy_hash"],
    )


def _validate_mode_evidence(manifest: DatasetManifest, mode: str) -> None:
    if mode == "plan":
        return
    if mode == "paper" and manifest.certification is DatasetCertification.PROVISIONAL:
        raise TradingCycleNotReadyError(
            "paper mode requires RESEARCH or PRODUCTION certification; "
            f"snapshot is {manifest.certification.value}"
        )
    if mode == "live":
        if manifest.certification is not DatasetCertification.PRODUCTION:
            raise TradingCycleNotReadyError(
                "live mode requires PRODUCTION certification; "
                f"snapshot is {manifest.certification.value}"
            )
        validate_production_manifest(manifest)


def _universe_gate(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the point-in-time universe and tradability gate when present."""
    gated = frame
    if "data_quality_status" in gated.columns:
        gated = gated.filter(pl.col("data_quality_status") == "eligible")
    if "is_universe" in gated.columns:
        gated = gated.filter(pl.col("is_universe"))
    if "tradable" in gated.columns:
        gated = gated.filter(pl.col("tradable"))
    return gated


def _drop_label_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove canonical label columns so scoring never observes a label."""
    from src.stocks.research.labels import (
        LABEL_AVAILABLE_COLUMN,
        RELEVANCE_COLUMN,
        RESIDUAL_O2O_LABEL,
    )

    drops = [
        c
        for c in frame.columns
        if c.startswith(("target_", "label_", "residual_"))
        or c in (LABEL_AVAILABLE_COLUMN, RELEVANCE_COLUMN, RESIDUAL_O2O_LABEL, "fwd_ret_5d")
    ]
    return frame.drop(drops)


def _build_intents(
    allocations: tuple[Allocation, ...],
    portfolio: PortfolioSnapshot,
    request: TradingCycleRequest,
) -> tuple[TradeIntent, ...]:
    targeted = {a.instrument.instrument_id for a in allocations}
    intents: list[TradeIntent] = []
    for index, allocation in enumerate(allocations):
        intents.append(
            TradeIntent(
                intent_id=f"{request.strategy_id}:{allocation.instrument.instrument_id}:{request.decision_time.isoformat()}:{index}",
                asset_kind=allocation.instrument.asset_kind,
                instrument_id=allocation.instrument.instrument_id,
                target_value=allocation.target_value,
                decision_time=request.decision_time,
                execution_time=request.execution_time,
                strategy_id=request.strategy_id,
                reason=allocation.reason or "score-rank-policy",
                idempotency_key=f"{request.strategy_id}:{allocation.instrument.instrument_id}:{request.decision_time.date().isoformat()}",
                account_snapshot_id=portfolio.account_snapshot_id,
            )
        )
    for position in portfolio.positions:
        instrument_id = position.instrument.instrument_id
        if instrument_id not in targeted and position.quantity > 0:
            intents.append(
                TradeIntent(
                    intent_id=f"{request.strategy_id}:{instrument_id}:{request.decision_time.isoformat()}:exit",
                    asset_kind=position.instrument.asset_kind,
                    instrument_id=instrument_id,
                    target_value=0.0,
                    decision_time=request.decision_time,
                    execution_time=request.execution_time,
                    strategy_id=request.strategy_id,
                    reason="rebalance-exit",
                    idempotency_key=f"{request.strategy_id}:{instrument_id}:{request.decision_time.date().isoformat()}:exit",
                    account_snapshot_id=portfolio.account_snapshot_id,
                )
            )
    return tuple(intents)


def _no_trade_result(
    request: TradingCycleRequest,
    manifest: DatasetManifest,
    portfolio: PortfolioSnapshot,
    dataset_hash: str,
    reason: str,
) -> TradingCycleResult:
    fingerprints = _fingerprints(manifest, request)
    return TradingCycleResult(
        status=CycleStatus.NO_TRADE,
        cycle_id=_cycle_id(request, manifest, portfolio, dataset_hash),
        decision_time=request.decision_time,
        dataset_hash=dataset_hash,
        artifact_id=request.artifact_id,
        account_snapshot_id=portfolio.account_snapshot_id,
        allocations=(),
        intents=(),
        selected_instruments=(),
        reasons=(reason,),
        universe_hash=fingerprints["universe_hash"],
        feature_hash=fingerprints["feature_hash"],
        label_hash=fingerprints["label_hash"],
        cost_hash=fingerprints["cost_hash"],
        risk_policy_hash=fingerprints["risk_policy_hash"],
    )


def _cycle_id(
    request: TradingCycleRequest,
    manifest: DatasetManifest,
    portfolio: PortfolioSnapshot,
    dataset_hash: str,
) -> str:
    key = "|".join(
        (
            request.strategy_id,
            dataset_hash,
            request.artifact_id,
            request.decision_time.isoformat(),
            portfolio.account_snapshot_id,
        )
    )
    return sha256(key.encode("utf-8")).hexdigest()


def _snapshot_hash(
    frame: pl.DataFrame,
    manifest: DatasetManifest,
    request: TradingCycleRequest,
) -> str:
    """Bind the cycle to concrete rows, not only their schema."""
    columns = sorted(frame.columns)
    sort_columns = [c for c in ("instrument_id", "session") if c in columns]
    ordered = frame.select(columns).sort(sort_columns) if sort_columns else frame.select(columns)
    row_digest = ordered.hash_rows(seed=0).to_numpy().tobytes()
    return sha256(
        manifest.schema_hash.encode("utf-8")
        + b"|"
        + row_digest
    ).hexdigest()


def _fingerprints(
    manifest: DatasetManifest,
    request: TradingCycleRequest,
) -> dict[str, str]:
    risk_policy = request.risk_policy
    policy_fields = "|".join(
        str(getattr(risk_policy, field))
        for field in StockRiskPolicy.__dataclass_fields__
    )
    return {
        "universe_hash": manifest.universe_policy_hash,
        "feature_hash": manifest.feature_set_hash,
        "label_hash": sha256(
            f"{manifest.label_definition}:{manifest.label_horizon_sessions}".encode()
        ).hexdigest(),
        "cost_hash": manifest.cost_source_hash,
        "risk_policy_hash": sha256(policy_fields.encode("utf-8")).hexdigest(),
    }
