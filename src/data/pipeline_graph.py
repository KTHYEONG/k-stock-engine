"""Declarative whole-scope refresh (collect → build → verify → register)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

from src.core.pit import PITDataError
from src.core.time import KRX_TZ, SessionCalendar
from src.data.dataset_registry import DatasetRegistry
from src.data.datasets import (
    DatasetIdentity,
    PublishedDataset,
    dataset_digest,
    dataset_reference,
    load_manifest,
    verify_dataset,
)
from src.data.fact_state import (
    fact_quarantine_file,
    file_digest,
    load_superseded_receipts,
    unresolved_events_file,
)
from src.data.receipt_catalog import ReceiptCatalog
from src.data.runtime import DataRuntime

_LOG = logging.getLogger(__name__)

_NON_STALENESS_PARAMS: Final = frozenset({"decision_time"})


@dataclass(frozen=True, slots=True)
class RefreshContext:
    """Immutable inputs for one refresh run under one scope.

    ``decision_time`` is the single availability cutoff every
    decision-time node (financial facts, quality) builds with; it is
    recorded in the report but never marks a node stale on its own, so
    an unchanged scope plans nothing to do. ``superseded_receipts`` is the
    operator-managed evidence of the facts build, read from the scope state
    (the same file ``normalize-dart-facts`` uses); the quality build resolves
    its quarantine from the facts dataset and its manual events from state.
    """

    runtime: DataRuntime
    registry: DatasetRegistry
    catalog: ReceiptCatalog
    decision_time: datetime
    superseded_receipts: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class BuildNode:
    """One registry kind produced by the scope refresh, in dependency order."""

    kind: str
    inputs: tuple[str, ...]
    bronze_sources: tuple[str, ...]
    build: Callable[[RefreshContext, Mapping[str, str]], PublishedDataset]


@dataclass(frozen=True, slots=True)
class RefreshReport:
    """Outcome of one refresh run, including a blocked collection gate."""

    decision_time: datetime
    status: str
    planned: tuple[str, ...]
    built: tuple[str, ...]
    datasets: Mapping[str, str]
    blocking_job: str | None
    collection: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class CollectionStepReport:
    """Terminal outcome of one collection step."""

    job: str
    status: str
    done: int
    pending_left: int
    requests_used: int


@dataclass(frozen=True, slots=True)
class CollectionReport:
    """Outcome of the collection phase gating a refresh."""

    steps: tuple[CollectionStepReport, ...]

    @property
    def complete(self) -> bool:
        """Whether every collection step reached ``complete``."""
        return bool(self.steps) and all(step.status == "complete" for step in self.steps)

    @property
    def blocking_job(self) -> str | None:
        """First step that did not reach ``complete``, if any."""
        for step in self.steps:
            if step.status != "complete":
                return step.job
        return None


def build_refresh_context(
    runtime: DataRuntime,
    *,
    decision_time: datetime | None = None,
    superseded_receipts: frozenset[str] | None = None,
) -> RefreshContext:
    """Bind one scope runtime to its registry, catalog and run cutoff."""
    moment = decision_time if decision_time is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        raise PITDataError("refresh decision_time must be timezone-aware")
    return RefreshContext(
        runtime=runtime,
        registry=DatasetRegistry(runtime.workspace.state_root),
        catalog=ReceiptCatalog(runtime.workspace.bronze_root / "catalog"),
        decision_time=moment,
        superseded_receipts=(
            superseded_receipts
            if superseded_receipts is not None
            else load_superseded_receipts(runtime.workspace.state_root)
        ),
    )


def _published(dataset_dir: Path) -> PublishedDataset:
    """Return the identity-bound publication pointer for one built directory."""
    manifest = load_manifest(Path(dataset_dir))
    return PublishedDataset(dataset_id=manifest.dataset_id, path=Path(dataset_dir), rows=manifest.rows)


def _build_ordinary_universe(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build the ordinary-share universe from catalog security-master sessions."""
    from src.data.ordinary_universe import catalog_master_sessions, materialize_ordinary_universe_from_catalog

    _ = inputs
    sessions = catalog_master_sessions(ctx.catalog)
    path = materialize_ordinary_universe_from_catalog(
        catalog=ctx.catalog,
        sessions=sessions,
        silver_root=ctx.runtime.workspace.silver_root,
    )
    return _published(path)


def _build_daily_market(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build Silver daily-market bars for the resolved universe sessions."""
    from src.data.daily_market_silver import materialize_daily_market_silver

    result = materialize_daily_market_silver(
        catalog=ctx.catalog,
        universe_root=ctx.runtime.workspace.silver_root,
        silver_root=ctx.runtime.workspace.silver_root,
        universe_dataset_id=inputs["ordinary_universe"],
    )
    return _published(result.dataset_path)


def _build_investor_flow_ls(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build Silver LS investor-flow rows for the resolved universe."""
    from src.data.investor_flow_silver import materialize_investor_flow_silver

    result = materialize_investor_flow_silver(
        catalog=ctx.catalog,
        universe_root=ctx.runtime.workspace.silver_root,
        silver_root=ctx.runtime.workspace.silver_root,
        universe_dataset_id=inputs["ordinary_universe"],
    )
    return _published(result.dataset_path)


def _build_market_panel(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build the decision-safe Gold market panel from resolved inputs."""
    from src.config.runtime import load_runtime_config
    from src.core.market_rules import load_krx_market_rules
    from src.data.market_panel import materialize_market_panel

    runtime = ctx.runtime
    rules = load_krx_market_rules(load_runtime_config().market_rules)
    result = materialize_market_panel(
        daily_market_path=runtime.workspace.silver_root / inputs["daily_market"],
        universe_path=runtime.workspace.silver_root / inputs["ordinary_universe"],
        rules=rules,
        gold_root=runtime.workspace.gold_root,
    )
    return _published(result.dataset_path)


def _build_investor_flow_kis_supplement(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build the provider-tagged KIS supplement for the resolved LS gap."""
    from src.data.flow_targets import investor_flow_targets_from_ids
    from src.data.investor_flow_kis_supplement import materialize_investor_flow_kis_supplement

    runtime = ctx.runtime
    targets = investor_flow_targets_from_ids(
        runtime.workspace.silver_root, inputs["ordinary_universe"], inputs["daily_market"]
    )
    result = materialize_investor_flow_kis_supplement(
        catalog=ctx.catalog,
        targets=targets,
        ls_flow_silver_path=runtime.workspace.silver_root / inputs["investor_flow_ls"],
        silver_root=runtime.workspace.silver_root,
    )
    return _published(result.dataset_path)


def _build_investor_flow(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Union certified LS flow and its KIS supplement into one dataset."""
    from src.data.investor_flow_union import materialize_investor_flow_union

    runtime = ctx.runtime
    result = materialize_investor_flow_union(
        ls_flow_silver_path=runtime.workspace.silver_root / inputs["investor_flow_ls"],
        kis_supplement_silver_path=runtime.workspace.silver_root / inputs["investor_flow_kis_supplement"],
        silver_root=runtime.workspace.silver_root,
    )
    return _published(result.dataset_path)


def _build_industry(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build the certified industry classification Silver snapshot."""
    from src.data.industry_silver import materialize_industry_classification_silver

    _ = inputs
    result = materialize_industry_classification_silver(
        catalog=ctx.catalog,
        silver_root=ctx.runtime.workspace.silver_root,
    )
    return _published(result.dataset_path)


def _build_financial_facts(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Rebuild the financial-facts Silver table from every verified Bronze receipt."""
    from src.data.incremental_normalization import refresh_dart_financial_facts

    _ = inputs
    runtime = ctx.runtime
    artifact = refresh_dart_financial_facts(
        bronze_root=runtime.workspace.bronze_root,
        silver_root=runtime.workspace.silver_root,
        artifact_root=runtime.workspace.state_root,
        decision_time=ctx.decision_time,
        calendar=_build_calendar(ctx),
        superseded_receipt_hashes=ctx.superseded_receipts,
    )
    return _published(Path(str(artifact.dataset_path)))


def _build_financial_quality(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build certified financial-quality evidence exactly as ``build-financial-quality`` does."""
    from src.data.financial_quality import materialize_financial_quality_from_files

    runtime = ctx.runtime
    facts_id = inputs["financial_facts"]
    state_root = runtime.workspace.state_root
    path, _ = materialize_financial_quality_from_files(
        runtime,
        facts_dataset_id=facts_id,
        decision_time=ctx.decision_time,
        quarantine_file=fact_quarantine_file(state_root, runtime.workspace.silver_root / facts_id),
        unresolved_events_file=unresolved_events_file(state_root),
    )
    return _published(path)


def _build_dividend_events(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build Silver cash-dividend events from dated decision filings."""
    from src.data.dividend_events import materialize_dividend_events

    _ = inputs
    runtime = ctx.runtime
    path = materialize_dividend_events(
        bronze_root=runtime.workspace.bronze_root,
        universe_root=runtime.workspace.silver_root,
        silver_root=runtime.workspace.silver_root,
        calendar=_build_calendar(ctx),
    )
    return _published(path)


def _build_reference_benchmarks(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build frictionless Gold reference benchmarks from the resolved panel."""
    from src.config.runtime import load_runtime_config
    from src.data.reference_benchmarks import load_benchmark_definitions, materialize_reference_benchmarks

    runtime = ctx.runtime
    version, definitions = load_benchmark_definitions(load_runtime_config().reference_benchmarks)
    result = materialize_reference_benchmarks(
        market_panel_path=runtime.workspace.gold_root / inputs["market_panel"],
        definitions=definitions,
        definitions_version=version,
        gold_root=runtime.workspace.gold_root,
    )
    return _published(result.dataset_path)


def _build_calendar(ctx: RefreshContext) -> SessionCalendar:
    """Year-bounded XKRX calendar for this run, so identities do not drift daily."""
    from src.core.krx_calendar import xkrx_calendar_through

    return xkrx_calendar_through(ctx.decision_time.astimezone(KRX_TZ).date())


def _preview_ordinary_universe(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be universe identity from catalog Bronze digests."""

    from src.data.datasets import DatasetLayer as _Layer
    from src.data.ordinary_universe import POLICY_VERSION, catalog_master_receipts, catalog_master_sessions

    sessions = catalog_master_sessions(ctx.catalog)
    receipts = catalog_master_receipts(ctx.catalog, sessions=sessions)
    source_hashes = [receipt.content_hash for receipt in receipts]
    return DatasetIdentity(
        kind="ordinary_universe",
        layer=_Layer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={"bronze_master": dataset_digest(source_hashes)},
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(session.isoformat() for session in sorted(sessions)).encode("utf-8")
            ).hexdigest()
        },
    )


def _preview_daily_market(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be daily-market identity from catalog Bronze digests."""
    from src.data.daily_market_silver import POLICY_VERSION, DailyMarketSilverPolicy
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.datasets import universe_sessions

    universe_id = ctx.registry.require("ordinary_universe")
    _, calendar = universe_sessions(ctx.runtime.workspace.silver_root, universe_id, allow_legacy=False)
    entries = ctx.catalog.latest(
        source="krx_daily_market",
        natural_keys={session.isoformat() for session in calendar},
    )
    source_hashes = [entries[session.isoformat()].content_hash for session in calendar]
    policy = DailyMarketSilverPolicy()
    return DatasetIdentity(
        kind="daily_market",
        layer=_Layer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
            "bronze_daily": dataset_digest(source_hashes),
        },
        params={
            "available_time": policy.available_time.isoformat(),
            "fluc_tolerance_pct": policy.fluc_tolerance_pct,
        },
    )


def _preview_investor_flow_ls(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be LS flow identity from the catalog blob digest."""
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.investor_flow_silver import POLICY_VERSION, InvestorFlowSilverPolicy

    universe_id = ctx.registry.require("ordinary_universe")
    bronze_flow = ctx.catalog.blob_digest(source="ls_investor_flow")
    policy = InvestorFlowSilverPolicy()
    return DatasetIdentity(
        kind="investor_flow_ls",
        layer=_Layer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
            "bronze_flow": bronze_flow,
        },
        params={
            "available_session_lag": policy.available_session_lag,
            "available_time": policy.available_time.isoformat(),
        },
    )


def _preview_market_panel(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be market-panel identity from resolved inputs and rules."""
    from src.config.runtime import load_runtime_config
    from src.core.market_rules import load_krx_market_rules
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.market_panel import POLICY_VERSION, MarketPanelPolicy, _rules_fingerprint

    daily_id = ctx.registry.require("daily_market")
    universe_id = ctx.registry.require("ordinary_universe")
    rules = load_krx_market_rules(load_runtime_config().market_rules)
    policy = MarketPanelPolicy()
    return DatasetIdentity(
        kind="market_panel",
        layer=_Layer.GOLD,
        policy_version=POLICY_VERSION,
        inputs={
            "daily_market": dataset_reference(daily_id, kind="daily_market"),
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
        },
        params={
            "adtv_short_sessions": policy.adtv_short_sessions,
            "adtv_long_sessions": policy.adtv_long_sessions,
            "return_vol_sessions": policy.return_vol_sessions,
            "rules_version": rules.version,
            "rules_fingerprint": _rules_fingerprint(rules),
        },
    )


def _preview_investor_flow_kis_supplement(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be KIS supplement identity from the catalog digest."""
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.investor_flow_kis_supplement import (
        POLICY_VERSION,
        InvestorFlowKisSupplementPolicy,
    )

    universe_id = ctx.registry.require("ordinary_universe")
    daily_id = ctx.registry.require("daily_market")
    ls_id = ctx.registry.require("investor_flow_ls")
    bronze_kis = ctx.catalog.blob_digest(source="kis_investor_flow")
    policy = InvestorFlowKisSupplementPolicy()
    return DatasetIdentity(
        kind="investor_flow_kis_supplement",
        layer=_Layer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
            "daily_market": dataset_reference(daily_id, kind="daily_market"),
            "ls": dataset_reference(ls_id, kind="investor_flow_ls"),
            "bronze_kis": bronze_kis,
        },
        params={
            "available_session_lag": policy.available_session_lag,
            "available_time": policy.available_time.isoformat(),
        },
    )


def _preview_investor_flow(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be flow-union identity from resolved LS and KIS inputs."""
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.investor_flow_union import POLICY_VERSION

    ls_id = ctx.registry.require("investor_flow_ls")
    kis_id = ctx.registry.require("investor_flow_kis_supplement")
    return DatasetIdentity(
        kind="investor_flow",
        layer=_Layer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={
            "ls": dataset_reference(ls_id, kind="investor_flow_ls"),
            "kis_supplement": dataset_reference(kis_id, kind="investor_flow_kis_supplement"),
        },
        params={},
    )


def _preview_industry(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be industry identity from the catalog blob digest."""
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.industry_silver import POLICY_VERSION

    bronze_classification = ctx.catalog.blob_digest(source="kis_industry")
    if bronze_classification == dataset_digest([]):
        raise PITDataError("no certified INDUSTRY Bronze evidence found")
    return DatasetIdentity(
        kind="industry",
        layer=_Layer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={"bronze_classification": bronze_classification},
        params={"symbols": None},
    )


def _preview_financial_facts(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be financial-facts identity from Bronze fact digests."""

    from src.data.datasets import DatasetLayer as _Layer
    from src.data.datasets import dataset_reference as _reference
    from src.data.datasets import load_manifest as _load
    from src.data.incremental_normalization import (
        AVAILABILITY_POLICY,
        _discover_fact_receipts,
        load_frozen_dart_ticker_bridge,
    )

    runtime = ctx.runtime
    receipts = _discover_fact_receipts(runtime.workspace.bronze_root)
    # 빌더와 같이 대체된 수집 기록은 입력 요약에서 제외한다(포함하면 갱신 판정이 영원히 어긋난다).
    receipt_hashes = [
        str(item["content_hash"]) for item in receipts if str(item["content_hash"]) not in ctx.superseded_receipts
    ]
    disclosure_digest = dataset_digest([])
    for candidate in sorted(runtime.workspace.silver_root.glob("disclosures_*"), reverse=True):
        try:
            _load(candidate)
        except PITDataError:
            continue
        disclosure_digest = _reference(candidate.name, kind="disclosures")
        break
    bridge_receipt_hash: str | None = None
    if (runtime.workspace.bronze_root / "dart_corp_codes").exists():
        _, bridge_receipt_hash = load_frozen_dart_ticker_bridge(
            bronze_root=runtime.workspace.bronze_root, decision_time=ctx.decision_time
        )
    calendar = _build_calendar(ctx)
    calendar_digest = hashlib.sha256(
        "\n".join(session.astimezone(UTC).isoformat() for session in calendar.sessions).encode("utf-8")
    ).hexdigest()
    return DatasetIdentity(
        kind="financial_facts",
        layer=_Layer.SILVER,
        policy_version="dart-incremental-v1",
        inputs={
            "bronze_facts": dataset_digest(receipt_hashes),
            "superseded": dataset_digest(sorted(ctx.superseded_receipts)),
            "disclosures": disclosure_digest,
            "ticker_bridge": dataset_digest([bridge_receipt_hash]) if bridge_receipt_hash else dataset_digest([]),
        },
        params={
            "decision_time": ctx.decision_time,
            "availability_policy": AVAILABILITY_POLICY,
            "calendar_digest": calendar_digest,
            "ticker_bridge": bridge_receipt_hash,
        },
    )


def _preview_financial_quality(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be financial-quality identity from the resolved facts."""
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.financial_quality import FinancialQualityPolicy

    facts_id = ctx.registry.require("financial_facts")
    state_root = ctx.runtime.workspace.state_root
    quarantine = fact_quarantine_file(state_root, ctx.runtime.workspace.silver_root / facts_id)
    manual = unresolved_events_file(state_root)
    policy = FinancialQualityPolicy()
    return DatasetIdentity(
        kind="financial_quality",
        layer=_Layer.SILVER,
        policy_version=policy.version,
        inputs={
            "facts": dataset_reference(facts_id, kind="financial_facts"),
            "quarantine": file_digest(quarantine),
            "unresolved_events": file_digest(manual),
        },
        params={
            "decision_time": ctx.decision_time,
            "required_facts": ",".join(policy.required_facts),
            "basis_preference": ",".join(policy.basis_preference),
        },
    )


def _preview_dividend_events(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be dividend-events identity from decision envelopes."""
    import json as _json

    from src.core.time import KRX_TZ
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.dividend_events import POLICY_VERSION, _iter_decision_envelopes

    bronze_root = ctx.runtime.workspace.bronze_root
    envelopes = _iter_decision_envelopes(bronze_root)
    envelope_hashes = [
        hashlib.sha256(
            _json.dumps(envelope, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        for envelope in envelopes
    ]
    bridge_paths = sorted((bronze_root / "dart_corp_codes").glob("*/payload.json"))
    bridge_receipt_hash = bridge_paths[-1].parent.name if bridge_paths else ""
    calendar = _build_calendar(ctx)
    return DatasetIdentity(
        kind="dividend_events",
        layer=_Layer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={
            "bronze_dividend_decisions": dataset_digest(envelope_hashes),
            "corp_code_bridge": dataset_digest([bridge_receipt_hash]),
        },
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(
                    session.astimezone(KRX_TZ).date().isoformat() for session in calendar.sessions
                ).encode("utf-8")
            ).hexdigest()
        },
    )


def _preview_reference_benchmarks(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be benchmarks identity from the resolved panel."""
    import json as _json

    from src.config.runtime import load_runtime_config
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.reference_benchmarks import load_benchmark_definitions

    panel_id = ctx.registry.require("market_panel")
    definitions_version, definitions = load_benchmark_definitions(load_runtime_config().reference_benchmarks)
    definition_payload = [
        {
            "benchmark_id": definition.benchmark_id,
            "weighting": definition.weighting.value,
            "min_adtv20_krw": definition.min_adtv20_krw,
        }
        for definition in sorted(definitions, key=lambda item: item.benchmark_id)
    ]
    return DatasetIdentity(
        kind="reference_benchmarks",
        layer=_Layer.GOLD,
        policy_version=definitions_version,
        inputs={"market_panel": dataset_reference(panel_id, kind="market_panel")},
        params={
            "definitions_version": definitions_version,
            "definitions": _json.dumps(definition_payload, sort_keys=True, separators=(",", ":")),
        },
    )


_PREVIEW_FNS: Final[dict[str, Callable[[RefreshContext], DatasetIdentity]]] = {
    "ordinary_universe": _preview_ordinary_universe,
    "daily_market": _preview_daily_market,
    "investor_flow_ls": _preview_investor_flow_ls,
    "market_panel": _preview_market_panel,
    "investor_flow_kis_supplement": _preview_investor_flow_kis_supplement,
    "investor_flow": _preview_investor_flow,
    "industry": _preview_industry,
    "financial_facts": _preview_financial_facts,
    "financial_quality": _preview_financial_quality,
    "dividend_events": _preview_dividend_events,
    "reference_benchmarks": _preview_reference_benchmarks,
}

SCOPE_GRAPH: Final[tuple[BuildNode, ...]] = (
    BuildNode(kind="ordinary_universe", inputs=(), bronze_sources=("krx_security_master",), build=_build_ordinary_universe),
    BuildNode(kind="dividend_events", inputs=(), bronze_sources=("corporate_actions",), build=_build_dividend_events),
    BuildNode(kind="industry", inputs=(), bronze_sources=("industry",), build=_build_industry),
    BuildNode(kind="financial_facts", inputs=(), bronze_sources=("financial_facts",), build=_build_financial_facts),
    BuildNode(
        kind="daily_market",
        inputs=("ordinary_universe",),
        bronze_sources=("krx_daily_market",),
        build=_build_daily_market,
    ),
    BuildNode(
        kind="investor_flow_ls",
        inputs=("ordinary_universe",),
        bronze_sources=("investor_flow",),
        build=_build_investor_flow_ls,
    ),
    BuildNode(
        kind="financial_quality",
        inputs=("financial_facts",),
        bronze_sources=(),
        build=_build_financial_quality,
    ),
    BuildNode(
        kind="market_panel",
        inputs=("daily_market", "ordinary_universe"),
        bronze_sources=(),
        build=_build_market_panel,
    ),
    BuildNode(
        kind="investor_flow_kis_supplement",
        inputs=("ordinary_universe", "daily_market", "investor_flow_ls"),
        bronze_sources=("investor_flow",),
        build=_build_investor_flow_kis_supplement,
    ),
    BuildNode(
        kind="investor_flow",
        inputs=("investor_flow_ls", "investor_flow_kis_supplement"),
        bronze_sources=(),
        build=_build_investor_flow,
    ),
    BuildNode(
        kind="reference_benchmarks",
        inputs=("market_panel",),
        bronze_sources=(),
        build=_build_reference_benchmarks,
    ),
)


def _validate_scope_graph(graph: tuple[BuildNode, ...]) -> tuple[BuildNode, ...]:
    """Fail closed when the scope graph has duplicates, unknown inputs or cycles."""
    kinds = [node.kind for node in graph]
    if len(set(kinds)) != len(kinds):
        raise PITDataError(f"scope graph has duplicate kinds: {sorted(kinds)}")
    known = set(kinds)
    for node in graph:
        for dependency in node.inputs:
            if dependency not in known:
                raise PITDataError(f"scope graph node {node.kind!r} depends on unknown kind {dependency!r}")
    remaining = {node.kind: set(node.inputs) for node in graph}
    ordered: list[str] = []
    while remaining:
        ready = sorted(kind for kind, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise PITDataError(f"scope graph has a dependency cycle: {sorted(remaining)}")
        for kind in ready:
            ordered.append(kind)
            del remaining[kind]
            for dependencies in remaining.values():
                dependencies.discard(kind)
    return graph


_VALIDATED_GRAPH: Final = _validate_scope_graph(SCOPE_GRAPH)


def _expected_matches_registered(expected: DatasetIdentity, dataset_dir: Path) -> bool:
    """Whether the registered manifest already describes the would-be identity.

    ``decision_time`` is a run-scoped build cutoff recorded in the report; it
    never marks a node stale on its own.
    """
    try:
        manifest = load_manifest(Path(dataset_dir))
    except PITDataError:
        return False
    if expected.layer is not manifest.layer or expected.policy_version != manifest.policy_version:
        return False
    if dict(expected.inputs) != dict(manifest.inputs):
        return False
    expected_params = {name: value for name, value in expected.params.items() if name not in _NON_STALENESS_PARAMS}
    manifest_params = {name: value for name, value in manifest.params.items() if name not in _NON_STALENESS_PARAMS}
    return expected_params == manifest_params


def _registered_dataset_dir(ctx: RefreshContext, dataset_id: str) -> Path | None:
    """Return the Silver or Gold directory for one registered id, if unambiguous."""
    candidates = [
        ctx.runtime.workspace.silver_root / dataset_id,
        ctx.runtime.workspace.gold_root / dataset_id,
    ]
    matches = tuple(path for path in candidates if (path / "manifest.json").is_file())
    return matches[0] if len(matches) == 1 else None


def plan_refresh(ctx: RefreshContext) -> tuple[BuildNode, ...]:
    """Plan stale scope nodes in topological order without building anything.

    A node is stale when its kind is unregistered, its manifest is
    unreadable, its would-be identity (Bronze digests and upstream ids)
    differs from the registered manifest, or any upstream node is stale.
    A registry kind outside the scope graph fails closed.
    """
    snapshot = dict(ctx.registry.snapshot())
    known_kinds = {node.kind for node in _VALIDATED_GRAPH}
    unknown = sorted(set(snapshot) - known_kinds)
    if unknown:
        raise PITDataError(f"dataset registry has kinds outside the scope graph: {unknown}")
    stale: set[str] = set()
    plan: list[BuildNode] = []
    for node in _VALIDATED_GRAPH:
        dataset_id = snapshot.get(node.kind)
        if dataset_id is None or any(dependency in stale for dependency in node.inputs):
            stale.add(node.kind)
            plan.append(node)
            continue
        dataset_dir = _registered_dataset_dir(ctx, dataset_id)
        if dataset_dir is None:
            stale.add(node.kind)
            plan.append(node)
            continue
        try:
            expected = _PREVIEW_FNS[node.kind](ctx)
        except Exception as exc:  # noqa: BLE001 - any unreadable input means rebuild, build fails loudly
            _LOG.warning("[DATA] stage=refresh_plan kind=%s status=preview_failed error=%s", node.kind, exc)
            stale.add(node.kind)
            plan.append(node)
            continue
        if not _expected_matches_registered(expected, dataset_dir):
            stale.add(node.kind)
            plan.append(node)
    return tuple(plan)


def _disk_dataset_ids(ctx: RefreshContext, extra: set[str]) -> set[str]:
    """Collect physically present v2 dataset ids for lineage verification."""
    known = set(extra) | set(ctx.registry.retired())
    for root in (ctx.runtime.workspace.silver_root, ctx.runtime.workspace.gold_root):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            try:
                known.add(load_manifest(path).dataset_id)
            except PITDataError:
                continue
    return known


def run_refresh(
    ctx: RefreshContext, *, dry_run: bool, emit: Callable[[Mapping[str, object]], None]
) -> RefreshReport:
    """Rebuild stale nodes in order, verifying each and registering atomically.

    Every rebuilt output is verified before the next node builds; the
    registry moves all rebuilt kinds in one atomic write only after every
    node succeeded. Any failure leaves the registry unchanged while the
    rebuilt datasets stay on disk for ``prune-datasets``.
    """
    plan = plan_refresh(ctx)
    planned = tuple(node.kind for node in plan)
    for kind in planned:
        emit({"type": "plan", "kind": kind, "status": "stale"})
        _LOG.info("[DATA] stage=refresh_plan kind=%s status=stale", kind)
    if dry_run:
        return RefreshReport(
            decision_time=ctx.decision_time,
            status="dry_run",
            planned=planned,
            built=(),
            datasets=MappingProxyType({}),
            blocking_job=None,
            collection=(),
        )
    snapshot = dict(ctx.registry.snapshot())
    built: dict[str, PublishedDataset] = {}
    known = _disk_dataset_ids(ctx, set(snapshot.values()))
    for node in plan:
        try:
            resolved = {
                dependency: built[dependency].dataset_id if dependency in built else snapshot[dependency]
                for dependency in node.inputs
            }
        except KeyError as exc:
            raise PITDataError(f"refresh input is not registered: {exc}") from exc
        outcome = node.build(ctx, MappingProxyType(resolved))
        verification = verify_dataset(outcome.path, known_ids=known.__contains__)
        if not verification.passed:
            raise PITDataError(f"refresh verification failed for {node.kind}: {'; '.join(verification.failures)}")
        built[node.kind] = outcome
        known.add(outcome.dataset_id)
        emit({"type": "dataset", "kind": node.kind, "dataset_id": outcome.dataset_id, "status": "built"})
        _LOG.info("[DATA] stage=refresh_build kind=%s dataset=%s status=built", node.kind, outcome.dataset_id)
    if built:
        ctx.registry.register_many({kind: outcome.dataset_id for kind, outcome in built.items()})
    return RefreshReport(
        decision_time=ctx.decision_time,
        status="complete",
        planned=planned,
        built=tuple(built),
        datasets=MappingProxyType({kind: outcome.dataset_id for kind, outcome in built.items()}),
        blocking_job=None,
        collection=(),
    )


def _run_dart_collection_step(
    ctx: RefreshContext, job_name: str, *, dry_run: bool, emit: Callable[[Mapping[str, object]], None]
) -> CollectionStepReport:
    """Run one budgeted DART collection job without duplicating runner wiring."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.dart_backfill import build_scoped_dart_collector
    from src.data.jobs.dart import resolve_dart_job
    from src.data.jobs.runner import build_job_context, run_job
    from src.integrations.quota import ProviderQuotaStateStore

    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    collector = None
    if not dry_run:  # pragma: no cover - live provider path
        collector = build_scoped_dart_collector(
            provider=provider,
            quota_store=ProviderQuotaStateStore(ctx.runtime.workspace.state_root / "quota"),
            key_env=provider.default_key_env,
        )
    job_ctx = build_job_context(
        runtime=ctx.runtime, provider=provider, key_env=provider.default_key_env, collector=collector
    )
    report = run_job(
        resolve_dart_job(job_name),
        job_ctx,
        chunk_size=provider.dart.batch_identities,
        max_chunks=None,
        dry_run=dry_run,
        emit=emit,
    )
    return CollectionStepReport(
        job=job_name, status=report.status, done=report.done,
        pending_left=report.pending_left, requests_used=report.requests_used,
    )


def _run_krx_collection_step(
    ctx: RefreshContext, job_name: str, *, dry_run: bool, emit: Callable[[Mapping[str, object]], None]
) -> CollectionStepReport:
    """Run one KRX session collection job without duplicating runner wiring."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.jobs.krx import KRX_CHUNK_SIZE, build_krx_job_context, resolve_krx_job
    from src.data.jobs.runner import run_job
    from src.integrations.krx.client import build_scoped_krx_client
    from src.integrations.quota import ProviderQuotaStateStore

    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    collector = None
    if not dry_run:  # pragma: no cover - live provider path
        collector = build_scoped_krx_client(
            policy=provider.krx,
            quota_store=ProviderQuotaStateStore(ctx.runtime.workspace.state_root / "quota"),
        )
    job_ctx = build_krx_job_context(runtime=ctx.runtime, provider=provider, collector=collector)
    report = run_job(
        resolve_krx_job(job_name),
        job_ctx,
        chunk_size=KRX_CHUNK_SIZE,
        max_chunks=None,
        dry_run=dry_run,
        emit=emit,
    )
    return CollectionStepReport(
        job=job_name, status=report.status, done=report.done,
        pending_left=report.pending_left, requests_used=report.requests_used,
    )


def _run_ls_flow_collection_step(
    ctx: RefreshContext, *, dry_run: bool, emit: Callable[[Mapping[str, object]], None]
) -> CollectionStepReport:
    """Run the range-planned LS investor-flow job through the budgeted runner."""
    from src.config import load_provider_policy, load_runtime_config
    from src.core.pit import PITDataError
    from src.data.jobs.flow import LS_CHUNK_SIZE, LsInvestorFlowJob, build_ls_job_context
    from src.data.jobs.runner import run_job
    from src.integrations.ls.client import build_scoped_ls_client
    from src.integrations.ls.investor_flow import LsInvestorFlowCollector
    from src.integrations.quota import ProviderQuotaStateStore

    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    collector = None
    if not dry_run:  # pragma: no cover - live provider path
        collector = LsInvestorFlowCollector(
            client=build_scoped_ls_client(
                policy=provider.ls,
                quota_store=ProviderQuotaStateStore(ctx.runtime.workspace.state_root / "quota"),
                token_cache_dir=runtime_config.logs_root,
            ),
        )
    job_ctx = build_ls_job_context(runtime=ctx.runtime, provider=provider, collector=collector)
    try:
        report = run_job(
            LsInvestorFlowJob(),
            job_ctx,
            chunk_size=LS_CHUNK_SIZE,
            max_chunks=None,
            dry_run=dry_run,
            emit=emit,
        )
    except PITDataError:
        if dry_run:
            return CollectionStepReport(
                job="ls_investor_flow", status="dry_run", done=0, pending_left=0, requests_used=0
            )
        raise
    return CollectionStepReport(
        job="ls_investor_flow", status=report.status, done=report.done,
        pending_left=report.pending_left, requests_used=report.requests_used,
    )


def _run_kis_flow_collection_step(
    ctx: RefreshContext, *, dry_run: bool, emit: Callable[[Mapping[str, object]], None]
) -> CollectionStepReport:
    """Run the range-planned KIS investor-flow job through the budgeted runner."""
    from src.config import load_provider_policy, load_runtime_config
    from src.core.pit import PITDataError
    from src.data.jobs.flow import KIS_CHUNK_SIZE, KisInvestorFlowJob, build_kis_job_context
    from src.data.jobs.runner import run_job
    from src.integrations.kis.client import KisClient, KisCredentials
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    collector = None
    if not dry_run:  # pragma: no cover - live provider path
        collector = KisInvestorFlowCollector(
            client=KisClient(
                KisCredentials.from_env(provider.kis),
                token_cache_dir=Path(runtime_config.logs_root),
                policy=provider.kis,
            )
        )
    job_ctx = build_kis_job_context(runtime=ctx.runtime, provider=provider, collector=collector)
    try:
        report = run_job(
            KisInvestorFlowJob(),
            job_ctx,
            chunk_size=KIS_CHUNK_SIZE,
            max_chunks=None,
            dry_run=dry_run,
            emit=emit,
        )
    except PITDataError:
        if dry_run:
            # LS Silver가 아직 갱신 전이면 KIS 대상을 셀 수 없다. 0건으로 보고하면 "할 일 없음"으로 오해된다.
            return CollectionStepReport(
                job="kis_investor_flow", status="blocked_until_ls_build", done=0, pending_left=0, requests_used=0
            )
        raise
    return CollectionStepReport(
        job="kis_investor_flow", status=report.status, done=report.done,
        pending_left=report.pending_left, requests_used=report.requests_used,
    )


CollectionStep = tuple[str, Callable[..., CollectionStepReport]]


def _default_collection_steps() -> tuple[CollectionStep, ...]:
    """Collection jobs in refresh order, from disclosures to investor flow."""
    return (
        (
            "dart_disclosures",
            lambda ctx, *, dry_run, emit: _run_dart_collection_step(
                ctx, "dart_disclosures", dry_run=dry_run, emit=emit
            ),
        ),
        (
            "dart_facts",
            lambda ctx, *, dry_run, emit: _run_dart_collection_step(ctx, "dart_facts", dry_run=dry_run, emit=emit),
        ),
        (
            "dividend_decisions",
            lambda ctx, *, dry_run, emit: _run_dart_collection_step(
                ctx, "dividend_decisions", dry_run=dry_run, emit=emit
            ),
        ),
        (
            "krx_daily_market",
            lambda ctx, *, dry_run, emit: _run_krx_collection_step(ctx, "krx_daily_market", dry_run=dry_run, emit=emit),
        ),
        (
            "krx_security_master",
            lambda ctx, *, dry_run, emit: _run_krx_collection_step(
                ctx, "krx_security_master", dry_run=dry_run, emit=emit
            ),
        ),
        ("ls_investor_flow", _run_ls_flow_collection_step),
        ("kis_investor_flow", _run_kis_flow_collection_step),
    )


def run_collection_jobs(
    ctx: RefreshContext,
    *,
    dry_run: bool,
    emit: Callable[[Mapping[str, object]], None],
    steps: tuple[CollectionStep, ...] | None = None,
) -> CollectionReport:
    """Run collection steps in order, reporting every step without building."""
    reports: list[CollectionStepReport] = []
    for name, step in steps if steps is not None else _default_collection_steps():
        report = step(ctx, dry_run=dry_run, emit=emit)
        if report.job != name:
            raise PITDataError(f"collection step name mismatch: expected={name!r} actual={report.job!r}")
        emit(
            {
                "type": "collection",
                "job": report.job,
                "status": report.status,
                "done": report.done,
                "pending_left": report.pending_left,
                "requests_used": report.requests_used,
            }
        )
        _LOG.info(
            "[DATA] stage=refresh_collect job=%s status=%s done=%d pending_left=%d",
            report.job,
            report.status,
            report.done,
            report.pending_left,
        )
        reports.append(report)
    return CollectionReport(steps=tuple(reports))


def _phase_split(plan: tuple[BuildNode, ...]) -> tuple[tuple[BuildNode, ...], tuple[BuildNode, ...]]:
    """Split a build plan at ``investor_flow_ls`` in graph order."""
    order = [node.kind for node in _VALIDATED_GRAPH]
    try:
        cutoff = order.index("investor_flow_ls")
    except ValueError:  # pragma: no cover - the scope graph always contains the LS node
        return (plan, ())
    first_kinds = set(order[: cutoff + 1])
    first = tuple(node for node in plan if node.kind in first_kinds)
    second = tuple(node for node in plan if node.kind not in first_kinds)
    return (first, second)


def _build_nodes(
    ctx: RefreshContext,
    nodes: tuple[BuildNode, ...],
    snapshot: dict[str, str],
    built: dict[str, PublishedDataset],
    known: set[str],
    emit: Callable[[Mapping[str, object]], None],
) -> None:
    """Build one ordered subset, verifying each and emitting progress."""
    for node in nodes:
        try:
            resolved = {
                dependency: built[dependency].dataset_id if dependency in built else snapshot[dependency]
                for dependency in node.inputs
            }
        except KeyError as exc:
            raise PITDataError(f"refresh input is not registered: {exc}") from exc
        outcome = node.build(ctx, MappingProxyType(resolved))
        verification = verify_dataset(outcome.path, known_ids=known.__contains__)
        if not verification.passed:
            raise PITDataError(f"refresh verification failed for {node.kind}: {'; '.join(verification.failures)}")
        built[node.kind] = outcome
        known.add(outcome.dataset_id)
        emit({"type": "dataset", "kind": node.kind, "dataset_id": outcome.dataset_id, "status": "built"})
        _LOG.info("[DATA] stage=refresh_build kind=%s dataset=%s status=built", node.kind, outcome.dataset_id)


def refresh_scope(
    ctx: RefreshContext,
    *,
    collect: bool,
    dry_run: bool,
    emit: Callable[[Mapping[str, object]], None],
    steps: tuple[CollectionStep, ...] | None = None,
) -> RefreshReport:
    """Run one scope refresh with KIS collection after the LS build.

    With ``collect``, the order is: DART/KRX/LS collection, build and register
    the graph up to ``investor_flow_ls``, the ``kis_investor_flow`` collection
    step, then the remaining builds. KIS collection needs a current LS Silver
    dataset, so it cannot run before the LS build. ``dry_run`` reports all
    steps in the same order without requests or writes.
    """
    resolved_steps = steps if steps is not None else _default_collection_steps()
    has_kis = any(name == "kis_investor_flow" for name, _ in resolved_steps)
    if not collect or not has_kis:
        summaries: tuple[Mapping[str, object], ...] = ()
        if collect:
            collection = run_collection_jobs(ctx, dry_run=dry_run, emit=emit, steps=steps)
            summaries = tuple(
                MappingProxyType(
                    {
                        "job": step.job,
                        "status": step.status,
                        "done": step.done,
                        "pending_left": step.pending_left,
                        "requests_used": step.requests_used,
                    }
                )
                for step in collection.steps
            )
            if not dry_run and not collection.complete:
                return RefreshReport(
                    decision_time=ctx.decision_time,
                    status="blocked",
                    planned=(),
                    built=(),
                    datasets=MappingProxyType({}),
                    blocking_job=collection.blocking_job,
                    collection=summaries,
                )
        report = run_refresh(ctx, dry_run=dry_run, emit=emit)
        return replace(report, collection=summaries)
    pre_steps = tuple(item for item in resolved_steps if item[0] != "kis_investor_flow")
    kis_steps = tuple(item for item in resolved_steps if item[0] == "kis_investor_flow")
    if dry_run and collect:
        pre_collection = run_collection_jobs(ctx, dry_run=True, emit=emit, steps=pre_steps)
        plan = plan_refresh(ctx)
        first, second = _phase_split(plan)
        for kind in tuple(node.kind for node in first):
            emit({"type": "plan", "kind": kind, "status": "stale"})
            _LOG.info("[DATA] stage=refresh_plan kind=%s status=stale", kind)
        kis_collection = run_collection_jobs(ctx, dry_run=True, emit=emit, steps=kis_steps)
        for kind in tuple(node.kind for node in second):
            emit({"type": "plan", "kind": kind, "status": "stale"})
            _LOG.info("[DATA] stage=refresh_plan kind=%s status=stale", kind)
        summaries = tuple(
            MappingProxyType(
                {
                    "job": step.job,
                    "status": step.status,
                    "done": step.done,
                    "pending_left": step.pending_left,
                    "requests_used": step.requests_used,
                }
            )
            for step in (*pre_collection.steps, *kis_collection.steps)
        )
        return RefreshReport(
            decision_time=ctx.decision_time,
            status="dry_run",
            planned=tuple(node.kind for node in plan),
            built=(),
            datasets=MappingProxyType({}),
            blocking_job=None,
            collection=summaries,
        )
    pre_collection = run_collection_jobs(ctx, dry_run=False, emit=emit, steps=pre_steps)
    pre_summaries = [
        MappingProxyType(
            {
                "job": step.job,
                "status": step.status,
                "done": step.done,
                "pending_left": step.pending_left,
                "requests_used": step.requests_used,
            }
        )
        for step in pre_collection.steps
    ]
    if not pre_collection.complete:
        return RefreshReport(
            decision_time=ctx.decision_time,
            status="blocked",
            planned=(),
            built=(),
            datasets=MappingProxyType({}),
            blocking_job=pre_collection.blocking_job,
            collection=tuple(pre_summaries),
        )
    plan = plan_refresh(ctx)
    planned_kinds = tuple(node.kind for node in plan)
    for kind in planned_kinds:
        emit({"type": "plan", "kind": kind, "status": "stale"})
        _LOG.info("[DATA] stage=refresh_plan kind=%s status=stale", kind)
    first, _ = _phase_split(plan)
    snapshot = dict(ctx.registry.snapshot())
    built: dict[str, PublishedDataset] = {}
    known = _disk_dataset_ids(ctx, set(snapshot.values()))
    _build_nodes(ctx, first, snapshot, built, known, emit)
    if built:
        ctx.registry.register_many({kind: outcome.dataset_id for kind, outcome in built.items()})
        snapshot = dict(ctx.registry.snapshot())
    kis_collection = run_collection_jobs(ctx, dry_run=False, emit=emit, steps=kis_steps)
    summaries = tuple(pre_summaries) + tuple(
        MappingProxyType(
            {
                "job": step.job,
                "status": step.status,
                "done": step.done,
                "pending_left": step.pending_left,
                "requests_used": step.requests_used,
            }
        )
        for step in kis_collection.steps
    )
    if not kis_collection.complete:
        return RefreshReport(
            decision_time=ctx.decision_time,
            status="blocked",
            planned=planned_kinds,
            built=tuple(built),
            datasets=MappingProxyType({kind: outcome.dataset_id for kind, outcome in built.items()}),
            blocking_job=kis_collection.blocking_job,
            collection=summaries,
        )
    remaining = plan_refresh(ctx)
    _, second = _phase_split(remaining)
    # Nodes already built in the first phase are registered, so the fresh plan
    # only contains what is still stale; build exactly that remainder.
    _build_nodes(ctx, second, snapshot, built, known, emit)
    if second:
        ctx.registry.register_many(
            {node.kind: built[node.kind].dataset_id for node in second if node.kind in built}
        )
    return RefreshReport(
        decision_time=ctx.decision_time,
        status="complete",
        planned=planned_kinds,
        built=tuple(built),
        datasets=MappingProxyType({kind: outcome.dataset_id for kind, outcome in built.items()}),
        blocking_job=None,
        collection=summaries,
    )


__all__ = [
    "SCOPE_GRAPH",
    "BuildNode",
    "CollectionReport",
    "CollectionStep",
    "CollectionStepReport",
    "RefreshContext",
    "RefreshReport",
    "build_refresh_context",
    "plan_refresh",
    "refresh_scope",
    "run_collection_jobs",
    "run_refresh",
]
