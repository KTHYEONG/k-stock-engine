"""Declarative whole-scope refresh (collect → build → verify → register)."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

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


def node_peak_rss_mb(*, proc_root: Path = Path("/proc")) -> float | None:
    """Peak resident set in MB since the last reset."""
    try:
        text = (Path(proc_root) / "self" / "status").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            match = re.fullmatch(r"VmHWM:\s+([0-9]+)\s+kB\s*", line)
            if match is not None:
                return float(int(match[1])) / 1024.0
    except (OSError, ValueError, OverflowError):
        return None
    return None


def reset_peak_rss(*, proc_root: Path = Path("/proc")) -> bool:
    """Reset the kernel peak-RSS watermark; False when unsupported."""
    try:
        (Path(proc_root) / "self" / "clear_refs").write_text("5", encoding="ascii")
    except (OSError, ValueError):
        return False
    return True


def _build_with_peak(
    node: BuildNode, ctx: RefreshContext, inputs: Mapping[str, str]
) -> tuple[PublishedDataset, float | None]:
    """Reset the watermark, run one node build, then read its peak."""
    reset_peak_rss()
    outcome = node.build(ctx, inputs)
    return outcome, node_peak_rss_mb()


def _dataset_payload(kind: str, dataset_id: str, peak: float | None) -> Mapping[str, object]:
    """Dataset progress event, carrying the peak only when available."""
    payload: dict[str, object] = {"type": "dataset", "kind": kind, "dataset_id": dataset_id, "status": "built"}
    if peak is not None:
        payload["peak_rss_mb"] = peak
    return payload


def _log_build(kind: str, dataset_id: str, peak: float | None) -> None:
    """Emit the refresh_build line, carrying the peak only when available."""
    if peak is not None:
        _LOG.info(
            "[DATA] stage=refresh_build kind=%s dataset=%s status=built peak_rss_mb=%d",
            kind,
            dataset_id,
            int(peak),
        )
    else:
        _LOG.info("[DATA] stage=refresh_build kind=%s dataset=%s status=built", kind, dataset_id)


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
    deferred_build_kinds: frozenset[str] = frozenset()
    retry_krx_session: date | None = None


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
    degraded_jobs: tuple[str, ...] = ()
    node_peak_rss_mb: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


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
    optional: frozenset[str] = frozenset()

    @property
    def complete(self) -> bool:
        """Whether every required collection step reached ``complete``."""
        return bool(self.steps) and all(
            step.status == "complete" for step in self.steps if step.job not in self.optional
        )

    @property
    def blocking_job(self) -> str | None:
        """First required step that did not reach ``complete``, if any."""
        for step in self.steps:
            if step.status != "complete" and step.job not in self.optional:
                return step.job
        return None

    @property
    def degraded_jobs(self) -> tuple[str, ...]:
        """Optional steps that did not reach ``complete``, in step order."""
        return tuple(
            step.job for step in self.steps if step.status != "complete" and step.job in self.optional
        )


OPTIONAL_COLLECTION_JOBS: Final[frozenset[str]] = frozenset({
    "kind_notice_search",
    "kind_notice_documents",
    "kis_industry",
    "dbsec_ticker_master",
    "dbsec_daily_chart",
    "dbsec_closing_bar",
})


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
    current = ctx.registry.snapshot().get("ordinary_universe")
    path = materialize_ordinary_universe_from_catalog(
        catalog=ctx.catalog,
        sessions=sessions,
        silver_root=ctx.runtime.workspace.silver_root,
        reuse_from=_registered_dataset_dir(ctx, current) if current is not None else None,
    )
    return _published(path)


def _build_daily_market(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build Silver daily-market bars for the resolved universe sessions."""
    from src.data.daily_market_silver import materialize_daily_market_silver

    current = ctx.registry.snapshot().get("daily_market")
    result = materialize_daily_market_silver(
        catalog=ctx.catalog,
        universe_root=ctx.runtime.workspace.silver_root,
        silver_root=ctx.runtime.workspace.silver_root,
        universe_dataset_id=inputs["ordinary_universe"],
        reuse_from=_registered_dataset_dir(ctx, current) if current is not None else None,
    )
    return _published(result.dataset_path)


def _build_hedge_series(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build the KOSDAQ-150 hedge series Silver from the registered universe; returns the published dataset."""
    from src.config import load_runtime_config
    from src.data.hedge_series_silver import load_hedge_series_config, materialize_hedge_series_silver

    result = materialize_hedge_series_silver(
        catalog=ctx.catalog,
        universe_root=ctx.runtime.workspace.silver_root,
        silver_root=ctx.runtime.workspace.silver_root,
        config=load_hedge_series_config(load_runtime_config().hedge_series),
        universe_dataset_id=inputs["ordinary_universe"],
    )
    return _published(result.dataset_path)


def _build_trend_series(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build the KOSPI-200 trend series Silver from the registered universe; returns the published dataset."""
    from src.config import load_runtime_config
    from src.data.hedge_series_silver import load_hedge_series_config, materialize_hedge_series_silver

    result = materialize_hedge_series_silver(
        catalog=ctx.catalog,
        universe_root=ctx.runtime.workspace.silver_root,
        silver_root=ctx.runtime.workspace.silver_root,
        config=load_hedge_series_config(load_runtime_config().trend_series),
        universe_dataset_id=inputs["ordinary_universe"],
    )
    return _published(result.dataset_path)


def _build_cash_series(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build the cash-proxy series Silver from the registered universe; returns the published dataset."""
    from src.config import load_runtime_config
    from src.data.cash_series_silver import load_cash_series_config, materialize_cash_series_silver

    result = materialize_cash_series_silver(
        catalog=ctx.catalog,
        universe_root=ctx.runtime.workspace.silver_root,
        silver_root=ctx.runtime.workspace.silver_root,
        config=load_cash_series_config(load_runtime_config().cash_series),
        universe_dataset_id=inputs["ordinary_universe"],
    )
    return _published(result.dataset_path)


def _build_investor_flow_ls(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build Silver LS investor-flow rows for the resolved universe."""
    from src.data.investor_flow_silver import (
        load_investor_flow_build_config,
        materialize_investor_flow_silver,
        parse_fingerprint,
    )
    from src.data.parse_cache import ParseCache

    with ParseCache(
        _parse_cache_path(ctx, "ls_investor_flow"),
        namespace="ls_investor_flow-v1",
        fingerprint=parse_fingerprint(),
    ) as cache:
        result = materialize_investor_flow_silver(
            catalog=ctx.catalog,
            universe_root=ctx.runtime.workspace.silver_root,
            silver_root=ctx.runtime.workspace.silver_root,
            universe_dataset_id=inputs["ordinary_universe"],
            parse_cache=cache,
            ticker_buckets=load_investor_flow_build_config().ticker_buckets,
        )
        cache.prune({blob.content_hash for blob in ctx.catalog.blobs(source="ls_investor_flow", usable=True)})
    return _published(result.dataset_path)


def _build_market_panel(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build the decision-safe Gold market panel from resolved inputs."""
    from src.config.runtime import load_runtime_config
    from src.core.market_rules import load_krx_market_rules
    from src.data.market_panel import materialize_market_panel

    runtime = ctx.runtime
    rules = load_krx_market_rules(load_runtime_config().market_rules)
    market_actions_id = inputs.get("market_actions")
    result = materialize_market_panel(
        daily_market_path=runtime.workspace.silver_root / inputs["daily_market"],
        universe_path=runtime.workspace.silver_root / inputs["ordinary_universe"],
        rules=rules,
        gold_root=runtime.workspace.gold_root,
        market_actions_path=(
            runtime.workspace.silver_root / market_actions_id if market_actions_id is not None else None
        ),
    )
    return _published(result.dataset_path)


def _daily_flag_cache_path(ctx: RefreshContext) -> Path:
    """Single cache location for per-session daily admin flags."""
    return ctx.runtime.workspace.state_root / "cache" / "daily_admin_flags.parquet"


def _parse_cache_path(ctx: RefreshContext, namespace: str) -> Path:
    """Single cache location for content-addressed parse results."""
    return ctx.runtime.workspace.state_root / "cache" / f"parse_{namespace}.sqlite"


def warm_daily_flag_cache(ctx: RefreshContext) -> int:
    """Populate the daily admin-flag cache from Bronze daily pages; returns covered session count."""
    from src.data.market_actions import _load_daily_flag_rows

    return len(_load_daily_flag_rows(ctx.catalog, _build_calendar(ctx), flag_cache=_daily_flag_cache_path(ctx)))


def _build_market_actions(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build Silver exchange market actions from disclosure titles and daily flags."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.jobs.universe import read_corp_code_bridge
    from src.data.market_actions import materialize_market_actions

    _ = inputs
    mapping, _ = read_corp_code_bridge(ctx.catalog)
    provider = load_provider_policy(load_runtime_config())
    path = materialize_market_actions(
        catalog=ctx.catalog,
        silver_root=ctx.runtime.workspace.silver_root,
        calendar=_build_calendar(ctx),
        bridge=dict(mapping),
        kind_keywords=tuple(provider.kind.search_keywords),
        kind_coverage_start=ctx.runtime.scope.evidence_start,
        kind_max_stale_sessions=provider.kind.max_stale_sessions,
        flag_cache=_daily_flag_cache_path(ctx),
    )
    return _published(path)


def _build_investor_flow_kis_supplement(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build the provider-tagged KIS supplement for the resolved LS gap."""
    from src.data.flow_targets import investor_flow_targets_from_ids
    from src.data.investor_flow_kis_supplement import materialize_investor_flow_kis_supplement, parse_fingerprint
    from src.data.parse_cache import ParseCache

    runtime = ctx.runtime
    targets = investor_flow_targets_from_ids(
        runtime.workspace.silver_root, inputs["ordinary_universe"], inputs["daily_market"]
    )
    with ParseCache(
        _parse_cache_path(ctx, "kis_investor_flow"),
        namespace="kis_investor_flow-v1",
        fingerprint=parse_fingerprint(),
    ) as cache:
        result = materialize_investor_flow_kis_supplement(
            catalog=ctx.catalog,
            targets=targets,
            ls_flow_silver_path=runtime.workspace.silver_root / inputs["investor_flow_ls"],
            silver_root=runtime.workspace.silver_root,
            parse_cache=cache,
        )
        cache.prune({blob.content_hash for blob in ctx.catalog.blobs(source="kis_investor_flow", usable=True)})
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
    from src.config import load_provider_policy, load_runtime_config
    from src.data.incremental_normalization import refresh_dart_financial_facts

    _ = inputs
    runtime = ctx.runtime
    provider = load_provider_policy(load_runtime_config())
    artifact = refresh_dart_financial_facts(
        bronze_root=runtime.workspace.bronze_root,
        silver_root=runtime.workspace.silver_root,
        artifact_root=runtime.workspace.state_root,
        decision_time=ctx.decision_time,
        calendar=_build_calendar(ctx),
        superseded_receipt_hashes=ctx.superseded_receipts,
        trust_document_facts=bool(provider.dart.document_parser.trusted),
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
    from src.config import load_provider_policy, load_runtime_config
    from src.data.dividend_events import materialize_dividend_events, parse_fingerprint
    from src.data.evidence_sources import DIVIDEND_DECISION_SOURCE
    from src.data.parse_cache import ParseCache

    runtime = ctx.runtime
    policy = load_provider_policy(load_runtime_config()).dividends
    with ParseCache(
        _parse_cache_path(ctx, "dividend_decision"),
        namespace="dividend_decision-v1",
        fingerprint=parse_fingerprint(),
    ) as cache:
        path = materialize_dividend_events(
            catalog=ctx.catalog,
            silver_root=runtime.workspace.silver_root,
            calendar=_build_calendar(ctx),
            daily_market_path=runtime.workspace.silver_root / inputs["daily_market"],
            policy=policy,
            parse_cache=cache,
        )
        cache.prune(
            {blob.content_hash for blob in ctx.catalog.blobs(source=DIVIDEND_DECISION_SOURCE, usable=True)}
        )
    return _published(path)


def _build_earnings_releases(ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
    """Build Silver early-earnings releases from preliminary and profit-change filings."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.earnings_releases import materialize_earnings_releases

    _ = inputs
    runtime = ctx.runtime
    policy = load_provider_policy(load_runtime_config()).earnings_releases
    path = materialize_earnings_releases(
        catalog=ctx.catalog,
        silver_root=runtime.workspace.silver_root,
        calendar=_build_calendar(ctx),
        policy=policy,
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
    from src.data.ordinary_universe import REVISION as ORDINARY_UNIVERSE_REVISION
    from src.data.ordinary_universe import catalog_master_receipts, catalog_master_sessions

    sessions = catalog_master_sessions(ctx.catalog)
    receipts = catalog_master_receipts(ctx.catalog, sessions=sessions)
    source_hashes = [receipt.content_hash for receipt in receipts]
    return DatasetIdentity(
        kind="ordinary_universe",
        layer=_Layer.SILVER,
        policy_version=ORDINARY_UNIVERSE_REVISION,
        inputs={"bronze_master": dataset_digest(source_hashes)},
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(session.isoformat() for session in sorted(sessions)).encode("utf-8")
            ).hexdigest()
        },
    )


def _preview_daily_market(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be daily-market identity from catalog Bronze digests."""
    from src.data.daily_market_silver import REVISION as DAILY_MARKET_SILVER_REVISION
    from src.data.daily_market_silver import DailyMarketSilverPolicy
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
        policy_version=DAILY_MARKET_SILVER_REVISION,
        inputs={
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
            "bronze_daily": dataset_digest(source_hashes),
        },
        params={
            "available_time": policy.available_time.isoformat(),
            "fluc_tolerance_pct": policy.fluc_tolerance_pct,
        },
    )


def _index_series_source_hashes(
    ctx: RefreshContext, *, source: str, sessions: tuple[date, ...]
) -> list[str]:
    """Content hashes of the certified index-series pages, failing closed on gaps."""
    from src.data.receipt_catalog import EvidenceStatus

    entries = ctx.catalog.latest(source=source, natural_keys={session.isoformat() for session in sessions})
    hashes: list[str] = []
    for session in sessions:
        entry = entries.get(session.isoformat())
        if entry is None or entry.status is not EvidenceStatus.SUCCESS:
            raise PITDataError(f"KRX index series page is missing for {session}")
        if entry.as_of != session:
            raise PITDataError(f"KRX index series catalog date conflicts for {session}")
        hashes.append(entry.content_hash)
    return hashes


def _preview_index_series(ctx: RefreshContext, *, config_path_kind: str) -> DatasetIdentity:
    """Identity the hedge/trend materializer publishes for the current catalog state."""
    from src.config import load_runtime_config
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.datasets import dataset_digest, dataset_reference, universe_sessions
    from src.data.evidence_sources import KRX_HEDGE_SERIES_SOURCE, KRX_TREND_SERIES_SOURCE
    from src.data.hedge_series_silver import REVISION as HEDGE_SERIES_REVISION
    from src.data.hedge_series_silver import load_hedge_series_config

    runtime_config = load_runtime_config()
    config = load_hedge_series_config(
        runtime_config.hedge_series if config_path_kind == "hedge" else runtime_config.trend_series
    )
    universe_id = ctx.registry.require("ordinary_universe")
    _, calendar = universe_sessions(ctx.runtime.workspace.silver_root, universe_id, allow_legacy=False)
    sessions = tuple(day for day in calendar if day >= config.collection_start)
    source_hashes = _index_series_source_hashes(ctx, source=config.source, sessions=sessions)
    params: dict[str, str] = {
        "collection_start": config.collection_start.isoformat(),
        "index_name": config.index_name,
        "inverse_ticker": config.inverse_ticker,
        "available_time": config.available_time.isoformat(),
    }
    if config.index_class != "KOSDAQ":
        params["index_class"] = config.index_class
    if config.source != KRX_HEDGE_SERIES_SOURCE:
        params["source"] = config.source
    return DatasetIdentity(
        kind="trend_series" if config.source == KRX_TREND_SERIES_SOURCE else "hedge_series",
        layer=_Layer.SILVER,
        policy_version=HEDGE_SERIES_REVISION,
        inputs={
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
            "bronze_hedge": dataset_digest(source_hashes),
        },
        params=params,
    )


def _preview_hedge_series(ctx: RefreshContext) -> DatasetIdentity:
    """Identity the hedge series build would publish for the current catalog and registry state."""
    return _preview_index_series(ctx, config_path_kind="hedge")


def _preview_trend_series(ctx: RefreshContext) -> DatasetIdentity:
    """Identity the trend series build would publish for the current catalog and registry state."""
    return _preview_index_series(ctx, config_path_kind="trend")


def _preview_cash_series(ctx: RefreshContext) -> DatasetIdentity:
    """Identity the cash series build would publish for the current catalog and registry state."""
    from src.config import load_runtime_config
    from src.data.cash_series_silver import REVISION as CASH_SERIES_REVISION
    from src.data.cash_series_silver import load_cash_series_config
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.datasets import dataset_digest, dataset_reference, universe_sessions
    from src.data.evidence_sources import KRX_CASH_SERIES_SOURCE

    config = load_cash_series_config(load_runtime_config().cash_series)
    universe_id = ctx.registry.require("ordinary_universe")
    _, calendar = universe_sessions(ctx.runtime.workspace.silver_root, universe_id, allow_legacy=False)
    sessions = tuple(day for day in calendar if day >= config.collection_start)
    source_hashes = _index_series_source_hashes(ctx, source=KRX_CASH_SERIES_SOURCE, sessions=sessions)
    return DatasetIdentity(
        kind="cash_series",
        layer=_Layer.SILVER,
        policy_version=CASH_SERIES_REVISION,
        inputs={
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
            "bronze_cash": dataset_digest(source_hashes),
        },
        params={
            "collection_start": config.collection_start.isoformat(),
            "ticker": config.ticker,
            "available_time": config.available_time.isoformat(),
        },
    )


def _preview_investor_flow_ls(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be LS flow identity from the catalog blob digest."""
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.investor_flow_silver import REVISION as INVESTOR_FLOW_SILVER_REVISION
    from src.data.investor_flow_silver import InvestorFlowSilverPolicy

    universe_id = ctx.registry.require("ordinary_universe")
    bronze_flow = ctx.catalog.blob_digest(source="ls_investor_flow")
    policy = InvestorFlowSilverPolicy()
    return DatasetIdentity(
        kind="investor_flow_ls",
        layer=_Layer.SILVER,
        policy_version=INVESTOR_FLOW_SILVER_REVISION,
        inputs={
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
            "bronze_flow": bronze_flow,
        },
        params={
            "available_session_lag": policy.available_session_lag,
            "available_time": policy.available_time.isoformat(),
        },
    )


def _preview_market_actions(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be market-actions identity from Bronze digests."""
    import hashlib as _hashlib

    from src.core.time import KRX_TZ
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.jobs.universe import read_corp_code_bridge as _read_bridge
    from src.data.kind_notices import kind_source_digest
    from src.data.market_actions import REVISION as MARKET_ACTIONS_REVISION
    from src.data.market_actions import (
        corp_bridge_digest,
        daily_flags_source_digest,
        disclosure_source_digest,
        market_actions_dataset_inputs,
    )

    mapping, _ = _read_bridge(ctx.catalog)
    calendar = _build_calendar(ctx)
    return DatasetIdentity(
        kind="market_actions",
        layer=_Layer.SILVER,
        policy_version=MARKET_ACTIONS_REVISION,
        inputs=market_actions_dataset_inputs(
            bronze_disclosures=disclosure_source_digest(ctx.catalog),
            corp_code_bridge=corp_bridge_digest(dict(mapping)),
            bronze_daily=daily_flags_source_digest(ctx.catalog, calendar),
            bronze_kind=kind_source_digest(ctx.catalog),
        ),
        params={
            "calendar_digest": _hashlib.sha256(
                "\n".join(session.astimezone(KRX_TZ).isoformat() for session in calendar.sessions).encode(
                    "utf-8"
                )
            ).hexdigest(),
        },
    )


def _preview_market_panel(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be market-panel identity from resolved inputs and rules."""
    from src.config.runtime import load_runtime_config
    from src.core.market_rules import load_krx_market_rules
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.market_panel import REVISION as MARKET_PANEL_REVISION
    from src.data.market_panel import MarketPanelPolicy, _rules_fingerprint

    daily_id = ctx.registry.require("daily_market")
    universe_id = ctx.registry.require("ordinary_universe")
    market_actions_id = ctx.registry.require("market_actions")
    rules = load_krx_market_rules(load_runtime_config().market_rules)
    policy = MarketPanelPolicy()
    return DatasetIdentity(
        kind="market_panel",
        layer=_Layer.GOLD,
        policy_version=MARKET_PANEL_REVISION,
        inputs={
            "daily_market": dataset_reference(daily_id, kind="daily_market"),
            "universe": dataset_reference(universe_id, kind="ordinary_universe"),
            "market_actions": dataset_reference(market_actions_id, kind="market_actions"),
        },
        params={
            "adtv_short_sessions": policy.adtv_short_sessions,
            "adtv_long_sessions": policy.adtv_long_sessions,
            "return_vol_sessions": policy.return_vol_sessions,
            "block_administrative": policy.block_administrative,
            "delisting_block_max_sessions": policy.delisting_block_max_sessions,
            "rules_version": rules.version,
            "rules_fingerprint": _rules_fingerprint(rules),
        },
    )


def _preview_investor_flow_kis_supplement(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be KIS supplement identity from the catalog digest."""
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.investor_flow_kis_supplement import REVISION as INVESTOR_FLOW_KIS_SUPPLEMENT_REVISION
    from src.data.investor_flow_kis_supplement import InvestorFlowKisSupplementPolicy

    universe_id = ctx.registry.require("ordinary_universe")
    daily_id = ctx.registry.require("daily_market")
    ls_id = ctx.registry.require("investor_flow_ls")
    bronze_kis = ctx.catalog.blob_digest(source="kis_investor_flow")
    policy = InvestorFlowKisSupplementPolicy()
    return DatasetIdentity(
        kind="investor_flow_kis_supplement",
        layer=_Layer.SILVER,
        policy_version=INVESTOR_FLOW_KIS_SUPPLEMENT_REVISION,
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
    from src.data.investor_flow_union import REVISION as INVESTOR_FLOW_UNION_REVISION

    ls_id = ctx.registry.require("investor_flow_ls")
    kis_id = ctx.registry.require("investor_flow_kis_supplement")
    return DatasetIdentity(
        kind="investor_flow",
        layer=_Layer.SILVER,
        policy_version=INVESTOR_FLOW_UNION_REVISION,
        inputs={
            "ls": dataset_reference(ls_id, kind="investor_flow_ls"),
            "kis_supplement": dataset_reference(kis_id, kind="investor_flow_kis_supplement"),
        },
        params={},
    )


def _preview_industry(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be industry identity from the catalog blob digest."""
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.industry_silver import REVISION as INDUSTRY_SILVER_REVISION

    bronze_classification = ctx.catalog.blob_digest(source="kis_industry")
    if bronze_classification == dataset_digest([]):
        raise PITDataError("no certified INDUSTRY Bronze evidence found")
    return DatasetIdentity(
        kind="industry",
        layer=_Layer.SILVER,
        policy_version=INDUSTRY_SILVER_REVISION,
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
    receipts = _discover_fact_receipts(runtime.workspace.bronze_root, verify_payload=False)
    # 빌더와 같이 대체된 수집 기록은 입력 요약에서 제외한다(포함하면 갱신 판정이 영원히 어긋난다).
    receipt_hashes = [
        str(item["content_hash"]) for item in receipts if str(item["content_hash"]) not in ctx.superseded_receipts
    ]
    from src.data.datasets import read_dataset as _read_dataset

    disclosure_digest = dataset_digest([])
    candidates = sorted(
        (path for path in runtime.workspace.silver_root.glob("disclosures_*")
         if path.is_dir() and not path.is_symlink()),
        reverse=True,
    )
    for candidate in candidates:
        try:
            _load(candidate)
            _read_dataset(candidate).collect()
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
    from src.config import load_provider_policy, load_runtime_config
    from src.data.incremental_normalization import REVISION as INCREMENTAL_NORMALIZATION_REVISION
    from src.integrations.dart.document_statements import REVISION as DOCUMENT_STATEMENTS_REVISION

    try:
        _trusted = bool(load_provider_policy(load_runtime_config()).dart.document_parser.trusted)
    except Exception:  # pragma: no cover - missing provider preview fallback
        _trusted = False
    return DatasetIdentity(
        kind="financial_facts",
        layer=_Layer.SILVER,
        policy_version=INCREMENTAL_NORMALIZATION_REVISION,
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
            "trust_document_facts": _trusted,
            "document_parser_version": DOCUMENT_STATEMENTS_REVISION,
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
    from src.data.dividend_events import REVISION as DIVIDEND_EVENTS_REVISION
    from src.data.dividend_events import (
        _iter_decision_envelopes,
        dividend_dataset_inputs,
        dividend_policy_params,
    )

    envelopes = _iter_decision_envelopes(ctx.catalog)
    envelope_hashes = [
        hashlib.sha256(
            _json.dumps(envelope, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        for envelope in envelopes
    ]
    try:
        from src.data.jobs.universe import read_corp_code_bridge as _read_bridge

        _, bridge_receipt_hash = _read_bridge(ctx.catalog)
    except PITDataError:  # pragma: no cover - missing bridge preview fallback
        bridge_receipt_hash = ""
    from src.config import load_provider_policy, load_runtime_config

    policy = load_provider_policy(load_runtime_config()).dividends
    daily_market_id = ctx.registry.require("daily_market")
    calendar = _build_calendar(ctx)
    return DatasetIdentity(
        kind="dividend_events",
        layer=_Layer.SILVER,
        policy_version=DIVIDEND_EVENTS_REVISION,
        inputs=dividend_dataset_inputs(
            bronze_dividend_decisions=dataset_digest(envelope_hashes),
            corp_code_bridge=dataset_digest([bridge_receipt_hash]),
            daily_market_id=daily_market_id,
        ),
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(
                    session.astimezone(KRX_TZ).date().isoformat() for session in calendar.sessions
                ).encode("utf-8")
            ).hexdigest(),
            **dividend_policy_params(policy),
        },
    )


def _preview_earnings_releases(ctx: RefreshContext) -> DatasetIdentity:
    """Compute the would-be earnings-releases identity from release envelopes."""
    import json as _json

    from src.core.time import KRX_TZ
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.earnings_releases import REVISION as EARNINGS_RELEASES_REVISION
    from src.data.earnings_releases import _iter_release_envelopes, earnings_release_dataset_inputs

    envelopes = _iter_release_envelopes(ctx.catalog)
    ordered = sorted(envelopes, key=lambda item: str(item.get("rcept_no", "")))
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for envelope in ordered:
        key = str(envelope.get("rcept_no", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(envelope)
    envelope_hashes = [
        hashlib.sha256(
            _json.dumps(envelope, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        for envelope in unique
    ]
    try:
        from src.data.jobs.universe import read_corp_code_bridge as _read_bridge

        _, bridge_receipt_hash = _read_bridge(ctx.catalog)
    except PITDataError:  # pragma: no cover - missing bridge preview fallback
        bridge_receipt_hash = ""
    from src.config import load_provider_policy, load_runtime_config

    policy = load_provider_policy(load_runtime_config()).earnings_releases
    calendar = _build_calendar(ctx)
    return DatasetIdentity(
        kind="earnings_releases",
        layer=_Layer.SILVER,
        policy_version=EARNINGS_RELEASES_REVISION,
        inputs=earnings_release_dataset_inputs(
            bronze_earnings_releases=dataset_digest(envelope_hashes),
            corp_code_bridge=dataset_digest([bridge_receipt_hash]),
        ),
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(
                    session.astimezone(KRX_TZ).date().isoformat() for session in calendar.sessions
                ).encode("utf-8")
            ).hexdigest(),
            "max_filing_lag_days": policy.max_filing_lag_days,
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
    "hedge_series": _preview_hedge_series,
    "trend_series": _preview_trend_series,
    "cash_series": _preview_cash_series,
    "investor_flow_ls": _preview_investor_flow_ls,
    "market_actions": _preview_market_actions,
    "market_panel": _preview_market_panel,
    "investor_flow_kis_supplement": _preview_investor_flow_kis_supplement,
    "investor_flow": _preview_investor_flow,
    "industry": _preview_industry,
    "financial_facts": _preview_financial_facts,
    "financial_quality": _preview_financial_quality,
    "dividend_events": _preview_dividend_events,
    "earnings_releases": _preview_earnings_releases,
    "reference_benchmarks": _preview_reference_benchmarks,
}

SCOPE_GRAPH: Final[tuple[BuildNode, ...]] = (
    BuildNode(kind="ordinary_universe", inputs=(), bronze_sources=("krx_security_master",), build=_build_ordinary_universe),
    BuildNode(kind="industry", inputs=(), bronze_sources=("industry",), build=_build_industry),
    BuildNode(kind="financial_facts", inputs=(), bronze_sources=("financial_facts",), build=_build_financial_facts),
    BuildNode(
        kind="daily_market",
        inputs=("ordinary_universe",),
        bronze_sources=("krx_daily_market",),
        build=_build_daily_market,
    ),
    BuildNode(
        kind="hedge_series",
        inputs=("ordinary_universe",),
        bronze_sources=("krx_hedge_series",),
        build=_build_hedge_series,
    ),
    BuildNode(
        kind="trend_series",
        inputs=("ordinary_universe",),
        bronze_sources=("krx_trend_series",),
        build=_build_trend_series,
    ),
    BuildNode(
        kind="cash_series",
        inputs=("ordinary_universe",),
        bronze_sources=("krx_cash_series",),
        build=_build_cash_series,
    ),
    BuildNode(
        kind="dividend_events",
        inputs=("daily_market",),
        bronze_sources=("corporate_actions",),
        build=_build_dividend_events,
    ),
    BuildNode(kind="earnings_releases", inputs=(), bronze_sources=("disclosures",), build=_build_earnings_releases),
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
        kind="market_actions",
        inputs=("daily_market",),
        bronze_sources=("corporate_actions", "krx_daily_market"),
        build=_build_market_actions,
    ),
    BuildNode(
        kind="market_panel",
        inputs=("daily_market", "ordinary_universe", "market_actions"),
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
        if node.kind in ctx.deferred_build_kinds:
            continue
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
    """Collect physically present dataset ids for lineage verification."""
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
    peaks: dict[str, float] = {}
    known = _disk_dataset_ids(ctx, set(snapshot.values()))
    for node in plan:
        try:
            resolved = {
                dependency: built[dependency].dataset_id if dependency in built else snapshot[dependency]
                for dependency in node.inputs
            }
        except KeyError as exc:
            raise PITDataError(f"refresh input is not registered: {exc}") from exc
        outcome, peak = _build_with_peak(node, ctx, MappingProxyType(resolved))
        verification = verify_dataset(outcome.path, known_ids=known.__contains__)
        if not verification.passed:
            raise PITDataError(f"refresh verification failed for {node.kind}: {'; '.join(verification.failures)}")
        built[node.kind] = outcome
        known.add(outcome.dataset_id)
        if peak is not None:
            peaks[node.kind] = peak
        emit(_dataset_payload(node.kind, outcome.dataset_id, peak))
        _log_build(node.kind, outcome.dataset_id, peak)
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
        node_peak_rss_mb=MappingProxyType(dict(peaks)),
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
            host_pacer_path=ctx.runtime.workspace.state_root / "quota" / "dart_host_pacer.json",
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
    job_ctx = replace(job_ctx, retry_empty_session=ctx.retry_krx_session)
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


def _run_dbsec_collection_step(
    ctx: RefreshContext, job_name: str, *, dry_run: bool, emit: Callable[[Mapping[str, object]], None]
) -> CollectionStepReport:
    """Run one DB Securities fallback collection job without duplicating runner wiring."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.jobs.dbsec import build_dbsec_job_context, resolve_dbsec_job
    from src.data.jobs.runner import run_job
    from src.integrations.dbsec.client import build_scoped_dbsec_client
    from src.integrations.quota import ProviderQuotaStateStore

    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    collector = None
    if not dry_run:  # pragma: no cover - live provider path
        collector = build_scoped_dbsec_client(
            policy=provider.dbsec,
            quota_store=ProviderQuotaStateStore(ctx.runtime.workspace.state_root / "quota"),
        )
    job_ctx = build_dbsec_job_context(runtime=ctx.runtime, provider=provider, collector=collector)
    report = run_job(
        resolve_dbsec_job(job_name),
        job_ctx,
        chunk_size=int(provider.dbsec.chunk_units),
        max_chunks=None,
        dry_run=dry_run,
        emit=emit,
    )
    return CollectionStepReport(
        job=job_name, status=report.status, done=report.done,
        pending_left=report.pending_left, requests_used=report.requests_used,
    )


def _run_kind_collection_step(
    ctx: RefreshContext, job_name: str, *, dry_run: bool, emit: Callable[[Mapping[str, object]], None]
) -> CollectionStepReport:
    """Run one KIND notice collection job without duplicating runner wiring."""
    from src.config import load_provider_policy, load_runtime_config
    from src.config.secrets import environment_snapshot
    from src.data.kind_relay import kind_transport

    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    if kind_transport(environment_snapshot()) == "relay":
        from src.data.kind_relay_import import import_relay_bundle

        relay = import_relay_bundle(
            runtime=ctx.runtime, provider=provider, job=job_name,
            now=datetime.now(UTC), dry_run=dry_run,
        )
        emit(
            {
                "type": "collection",
                "job": job_name,
                "status": relay.status,
                "fetched_at": relay.fetched_at.isoformat() if relay.fetched_at is not None else None,
                "imported": relay.imported,
                "unchanged": relay.unchanged,
                "rejected": relay.rejected,
                "pending_left": relay.pending_left,
                "requests_used": 0,
            }
        )
        _LOG.info(
            "[DATA] stage=refresh_collect job=%s status=%s imported=%d unchanged=%d rejected=%d",
            job_name,
            relay.status,
            relay.imported,
            relay.unchanged,
            relay.rejected,
        )
        return CollectionStepReport(
            job=job_name, status=relay.status, done=relay.imported + relay.unchanged,
            pending_left=relay.pending_left, requests_used=0,
        )
    from src.data.jobs.kind import KIND_CHUNK_SIZE, build_kind_job_context, resolve_kind_job
    from src.data.jobs.runner import run_job
    from src.integrations.krx.kind import build_scoped_kind_client
    from src.integrations.quota import ProviderQuotaStateStore

    collector = None
    if not dry_run:  # pragma: no cover - live provider path
        collector = build_scoped_kind_client(
            policy=provider.kind,
            quota_store=ProviderQuotaStateStore(ctx.runtime.workspace.state_root / "quota"),
        )
    job_ctx = build_kind_job_context(runtime=ctx.runtime, provider=provider, collector=collector)
    report = run_job(
        resolve_kind_job(job_name),
        job_ctx,
        chunk_size=KIND_CHUNK_SIZE,
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
        if provider.kis.pool_slot is not None:
            from src.integrations.kis import build_pool_client

            collector = KisInvestorFlowCollector(client=build_pool_client(provider.kis))
        else:
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


def _run_industry_collection_step(
    ctx: RefreshContext, *, dry_run: bool, emit: Callable[[Mapping[str, object]], None]
) -> CollectionStepReport:
    """Top up this ISO week's KIS industry snapshot for current-universe tickers that lack one.

    KIS serves only the current classification, so the weekly snapshot is the point-in-time
    record. Reruns within the week fetch only missing tickers, so a covered week costs no request.
    ``pending_left`` counts tickers skipped this run (retried on the next run of the week).
    """
    from functools import partial

    import src.integrations.kis as kis
    from src.data.industry_collection import (
        collect_classification_with_isolation,
        current_universe_tickers,
        tickers_missing_weekly_snapshot,
    )
    from src.data.scoped_ingestion import ScopedBronzeWriter
    from src.integrations.kis import industry as kis_industry

    _ = emit
    universe = ctx.runtime.workspace.silver_root / ctx.registry.require("ordinary_universe")
    pending = tickers_missing_weekly_snapshot(
        ctx.catalog, current_universe_tickers(universe), today=datetime.now(UTC).date()
    )
    if not pending or dry_run:
        return CollectionStepReport(
            job="kis_industry", status="dry_run" if pending else "complete",
            done=0, pending_left=len(pending), requests_used=0,
        )
    summary = collect_classification_with_isolation(
        stage="kis_industry",
        collector_cls=partial(kis_industry.KisIndustryCollector, client=kis.default_client()),
        fetch_attr="fetch_industry_classification",
        bronze_root=ctx.runtime.workspace.bronze_root,
        writer=ScopedBronzeWriter(runtime=ctx.runtime, catalog=ctx.catalog),
        symbols=pending,
        # One shared client: its transport already enforces KisPolicy.min_interval_seconds per call.
        pace_seconds=0.0,
    )
    skipped = sorted(cast("dict[str, str]", summary["skipped"]))
    if skipped:
        _LOG.warning(
            "[DATA] stage=kis_industry status=partial requested=%d skipped=%d head=%s",
            len(pending), len(skipped), ",".join(skipped[:20]),
        )
    return CollectionStepReport(
        job="kis_industry", status="partial" if skipped else "complete",
        done=cast(int, summary["pages_collected"]), pending_left=len(skipped), requests_used=len(pending),
    )


CollectionStep = tuple[str, Callable[..., CollectionStepReport]]


def _default_collection_steps() -> tuple[CollectionStep, ...]:
    """Collection jobs in refresh order, from disclosures to investor flow."""
    return (
        (
            "dart_corp_codes",
            lambda ctx, *, dry_run, emit: _run_dart_collection_step(
                ctx, "dart_corp_codes", dry_run=dry_run, emit=emit
            ),
        ),
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
            "earnings_releases",
            lambda ctx, *, dry_run, emit: _run_dart_collection_step(
                ctx, "earnings_releases", dry_run=dry_run, emit=emit
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
        (
            "krx_hedge_series",
            lambda ctx, *, dry_run, emit: _run_krx_collection_step(
                ctx, "krx_hedge_series", dry_run=dry_run, emit=emit
            ),
        ),
        (
            "krx_trend_series",
            lambda ctx, *, dry_run, emit: _run_krx_collection_step(
                ctx, "krx_trend_series", dry_run=dry_run, emit=emit
            ),
        ),
        (
            "krx_cash_series",
            lambda ctx, *, dry_run, emit: _run_krx_collection_step(
                ctx, "krx_cash_series", dry_run=dry_run, emit=emit
            ),
        ),
        (
            "kind_notice_search",
            lambda ctx, *, dry_run, emit: _run_kind_collection_step(
                ctx, "kind_notice_search", dry_run=dry_run, emit=emit
            ),
        ),
        (
            "kind_notice_documents",
            lambda ctx, *, dry_run, emit: _run_kind_collection_step(
                ctx, "kind_notice_documents", dry_run=dry_run, emit=emit
            ),
        ),
        ("ls_investor_flow", _run_ls_flow_collection_step),
        ("kis_investor_flow", _run_kis_flow_collection_step),
        ("kis_industry", _run_industry_collection_step),
        (
            "dbsec_ticker_master",
            lambda ctx, *, dry_run, emit: _run_dbsec_collection_step(
                ctx, "dbsec_ticker_master", dry_run=dry_run, emit=emit
            ),
        ),
        (
            "dbsec_daily_chart",
            lambda ctx, *, dry_run, emit: _run_dbsec_collection_step(
                ctx, "dbsec_daily_chart", dry_run=dry_run, emit=emit
            ),
        ),
        (
            "dbsec_closing_bar",
            lambda ctx, *, dry_run, emit: _run_dbsec_collection_step(
                ctx, "dbsec_closing_bar", dry_run=dry_run, emit=emit
            ),
        ),
    )


def run_collection_jobs(
    ctx: RefreshContext,
    *,
    dry_run: bool,
    emit: Callable[[Mapping[str, object]], None],
    steps: tuple[CollectionStep, ...] | None = None,
) -> CollectionReport:
    """Run collection steps in order and report each; a step ``Exception`` becomes a ``failed`` report."""
    reports: list[CollectionStepReport] = []
    emit_failure: Exception | None = None

    def step_emit(payload: Mapping[str, object]) -> None:
        nonlocal emit_failure
        try:
            emit(payload)
        except Exception as exc:
            emit_failure = exc
            raise

    for name, step in steps if steps is not None else _default_collection_steps():
        emit_failure = None
        try:
            report = step(ctx, dry_run=dry_run, emit=step_emit)
        except Exception as exc:
            if exc is emit_failure:
                raise
            from src.data.jobs.dart import DisclosureCoverageError

            if isinstance(exc, DisclosureCoverageError) and name not in OPTIONAL_COLLECTION_JOBS:
                report = CollectionStepReport(
                    job=name, status="coverage_gap", done=0, pending_left=0, requests_used=0,
                )
                emit(
                    {
                        "type": "collection",
                        "job": report.job,
                        "status": report.status,
                        "done": report.done,
                        "pending_left": report.pending_left,
                        "requests_used": report.requests_used,
                        "periods": list(exc.periods),
                        "error": str(exc),
                    }
                )
                _LOG.warning(
                    "[DATA] stage=refresh_collect job=%s status=%s periods=%s",
                    report.job,
                    report.status,
                    ",".join(exc.periods),
                )
                reports.append(report)
                continue
            if name in OPTIONAL_COLLECTION_JOBS:
                report = CollectionStepReport(
                    job=name, status="failed", done=0, pending_left=0, requests_used=0,
                )
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
                _LOG.warning(
                    "[DATA] stage=refresh_collect job=%s status=failed error=%s",
                    report.job,
                    type(exc).__name__,
                )
                reports.append(report)
                continue
            report = CollectionStepReport(
                job=name, status="failed", done=0, pending_left=0, requests_used=0,
            )
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
            _LOG.error(
                "[DATA] stage=refresh_collect job=%s status=failed error=%s",
                report.job,
                type(exc).__name__,
                exc_info=exc,
            )
            reports.append(report)
            continue
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
    return CollectionReport(steps=tuple(reports), optional=OPTIONAL_COLLECTION_JOBS)


DART_COLLECTION_STEPS: Final[frozenset[str]] = frozenset({
    "dart_corp_codes",
    "dart_disclosures",
    "dart_facts",
    "dividend_decisions",
    "earnings_releases",
})

NODE_COLLECTION_GATES: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    "ordinary_universe": frozenset({"krx_security_master"}),
    "daily_market": frozenset({"krx_daily_market"}),
    "hedge_series": frozenset({"krx_hedge_series"}),
    "trend_series": frozenset({"krx_trend_series"}),
    "cash_series": frozenset({"krx_cash_series"}),
    "investor_flow_ls": frozenset({"ls_investor_flow"}),
    "investor_flow_kis_supplement": frozenset({"kis_investor_flow"}),
    "financial_facts": frozenset(DART_COLLECTION_STEPS),
    "dividend_events": frozenset(DART_COLLECTION_STEPS),
    "earnings_releases": frozenset(DART_COLLECTION_STEPS),
    "market_actions": frozenset(DART_COLLECTION_STEPS | {"krx_daily_market"}),
    "industry": frozenset(),
    "financial_quality": frozenset(),
    "market_panel": frozenset(),
    "investor_flow": frozenset(),
    "reference_benchmarks": frozenset(),
})

KIS_PREREQUISITE_KINDS: Final[tuple[str, ...]] = (
    "ordinary_universe",
    "daily_market",
    "investor_flow_ls",
)

FLOW_COMPLETION_KINDS: Final[tuple[str, ...]] = (
    "investor_flow_kis_supplement",
    "investor_flow",
)


def _effective_gate(
    kind: str,
    inputs_by_kind: Mapping[str, tuple[str, ...]],
    memo: dict[str, frozenset[str]],
) -> frozenset[str]:
    """Own gate plus transitive input gates for one kind."""
    cached = memo.get(kind)
    if cached is not None:
        return cached
    own = NODE_COLLECTION_GATES.get(kind, frozenset())
    union: set[str] = set(own)
    for dependency in inputs_by_kind.get(kind, ()):
        union |= set(_effective_gate(dependency, inputs_by_kind, memo))
    result = frozenset(union)
    memo[kind] = result
    return result


def _closure(seeds: tuple[str, ...], inputs_by_kind: Mapping[str, tuple[str, ...]]) -> set[str]:
    """Seed kinds plus their transitive inputs."""
    closed: set[str] = set()
    stack = list(seeds)
    while stack:
        kind = stack.pop()
        if kind in closed:
            continue
        if kind not in inputs_by_kind:
            continue
        closed.add(kind)
        stack.extend(inputs_by_kind[kind])
    return closed


def _phase_split(plan: tuple[BuildNode, ...]) -> tuple[tuple[BuildNode, ...], tuple[BuildNode, ...], tuple[BuildNode, ...]]:
    """Split a plan into (KIS prerequisites, flow completion, remainder), each in graph order."""
    inputs_by_kind = {node.kind: node.inputs for node in _VALIDATED_GRAPH}
    present = set(inputs_by_kind)
    phase_a = _closure(KIS_PREREQUISITE_KINDS, inputs_by_kind) & present
    phase_b = (_closure(FLOW_COMPLETION_KINDS, inputs_by_kind) & present) - phase_a
    wanted = {node.kind for node in plan}
    first = tuple(node for node in _VALIDATED_GRAPH if node.kind in phase_a and node.kind in wanted)
    flow_done = tuple(node for node in _VALIDATED_GRAPH if node.kind in phase_b and node.kind in wanted)
    rest = tuple(node for node in _VALIDATED_GRAPH if node.kind not in phase_a and node.kind not in phase_b and node.kind in wanted)
    return (first, flow_done, rest)


def _blocked_kinds(
    plan: tuple[BuildNode, ...], incomplete_steps: frozenset[str]
) -> frozenset[str]:
    """Kinds in ``plan`` whose effective gate intersects ``incomplete_steps``."""
    if not incomplete_steps:
        return frozenset()
    inputs_by_kind = {node.kind: node.inputs for node in _VALIDATED_GRAPH}
    memo: dict[str, frozenset[str]] = {}
    return frozenset(
        node.kind
        for node in plan
        if _effective_gate(node.kind, inputs_by_kind, memo) & incomplete_steps
    )


def _build_nodes(
    ctx: RefreshContext,
    nodes: tuple[BuildNode, ...],
    snapshot: dict[str, str],
    built: dict[str, PublishedDataset],
    known: set[str],
    emit: Callable[[Mapping[str, object]], None],
    peaks: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build one ordered subset, verifying each and emitting progress."""
    owned: dict[str, float] = {} if peaks is None else peaks
    for node in nodes:
        try:
            resolved = {
                dependency: built[dependency].dataset_id if dependency in built else snapshot[dependency]
                for dependency in node.inputs
            }
        except KeyError as exc:
            raise PITDataError(f"refresh input is not registered: {exc}") from exc
        outcome, peak = _build_with_peak(node, ctx, MappingProxyType(resolved))
        verification = verify_dataset(outcome.path, known_ids=known.__contains__)
        if not verification.passed:
            raise PITDataError(f"refresh verification failed for {node.kind}: {'; '.join(verification.failures)}")
        built[node.kind] = outcome
        known.add(outcome.dataset_id)
        if peak is not None:
            owned[node.kind] = peak
        emit(_dataset_payload(node.kind, outcome.dataset_id, peak))
        _log_build(node.kind, outcome.dataset_id, peak)
    return owned


def refresh_scope(
    ctx: RefreshContext,
    *,
    collect: bool,
    dry_run: bool,
    emit: Callable[[Mapping[str, object]], None],
    steps: tuple[CollectionStep, ...] | None = None,
    collect_steps: frozenset[str] | None = None,
) -> RefreshReport:
    """Collect, then build flow prerequisites, collect KIS, build flow completion, then the remainder."""
    resolved_steps = steps if steps is not None else _default_collection_steps()
    if collect_steps is not None:
        known_steps = {name for name, _ in resolved_steps}
        unknown_steps = sorted(set(collect_steps) - known_steps)
        if unknown_steps:
            raise ValueError(f"unknown collection steps: {unknown_steps}")
        resolved_steps = tuple(item for item in resolved_steps if item[0] in collect_steps)
    has_kis = any(name == "kis_investor_flow" for name, _ in resolved_steps)
    if not collect or not has_kis:
        summaries: tuple[Mapping[str, object], ...] = ()
        if collect:
            collection = run_collection_jobs(ctx, dry_run=dry_run, emit=emit, steps=resolved_steps)
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
            if not dry_run and collection.blocking_job is not None:
                return RefreshReport(
                    decision_time=ctx.decision_time,
                    status="blocked",
                    planned=(),
                    built=(),
                    datasets=MappingProxyType({}),
                    blocking_job=collection.blocking_job,
                    collection=summaries,
                    degraded_jobs=collection.degraded_jobs,
                )
        report = run_refresh(ctx, dry_run=dry_run, emit=emit)
        if dry_run or not collect:
            return replace(report, collection=summaries)
        return replace(report, collection=summaries, degraded_jobs=collection.degraded_jobs)
    pre_steps = tuple(item for item in resolved_steps if item[0] != "kis_investor_flow")
    kis_steps = tuple(item for item in resolved_steps if item[0] == "kis_investor_flow")
    if dry_run and collect:
        pre_collection = run_collection_jobs(ctx, dry_run=True, emit=emit, steps=pre_steps)
        plan = plan_refresh(ctx)
        first, flow_done, rest = _phase_split(plan)
        for kind in tuple(node.kind for node in first):
            emit({"type": "plan", "kind": kind, "status": "stale"})
            _LOG.info("[DATA] stage=refresh_plan kind=%s status=stale", kind)
        kis_collection = run_collection_jobs(ctx, dry_run=True, emit=emit, steps=kis_steps)
        for kind in tuple(node.kind for node in (*flow_done, *rest)):
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
    inputs_by_kind = {node.kind: node.inputs for node in _VALIDATED_GRAPH}
    memo: dict[str, frozenset[str]] = {}
    full_phase_a, _, _ = _phase_split(tuple(_VALIDATED_GRAPH))
    flow_gate: set[str] = set()
    for node in full_phase_a:
        flow_gate |= set(_effective_gate(node.kind, inputs_by_kind, memo))
    if any(
        step.job in flow_gate and step.status != "complete" for step in pre_collection.steps
    ):
        return RefreshReport(
            decision_time=ctx.decision_time,
            status="blocked",
            planned=(),
            built=(),
            datasets=MappingProxyType({}),
            blocking_job=pre_collection.blocking_job,
            collection=tuple(pre_summaries),
            degraded_jobs=pre_collection.degraded_jobs,
        )
    plan = plan_refresh(ctx)
    planned_kinds = tuple(node.kind for node in plan)
    for kind in planned_kinds:
        emit({"type": "plan", "kind": kind, "status": "stale"})
        _LOG.info("[DATA] stage=refresh_plan kind=%s status=stale", kind)
    order_index = {node.kind: position for position, node in enumerate(_VALIDATED_GRAPH)}
    first, _, _ = _phase_split(plan)
    snapshot = dict(ctx.registry.snapshot())
    built: dict[str, PublishedDataset] = {}
    peaks: dict[str, float] = {}
    known = _disk_dataset_ids(ctx, set(snapshot.values()))
    _build_nodes(ctx, first, snapshot, built, known, emit, peaks)
    if any(node.kind in built for node in first):
        ctx.registry.register_many(
            {node.kind: built[node.kind].dataset_id for node in first if node.kind in built}
        )
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
    degraded = tuple(pre_collection.degraded_jobs) + tuple(kis_collection.degraded_jobs)
    if not kis_collection.complete:
        blocking = pre_collection.blocking_job or kis_collection.blocking_job
        ordered_built = tuple(sorted(built, key=lambda kind: order_index.get(kind, 0)))
        return RefreshReport(
            decision_time=ctx.decision_time,
            status="blocked",
            planned=planned_kinds,
            built=ordered_built,
            datasets=MappingProxyType({kind: built[kind].dataset_id for kind in ordered_built}),
            blocking_job=blocking,
            collection=summaries,
            degraded_jobs=degraded,
            node_peak_rss_mb=MappingProxyType({kind: peaks[kind] for kind in ordered_built if kind in peaks}),
        )
    remaining = plan_refresh(ctx)
    _, flow_done, rest = _phase_split(remaining)
    _build_nodes(ctx, flow_done, snapshot, built, known, emit, peaks)
    if any(node.kind in built for node in flow_done):
        ctx.registry.register_many(
            {node.kind: built[node.kind].dataset_id for node in flow_done if node.kind in built}
        )
        snapshot = dict(ctx.registry.snapshot())
    incomplete_required = frozenset(
        step.job
        for step in (*pre_collection.steps, *kis_collection.steps)
        if step.status != "complete" and step.job not in OPTIONAL_COLLECTION_JOBS
    )
    step_order = [step.job for step in (*pre_collection.steps, *kis_collection.steps)]
    blocked = _blocked_kinds(rest, incomplete_required)
    blocked_memo: dict[str, frozenset[str]] = {}
    buildable = []
    for node in rest:
        if node.kind not in blocked:
            buildable.append(node)
            continue
        gate = _effective_gate(node.kind, inputs_by_kind, blocked_memo)
        blocker = next((job for job in step_order if job in gate and job in incomplete_required), None)
        emit({"type": "plan", "kind": node.kind, "status": "skipped_blocked", "blocking_job": blocker})
        _LOG.warning(
            "[DATA] stage=refresh_build kind=%s status=skipped_blocked gate=%s",
            node.kind,
            blocker,
        )
    _build_nodes(ctx, tuple(buildable), snapshot, built, known, emit, peaks)
    if any(node.kind in built for node in buildable):
        ctx.registry.register_many(
            {node.kind: built[node.kind].dataset_id for node in buildable if node.kind in built}
        )
    ordered_built = tuple(sorted(built, key=lambda kind: order_index.get(kind, 0)))
    ordered_datasets = MappingProxyType({kind: built[kind].dataset_id for kind in ordered_built})
    ordered_peaks = MappingProxyType({kind: peaks[kind] for kind in ordered_built if kind in peaks})
    if pre_collection.blocking_job is not None:
        blocking = pre_collection.blocking_job
        return RefreshReport(
            decision_time=ctx.decision_time,
            status="blocked",
            planned=planned_kinds,
            built=ordered_built,
            datasets=ordered_datasets,
            blocking_job=blocking,
            collection=summaries,
            degraded_jobs=degraded,
            node_peak_rss_mb=ordered_peaks,
        )
    return RefreshReport(
        decision_time=ctx.decision_time,
        status="complete",
        planned=planned_kinds,
        built=ordered_built,
        datasets=ordered_datasets,
        blocking_job=None,
        collection=summaries,
        degraded_jobs=degraded,
        node_peak_rss_mb=ordered_peaks,
    )


__all__ = [
    "DART_COLLECTION_STEPS",
    "FLOW_COMPLETION_KINDS",
    "KIS_PREREQUISITE_KINDS",
    "NODE_COLLECTION_GATES",
    "OPTIONAL_COLLECTION_JOBS",
    "SCOPE_GRAPH",
    "BuildNode",
    "CollectionReport",
    "CollectionStep",
    "CollectionStepReport",
    "RefreshContext",
    "RefreshReport",
    "build_refresh_context",
    "node_peak_rss_mb",
    "plan_refresh",
    "refresh_scope",
    "reset_peak_rss",
    "run_collection_jobs",
    "run_refresh",
    "warm_daily_flag_cache",
]
