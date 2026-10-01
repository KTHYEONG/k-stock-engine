"""Collect-area commands: scoped persistence, coverage plans and provider jobs."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.data.cli.common import add_scoped_args, scoped_catalog, scoped_runtime
from src.data.cli.registry import Command

__all__ = ["COLLECT_COMMANDS"]

_LOG = logging.getLogger(__name__)


def _read_json_list(path: Path, *, label: str) -> list[Any]:
    from src.core.pit import PITDataError

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PITDataError(f"scoped {label} file is unreadable: {exc}") from exc
    if not isinstance(raw, list):
        raise PITDataError(f"scoped {label} file must hold a list")
    return raw


def _mapping_rows(raw: list[Any], *, label: str) -> list[dict[str, Any]]:
    from src.core.pit import PITDataError

    rows: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise PITDataError(f"scoped {label} entry must be a mapping")
        rows.append(dict(entry))
    return rows


def _read_scoped_payloads(path: Path) -> tuple[Any, ...]:
    """Scoped raw payloads from a JSON file with base64-encoded bodies."""
    from src.core.pit import EvidenceKind
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload

    rows = _mapping_rows(_read_json_list(path, label="payloads"), label="payloads")
    items: list[ScopedRawPayload] = []
    for row in rows:
        as_of_raw = row.get("as_of")
        items.append(
            ScopedRawPayload(
                kind=EvidenceKind(str(row.get("kind") or "")),
                source=str(row.get("source") or ""),
                natural_key=str(row.get("natural_key") or ""),
                as_of=date.fromisoformat(str(as_of_raw)) if as_of_raw else None,
                fiscal_period=str(row.get("fiscal_period") or "") or None,
                status=EvidenceStatus(str(row.get("status") or "")),
                payload=base64.b64decode(str(row.get("payload_b64") or "")),
                retrieved_at=datetime.fromisoformat(str(row.get("retrieved_at") or "")),
                source_label=str(row.get("source_label") or ""),
            )
        )
    return tuple(items)


def _run_dart_job(args: argparse.Namespace, *, job_name: str) -> dict[str, object]:
    """One budgeted DART job, emitting one JSON line per phase plus a summary."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.dart_backfill import build_scoped_dart_collector
    from src.data.jobs.dart import resolve_dart_job
    from src.data.jobs.runner import build_job_context, run_job
    from src.integrations.quota import ProviderQuotaStateStore

    runtime = scoped_runtime(args)
    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    key_env = getattr(args, "key_env", None) or provider.default_key_env
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    dry_run = bool(getattr(args, "dry_run", False))
    collector = None if dry_run else build_scoped_dart_collector(provider=provider, quota_store=quota_store, key_env=key_env)
    ctx = build_job_context(runtime=runtime, provider=provider, key_env=key_env, collector=collector)

    def _emit(payload: Mapping[str, object]) -> None:
        sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")

    report = run_job(
        resolve_dart_job(job_name), ctx,
        chunk_size=provider.dart.batch_identities, max_chunks=getattr(args, "max_chunks", None),
        dry_run=dry_run, emit=_emit,
    )
    _LOG.info(
        "[DATA] command=%s status=%s done=%d pending_left=%d requests_used=%d",
        job_name, report.status, report.done, report.pending_left, report.requests_used,
    )
    return {"job": job_name, "status": report.status, "done": report.done,
            "pending_left": report.pending_left, "requests_used": report.requests_used}


def _run_krx_job(args: argparse.Namespace, *, job_name: str) -> dict[str, object]:
    """One KRX session job, emitting one JSON line per phase plus a summary."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.jobs.krx import KRX_CHUNK_SIZE, build_krx_job_context, resolve_krx_job
    from src.data.jobs.runner import run_job
    from src.integrations.krx.client import build_scoped_krx_client
    from src.integrations.quota import ProviderQuotaStateStore

    runtime = scoped_runtime(args)
    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    dry_run = bool(getattr(args, "dry_run", False))
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    collector = None if dry_run else build_scoped_krx_client(policy=provider.krx, quota_store=quota_store)
    ctx = build_krx_job_context(runtime=runtime, provider=provider, collector=collector)

    def _emit(payload: Mapping[str, object]) -> None:
        sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")

    report = run_job(
        resolve_krx_job(job_name), ctx, chunk_size=KRX_CHUNK_SIZE,
        max_chunks=getattr(args, "max_chunks", None), dry_run=dry_run, emit=_emit,
    )
    _LOG.info(
        "[DATA] command=%s status=%s done=%d pending_left=%d requests_used=%d",
        job_name, report.status, report.done, report.pending_left, report.requests_used,
    )
    return {"job": job_name, "status": report.status, "done": report.done,
            "pending_left": report.pending_left, "requests_used": report.requests_used}


def _run_kind_job(args: argparse.Namespace, *, job_name: str) -> dict[str, object]:
    """One KIND notice job, emitting one JSON line per phase plus a summary."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.jobs.kind import KIND_CHUNK_SIZE, build_kind_job_context, resolve_kind_job
    from src.data.jobs.runner import run_job
    from src.integrations.krx.kind import build_scoped_kind_client
    from src.integrations.quota import ProviderQuotaStateStore

    runtime = scoped_runtime(args)
    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    dry_run = bool(getattr(args, "dry_run", False))
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    collector = None if dry_run else build_scoped_kind_client(policy=provider.kind, quota_store=quota_store)
    ctx = build_kind_job_context(runtime=runtime, provider=provider, collector=collector)

    def _emit(payload: Mapping[str, object]) -> None:
        sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")

    report = run_job(
        resolve_kind_job(job_name), ctx, chunk_size=KIND_CHUNK_SIZE,
        max_chunks=getattr(args, "max_chunks", None), dry_run=dry_run, emit=_emit,
    )
    _LOG.info(
        "[DATA] command=%s status=%s done=%d pending_left=%d requests_used=%d",
        job_name, report.status, report.done, report.pending_left, report.requests_used,
    )
    return {"job": job_name, "status": report.status, "done": report.done,
            "pending_left": report.pending_left, "requests_used": report.requests_used}


def _run_ls_job(args: argparse.Namespace) -> dict[str, object]:
    """One range-planned LS investor-flow job, emitting one JSON line per phase plus a summary."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.jobs.flow import LS_CHUNK_SIZE, LsInvestorFlowJob, build_ls_job_context
    from src.data.jobs.runner import run_job
    from src.integrations.ls.client import build_scoped_ls_client
    from src.integrations.ls.investor_flow import LsInvestorFlowCollector
    from src.integrations.quota import ProviderQuotaStateStore

    runtime = scoped_runtime(args)
    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    dry_run = bool(getattr(args, "dry_run", False))
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    collector: LsInvestorFlowCollector | None = None
    if not dry_run:
        client = build_scoped_ls_client(
            policy=provider.ls, quota_store=quota_store, token_cache_dir=runtime_config.logs_root,
        )
        collector = LsInvestorFlowCollector(client=client)
    ctx = build_ls_job_context(runtime=runtime, provider=provider, collector=collector)

    def _emit(payload: Mapping[str, object]) -> None:
        sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")

    report = run_job(
        LsInvestorFlowJob(since=getattr(args, "since", None)), ctx, chunk_size=LS_CHUNK_SIZE,
        max_chunks=getattr(args, "max_chunks", None), dry_run=dry_run, emit=_emit,
    )
    _LOG.info(
        "[DATA] command=ls_investor_flow status=%s done=%d pending_left=%d requests_used=%d",
        report.status, report.done, report.pending_left, report.requests_used,
    )
    return {"job": "ls_investor_flow", "status": report.status,
            "done": report.done, "pending_left": report.pending_left, "requests_used": report.requests_used}


def _run_kis_flow_job(args: argparse.Namespace) -> dict[str, object]:
    """One range-planned KIS investor-flow job, emitting one JSON line per phase plus a summary."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.jobs.flow import KIS_CHUNK_SIZE, KisInvestorFlowJob, build_kis_job_context
    from src.data.jobs.runner import run_job
    from src.integrations.kis.client import KisClient, KisCredentials
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    runtime = scoped_runtime(args)
    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    dry_run = bool(getattr(args, "dry_run", False))
    collector: KisInvestorFlowCollector | None = None
    if not dry_run:
        collector = KisInvestorFlowCollector(
            client=KisClient(
                KisCredentials.from_env(provider.kis),
                token_cache_dir=Path(runtime_config.logs_root),
                policy=provider.kis,
            )
        )
    ctx = build_kis_job_context(runtime=runtime, provider=provider, collector=collector)

    def _emit(payload: Mapping[str, object]) -> None:
        sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")

    report = run_job(
        KisInvestorFlowJob(since=getattr(args, "since", None)), ctx, chunk_size=KIS_CHUNK_SIZE,
        max_chunks=getattr(args, "max_chunks", None), dry_run=dry_run, emit=_emit,
    )
    _LOG.info(
        "[DATA] command=kis_investor_flow status=%s done=%d pending_left=%d requests_used=%d",
        report.status, report.done, report.pending_left, report.requests_used,
    )
    return {"job": "kis_investor_flow", "status": report.status,
            "done": report.done, "pending_left": report.pending_left, "requests_used": report.requests_used}


def _add_dry_run_max_chunks(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-chunks", type=int, required=False, default=None)


def _add_collect_scoped(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--payloads", type=Path, required=True)


def _add_flow_job(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--since", type=date.fromisoformat, required=False, default=None)
    _add_dry_run_max_chunks(parser)


def _add_krx_job(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    _add_dry_run_max_chunks(parser)


def _add_dart_job(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--key-env", type=str, required=False, default=None)
    _add_dry_run_max_chunks(parser)


def _add_classification(parser: argparse.ArgumentParser) -> None:
    from src.data.cli.common import _KIS_CLASSIFICATION_PACE_SECONDS

    add_scoped_args(parser)
    parser.add_argument("--symbols-from", type=Path, required=False, default=None)
    parser.add_argument("--pace-seconds", type=float, default=_KIS_CLASSIFICATION_PACE_SECONDS)


def _run_collect_scoped(args: argparse.Namespace) -> Mapping[str, object]:
    from src.data.scoped_ingestion import ScopedBronzeWriter

    runtime = scoped_runtime(args)
    writer = ScopedBronzeWriter(runtime=runtime, catalog=scoped_catalog(runtime))
    count = 0
    content_hash = ""
    for scoped_payload in _read_scoped_payloads(Path(args.payloads)):
        receipt = writer.persist(scoped_payload)
        count += 1
        content_hash = receipt.bronze_receipt.content_hash
    return {"scope_id": runtime.scope.scope_id, "receipts": count, "content_hash": content_hash}


def _run_collect_classification(args: argparse.Namespace, *, stage: str, collector: str, fetch_attr: str) -> Mapping[str, object]:
    import importlib

    from src.data.industry_collection import collect_classification_with_isolation, resolve_industry_symbols
    from src.data.scoped_ingestion import ScopedBronzeWriter

    runtime = scoped_runtime(args)
    module = importlib.import_module("src.integrations.kis.industry")
    return collect_classification_with_isolation(
        stage=stage, collector_cls=getattr(module, collector), fetch_attr=fetch_attr,
        bronze_root=runtime.workspace.bronze_root, symbols=resolve_industry_symbols(runtime, args.symbols_from),
        writer=ScopedBronzeWriter(runtime=runtime, catalog=scoped_catalog(runtime)),
        pace_seconds=float(args.pace_seconds),
    )


def _run_document_job(args: argparse.Namespace, *, job_name: str) -> dict[str, object]:
    """One document job, emitting one JSON line per phase plus a summary."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.dart_backfill import build_scoped_dart_collector
    from src.data.jobs.dart_documents import (
        DartBenchmarkDocumentFetchJob,
        DartDocumentFetchJob,
        DartDocumentReparseJob,
    )
    from src.data.jobs.runner import build_job_context, run_job
    from src.integrations.quota import ProviderQuotaStateStore

    runtime = scoped_runtime(args)
    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    key_env = getattr(args, "key_env", None) or provider.default_key_env
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    dry_run = bool(getattr(args, "dry_run", False))
    spec: Any
    if job_name == "dart_document_reparse":
        spec = DartDocumentReparseJob()
    elif job_name == "dart_benchmark_documents":
        spec = DartBenchmarkDocumentFetchJob()
    else:
        spec = DartDocumentFetchJob()
    collector = None
    if job_name in {"dart_document_fetch", "dart_benchmark_documents"} and not dry_run:  # pragma: no cover - live provider path
        collector = build_scoped_dart_collector(provider=provider, quota_store=quota_store, key_env=key_env)
    ctx = build_job_context(runtime=runtime, provider=provider, key_env=key_env, collector=collector)

    def _emit(payload: Mapping[str, object]) -> None:
        sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")

    report = run_job(
        spec, ctx,
        chunk_size=provider.dart.batch_identities, max_chunks=getattr(args, "max_chunks", None),
        dry_run=dry_run, emit=_emit,
    )
    _LOG.info(
        "[DATA] command=%s status=%s done=%d pending_left=%d requests_used=%d",
        job_name, report.status, report.done, report.pending_left, report.requests_used,
    )
    return {"job": job_name, "status": report.status, "done": report.done,
            "pending_left": report.pending_left, "requests_used": report.requests_used}


def _add_document_fetch_job(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--key-env", type=str, required=False, default=None)
    _add_dry_run_max_chunks(parser)


def _add_document_reparse_job(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    _add_dry_run_max_chunks(parser)


def _run_benchmark_documents(args: argparse.Namespace) -> Mapping[str, object]:
    """Parse stored benchmark documents and report same-filing precision."""
    from src.config import load_provider_policy, load_runtime_config
    from src.data.cli.common import CommandFailed, scoped_catalog
    from src.data.dart_document_benchmark import BENCHMARK_SEED, run_benchmark, select_benchmark_filings

    runtime = scoped_runtime(args)
    runtime_config = load_runtime_config()
    provider = load_provider_policy(runtime_config)
    catalog = scoped_catalog(runtime)
    policy = provider.dart.document_parser
    filings = select_benchmark_filings(catalog, size=policy.benchmark_sample, seed=BENCHMARK_SEED)
    report = run_benchmark(runtime, filings=filings)
    payload: dict[str, object] = dict(report.to_dict())
    if report.precision < policy.min_precision:
        raise CommandFailed(1, payload)
    return payload


def _add_benchmark_documents(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)


COLLECT_COMMANDS: tuple[Command, ...] = (
    Command("collect-scoped", "Persist scoped raw payloads to Bronze and catalog", _add_collect_scoped, _run_collect_scoped),
    Command("collect-krx-daily-market", "Collect KRX daily-market pages for completed sessions", _add_krx_job,
            lambda args: _run_krx_job(args, job_name="krx_daily_market")),
    Command("collect-krx-security-master", "Collect KRX security-master snapshots for completed sessions", _add_krx_job,
            lambda args: _run_krx_job(args, job_name="krx_security_master")),
    Command("collect-krx-hedge-series", "Collect KRX hedge-series pages (KOSDAQ150 index, inverse ETF) for completed sessions", _add_krx_job,
            lambda args: _run_krx_job(args, job_name="krx_hedge_series")),
    Command("collect-kind-notices", "Collect KIND exchange-notice search windows", _add_krx_job,
            lambda args: _run_kind_job(args, job_name="kind_notice_search")),
    Command("collect-kind-documents", "Collect KIND exchange-notice bodies", _add_krx_job,
            lambda args: _run_kind_job(args, job_name="kind_notice_documents")),
    Command("collect-ls-investor-flow", "Collect range-planned LS investor-flow windows", _add_flow_job, _run_ls_job),
    Command("collect-kis-investor-flow", "Collect range-planned KIS investor-flow pages", _add_flow_job, _run_kis_flow_job),
    Command("collect-dart-disclosures", "Collect market-wide DART disclosure windows", _add_dart_job,
            lambda args: _run_dart_job(args, job_name="dart_disclosures")),
    Command("collect-dart-corp-codes", "Refresh the DART corp-code bridge when the universe needs it", _add_dart_job,
            lambda args: _run_dart_job(args, job_name="dart_corp_codes")),
    Command("collect-dart-facts", "Collect DART periodic-report facts for eligible filings", _add_dart_job,
            lambda args: _run_dart_job(args, job_name="dart_facts")),
    Command("collect-dividend-decisions", "Collect DART cash-dividend decision archives", _add_dart_job,
            lambda args: _run_dart_job(args, job_name="dividend_decisions")),
    Command("collect-earnings-releases", "Collect DART preliminary-result and profit-change archives", _add_dart_job,
            lambda args: _run_dart_job(args, job_name="earnings_releases")),
    Command("reparse-dart-documents", "Re-derive document fact pages from stored archives", _add_document_reparse_job,
            lambda args: _run_document_job(args, job_name="dart_document_reparse")),
    Command("collect-dart-documents", "Fetch document archives for relevant document-path identities", _add_document_fetch_job,
            lambda args: _run_document_job(args, job_name="dart_document_fetch")),
    Command("collect-dart-benchmark-documents", "Fetch document archives for benchmark filings", _add_document_fetch_job,
            lambda args: _run_document_job(args, job_name="dart_benchmark_documents")),
    Command("benchmark-dart-documents", "Benchmark parsed documents against same-filing standard labels", _add_benchmark_documents,
            _run_benchmark_documents),
    Command("collect-industry-classification", "Collect current KIS industry classifications to Bronze", _add_classification,
            lambda args: _run_collect_classification(args, stage="collect-industry-classification",
                                                     collector="KisIndustryCollector",
                                                     fetch_attr="fetch_industry_classification")),
    Command("collect-stock-classification", "Collect current KIS KSIC stock classifications to Bronze", _add_classification,
            lambda args: _run_collect_classification(args, stage="collect-stock-classification",
                                                     collector="KisStockClassificationCollector",
                                                     fetch_attr="fetch_stock_classification")),
)
