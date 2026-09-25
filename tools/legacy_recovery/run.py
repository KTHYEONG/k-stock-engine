"""Recover quarantined legacy filings under a benchmark gate.

Stages, each resumable: (1) list quarantined filings from the quarantine file;
(2) ensure their archives are in Bronze, fetching missing ones through the
scoped collector; (3) extract and verify; (4) evaluate against the labeled set;
(5) for promotable fact classes only, persist attested pages; (6) write the report.

Run from the repo root after the quarantine file exists::

    uv run python tools/legacy_recovery/run.py \\
        --key-env OPENDART_API_KEY_2 \\
        --quarantine-file data/state/kr_swing_2019_v1/dart_fact_quarantine_<hash>.json

Follow-up, in order: ``normalize-dart-facts`` (with the superseded receipts
file), ``build-financial-quality``, delete superseded Silver generations.

This tool never modifies quarantine or Silver files directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

from tools.legacy_recovery.benchmark import (
    FACT_CLASSES,
    PromotionGate,
    build_labeled_set,
    evaluate,
    filing_key,
    score_filings,
)
from tools.legacy_recovery.extract import ExtractedStatement, extract_statements
from tools.legacy_recovery.verify import (
    Verdict,
    statement_fact_class,
    verify_filing,
)

__all__ = ["RecoveryResult", "main", "run_recovery"]

_ATTESTED_KIND: Final = "legacy_document_verified"
_MAPPING_VERSION: Final = "dart-fact-map-v1"
_REPORT_JSON_NAME: Final = "legacy_recovery_report.json"
_REPORT_MD_NAME: Final = "legacy_recovery_report.md"


class ArchiveCollector(Protocol):
    """Source of ``document.xml`` archives; the scoped collector in production."""

    def fetch_document_archive(self, rcept_no: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    status: str
    benchmark_id: str
    report: Mapping[str, Any]
    persisted_receipts: tuple[str, ...]
    remaining_archives: int
    fetched_archives: int


@dataclass(frozen=True, slots=True)
class _FilingBundle:
    filing_id: str
    corp_code: str
    fiscal_period: str
    company_id: str
    published_at: str
    statements: tuple[ExtractedStatement, ...]
    verdicts: tuple[Verdict, ...]


def _emit(**fields: object) -> None:
    sys.stdout.write(json.dumps(fields, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def load_gate_config(path: Path) -> PromotionGate:
    """Read gate thresholds from the tool config file (no literals in logic)."""
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read gate config {path}: {exc}") from exc
    try:
        section = raw["promotion_gate"]
        rate = float(section["min_exact_match_rate"])
        count = int(section["min_accepted_filings"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid gate config {path}: {exc}") from exc
    if not 0.0 <= rate <= 1.0 or count < 1:
        raise SystemExit(f"invalid gate config {path}: thresholds out of range")
    return PromotionGate(min_exact_match_rate=rate, min_accepted_filings=count)


def load_quarantine_entries(path: Path) -> list[dict[str, Any]]:
    """Read the quarantine file as a list of filing mappings."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read quarantine file {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"quarantine file {path} must hold a JSON list")
    return [item for item in payload if isinstance(item, dict)]


def index_local_archives(bronze_root: Path) -> dict[str, bytes]:
    """Map receipt number to archive bytes for verified Bronze document archives."""
    indexed: dict[str, tuple[str, bytes]] = {}
    try:
        receipts = sorted((Path(bronze_root) / "dart_documents").rglob("receipt.json"))
    except OSError:
        return {}
    for receipt_path in receipts:
        try:
            meta = json.loads(receipt_path.read_text(encoding="utf-8"))
            payload = (receipt_path.parent / "payload.zip").read_bytes()
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        rcept_no = str(meta.get("rcept_no") or "").strip()
        digest = hashlib.sha256(payload).hexdigest()
        if (
            not rcept_no
            or digest != receipt_path.parent.name
            or meta.get("sha256") != digest
        ):
            continue
        retrieved = str(meta.get("retrieved_at") or "")
        current = indexed.get(rcept_no)
        if current is None or retrieved > current[0]:
            indexed[rcept_no] = (retrieved, payload)
    return {key: value for key, (_, value) in indexed.items()}


def _iter_fact_pages(bronze_root: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    try:
        receipt_paths = sorted((Path(bronze_root) / "financial_facts").rglob("receipt.json"))
    except OSError:
        return pages
    for receipt_path in receipt_paths:
        try:
            payload = json.loads((receipt_path.parent / "payload.json").read_bytes())
        except (OSError, ValueError):
            continue
        if isinstance(payload, list):
            pages.append({"records": payload})
        elif isinstance(payload, dict):
            pages.append(payload)
    return pages


def find_legacy_pages(bronze_root: Path) -> dict[str, dict[str, Any]]:
    """Map filing id to its original ``legacy_document`` Bronze page."""
    found: dict[str, dict[str, Any]] = {}
    for page in _iter_fact_pages(bronze_root):
        if not isinstance(page, dict) or str(page.get("source_kind") or "") != "legacy_document":
            continue
        identity = page.get("identity")
        identity_map = identity if isinstance(identity, dict) else {}
        filing_id = str(
            identity_map.get("filing_id")
            or identity_map.get("rcept_no")
            or page.get("filing_id")
            or ""
        ).strip()
        if not filing_id:
            records = page.get("records")
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, dict):
                        filing_id = str(
                            record.get("filing_id") or record.get("rcept_no") or ""
                        ).strip()
                        if filing_id:
                            break
        if filing_id and filing_id not in found:
            found[filing_id] = page
    return found


def _is_fetchable(filing_id: str) -> bool:
    return len(filing_id) == 14 and filing_id.isdigit()


def _ensure_archives(
    *,
    entries: Sequence[Mapping[str, Any]],
    local: dict[str, bytes],
    dry_run: bool,
    collector: ArchiveCollector | None,
    headroom_fn: Callable[[], int] | None,
    bronze_root: Path,
    now: datetime,
) -> tuple[int, int, int, str]:
    """Fetch missing archives through the scoped collector.

    Returns (fetched, remaining, unfetchable, fetch_status). Never raises for
    provider failures; those filings stay missing so the next run resumes them.
    """
    from src.data.dart_documents import DartDocumentStore

    missing = sorted(
        {
            str(entry.get("filing_id") or "").strip()
            for entry in entries
            if _is_fetchable(str(entry.get("filing_id") or "").strip())
            and str(entry.get("filing_id") or "").strip() not in local
        }
    )
    unfetchable = sum(
        1
        for entry in entries
        if not _is_fetchable(str(entry.get("filing_id") or "").strip())
    )
    if dry_run or not missing:
        return (0, len(missing), unfetchable, "dry_run" if dry_run else "complete")
    headroom = headroom_fn() if headroom_fn is not None else 0
    if headroom <= 0:
        return (0, len(missing), unfetchable, "budget_exhausted")
    if collector is None:
        return (0, len(missing), unfetchable, "budget_exhausted")
    health_check = getattr(collector, "health_check", None)
    if callable(health_check):
        try:
            health_check()
        except Exception:
            return (0, len(missing), unfetchable, "provider_unreachable")
    store = DartDocumentStore(bronze_root)
    fetched = 0
    remaining: list[str] = []
    for filing_id in missing:
        try:
            headroom = headroom_fn() if headroom_fn is not None else 0
        except Exception:
            headroom = 0
        if headroom <= 0:
            remaining.append(filing_id)
            continue
        try:
            archive = collector.fetch_document_archive(filing_id)
        except Exception:
            remaining.append(filing_id)
            continue
        if not archive:
            remaining.append(filing_id)
            continue
        try:
            store.store_archive(archive, rcept_no=filing_id, retrieved_at=now)
        except (ValueError, OSError):
            remaining.append(filing_id)
            continue
        local[filing_id] = bytes(archive)
        fetched += 1
    status = "complete" if not remaining else "budget_exhausted"
    return (fetched, len(remaining), unfetchable, status)


def _process_filings(
    *,
    entries: Sequence[Mapping[str, Any]],
    local: Mapping[str, bytes],
) -> tuple[dict[str, list[ExtractedStatement]], dict[str, _FilingBundle], int, int]:
    extracted: dict[str, list[ExtractedStatement]] = {}
    bundles: dict[str, _FilingBundle] = {}
    attempted = 0
    accepted_filings = 0
    for entry in entries:
        filing_id = str(entry.get("filing_id") or "").strip()
        if not filing_id or filing_id in bundles:
            continue
        archive = local.get(filing_id)
        if archive is None:
            continue
        corp = str(entry.get("dart_corp_code") or entry.get("company_id") or "").strip()
        period = str(entry.get("fiscal_period") or "").strip()
        if not corp or not period:
            continue
        attempted += 1
        statements = extract_statements(archive)
        verdicts = verify_filing(statements)
        bundles[filing_id] = _FilingBundle(
            filing_id=filing_id,
            corp_code=corp,
            fiscal_period=period,
            company_id=str(entry.get("company_id") or corp),
            published_at=str(entry.get("published_at") or ""),
            statements=tuple(statements),
            verdicts=tuple(verdicts),
        )
        accepted_here = False
        for statement, verdict in zip(statements, verdicts, strict=True):
            if not verdict.accepted:
                continue
            accepted_here = True
            extracted.setdefault(
                filing_key(corp, period, statement.basis), []
            ).append(statement)
        if accepted_here:
            accepted_filings += 1
    return (extracted, bundles, attempted, accepted_filings)


def _original_field(page: Mapping[str, Any], entry: Mapping[str, Any], name: str) -> str:
    records = page.get("records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and record.get(name) not in (None, ""):
                return str(record[name])
    for source in (page, entry):
        if isinstance(source, Mapping) and source.get(name) not in (None, ""):
            return str(source[name])
    identity = page.get("identity")
    if isinstance(identity, Mapping) and identity.get(name) not in (None, ""):
        return str(identity[name])
    return ""


def _build_attested_page(
    *,
    original: Mapping[str, Any],
    bundle: _FilingBundle,
    fact_class: str,
    benchmark_id: str,
    archive_hash: str,
) -> dict[str, Any] | None:
    accepted = [
        (statement, verdict)
        for statement, verdict in zip(bundle.statements, bundle.verdicts, strict=True)
        if verdict.accepted and statement_fact_class(statement) == fact_class
    ]
    if not accepted:
        return None
    identity_raw = original.get("identity")
    identity = dict(identity_raw) if isinstance(identity_raw, Mapping) else {}
    identity.setdefault("filing_id", bundle.filing_id)
    if not all(
        str(identity.get(name) or "").strip() for name in ("corp_code", "biz_year", "reprt_code")
    ):
        return None
    company_id = _original_field(original, {"company_id": bundle.company_id}, "company_id") or bundle.corp_code
    ticker = _original_field(original, {}, "ticker")
    corp_code = str(identity.get("corp_code") or bundle.corp_code)
    published_at = (
        _original_field(original, {"published_at": bundle.published_at}, "published_at")
        or bundle.published_at
    )
    mapping_version = str(original.get("mapping_version") or _MAPPING_VERSION)
    record_checks: dict[str, list[str]] = {}
    records: list[dict[str, Any]] = []
    for statement, verdict in accepted:
        consolidated = statement.basis == "consolidated"
        for fact in sorted(statement.values):
            records.append(
                {
                    "company_id": company_id,
                    "corp_code": corp_code,
                    "ticker": ticker,
                    "filing_id": bundle.filing_id,
                    "fiscal_period": bundle.fiscal_period,
                    "fact": fact,
                    "published_at": published_at,
                    "value": float(statement.values[fact]),
                    "unit": "KRW",
                    "consolidated": consolidated,
                    "restatement_id": "r0",
                    "source_kind": _ATTESTED_KIND,
                    "mapping_version": mapping_version,
                    "raw_document_hash": archive_hash,
                    "verification": {
                        "benchmark_id": benchmark_id,
                        "fact_class": fact_class,
                        "checks": list(verdict.checks),
                    },
                }
            )
            record_checks.setdefault(fact, list(verdict.checks))
    checks = sorted({check for checks in record_checks.values() for check in checks})
    page: dict[str, Any] = {
        "source_kind": _ATTESTED_KIND,
        "identity": identity,
        "records": records,
        "verification": {
            "benchmark_id": benchmark_id,
            "fact_class": fact_class,
            "checks": checks,
        },
        "mapping_version": mapping_version,
        "raw_document_hash": archive_hash,
        "company_id": company_id,
        "corp_code": corp_code,
        "ticker": ticker,
        "filing_id": bundle.filing_id,
        "fiscal_period": bundle.fiscal_period,
        "published_at": published_at,
    }
    return page


def _report_sections(
    *,
    gate: PromotionGate,
    extracted: Mapping[str, Sequence[ExtractedStatement]],
    scores: Sequence[Any],
    results: Sequence[Any],
    bundles: Mapping[str, _FilingBundle],
    quarantine_count: int,
) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for result in results:
        rate = result.exact_matches / result.accepted_filings if result.accepted_filings else 0.0
        classes[result.fact_class] = {
            "accepted_filings": result.accepted_filings,
            "exact_matches": result.exact_matches,
            "exact_match_rate": round(rate, 4),
            "promotable": result.promotable,
            "min_exact_match_rate": gate.min_exact_match_rate,
            "min_accepted_filings": gate.min_accepted_filings,
        }
    attempted_classes: dict[str, set[str]] = {name: set() for name in FACT_CLASSES}
    for key, statements in extracted.items():
        for statement in statements:
            attempted_classes[statement_fact_class(statement)].add(key)
    accepted_classes: dict[str, set[str]] = {name: set() for name in FACT_CLASSES}
    for score in scores:
        if score_filing_accepted(score):
            accepted_classes[score.fact_class].add(score.key)
    acceptance_rate = {
        "attempted_filings": len(bundles),
        "accepted_filings": sum(1 for key in bundles if _bundle_accepted(bundles[key])),
        "by_class": {
            name: {
                "attempted_filings": len(attempted_classes[name]),
                "accepted_filings": len(accepted_classes[name]),
            }
            for name in FACT_CLASSES
        },
    }
    by_year: dict[str, Any] = {}
    by_period: dict[str, Any] = {}
    for score in scores:
        if not score.scored:
            continue
        period = _score_period(score.key, bundles)
        year = period[:4] if len(period) >= 4 else "unknown"
        for bucket, label in ((by_year, year), (by_period, period or "unknown")):
            cell = bucket.setdefault(
                label, {name: {"accepted": 0, "exact": 0} for name in FACT_CLASSES}
            )
            cell[score.fact_class]["accepted"] += 1
            if score.exact:
                cell[score.fact_class]["exact"] += 1
    return {
        "gate": {
            "min_exact_match_rate": gate.min_exact_match_rate,
            "min_accepted_filings": gate.min_accepted_filings,
        },
        "classes": classes,
        "acceptance_rate": acceptance_rate,
        "quarantine_filings": quarantine_count,
        "by_year": by_year,
        "by_period": by_period,
    }


def score_filing_accepted(score: Any) -> bool:
    """Return True when a filing score carries accepted (scored or not) facts."""
    return bool(score.verified_facts)


def _bundle_accepted(bundle: _FilingBundle) -> bool:
    return any(verdict.accepted for verdict in bundle.verdicts)


def _score_period(key: str, bundles: Mapping[str, _FilingBundle]) -> str:
    for bundle in bundles.values():
        for statement in bundle.statements:
            candidate = filing_key(bundle.corp_code, bundle.fiscal_period, statement.basis)
            if candidate == key:
                return bundle.fiscal_period
    return ""


def _write_reports(report_dir: Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / _REPORT_JSON_NAME
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    body = report
    lines = [
        "# Legacy recovery report",
        "",
        f"Status: `{body.get('status')}`",
        f"Benchmark ID: `{body.get('benchmark_id')}`",
        "",
        "## Gate",
    ]
    gate = body.get("gate", {})
    lines.append(
        f"Exact-match rate >= `{gate.get('min_exact_match_rate')}`, "
        f"accepted filings >= `{gate.get('min_accepted_filings')}`."
    )
    lines.extend(["", "## Fact classes", ""])
    classes = body.get("classes", {})
    for name in FACT_CLASSES:
        cell = classes.get(name, {})
        lines.append(
            f"- `{name}`: accepted `{cell.get('accepted_filings', 0)}`, "
            f"exact `{cell.get('exact_matches', 0)}`, "
            f"rate `{cell.get('exact_match_rate', 0.0)}`, "
            f"promotable `{cell.get('promotable', False)}`."
        )
    acceptance = body.get("acceptance_rate", {})
    lines.extend(
        [
            "",
            "## Acceptance",
            f"Attempted filings: `{acceptance.get('attempted_filings', 0)}`, "
            f"accepted: `{acceptance.get('accepted_filings', 0)}`.",
            "",
            "## Archives",
            f"Fetched: `{body.get('fetched_archives', 0)}`, "
            f"remaining: `{body.get('remaining_archives', 0)}`, "
            f"unfetchable: `{body.get('unfetchable_filings', 0)}`.",
            f"Persisted receipts: `{len(body.get('persisted_receipts', []))}`.",
            f"Unscored promoted filings: `{body.get('unscored_promoted', {})}`.",
        ]
    )
    if body.get("dry_run"):
        lines.extend(["", "Dry run: no API calls were made and nothing was persisted."])
    lines.extend(
        [
            "",
            "## Follow-up",
            "Run `normalize-dart-facts` (with the superseded receipts file), "
            "then `build-financial-quality`, then delete superseded Silver generations.",
            "",
        ]
    )
    md_path = target / _REPORT_MD_NAME
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return (json_path, md_path)


def run_recovery(
    *,
    bronze_root: Path,
    quarantine_file: Path,
    report_dir: Path,
    gate: PromotionGate,
    batch_size: int = 50,
    dry_run: bool = False,
    now: datetime | None = None,
    collector: ArchiveCollector | None = None,
    headroom_fn: Callable[[], int] | None = None,
    persist_fn: Callable[[list[dict[str, Any]]], list[str]] | None = None,
) -> RecoveryResult:
    """Recover quarantined legacy filings under the benchmark gate.

    Args:
        bronze_root: Scope Bronze root holding ``financial_facts/`` and
            ``dart_documents/``.
        quarantine_file: Quarantine JSON written beside the fact refresh.
        report_dir: Destination for the JSON and markdown report.
        gate: Promotion thresholds read from the tool config file.
        batch_size: Attested pages persisted per catalog revision.
        dry_run: Resolve, extract, and evaluate from local data only; makes
            zero API calls and persists nothing.
        now: Retrieval timestamp for fetched archives.
        collector: Archive source; required only when archives are missing
            and the run is not a dry run.
        headroom_fn: Scoped quota headroom probe; absence means no fetch.
        persist_fn: Attested-page sink; absence means persist nothing.

    Returns:
        The recovery result bound to the written report.
    """
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    moment = now if now is not None and now.tzinfo is not None else datetime.now(UTC)
    bronze_path = Path(bronze_root)
    entries = load_quarantine_entries(quarantine_file)
    local = index_local_archives(bronze_path)
    fetched, remaining, unfetchable, fetch_status = _ensure_archives(
        entries=entries,
        local=local,
        dry_run=dry_run,
        collector=collector,
        headroom_fn=headroom_fn,
        bronze_root=bronze_path,
        now=moment,
    )
    extracted, bundles, attempted, _ = _process_filings(entries=entries, local=local)
    labeled = build_labeled_set(bronze_path, quarantine_file)
    scores = score_filings(extracted, labeled)
    results = evaluate(extracted, labeled, gate)
    promotable = {result.fact_class for result in results if result.promotable}
    sections = _report_sections(
        gate=gate,
        extracted=extracted,
        scores=scores,
        results=results,
        bundles=bundles,
        quarantine_count=len(entries),
    )
    benchmark_id = hashlib.sha256(
        json.dumps(sections, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    legacy_pages = find_legacy_pages(bronze_path)
    pages: list[dict[str, Any]] = []
    skipped = 0
    for filing_id, bundle in sorted(bundles.items()):
        original = legacy_pages.get(filing_id)
        if original is None:
            if any(
                verdict.accepted
                and statement_fact_class(statement) in promotable
                for statement, verdict in zip(
                    bundle.statements, bundle.verdicts, strict=True
                )
            ):
                skipped += 1
            continue
        archive_hash = hashlib.sha256(local[filing_id]).hexdigest()
        for fact_class in FACT_CLASSES:
            if fact_class not in promotable:
                continue
            page = _build_attested_page(
                original=original,
                bundle=bundle,
                fact_class=fact_class,
                benchmark_id=benchmark_id,
                archive_hash=archive_hash,
            )
            if page is None:
                skipped += 1
                continue
            pages.append(page)
    scored_keys = {(score.key, score.fact_class) for score in scores if score.scored}
    unscored_promoted = {
        name: sum(
            1
            for bundle in bundles.values()
            if _bundle_class_promoted(bundle, name, promotable)
            and not any(
                (key, name) in scored_keys
                for key in (
                    filing_key(bundle.corp_code, bundle.fiscal_period, basis)
                    for basis in ("consolidated", "separate")
                )
            )
        )
        for name in FACT_CLASSES
    }
    persisted: list[str] = []
    if pages and not dry_run and persist_fn is not None:
        batches = [pages[idx : idx + int(batch_size)] for idx in range(0, len(pages), int(batch_size))]
        for batch in batches:
            persisted.extend(persist_fn(batch))
    if dry_run:
        status = "dry_run"
    elif not promotable:
        status = "no_promotable_class"
    elif fetch_status != "complete":
        status = fetch_status
    else:
        status = "complete"
    report: dict[str, Any] = {
        **sections,
        "benchmark_id": benchmark_id,
        "status": status,
        "dry_run": dry_run,
        "fetched_archives": fetched,
        "remaining_archives": remaining,
        "unfetchable_filings": unfetchable,
        "attempted_filings": attempted,
        "persisted_receipts": list(persisted),
        "persisted_pages": len(pages),
        "skipped_without_identity": skipped,
        "unscored_promoted": unscored_promoted,
        "quarantine_file": str(quarantine_file),
    }
    _write_reports(report_dir, report)
    return RecoveryResult(
        status=status,
        benchmark_id=benchmark_id,
        report=report,
        persisted_receipts=tuple(persisted),
        remaining_archives=remaining,
        fetched_archives=fetched,
    )


def _bundle_class_promoted(
    bundle: _FilingBundle, fact_class: str, promotable: set[str]
) -> bool:
    return fact_class in promotable and any(
        verdict.accepted and statement_fact_class(statement) == fact_class
        for statement, verdict in zip(bundle.statements, bundle.verdicts, strict=True)
    )


class _ScopedArchiveCollector:
    """Adapt the scoped DART collector to the archive fetcher seam."""

    def __init__(self, collector: Any) -> None:
        self._collector = collector

    def health_check(self) -> None:
        check = getattr(self._collector, "health_check", None)
        if callable(check):
            check()

    def fetch_document_archive(self, rcept_no: str) -> bytes:
        direct = getattr(self._collector, "fetch_document_archive", None)
        if callable(direct):
            return bytes(direct(rcept_no))
        client = getattr(self._collector, "_client", None)
        fetch = getattr(client, "fetch_document_archive", None)
        if not callable(fetch):
            raise ValueError("scoped collector cannot fetch document archives")
        return bytes(fetch(rcept_no))


def _default_paths() -> tuple[Path, Path, Path]:
    root = Path.cwd()
    return (
        root / "config" / "research" / "kr_swing_2019_v1.toml",
        root / "tools" / "legacy_recovery" / "config.toml",
        root / "docs" / "research",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Recover quarantined legacy filings under a benchmark gate."""
    default_scope, default_gate, default_reports = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarantine-file", required=True, help="quarantine JSON path")
    parser.add_argument(
        "--key-env",
        default=os.environ.get("PRIMARY_DART_KEY_ENV", "OPENDART_API_KEY"),
        help="environment variable of the OpenDART key",
    )
    parser.add_argument("--data-root", default=str(Path.cwd() / "data"))
    parser.add_argument("--scope-config", default=str(default_scope))
    parser.add_argument("--config", default=str(default_gate))
    parser.add_argument("--report-dir", default=str(default_reports))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from src.data.cli import load_data_runtime  # noqa: PLC0415
    from src.data.collection import dart_fact_scoped_payload  # noqa: PLC0415
    from src.data.dart_backfill import (  # noqa: PLC0415
        build_scoped_dart_collector,
        scoped_dart_request_headroom,
    )
    from src.data.receipt_catalog import ReceiptCatalog  # noqa: PLC0415
    from src.data.research_scope import PRIMARY_DART_KEY_ENV  # noqa: PLC0415
    from src.data.scoped_ingestion import ScopedBronzeWriter  # noqa: PLC0415
    from src.integrations.quota import ProviderQuotaStateStore  # noqa: PLC0415

    key_env = str(args.key_env or PRIMARY_DART_KEY_ENV)
    gate = load_gate_config(Path(args.config))
    try:
        runtime = load_data_runtime(
            scope_config=Path(args.scope_config), data_root=Path(args.data_root)
        )
    except Exception as exc:
        _emit(stage="abort", reason="runtime", error=str(exc)[:200])
        return 2
    bronze_root = runtime.workspace.bronze_root
    catalog = ReceiptCatalog(bronze_root / "catalog")
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    headroom_fn = lambda: scoped_dart_request_headroom(  # noqa: E731
        runtime=runtime, quota_store=quota_store, key_env=key_env
    )
    _emit(
        stage="plan",
        key_env=key_env,
        quarantine_file=str(args.quarantine_file),
        dry_run=bool(args.dry_run),
        headroom_now=headroom_fn(),
    )
    collector: ArchiveCollector | None = None
    if not args.dry_run:
        if not os.environ.get(key_env):
            _emit(stage="abort", reason="key_missing", key_env=key_env)
            return 2
        try:
            scoped = build_scoped_dart_collector(
                runtime=runtime, quota_store=quota_store, key_env=key_env
            )
        except (ValueError, Exception) as exc:
            _emit(stage="abort", reason="collector", error=str(exc)[:200])
            return 2
        collector = _ScopedArchiveCollector(scoped)

    def _persist(pages: list[dict[str, Any]]) -> list[str]:
        from datetime import UTC as _UTC  # noqa: PLC0415
        from datetime import datetime as _datetime  # noqa: PLC0415

        writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
        moment = _datetime.now(_UTC)
        hashes: list[str] = []
        for idx in range(0, len(pages), int(args.batch_size)):
            batch = pages[idx : idx + int(args.batch_size)]
            receipts = writer.persist_many(
                [
                    dart_fact_scoped_payload(page=page, retrieved_at=moment)
                    for page in batch
                ]
            )
            hashes.extend(r.bronze_receipt.content_hash for r in receipts)
        return hashes

    try:
        result = run_recovery(
            bronze_root=bronze_root,
            quarantine_file=Path(args.quarantine_file),
            report_dir=Path(args.report_dir),
            gate=gate,
            batch_size=int(args.batch_size),
            dry_run=bool(args.dry_run),
            collector=collector,
            headroom_fn=headroom_fn,
            persist_fn=None if args.dry_run else _persist,
        )
    except SystemExit as exc:
        raise exc
    except Exception as exc:
        _emit(stage="abort", reason="recovery", error=str(exc)[:200])
        return 1
    _emit(
        stage="done",
        status=result.status,
        benchmark_id=result.benchmark_id,
        persisted=len(result.persisted_receipts),
        remaining=result.remaining_archives,
        fetched=result.fetched_archives,
    )
    if result.status == "provider_unreachable":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
