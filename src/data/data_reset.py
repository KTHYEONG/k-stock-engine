"""Remove precisely enumerated legacy data roots after rebase verification."""
from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePath
from typing import Any

from src.data.rebase import RebaseReport, RetentionDecision
from src.data.receipt_catalog import ReceiptCatalog
from src.data.research_scope import ResearchScope
from src.data.runtime import DataRuntime
from src.data.schemas import PITDataError
from src.data.scope_coverage import CoverageRequirement, build_scope_coverage_report
from src.data.scoped_ingestion import CORP_CODE_SOURCE

__all__ = [
    "LEGACY_REMOVAL_TARGETS",
    "ResetVerification",
    "remove_verified_legacy_data",
    "verify_legacy_removal",
]

LEGACY_REMOVAL_TARGETS: tuple[PurePath, ...] = (
    PurePath("archive"),
    PurePath("artifacts"),
    PurePath("bronze/stocks"),
    PurePath("silver/stocks"),
    PurePath("gold/stocks"),
)

_REQUIRED_SOURCES: tuple[str, ...] = ("krx_daily_market", "financial_facts")
_FISCAL_PATTERN = re.compile(r"\d{4}Q[1-4]")


@dataclass(frozen=True, slots=True)
class ResetVerification:
    """Checks required before removal of precisely enumerated legacy data roots."""

    scope_hash: str
    rebase_report_hash: str
    catalog_revision_hash: str
    verified_targets: tuple[Path, ...]


def _fiscal_key(period: str) -> int:
    return int(period[:4]) * 4 + int(period[5])


def _check_target_boundary(*, target: Path, root_resolved: Path) -> None:
    if target.is_symlink():
        raise PITDataError(f"refusing removal: symlink target {target}")
    resolved = target.resolve()
    if not any(resolved == root_resolved / relative for relative in LEGACY_REMOVAL_TARGETS):
        raise PITDataError(f"refusing removal: target outside enumerated legacy roots {target}")


def _read_catalog_entries(*, catalog_root: Path, catalog_revision_hash: str) -> list[dict[str, Any]]:
    revision_path = catalog_root / f"{catalog_revision_hash}.json"
    pointer = catalog_root / "latest.json"
    if not revision_path.is_file() or not pointer.is_file():
        raise PITDataError("matching receipt-catalog revision is missing")
    entries = json.loads(revision_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise PITDataError("matching receipt-catalog revision is missing")
    return [item for item in entries if isinstance(item, dict)]


def _entry_out_of_bounds(entry: Mapping[str, Any], *, scope: ResearchScope) -> bool:
    source = str(entry.get("source") or "")
    as_of_raw = entry.get("as_of")
    as_of = date.fromisoformat(str(as_of_raw)) if isinstance(as_of_raw, str) and as_of_raw.strip() else None
    fiscal_raw = entry.get("fiscal_period")
    has_fiscal_period = fiscal_raw not in (None, "")
    if as_of is None:
        out_of_bounds = source != CORP_CODE_SOURCE
    elif source == "financial_facts" and has_fiscal_period:
        # FY2025 annual facts are ordinarily published in 2026.  The fiscal
        # period, rather than publication date, bounds retained accounting
        # evidence; publication date remains its PIT availability timestamp.
        out_of_bounds = as_of < scope.evidence_start
    else:
        out_of_bounds = as_of < scope.evidence_start or as_of > scope.completed_end
    if has_fiscal_period:
        fiscal_period = str(fiscal_raw)
        out_of_bounds = (
            out_of_bounds
            or not _FISCAL_PATTERN.fullmatch(fiscal_period)
            or _fiscal_key(fiscal_period) < _fiscal_key(scope.features.fundamental_fiscal_start)
        )
    return out_of_bounds


def _require_catalog_bounds(*, runtime: DataRuntime, catalog_root: Path, catalog_revision_hash: str) -> None:
    entries = _read_catalog_entries(catalog_root=catalog_root, catalog_revision_hash=catalog_revision_hash)
    for entry in entries:
        if _entry_out_of_bounds(entry, scope=runtime.scope):
            raise PITDataError(f"retained catalog entry {entry.get('natural_key')!r} is outside scope limits")


def _claimed_requirements(decisions: tuple[RetentionDecision, ...]) -> tuple[CoverageRequirement, ...]:
    return tuple(
        CoverageRequirement(source=item.source, natural_key=item.natural_key, as_of=None, fiscal_period=None, required=True)
        for item in decisions
        if item.retained and item.natural_key
    )


def _require_mandatory_coverage(
    *, runtime: DataRuntime, catalog: ReceiptCatalog, decisions: tuple[RetentionDecision, ...]
) -> None:
    for source in _REQUIRED_SOURCES:
        if not catalog.successful_keys(source=source):
            raise PITDataError(f"mandatory raw coverage is unresolved for {source!r}")
    report = build_scope_coverage_report(
        scope=runtime.scope, requirements=_claimed_requirements(decisions), catalog=catalog
    )
    if report.missing or report.unresolved:
        raise PITDataError("mandatory raw coverage is unresolved; legacy output cannot substitute raw evidence")


def verify_legacy_removal(
    *, runtime: DataRuntime, rebase_report: RebaseReport, data_root: Path
) -> ResetVerification:
    """Verify the rebase and exact target boundaries without deleting any path."""
    root = Path(data_root)
    root_resolved = root.resolve()
    if not rebase_report.catalog_revision_hash or rebase_report.retained_payload_count == 0:
        raise PITDataError("rebase report is not a successful non-dry-run report")
    if rebase_report.scope_hash != runtime.scope.content_hash:
        raise PITDataError("rebase report scope hash does not match the active scope")
    catalog_root = runtime.workspace.bronze_root / "catalog"
    catalog = ReceiptCatalog(catalog_root)
    _require_catalog_bounds(runtime=runtime, catalog_root=catalog_root, catalog_revision_hash=rebase_report.catalog_revision_hash)
    _require_mandatory_coverage(runtime=runtime, catalog=catalog, decisions=rebase_report.decisions)
    verified: list[Path] = []
    for relative in LEGACY_REMOVAL_TARGETS:
        target = root / relative
        _check_target_boundary(target=target, root_resolved=root_resolved)
        if target.exists():
            verified.append(target)
    return ResetVerification(
        scope_hash=runtime.scope.content_hash,
        rebase_report_hash=rebase_report.report_path.parent.name,
        catalog_revision_hash=rebase_report.catalog_revision_hash,
        verified_targets=tuple(verified),
    )


def _find_state_root(*, data_root: Path, rebase_report_hash: str) -> Path:
    state_base = Path(data_root) / "state"
    if state_base.is_dir():
        for scope_dir in sorted(state_base.iterdir(), key=lambda item: item.name):
            if scope_dir.is_dir() and (scope_dir / "rebase" / rebase_report_hash / "report.json").is_file():
                return scope_dir
    raise PITDataError("scoped state for the rebase report is missing")


def remove_verified_legacy_data(
    *, verification: ResetVerification, data_root: Path, apply: bool
) -> tuple[Path, ...]:
    """Delete only verified legacy targets when apply is explicitly true."""
    root = Path(data_root)
    root_resolved = root.resolve()
    for target in verification.verified_targets:
        _check_target_boundary(target=target, root_resolved=root_resolved)
    if not apply:
        return verification.verified_targets
    verified_set = {Path(item) for item in verification.verified_targets}
    removed: list[Path] = []
    absent: list[str] = []
    for relative in LEGACY_REMOVAL_TARGETS:
        target = root / relative
        if target.exists():
            if target not in verified_set:
                raise PITDataError(f"refusing removal: unverified target {target}")
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed.append(target)
        else:
            absent.append(relative.as_posix())
    remaining = [relative.as_posix() for relative in LEGACY_REMOVAL_TARGETS if (root / relative).exists()]
    if remaining or not root.is_dir():
        raise PITDataError(f"legacy removal is incomplete: {remaining}")
    state_root = _find_state_root(data_root=root, rebase_report_hash=verification.rebase_report_hash)
    record_dir = state_root / "reset" / verification.scope_hash
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "removed.json").write_text(
        json.dumps(
            {
                "scope_hash": verification.scope_hash,
                "rebase_report_hash": verification.rebase_report_hash,
                "catalog_revision_hash": verification.catalog_revision_hash,
                "removed": [item.as_posix() for item in removed],
                "absent": sorted(absent),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return tuple(removed)
