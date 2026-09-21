"""Rebase legacy raw receipts into one scoped Bronze workspace."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.research_scope import ResearchScope
from src.data.runtime import DataRuntime
from src.data.schemas import EvidenceKind
from src.data.scoped_ingestion import (
    CORP_CODE_SOURCE,
    FACT_SOURCE,
    ScopedBronzeWriter,
    ScopedRawPayload,
    dart_fact_natural_key,
)

__all__ = ["RebaseReport", "RetentionDecision", "materialize_scoped_bronze"]

_REPRT_QUARTER = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}
_FISCAL_PATTERN = re.compile(r"\d{4}Q[1-4]")
_PRICE_SOURCE = "krx_daily_market"
_ACTION_SOURCE = "dart_corporate_actions"
_CORP_MAP_KEY = "dart_corp_codes"
_CORP_MAP_DIR = "dart_corp_codes"


@dataclass(frozen=True, slots=True)
class RetentionDecision:
    """Auditable keep or reject decision for one legacy raw receipt."""

    legacy_path: Path
    source: str
    natural_key: str | None
    retained: bool
    reason: str


@dataclass(frozen=True, slots=True)
class RebaseReport:
    """Immutable proof of evidence copied into a single Scope workspace."""

    scope_hash: str
    content_hash: str
    decisions: tuple[RetentionDecision, ...]
    catalog_revision_hash: str
    retained_payload_count: int
    rejected_payload_count: int
    report_path: Path


def _fiscal_key(period: str) -> int:
    return int(period[:4]) * 4 + int(period[5])


def _parse_iso_day(value: object) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _parse_compact_day(value: object) -> date | None:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        except ValueError:
            return None
    return _parse_iso_day(text)


def _daily_market_day(payload: Mapping[str, Any]) -> date | None:
    found = _parse_iso_day(payload.get("session"))
    if found is not None:
        return found
    records = payload.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in ("BAS_DD", "bas_dd", "session"):
                found = _parse_compact_day(record.get(key))
                if found is not None:
                    return found
    return None


def _action_day(payload: Mapping[str, Any]) -> date | None:
    return _parse_iso_day(payload.get("end")) or _parse_iso_day(payload.get("start"))


def _string_field(payload: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = payload.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    identity = payload.get("identity")
    if isinstance(identity, dict):
        for name in names:
            value = identity.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _fact_fiscal_period(payload: Mapping[str, Any]) -> str | None:
    explicit = _string_field(payload, "fiscal_period")
    if explicit:
        return explicit
    biz_year = _string_field(payload, "biz_year", "bsns_year")
    reprt_code = _string_field(payload, "reprt_code", "report_code")
    quarter = _REPRT_QUARTER.get(reprt_code)
    if len(biz_year) == 4 and biz_year.isdigit() and quarter is not None:
        return f"{biz_year}Q{quarter}"
    return None


def _iter_legacy_payloads(bronze_source: Path) -> list[tuple[str, Path]]:
    if not bronze_source.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for kind_dir in sorted(bronze_source.iterdir(), key=lambda item: item.name):
        if not kind_dir.is_dir() or kind_dir.is_symlink():
            continue
        for receipt_dir in sorted(kind_dir.iterdir(), key=lambda item: item.name):
            if not receipt_dir.is_dir() or receipt_dir.is_symlink():
                continue
            payload_path = receipt_dir / "payload.json"
            if payload_path.is_file():
                found.append((kind_dir.name, payload_path))
    return found


def _check_duplicate(
    *, source: str, natural_key: str, legacy_path: Path, raw: bytes, seen: dict[tuple[str, str], str]
) -> RetentionDecision | None:
    payload_hash = hashlib.sha256(raw).hexdigest()
    previous = seen.get((source, natural_key))
    if previous is None:
        seen[(source, natural_key)] = payload_hash
        return None
    reason = "duplicate_receipt" if previous == payload_hash else "duplicate_conflict"
    return RetentionDecision(legacy_path, source, natural_key, False, reason)


def _retain_price(
    *,
    scope: ResearchScope,
    legacy_path: Path,
    payload: Mapping[str, Any],
    raw: bytes,
    retrieved_at: datetime,
    seen: dict[tuple[str, str], str],
    is_action: bool,
) -> tuple[RetentionDecision, ScopedRawPayload | None]:
    provider_day = _action_day(payload) if is_action else _daily_market_day(payload)
    source = _ACTION_SOURCE if is_action else _PRICE_SOURCE
    if provider_day is None:
        return RetentionDecision(legacy_path, source, None, False, "missing_provider_date"), None
    if is_action:
        corp_code = _string_field(payload, "corp_code") or "unknown"
        endpoint = _string_field(payload, "endpoint") or "decisions"
        natural_key = f"{corp_code}:{endpoint}:{provider_day.isoformat()}"
    else:
        natural_key = provider_day.isoformat()
    if provider_day < scope.evidence_start or provider_day > scope.completed_end:
        return RetentionDecision(legacy_path, source, natural_key, False, "out_of_scope_provider_date"), None
    duplicate = _check_duplicate(source=source, natural_key=natural_key, legacy_path=legacy_path, raw=raw, seen=seen)
    if duplicate is not None:
        return duplicate, None
    kind = EvidenceKind.CORPORATE_ACTIONS if is_action else EvidenceKind.DAILY_MARKET
    scoped = ScopedRawPayload(
        kind=kind,
        source=source,
        natural_key=natural_key,
        as_of=provider_day,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        payload=raw,
        retrieved_at=retrieved_at,
        source_label=f"legacy:{legacy_path}",
    )
    return RetentionDecision(legacy_path, source, natural_key, True, "in_scope_verified"), scoped


def _retain_fact(
    *,
    scope: ResearchScope,
    legacy_path: Path,
    payload: Mapping[str, Any],
    raw: bytes,
    retrieved_at: datetime,
    seen: dict[tuple[str, str], str],
) -> tuple[RetentionDecision, ScopedRawPayload | None]:
    corp_code = _string_field(payload, "corp_code")
    biz_year = _string_field(payload, "biz_year", "bsns_year")
    reprt_code = _string_field(payload, "reprt_code", "report_code")
    if not corp_code or not biz_year or not reprt_code:
        return RetentionDecision(legacy_path, FACT_SOURCE, None, False, "missing_payload_identity"), None
    natural_key = dart_fact_natural_key(corp_code=corp_code, biz_year=biz_year, reprt_code=reprt_code)
    fiscal_period = _fact_fiscal_period(payload)
    if fiscal_period is None:
        return RetentionDecision(legacy_path, FACT_SOURCE, natural_key, False, "missing_fiscal_period"), None
    if not _FISCAL_PATTERN.fullmatch(fiscal_period):
        return RetentionDecision(legacy_path, FACT_SOURCE, natural_key, False, "invalid_fiscal_period"), None
    if _fiscal_key(fiscal_period) < _fiscal_key(scope.features.fundamental_fiscal_start):
        return RetentionDecision(legacy_path, FACT_SOURCE, natural_key, False, "out_of_scope_fiscal_period"), None
    provider_day = _parse_iso_day(_string_field(payload, "published_at")) or retrieved_at.date()
    duplicate = _check_duplicate(source=FACT_SOURCE, natural_key=natural_key, legacy_path=legacy_path, raw=raw, seen=seen)
    if duplicate is not None:
        return duplicate, None
    scoped = ScopedRawPayload(
        kind=EvidenceKind.FINANCIAL_FACTS,
        source=FACT_SOURCE,
        natural_key=natural_key,
        as_of=provider_day,
        fiscal_period=fiscal_period,
        status=EvidenceStatus.SUCCESS,
        payload=raw,
        retrieved_at=retrieved_at,
        source_label=f"legacy:{legacy_path}",
    )
    return RetentionDecision(legacy_path, FACT_SOURCE, natural_key, True, "in_scope_verified"), scoped


def _retain_corp_map(
    *,
    legacy_path: Path,
    raw: bytes,
    retrieved_at: datetime,
    seen: dict[tuple[str, str], str],
) -> tuple[RetentionDecision, ScopedRawPayload | None]:
    duplicate = _check_duplicate(
        source=CORP_CODE_SOURCE, natural_key=_CORP_MAP_KEY, legacy_path=legacy_path, raw=raw, seen=seen
    )
    if duplicate is not None:
        return duplicate, None
    scoped = ScopedRawPayload(
        kind=EvidenceKind.SECURITY_MASTER,
        source=CORP_CODE_SOURCE,
        natural_key=_CORP_MAP_KEY,
        as_of=None,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        payload=raw,
        retrieved_at=retrieved_at,
        source_label=f"legacy:{legacy_path}",
    )
    return RetentionDecision(legacy_path, CORP_CODE_SOURCE, _CORP_MAP_KEY, True, "in_scope_verified"), scoped


def _verify_receipt(
    *, kind_name: str, legacy_path: Path, raw: bytes, scope: ResearchScope
) -> tuple[RetentionDecision | None, datetime | None]:
    receipt_path = legacy_path.parent / "receipt.json"
    if not receipt_path.is_file():
        if kind_name == _CORP_MAP_DIR:
            start = scope.evidence_start
            return None, datetime(start.year, start.month, start.day, tzinfo=UTC)
        return RetentionDecision(legacy_path, kind_name, None, False, "missing_legacy_receipt"), None
    try:
        meta = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    if not isinstance(meta, dict):
        return RetentionDecision(legacy_path, kind_name, None, False, "malformed_legacy_receipt"), None
    content_hash = meta.get("content_hash")
    if not isinstance(content_hash, str) or not content_hash:
        return RetentionDecision(legacy_path, kind_name, None, False, "malformed_legacy_receipt"), None
    if content_hash != hashlib.sha256(raw).hexdigest():
        return RetentionDecision(legacy_path, kind_name, None, False, "receipt_hash_mismatch"), None
    if str(meta.get("kind") or "") != kind_name:
        return RetentionDecision(legacy_path, kind_name, None, False, "receipt_kind_mismatch"), None
    try:
        retrieved_at = datetime.fromisoformat(str(meta["retrieved_at"]))
    except (KeyError, ValueError):
        return RetentionDecision(legacy_path, kind_name, None, False, "malformed_legacy_receipt"), None
    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.replace(tzinfo=UTC)
    return None, retrieved_at


def _decide(
    *,
    scope: ResearchScope,
    kind_name: str,
    legacy_path: Path,
    raw: bytes,
    seen: dict[tuple[str, str], str],
) -> tuple[RetentionDecision, ScopedRawPayload | None]:
    rejection, retrieved_at = _verify_receipt(kind_name=kind_name, legacy_path=legacy_path, raw=raw, scope=scope)
    if rejection is not None or retrieved_at is None:
        assert rejection is not None
        return rejection, None
    try:
        payload = json.loads(raw)
    except ValueError:
        return RetentionDecision(legacy_path, kind_name, None, False, "malformed_legacy_payload"), None
    if kind_name == EvidenceKind.DAILY_MARKET.value:
        if not isinstance(payload, dict):
            return RetentionDecision(legacy_path, kind_name, None, False, "malformed_legacy_payload"), None
        return _retain_price(
            scope=scope, legacy_path=legacy_path, payload=payload,
            raw=raw, retrieved_at=retrieved_at, seen=seen, is_action=False,
        )
    if kind_name == EvidenceKind.CORPORATE_ACTIONS.value:
        if not isinstance(payload, dict):
            return RetentionDecision(legacy_path, kind_name, None, False, "malformed_legacy_payload"), None
        return _retain_price(
            scope=scope, legacy_path=legacy_path, payload=payload,
            raw=raw, retrieved_at=retrieved_at, seen=seen, is_action=True,
        )
    if kind_name == EvidenceKind.FINANCIAL_FACTS.value:
        if not isinstance(payload, dict):
            return RetentionDecision(legacy_path, kind_name, None, False, "malformed_legacy_payload"), None
        return _retain_fact(
            scope=scope, legacy_path=legacy_path, payload=payload,
            raw=raw, retrieved_at=retrieved_at, seen=seen,
        )
    if kind_name == _CORP_MAP_DIR:
        return _retain_corp_map(
            legacy_path=legacy_path, raw=raw, retrieved_at=retrieved_at, seen=seen,
        )
    return RetentionDecision(legacy_path, kind_name, None, False, "unsupported_legacy_source"), None


def _catalog_revision_hash(catalog_root: Path) -> str:
    pointer = catalog_root / "latest.json"
    if not pointer.is_file():
        return ""
    raw = json.loads(pointer.read_text(encoding="utf-8"))
    revision = raw.get("revision", "") if isinstance(raw, dict) else ""
    return Path(str(revision)).stem if revision else ""


def _decision_to_dict(decision: RetentionDecision) -> dict[str, object]:
    return {
        "legacy_path": str(decision.legacy_path),
        "source": decision.source,
        "natural_key": decision.natural_key,
        "retained": decision.retained,
        "reason": decision.reason,
    }


def materialize_scoped_bronze(
    *, runtime: DataRuntime, legacy_data_root: Path, dry_run: bool
) -> RebaseReport:
    """Classify legacy raw receipts and retain only independently verified in-scope source evidence."""
    scope = runtime.scope
    bronze_source = Path(legacy_data_root) / "bronze"
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    writer = None if dry_run else ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    decisions: list[RetentionDecision] = []
    retained_hashes: list[str] = []
    seen: dict[tuple[str, str], str] = {}
    pending: list[ScopedRawPayload] = []
    for kind_name, legacy_path in _iter_legacy_payloads(bronze_source):
        raw = legacy_path.read_bytes()
        decision, scoped = _decide(scope=scope, kind_name=kind_name, legacy_path=legacy_path, raw=raw, seen=seen)
        decisions.append(decision)
        if decision.retained:
            retained_hashes.append(hashlib.sha256(raw).hexdigest())
            if scoped is not None and writer is not None:
                pending.append(scoped)
    if writer is not None:
        for scoped in pending:
            writer.persist(scoped)
    ordered = tuple(sorted(decisions, key=lambda item: (item.source, item.natural_key or "", str(item.legacy_path))))
    digest = hashlib.sha256()
    for payload_hash in sorted(retained_hashes):
        digest.update(payload_hash.encode("utf-8"))
        digest.update(b"\x00")
    content_hash = digest.hexdigest()
    scope_hash = scope.content_hash
    catalog_revision_hash = "" if dry_run else _catalog_revision_hash(runtime.workspace.bronze_root / "catalog")
    retained_count = sum(1 for item in ordered if item.retained)
    canonical = json.dumps(
        {
            "catalog_revision_hash": catalog_revision_hash,
            "content_hash": content_hash,
            "decisions": [
                {
                    "legacy_path": str(item.legacy_path),
                    "source": item.source,
                    "natural_key": item.natural_key,
                    "retained": item.retained,
                    "reason": item.reason,
                }
                for item in ordered
            ],
            "scope_hash": scope_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    report_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report_dir = runtime.workspace.state_root / "rebase" / report_hash
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "scope_hash": scope_hash,
                "content_hash": content_hash,
                "catalog_revision_hash": catalog_revision_hash,
                "retained_payload_count": retained_count,
                "rejected_payload_count": len(ordered) - retained_count,
                "decisions": [_decision_to_dict(item) for item in ordered],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return RebaseReport(
        scope_hash=scope_hash,
        content_hash=content_hash,
        decisions=ordered,
        catalog_revision_hash=catalog_revision_hash,
        retained_payload_count=retained_count,
        rejected_payload_count=len(ordered) - retained_count,
        report_path=report_path,
    )
