"""Scoped DART fact batching against the receipt catalog and provider quota."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime

from src.config.errors import ConfigError
from src.config.providers import ProviderPolicy
from src.config.secrets import read_secret
from src.core.pit import PITDataError
from src.data.receipt_catalog import ReceiptCatalog
from src.data.runtime import DataRuntime
from src.data.scope_coverage import CoverageRequirement
from src.data.scoped_ingestion import FACT_SOURCE, dart_fact_natural_key
from src.integrations.dart.client import dart_ledger_for_key
from src.integrations.dart.xbrl import DartXbrlCollector
from src.integrations.quota import ProviderQuotaStateStore

__all__ = [
    "DartFactBatchPlan",
    "build_scoped_dart_collector",
    "build_scoped_dart_fact_batch",
    "scoped_dart_request_headroom",
]

_FISCAL_PATTERN = re.compile(r"\d{4}Q[1-4]")


def _period_key(period: str) -> int:
    return int(period[:4]) * 4 + int(period[5])


def _identity_fiscal_period(identity: Mapping[str, str]) -> str:
    raw = str(identity.get("fiscal_period") or "").strip()
    if raw:
        if not _FISCAL_PATTERN.fullmatch(raw):
            raise PITDataError(f"invalid fiscal period {raw!r}")
        return raw
    biz_year = str(identity.get("biz_year") or "").strip()
    reprt_code = str(identity.get("reprt_code") or "").strip()
    quarter = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}.get(reprt_code)
    if not biz_year.isdigit() or quarter is None:
        raise PITDataError("DART fact identity is missing a derivable fiscal period")
    return f"{int(biz_year)}Q{quarter}"


@dataclass(frozen=True, slots=True)
class DartFactBatchPlan:
    """Quota-bounded DART fact identities selected from in-scope filing evidence."""

    scope_hash: str
    plan_id: str
    identities: tuple[Mapping[str, str], ...]
    missing_without_filing: tuple[CoverageRequirement, ...]
    estimated_request_ceiling: int
    available_request_headroom: int


def _quota_provider(*, provider: ProviderPolicy, key_env: str, api_key: str) -> str:
    """Ledger name for one key: the primary key keeps its historical name."""
    return dart_ledger_for_key(key_env=key_env, primary_key_env=provider.primary_key_env, api_key=api_key)


def scoped_dart_request_headroom(
    *,
    provider: ProviderPolicy,
    quota_store: ProviderQuotaStateStore,
    key_env: str | None = None,
    now: datetime | None = None,
) -> int:
    """Request capacity for one key after reserving its quota headroom.

    Each key is metered in its own ledger against the budget and reserve its
    provider policy declares, so a run of one key never consumes another's
    headroom.
    """
    resolved = key_env or provider.primary_key_env
    policy = provider.dart_key(resolved)
    api_key = read_secret(resolved, default="")
    remaining = quota_store.remaining_daily_attempts(
        provider=_quota_provider(provider=provider, key_env=resolved, api_key=api_key),
        now=now or datetime.now(UTC),
        daily_limit=policy.daily_budget,
    )
    return max(0, remaining - policy.daily_reserve)


def build_scoped_dart_collector(
    *,
    provider: ProviderPolicy,
    quota_store: ProviderQuotaStateStore,
    key_env: str | None = None,
) -> DartXbrlCollector:
    """Sole DART collector for one declared key from its provider policy.

    Raises:
        ConfigError: the key has no declared policy.
        ValueError: the key's environment variable is unset.
    """
    resolved = key_env or provider.primary_key_env
    policy = provider.dart_key(resolved)
    try:
        api_key = read_secret(resolved)
    except ConfigError as exc:
        raise ValueError(f"{resolved} is not set") from exc
    return DartXbrlCollector(
        api_key=api_key,
        quota_store=quota_store,
        quota_provider=_quota_provider(provider=provider, key_env=resolved, api_key=api_key),
        max_workers=policy.max_workers,
        min_interval=policy.min_interval_seconds,
        daily_request_limit=policy.daily_budget,
    )


def build_scoped_dart_fact_batch(
    *,
    runtime: DataRuntime,
    catalog: ReceiptCatalog,
    filing_identities: Collection[Mapping[str, str]],
    offset: int,
    limit: int,
    provider: ProviderPolicy,
    key_env: str | None = None,
    quota_store: ProviderQuotaStateStore | None = None,
    now: datetime | None = None,
) -> DartFactBatchPlan:
    """Select retained 2019+ filing identities lacking successful fact evidence without disclosure-list discovery."""
    scope = runtime.scope
    if offset < 0 or limit < 1:
        raise PITDataError("offset must be nonnegative and limit must be positive")
    floor = scope.features.fundamental_fiscal_start
    full: dict[tuple[str, str, str], dict[str, str]] = {}
    missing: list[CoverageRequirement] = []
    for raw_identity in filing_identities:
        item = dict(raw_identity)
        corp_code = str(item.get("corp_code") or "").strip()
        biz_year = str(item.get("biz_year") or "").strip()
        reprt_code = str(item.get("reprt_code") or "").strip()
        if not corp_code or not biz_year or not reprt_code:
            raise PITDataError("DART fact identity is missing corp code, business year, or report code")
        filing_id = str(item.get("filing_id") or item.get("rcept_no") or "").strip()
        published_at = str(item.get("published_at") or item.get("available_at") or "").strip()
        natural_key = dart_fact_natural_key(corp_code=corp_code, biz_year=biz_year, reprt_code=reprt_code)
        if not filing_id or not published_at:
            as_of_raw = str(item.get("as_of") or "").strip()
            missing.append(
                CoverageRequirement(
                    source=FACT_SOURCE,
                    natural_key=natural_key,
                    as_of=date.fromisoformat(as_of_raw[:10]) if as_of_raw else None,
                    fiscal_period=_identity_fiscal_period(item),
                    required=True,
                )
            )
            continue
        if _period_key(_identity_fiscal_period(item)) < _period_key(floor):
            continue
        key = (corp_code, biz_year, reprt_code)
        current = full.get(key)
        if current is None or (published_at, filing_id) > (
            str(current.get("published_at") or current.get("available_at") or ""),
            str(current.get("filing_id") or current.get("rcept_no") or ""),
        ):
            full[key] = item
    covered = catalog.successful_keys(source=FACT_SOURCE, fiscal_start=floor)
    candidates = sorted(
        (
            item
            for key, item in full.items()
            if dart_fact_natural_key(corp_code=key[0], biz_year=key[1], reprt_code=key[2]) not in covered
        ),
        key=lambda item: dart_fact_natural_key(
            corp_code=str(item.get("corp_code") or ""),
            biz_year=str(item.get("biz_year") or ""),
            reprt_code=str(item.get("reprt_code") or ""),
        ),
    )
    page = candidates[offset : offset + limit]
    store = quota_store or ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    request_headroom = scoped_dart_request_headroom(
        provider=provider, quota_store=store, now=now, key_env=key_env
    )
    allowance = min(
        len(page),
        provider.dart.batch_identities,
        request_headroom // 3,
    )
    selected = tuple(page[:allowance])
    digest = hashlib.sha256()
    digest.update(scope.content_hash.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps([dict(item) for item in selected], sort_keys=True).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(f"{offset}:{limit}".encode())
    return DartFactBatchPlan(
        scope_hash=scope.content_hash,
        plan_id=f"dart-facts-{digest.hexdigest()[:16]}",
        identities=selected,
        missing_without_filing=tuple(sorted(missing, key=lambda item: (item.source, item.natural_key))),
        estimated_request_ceiling=len(selected) * 3,
        available_request_headroom=request_headroom,
    )
