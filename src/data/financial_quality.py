"""PIT financial-completeness evidence used to gate Gold feature generation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast
from zoneinfo import ZoneInfo

import polars as pl

from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    canonical_content_hash,
    dataset_digest,
    dataset_reference,
    load_manifest,
    publish_dataset,
    read_dataset,
)
from src.data.schemas import PITDataError

_FISCAL_RE = re.compile(r"^(\d{4})Q([1-4])$")
_QUALITY_COLUMNS = [
    "company_id",
    "fiscal_period",
    "accounting_basis",
    "available_at",
    "financial_complete",
    "exclusion_reason",
    "missing_facts_json",
    "source_filing_ids_json",
    "published_at",
    "source_hash",
]


@dataclass(frozen=True, slots=True)
class FinancialQualityPolicy:
    """Define the non-imputed financial facts required for Gold eligibility."""

    version: str = "financial-quality-v1"
    required_facts: tuple[str, ...] = (
        "sales",
        "gross_profit",
        "operating_profit",
        "net_income",
        "assets",
        "equity",
        "operating_cash_flow",
    )
    basis_preference: tuple[str, ...] = ("consolidated", "separate")

    def __post_init__(self) -> None:
        if not self.version.strip() or not self.required_facts:
            raise ValueError("financial quality policy must name required facts")  # pragma: no cover
        if len(set(self.required_facts)) != len(self.required_facts):
            raise ValueError("financial quality required facts must be unique")  # pragma: no cover


_DEFAULT_POLICY = FinancialQualityPolicy()


@dataclass(frozen=True, slots=True)
class FinancialQualityEvent:
    """Represent an official filing that cannot provide a usable fiscal-period value."""

    company_id: str
    fiscal_period: str
    filing_id: str
    published_at: datetime
    available_at: datetime
    reason: str

    def __post_init__(self) -> None:
        if not self.company_id or not _FISCAL_RE.fullmatch(self.fiscal_period):
            raise ValueError("financial quality event identity is invalid")  # pragma: no cover
        if not self.filing_id or not self.reason.strip():
            raise ValueError("financial quality event filing and reason are required")  # pragma: no cover
        if self.published_at.tzinfo is None or self.available_at.tzinfo is None:
            raise ValueError("financial quality event timestamps must be timezone-aware")  # pragma: no cover
        if self.available_at < self.published_at:
            raise ValueError("financial quality event cannot precede publication")  # pragma: no cover


UNVERIFIED_LEGACY_REASON: Final = "unverified_legacy_extraction"

# A listed filer must hold at least this much in total assets. Anything smaller
# is a unit error (millions recorded as won) or a broken extraction, never a
# truthful balance sheet. Named so the plausibility floor stays auditable.
_ASSETS_FLOOR_KRW: Final = 100_000_000.0

# Equity cannot exceed assets beyond this relative tolerance. The margin keeps
# rounding-scale restatements from flagging while still catching sign flips and
# swapped columns. Named so the plausibility tolerance stays auditable.
_EQUITY_EXCEEDS_ASSETS_TOLERANCE: Final = 0.0001


def _parse_quality_timestamp(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise PITDataError(f"financial quality event has an invalid {field}: {value!r}") from exc
    else:
        raise PITDataError(f"financial quality event lacks {field}")
    if parsed.tzinfo is None:
        raise PITDataError(f"financial quality event has a naive {field}")
    return parsed


def quarantine_events(quarantine: Sequence[Mapping[str, object]]) -> tuple[FinancialQualityEvent, ...]:
    """Convert quarantined-filing records into unresolved quality events.

    A filing that exists but whose values were withheld must make its fiscal
    period visibly incomplete from the moment it was observable; silently
    dropping it would let the previous period look like the latest complete
    state.

    Args:
        quarantine: Records with company_id, fiscal_period, filing_id, published_at, available_at (ISO-8601 UTC strings).

    Returns:
        One event per (company_id, fiscal_period, filing_id), reason ``unverified_legacy_extraction``.

    Raises:
        PITDataError: a record lacks a field, has a naive timestamp, has an invalid fiscal period, or precedes its publication.
    """
    try:
        records = list(quarantine)
    except TypeError as exc:
        raise PITDataError("financial quality quarantine must be a sequence") from exc
    events: dict[tuple[str, str, str], FinancialQualityEvent] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise PITDataError("financial quality quarantine record must be a mapping")
        company_id = str(record.get("company_id") or "").strip()
        fiscal_period = str(record.get("fiscal_period") or "").strip()
        filing_id = str(record.get("filing_id") or "").strip()
        if not company_id:
            raise PITDataError("financial quality quarantine record lacks company_id")
        if not fiscal_period or _FISCAL_RE.fullmatch(fiscal_period) is None:
            raise PITDataError(f"financial quality quarantine record has an invalid fiscal period: {fiscal_period!r}")
        if not filing_id:
            raise PITDataError("financial quality quarantine record lacks filing_id")
        published_at = _parse_quality_timestamp(record.get("published_at"), field="published_at")
        available_at = _parse_quality_timestamp(record.get("available_at"), field="available_at")
        try:
            event = FinancialQualityEvent(
                company_id=company_id,
                fiscal_period=fiscal_period,
                filing_id=filing_id,
                published_at=published_at,
                available_at=available_at,
                reason=UNVERIFIED_LEGACY_REASON,
            )
        except ValueError as exc:
            raise PITDataError(str(exc)) from exc
        key = (company_id, fiscal_period, filing_id)
        previous = events.get(key)
        if previous is None or (event.available_at, event.published_at) > (
            previous.available_at,
            previous.published_at,
        ):
            events[key] = event
    return tuple(
        sorted(events.values(), key=lambda e: (e.available_at, e.company_id, e.fiscal_period, e.filing_id))
    )


def implausible_balance_flags(*, assets: float | None, equity: float | None) -> tuple[str, ...]:
    """Return report-only labels for balance-sheet values no filer could truthfully report.

    Labels: ``assets_nonpositive`` (assets ≤ 0), ``equity_exceeds_assets``
    (equity > assets with a 0.01% tolerance), ``assets_below_floor``
    (0 < assets < 100,000,000 KRW). None inputs yield no label. The result is a
    diagnostic for trusted pages; it never excludes data.
    """
    if assets is None:
        return ()
    labels: list[str] = []
    if assets <= 0:
        labels.append("assets_nonpositive")
    elif assets < _ASSETS_FLOOR_KRW:
        labels.append("assets_below_floor")
    if equity is not None and equity - assets > abs(assets) * _EQUITY_EXCEEDS_ASSETS_TOLERANCE:
        labels.append("equity_exceeds_assets")
    return tuple(labels)


def _fiscal_key(period: str) -> tuple[int, int]:
    matched = _FISCAL_RE.fullmatch(period)
    if matched is None:
        return (-1, -1)  # pragma: no cover
    return (int(matched.group(1)), int(matched.group(2)))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise PITDataError("financial quality timestamps must be timezone-aware")  # pragma: no cover
    return value.astimezone(UTC)


def _empty_quality_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "company_id": pl.Series([], dtype=pl.String),
            "fiscal_period": pl.Series([], dtype=pl.String),
            "accounting_basis": pl.Series([], dtype=pl.String),
            "available_at": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
            "financial_complete": pl.Series([], dtype=pl.Boolean),
            "exclusion_reason": pl.Series([], dtype=pl.String),
            "missing_facts_json": pl.Series([], dtype=pl.String),
            "source_filing_ids_json": pl.Series([], dtype=pl.String),
            "published_at": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
            "source_hash": pl.Series([], dtype=pl.String),
        }
    )


def build_financial_quality_events(
    financial_facts: pl.DataFrame,
    *,
    unresolved_events: Collection[FinancialQualityEvent],
    decision_time: datetime,
    policy: FinancialQualityPolicy = _DEFAULT_POLICY,
) -> pl.DataFrame:
    """Build event-time financial completeness without repairing missing values.

    Each fact availability transition emits a snapshot. Consumers can therefore
    reproduce the exact completeness state observable at any past decision.
    """
    if decision_time.tzinfo is None:
        raise PITDataError("financial quality decision_time must be timezone-aware")  # pragma: no cover
    required = set(policy.required_facts)
    records: list[dict[str, object]] = []
    if not financial_facts.is_empty():
        needed = {"company_id", "fiscal_period", "filing_id", "fact", "published_at", "available_at", "value", "unit", "consolidated"}
        missing = sorted(needed - set(financial_facts.columns))
        if missing:
            raise PITDataError(f"financial quality source lacks columns: {missing}")  # pragma: no cover
        tz = getattr(financial_facts["available_at"].dtype, "time_zone", None)
        target = decision_time.astimezone(ZoneInfo(str(tz))) if tz else decision_time
        scoped = (
            financial_facts.filter(pl.col("available_at") <= target)
            .filter(pl.col("fact").is_in(sorted(required)))
            .select(["company_id", "fiscal_period", "filing_id", "fact", "published_at", "available_at", "value", "unit", "consolidated"])
            .sort(["company_id", "fiscal_period", "consolidated", "available_at", "filing_id", "fact"])
        )
        states: dict[tuple[str, str, str], dict[str, tuple[float, str, datetime]]] = {}
        active_key: tuple[str, str, str] | None = None
        active_at: datetime | None = None

        def emit() -> None:
            if active_key is None or active_at is None:
                return
            company_id, fiscal_period, basis = active_key
            state = states[active_key]
            missing_facts = sorted(required - set(state))
            filing_ids = sorted({value[1] for value in state.values()})
            published_at = max(value[2] for value in state.values()) if state else active_at
            complete = not missing_facts
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "company_id": company_id,
                        "fiscal_period": fiscal_period,
                        "basis": basis,
                        "available_at": active_at.isoformat(),
                        "filings": filing_ids,
                        "missing": missing_facts,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "company_id": company_id,
                    "fiscal_period": fiscal_period,
                    "accounting_basis": basis,
                    "available_at": active_at,
                    "financial_complete": complete,
                    "exclusion_reason": "" if complete else "missing_required_facts",
                    "missing_facts_json": json.dumps(missing_facts, separators=(",", ":")),
                    "source_filing_ids_json": json.dumps(filing_ids, separators=(",", ":")),
                    "published_at": published_at,
                    "source_hash": digest,
                }
            )

        for row in scoped.iter_rows(named=True):
            company_id = str(row["company_id"] or "").strip()
            fiscal_period = str(row["fiscal_period"] or "").strip()
            fact = str(row["fact"] or "").strip()
            available_at = row["available_at"]
            published_at = row["published_at"]
            if not company_id or _fiscal_key(fiscal_period) == (-1, -1) or not isinstance(available_at, datetime) or not isinstance(published_at, datetime):
                continue  # pragma: no cover
            try:
                value = float(row["value"])
            except (TypeError, ValueError):  # pragma: no cover
                continue
            if str(row["unit"] or "") != "KRW" or not math.isfinite(value):
                continue
            basis = "consolidated" if bool(row["consolidated"]) else "separate"
            key = (company_id, fiscal_period, basis)
            current_at = _as_utc(available_at)
            if active_key != key or active_at != current_at:
                emit()
                active_key = key
                active_at = current_at
            states.setdefault(key, {})[fact] = (value, str(row["filing_id"]), _as_utc(published_at))
        emit()

    records.extend(
        {
            "company_id": event.company_id,
            "fiscal_period": event.fiscal_period,
            "accounting_basis": "unknown",
            "available_at": _as_utc(event.available_at),
            "financial_complete": False,
            "exclusion_reason": event.reason,
            "missing_facts_json": json.dumps(sorted(required), separators=(",", ":")),
            "source_filing_ids_json": json.dumps([event.filing_id], separators=(",", ":")),
            "published_at": _as_utc(event.published_at),
            "source_hash": hashlib.sha256(
                f"{event.company_id}:{event.fiscal_period}:{event.filing_id}:{event.reason}".encode()
            ).hexdigest(),
        }
        for event in unresolved_events
        if _as_utc(event.available_at) <= decision_time
    )
    if not records:
        return _empty_quality_frame()
    return pl.DataFrame(records, schema_overrides={"available_at": pl.Datetime(time_zone="UTC"), "published_at": pl.Datetime(time_zone="UTC")}).select(_QUALITY_COLUMNS).sort(["company_id", "fiscal_period", "accounting_basis", "available_at"])


def eligible_companies_from_quality(
    events: pl.DataFrame,
    *,
    decision_time: datetime,
    company_ids: Collection[str],
    policy: FinancialQualityPolicy = _DEFAULT_POLICY,
) -> frozenset[str]:
    """Return companies with a complete latest fiscal period visible at decision time."""
    if decision_time.tzinfo is None:
        raise PITDataError("financial quality decision_time must be timezone-aware")  # pragma: no cover
    candidates = frozenset(str(company_id) for company_id in company_ids if str(company_id))
    if not candidates or events.is_empty():
        return frozenset()
    needed = {"company_id", "fiscal_period", "accounting_basis", "available_at", "financial_complete"}
    if missing := sorted(needed - set(events.columns)):
        raise PITDataError(f"financial quality rows lack columns: {missing}")  # pragma: no cover
    tz = getattr(events["available_at"].dtype, "time_zone", None)
    target = decision_time.astimezone(ZoneInfo(str(tz))) if tz else decision_time
    scoped = events.filter(pl.col("company_id").is_in(sorted(candidates)) & (pl.col("available_at") <= target))
    by_company: dict[str, list[dict[str, object]]] = {}
    for row in scoped.iter_rows(named=True):
        period = str(row["fiscal_period"])
        if _fiscal_key(period) != (-1, -1):
            by_company.setdefault(str(row["company_id"]), []).append(row)
    eligible: set[str] = set()
    for company_id, rows in by_company.items():
        latest_period = max((str(row["fiscal_period"]) for row in rows), key=_fiscal_key)
        latest_rows = [row for row in rows if row["fiscal_period"] == latest_period]
        latest_by_basis: dict[str, dict[str, object]] = {}
        for row in latest_rows:
            basis = str(row["accounting_basis"])
            previous = latest_by_basis.get(basis)
            row_available_at = cast(datetime, row["available_at"])
            previous_available_at = (
                cast(datetime, previous["available_at"]) if previous is not None else None
            )
            if previous_available_at is None or row_available_at > previous_available_at:
                latest_by_basis[basis] = row
        if any(bool(latest_by_basis.get(basis, {}).get("financial_complete")) for basis in policy.basis_preference):
            eligible.add(company_id)
    return frozenset(eligible)


def _v2_digest(value: str | None, values: list[str]) -> str:
    if value is None:
        return dataset_digest(values)
    text = str(value)
    return text if text.startswith("bronze:") else dataset_digest([text])


def _materialize_financial_quality_v2(
    events: pl.DataFrame,
    *,
    layer_root: Path,
    decision_time: datetime,
    facts_dataset_id: str,
    quarantine_digest: str | None,
    unresolved_events_digest: str | None,
    policy: FinancialQualityPolicy,
) -> Path:
    if not facts_dataset_id.strip():
        raise PITDataError("financial quality materialization requires a facts dataset id")
    if events.is_empty():
        raise PITDataError("financial quality materialization requires events")
    if list(events.columns) != _QUALITY_COLUMNS:
        raise PITDataError("financial quality events have an unexpected schema")
    if events.filter(pl.col("available_at") > decision_time).height:
        raise PITDataError("financial quality events contain a row after decision_time")
    quality_report_hash = canonical_content_hash(events, _QUALITY_COLUMNS)
    identity = DatasetIdentity(
        kind="financial_quality",
        layer=DatasetLayer.SILVER,
        policy_version=policy.version,
        inputs={
            "facts": dataset_reference(facts_dataset_id, kind="financial_facts"),
            "quarantine": _v2_digest(quarantine_digest, []),
            "unresolved_events": _v2_digest(unresolved_events_digest, []),
        },
        params={
            "decision_time": decision_time,
            "required_facts": ",".join(policy.required_facts),
            "basis_preference": ",".join(policy.basis_preference),
        },
    )
    published = publish_dataset(
        layer_root=Path(layer_root),
        identity=identity,
        partitions={"part-00000.parquet": events},
        details={
            "quality_report_hash": quality_report_hash,
            "decision_time": decision_time.isoformat(),
            "required_facts": list(policy.required_facts),
            "basis_preference": list(policy.basis_preference),
        },
    )
    return published.path


def materialize_financial_quality(
    events: pl.DataFrame,
    *,
    root: Path | None = None,
    layer_root: Path | None = None,
    dataset_id: str | None = None,
    decision_time: datetime,
    source_dataset_id: str | None = None,
    facts_dataset_id: str | None = None,
    quarantine_digest: str | None = None,
    unresolved_events_digest: str | None = None,
    policy: FinancialQualityPolicy = _DEFAULT_POLICY,
    certification: object | None = None,
) -> Path:
    """Publish quality evidence through the v2 identity-bound dataset contract."""

    _ = dataset_id, certification
    destination = layer_root if layer_root is not None else root
    if destination is None:
        raise PITDataError("financial quality materialization requires a Silver layer root")
    return _materialize_financial_quality_v2(
        events,
        layer_root=Path(destination),
        decision_time=decision_time,
        facts_dataset_id=facts_dataset_id or source_dataset_id or "",
        quarantine_digest=quarantine_digest,
        unresolved_events_digest=unresolved_events_digest,
        policy=policy,
    )


def load_latest_financial_quality(*, root: Path, decision_time: datetime) -> pl.DataFrame:
    """Load the newest v2-certified quality dataset."""

    if decision_time.tzinfo is None:
        raise PITDataError("financial quality decision_time must be timezone-aware")
    table_root = Path(root)
    candidates: list[tuple[datetime, str]] = []
    if table_root.is_dir():
        for path in table_root.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            try:
                manifest = load_manifest(path)
            except PITDataError:
                continue
            if manifest.kind == "financial_quality":
                candidates.append((manifest.created_at, path.name))
    if not candidates:
        raise PITDataError("missing certified financial quality dataset")
    _, dataset_id = max(candidates)
    return read_dataset(table_root / dataset_id).filter(pl.col("available_at") <= decision_time).collect()
