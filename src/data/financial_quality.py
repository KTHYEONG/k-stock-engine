"""PIT financial-completeness evidence used to gate Gold feature generation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import polars as pl

from src.core.datasets import HIVE_PARTITION_LAYOUT, DatasetCertification, make_manifest
from src.core.instruments import AssetKind
from src.data.schemas import PITDataError
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

_FISCAL_RE = re.compile(r"^(\d{4})Q([1-4])$")
_QUALITY_FEATURE_SET = "stock_financial_quality_v1"
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


def materialize_financial_quality(
    events: pl.DataFrame,
    *,
    root: Path,
    dataset_id: str,
    decision_time: datetime,
    source_dataset_id: str,
    policy: FinancialQualityPolicy = _DEFAULT_POLICY,
    certification: DatasetCertification = DatasetCertification.RESEARCH,
) -> Path:
    """Persist immutable quality evidence bound to one financial Silver dataset."""
    if decision_time.tzinfo is None or not source_dataset_id.strip():
        raise PITDataError("financial quality materialization requires PIT source identity")  # pragma: no cover
    if events.is_empty():
        raise PITDataError("financial quality materialization requires events")  # pragma: no cover
    if list(events.columns) != _QUALITY_COLUMNS:
        raise PITDataError("financial quality events have an unexpected schema")  # pragma: no cover
    content_hash = canonical_content_hash(events, _QUALITY_COLUMNS)
    time_start = events["available_at"].min()
    time_end = events["available_at"].max()
    if not isinstance(time_start, datetime) or not isinstance(time_end, datetime):
        raise PITDataError("financial quality events lack available_at timestamps")  # pragma: no cover
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=_QUALITY_COLUMNS,
        feature_set=_QUALITY_FEATURE_SET,
        label_definition="none",
        label_horizon_sessions=1,
        time_start=time_start,
        time_end=time_end,
        provider_version="dart-financial-quality-v1",
        universe_policy_version=policy.version,
        row_count=events.height,
        generated_time=decision_time,
        certification=certification,
        quality_report_hash=source_dataset_id,
        schema_version="v2",
        content_hash=content_hash,
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    return ParquetDatasetStore(Path(root) / "financial_quality").write_partitioned(
        events,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=_QUALITY_FEATURE_SET,
        decision_time=decision_time,
        content_manifest={"source_financial_dataset_id": source_dataset_id, "policy_version": policy.version},
    )


def load_latest_financial_quality(*, root: Path, decision_time: datetime) -> pl.DataFrame:
    """Load the newest certified quality dataset; rows remain PIT-filtered by consumers."""
    if decision_time.tzinfo is None:
        raise PITDataError("financial quality decision_time must be timezone-aware")  # pragma: no cover
    table_root = Path(root) / "financial_quality"
    store = ParquetDatasetStore(table_root)
    candidates: list[tuple[datetime, datetime, str]] = []
    if table_root.exists():
        for path in table_root.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue  # pragma: no cover
            try:
                manifest = store.read_manifest(path.name)
            except (FileNotFoundError, ValueError, OSError):  # pragma: no cover
                continue
            generated = getattr(manifest, "generated_time", None)
            if isinstance(generated, datetime) and generated.tzinfo is not None and getattr(manifest, "feature_set", "") == _QUALITY_FEATURE_SET:
                candidates.append((generated, manifest.time_end, path.name))
    if not candidates:
        raise PITDataError("missing certified financial quality dataset")  # pragma: no cover
    _, manifest_time_end, dataset_id = max(candidates)
    try:
        # Manifest coverage may end after a historical replay decision.  The
        # immutable companion is verified as a whole, then its rows are
        # filtered at each session by eligible_companies_from_quality.
        return store.read(
            dataset_id,
            AssetKind.STOCK,
            _QUALITY_FEATURE_SET,
            max(decision_time, manifest_time_end),
        )
    except (FileNotFoundError, ValueError, OSError) as exc:  # pragma: no cover
        raise PITDataError("invalid certified financial quality dataset") from exc
