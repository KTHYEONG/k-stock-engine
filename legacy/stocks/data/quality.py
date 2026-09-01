"""Deterministic stock-panel quality contracts and classification.

Every source row is routed into exactly one of three immutable outputs
(``eligible`` / ``quarantined`` / ``non_equity``) based on an effective-dated
security master, structural bar invariants, and corporate-action interval
coverage. ``unclassified`` is a quarantine reason, never a fourth successful
class: an identifier without a matching effective-dated master record is
quarantined rather than aborting a provisional migration or entering the stock
universe.

All panel-scale work is vectorized Polars; no ``apply``/``map_rows`` over
market rows, and report ordering and hashes are deterministic across identical
bytes.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time
from enum import StrEnum

import polars as pl

from src.core.datasets import DatasetCertification
from legacy.stocks.research.datasets import (
    ELIGIBLE_STATUS,
    QUALITY_REASON_COLUMN,
    QUALITY_STATUS_COLUMN,
    QUARANTINED_STATUS,
)

__all__ = [
    "NON_EQUITY_STATUS",
    "CorporateActionInterval",
    "CorporateActionSnapshot",
    "FeatureAvailabilityRecord",
    "InstrumentClassification",
    "InstrumentMasterRecord",
    "InstrumentMasterSnapshot",
    "KRXSessionCalendar",
    "StockDataQualityPolicy",
    "StockDataQualityReport",
    "validate_canonical_stock_panel",
]

NON_EQUITY_STATUS = "non_equity"

REASON_UNCLASSIFIED = "unclassified_instrument"
REASON_NON_EQUITY = "non_equity_instrument"
REASON_INDEX = "index_instrument"
REASON_OUTSIDE_INTERVAL = "outside_tradable_interval"
REASON_OHLC = "non_positive_or_missing_ohlc"
REASON_ORDERING = "invalid_ohlc_ordering"
REASON_EXECUTABLE = "non_executable_bar"
REASON_CAPITALIZATION = "negative_capitalization"
REASON_ACTION_COVERAGE = "uncovered_action_interval"
REASON_NON_SESSION = "non_calendar_session"

INDEX_TICKERS = ("KOSPI", "KOSDAQ")
_TICKER_RE = re.compile(r"^\d{6}$")
_KRX_CLOSE_TIME = time(15, 30)
_KRX_AVAILABLE_TIME = time(15, 31)
_OHLC_COLUMNS = ("open", "high", "low", "close")

PROVISIONAL_AVAILABILITY_POLICY = "provisional-close-plus-1min"
# Baseline evidence observed on the numeric-stock cross-section; reported only,
# never used as an accept/reject band (reported trading value may use VWAP or
# vendor-specific rounding).
TRADING_VALUE_RATIO_BASELINE = (0.7615, 1.4986)


class InstrumentClassification(StrEnum):
    """The three immutable routing classes for a source identifier at a session."""

    COMMON_STOCK = "common_stock"
    NON_EQUITY = "non_equity"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True, slots=True)
class InstrumentMasterRecord:
    """One effective-dated security-master interval for a source identifier."""

    source_identifier: str
    instrument_id: str
    asset_type: str
    is_common_stock: bool
    listed_from: date
    delisted_on: date | None = None
    tradable_from: date | None = None
    tradable_to: date | None = None
    available_time: datetime | None = None

    def __post_init__(self) -> None:
        if not self.source_identifier:
            raise ValueError("source_identifier must be non-empty")
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        if not self.asset_type:
            raise ValueError("asset_type must be non-empty")
        if self.delisted_on is not None and self.delisted_on <= self.listed_from:
            raise ValueError("delisted_on must be after listed_from")
        if (
            self.tradable_from is not None
            and self.tradable_to is not None
            and self.tradable_to < self.tradable_from
        ):
            raise ValueError("tradable_to must be on or after tradable_from")
        if self.available_time is not None and self.available_time.tzinfo is None:
            raise ValueError("available_time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class InstrumentMasterSnapshot:
    """Versioned, immutable security-master snapshot.

    Keyed by source identifier with effective intervals; ``content_hash`` binds
    the exact bytes of every record so the report and dataset lineage can carry
    a stable fingerprint.
    """

    version: str
    records: tuple[InstrumentMasterRecord, ...]
    generated_time: datetime

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")
        if not self.records:
            raise ValueError("master snapshot must contain at least one record")
        keys = [
            (r.source_identifier, r.listed_from, r.delisted_on, r.tradable_from, r.tradable_to)
            for r in self.records
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("master snapshot must not repeat effective intervals")

    @property
    def content_hash(self) -> str:
        lines = sorted(
            "|".join(
                (
                    r.source_identifier,
                    r.instrument_id,
                    r.asset_type,
                    str(int(r.is_common_stock)),
                    r.listed_from.isoformat(),
                    r.delisted_on.isoformat() if r.delisted_on else "",
                    r.tradable_from.isoformat() if r.tradable_from else "",
                    r.tradable_to.isoformat() if r.tradable_to else "",
                    r.available_time.isoformat() if r.available_time else "",
                )
            )
            for r in self.records
        )
        return hashlib.sha256(
            (f"{self.version}\n" + "\n".join(lines)).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class KRXSessionCalendar:
    """Versioned, effective-dated KRX session calendar."""

    version: str
    sessions: tuple[date, ...]
    generated_time: datetime

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")
        if not self.sessions:
            raise ValueError("calendar must contain at least one session")
        if list(self.sessions) != sorted(self.sessions):
            raise ValueError("sessions must be sorted ascending")
        if len(set(self.sessions)) != len(self.sessions):
            raise ValueError("sessions must not repeat")

    @property
    def content_hash(self) -> str:
        payload = "\n".join(d.isoformat() for d in self.sessions)
        return hashlib.sha256((f"{self.version}\n{payload}").encode()).hexdigest()

    def is_session(self, day: date) -> bool:
        index = bisect.bisect_left(self.sessions, day)
        return index < len(self.sessions) and self.sessions[index] == day

    def previous_session(self, day: date) -> date | None:
        index = bisect.bisect_left(self.sessions, day)
        return self.sessions[index - 1] if index > 0 else None


@dataclass(frozen=True, slots=True)
class CorporateActionInterval:
    """One action or no-action record for an ``(instrument, prev, session)`` pair."""

    instrument_id: str
    previous_session: date
    session: date
    action_code: str
    adjustment_factor: float = 1.0

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        if not self.action_code:
            raise ValueError("action_code must be non-empty")
        if self.previous_session >= self.session:
            raise ValueError("previous_session must be before session")
        if not math.isfinite(self.adjustment_factor) or self.adjustment_factor <= 0:
            raise ValueError("adjustment_factor must be finite and positive")
        if self.action_code == "no_action" and not math.isclose(self.adjustment_factor, 1.0):
            raise ValueError("no_action adjustment_factor must equal 1")


@dataclass(frozen=True, slots=True)
class CorporateActionSnapshot:
    """Versioned, immutable action/no-action interval snapshot."""

    version: str
    intervals: tuple[CorporateActionInterval, ...]
    generated_time: datetime

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")
        if not self.intervals:
            raise ValueError("corporate action snapshot must contain at least one interval")
        keys = [(i.instrument_id, i.previous_session, i.session) for i in self.intervals]
        if len(set(keys)) != len(keys):
            raise ValueError("corporate action intervals must not repeat")

    @property
    def content_hash(self) -> str:
        lines = sorted(
            "|".join(
                (
                    i.instrument_id,
                    i.previous_session.isoformat(),
                    i.session.isoformat(),
                    i.action_code,
                    repr(i.adjustment_factor),
                )
            )
            for i in self.intervals
        )
        return hashlib.sha256(
            (f"{self.version}\n" + "\n".join(lines)).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureAvailabilityRecord:
    """Field-level availability evidence for one feature column."""

    feature_name: str
    source_field: str
    availability_rule: str
    source_version: str
    source_hash: str
    null_rate: float
    use_class: str
    available_time: datetime | None = None

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("feature_name must be non-empty")
        if not self.availability_rule or not self.source_version or not self.source_hash:
            raise ValueError("feature availability evidence must identify rule, version, and hash")
        if not 0.0 <= self.null_rate <= 1.0:
            raise ValueError("null_rate must be within [0, 1]")
        if self.use_class not in ("research", "audit"):
            raise ValueError(f"unknown use_class {self.use_class!r}")
        if self.available_time is not None and self.available_time.tzinfo is None:
            raise ValueError("available_time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class StockDataQualityPolicy:
    """Policy that decides certification requirements and availability evidence.

    ``PROVISIONAL`` may be produced from the present source with a numeric-code
    heuristic (a formatting diagnostic only). ``RESEARCH``/``PRODUCTION``
    require a validated security master and availability evidence; only
    ``PRODUCTION`` requires exhaustive corporate-action coverage. The
    close+1-minute rule produces PROVISIONAL data only; higher tiers require
    the calendar/availability evidence supplied here.
    """

    certification: DatasetCertification = DatasetCertification.PROVISIONAL
    calendar: KRXSessionCalendar | None = None
    availability_policy: str = PROVISIONAL_AVAILABILITY_POLICY
    feature_availability: tuple[FeatureAvailabilityRecord, ...] = ()
    trading_value_ratio_baseline: tuple[float, float] = TRADING_VALUE_RATIO_BASELINE

    @property
    def availability_policy_hash(self) -> str:
        lines = [self.availability_policy]
        lines.extend(
            "|".join(
                (
                    f.feature_name,
                    f.source_field,
                    f.availability_rule,
                    f.source_version,
                    f.source_hash,
                    repr(f.null_rate),
                    f.use_class,
                    f.available_time.isoformat() if f.available_time else "",
                )
            )
            for f in sorted(self.feature_availability, key=lambda r: r.feature_name)
        )
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    def requires_action_coverage(self) -> bool:
        return self.certification is DatasetCertification.PRODUCTION


@dataclass(frozen=True, slots=True)
class StockDataQualityReport:
    """Immutable quality outcome: partitioned frames plus deterministic evidence.

    ``eligible`` / ``quarantined`` / ``non_equity`` hold the routed source rows
    (with lineage, identity, and reason columns). ``to_json_dict`` is the
    machine-readable ``quality_report.json`` body, byte-deterministic for the
    same inputs, and ``quality_report_hash`` binds those exact bytes.
    """

    certification: DatasetCertification
    reason_counts: dict[str, int]
    affected_identifiers: dict[str, tuple[str, ...]]
    affected_files: dict[str, tuple[str, ...]]
    column_null_rates: dict[str, float]
    fully_null_columns: tuple[str, ...]
    coverage_interval: tuple[str, str]
    action_coverage: dict[str, int]
    trading_value_ratio: dict[str, float]
    benchmark_routed_identifiers: tuple[str, ...]
    hashes: dict[str, str]
    content_hash: str = ""
    eligible: pl.DataFrame | None = None
    quarantined: pl.DataFrame | None = None
    non_equity: pl.DataFrame | None = None
    source_root: str = ""
    generated_time: datetime | None = None

    @property
    def eligible_row_count(self) -> int:
        return self.eligible.height if self.eligible is not None else 0

    @property
    def quarantined_row_count(self) -> int:
        return self.quarantined.height if self.quarantined is not None else 0

    @property
    def non_equity_row_count(self) -> int:
        return self.non_equity.height if self.non_equity is not None else 0

    def with_content_hash(self, content_hash: str) -> StockDataQualityReport:
        """Return a copy carrying the canonical dataset content hash."""
        return replace(self, content_hash=content_hash)

    def with_generated_time(self, generated_time: datetime) -> StockDataQualityReport:
        """Return a copy with the deterministic curation timestamp."""
        return replace(self, generated_time=generated_time)

    def to_json_dict(self) -> dict[str, object]:
        # ``generated_time`` is provenance metadata only and is deliberately
        # excluded from the serialized report so identical source bytes and
        # evidence always produce byte-equivalent reports and hashes.
        return {
            "certification": self.certification.value,
            "row_counts": {
                "eligible": self.eligible_row_count,
                "quarantined": self.quarantined_row_count,
                "non_equity": self.non_equity_row_count,
            },
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "affected_identifiers": {
                reason: sorted(ids) for reason, ids in sorted(self.affected_identifiers.items())
            },
            "affected_files": {
                reason: sorted(files) for reason, files in sorted(self.affected_files.items())
            },
            "column_null_rates": dict(sorted(self.column_null_rates.items())),
            "fully_null_columns": sorted(self.fully_null_columns),
            "coverage_interval": [self.coverage_interval[0], self.coverage_interval[1]],
            "action_coverage": dict(sorted(self.action_coverage.items())),
            "trading_value_ratio": dict(sorted(self.trading_value_ratio.items())),
            "benchmark_routed_identifiers": sorted(self.benchmark_routed_identifiers),
            "hashes": dict(sorted(self.hashes.items())),
            "content_hash": self.content_hash,
            "source_root": self.source_root,
        }


def _utc_close(session: pl.Expr) -> pl.Expr:
    return (
        session.dt.combine(pl.lit(_KRX_CLOSE_TIME))
        .dt.replace_time_zone("Asia/Seoul")
        .dt.convert_time_zone("UTC")
    )


def _utc_available(session: pl.Expr) -> pl.Expr:
    return (
        session.dt.combine(pl.lit(_KRX_AVAILABLE_TIME))
        .dt.replace_time_zone("Asia/Seoul")
        .dt.convert_time_zone("UTC")
    )


def _master_lookup(master: InstrumentMasterSnapshot) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_identifier": [r.source_identifier for r in master.records],
            "instrument_id": [r.instrument_id for r in master.records],
            "is_common_stock": [r.is_common_stock for r in master.records],
            "asset_type": [r.asset_type for r in master.records],
            "listed_from": [r.listed_from for r in master.records],
            "delisted_on": [r.delisted_on for r in master.records],
            "tradable_from": [r.tradable_from for r in master.records],
            "tradable_to": [r.tradable_to for r in master.records],
        }
    )


def validate_canonical_stock_panel(
    frame: pl.DataFrame,
    instrument_master: InstrumentMasterSnapshot | None,
    corporate_actions: CorporateActionSnapshot | None,
    policy: StockDataQualityPolicy,
) -> StockDataQualityReport:
    """Route every source row to eligible/quarantined/non_equity with evidence.

    Vectorized classification (security master effective at the session, or the
    provisional six-digit diagnostic), structural bar invariants, action/no-action
    interval coverage gating for derived returns, trading-value identity ratio
    reporting, and deterministic per-column null statistics.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars DataFrame")
    missing = [c for c in ("date", "ticker") if c not in frame.columns]
    if missing:
        raise ValueError(f"panel must carry {', '.join(missing)}")

    requires_certified_inputs = policy.certification is not DatasetCertification.PROVISIONAL
    if requires_certified_inputs and instrument_master is None:
        raise ValueError(
            f"{policy.certification.value} requires a validated InstrumentMasterSnapshot"
        )
    if policy.requires_action_coverage() and corporate_actions is None:
        raise ValueError(f"{policy.certification.value} requires a CorporateActionSnapshot")
    if requires_certified_inputs and policy.calendar is None:
        raise ValueError(f"{policy.certification.value} requires a KRXSessionCalendar")
    if requires_certified_inputs:
        _validate_feature_availability(frame, policy)

    working = frame.with_columns(pl.col("date").cast(pl.Date).alias("session"))
    working = _classify_rows(working, instrument_master)
    working = _apply_bar_invariants(working)
    working = _apply_calendar_membership(working, policy.calendar, policy)
    working = _apply_action_coverage(working, corporate_actions, policy)

    eligible = working.filter(pl.col(QUALITY_STATUS_COLUMN) == ELIGIBLE_STATUS)
    quarantined = working.filter(pl.col(QUALITY_STATUS_COLUMN) == QUARANTINED_STATUS)
    non_equity = working.filter(pl.col(QUALITY_STATUS_COLUMN) == NON_EQUITY_STATUS)

    coverage_interval = _interval_bounds(eligible, working)
    reason_counts = _reason_counts(quarantined)
    affected_identifiers = _affected_by_reason(quarantined, "instrument_id")
    affected_files = _affected_by_reason(quarantined, "source_file")
    column_null_rates = _column_null_rates(frame)
    fully_null = tuple(
        name for name, rate in sorted(column_null_rates.items()) if rate >= 1.0
    )
    action_coverage = _action_coverage_counts(working)
    trading_value_ratio = _trading_value_ratio(working)
    benchmark_ids = tuple(sorted(non_equity["instrument_id"].unique().to_list()))

    hashes = {
        "master": instrument_master.content_hash if instrument_master else "",
        "calendar": policy.calendar.content_hash if policy.calendar else "",
        "actions": corporate_actions.content_hash if corporate_actions else "",
        "availability_policy": policy.availability_policy_hash,
        "quality_report": "",
    }

    report = StockDataQualityReport(
        certification=policy.certification,
        reason_counts=reason_counts,
        affected_identifiers=affected_identifiers,
        affected_files=affected_files,
        column_null_rates=column_null_rates,
        fully_null_columns=fully_null,
        coverage_interval=coverage_interval,
        action_coverage=action_coverage,
        trading_value_ratio=trading_value_ratio,
        benchmark_routed_identifiers=benchmark_ids,
        hashes=hashes,
        eligible=eligible,
        quarantined=quarantined,
        non_equity=non_equity,
        generated_time=datetime.now(UTC),
    )
    report_hash = hashlib.sha256(
        json.dumps(report.to_json_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return replace(report, hashes={**hashes, "quality_report": report_hash})


def _classify_rows(
    working: pl.DataFrame,
    instrument_master: InstrumentMasterSnapshot | None,
) -> pl.DataFrame:
    session = pl.col("session").cast(pl.Date)
    instrument_id = (pl.lit("KRX:") + pl.col("ticker").cast(pl.Utf8)).alias("instrument_id")
    if instrument_master is not None:
        lookup = _master_lookup(instrument_master)
        working = working.join(
            lookup,
            left_on="ticker",
            right_on="source_identifier",
            how="left",
        )
        listed_ok = (session >= pl.col("listed_from").cast(pl.Date)) & (
            pl.col("delisted_on").is_null()
            | (session < pl.col("delisted_on").cast(pl.Date))
        )
        tradable_ok = (
            pl.col("tradable_from").is_null()
            | (session >= pl.col("tradable_from").cast(pl.Date))
        ) & (
            pl.col("tradable_to").is_null()
            | (session <= pl.col("tradable_to").cast(pl.Date))
        )
        canonical_id = pl.when(pl.col("instrument_id").is_not_null()).then(
            pl.col("instrument_id")
        ).otherwise(pl.lit("KRX:") + pl.col("ticker").cast(pl.Utf8))
        classification = (
            pl.when(pl.col("is_common_stock").is_null())
            .then(pl.lit(InstrumentClassification.UNCLASSIFIED))
            .when(~pl.col("is_common_stock"))
            .then(pl.lit(InstrumentClassification.NON_EQUITY))
            .when(~(listed_ok & tradable_ok))
            .then(pl.lit(InstrumentClassification.UNCLASSIFIED))
            .otherwise(pl.lit(InstrumentClassification.COMMON_STOCK))
        )
        return working.with_columns(
            canonical_id.alias("instrument_id"),
            classification.alias("_classification"),
        )
    heuristic = (
        pl.when(pl.col("ticker").cast(pl.Utf8).is_in(INDEX_TICKERS))
        .then(pl.lit(InstrumentClassification.NON_EQUITY))
        .when(pl.col("ticker").cast(pl.Utf8).str.contains(_TICKER_RE.pattern))
        .then(pl.lit(InstrumentClassification.COMMON_STOCK))
        .otherwise(pl.lit(InstrumentClassification.UNCLASSIFIED))
    )
    return working.with_columns(
        instrument_id,
        heuristic.alias("_classification"),
    )


def _apply_bar_invariants(working: pl.DataFrame) -> pl.DataFrame:
    ohlc_columns = [c for c in _OHLC_COLUMNS if c in working.columns]
    missing = [c for c in _OHLC_COLUMNS if c not in working.columns]
    if missing:
        raise ValueError(f"panel must carry {', '.join(missing)}")

    ohlc_invalid = pl.any_horizontal(
        [
            pl.col(c).is_null() | (pl.col(c) <= 0) | ~pl.col(c).is_finite()
            for c in ohlc_columns
        ]
    )
    ordering_invalid = (
        (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.max_horizontal("open", "close") > pl.col("high"))
    )
    executable = pl.lit(False)
    if "volume" in working.columns:
        executable = executable | pl.col("volume").is_null() | (pl.col("volume") <= 0)
    if "trading_value" in working.columns:
        executable = executable | pl.col("trading_value").is_null() | (
            pl.col("trading_value") <= 0
        )
    capitalization = pl.lit(False)
    if "market_cap" in working.columns:
        capitalization = pl.col("market_cap") < 0

    is_stock = pl.col("_classification") == InstrumentClassification.COMMON_STOCK
    reason = (
        pl.when(pl.col("_classification") == InstrumentClassification.UNCLASSIFIED)
        .then(pl.lit(REASON_UNCLASSIFIED))
        .when(pl.col("_classification") == InstrumentClassification.NON_EQUITY)
        .then(
            pl.when(pl.col("ticker").cast(pl.Utf8).is_in(INDEX_TICKERS))
            .then(pl.lit(REASON_INDEX))
            .otherwise(pl.lit(REASON_NON_EQUITY))
        )
        .when(is_stock & ohlc_invalid)
        .then(pl.lit(REASON_OHLC))
        .when(is_stock & ordering_invalid & ~ohlc_invalid)
        .then(pl.lit(REASON_ORDERING))
        .when(is_stock & executable & ~ohlc_invalid)
        .then(pl.lit(REASON_EXECUTABLE))
        .when(is_stock & capitalization)
        .then(pl.lit(REASON_CAPITALIZATION))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )

    status = (
        pl.when(pl.col("_classification") == InstrumentClassification.NON_EQUITY)
        .then(pl.lit(NON_EQUITY_STATUS))
        .when(is_stock & reason.is_null())
        .then(pl.lit(ELIGIBLE_STATUS))
        .otherwise(pl.lit(QUARANTINED_STATUS))
    )

    return working.with_columns(
        reason.alias(QUALITY_REASON_COLUMN),
        status.alias(QUALITY_STATUS_COLUMN),
        _utc_close(pl.col("session")).alias("observation_time"),
        _utc_available(pl.col("session")).alias("available_time"),
    )


_AVAILABILITY_BASE_COLUMNS = frozenset(
    {
        "date",
        "ticker",
        "instrument_id",
        "session",
        "observation_time",
        "available_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trading_value",
        "market_cap",
        "sector",
        "name",
        "market",
        "SECT_TP_NM",
        "MARKET",
        "fiscal_year",
        "year",
        "reprt_code",
        "reprt_code_right",
        "disclosure_date",
        "disclosure_date_right",
        QUALITY_STATUS_COLUMN,
        QUALITY_REASON_COLUMN,
        "source_file",
        "action_interval_covered",
    }
)
_FORWARD_COLUMNS = ("target_", "label_")


def _validate_feature_availability(
    frame: pl.DataFrame, policy: StockDataQualityPolicy
) -> None:
    feature_columns = [
        column
        for column in frame.columns
        if column not in _AVAILABILITY_BASE_COLUMNS
        and not column.startswith(_FORWARD_COLUMNS)
    ]
    records = {record.feature_name: record for record in policy.feature_availability}
    missing = [column for column in feature_columns if column not in records]
    if missing:
        raise ValueError(
            f"research requires FeatureAvailabilityRecord for {missing}"
        )
    not_research = [
        column for column in feature_columns if records[column].use_class != "research"
    ]
    if not_research:
        raise ValueError(
            f"research features must use use_class='research': {not_research}"
        )


def _apply_calendar_membership(
    working: pl.DataFrame,
    calendar: KRXSessionCalendar | None,
    policy: StockDataQualityPolicy,
) -> pl.DataFrame:
    if policy.certification is DatasetCertification.PROVISIONAL or calendar is None:
        return working
    session_ok = pl.col("session").is_in(calendar.sessions)
    is_stock = pl.col("_classification") == InstrumentClassification.COMMON_STOCK
    return working.with_columns(
        pl.when(is_stock & ~session_ok & pl.col(QUALITY_REASON_COLUMN).is_null())
        .then(pl.lit(REASON_NON_SESSION))
        .otherwise(pl.col(QUALITY_REASON_COLUMN))
        .alias(QUALITY_REASON_COLUMN),
        pl.when(is_stock & ~session_ok & pl.col(QUALITY_REASON_COLUMN).is_null())
        .then(pl.lit(QUARANTINED_STATUS))
        .otherwise(pl.col(QUALITY_STATUS_COLUMN))
        .alias(QUALITY_STATUS_COLUMN),
    )


def _apply_action_coverage(
    working: pl.DataFrame,
    corporate_actions: CorporateActionSnapshot | None,
    policy: StockDataQualityPolicy,
) -> pl.DataFrame:
    out = working.sort(["instrument_id", "session"]).with_columns(
        pl.col("session")
        .shift(1)
        .over("instrument_id", order_by="session")
        .alias("_previous_session")
    )
    covered = pl.lit(None, dtype=pl.Boolean).alias("action_interval_covered")
    if corporate_actions is not None:
        action_df = pl.DataFrame(
            {
                "instrument_id": [i.instrument_id for i in corporate_actions.intervals],
                "_previous_session": [i.previous_session for i in corporate_actions.intervals],
                "session": [i.session for i in corporate_actions.intervals],
                "_action_match": [True] * len(corporate_actions.intervals),
            }
        )
        out = out.join(action_df, on=["instrument_id", "_previous_session", "session"], how="left")
        covered = (pl.col("_previous_session").is_null() | pl.col("_action_match").is_not_null())
        covered = covered.alias("action_interval_covered")

        if policy.requires_action_coverage():
            is_stock = pl.col("_classification") == InstrumentClassification.COMMON_STOCK
            uncovered = (
                pl.col("_previous_session").is_not_null()
                & pl.col("_action_match").is_null()
            )
            out = out.with_columns(
                pl.when(is_stock & uncovered)
                .then(pl.lit(REASON_ACTION_COVERAGE))
                .otherwise(pl.col(QUALITY_REASON_COLUMN))
                .alias(QUALITY_REASON_COLUMN),
                pl.when(is_stock & uncovered)
                .then(pl.lit(QUARANTINED_STATUS))
                .otherwise(pl.col(QUALITY_STATUS_COLUMN))
                .alias(QUALITY_STATUS_COLUMN),
            )
    return out.with_columns(covered)


def _interval_bounds(
    eligible: pl.DataFrame,
    working: pl.DataFrame,
) -> tuple[str, str]:
    source = eligible if eligible.height else working
    if source.height == 0:
        return "", ""
    return _iso_min(source["session"].min()), _iso_min(source["session"].max())


def _iso_min(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return ""


def _reason_counts(quarantined: pl.DataFrame) -> dict[str, int]:
    if quarantined.height == 0:
        return {}
    grouped = (
        quarantined.group_by(QUALITY_REASON_COLUMN)
        .len()
        .rename({QUALITY_REASON_COLUMN: "reason", "len": "count"})
    )
    return {
        str(row["reason"]): int(row["count"])
        for row in grouped.sort("reason").iter_rows(named=True)
    }


def _affected_by_reason(frame: pl.DataFrame, column: str) -> dict[str, tuple[str, ...]]:
    if frame.height == 0 or column not in frame.columns:
        return {}
    grouped = frame.group_by(QUALITY_REASON_COLUMN).agg(pl.col(column).unique())
    result: dict[str, tuple[str, ...]] = {}
    for row in grouped.iter_rows(named=True):
        reason = str(row[QUALITY_REASON_COLUMN])
        values = row[column]
        result[reason] = tuple(sorted(str(v) for v in (values or [])))
    return result


def _column_null_rates(frame: pl.DataFrame) -> dict[str, float]:
    height = frame.height
    if height == 0:
        return dict.fromkeys(frame.columns, 1.0)
    return {
        name: float(frame[name].null_count()) / height
        for name in frame.columns
    }


def _action_coverage_counts(working: pl.DataFrame) -> dict[str, int]:
    if "action_interval_covered" not in working.columns:
        return {"covered": 0, "uncovered": 0}
    # Only rows that actually cross an interval (previous session exists)
    # contribute to coverage; a first session has no interval to cover.
    intervals = working.filter(pl.col("_previous_session").is_not_null())
    counts = intervals.group_by("action_interval_covered").len()
    mapping = {
        bool(row["action_interval_covered"]): int(row["len"])
        for row in counts.iter_rows(named=True)
        if row["action_interval_covered"] is not None
    }
    return {
        "covered": mapping.get(True, 0),
        "uncovered": mapping.get(False, 0),
    }


def _trading_value_ratio(working: pl.DataFrame) -> dict[str, float]:
    required = {"trading_value", "close", "volume"}
    if not required.issubset(working.columns):
        return {}
    valid = working.filter(
        (pl.col("_classification") == InstrumentClassification.COMMON_STOCK)
        & pl.col("close").is_not_null()
        & (pl.col("close") > 0)
        & pl.col("volume").is_not_null()
        & (pl.col("volume") > 0)
        & (pl.col("close") * pl.col("volume") > 0)
    )
    if valid.height == 0:
        return {}
    ratio = (pl.col("trading_value") / (pl.col("close") * pl.col("volume"))).alias("ratio")
    stats = valid.select(
        ratio.min().alias("min"),
        ratio.quantile(0.5).alias("median"),
        ratio.max().alias("max"),
    ).to_dicts()[0]
    return {
        "min": _as_float(stats["min"]),
        "median": _as_float(stats["median"]),
        "max": _as_float(stats["max"]),
        "sample_rows": float(valid.height),
    }


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, pl.Series):
        item = value.item()
        if isinstance(item, (int, float)):
            return float(item)
    return 0.0
