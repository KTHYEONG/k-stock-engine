"""Gold-layer input validation and eligibility auditing.

Bridges certified Silver tables to backtest-consumable Gold inputs by enforcing:
  - 60-trading-day warmup before the validation window start
  - Per-instrument bar continuity (session coverage, OHLC sanity, duplicate guards)
  - DART fact eligibility: 4 consecutive quarters + required fact set
  - Corporate-action sentinel exclusion (no_action implies no price-adjustment data)

All exclusion decisions are recorded with a structured reason; no silent imputation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl

from src.core.datasets import DatasetCertification
from src.core.time import KRX_TZ, SessionCalendar
from src.data.schemas import PITDataError

# ──────────────────────────────────────────────────────────────────
# Domain types
# ──────────────────────────────────────────────────────────────────

WARMUP_SESSIONS = 60  # minimum trading sessions before validation window

_REQUIRED_FACTS: frozenset[str] = frozenset(
    ["sales", "operating_profit", "net_income", "assets", "equity", "operating_cash_flow", "gross_profit"]
)

_FISCAL_RE = re.compile(r"^(\d{4})Q([1-4])$")

_LOGGER = logging.getLogger(__name__)


class BarExclusionReason(StrEnum):
    MISSING_SESSIONS = "missing_sessions"
    DUPLICATE_SESSIONS = "duplicate_sessions"
    OHLC_VIOLATION = "ohlc_violation"
    NEGATIVE_VOLUME = "negative_volume"
    NEGATIVE_TRADING_VALUE = "negative_trading_value"
    INSUFFICIENT_HISTORY = "insufficient_history"


class DartExclusionReason(StrEnum):
    INSUFFICIENT_QUARTERS = "insufficient_quarters"
    MISSING_REQUIRED_FACTS = "missing_required_facts"
    NO_PIT_FACTS_AVAILABLE = "no_pit_facts_available"


class CorporateActionExclusionReason(StrEnum):
    SENTINEL_NO_ACTION = "sentinel_no_action"  # only sentinel → no real price-adj data


@dataclass(frozen=True, slots=True)
class BarAuditResult:
    instrument_id: str
    eligible: bool
    exclusion_reasons: tuple[BarExclusionReason, ...]
    sessions_found: int
    sessions_expected: int


@dataclass(frozen=True, slots=True)
class DartEligibilityResult:
    company_id: str
    eligible: bool
    exclusion_reasons: tuple[DartExclusionReason, ...]
    consecutive_quarters_found: int
    required_facts_present: frozenset[str]


@dataclass(frozen=True, slots=True)
class WarmupCheckResult:
    warmup_ok: bool
    warmup_sessions_found: int
    warmup_sessions_required: int
    first_validation_session: datetime | None


@dataclass(frozen=True, slots=True)
class GoldAuditManifest:
    """Reproducible artifact summarising all pre-Gold exclusion decisions."""

    warmup: WarmupCheckResult
    bar_audit: tuple[BarAuditResult, ...]
    dart_eligibility: tuple[DartEligibilityResult, ...]
    ca_excluded_instrument_ids: frozenset[str]
    eligible_instrument_ids: frozenset[str]
    manifest_hash: str


# ──────────────────────────────────────────────────────────────────
# 1. Warmup check
# ──────────────────────────────────────────────────────────────────

def check_warmup_sessions(
    calendar: SessionCalendar,
    *,
    validation_start: date,
    warmup_required: int = WARMUP_SESSIONS,
) -> WarmupCheckResult:
    """Verify ≥ warmup_required trading sessions exist before validation_start.

    Sessions strictly before validation_start are counted as warmup.
    """
    warmup_sessions = [
        s for s in calendar.sessions
        if s.astimezone(KRX_TZ).date() < validation_start
    ]
    found = len(warmup_sessions)
    first_val: datetime | None = None
    for s in calendar.sessions:
        if s.astimezone(KRX_TZ).date() >= validation_start:
            first_val = s
            break
    return WarmupCheckResult(
        warmup_ok=found >= warmup_required,
        warmup_sessions_found=found,
        warmup_sessions_required=warmup_required,
        first_validation_session=first_val,
    )


# ──────────────────────────────────────────────────────────────────
# 2. Bar continuity audit
# ──────────────────────────────────────────────────────────────────

def audit_bar_continuity(
    daily_market: pl.DataFrame,
    calendar: SessionCalendar,
    *,
    window_start: date,
    window_end: date,
) -> tuple[BarAuditResult, ...]:
    """Audit daily bar completeness and validity for each instrument in the window.

    Each instrument is checked for: missing sessions, duplicate sessions, OHLC
    violations, and negative volume/trading_value.
    """
    window_sessions: list[datetime] = [
        s for s in calendar.sessions
        if window_start <= s.astimezone(KRX_TZ).date() <= window_end
    ]
    if not window_sessions:
        return ()
    if daily_market.is_empty():
        return ()

    expected_count = len(window_sessions)
    # Compare by KRX local date — calendar stores 00:00, daily_market 09:00 for the same session
    # Filter daily_market to window using Polars
    try:
        in_window = daily_market.filter(
            pl.col("session").dt.date().is_between(window_start, window_end)
        )
    except Exception:
        rows = daily_market.to_dicts()
        in_window = pl.DataFrame(
            [r for r in rows if r.get("session") is not None
             and window_start <= _to_date(r["session"]) <= window_end]
        ) if rows else daily_market.clear()

    if in_window.is_empty():
        return ()

    # Aggregate in Polars instead of converting hundreds of thousands of bars
    # into Python dictionaries; this is the measured Gold OOM hotspot.
    iid = pl.col("instrument_id").cast(pl.String, strict=False)
    o = pl.col("open").cast(pl.Float64, strict=False)
    h = pl.col("high").cast(pl.Float64, strict=False)
    lo = pl.col("low").cast(pl.Float64, strict=False)
    c = pl.col("close").cast(pl.Float64, strict=False)
    invalid_ohlc = (
        o.is_null() | h.is_null() | lo.is_null() | c.is_null()
        | o.is_nan() | h.is_nan() | lo.is_nan() | c.is_nan()
        | (lo > o) | (o > h) | (lo > c) | (c > h)
    )
    volume = pl.col("volume").cast(pl.Float64, strict=False)
    trading_value = pl.col("trading_value").cast(pl.Float64, strict=False)
    aggregates = (
        in_window.with_columns(iid.alias("_instrument_id"))
        .filter(pl.col("_instrument_id").is_not_null() & (pl.col("_instrument_id") != ""))
        .with_columns(
            pl.col("session").dt.date().alias("_session_date"),
            invalid_ohlc.alias("_invalid_ohlc"),
            (volume < 0).fill_null(False).alias("_negative_volume"),
            (trading_value < 0).fill_null(False).alias("_negative_trading_value"),
        )
        .group_by("_instrument_id")
        .agg(
            pl.col("_session_date").count().alias("_session_row_count"),
            pl.col("_session_date").n_unique().alias("_session_count"),
            pl.col("_invalid_ohlc").any().alias("_invalid_ohlc_any"),
            pl.col("_negative_volume").any().alias("_negative_volume_any"),
            pl.col("_negative_trading_value").any().alias("_negative_trading_value_any"),
        )
        .sort("_instrument_id")
    )

    results: list[BarAuditResult] = []
    for row in aggregates.iter_rows(named=True):
        reasons: set[BarExclusionReason] = set()
        if int(row["_session_row_count"]) != int(row["_session_count"]):
            reasons.add(BarExclusionReason.DUPLICATE_SESSIONS)
        if int(row["_session_count"]) != expected_count:
            reasons.add(BarExclusionReason.MISSING_SESSIONS)
        if bool(row["_invalid_ohlc_any"]):
            reasons.add(BarExclusionReason.OHLC_VIOLATION)
        if bool(row["_negative_volume_any"]):
            reasons.add(BarExclusionReason.NEGATIVE_VOLUME)
        if bool(row["_negative_trading_value_any"]):
            reasons.add(BarExclusionReason.NEGATIVE_TRADING_VALUE)
        unique_reasons = tuple(sorted(reasons, key=lambda reason: reason.value))
        results.append(
            BarAuditResult(
                instrument_id=str(row["_instrument_id"]),
                eligible=len(unique_reasons) == 0,
                exclusion_reasons=unique_reasons,
                sessions_found=int(row["_session_count"]),
                sessions_expected=expected_count,
            )
        )
    return tuple(results)



# ──────────────────────────────────────────────────────────────────
# 3. DART fact eligibility
# ──────────────────────────────────────────────────────────────────

def audit_dart_fact_eligibility(
    financial_facts: pl.DataFrame,
    *,
    decision_time: datetime,
    required_facts: frozenset[str] = _REQUIRED_FACTS,
    min_consecutive_quarters: int = 4,
) -> tuple[DartEligibilityResult, ...]:
    """Determine per-company DART fact availability as of decision_time.

    A company is DART-eligible when:
    - ≥ min_consecutive_quarters consecutive fiscal quarters are available PIT.
    - All required_facts are present in the latest quarter.
    Ineligible companies receive explicit reasons; no silent imputation.
    """
    if financial_facts.is_empty():
        return ()

    # PIT filter
    pit_rows: list[dict[str, Any]] = []
    for row in financial_facts.to_dicts():
        av = row.get("available_at")
        if av is None:
            continue
        try:
            if av <= decision_time:
                pit_rows.append(row)
        except TypeError:
            continue

    if not pit_rows:
        return ()

    # Resolve best value per (company_id, fiscal_period, fact)
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in pit_rows:
        if not row.get("consolidated"):
            continue
        cid = str(row.get("company_id") or "")
        period = str(row.get("fiscal_period") or "")
        fact = str(row.get("fact") or "")
        if not cid or not period or not fact:
            continue
        if _parse_fiscal_key(period) == (-1, -1):
            continue
        by_key.setdefault((cid, period, fact), []).append(row)

    periods_by_company: dict[str, set[str]] = {}
    facts_by_company_period: dict[tuple[str, str], set[str]] = {}

    for (cid, period, fact), cands in by_key.items():
        try:
            best = max(cands, key=lambda r: r["available_at"])
            val = best.get("value")
            fv = float(val)  # type: ignore[arg-type]
            if math.isfinite(fv):
                periods_by_company.setdefault(cid, set()).add(period)
                facts_by_company_period.setdefault((cid, period), set()).add(fact)
        except (TypeError, ValueError):
            continue

    results: list[DartEligibilityResult] = []
    for cid in sorted(periods_by_company):
        all_periods = sorted(
            periods_by_company[cid],
            key=_parse_fiscal_key,
        )
        if not all_periods:
            results.append(_dart_result(cid, [DartExclusionReason.NO_PIT_FACTS_AVAILABLE], 0, frozenset()))
            continue

        latest = all_periods[-1]
        consec = _count_consecutive_quarters(latest, periods_by_company[cid])
        present_facts: frozenset[str] = frozenset(facts_by_company_period.get((cid, latest), set()))
        missing = required_facts - present_facts

        reasons: list[DartExclusionReason] = []
        if consec < min_consecutive_quarters:
            reasons.append(DartExclusionReason.INSUFFICIENT_QUARTERS)
        if missing:
            reasons.append(DartExclusionReason.MISSING_REQUIRED_FACTS)

        results.append(_dart_result(cid, reasons, consec, present_facts))

    return tuple(results)


def _dart_result(
    cid: str,
    reasons: list[DartExclusionReason],
    consec: int,
    present: frozenset[str],
) -> DartEligibilityResult:
    return DartEligibilityResult(
        company_id=cid,
        eligible=len(reasons) == 0,
        exclusion_reasons=tuple(sorted(set(reasons), key=lambda r: r.value)),
        consecutive_quarters_found=consec,
        required_facts_present=present,
    )


def _count_consecutive_quarters(latest: str, available_periods: set[str]) -> int:
    """Count consecutive quarters in available_periods ending at latest."""
    count = 0
    cur = latest
    while cur in available_periods:
        count += 1
        prev = _prev_quarter(cur)
        if prev is None:
            break
        cur = prev
    return count


def _prev_quarter(period: str) -> str | None:
    parsed = _parse_fiscal_key(period)
    if parsed == (-1, -1):
        return None
    year, q = parsed
    total = year * 4 + (q - 1) - 1
    if total < 0:
        return None
    return f"{total // 4}Q{(total % 4) + 1}"


def _parse_fiscal_key(period: str) -> tuple[int, int]:
    m = _FISCAL_RE.match(str(period))
    if not m:
        return (-1, -1)
    return int(m.group(1)), int(m.group(2))


# ──────────────────────────────────────────────────────────────────
# 4. Corporate action sentinel exclusion
# ──────────────────────────────────────────────────────────────────

def exclude_sentinel_corporate_actions(
    corporate_actions: pl.DataFrame,
    candidate_instrument_ids: frozenset[str],
    *,
    window_start: date | None = None,
    window_end: date | None = None,
) -> frozenset[str]:
    """Return instrument IDs excluded due to sentinel-only CA data.

    Instruments with only ``no_action`` rows (or absent from the CA table) are
    excluded because no verified price-adjustment data is available.
    """
    if corporate_actions.is_empty():
        return candidate_instrument_ids

    by_instrument: dict[str, list[tuple[date, date]]] = {}
    for row in corporate_actions.to_dicts():
        iid = str(row.get("instrument_id") or "")
        if not iid or iid == "KRX:__NO_ACTION__":
            continue
        try:
            start = row["effective_date"].astimezone(KRX_TZ).date()
            end = (row.get("coverage_end") or row["effective_date"]).astimezone(KRX_TZ).date()
        except (AttributeError, TypeError):
            continue
        if end >= start:
            by_instrument.setdefault(iid, []).append((start, end))

    excluded: set[str] = set()
    for iid in candidate_instrument_ids:
        ranges = sorted(by_instrument.get(iid, []))
        if not ranges:
            excluded.add(iid)
            continue
        if window_start is None or window_end is None:
            types = {
                str(row.get("type") or "")
                for row in corporate_actions.to_dicts()
                if str(row.get("instrument_id") or "") == iid
            }
            if types == {"no_action"}:
                excluded.add(iid)
            continue
        cursor = window_start
        for start, end in ranges:
            if end < cursor:
                continue
            if start > cursor:
                break
            cursor = max(cursor, end)
            if cursor >= window_end:
                break
        if cursor < window_end:
            excluded.add(iid)

    return frozenset(excluded)


# ──────────────────────────────────────────────────────────────────
# 5. Composite Gold audit
# ──────────────────────────────────────────────────────────────────

def build_gold_audit_manifest(
    *,
    calendar: SessionCalendar,
    security_master: pl.DataFrame,
    daily_market: pl.DataFrame,
    financial_facts: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    decision_time: datetime,
    validation_start: date,
    validation_end: date,
) -> GoldAuditManifest:
    """Run all Gold-layer pre-flight checks and produce a reproducible manifest.

    Warmup failure is recorded but does not raise; caller inspects manifest.warmup.warmup_ok.
    All instrument exclusions carry explicit reasons.
    """
    warmup = check_warmup_sessions(calendar, validation_start=validation_start)

    bar_audit = audit_bar_continuity(
        daily_market,
        calendar,
        window_start=validation_start,
        window_end=validation_end,
    )

    dart_elig = audit_dart_fact_eligibility(financial_facts, decision_time=decision_time)

    dart_eligible_companies: frozenset[str] = frozenset(
        d.company_id for d in dart_elig if d.eligible
    )

    # PIT-safe company_id → instrument_id mapping
    company_to_instruments: dict[str, set[str]] = {}
    if not security_master.is_empty():
        cols = [c for c in ("company_id", "instrument_id", "available_at") if c in security_master.columns]
        sm_subset = security_master.select(cols).unique() if len(cols) == 3 else security_master
        for row in sm_subset.to_dicts():
            av = row.get("available_at")
            try:
                if av is not None and av <= decision_time:
                    cid = str(row.get("company_id") or "")
                    iid = str(row.get("instrument_id") or "")
                    if cid and iid:
                        company_to_instruments.setdefault(cid, set()).add(iid)
            except TypeError:
                continue

    dart_eligible_instruments: frozenset[str] = frozenset(
        iid
        for cid in dart_eligible_companies
        for iid in company_to_instruments.get(cid, set())
    )

    bar_eligible_instruments: frozenset[str] = frozenset(
        r.instrument_id for r in bar_audit if r.eligible
    )

    candidate_ids = bar_eligible_instruments & dart_eligible_instruments
    ca_excluded = exclude_sentinel_corporate_actions(
        corporate_actions, candidate_ids, window_start=validation_start, window_end=validation_end
    )
    eligible = candidate_ids - ca_excluded

    hash_parts = [
        f"warmup:{warmup.warmup_ok}:{warmup.warmup_sessions_found}",
        f"validation:{validation_start.isoformat()}:{validation_end.isoformat()}",
        f"bar_eligible:{','.join(sorted(bar_eligible_instruments))}",
        f"dart_eligible:{','.join(sorted(dart_eligible_instruments))}",
        f"ca_excluded:{','.join(sorted(ca_excluded))}",
        f"eligible:{','.join(sorted(eligible))}",
    ]
    manifest_hash = hashlib.sha256("\n".join(hash_parts).encode("utf-8")).hexdigest()

    return GoldAuditManifest(
        warmup=warmup,
        bar_audit=bar_audit,
        dart_eligibility=dart_elig,
        ca_excluded_instrument_ids=ca_excluded,
        eligible_instrument_ids=eligible,
        manifest_hash=manifest_hash,
    )


def write_gold_audit_artifact(manifest: GoldAuditManifest, artifact_path: Path) -> Path:
    """Persist the Gold audit manifest as a reproducible JSON artifact."""
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "manifest_hash": manifest.manifest_hash,
        "warmup": {
            "ok": manifest.warmup.warmup_ok,
            "sessions_found": manifest.warmup.warmup_sessions_found,
            "sessions_required": manifest.warmup.warmup_sessions_required,
            "first_validation_session": (
                manifest.warmup.first_validation_session.isoformat()
                if manifest.warmup.first_validation_session
                else None
            ),
        },
        "bar_audit": {
            "eligible_count": sum(1 for r in manifest.bar_audit if r.eligible),
            "ineligible_count": sum(1 for r in manifest.bar_audit if not r.eligible),
            "by_reason": _count_reasons(manifest.bar_audit),
            "ineligible_instruments": [
                {
                    "instrument_id": r.instrument_id,
                    "reasons": [x.value for x in r.exclusion_reasons],
                    "sessions_found": r.sessions_found,
                    "sessions_expected": r.sessions_expected,
                }
                for r in manifest.bar_audit
                if not r.eligible
            ],
        },
        "dart_eligibility": {
            "eligible_count": sum(1 for d in manifest.dart_eligibility if d.eligible),
            "ineligible_count": sum(1 for d in manifest.dart_eligibility if not d.eligible),
            "by_reason": _count_reasons(manifest.dart_eligibility),
            "ineligible_companies": [
                {
                    "company_id": d.company_id,
                    "reasons": [x.value for x in d.exclusion_reasons],
                    "consecutive_quarters_found": d.consecutive_quarters_found,
                    "missing_facts": sorted(_REQUIRED_FACTS - d.required_facts_present),
                }
                for d in manifest.dart_eligibility
                if not d.eligible
            ],
        },
        "corporate_action_excluded": {
            "count": len(manifest.ca_excluded_instrument_ids),
            "instrument_ids": sorted(manifest.ca_excluded_instrument_ids),
            "reason": CorporateActionExclusionReason.SENTINEL_NO_ACTION.value,
        },
        "eligible_instruments": {
            "count": len(manifest.eligible_instrument_ids),
            "instrument_ids": sorted(manifest.eligible_instrument_ids),
        },
    }

    artifact_path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact_path

def _assert_no_availability_repair(
    *,
    table_name: str,
    frame: pl.DataFrame,
    final_decision_time: datetime,
) -> None:
    """Reject future availability instead of rewriting it (fail-closed PIT).

    Rows with available_at after the consuming decision remain unavailable;
    downstream repair is forbidden, so the frame is left unchanged on success
    and raises before any mutation on failure.
    """
    if final_decision_time.tzinfo is None:
        raise PITDataError(f"{table_name} final_decision_time must be timezone-aware")
    if frame.is_empty() or "available_at" not in frame.columns:
        return
    try:
        # Silver partitions may preserve their source timezone (e.g. KRX) while
        # the caller supplies an UTC decision instant; compare like-for-like.
        from src.data.replay import _literal_in_column_tz

        literal = _literal_in_column_tz(frame["available_at"].dtype, final_decision_time)
        future = frame.filter(pl.col("available_at") > literal)
    except Exception as exc:
        raise PITDataError(f"{table_name} available_at comparison failed") from exc
    if future.height > 0:
        raise PITDataError(
            f"{table_name} has {future.height} rows with future available_at after {final_decision_time.isoformat()}"
        )


@dataclass(frozen=True, slots=True)
class GoldRunReport:
    manifest: GoldAuditManifest
    universe_decisions_count: int
    eligible_decisions_count: int
    feature_rows_count: int
    universe_path: str | None
    features_path: str | None
    summary_artifact_path: str


def materialize_gold_window(
    *,
    calendar: SessionCalendar | None = None,
    security_master: pl.DataFrame | None = None,
    daily_market: pl.DataFrame | None = None,
    financial_facts: pl.DataFrame | None = None,
    corporate_actions: pl.DataFrame | None = None,
    investor_flow: pl.DataFrame | None = None,
    validation_start: date,
    validation_end: date,
    decision_time: datetime,
    artifact_root: Path,
    gold_root: Path | None = None,
    universe_policy: Any | None = None,
    qvef_policy: Any | None = None,
    silver_root: Path | None = None,
) -> GoldRunReport:
    """Run Gold-layer audit, generate daily historical universe decisions, and build QVEF features.

    Implements tasks 1-4 of docs/next.md:
      1. Historical universe U_t for every validation session with explicit exclusion reasons
      2. 60-trading-day warmup and bar continuity pre-flight audit
      3. DART fact 4-quarter eligibility and corporate-action sentinel exclusion
      4. PIT-safe lagged feature matrix with provenance
    """
    from datetime import time as dt_time

    from src.data.replay import PITReplayReader, StreamingGoldWriter
    from src.features.contracts import QvefFeaturePolicy, QvefFeatureRow
    from src.features.materialize import materialize_qvef_features
    from src.features.qvef import build_qvef_features
    from src.strategy.universe import (
        UniverseDecision,
        UniversePolicy,
        build_historical_universe,
        materialize_historical_universe,
    )

    u_policy = universe_policy if universe_policy is not None else UniversePolicy()
    f_policy = qvef_policy if qvef_policy is not None else QvefFeaturePolicy()
    reader: Any | None = None
    stream_writer: StreamingGoldWriter | None = None
    has_complete_frames = (
        calendar is not None
        and security_master is not None
        and daily_market is not None
        and financial_facts is not None
        and corporate_actions is not None
    )
    if silver_root is not None and not has_complete_frames:
        from src.data.gold_loader import load_gold_window_inputs as _load_inputs

        _inputs = _load_inputs(
            silver_root=Path(silver_root),
            validation_start=validation_start,
            validation_end=validation_end,
            decision_time=decision_time,
            universe_policy=u_policy if isinstance(u_policy, UniversePolicy) else None,
            qvef_policy=f_policy if isinstance(f_policy, QvefFeaturePolicy) else None,
        )
        calendar = _inputs.calendar
        security_master = _inputs.security_master
        daily_market = _inputs.daily_market
        financial_facts = _inputs.financial_facts
        corporate_actions = _inputs.corporate_actions
        investor_flow = _inputs.investor_flow
        assert calendar is not None
    if (
        calendar is None
        or security_master is None
        or daily_market is None
        or financial_facts is None
        or corporate_actions is None
    ):
        raise PITDataError("materialize_gold_window requires calendar and Silver frames")
    flow_df = investor_flow if investor_flow is not None else pl.DataFrame()

    # Align batch-ingestion available_at timestamps if present
    # PIT fail-closed: future availability is rejected, never rewritten.
    _assert_no_availability_repair(table_name="daily_market", frame=daily_market, final_decision_time=decision_time)
    _assert_no_availability_repair(table_name="security_master", frame=security_master, final_decision_time=decision_time)
    _assert_no_availability_repair(table_name="financial_facts", frame=financial_facts, final_decision_time=decision_time)
    _assert_no_availability_repair(table_name="corporate_actions", frame=corporate_actions, final_decision_time=decision_time)
    _assert_no_availability_repair(table_name="investor_flow", frame=flow_df, final_decision_time=decision_time)

    if not security_master.is_empty() and "valid_from" in security_master.columns and "listing_date" in security_master.columns:
        earliest_vf = security_master.group_by("instrument_id").agg(pl.col("valid_from").min().alias("_min_vf"))
        security_master = security_master.join(earliest_vf, on="instrument_id").with_columns(
            pl.when(pl.col("listing_date") == pl.col("valid_from"))
            .then(pl.col("_min_vf"))
            .otherwise(pl.col("listing_date"))
            .alias("listing_date")
        ).drop("_min_vf")

    if has_complete_frames or silver_root is not None:
        assert calendar is not None
        assert security_master is not None
        assert daily_market is not None
        assert financial_facts is not None
        assert corporate_actions is not None
        flow_for_reader = investor_flow if investor_flow is not None else pl.DataFrame()
        reader = PITReplayReader.from_frames(calendar=calendar, security_master=security_master, daily_market=daily_market, investor_flow=flow_for_reader, financial_facts=financial_facts, corporate_actions=corporate_actions)

    # 1-3. Pre-flight audit manifest
    manifest = build_gold_audit_manifest(
        calendar=calendar,
        security_master=security_master,
        daily_market=daily_market,
        financial_facts=financial_facts,
        corporate_actions=corporate_actions,
        decision_time=decision_time,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    audit_path = Path(artifact_root) / "gold_audit" / f"{manifest.manifest_hash[:16]}.json"
    write_gold_audit_artifact(manifest, audit_path)

    # Filter sessions in validation window
    val_sessions = [
        s for s in calendar.sessions
        if validation_start <= s.astimezone(KRX_TZ).date() <= validation_end
    ]

    if reader is not None and val_sessions and "available_at" in daily_market.columns:
        # Avoid replaying hundreds of sessions when the source has no PIT-valid
        # market bars at the first decision (typically a retrieval-time leak).
        from src.data.replay import _literal_in_column_tz

        first_decision = datetime.combine(
            val_sessions[0].astimezone(KRX_TZ).date(), dt_time(15, 30), tzinfo=KRX_TZ
        )
        available_min = daily_market["available_at"].min()
        if isinstance(available_min, datetime):
            literal = _literal_in_column_tz(daily_market["available_at"].dtype, first_decision)
            if available_min > literal:
                raise PITDataError(
                    "daily_market has no PIT-available rows at validation start"
                )

    if reader is not None and gold_root is not None:
        stream_writer = StreamingGoldWriter(
            root=Path(gold_root),
            dataset_id=manifest.manifest_hash[:32],
            decision_time=decision_time,
            certification=DatasetCertification.RESEARCH,
            source_hashes={
                "calendar": manifest.manifest_hash[:16],
                "security_master": manifest.manifest_hash[16:32],
                "quality_report": manifest.manifest_hash,
            },
            expected_sessions=tuple(s for s in val_sessions if s <= decision_time),
            require_scores=False,
        )

    all_universe: list[UniverseDecision] = []
    all_features: list[QvefFeatureRow] = []
    universe_count = 0
    eligible_count = 0
    feature_count = 0
    completed_sessions = 0
    import time as _time

    _replay_start = _time.monotonic()
    _LOGGER.info(
        "[DATA] stage=gold_replay_start daily_rows=%d master_rows=%d sessions=%d",
        daily_market.height,
        security_master.height,
        len(val_sessions),
    )

    for session in val_sessions:
        # Market close of session for end-of-day daily decisions
        sess_dt = datetime.combine(
            session.astimezone(KRX_TZ).date(), dt_time(15, 30), tzinfo=KRX_TZ
        )
        if sess_dt > decision_time:
            continue

        replay = None
        if reader is not None:
            # PITReplayReader.session_input via frame-route reader.
            replay = reader.session_input(
                session=session,
                decision_time=sess_dt,
                universe_policy=u_policy,
                qvef_policy=f_policy,
            )
            u_decisions = build_historical_universe(
                decision_session=session,
                decision_time=sess_dt,
                calendar=calendar,
                security_master=replay.security_master,
                daily_market=replay.daily_market,
                corporate_actions=replay.corporate_actions,
                policy=u_policy,
            )
        else:
            u_decisions = build_historical_universe(
                decision_session=session,
                decision_time=sess_dt,
                calendar=calendar,
                security_master=security_master,
                daily_market=daily_market,
                corporate_actions=corporate_actions,
                policy=u_policy,
            )
        if stream_writer is not None:
            stream_writer.append_universe(u_decisions)
        else:
            all_universe.extend(u_decisions)
        universe_count += len(u_decisions)
        eligible_count += sum(1 for item in u_decisions if item.eligible)

        eligible = tuple(u for u in u_decisions if u.eligible)
        if eligible:
            if reader is not None:
                assert replay is not None
                f_rows = build_qvef_features(
                    decision_session=session,
                    decision_time=sess_dt,
                    calendar=calendar,
                    universe=eligible,
                    security_master=replay.security_master,
                    daily_market=replay.daily_market,
                    investor_flow=replay.investor_flow,
                    financial_facts=replay.financial_facts,
                    policy=f_policy,
                )
            else:
                f_rows = build_qvef_features(
                    decision_session=session,
                    decision_time=sess_dt,
                    calendar=calendar,
                    universe=eligible,
                    security_master=security_master,
                    daily_market=daily_market,
                    investor_flow=flow_df,
                    financial_facts=financial_facts,
                    policy=f_policy,
                )
            if stream_writer is not None:
                stream_writer.append_features(f_rows)
            else:
                all_features.extend(f_rows)
            feature_count += len(f_rows)
        completed_sessions += 1
        completed = completed_sessions
        if completed % 500 == 0:
            _LOGGER.info(
                "[SYS] stage=gold_replay_progress completed=%d elapsed_ms=%d",
                completed,
                int((_time.monotonic() - _replay_start) * 1000),
            )

    _LOGGER.info(
        "[SYS] stage=gold_replay_done completed=%d elapsed_ms=%d",
        completed_sessions,
        int((_time.monotonic() - _replay_start) * 1000),
    )
    u_path_str: str | None = None
    f_path_str: str | None = None

    if stream_writer is not None:
        published = stream_writer.close()
        u_path_str = str(published["universe"])
        f_path_str = str(published["qvef"]) if "qvef" in published else None
    elif gold_root is not None and all_universe:
        from src.storage.parquet_datasets import ParquetDatasetStore

        dataset_id = manifest.manifest_hash[:32]
        universe_root = Path(gold_root) / "universe"
        existing_universe = universe_root / dataset_id
        if existing_universe.exists():
            stored = ParquetDatasetStore(universe_root).read_manifest(dataset_id)
            if stored.quality_report_hash != manifest.manifest_hash:
                raise PITDataError("existing Gold universe has a different audit manifest")
            u_path = existing_universe
        else:
            u_path = materialize_historical_universe(
                tuple(all_universe), root=universe_root, dataset_id=dataset_id,
                decision_time=decision_time, policy=u_policy, provider_version="official-pit-v1",
                calendar_hash=manifest.manifest_hash[:16], master_hash=manifest.manifest_hash[16:32],
                quality_report_hash=manifest.manifest_hash, certification=DatasetCertification.RESEARCH,
            )
        u_path_str = str(u_path)

        if all_features:
            feature_root = Path(gold_root) / "qvef"
            existing_features = feature_root / dataset_id
            if existing_features.exists():
                stored = ParquetDatasetStore(feature_root).read_manifest(dataset_id)
                if stored.quality_report_hash != manifest.manifest_hash:
                    raise PITDataError("existing Gold features have a different audit manifest")
                f_path = existing_features
            else:
                f_path = materialize_qvef_features(
                    tuple(all_features), root=feature_root, dataset_id=dataset_id,
                    decision_time=decision_time, policy=f_policy, provider_version="official-pit-v1",
                    calendar_hash=manifest.manifest_hash[:16], master_hash=manifest.manifest_hash[16:32],
                    quality_report_hash=manifest.manifest_hash, certification=DatasetCertification.RESEARCH,
                )
            f_path_str = str(f_path)

    # Write summary artifact
    summary_path = Path(artifact_root) / "gold_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "manifest_hash": manifest.manifest_hash,
        "validation_start": validation_start.isoformat(),
        "validation_end": validation_end.isoformat(),
        "sessions_evaluated": len(val_sessions),
        "total_universe_decisions": universe_count,
        "eligible_universe_decisions": eligible_count,
        "total_feature_rows": feature_count,
        "universe_path": u_path_str,
        "features_path": f_path_str,
        "audit_artifact_path": str(audit_path),
    }
    summary_path.write_text(
        json.dumps(summary_payload, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return GoldRunReport(
        manifest=manifest,
        universe_decisions_count=universe_count,
        eligible_decisions_count=eligible_count,
        feature_rows_count=feature_count,
        universe_path=u_path_str,
        features_path=f_path_str,
        summary_artifact_path=str(summary_path),
    )


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


def _to_date(v: Any) -> date:
    if isinstance(v, datetime):
        return v.astimezone(KRX_TZ).date()
    if isinstance(v, date):
        return v
    raise TypeError(f"cannot convert {type(v)} to date")


def _count_reasons(results: tuple[Any, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        for reason in r.exclusion_reasons:
            counts[reason.value] = counts.get(reason.value, 0) + 1
    return counts
