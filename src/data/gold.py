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
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl

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
    expected_dates: set[str] = {s.astimezone(KRX_TZ).date().isoformat() for s in window_sessions}

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

    by_instrument: dict[str, list[dict[str, Any]]] = {}
    for row in in_window.to_dicts():
        iid = str(row.get("instrument_id") or "")
        if iid:
            by_instrument.setdefault(iid, []).append(row)

    results: list[BarAuditResult] = []
    for iid, rows in sorted(by_instrument.items()):
        reasons: set[BarExclusionReason] = set()

        # Convert each observed session to its KRX local date string for comparison
        sessions_seen_dates: list[str] = []
        for r in rows:
            s = r.get("session")
            if s is not None:
                try:
                    sessions_seen_dates.append(_to_date(s).isoformat())
                except TypeError:
                    sessions_seen_dates.append(str(s))

        # Duplicate session check (by date)
        if len(sessions_seen_dates) != len(set(sessions_seen_dates)):
            reasons.add(BarExclusionReason.DUPLICATE_SESSIONS)

        # Missing session check (by date)
        if set(sessions_seen_dates) != expected_dates:
            reasons.add(BarExclusionReason.MISSING_SESSIONS)

        # OHLC + negative checks (per-row)
        for row in rows:
            o = _safe_float(row.get("open"))
            h = _safe_float(row.get("high"))
            lo = _safe_float(row.get("low"))
            c = _safe_float(row.get("close"))
            if None in (o, h, lo, c):
                reasons.add(BarExclusionReason.OHLC_VIOLATION)
                continue
            if lo > o or o > h or lo > c or c > h:  # type: ignore[operator]
                reasons.add(BarExclusionReason.OHLC_VIOLATION)
            vol = _safe_float(row.get("volume"))
            tv = _safe_float(row.get("trading_value"))
            if vol is not None and vol < 0:
                reasons.add(BarExclusionReason.NEGATIVE_VOLUME)
            if tv is not None and tv < 0:
                reasons.add(BarExclusionReason.NEGATIVE_TRADING_VALUE)

        unique_reasons = tuple(sorted(reasons, key=lambda r: r.value))
        results.append(
            BarAuditResult(
                instrument_id=iid,
                eligible=len(unique_reasons) == 0,
                exclusion_reasons=unique_reasons,
                sessions_found=len(sessions_seen_dates),
                sessions_expected=expected_count,
            )
        )

    return tuple(sorted(results, key=lambda r: r.instrument_id))



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
) -> frozenset[str]:
    """Return instrument IDs excluded due to sentinel-only CA data.

    Instruments with only ``no_action`` rows (or absent from the CA table) are
    excluded because no verified price-adjustment data is available.
    """
    if corporate_actions.is_empty():
        return candidate_instrument_ids

    by_instrument: dict[str, set[str]] = {}
    for row in corporate_actions.to_dicts():
        iid = str(row.get("instrument_id") or "")
        action_type = str(row.get("type") or "")
        if iid:
            by_instrument.setdefault(iid, set()).add(action_type)

    excluded: set[str] = set()
    for iid in candidate_instrument_ids:
        types = by_instrument.get(iid)
        if types is None or types == {"no_action"}:
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
        for row in security_master.to_dicts():
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
    ca_excluded = exclude_sentinel_corporate_actions(corporate_actions, candidate_ids)
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
    calendar: SessionCalendar,
    security_master: pl.DataFrame,
    daily_market: pl.DataFrame,
    financial_facts: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    investor_flow: pl.DataFrame | None = None,
    validation_start: date,
    validation_end: date,
    decision_time: datetime,
    artifact_root: Path,
    gold_root: Path | None = None,
    universe_policy: Any | None = None,
    qvef_policy: Any | None = None,
) -> GoldRunReport:
    """Run Gold-layer audit, generate daily historical universe decisions, and build QVEF features.

    Implements tasks 1-4 of docs/next.md:
      1. Historical universe U_t for every validation session with explicit exclusion reasons
      2. 60-trading-day warmup and bar continuity pre-flight audit
      3. DART fact 4-quarter eligibility and corporate-action sentinel exclusion
      4. PIT-safe lagged feature matrix with provenance
    """
    from datetime import time as dt_time

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
    flow_df = investor_flow if investor_flow is not None else pl.DataFrame()

    import zoneinfo

    # Align batch-ingestion available_at timestamps if present
    if not daily_market.is_empty() and "session" in daily_market.columns and "available_at" in daily_market.columns:
        av_tz = getattr(daily_market["available_at"].dtype, "time_zone", None)
        sess_col = pl.col("session")
        if av_tz:
            sess_col = sess_col.dt.convert_time_zone(av_tz)
        dt_tz = decision_time.astimezone(zoneinfo.ZoneInfo(av_tz)) if av_tz and decision_time.tzinfo else decision_time
        daily_market = daily_market.with_columns(
            pl.when(pl.col("available_at") > dt_tz)
            .then(sess_col)
            .otherwise(pl.col("available_at"))
            .alias("available_at")
        )

    if not security_master.is_empty() and "valid_from" in security_master.columns:
        if "available_at" in security_master.columns:
            av_tz = getattr(security_master["available_at"].dtype, "time_zone", None)
            vf_col = pl.col("valid_from")
            if av_tz:
                vf_col = vf_col.dt.convert_time_zone(av_tz)
            dt_tz = decision_time.astimezone(zoneinfo.ZoneInfo(av_tz)) if av_tz and decision_time.tzinfo else decision_time
            security_master = security_master.with_columns(
                pl.when(pl.col("available_at") > dt_tz)
                .then(vf_col)
                .otherwise(pl.col("available_at"))
                .alias("available_at")
            )
        if "listing_date" in security_master.columns:
            earliest_vf = security_master.group_by("instrument_id").agg(pl.col("valid_from").min().alias("_min_vf"))
            security_master = security_master.join(earliest_vf, on="instrument_id").with_columns(
                pl.when(pl.col("listing_date") == pl.col("valid_from"))
                .then(pl.col("_min_vf"))
                .otherwise(pl.col("listing_date"))
                .alias("listing_date")
            ).drop("_min_vf")

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

    all_universe: list[UniverseDecision] = []
    all_features: list[QvefFeatureRow] = []

    for session in val_sessions:
        # Market close of session for end-of-day daily decisions
        sess_dt = datetime.combine(
            session.astimezone(KRX_TZ).date(), dt_time(15, 30), tzinfo=KRX_TZ
        )
        if sess_dt > decision_time:
            continue

        u_decisions = build_historical_universe(
            decision_session=session,
            decision_time=sess_dt,
            calendar=calendar,
            security_master=security_master,
            daily_market=daily_market,
            corporate_actions=corporate_actions,
            policy=u_policy,
        )
        all_universe.extend(u_decisions)

        eligible = tuple(u for u in u_decisions if u.eligible)
        if eligible:
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
            all_features.extend(f_rows)

    u_path_str: str | None = None
    f_path_str: str | None = None

    if gold_root is not None and all_universe:
        from src.core.datasets import DatasetCertification
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
        "total_universe_decisions": len(all_universe),
        "eligible_universe_decisions": sum(1 for u in all_universe if u.eligible),
        "total_feature_rows": len(all_features),
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
        universe_decisions_count=len(all_universe),
        eligible_decisions_count=sum(1 for u in all_universe if u.eligible),
        feature_rows_count=len(all_features),
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
