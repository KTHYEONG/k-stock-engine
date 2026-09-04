"""Historical eligible universe — PIT-bounded, deterministic."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl

from src.core.datasets import (
    HIVE_PARTITION_LAYOUT,
    DatasetCertification,
    make_manifest,
)
from src.core.instruments import AssetKind
from src.core.time import SessionCalendar
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash


@dataclass(frozen=True, slots=True)
class UniversePolicy:
    version: str = "champion-v1-universe"
    minimum_listing_sessions: int = 252
    liquidity_window_sessions: int = 60
    minimum_median_trading_value_krw: float = 2_000_000_000.0


class ExclusionReason(StrEnum):
    AMBIGUOUS_MASTER = "ambiguous_master"
    DELISTED = "delisted"
    FINANCIAL_SECTOR = "financial_sector"
    INELIGIBLE_STATUS = "ineligible_status"
    INSUFFICIENT_LIQUIDITY_HISTORY = "insufficient_liquidity_history"
    INSUFFICIENT_LISTING_AGE = "insufficient_listing_age"
    INVALID_MARKET = "invalid_market"
    INVALID_MASTER = "invalid_master"
    LIQUIDITY_BELOW_THRESHOLD = "liquidity_below_threshold"
    MISSING_MASTER = "missing_master"
    MISSING_SECTOR = "missing_sector"
    NON_COMMON_SHARE_CLASS = "non_common_share_class"
    NOT_LISTED = "not_listed"


@dataclass(frozen=True, slots=True)
class UniverseDecision:
    decision_session: datetime
    instrument_id: str
    eligible: bool
    exclusion_reasons: tuple[ExclusionReason, ...]
    listing_age_sessions: int | None
    median_trading_value_60: float | None


def _validate_build_inputs(
    *,
    decision_session: datetime,
    decision_time: datetime,
    calendar: SessionCalendar,
    policy: UniversePolicy,
) -> None:
    if decision_session.tzinfo is None or decision_time.tzinfo is None:
        raise ValueError("decision_session and decision_time must be timezone-aware")
    if decision_session > decision_time:
        raise ValueError("decision_session must not be after decision_time")
    if decision_session not in calendar.sessions:
        raise ValueError("calendar does not contain decision_session")
    if policy.minimum_listing_sessions <= 0:
        raise ValueError("minimum_listing_sessions must be positive")
    if policy.liquidity_window_sessions <= 0:
        raise ValueError("liquidity_window_sessions must be positive")
    if policy.minimum_median_trading_value_krw <= 0:
        raise ValueError("minimum_median_trading_value_krw must be positive")
    if not policy.version or not policy.version.strip():
        raise ValueError("policy version must be non-empty")


def _pit_filter(frame: pl.DataFrame, decision_time: datetime) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "available_at" not in frame.columns:
        # Availability is required evidence; without it no row is PIT-safe.
        return frame.clear()
    # Filter rows where available_at <= decision_time
    try:
        return frame.filter(pl.col("available_at") <= decision_time)
    except Exception:
        # Fallback python filter if dtype mismatch
        rows = frame.to_dicts()
        kept = [r for r in rows if r.get("available_at") is not None and r["available_at"] <= decision_time]
        if not kept:
            return frame.clear()
        return pl.DataFrame(kept)


def build_historical_universe(
    *,
    decision_session: datetime,
    decision_time: datetime,
    calendar: SessionCalendar,
    security_master: pl.DataFrame,
    daily_market: pl.DataFrame,
    policy: UniversePolicy = UniversePolicy(),  # noqa: B008
) -> tuple[UniverseDecision, ...]:
    _validate_build_inputs(
        decision_session=decision_session,
        decision_time=decision_time,
        calendar=calendar,
        policy=policy,
    )

    master_filtered = _pit_filter(security_master, decision_time)
    daily_filtered = _pit_filter(daily_market, decision_time)

    # Group master rows by instrument_id
    master_by_id: dict[str, list[dict[str, Any]]] = {}
    if not master_filtered.is_empty():
        for row in master_filtered.to_dicts():
            iid = row.get("instrument_id")
            if iid is None:
                continue
            iid_s = str(iid)
            master_by_id.setdefault(iid_s, []).append(row)
        # Sort each group's rows by valid_from for deterministic resolution (O(M log M))
        for iid in master_by_id:
            try:  # noqa: SIM105
                master_by_id[iid].sort(key=lambda r: (r.get("valid_from") is None, str(r.get("valid_from"))))
            except TypeError:
                pass

    # Group daily rows by instrument_id (O(D log D))
    daily_by_instrument: dict[str, list[dict[str, Any]]] = {}
    if not daily_filtered.is_empty():
        for row in daily_filtered.to_dicts():
            iid = row.get("instrument_id")
            if iid is None:
                continue
            iid_s = str(iid)
            daily_by_instrument.setdefault(iid_s, []).append(row)
        for iid in daily_by_instrument:
            try:  # noqa: SIM105
                daily_by_instrument[iid].sort(key=lambda r: str(r.get("session")))
            except TypeError:
                pass

    instrument_ids = set(master_by_id.keys()) | set(daily_by_instrument.keys())
    # Ensure deterministic ordering
    sorted_ids = sorted(instrument_ids)

    # Precompute calendar index for decision_session
    hi_idx = calendar.sessions.index(decision_session)
    window = policy.liquidity_window_sessions

    decisions: list[UniverseDecision] = []
    seen: set[str] = set()

    for instrument_id in sorted_ids:
        if instrument_id in seen:
            raise ValueError(f"duplicate instrument identity {instrument_id!r}")
        seen.add(instrument_id)

        reasons: list[ExclusionReason] = []
        listing_age: int | None = None
        median_val: float | None = None

        master_rows = master_by_id.get(instrument_id, [])
        # Resolve master as of decision_session
        active: list[dict[str, Any]] = []
        for r in master_rows:
            vf = r.get("valid_from")
            vt = r.get("valid_to")
            if vf is None:
                continue
            try:
                if vf <= decision_session and (vt is None or decision_session <= vt):
                    active.append(r)
            except TypeError:
                continue

        master_row: dict[str, Any] | None = None
        if not active:
            # No active master for instrument: if instrument only seen in daily_market, it's missing master
            # If seen in master but none active, also missing
            reasons.append(ExclusionReason.MISSING_MASTER)
        else:
            # Find latest valid_from
            try:
                max_vf = max(r["valid_from"] for r in active)
            except Exception:
                reasons.append(ExclusionReason.INVALID_MASTER)
                max_vf = None
            if max_vf is not None:
                candidates = [r for r in active if r.get("valid_from") == max_vf]
                if len(candidates) > 1:
                    reasons.append(ExclusionReason.AMBIGUOUS_MASTER)
                elif len(candidates) == 1:
                    master_row = candidates[0]
                else:
                    reasons.append(ExclusionReason.MISSING_MASTER)

        if master_row is not None:
            market = master_row.get("market")
            share_class = master_row.get("share_class")
            sector = master_row.get("sector")
            status = master_row.get("status")
            listing_date = master_row.get("listing_date")
            delisting_date = master_row.get("delisting_date")

            # Market
            if market not in ("KOSPI", "KOSDAQ"):
                reasons.append(ExclusionReason.INVALID_MARKET)
            # Share class
            if share_class != "common":
                reasons.append(ExclusionReason.NON_COMMON_SHARE_CLASS)
            # Sector
            if sector == "Financials":
                reasons.append(ExclusionReason.FINANCIAL_SECTOR)
            # Sector is optional for universe eligibility; QVEF handles an
            # absent industry classification with a global cross-section.
            # Status
            if status != "listed":
                reasons.append(ExclusionReason.INELIGIBLE_STATUS)
            # Listing date
            if listing_date is None or (isinstance(listing_date, float) and math.isnan(listing_date)):
                reasons.append(ExclusionReason.INVALID_MASTER)
                listing_age = None
            else:
                try:
                    if listing_date > decision_session:
                        reasons.append(ExclusionReason.NOT_LISTED)
                        listing_age = 0
                        if listing_age < policy.minimum_listing_sessions:  # noqa: SIM102
                            if ExclusionReason.INSUFFICIENT_LISTING_AGE not in reasons:
                                reasons.append(ExclusionReason.INSUFFICIENT_LISTING_AGE)
                    else:
                        lo = None
                        for idx, s in enumerate(calendar.sessions):
                            try:
                                if s >= listing_date:
                                    lo = idx
                                    break
                            except TypeError:
                                continue
                        if lo is None or lo > hi_idx:  # noqa: SIM108
                            listing_age = 0
                        else:
                            listing_age = hi_idx - lo + 1
                        if listing_age < policy.minimum_listing_sessions:
                            reasons.append(ExclusionReason.INSUFFICIENT_LISTING_AGE)
                except TypeError:
                    reasons.append(ExclusionReason.INVALID_MASTER)
                    listing_age = None

            # Delisting
            if delisting_date is not None and not (isinstance(delisting_date, float) and math.isnan(delisting_date)):
                try:
                    if delisting_date <= decision_session:  # noqa: SIM102
                        if ExclusionReason.DELISTED not in reasons:
                            reasons.append(ExclusionReason.DELISTED)
                except TypeError:
                    pass
        else:
            # No valid master: listing_age stays None
            listing_age = None

        # Liquidity: trailing 60 sessions ending at decision_session inclusive
        if hi_idx - window + 1 < 0:
            reasons.append(ExclusionReason.INSUFFICIENT_LIQUIDITY_HISTORY)
            median_val = None
        else:
            expected_sessions = calendar.sessions[hi_idx - window + 1 : hi_idx + 1]
            # Build session -> values for this instrument
            daily_rows = daily_by_instrument.get(instrument_id, [])
            # Map expected session to list
            sess_to_vals: dict[datetime, list[float]] = {s: [] for s in expected_sessions}
            invalid_liquidity = False
            for r in daily_rows:
                sess = r.get("session")
                if sess not in sess_to_vals:
                    continue
                tv = r.get("trading_value")
                try:
                    if tv is None or (isinstance(tv, float) and (math.isnan(tv) or math.isinf(tv))):
                        invalid_liquidity = True
                        break
                    tv_f = float(tv)
                    if not math.isfinite(tv_f) or tv_f < 0:
                        invalid_liquidity = True
                        break
                    sess_to_vals[sess].append(tv_f)
                except Exception:
                    invalid_liquidity = True
                    break

            values: list[float] = []
            if not invalid_liquidity:
                for sess in expected_sessions:
                    lst = sess_to_vals.get(sess, [])
                    if len(lst) != 1:
                        invalid_liquidity = True
                        break
                    values.append(lst[0])

            if invalid_liquidity or len(values) != window:
                if ExclusionReason.INSUFFICIENT_LIQUIDITY_HISTORY not in reasons:
                    reasons.append(ExclusionReason.INSUFFICIENT_LIQUIDITY_HISTORY)
                median_val = None
            else:
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                if n % 2 == 1:
                    median_val = float(sorted_vals[n // 2])
                else:
                    median_val = (float(sorted_vals[n // 2 - 1]) + float(sorted_vals[n // 2])) / 2.0
                if median_val < policy.minimum_median_trading_value_krw:
                    reasons.append(ExclusionReason.LIQUIDITY_BELOW_THRESHOLD)

        # Deduplicate and sort reasons for stability
        unique_reasons = sorted(set(reasons), key=lambda r: r.value)
        eligible = len(unique_reasons) == 0
        if eligible:
            exclusion_tuple: tuple[ExclusionReason, ...] = ()
        else:
            exclusion_tuple = tuple(unique_reasons)

        decisions.append(
            UniverseDecision(
                decision_session=decision_session,
                instrument_id=instrument_id,
                eligible=eligible,
                exclusion_reasons=exclusion_tuple,
                listing_age_sessions=listing_age,
                median_trading_value_60=median_val,
            )
        )

    # Ensure sorted by instrument_id and no duplicates (already)
    decisions_sorted = tuple(sorted(decisions, key=lambda d: d.instrument_id))
    # Final duplicate check
    ids_out = [d.instrument_id for d in decisions_sorted]
    if len(ids_out) != len(set(ids_out)):
        raise ValueError("duplicate output instrument identities")
    return decisions_sorted


def materialize_historical_universe(
    decisions: tuple[UniverseDecision, ...],
    *,
    root: Path,
    dataset_id: str,
    decision_time: datetime,
    policy: UniversePolicy,
    provider_version: str,
    calendar_hash: str,
    master_hash: str,
    quality_report_hash: str,
    certification: DatasetCertification,
) -> Path:
    if not decisions:
        raise ValueError("decisions must be non-empty")
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not policy.version or not policy.version.strip():
        raise ValueError("policy version must be non-empty")
    if not provider_version or not provider_version.strip():
        raise ValueError("provider_version must be non-empty")
    if not calendar_hash or not calendar_hash.strip():
        raise ValueError("calendar_hash must be non-empty")
    if not master_hash or not master_hash.strip():
        raise ValueError("master_hash must be non-empty")
    if not quality_report_hash or not quality_report_hash.strip():
        raise ValueError("quality_report_hash must be non-empty")
    if certification not in (DatasetCertification.RESEARCH, DatasetCertification.PRODUCTION):
        raise ValueError("certification must be RESEARCH or PRODUCTION")
    if not dataset_id or not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")

    root = Path(root)
    dataset_dir = root / dataset_id
    if dataset_dir.exists():
        raise ValueError(f"dataset already exists: {dataset_id}")

    # Validate uniqueness of (decision_session, instrument_id)
    seen_keys: set[tuple[datetime, str]] = set()
    for d in decisions:
        key = (d.decision_session, d.instrument_id)
        if key in seen_keys:
            raise ValueError(f"duplicate key {key!r}")
        seen_keys.add(key)
        if d.eligible and d.exclusion_reasons:
            raise ValueError(f"eligible decision must have no exclusion reasons: {d.instrument_id}")
        if not d.eligible and not d.exclusion_reasons:
            raise ValueError(f"ineligible decision must have exclusion reasons: {d.instrument_id}")

    # Build frame
    rows: list[dict[str, Any]] = []
    for d in decisions:
        # exclusion_reasons stored as comma-separated string for Parquet
        excl_str = ",".join(r.value for r in d.exclusion_reasons)
        rows.append(
            {
                "decision_session": d.decision_session,
                "instrument_id": d.instrument_id,
                "eligible": bool(d.eligible),
                "exclusion_reasons": excl_str,
                "listing_age_sessions": d.listing_age_sessions,
                "median_trading_value_60": d.median_trading_value_60,
                "policy_version": policy.version,
                "generated_at": decision_time,
            }
        )

    frame = pl.DataFrame(rows)
    # Ensure types: decision_session datetime, generated_at datetime
    ordered_columns = [
        "decision_session",
        "instrument_id",
        "eligible",
        "exclusion_reasons",
        "listing_age_sessions",
        "median_trading_value_60",
        "policy_version",
        "generated_at",
    ]
    frame = frame.select(ordered_columns)

    # Time bounds for manifest
    sessions = [d.decision_session for d in decisions]
    time_start = min(sessions)
    time_end = max(sessions)

    content_hash = canonical_content_hash(frame, ordered_columns)

    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set="stock_historical_eligible_universe_v1",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=time_start,
        time_end=time_end,
        provider_version=provider_version,
        universe_policy_version=policy.version,
        row_count=frame.height,
        generated_time=decision_time,
        certification=certification,
        calendar_hash=calendar_hash,
        master_hash=master_hash,
        quality_report_hash=quality_report_hash,
        schema_version="v2",
        content_hash=content_hash,
        storage_layout=HIVE_PARTITION_LAYOUT,
    )

    store = ParquetDatasetStore(root)
    path = store.write_partitioned(
        frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set="stock_historical_eligible_universe_v1",
        decision_time=decision_time,
    )
    return path
