"""PIT backtest session builder from certified Silver snapshots."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Any

import polars as pl

from src.core.instruments import AssetKind, Instrument
from src.core.ledger import LedgerActionType, LedgerCorporateAction
from src.core.time import SessionCalendar
from src.data.schemas import PITDataError, SilverTable
from src.data.snapshot import PITSnapshotRepository
from src.engine.backtest import BacktestSession
from src.engine.fill_model import HistoricalBar


@dataclass(frozen=True, slots=True)
class BacktestMarketInputsPolicy:
    version: str = "korean-equity-market-inputs-v1"
    adtv_sessions: int = 20
    volatility_sessions: int = 60
    market_volatility_sessions: int = 60
    annualization_sessions: int = 252
    unexplained_price_jump_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (
            self.version != "korean-equity-market-inputs-v1"
            or self.adtv_sessions != 20
            or self.volatility_sessions != 60
            or self.market_volatility_sessions != 60
            or self.annualization_sessions != 252
            or self.unexplained_price_jump_threshold != 0.5
        ):
            raise ValueError("BacktestMarketInputsPolicy constants are immutable")


_ALLOWED_ACTION_TYPES = frozenset({"split", "reverse_split", "dividend", "bonus_issue"})

_REL_TOL = 1e-6
_ABS_TOL = 1e-6


def _close_enough(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(_ABS_TOL, _REL_TOL * max(abs(actual), abs(expected)))


@dataclass(frozen=True, slots=True)
class CorporateActionCoverage:
    actions_by_session: Mapping[datetime, tuple[LedgerCorporateAction, ...]]
    research_returns_by_key: Mapping[tuple[datetime, str], float]


@dataclass(frozen=True, slots=True)
class CorporateActionEvidenceResolution:
    eligible_daily_market: pl.DataFrame
    verified_corporate_actions: pl.DataFrame
    excluded_instruments: frozenset[str]
    exclusion_reasons: Mapping[str, tuple[str, ...]]


def find_unexplained_price_discontinuities(
    *, daily_market: pl.DataFrame, verified_actions: pl.DataFrame, threshold: float
) -> pl.DataFrame:
    """Collect compact jump sessions lacking verified cover with a lazy scan."""
    windowed = (
        daily_market.lazy()
        .select(
            pl.col("session"),
            pl.col("instrument_id").cast(pl.String),
            pl.col("close").cast(pl.Float64),
        )
        .sort(["instrument_id", "session"])
    )
    invalid = (
        windowed.filter(~pl.col("close").is_finite() | (pl.col("close") <= 0))
        .select("instrument_id")
        .limit(1)
        .collect()
    )
    if invalid.height > 0:
        raise PITDataError("invalid corporate-action market value; certification blocked")
    jumps = (
        windowed.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1.0).alias("_raw_return")
        )
        .filter(pl.col("_raw_return").is_finite() & (pl.col("_raw_return").abs() > float(threshold)))
        .select("instrument_id", "session")
    )
    if verified_actions.height == 0:
        return jumps.collect()
    if "effective_session" in verified_actions.columns:
        key_col = "effective_session"
    elif "effective_date" in verified_actions.columns:
        key_col = "effective_date"
    else:
        raise PITDataError("verified corporate actions lack effective session; certification blocked")
    keys = verified_actions.lazy().select(
        pl.col("instrument_id").cast(pl.String).alias("instrument_id"),
        pl.col(key_col).alias("session"),
    )
    return jumps.join(keys, on=["instrument_id", "session"], how="anti").collect()


def resolve_backtest_corporate_action_evidence(
    *,
    daily_market: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    calendar: SessionCalendar,
    policy: BacktestMarketInputsPolicy,
) -> CorporateActionEvidenceResolution:
    threshold = float(policy.unexplained_price_jump_threshold)
    action_rows: list[dict[str, Any]] = corporate_actions.to_dicts() if corporate_actions.height > 0 else []
    unresolved_reasons: dict[str, set[str]] = {}
    for row in action_rows:
        iid = str(row.get("instrument_id", ""))
        if "evidence_status" not in row or "evidence_reason" not in row:
            raise PITDataError(
                f"legacy corporate-action evidence status missing for {iid!r}; certification blocked"
            )
        status = str(row["evidence_status"] or "")
        atype = str(row.get("action_type", row.get("type", "")))
        if status != "verified" or atype == "unresolved":
            reason = row.get("evidence_reason")
            label = str(reason).strip() if isinstance(reason, str) and reason.strip() else "unresolved_evidence"
            unresolved_reasons.setdefault(iid, set()).add(label)
    if "evidence_status" in corporate_actions.columns:
        verified = corporate_actions.filter(pl.col("evidence_status") == "verified")
    else:
        verified = corporate_actions
    jumps = find_unexplained_price_discontinuities(daily_market=daily_market, verified_actions=verified, threshold=threshold)
    jump_reasons: dict[str, set[str]] = {
        iid: {"unexplained_price_discontinuity"}
        for iid in jumps.get_column("instrument_id").unique().to_list()
        if iid not in unresolved_reasons
    }
    excluded: set[str] = set(unresolved_reasons) | set(jump_reasons)
    exclusion_reasons: dict[str, tuple[str, ...]] = {}
    for iid in excluded:
        merged = sorted(unresolved_reasons.get(iid, set()) | jump_reasons.get(iid, set()))
        exclusion_reasons[iid] = tuple(merged)
    if excluded:
        eligible = daily_market.filter(~pl.col("instrument_id").is_in(sorted(excluded)))
        if "evidence_status" in corporate_actions.columns:
            verified = corporate_actions.filter(
                (pl.col("evidence_status") == "verified") & (~pl.col("instrument_id").is_in(sorted(excluded)))
            )
        else:
            verified = corporate_actions.filter(~pl.col("instrument_id").is_in(sorted(excluded)))
    else:
        eligible = daily_market
        if "evidence_status" in corporate_actions.columns:
            verified = corporate_actions.filter(pl.col("evidence_status") == "verified")
        else:
            verified = corporate_actions
    return CorporateActionEvidenceResolution(
        eligible_daily_market=eligible,
        verified_corporate_actions=verified,
        excluded_instruments=frozenset(excluded),
        exclusion_reasons=dict(exclusion_reasons),
    )


def _coerce_session(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _require_finite_positive(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PITDataError(f"invalid corporate-action market value for {field}; certification blocked") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise PITDataError(f"invalid corporate-action market value for {field}; certification blocked")
    return parsed


def validate_corporate_action_coverage(
    *,
    daily_market: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    calendar: SessionCalendar,
    decision_time_of: Callable[[datetime], datetime],
    policy: BacktestMarketInputsPolicy,
) -> CorporateActionCoverage:
    threshold = float(policy.unexplained_price_jump_threshold)
    action_rows: list[dict[str, Any]] = corporate_actions.to_dicts() if corporate_actions.height > 0 else []
    for row in action_rows:
        if str(row.get("evidence_status", "verified") or "verified") != "verified":
            raise PITDataError(f"unresolved corporate-action evidence for {row.get('instrument_id')!r}; certification blocked")
        raw_type = str(row.get("action_type", row.get("type", "")))
        if raw_type == "unresolved":
            raise PITDataError(f"unresolved corporate-action evidence for {row.get('instrument_id')!r}; certification blocked")
    for row in action_rows:
        raw_type = str(row.get("action_type", row.get("type", "")))
        if raw_type == "no_action":
            raise PITDataError("legacy no_action corporate-action evidence requires rebuild")
    seen_keys: set[tuple[str, str, str]] = set()
    by_session: dict[datetime, list[LedgerCorporateAction]] = {}
    meta_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in action_rows:
        raw_type = str(row.get("action_type", row.get("type", "")))
        if raw_type in ("", "none"):
            raise PITDataError(f"unsupported corporate action type {raw_type!r}")  # pragma: no cover
        eff_raw = row.get("effective_session", row.get("effective_date"))
        avail = row.get("available_at")
        iid = str(row.get("instrument_id", ""))
        if eff_raw is None:
            raise PITDataError(f"invalid corporate action timing for {iid!r}")
        eff = _coerce_session(eff_raw)
        if eff.tzinfo is None:
            raise PITDataError(f"invalid corporate action timing for {iid!r}")
        key = (iid, eff.isoformat(), raw_type)
        if key in seen_keys:
            raise PITDataError(f"ambiguous PIT corporate action rows for {iid!r}")
        seen_keys.add(key)
        if raw_type not in _ALLOWED_ACTION_TYPES:
            raise PITDataError(f"unsupported corporate action type {raw_type!r}")
        if not isinstance(avail, datetime) or avail.tzinfo is None:
            raise PITDataError(f"invalid corporate action timing for {iid!r}")
        session_open = next((s for s in calendar.sessions if s == eff), None)
        if session_open is None:
            raise PITDataError(f"corporate action session is outside calendar for {iid!r}")
        decision_time = decision_time_of(session_open)
        if decision_time.tzinfo is None or avail >= decision_time or avail > session_open:
            raise PITDataError(f"late corporate action for {iid!r}")
        factor = float(row.get("factor", 2.0 if raw_type != "dividend" else 1.0))
        cash = float(row.get("cash_amount", 0.0))
        if raw_type in ("bonus_issue", "split"):
            if not math.isfinite(factor) or factor <= 1.0:
                raise PITDataError(f"invalid corporate action factor for {iid!r}")  # pragma: no cover
            ledger_type = LedgerActionType.SPLIT
        elif raw_type == "reverse_split":
            if not math.isfinite(factor) or not 0.0 < factor < 1.0:
                raise PITDataError(f"invalid corporate action factor for {iid!r}")  # pragma: no cover
            ledger_type = LedgerActionType.REVERSE_SPLIT
        else:
            if not math.isfinite(factor) or factor != 1.0:
                raise PITDataError(f"invalid corporate action factor for {iid!r}")  # pragma: no cover
            if not math.isfinite(cash) or cash < 0:
                raise PITDataError(f"invalid corporate action cash amount for {iid!r}")
            ledger_type = LedgerActionType.DIVIDEND
        action = LedgerCorporateAction(
            action_id=str(row.get("action_id", f"{iid}:{raw_type}:{eff.isoformat()}")),
            instrument_id=iid,
            action_type=ledger_type,
            effective_time=eff,
            factor=float(factor),
            cash_amount=float(cash),
        )
        by_session.setdefault(eff, []).append(action)
        meta_by_key[(iid, eff.isoformat())] = {"raw_type": raw_type, "factor": float(factor), "cash": float(cash)}
    bar_rows = daily_market.to_dicts()
    has_shares = "shares_outstanding" in daily_market.columns and "market_cap" in daily_market.columns
    by_instrument: dict[str, list[dict[str, Any]]] = {}
    for row in bar_rows:
        iid = str(row["instrument_id"])
        by_instrument.setdefault(iid, []).append(row)
    research_returns: dict[tuple[datetime, str], float] = {}
    bonus_rows = [
        row
        for row in action_rows
        if str(row.get("action_type", row.get("type", ""))) == "bonus_issue"
    ]
    has_settlement = "share_listing_date" in corporate_actions.columns and "share_delta" in corporate_actions.columns
    if bonus_rows and not has_settlement:
        raise PITDataError("legacy bonus_issue corporate-action evidence requires rebuild")
    for iid, rows in by_instrument.items():
        ordered = sorted(rows, key=lambda r: _coerce_session(r["session"]))
        for prev_row, curr_row in pairwise(ordered):
            curr_session = _coerce_session(curr_row["session"])
            prev_close = _require_finite_positive(prev_row.get("close"), field="close")
            curr_close = _require_finite_positive(curr_row.get("close"), field="close")
            raw_return = curr_close / prev_close - 1.0
            meta = meta_by_key.get((iid, curr_session.isoformat()))
            if abs(raw_return) > threshold:
                if meta is None:
                    raise PITDataError(f"unexplained price discontinuity for {iid!r}")
                raw_type = str(meta["raw_type"])
                factor = float(meta["factor"])
                cash = float(meta["cash"])
                if raw_type in ("split", "reverse_split", "bonus_issue"):
                    adjusted = factor * curr_close / prev_close - 1.0
                    if abs(adjusted) > threshold:
                        raise PITDataError(f"unreconciled corporate action factor for {iid!r}")  # pragma: no cover
                    research_returns[(curr_session, iid)] = adjusted
                else:
                    adjusted = (curr_close + cash) / prev_close - 1.0
                    if abs(adjusted) > threshold:
                        raise PITDataError(f"unreconciled corporate action factor for {iid!r}")  # pragma: no cover
                    research_returns[(curr_session, iid)] = adjusted
                if has_shares and raw_type in ("split", "reverse_split"):
                    prev_shares = _require_finite_positive(prev_row.get("shares_outstanding"), field="shares_outstanding")
                    curr_shares = _require_finite_positive(curr_row.get("shares_outstanding"), field="shares_outstanding")
                    curr_cap = _require_finite_positive(curr_row.get("market_cap"), field="market_cap")
                    if not _close_enough(curr_shares, prev_shares * factor):
                        raise PITDataError(f"unreconciled corporate action shares for {iid!r}")  # pragma: no cover
                    if not _close_enough(curr_cap, curr_close * curr_shares):
                        raise PITDataError(f"unreconciled corporate action market cap for {iid!r}")  # pragma: no cover
            else:  # pragma: no cover
                if meta is not None:
                    raw_type = str(meta["raw_type"])
                    factor = float(meta["factor"])
                    if raw_type in ("split", "reverse_split", "bonus_issue"):
                        adjusted = factor * curr_close / prev_close - 1.0
                        if abs(adjusted) > threshold:
                            raise PITDataError(f"unreconciled corporate action factor for {iid!r}")
                        research_returns[(curr_session, iid)] = adjusted
                        if has_shares and raw_type in ("split", "reverse_split"):
                            prev_shares = _require_finite_positive(prev_row.get("shares_outstanding"), field="shares_outstanding")
                            curr_shares = _require_finite_positive(curr_row.get("shares_outstanding"), field="shares_outstanding")
                            curr_cap = _require_finite_positive(curr_row.get("market_cap"), field="market_cap")
                            if not _close_enough(curr_shares, prev_shares * factor):
                                raise PITDataError(f"unreconciled corporate action shares for {iid!r}")
                            if not _close_enough(curr_cap, curr_close * curr_shares):
                                raise PITDataError(f"unreconciled corporate action market cap for {iid!r}")
                    else:
                        research_returns[(curr_session, iid)] = (curr_close + float(meta["cash"])) / prev_close - 1.0
                else:
                    research_returns[(curr_session, iid)] = raw_return
    bonus_settlement = bonus_rows if has_settlement else []
    sessions_by_instrument: dict[str, list[datetime]] = {}
    bars_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for brow in bar_rows:
        biid = str(brow.get("instrument_id"))
        bsess = _coerce_session(brow.get("session"))
        sessions_by_instrument.setdefault(biid, []).append(bsess)
        bars_by_key[(biid, bsess.isoformat())] = brow
    for biid in sessions_by_instrument:
        sessions_by_instrument[biid] = sorted(sessions_by_instrument[biid])
    seen_settlement_ids: set[str] = set()
    has_duplicate_settlement_id = False
    all_rows_ok = True
    settlement_by_key: dict[tuple[str, str], list[Any]] = {}
    for row in bonus_settlement:
        riid = str(row.get("instrument_id", ""))
        aid = str(row.get("action_id", ""))
        has_duplicate_settlement_id = has_duplicate_settlement_id or (aid in seen_settlement_ids)
        seen_settlement_ids.add(aid)
        raw_delta = row.get("share_delta")
        delta = raw_delta if isinstance(raw_delta, int) and not isinstance(raw_delta, bool) and raw_delta > 0 else 0
        raw_listing = row.get("share_listing_date")
        listing_sess = raw_listing if isinstance(raw_listing, datetime) else None
        eff_session = _coerce_session(row.get("effective_session", row.get("effective_date")))
        row_ok = delta > 0 and listing_sess is not None and listing_sess.tzinfo is not None and listing_sess >= eff_session
        all_rows_ok = all_rows_ok and row_ok
        list_key = (riid, listing_sess.isoformat() if isinstance(listing_sess, datetime) else "")
        prev_entry = settlement_by_key.get(list_key)
        prev_total = prev_entry[2] if prev_entry is not None else 0
        prev_eff = prev_entry[1] if prev_entry is not None else eff_session
        merged_eff = prev_eff if prev_eff <= eff_session else eff_session
        merged_sess = prev_entry[0] if prev_entry is not None else listing_sess
        settlement_by_key[list_key] = [merged_sess, merged_eff, prev_total + delta]
    settlement_failed = False
    for (liid, liso), (lsess, min_eff, total_delta) in settlement_by_key.items():
        current_bar = bars_by_key.get((liid, liso))
        ordered_iid = sessions_by_instrument.get(liid, [])
        earlier = [s for s in ordered_iid if lsess is not None and s < lsess]
        previous_session = earlier[-1] if earlier else None
        previous_bar = bars_by_key.get((liid, previous_session.isoformat())) if previous_session is not None else None
        current_close = _require_finite_positive(current_bar.get("close"), field="close") if current_bar is not None else float("nan")
        current_shares = _require_finite_positive(current_bar.get("shares_outstanding"), field="shares_outstanding") if current_bar is not None else float("nan")
        current_cap = _require_finite_positive(current_bar.get("market_cap"), field="market_cap") if current_bar is not None else float("nan")
        previous_shares = _require_finite_positive(previous_bar.get("shares_outstanding"), field="shares_outstanding") if previous_bar is not None else float("nan")
        if (
            not all_rows_ok
            or has_duplicate_settlement_id
            or total_delta <= 0
            or current_bar is None
            or previous_bar is None
            or lsess is None
            or lsess.tzinfo is None
            or lsess < min_eff
            or current_shares != previous_shares + total_delta
            or not _close_enough(current_cap, current_close * current_shares)
        ):
            settlement_failed = True
    if settlement_failed:
        raise PITDataError("unreconciled corporate action listed shares at listing session; certification blocked")
    actions_map = {session: tuple(sorted(actions, key=lambda a: (a.instrument_id, a.action_id))) for session, actions in by_session.items()}
    return CorporateActionCoverage(actions_by_session=actions_map, research_returns_by_key=research_returns)


def _frame_for(repository: PITSnapshotRepository) -> pl.DataFrame | None:
    frames = getattr(repository, "_frames", {})
    frame = frames.get(SilverTable.DAILY_MARKET)
    return frame


def _rolling_inputs(
    full: pl.DataFrame,
    policy: BacktestMarketInputsPolicy,
    session_calendar: tuple[datetime, ...],
    adjusted_returns: Mapping[tuple[datetime, str], float] | None = None,
) -> tuple[dict[tuple[datetime, str], float], dict[tuple[datetime, str], float], dict[datetime, float]]:
    required = {"session", "instrument_id", "close", "trading_value", "market_cap"}
    missing = sorted(required - set(full.columns))
    if missing:
        raise PITDataError(f"daily market missing rolling columns: {missing}")
    ordered = (
        full.lazy()
        .select(
            pl.col("session").alias("session"),
            pl.col("instrument_id").cast(pl.String).alias("instrument_id"),
            pl.col("close").cast(pl.Float64).alias("close"),
            pl.col("trading_value").cast(pl.Float64).alias("trading_value"),
            pl.col("market_cap").cast(pl.Float64).alias("market_cap"),
        )
        .sort(["instrument_id", "session"])
        .collect()
    )
    sessions = tuple(session_calendar)
    session_index = {session: index for index, session in enumerate(sessions)}
    if len(session_index) != len(sessions):
        raise PITDataError("session calendar contains duplicate sessions")
    adtv: dict[tuple[datetime, str], float] = {}
    vols: dict[tuple[datetime, str], float] = {}
    history: dict[str, dict[datetime, tuple[float, float, float]]] = {}
    for row in ordered.to_dicts():
        sess = _coerce_session(row["session"])
        if sess not in session_index:
            raise PITDataError(f"daily market session is outside calendar: {sess!r}")
        values = (
            float(row["close"]),
            float(row["trading_value"]),
            float(row["market_cap"]),
        )
        if not all(math.isfinite(value) for value in values) or values[0] <= 0 or values[2] <= 0:
            raise PITDataError(f"invalid rolling market input for {row['instrument_id']!r}")
        history.setdefault(str(row["instrument_id"]), {})[sess] = values
    scale = math.sqrt(float(policy.annualization_sessions))
    for iid, points in history.items():
        for idx, session in enumerate(sessions):
            current = points.get(session)
            if current is None:
                continue
            adtv_sessions = sessions[max(0, idx - policy.adtv_sessions + 1) : idx + 1]
            if len(adtv_sessions) == policy.adtv_sessions and all(item in points for item in adtv_sessions):
                adtv_values = sorted(points[item][1] for item in adtv_sessions)
                middle = len(adtv_values) // 2
                median = adtv_values[middle] if len(adtv_values) % 2 else (adtv_values[middle - 1] + adtv_values[middle]) / 2.0
                adtv[(session, iid)] = float(median)
            volatility_sessions = sessions[max(0, idx - policy.volatility_sessions) : idx + 1]
            if len(volatility_sessions) == policy.volatility_sessions + 1 and all(item in points for item in volatility_sessions):
                returns = []
                for pos in range(1, len(volatility_sessions)):
                    key = (volatility_sessions[pos], iid)
                    if adjusted_returns is not None and key in adjusted_returns:
                        returns.append(float(adjusted_returns[key]))
                    else:
                        returns.append(points[volatility_sessions[pos]][0] / points[volatility_sessions[pos - 1]][0] - 1.0)  # pragma: no cover
                mean = sum(returns) / len(returns)
                variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
                volatility = math.sqrt(variance) * scale
                if math.isfinite(volatility) and volatility > 0:
                    vols[(session, iid)] = volatility
    market_vol: dict[datetime, float] = {}
    market_returns: dict[datetime, float] = {}
    for idx, session in enumerate(sessions):
        if idx == 0:
            continue
        weighted_returns: list[tuple[float, float]] = []
        previous = sessions[idx - 1]
        for iid_key, points in history.items():
            prior = points.get(previous)
            current = points.get(session)
            if prior is not None and current is not None:
                key = (session, iid_key)
                if adjusted_returns is not None and key in adjusted_returns:
                    weighted_returns.append((prior[2], float(adjusted_returns[key])))
                else:
                    weighted_returns.append((prior[2], current[0] / prior[0] - 1.0))
        total_cap = sum(cap for cap, _ in weighted_returns)
        if total_cap > 0 and weighted_returns:
            market_returns[session] = sum(cap / total_cap * ret for cap, ret in weighted_returns)
    for idx, session in enumerate(sessions):
        return_sessions = sessions[max(1, idx - policy.market_volatility_sessions + 1) : idx + 1]
        if len(return_sessions) != policy.market_volatility_sessions or not all(item in market_returns for item in return_sessions):
            continue
        returns = [market_returns[item] for item in return_sessions]
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        volatility = math.sqrt(variance) * scale
        if math.isfinite(volatility) and volatility > 0:
            market_vol[session] = volatility
    return adtv, vols, market_vol


def _resolve_sector(
    *,
    security_master: pl.DataFrame | None,
    instrument_id: str,
    session: datetime,
    decision_time: datetime,
    master_index: Mapping[str, tuple[dict[str, Any], ...]] | None = None,
) -> str:
    if security_master is None or security_master.is_empty():
        raise PITDataError("missing PIT security master")
    rows = master_index.get(instrument_id, ()) if master_index is not None else security_master.to_dicts()
    candidates = [
        row
        for row in rows
        if str(row.get("instrument_id")) == instrument_id
        and isinstance(row.get("available_at"), datetime)
        and row["available_at"] <= decision_time
        and _coerce_session(row.get("valid_from", session)) <= session
        and session <= _coerce_session(row.get("valid_to", session))
    ]
    sectors = {str(row.get("sector", "")) for row in candidates}
    sectors = {s for s in sectors if s and s != "__GLOBAL__"}
    if len(sectors) != 1 or len(candidates) != 1:
        raise PITDataError(f"invalid PIT sector for {instrument_id!r}")
    return next(iter(sectors))


def build_backtest_sessions(
    *,
    snapshot_repository: PITSnapshotRepository,
    calendar: SessionCalendar,
    start: datetime,
    end: datetime,
    decision_time_of: Callable[[datetime], datetime],
    security_master: pl.DataFrame | None = None,
    corporate_actions: pl.DataFrame | None = None,
    policy: BacktestMarketInputsPolicy = BacktestMarketInputsPolicy(),  # noqa: B008
) -> tuple[BacktestSession, ...]:
    if start.tzinfo is None or end.tzinfo is None:
        raise PITDataError("start and end must be timezone-aware")
    if start > end:
        raise PITDataError("start must not be after end")
    ordered = tuple(sorted(calendar.sessions))
    decisions = tuple(s for s in ordered if start <= s <= end)
    if not decisions:
        raise PITDataError("no sessions in requested range")
    full = _frame_for(snapshot_repository)
    if full is None or full.height == 0:
        raise PITDataError("missing daily market bars for requested coverage")
    required_cols = {
        "session",
        "instrument_id",
        "open",
        "close",
        "volume",
        "trading_value",
        "available_at",
    }
    missing_cols = [c for c in required_cols if c not in full.columns]
    if missing_cols:
        raise PITDataError(f"daily market missing columns: {missing_cols}")
    if full.select(["session", "instrument_id"]).is_duplicated().any():
        raise PITDataError("duplicate bar for (session, instrument_id) pair")
    bad_mask = (
        pl.col("open").is_null()
        | (pl.col("open") <= 0)
        | pl.col("close").is_null()
        | (pl.col("close") <= 0)
    )
    if full.select(bad_mask.any().alias("_bad")).item(0, 0):
        raise PITDataError("non-positive price bar detected (open or close <= 0)")
    distinct_sessions = full.select("session").unique()["session"].to_list()
    decision_times = [decision_time_of(s) for s in distinct_sessions]
    decision_frame = pl.DataFrame({"session": distinct_sessions, "decision_time": decision_times})
    # Historical feeds use both KST and UTC timestamp annotations.  Compare
    # instants in the bar's timezone so a valid receipt is not rejected (or a
    # leak missed) merely because the two frames carry different tz labels.
    available_dtype = full.schema.get("available_at")
    available_tz = getattr(available_dtype, "time_zone", None)
    decision_dtype = decision_frame.schema.get("decision_time")
    decision_tz = getattr(decision_dtype, "time_zone", None)
    if available_tz and decision_tz and available_tz != decision_tz:
        decision_frame = decision_frame.with_columns(
            pl.col("decision_time").dt.convert_time_zone(available_tz)
        )
    joined = full.join(decision_frame, on="session", how="left")
    if joined.select((pl.col("available_at") > pl.col("decision_time")).any().alias("_leak")).item(0, 0):
        raise PITDataError("available_at after decision time detected")
    partitioned = full.partition_by("session", as_dict=True)
    by_session = {k[0] if isinstance(k, tuple) else k: v for k, v in partitioned.items()}
    for session_open in decisions:
        idx = ordered.index(session_open)
        next_open = ordered[idx + 1] if idx + 1 < len(ordered) else None
        if next_open is None or next_open not in by_session:
            raise PITDataError(f"missing next session bar after {session_open}")
    if security_master is None or security_master.is_empty():
        raise PITDataError("missing PIT security master")
    if corporate_actions is None:
        raise PITDataError("missing PIT corporate actions")
    master_index: dict[str, tuple[dict[str, Any], ...]] = {}
    if security_master is not None:
        grouped_master: dict[str, list[dict[str, Any]]] = {}
        for row in security_master.to_dicts():
            grouped_master.setdefault(str(row.get("instrument_id")), []).append(row)
        master_index = {iid: tuple(rows) for iid, rows in grouped_master.items()}
    resolution = resolve_backtest_corporate_action_evidence(daily_market=full, corporate_actions=corporate_actions, calendar=calendar, policy=policy)
    full = resolution.eligible_daily_market
    corporate_actions = resolution.verified_corporate_actions
    partitioned = full.partition_by("session", as_dict=True)
    by_session = {k[0] if isinstance(k, tuple) else k: v for k, v in partitioned.items()}
    coverage = validate_corporate_action_coverage(daily_market=full, corporate_actions=corporate_actions, calendar=calendar, decision_time_of=decision_time_of, policy=policy)
    adtv_map, vol_map, market_vol_map = _rolling_inputs(full, policy, ordered, coverage.research_returns_by_key)
    for session_open in decisions:
        decision_time = decision_time_of(session_open)
        for instrument_id in by_session[session_open].get_column("instrument_id").to_list():
            _resolve_sector(
                security_master=security_master,
                instrument_id=str(instrument_id),
                session=session_open,
                decision_time=decision_time,
                master_index=master_index,
            )
    requested_keys = {
        (session_open, str(instrument_id))
        for session_open in decisions
        for instrument_id in by_session[session_open].get_column("instrument_id").to_list()
    }
    if requested_keys - adtv_map.keys() or requested_keys - vol_map.keys() or any(
        session_open not in market_vol_map for session_open in decisions
    ):
        raise PITDataError("insufficient rolling PIT market history")
    sessions: list[BacktestSession] = []
    for session_open in decisions:
        decision_time = decision_time_of(session_open)
        if decision_time.tzinfo is None:
            raise PITDataError("decision_time must be timezone-aware")
        idx = ordered.index(session_open)
        next_open = ordered[idx + 1] if idx + 1 < len(ordered) else None
        if next_open is None or next_open not in by_session:
            raise PITDataError(f"missing next session bar after {session_open}")
        part = by_session[session_open]
        instrument_ids = part.get_column("instrument_id").to_list()
        opens = part.get_column("open").cast(pl.Float64).to_list()
        closes = part.get_column("close").cast(pl.Float64).to_list()
        caps = part.get_column("market_cap").cast(pl.Float64).to_list()
        order = sorted(range(len(instrument_ids)), key=lambda i: str(instrument_ids[i]))
        bars = tuple(
            HistoricalBar(
                session_open,
                str(instrument_ids[i]),
                float(opens[i]),
                float(closes[i]),
                float(adtv_map[(session_open, str(instrument_ids[i]))]),
                float(vol_map[(session_open, str(instrument_ids[i]))]),
            )
            for i in order
        )
        mark_prices = {str(instrument_ids[i]): float(closes[i]) for i in range(len(instrument_ids))}
        adtv20 = {str(instrument_ids[i]): float(adtv_map[(session_open, str(instrument_ids[i]))]) for i in range(len(instrument_ids))}
        volatilities = {str(instrument_ids[i]): float(vol_map[(session_open, str(instrument_ids[i]))]) for i in range(len(instrument_ids))}
        market_caps = {str(instrument_ids[i]): float(caps[i]) for i in range(len(instrument_ids))}
        sectors = {
            str(instrument_ids[i]): _resolve_sector(
                security_master=security_master,
                instrument_id=str(instrument_ids[i]),
                session=session_open,
                decision_time=decision_time,
                master_index=master_index,
            )
            for i in range(len(instrument_ids))
        }
        instruments: dict[str, Any] = {
            iid: Instrument(iid, AssetKind.STOCK, "KRX", iid.split(":")[-1], "KRW") for iid in mark_prices
        }
        market_snapshot: dict[str, Any] = {
            "mark_prices": mark_prices,
            "adtv20": adtv20,
            "volatilities": volatilities,
            "sectors": sectors,
            "instruments": instruments,
            "market_volatility": float(market_vol_map[session_open]),
            "market_caps": market_caps,
        }
        sessions.append(
            BacktestSession(
                session_open=session_open,
                decision_time=decision_time,
                bars=bars,
                actions=coverage.actions_by_session.get(session_open, ()),
                market_snapshot=market_snapshot,
            )
        )
    return tuple(sessions)
