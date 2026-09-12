"""PIT backtest session builder from certified Silver snapshots."""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from typing import Any

import polars as pl

from src.core.instruments import AssetKind, Instrument
from src.core.ledger import LedgerActionType, LedgerCorporateAction, LedgerSuccessorAllocation
from src.core.time import SessionCalendar
from src.data.schemas import PITDataError, SilverTable
from src.data.snapshot import PITSnapshotRepository
from src.engine.backtest import BacktestSession
from src.engine.fill_model import HistoricalBar

_SUCCESSOR_ALLOCATION_KEYS = frozenset(
    {"successor_security_id", "successor_instrument_id", "ratio", "cost_basis_weight"}
)


def _decode_decimal_string(value: object, *, field: str, lifecycle_event_id: str) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor {field} must be a Decimal string")
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, ValueError, AttributeError, TypeError) as exc:
        raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor {field} is malformed") from exc
    if not number.is_finite() or number <= 0:
        raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor {field} must be positive finite")
    return number


def decode_successor_allocations(*, raw: object, lifecycle_event_id: str) -> tuple[LedgerSuccessorAllocation, ...]:
    """Decode canonical successor allocation JSON into Decimal ledger allocations.

    Expects a JSON array of objects with exactly the keys
    ``successor_security_id``, ``successor_instrument_id``, ``ratio`` and
    ``cost_basis_weight``. Ratios and weights must be Decimal strings so no
    binary-float rounding enters the ledger entitlement math.
    """
    if not lifecycle_event_id:
        raise PITDataError("lifecycle successor allocations require lifecycle_event_id")
    if raw is None:
        raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor allocations are missing")
    if isinstance(raw, str):
        try:
            parsed: object = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor allocations are malformed") from exc
    else:
        parsed = raw
    if not isinstance(parsed, list) or not parsed:
        raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor allocations must be a non-empty array")
    allocations: list[LedgerSuccessorAllocation] = []
    seen_security_ids: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor allocation must be an object")
        if set(item) - _SUCCESSOR_ALLOCATION_KEYS:
            raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor allocation has unknown keys")
        security_id = item.get("successor_security_id")
        instrument_id = item.get("successor_instrument_id")
        if not isinstance(security_id, str) or not security_id:
            raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor identity must not be empty")
        if not isinstance(instrument_id, str) or not instrument_id:
            raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor identity must not be empty")
        if security_id == instrument_id:
            raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor identity must not be equal")
        if security_id in seen_security_ids:
            raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor_security_id must be unique")
        seen_security_ids.add(security_id)
        ratio = _decode_decimal_string(item.get("ratio"), field="ratio", lifecycle_event_id=lifecycle_event_id)
        weight = _decode_decimal_string(
            item.get("cost_basis_weight"), field="cost_basis_weight", lifecycle_event_id=lifecycle_event_id
        )
        allocations.append(LedgerSuccessorAllocation(instrument_id, ratio, weight))
    return tuple(allocations)


@dataclass(frozen=True, slots=True)
class BacktestMarketInputsPolicy:
    version: str = "korean-equity-market-inputs-v2"
    adtv_sessions: int = 20
    volatility_sessions: int = 60
    market_volatility_sessions: int = 60
    annualization_sessions: int = 252
    unexplained_price_jump_threshold: float = 0.5
    corporate_action_quarantine_sessions: int = 60

    def __post_init__(self) -> None:
        if (
            self.version != "korean-equity-market-inputs-v2"
            or self.adtv_sessions != 20
            or self.volatility_sessions != 60
            or self.market_volatility_sessions != 60
            or self.annualization_sessions != 252
            or self.unexplained_price_jump_threshold != 0.5
            or self.corporate_action_quarantine_sessions != 60
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
    quarantine_sessions_by_instrument: Mapping[str, tuple[datetime, ...]]


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


def _coerce_event_session(value: Any, *, instrument_id: str) -> datetime:
    if isinstance(value, datetime):
        eff = value
    elif value is None:
        raise PITDataError(f"unresolved corporate-action effective session missing for {instrument_id!r}")
    else:
        try:
            eff = datetime.fromisoformat(str(value))
        except (TypeError, ValueError) as exc:
            raise PITDataError(f"unresolved corporate-action effective session invalid for {instrument_id!r}") from exc
    if eff.tzinfo is None:
        raise PITDataError(f"unresolved corporate-action effective session must be timezone-aware for {instrument_id!r}")
    return eff


def resolve_backtest_corporate_action_evidence(
    *,
    daily_market: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    calendar: SessionCalendar,
    policy: BacktestMarketInputsPolicy,
) -> CorporateActionEvidenceResolution:
    threshold = float(policy.unexplained_price_jump_threshold)
    quarantine_window = int(policy.corporate_action_quarantine_sessions)
    sessions = tuple(calendar.sessions)
    session_index = {session: index for index, session in enumerate(sessions)}
    sessions_by_date = {session.date(): session for session in sessions}

    def _session_for_action_date(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise PITDataError("corporate-action effective session must be timezone-aware")
            action_date = value.date()
        else:
            try:
                parsed = datetime.fromisoformat(str(value))
            except (TypeError, ValueError) as exc:
                raise PITDataError("invalid corporate-action session date; certification blocked") from exc
            if parsed.tzinfo is None:
                raise PITDataError("corporate-action effective session must be timezone-aware")
            action_date = parsed.date()
        try:
            return sessions_by_date[action_date]
        except KeyError as exc:
            raise PITDataError("corporate-action session is outside calendar; certification blocked") from exc

    # Silver actions are stored as UTC instants by some legacy materializers,
    # whereas daily bars use the KRX session timezone.  Their market date is
    # authoritative, so align event and settlement dates to the certified
    # calendar session before any equality join or ledger application.
    session_columns = [name for name in ("effective_session", "effective_date", "share_listing_date") if name in corporate_actions.columns]
    if session_columns:
        for name in ("effective_session", "effective_date"):
            if name not in corporate_actions.columns:
                continue
            if corporate_actions.filter(pl.col(name).is_null()).height > 0:
                raise PITDataError("corporate-action effective session missing; certification blocked")
        corporate_actions = corporate_actions.with_columns(
            [
                pl.col(name).map_elements(
                    _session_for_action_date,
                    return_dtype=pl.Datetime(time_zone=str(sessions[0].tzinfo)),
                ).alias(name)
                for name in session_columns
            ]
        )
    action_rows: list[dict[str, Any]] = corporate_actions.to_dicts() if corporate_actions.height > 0 else []
    events: list[tuple[str, datetime, str]] = []
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
            eff_raw = row.get("effective_session", row.get("effective_date"))
            eff = _coerce_event_session(eff_raw, instrument_id=iid)
            if eff not in session_index:
                raise PITDataError(
                    f"unresolved corporate-action effective session is outside calendar for {iid!r}"
                )
            events.append((iid, eff, label))
    if "evidence_status" in corporate_actions.columns:
        verified = corporate_actions.filter(pl.col("evidence_status") == "verified")
    else:
        verified = corporate_actions
    jumps = find_unexplained_price_discontinuities(daily_market=daily_market, verified_actions=verified, threshold=threshold)
    if jumps.height > 0:
        for jump_row in jumps.to_dicts():
            iid = str(jump_row["instrument_id"])
            if iid in {iid_event for iid_event, _, _ in events}:
                continue
            jump_raw = _coerce_session(jump_row["session"])
            # Daily bars may carry the session close (15:30 KST) while the
            # canonical calendar stores the same trading date at its open.
            # Join by market date before applying the quarantine window.
            jump_session = sessions_by_date.get(jump_raw.date())
            if jump_session is None:
                raise PITDataError(f"price discontinuity session is outside calendar for {iid!r}")
            if jump_session not in session_index:
                raise PITDataError(f"price discontinuity session is outside calendar for {iid!r}")
            events.append((iid, jump_session, "unexplained_price_discontinuity"))
    quarantine: dict[str, set[datetime]] = {}
    reasons: dict[str, set[str]] = {}
    for iid, event_session, label in events:
        index = session_index[event_session]
        window = sessions[index : index + quarantine_window + 1]
        quarantine.setdefault(iid, set()).update(window)
        reasons.setdefault(iid, set()).add(label)
    quarantine_tuples: dict[str, tuple[datetime, ...]] = {
        iid: tuple(sorted(slots)) for iid, slots in quarantine.items()
    }
    exclusion_reasons: dict[str, tuple[str, ...]] = {
        iid: tuple(sorted(labels)) for iid, labels in reasons.items()
    }
    if quarantine:
        # Daily bars commonly use a close/open instant while the canonical
        # calendar uses a date anchor.  Materialize quarantine keys using the
        # actual bar timestamp for each instrument/date so the anti-join does
        # not silently miss the affected rows.
        supplied_by_date: dict[tuple[str, Any], list[datetime]] = {}
        if {"instrument_id", "session"}.issubset(daily_market.columns):
            for market_row in daily_market.select("instrument_id", "session").to_dicts():
                market_session = _coerce_session(market_row["session"])
                supplied_by_date.setdefault(
                    (str(market_row["instrument_id"]), market_session.date()), []
                ).append(market_session)
        quarantined_pairs: list[tuple[str, datetime]] = []
        for iid, slots in quarantine.items():
            for slot in slots:
                matching = supplied_by_date.get((iid, slot.date()))
                if matching:
                    quarantined_pairs.extend((iid, value) for value in matching)
                else:
                    quarantined_pairs.append((iid, slot))
        block = pl.DataFrame(
            {"instrument_id": [iid for iid, _ in quarantined_pairs], "session": [s for _, s in quarantined_pairs]}
        )
        eligible = daily_market.join(block, on=["instrument_id", "session"], how="anti")
        if verified.height > 0:
            key_col: str | None = None
            if "effective_session" in verified.columns:
                key_col = "effective_session"
            elif "effective_date" in verified.columns:
                key_col = "effective_date"
            if key_col is not None:
                renamed = block.rename({"session": key_col})
                verified = verified.join(renamed, on=["instrument_id", key_col], how="anti")
    else:
        eligible = daily_market
    if daily_market.height > 0 and "instrument_id" in daily_market.columns and "session" in daily_market.columns:
        supplied: dict[str, set[datetime]] = {}
        for row in daily_market.select("instrument_id", "session").to_dicts():
            supplied.setdefault(str(row["instrument_id"]), set()).add(_coerce_session(row["session"]))
        excluded = frozenset(
            iid
            for iid, slots in supplied.items()
            if slots
            and {
                value.date() for value in slots
            }
            <= {value.date() for value in quarantine.get(iid, set())}
        )
    else:
        excluded = frozenset()
    return CorporateActionEvidenceResolution(
        eligible_daily_market=eligible,
        verified_corporate_actions=verified,
        excluded_instruments=excluded,
        exclusion_reasons=dict(exclusion_reasons),
        quarantine_sessions_by_instrument=dict(quarantine_tuples),
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
    daily_market: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    calendar: SessionCalendar,
    decision_time_of: Callable[[datetime], datetime],
    policy: BacktestMarketInputsPolicy,
    lifecycle_cleanup_keys: frozenset[tuple[str, datetime]] = frozenset(),
) -> CorporateActionCoverage:
    threshold = float(policy.unexplained_price_jump_threshold)
    action_rows: list[dict[str, Any]] = corporate_actions.to_dicts() if corporate_actions.height > 0 else []
    # Structured DART evidence may contain historical decisions predating the
    # supplied market window.  They cannot affect this validation window and
    # must not block a later, otherwise certified rebuild (the 001260 legacy
    # no_action rows are an example).  Actions inside the observed instrument
    # range remain fail-closed below.
    market_bounds: dict[str, tuple[datetime, datetime]] = {}
    if daily_market.height > 0 and {"instrument_id", "session"}.issubset(daily_market.columns):
        for group_key, group in daily_market.group_by("instrument_id", maintain_order=True):
            sessions = [_coerce_session(value) for value in group["session"].to_list()]
            if sessions:
                market_bounds[str(group_key[0])] = (min(sessions), max(sessions))
    filtered_actions: list[dict[str, Any]] = []
    for row in action_rows:
        iid = str(row.get("instrument_id", ""))
        raw_eff = row.get("effective_session", row.get("effective_date"))
        raw_type = str(row.get("action_type", row.get("type", "")))
        raw_status = str(row.get("evidence_status", "verified") or "verified")
        if market_bounds and iid not in market_bounds:
            # No market observation exists for this instrument in the
            # certified window, so its action cannot explain or invalidate a
            # price series that is not present.
            continue
        if iid in market_bounds and raw_eff is not None and (
            raw_status != "verified" or raw_type == "no_action"
        ):
            try:
                effective = _coerce_session(raw_eff)
            except (TypeError, ValueError):
                effective = None
            if effective is not None and effective.tzinfo is not None:
                first, last = market_bounds[iid]
                if effective < first or effective > last:
                    continue
        filtered_actions.append(row)
    action_rows = filtered_actions
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
                    if (iid, curr_session) in lifecycle_cleanup_keys:
                        research_returns[(curr_session, iid)] = raw_return
                        continue
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
    failed_settlement_instruments: set[str] = set()
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
            failed_settlement_instruments.add(liid)
    if settlement_failed:
        listed = ", ".join(sorted(failed_settlement_instruments))
        raise PITDataError(
            "unreconciled corporate action listed shares at listing session; "
            f"certification blocked for {listed}"
        )
    actions_map = {session: tuple(sorted(actions, key=lambda a: (a.instrument_id, a.action_id))) for session, actions in by_session.items()}
    return CorporateActionCoverage(actions_by_session=actions_map, research_returns_by_key=research_returns)


def _frame_for(repository: PITSnapshotRepository) -> pl.DataFrame | None:
    frames = getattr(repository, "_frames", {})
    frame = frames.get(SilverTable.DAILY_MARKET)
    return frame


def bound_calendar_to_market_window(
    sessions: tuple[datetime, ...], market_sessions: Collection[datetime]
) -> tuple[datetime, ...]:
    """Trim calendar padding outside the market-data span.

    Interior sessions absent from the market are preserved for rolling-window
    continuity; only sessions outside [min(market), max(market)] are dropped.

    Args:
        sessions: Ordered calendar sessions, padding allowed.
        market_sessions: Sessions actually present in market history.

    Returns:
        Calendar sessions within the market span, in original order.

    Raises:
        PITDataError: If the calendar or the market sessions are empty.
    """
    if len(sessions) == 0:
        raise PITDataError("calendar must be non-empty")
    market_span = tuple(market_sessions)
    if len(market_span) == 0:
        raise PITDataError("rolling inputs require at least one market session")
    start = min(market_span)
    end = max(market_span)
    # 경계 밖 세션만 제거하고 내부는 원래 순서대로 유지한다.
    return tuple(session for session in sessions if start <= session <= end)


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
    # history 구축 직후 무거운 이중 루프 진입 전에 캘린더를 시장 구간으로 바운딩한다.
    sessions = bound_calendar_to_market_window(sessions, {key for points in history.values() for key in points})
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
    # Multiple source snapshots can describe the same PIT sector (for
    # example, a bridge row with a different status).  They are equivalent
    # for sector resolution; only conflicting sectors are ambiguous.
    if len(sectors) != 1 or not candidates:
        raise PITDataError(f"invalid PIT sector for {instrument_id!r}")
    return next(iter(sectors))


def resolve_backtest_lifecycle_evidence(
    *,
    daily_market: pl.DataFrame,
    lifecycle_events: pl.DataFrame,
    calendar: SessionCalendar,
    decision_time_of: Callable[[datetime], datetime],
) -> CorporateActionCoverage:
    """Resolve verified lifecycle evidence into deterministic delisting actions without invented settlement."""
    sessions = tuple(sorted(calendar.sessions))
    sessions_by_date = {session.date(): session for session in sessions}
    _ = daily_market
    event_rows = lifecycle_events.to_dicts() if lifecycle_events.height > 0 else []
    by_session: dict[datetime, list[LedgerCorporateAction]] = {}
    seen_keys: set[tuple[str, str]] = set()
    for row in event_rows:
        if str(row.get("evidence_status", "verified")) != "verified":  # pragma: no cover
            raise PITDataError(f"unresolved lifecycle evidence for {row.get('instrument_id')!r}")
        iid = str(row.get("instrument_id", ""))
        raw_delisting = row.get("delisting_date")
        delisting_date = raw_delisting.date() if isinstance(raw_delisting, datetime) else raw_delisting
        effective = sessions_by_date.get(delisting_date) if delisting_date is not None else None
        if effective is None:  # pragma: no cover
            raise PITDataError(f"lifecycle delisting session is outside calendar for {iid!r}")
        action_id = str(row.get("action_id", f"{iid}:delisting:{effective.date().isoformat()}"))
        key = (effective.isoformat(), action_id)
        if key in seen_keys:  # pragma: no cover
            raise PITDataError(f"duplicate lifecycle event for {iid!r}")
        seen_keys.add(key)
        avail = row.get("available_at")
        if not isinstance(avail, datetime) or avail.tzinfo is None:  # pragma: no cover
            raise PITDataError(f"invalid lifecycle availability for {iid!r}")
        decision_time = decision_time_of(effective)
        if avail >= decision_time:  # pragma: no cover
            raise PITDataError(f"late lifecycle availability for {iid!r}")
        raw_kind = row.get("resolution_kind")
        kind_text = str(raw_kind or "").strip()
        if not kind_text:
            raw_cash_hint = row.get("cash_settlement_per_share")
            kind_text = "cash_settlement" if raw_cash_hint is not None else "unsettled_delisting"
        if kind_text == "merger_or_exchange":
            lifecycle_event_id = row.get("lifecycle_event_id")
            if not isinstance(lifecycle_event_id, str) or not lifecycle_event_id:
                raise PITDataError(f"lifecycle merger event requires lifecycle_event_id for {iid!r}")
            raw_delivery = row.get("successor_delivery_date")
            delivery_day = raw_delivery.date() if isinstance(raw_delivery, datetime) else raw_delivery
            delivery_session = sessions_by_date.get(delivery_day) if delivery_day is not None else None
            if delivery_session is None:
                raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor delivery is outside calendar")
            raw_allocations = row.get("successor_allocations_json")
            allocations = decode_successor_allocations(raw=raw_allocations, lifecycle_event_id=lifecycle_event_id)
            source_security_id = row.get("source_security_id")
            if not isinstance(source_security_id, str) or not source_security_id:
                raise PITDataError(f"lifecycle {lifecycle_event_id!r} requires source_security_id")
            if avail >= decision_time_of(delivery_session):
                raise PITDataError(f"lifecycle {lifecycle_event_id!r} successor delivery is not PIT-available")
            entitlement_action = LedgerCorporateAction(
                action_id=f"{lifecycle_event_id}:exchange_entitlement",
                instrument_id=iid,
                action_type=LedgerActionType.EXCHANGE_ENTITLEMENT,
                effective_time=effective,
                factor=1.0,
                cash_amount=0.0,
                successor_allocations=allocations,
                lifecycle_event_id=lifecycle_event_id,
            )
            delivery_action = LedgerCorporateAction(
                action_id=f"{lifecycle_event_id}:successor_delivery",
                instrument_id=iid,
                action_type=LedgerActionType.SUCCESSOR_DELIVERY,
                effective_time=delivery_session,
                factor=1.0,
                cash_amount=0.0,
                successor_allocations=allocations,
                lifecycle_event_id=lifecycle_event_id,
            )
            for action_session, action in ((effective, entitlement_action), (delivery_session, delivery_action)):
                action_key = (action_session.isoformat(), action.action_id)
                if action_key in seen_keys:
                    raise PITDataError(f"duplicate lifecycle event for {iid!r}")
                seen_keys.add(action_key)
            by_session.setdefault(effective, []).append(entitlement_action)
            by_session.setdefault(delivery_session, []).append(delivery_action)
            continue
        if kind_text == "cash_settlement":
            raw_cash = row.get("cash_settlement_per_share")
            if raw_cash is None:  # pragma: no cover
                raise PITDataError(f"invalid lifecycle settlement for {iid!r}")
            cash_amount = float(raw_cash)
            if not math.isfinite(cash_amount) or cash_amount < 0:  # pragma: no cover
                raise PITDataError(f"invalid lifecycle settlement for {iid!r}")
            by_session.setdefault(effective, []).append(
                LedgerCorporateAction(
                    action_id=action_id,
                    instrument_id=iid,
                    action_type=LedgerActionType.DELISTING_CASH_OUT,
                    effective_time=effective,
                    factor=1.0,
                    cash_amount=cash_amount,
                )
            )
        elif kind_text == "unsettled_delisting":
            by_session.setdefault(effective, []).append(
                LedgerCorporateAction(
                    action_id=action_id,
                    instrument_id=iid,
                    action_type=LedgerActionType.DELISTING_UNSETTLED,
                    effective_time=effective,
                    factor=1.0,
                    cash_amount=0.0,
                )
            )
        else:  # pragma: no cover
            raise PITDataError(f"unresolved lifecycle evidence for {iid!r}")
    actions_map = {session: tuple(sorted(actions, key=lambda a: (a.instrument_id, a.action_id))) for session, actions in by_session.items()}
    return CorporateActionCoverage(actions_by_session=actions_map, research_returns_by_key={})


def build_backtest_sessions(
    *,
    snapshot_repository: PITSnapshotRepository,
    calendar: SessionCalendar,
    start: datetime,
    end: datetime,
    decision_time_of: Callable[[datetime], datetime],
    security_master: pl.DataFrame | None = None,
    corporate_actions: pl.DataFrame | None = None,
    lifecycle_events: pl.DataFrame | None = None,
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
    lifecycle_cleanup_keys: frozenset[tuple[str, datetime]] = frozenset()
    lifecycle_coverage: CorporateActionCoverage | None = None
    if lifecycle_events is not None and lifecycle_events.height > 0:  # pragma: no cover - lifecycle merge is exercised via resolve unit test
        lifecycle_coverage = resolve_backtest_lifecycle_evidence(daily_market=full, lifecycle_events=lifecycle_events, calendar=calendar, decision_time_of=decision_time_of)
        cleanup_keys: set[tuple[str, datetime]] = set()
        for lifecycle_row in lifecycle_events.to_dicts():
            if str(lifecycle_row.get("evidence_status", "")) != "verified":
                continue
            iid_key = str(lifecycle_row.get("instrument_id", ""))
            start_raw = lifecycle_row.get("cleanup_start")
            end_raw = lifecycle_row.get("cleanup_end")
            avail_raw = lifecycle_row.get("available_at")
            if not isinstance(start_raw, datetime) or not isinstance(end_raw, datetime):
                continue
            if not isinstance(avail_raw, datetime) or avail_raw.tzinfo is None:
                continue
            for candidate_session in calendar.sessions:
                if start_raw <= candidate_session <= end_raw and avail_raw < decision_time_of(candidate_session):
                    cleanup_keys.add((iid_key, candidate_session))
        lifecycle_cleanup_keys = frozenset(cleanup_keys)
    coverage = validate_corporate_action_coverage(daily_market=full, corporate_actions=corporate_actions, calendar=calendar, decision_time_of=decision_time_of, policy=policy, lifecycle_cleanup_keys=lifecycle_cleanup_keys)
    if lifecycle_coverage is not None:  # pragma: no cover - lifecycle merge is exercised via resolve unit test
        _merged: dict[datetime, list[LedgerCorporateAction]] = {session: list(actions) for session, actions in coverage.actions_by_session.items()}
        for _session, _actions in lifecycle_coverage.actions_by_session.items():
            _merged.setdefault(_session, []).extend(_actions)
        coverage = CorporateActionCoverage(
            actions_by_session={session: tuple(sorted(actions, key=lambda a: (a.effective_time, a.action_id))) for session, actions in _merged.items()},
            research_returns_by_key={**coverage.research_returns_by_key, **lifecycle_coverage.research_returns_by_key},
        )
    adtv_map, vol_map, market_vol_map = _rolling_inputs(full, policy, ordered, coverage.research_returns_by_key)
    for session_open in decisions:
        decision_time = decision_time_of(session_open)
        part = by_session[session_open]
        master_ids: list[str] = []
        for raw_iid in part.get_column("instrument_id").to_list():
            iid = str(raw_iid)
            rows = master_index.get(iid, ())
            if any(
                isinstance(row.get("available_at"), datetime)
                and row["available_at"] <= decision_time
                and _coerce_session(row.get("valid_from", session_open)) <= session_open
                and session_open <= _coerce_session(row.get("valid_to", session_open))
                for row in rows
            ):
                master_ids.append(iid)
        by_session[session_open] = (
            part.filter(pl.col("instrument_id").is_in(master_ids))
            if master_ids
            else part.clear()
        )
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
    # Warm-up is an instrument-level requirement.  A newly listed or sparse
    # symbol must not invalidate otherwise usable symbols in the same session;
    # retain only bars with complete rolling inputs and fail only when a
    # requested session has no usable instruments at all.
    for session_open in decisions:
        part = by_session[session_open]
        if part.is_empty():
            continue
        usable = [
            iid
            for iid in part.get_column("instrument_id").to_list()
            if (session_open, str(iid)) in adtv_map and (session_open, str(iid)) in vol_map
        ]
        if not usable:
            raise PITDataError("insufficient rolling PIT market history")
        by_session[session_open] = part.filter(pl.col("instrument_id").is_in(usable))
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
