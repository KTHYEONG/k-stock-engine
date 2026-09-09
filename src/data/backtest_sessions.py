"""PIT backtest session builder from certified Silver snapshots."""
from __future__ import annotations

import math
from collections.abc import Callable
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


_ALLOWED_ACTION_TYPES = frozenset({"split", "reverse_split", "dividend"})


def _coerce_session(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def validate_corporate_action_coverage(
    *,
    daily_market: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    calendar: SessionCalendar,
    decision_time_of: Callable[[datetime], datetime],
    policy: BacktestMarketInputsPolicy,
) -> dict[datetime, tuple[LedgerCorporateAction, ...]]:
    threshold = float(policy.unexplained_price_jump_threshold)
    action_rows: list[dict[str, Any]] = corporate_actions.to_dicts() if corporate_actions.height > 0 else []
    seen_keys: set[tuple[str, str, str]] = set()
    by_session: dict[datetime, list[LedgerCorporateAction]] = {}
    for row in action_rows:
        raw_type = str(row.get("action_type", row.get("type", "")))
        if raw_type in ("no_action", "", "none"):
            continue
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
        if decision_time.tzinfo is None or avail > session_open or avail > decision_time:
            raise PITDataError(f"late corporate action for {iid!r}")
        action_type = LedgerActionType(raw_type)
        factor = float(row.get("factor", 2.0 if raw_type != "dividend" else 1.0))
        cash = float(row.get("cash_amount", 0.0))
        action = LedgerCorporateAction(
            action_id=str(row.get("action_id", f"{iid}:{raw_type}:{eff.isoformat()}")),
            instrument_id=iid,
            action_type=action_type,
            effective_time=eff,
            factor=factor,
            cash_amount=cash,
        )
        by_session.setdefault(eff, []).append(action)
    bar_rows = daily_market.to_dicts()
    by_instrument: dict[str, list[tuple[datetime, float]]] = {}
    for row in bar_rows:
        iid = str(row["instrument_id"])
        by_instrument.setdefault(iid, []).append((_coerce_session(row["session"]), float(row["close"])))
    split_cover: set[tuple[str, str]] = set()
    for (iid, eff_iso, raw_type) in seen_keys:
        if raw_type in ("split", "reverse_split"):
            split_cover.add((iid, eff_iso))
    for iid, points in by_instrument.items():
        ordered = sorted(points, key=lambda p: p[0])
        for prev, curr in pairwise(ordered):
            prev_close = prev[1]
            if prev_close <= 0:
                continue
            jump = abs(curr[1] / prev_close - 1.0)
            if jump > threshold and (iid, curr[0].isoformat()) not in split_cover:
                raise PITDataError(f"unexplained price discontinuity for {iid!r}")
    return {session: tuple(actions) for session, actions in by_session.items()}


def _frame_for(repository: PITSnapshotRepository) -> pl.DataFrame | None:
    frames = getattr(repository, "_frames", {})
    frame = frames.get(SilverTable.DAILY_MARKET)
    return frame


def _rolling_inputs(
    full: pl.DataFrame,
    policy: BacktestMarketInputsPolicy,
    session_calendar: tuple[datetime, ...],
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
                returns = [
                    points[volatility_sessions[pos]][0] / points[volatility_sessions[pos - 1]][0] - 1.0
                    for pos in range(1, len(volatility_sessions))
                ]
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
        for points in history.values():
            prior = points.get(previous)
            current = points.get(session)
            if prior is not None and current is not None:
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
) -> str:
    if security_master is None or security_master.is_empty():
        raise PITDataError("missing PIT security master")
    candidates = [
        row
        for row in security_master.to_dicts()
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
    adtv_map, vol_map, market_vol_map = _rolling_inputs(full, policy, ordered)
    action_map = validate_corporate_action_coverage(
        daily_market=full.select(["session", "instrument_id", "close", "available_at"]),
        corporate_actions=corporate_actions,
        calendar=calendar,
        decision_time_of=decision_time_of,
        policy=policy,
    )
    for session_open in decisions:
        decision_time = decision_time_of(session_open)
        for instrument_id in by_session[session_open].get_column("instrument_id").to_list():
            _resolve_sector(
                security_master=security_master,
                instrument_id=str(instrument_id),
                session=session_open,
                decision_time=decision_time,
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
                actions=action_map.get(session_open, ()),
                market_snapshot=market_snapshot,
            )
        )
    return tuple(sessions)
