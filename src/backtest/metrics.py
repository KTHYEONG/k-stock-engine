"""Flow-neutral performance summaries over engine results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from src.backtest.engine import BacktestResult
from src.backtest.ledger import JournalKind
from src.backtest.market import MarketArrays


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    log_growth_annualized: float
    log_growth_by_year: Mapping[int, float]
    twr_total: float
    mwr_annualized: float | None
    final_nav: int
    total_external_flow: int
    turnover_annualized: float
    cost_drag_annualized: float
    reject_counts: Mapping[str, int]
    max_participation: float
    price_return_only: bool


def _xirr(cashflows: list[tuple[date, float]]) -> float | None:
    base = cashflows[0][0]
    terms = [((day - base).days / 365.0, amount) for day, amount in cashflows]

    def npv(rate: float) -> float:
        return sum(amount / ((1.0 + rate) ** years) for years, amount in terms)

    low, high = -0.999999, 1.0
    for _ in range(100):
        if npv(high) < 0.0:
            break
        high = high * 2.0 + 1.0
    else:
        return None
    for _ in range(200):
        mid = (low + high) / 2.0
        if npv(mid) > 0.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def summarize(
    *, result: BacktestResult, arrays: MarketArrays, sessions_per_year: int
) -> PerformanceSummary:
    """Summarize a run with flow-neutral growth as the primary metric.

    Daily TWR return is ``(nav_t − flow_t) / nav_{t−1} − 1`` with flows booked
    pre-open, so a deposit alone produces zero return; log growth is the mean
    of ``ln(1 + r)`` scaled by ``sessions_per_year``. Turnover and cost drag
    are mean daily traded notionals and explicit costs (commission plus sell
    tax) over prior NAV, scaled the same way. Participation of a fill is its
    notional over decision-session adtv20.
    """
    if (
        isinstance(sessions_per_year, bool)
        or not isinstance(sessions_per_year, int)
        or sessions_per_year <= 0
    ):
        raise ValueError(f"sessions_per_year must be a positive int, got {sessions_per_year!r}")
    records = list(result.nav)
    dates = [arrays.sessions[record.session_idx] for record in records]
    dated_rets: list[tuple[date, float]] = []
    prev_nav: int | None = None
    prev_ext = 0
    for record, session_date in zip(records, dates):
        flow = record.external_flow - prev_ext
        prev_ext = record.external_flow
        if prev_nav is not None and prev_nav != 0:
            dated_rets.append((session_date, (record.nav - flow) / prev_nav - 1.0))
        prev_nav = record.nav
    growth = [math.log1p(ret) for _, ret in dated_rets if 1.0 + ret > 0.0]
    log_growth_annualized = sum(growth) / len(growth) * sessions_per_year if growth else 0.0
    by_year: dict[int, list[float]] = {}
    for session_date, ret in dated_rets:
        if 1.0 + ret > 0.0:
            by_year.setdefault(session_date.year, []).append(math.log1p(ret))
    log_growth_by_year = {
        year: sum(values) / len(values) * sessions_per_year
        for year, values in sorted(by_year.items())
    }
    twr_total = math.prod([1.0 + ret for _, ret in dated_rets], start=1.0) - 1.0
    cashflows: list[tuple[date, float]] = []
    prev_ext = 0
    for record, session_date in zip(records, dates):
        flow = record.external_flow - prev_ext
        prev_ext = record.external_flow
        if flow:
            cashflows.append((session_date, -float(flow)))
    if records:
        cashflows.append((dates[-1], float(records[-1].nav)))
    if not any(amount > 0 for _, amount in cashflows) or not any(
        amount < 0 for _, amount in cashflows
    ):
        mwr_annualized: float | None = None
    else:
        mwr_annualized = _xirr(cashflows)
    nav_by_session = {record.session_idx: record.nav for record in records}
    ordered_sessions = sorted(nav_by_session)
    base_nav = {
        session_idx: nav_by_session[ordered_sessions[pos - 1]] if pos > 0 else 0
        for pos, session_idx in enumerate(ordered_sessions)
    }
    traded: dict[int, float] = {}
    spent: dict[int, float] = {}
    for entry in result.journal:
        if entry.kind is JournalKind.BUY or entry.kind is JournalKind.SELL:
            traded[entry.session_idx] = traded.get(entry.session_idx, 0.0) + abs(entry.cash_delta)
        elif entry.kind is JournalKind.COMMISSION or entry.kind is JournalKind.SELL_TAX:
            spent[entry.session_idx] = spent.get(entry.session_idx, 0.0) + abs(entry.cash_delta)
    daily_turnover = [
        traded.get(session_idx, 0.0) / base_nav[session_idx]
        for session_idx in ordered_sessions
        if base_nav[session_idx]
    ]
    daily_cost = [
        spent.get(session_idx, 0.0) / base_nav[session_idx]
        for session_idx in ordered_sessions
        if base_nav[session_idx]
    ]
    turnover_annualized = (
        sum(daily_turnover) / len(daily_turnover) * sessions_per_year if daily_turnover else 0.0
    )
    cost_drag_annualized = (
        sum(daily_cost) / len(daily_cost) * sessions_per_year if daily_cost else 0.0
    )
    reject_counts: dict[str, int] = {}
    for reject in result.rejects:
        reject_counts[reject.reason] = reject_counts.get(reject.reason, 0) + 1
    adtv = arrays.float_fields["adtv20"]
    peak = 0.0
    for fill in result.fills:
        session_idx = fill.order.decision_session_idx
        peak = max(peak, (fill.quantity * fill.price) / float(adtv[session_idx, fill.order.instrument_idx]))
    return PerformanceSummary(
        log_growth_annualized=log_growth_annualized,
        log_growth_by_year=log_growth_by_year,
        twr_total=twr_total,
        mwr_annualized=mwr_annualized,
        final_nav=records[-1].nav,
        total_external_flow=records[-1].external_flow,
        turnover_annualized=turnover_annualized,
        cost_drag_annualized=cost_drag_annualized,
        reject_counts=reject_counts,
        max_participation=peak,
        price_return_only=not result.dividends_integrated,
    )
