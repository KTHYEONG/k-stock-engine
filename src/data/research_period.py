"""Research evaluation period policy for QVEF-driven strategies.

The earliest decision date a QVEF strategy can be evaluated on is bounded by the
first fiscal year OpenDART ``fnlttSinglAcntAll`` serves (FY2015) plus the
fundamental lookback: a trailing-twelve-month window (4 quarters) and the
year-over-year earnings-momentum comparison quarter (latest - 4), i.e. 5 quarters.
Evaluation is split into a development segment used for research iteration and a
holdout segment reserved for the final out-of-sample check.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from src.core.ledger import LedgerNav
from src.data.dart_backfill import QVEF_FUNDAMENTAL_LOOKBACK_QUARTERS, _publication_cutoff
from src.validation.metrics import calculate_ledger_metrics

__all__ = [
    "OPENDART_FIRST_FISCAL_YEAR",
    "ResearchPeriodPolicy",
    "earliest_evaluable_decision_date",
    "summarize_research_segments",
]

# OpenDART 단일회사 전체 재무제표 API bsns_year 제공 하한 (공식 개발가이드)
OPENDART_FIRST_FISCAL_YEAR = 2015
_KRX_TZ = ZoneInfo("Asia/Seoul")


def earliest_evaluable_decision_date(
    *,
    first_fiscal_year: int = OPENDART_FIRST_FISCAL_YEAR,
    lookback_quarters: int = QVEF_FUNDAMENTAL_LOOKBACK_QUARTERS,
) -> date:
    """Return the first date whose latest available quarter has a full fundamental lookback.

    Raises:
        ValueError: If first_fiscal_year or lookback_quarters is invalid.
    """
    if isinstance(first_fiscal_year, bool) or not isinstance(first_fiscal_year, int) or not 2000 <= first_fiscal_year <= 2100:
        raise ValueError(f"invalid first_fiscal_year {first_fiscal_year!r}")
    if isinstance(lookback_quarters, bool) or not isinstance(lookback_quarters, int) or lookback_quarters < 1:
        raise ValueError(f"invalid lookback_quarters {lookback_quarters!r}")
    # 하한 분기(FYQ1)에서 lookback-1 분기 뒤가 전체 룩백을 갖는 첫 '최신 분기'
    total = first_fiscal_year * 4 + (lookback_quarters - 1)
    latest = f"{total // 4}Q{total % 4 + 1}"
    return _publication_cutoff(latest) + timedelta(days=1)


@dataclass(frozen=True, slots=True)
class ResearchPeriodPolicy:
    first_fiscal_year: int = OPENDART_FIRST_FISCAL_YEAR
    evaluation_start: date = date(2017, 4, 1)
    development_end: date = date(2023, 12, 31)
    holdout_start: date = date(2024, 1, 1)

    def __post_init__(self) -> None:
        if isinstance(self.first_fiscal_year, bool) or not isinstance(self.first_fiscal_year, int) or self.first_fiscal_year < OPENDART_FIRST_FISCAL_YEAR:
            raise ValueError(f"invalid first_fiscal_year {self.first_fiscal_year!r}: OpenDART serves FY{OPENDART_FIRST_FISCAL_YEAR}+")
        floor = earliest_evaluable_decision_date(first_fiscal_year=self.first_fiscal_year)
        if self.evaluation_start < floor:
            raise ValueError(f"invalid evaluation_start {self.evaluation_start}: fundamentals lookback requires >= {floor}")
        if self.development_end < self.evaluation_start:
            raise ValueError("invalid development_end: must be on or after evaluation_start")
        if self.holdout_start != self.development_end + timedelta(days=1):
            raise ValueError("invalid holdout_start: must immediately follow development_end")


def _segment_summary(marks: tuple[LedgerNav, ...], in_segment: int) -> dict[str, Any]:
    metrics = calculate_ledger_metrics(marks)
    inside = marks[len(marks) - in_segment :]
    yearly: dict[str, float] = {}
    # 연도별 수익률은 직전 연도 마지막 NAV 기준으로 연쇄해 연초 첫 세션 수익을 누락하지 않음
    base = float(marks[0].nav)
    last_nav = base
    current_year: int | None = None
    for mark in inside:
        year = mark.as_of.astimezone(_KRX_TZ).year
        if current_year is not None and year != current_year:
            base = last_nav
        yearly[str(year)] = float(mark.nav) / base - 1.0
        last_nav = float(mark.nav)
        current_year = year
    return {
        "start": inside[0].as_of.astimezone(_KRX_TZ).date().isoformat(),
        "end": inside[-1].as_of.astimezone(_KRX_TZ).date().isoformat(),
        "sessions": in_segment,
        "total_return": float(marks[-1].nav) / float(marks[0].nav) - 1.0,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "annualized_volatility": metrics.annualized_volatility,
        "yearly_returns": yearly,
    }


def summarize_research_segments(
    daily_nav: tuple[LedgerNav, ...],
    *,
    policy: ResearchPeriodPolicy | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize NAV performance separately for the development and holdout segments.

    Each segment is chained from the last mark before its start so the first in-segment
    session return is included. Segments with no in-segment marks are omitted.
    """
    resolved = policy if policy is not None else ResearchPeriodPolicy()
    bounds = {
        "development": (resolved.evaluation_start, resolved.development_end),
        "holdout": (resolved.holdout_start, date.max),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, (start, end) in bounds.items():
        before: LedgerNav | None = None
        inside: list[LedgerNav] = []
        for mark in daily_nav:
            day = mark.as_of.astimezone(_KRX_TZ).date()
            if day < start:
                before = mark
            elif day <= end:
                inside.append(mark)
        if not inside:
            continue
        marks = ((before,) if before is not None else ()) + tuple(inside)
        if len(marks) < 2:
            continue
        out[name] = _segment_summary(marks, len(inside))
    return out
