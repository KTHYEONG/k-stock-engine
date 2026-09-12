"""Single-pass backtest exclusion plan for corporate actions and terminal closes."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from src.core.time import KRX_TZ, SessionCalendar
from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence
from src.data.schemas import PITDataError

MISSING_TERMINAL_CLOSE_REASON = "missing_terminal_close"


@dataclass(frozen=True, slots=True)
class BacktestExclusionPlan:
    """Structured single-prepass exclusion decision for a backtest run.

    Attributes:
        eligible_daily_market: Daily bars with every excluded instrument removed.
        corporate_action_instruments: Instruments quarantined by CA evidence.
        missing_terminal_close_instruments: Instruments lacking a terminal close.
        reasons: Exclusion labels by instrument.
        quarantine_sessions_by_instrument: Quarantined sessions by instrument.
    """

    eligible_daily_market: pl.DataFrame
    corporate_action_instruments: frozenset[str]
    missing_terminal_close_instruments: frozenset[str]
    reasons: Mapping[str, tuple[str, ...]]
    quarantine_sessions_by_instrument: Mapping[str, tuple[datetime, ...]]

    @property
    def excluded_instruments(self) -> frozenset[str]:
        """Return the union of every excluded instrument."""
        return self.corporate_action_instruments | self.missing_terminal_close_instruments

    def reason_counts(self) -> dict[str, int]:
        """Count excluded instruments by reason label."""
        counts: dict[str, int] = {}
        for labels in self.reasons.values():
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        return counts


def resolve_backtest_exclusion_plan(
    *,
    daily_market: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    calendar: SessionCalendar,
    policy: BacktestMarketInputsPolicy,
    terminal_session: datetime,
) -> BacktestExclusionPlan:
    """Resolve every backtest exclusion in one structured prepass.

    The two-stage CLI decision runs here in its original order: corporate-action
    quarantine first, then the terminal-close check on the post-CA frame so a
    CA-excluded instrument is never double-counted.

    Args:
        daily_market: Certified daily bars for the coverage window.
        corporate_actions: Certified corporate-action evidence.
        calendar: Certified session calendar.
        policy: Market-input policy for CA evidence resolution.
        terminal_session: Session whose KRX date bounds the close check.

    Returns:
        Plan with the eligible frame and the disjoint exclusion sets.

    Raises:
        PITDataError: If the market frame is empty or the terminal session
            falls outside the certified calendar.
    """
    if daily_market.height == 0:
        raise PITDataError("daily market must be non-empty")
    terminal_date = terminal_session.astimezone(KRX_TZ).date()
    calendar_dates = {session.astimezone(KRX_TZ).date() for session in calendar.sessions}
    if terminal_date not in calendar_dates:
        raise PITDataError("terminal session is outside the certified calendar")
    resolution = resolve_backtest_corporate_action_evidence(
        daily_market=daily_market,
        corporate_actions=corporate_actions,
        calendar=calendar,
        policy=policy,
    )
    # CA 제외 종목을 먼저 확정한다 (현행 순서 등가).
    corporate_action_instruments = frozenset(resolution.exclusion_reasons)
    if corporate_action_instruments:
        post_ca_market = resolution.eligible_daily_market.filter(
            ~pl.col("instrument_id").is_in(sorted(corporate_action_instruments))
        )
    else:
        post_ca_market = resolution.eligible_daily_market
    # terminal-close 판정은 CA 적용 후 프레임에서 계산해 중복 계상을 막는다.
    last_bars = post_ca_market.group_by("instrument_id").agg(pl.col("session").max().alias("_last_session"))
    missing: set[str] = set()
    for row in last_bars.to_dicts():
        last_session = row["_last_session"]
        if last_session.astimezone(KRX_TZ).date() < terminal_date:
            missing.add(str(row["instrument_id"]))
    missing_terminal_close_instruments = frozenset(missing)
    if corporate_action_instruments or missing_terminal_close_instruments:
        eligible_daily_market = post_ca_market.filter(
            ~pl.col("instrument_id").is_in(sorted(corporate_action_instruments | missing_terminal_close_instruments))
        )
    else:
        eligible_daily_market = post_ca_market
    reasons: dict[str, tuple[str, ...]] = {
        instrument_id: tuple(labels) for instrument_id, labels in resolution.exclusion_reasons.items()
    }
    for instrument_id in missing_terminal_close_instruments:
        reasons[instrument_id] = (MISSING_TERMINAL_CLOSE_REASON,)
    return BacktestExclusionPlan(
        eligible_daily_market=eligible_daily_market,
        corporate_action_instruments=corporate_action_instruments,
        missing_terminal_close_instruments=missing_terminal_close_instruments,
        reasons=reasons,
        quarantine_sessions_by_instrument=dict(resolution.quarantine_sessions_by_instrument),
    )
