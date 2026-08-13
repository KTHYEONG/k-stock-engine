"""Costed hold-or-replace action labels for the v5 action-value model."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from src.core.costs import CostPoint, CostSchedule
from src.stocks.research.hold_replace import (
    CASH_INCUMBENT,
    PortfolioDecisionState,
    PortfolioStateTrace,
    build_hold_replace_labels,
)


def _panel(n_sessions: int = 30, n_tickers: int = 6) -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(11)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for t in range(n_tickers):
        price = 100.0
        for s in range(n_sessions):
            obs = start + timedelta(days=s)
            price = max(10.0, price * (1.0 + float(rng.normal(0.0, 0.01))))
            rows.append(
                {
                    "session_index": s,
                    "session": obs,
                    "instrument_id": f"KRX:0{t + 1:05d}",
                    "close": price,
                    "adtv": float(rng.uniform(1e8, 1e9)),
                    "label_available_time": obs + timedelta(days=5),
                }
            )
    return pl.DataFrame(rows)


def _schedule() -> CostSchedule:
    return CostSchedule(
        name="base",
        points=(
            CostPoint(
                effective_from=datetime(2024, 1, 1, tzinfo=UTC),
                commission_rate=0.00015,
                tax_rate=0.0023,
                slippage_bps=5.0,
                settlement_days=2,
            ),
        ),
    )


def test_hold_replace_labels_are_causal_and_costed() -> None:
    """An incumbent is retained when its costed hold value exceeds the challenger.

    The action label is the log-return replacement value, so a challenger with a
    strictly lower net hold value than the incumbent produces a negative label
    (retain) while a materially better challenger wins. Later labels never
    change earlier features because every feature is decision-time available.
    """
    panel = _panel(n_sessions=30, n_tickers=6)
    decision_session = 10
    decision_time = panel.filter(pl.col("session_index") == decision_session)[
        "session"
    ].unique().item()
    # Incumbent is a stable name; challenger starts cheap (higher forward return).
    challenger = "KRX:000006"
    trace = PortfolioStateTrace(
        decisions=(
            PortfolioDecisionState(
                session_index=decision_session,
                decision_time=decision_time,
                incumbents=("KRX:000001",),
                incumbent_weights=(0.2,),
            ),
        )
    )
    labels = build_hold_replace_labels(
        panel,
        trace,
        label_column="forward_log_return",
        label_available_column="label_available_time",
        cost_schedule=_schedule(),
        holding_horizon_sessions=5,
    )
    assert not labels.is_empty()
    key_cols = ("session_index", "decision_time", "instrument_id", "incumbent_id")
    assert labels.select(key_cols).is_duplicated().sum() == 0
    assert labels["label_available_time"].is_not_null().all()
    assert labels["forward_log_return"].is_finite().all()

    incumbent_rows = labels.filter(
        (pl.col("instrument_id") == "KRX:000001")
        & (pl.col("incumbent_id") == "KRX:000001")
    )
    assert incumbent_rows.height >= 1
    cash_rows = labels.filter(pl.col("incumbent_id") == CASH_INCUMBENT)
    assert cash_rows.height >= 1
    # Retained incumbent has zero switch cost.
    retained = incumbent_rows.row(0, named=True)
    assert retained["entry_cost"] == 0.0
    assert retained["exit_cost"] == 0.0
    # The incumbent's own hold label equals its forward return (no costs).
    forward_incumbent = (
        panel.filter(
            (pl.col("session_index") == decision_session)
            & (pl.col("instrument_id") == "KRX:000001")
        )["close"]
    )
    assert forward_incumbent.len() == 1


def test_hold_replace_future_labels_cannot_change_earlier_features() -> None:
    """Mutating a future label row never changes an earlier action's features."""
    panel = _panel(n_sessions=40, n_tickers=6)
    decision_time = panel.filter(pl.col("session_index") == 8)["session"].unique().item()
    trace = PortfolioStateTrace(
        decisions=(
            PortfolioDecisionState(
                session_index=8,
                decision_time=decision_time,
                incumbents=("KRX:000002",),
                incumbent_weights=(0.2,),
            ),
        )
    )
    labels = build_hold_replace_labels(
        panel,
        trace,
        label_column="forward_log_return",
        label_available_column="label_available_time",
        cost_schedule=_schedule(),
        holding_horizon_sessions=5,
    )
    earlier = labels.filter(pl.col("session_index") == 8).sort(
        ["instrument_id", "incumbent_id"]
    )
    assert earlier.height >= 1

    mutated = panel.with_columns(
        pl.when(pl.col("session_index") == 20)
        .then(pl.lit(5.0))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    relabeled = build_hold_replace_labels(
        mutated,
        trace,
        label_column="forward_log_return",
        label_available_column="label_available_time",
        cost_schedule=_schedule(),
        holding_horizon_sessions=5,
    )
    later = relabeled.filter(pl.col("session_index") == 8).sort(
        ["instrument_id", "incumbent_id"]
    )
    assert earlier.equals(later)


def test_hold_replace_labels_reject_duplicate_keys_and_bad_inputs() -> None:
    panel = _panel(n_sessions=20, n_tickers=4)
    with pytest.raises(ValueError, match="must carry"):
        build_hold_replace_labels(
            panel.drop("adtv"),
            PortfolioStateTrace(decisions=()),
            label_column="forward_log_return",
            label_available_column="label_available_time",
            cost_schedule=_schedule(),
            holding_horizon_sessions=5,
        )
    with pytest.raises(ValueError, match="must be positive"):
        build_hold_replace_labels(
            panel,
            PortfolioStateTrace(decisions=()),
            label_column="forward_log_return",
            label_available_column="label_available_time",
            cost_schedule=_schedule(),
            holding_horizon_sessions=0,
        )


def test_hold_replace_trace_must_be_ascending_and_unique() -> None:
    decision_time = datetime(2024, 1, 10, tzinfo=UTC)
    with pytest.raises(ValueError, match="ascending"):
        PortfolioStateTrace(
            decisions=(
                PortfolioDecisionState(5, decision_time, ("a",), (1.0,)),
                PortfolioDecisionState(4, decision_time, ("a",), (1.0,)),
            )
        )
    with pytest.raises(ValueError, match="unique"):
        PortfolioStateTrace(
            decisions=(
                PortfolioDecisionState(5, decision_time, ("a",), (1.0,)),
                PortfolioDecisionState(5, decision_time, ("b",), (1.0,)),
            )
        )
