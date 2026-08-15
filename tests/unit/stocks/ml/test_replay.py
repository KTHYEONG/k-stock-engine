"""NetAlphaPolicyReplay canonical schema and decimal economics tests."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from src.core.costs import default_base_schedule
from src.stocks.ml.contracts import PortfolioSettings, RiskSettings
from src.stocks.ml.models import SCORE_COLUMN
from src.stocks.ml.replay import NetAlphaPolicyReplay
from tests.fixtures.stocks.helpers import stock_liquidity_model

_PORTFOLIO = PortfolioSettings()
_RISK = RiskSettings()


def _scored(
    rows: list[tuple[str, object, float]],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": [r[0] for r in rows],
            "session": [r[1] for r in rows],
            SCORE_COLUMN: [r[2] for r in rows],
        }
    )


def _realized(
    rows: list[tuple[str, object, float, float, float, float]],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": [r[0] for r in rows],
            "session": [r[1] for r in rows],
            "risk_residual": [r[2] for r in rows],
            "open": [r[3] for r in rows],
            "adtv_20d": [r[4] for r in rows],
            "volatility_20d": [r[5] for r in rows],
        }
    )


def test_evaluate_empty_scored_panel_returns_no_blocks() -> None:
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK
    ).evaluate(
        pl.DataFrame(
            {
                "instrument_id": [],
                "session": [],
                "predicted_net_alpha": [],
            }
        )
    )
    assert evaluation.blocks == ()
    assert evaluation.orders == ()


def test_evaluate_realized_none_is_score_only_planning() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored(
        [("KRX:00001", session, 0.02), ("KRX:00002", session, 0.01)]
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    ).evaluate(scored, realized=None)
    assert len(evaluation.orders) == 2
    assert evaluation.blocks == ()


def test_evaluate_rejects_missing_realized_columns() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored([("KRX:00001", session, 0.02)])
    broken = _realized([("KRX:00001", session, 0.05, 100.0, 1.0e8, 0.02)]).drop(
        "adtv_20d"
    )
    with pytest.raises(ValueError, match="realized frame missing canonical columns"):
        NetAlphaPolicyReplay(
            3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
        ).evaluate(scored, broken)


def test_evaluate_rejects_non_finite_realized() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored([("KRX:00001", session, 0.02)])
    bad = _realized([("KRX:00001", session, float("nan"), 100.0, 1.0e8, 0.02)])
    with pytest.raises(ValueError, match="non-finite outcome/cost columns"):
        NetAlphaPolicyReplay(
            3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
        ).evaluate(scored, bad)


def test_evaluate_rejects_duplicate_realized_keys() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored([("KRX:00001", session, 0.02)])
    dup = _realized(
        [
            ("KRX:00001", session, 0.05, 100.0, 1.0e8, 0.02),
            ("KRX:00001", session, 0.06, 100.0, 1.0e8, 0.02),
        ]
    )
    with pytest.raises(ValueError, match="duplicate instrument/session keys"):
        NetAlphaPolicyReplay(
            3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
        ).evaluate(scored, dup)


def test_evaluate_sorts_descending_with_instrument_id_tiebreak() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored(
        [
            ("KRX:00002", session, 0.01),
            ("KRX:00001", session, 0.03),
            ("KRX:00003", session, 0.03),
            ("KRX:00004", session, 0.02),
        ]
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    ).evaluate(scored, realized=None)
    instruments = [order.instrument_id for order in evaluation.orders]
    assert instruments == ["KRX:00001", "KRX:00003", "KRX:00004", "KRX:00002"]


def test_evaluate_zero_weight_cross_section_creates_no_orders() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    below_band = 0.0004
    scored = _scored(
        [
            ("KRX:00001", session, below_band),
            ("KRX:00002", session, below_band),
        ]
    )
    realized = _realized(
        [
            ("KRX:00001", session, 0.05, 100.0, 1.0e8, 0.02),
            ("KRX:00002", session, 0.05, 100.0, 1.0e8, 0.02),
        ]
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    ).evaluate(scored, realized)
    assert evaluation.orders == ()
    assert evaluation.blocks == ()
    assert sum(order.order_size for order in evaluation.orders) == 0.0


def test_evaluate_block_growth_uses_decimal_dynamic_cost() -> None:
    liquidity = stock_liquidity_model()
    sessions = [
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC),
    ]
    scored = _scored(
        [
            (f"KRX:{i:05d}", session, 0.05)
            for i, session in enumerate(sessions)
        ]
    )
    realized = _realized(
        [
            (f"KRX:{i:05d}", session, 0.03, 100.0, 1.0e8, 0.02)
            for i, session in enumerate(sessions)
        ]
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK,
        cost_schedule=default_base_schedule(),
        liquidity_model=liquidity,
    ).evaluate(scored, realized)
    assert len(evaluation.blocks) == 1
    assert evaluation.period_count == 1
    assert evaluation.active_cohort_count == 1
    assert len(evaluation.orders) == 3
    point = default_base_schedule().cost_for(sessions[0])
    expected_costs: list[float] = []
    for order in evaluation.orders:
        slippage_bps = liquidity.slippage_bps(
            notional=order.order_size,
            adtv_20d=1.0e8,
            daily_volatility=0.02,
            reference_price=100.0,
            effective_time=order.decision_session,
        )
        expected_costs.append(
            2.0 * point.commission_rate + point.tax_rate + 2.0 * slippage_bps / 10_000.0
        )
    expected_growth = (
        sum(0.03 - cost for cost in expected_costs) / len(expected_costs)
    )
    assert evaluation.blocks[0].net_return == pytest.approx(expected_growth, abs=1e-12)
    assert evaluation.blocks[0].net_return == evaluation.block_net_returns[0]
    assert evaluation.block_log_excess == evaluation.block_net_returns


def test_evaluate_realized_requires_liquidity_model() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored([("KRX:00001", session, 0.05)])
    realized = _realized([("KRX:00001", session, 0.03, 100.0, 1.0e8, 0.02)])
    with pytest.raises(ValueError, match="realized replay requires a liquidity model"):
        NetAlphaPolicyReplay(3, _PORTFOLIO, _RISK).evaluate(scored, realized)


def test_evaluate_uses_calibrated_lower_bound_as_economic_score() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001", "KRX:00002"],
            "session": [session, session],
            SCORE_COLUMN: [0.5, 0.6],
            "net_alpha_lower_bound": [0.004, 0.002],
        }
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    ).evaluate(scored, realized=None)
    instruments = [order.instrument_id for order in evaluation.orders]
    assert instruments == ["KRX:00001", "KRX:00002"]
    assert np.all([order.predicted_net_alpha > 0.0 for order in evaluation.orders])

def _session(value: str) -> datetime:
    return datetime.fromisoformat(value)


def test_evaluate_period_semantics_cover_complete_cohorts() -> None:
    liquidity = stock_liquidity_model()
    sessions = [_session(f"2024-01-0{i}T00:00:00+00:00") for i in range(1, 6)]
    rows = [
        (f"KRX:{i:05d}", session, 0.05, 0.03, 100.0, 1.0e8, 0.02)
        for i, session in enumerate(sessions)
    ]
    scored = pl.DataFrame(
        {
            "instrument_id": [r[0] for r in rows],
            "session": [r[1] for r in rows],
            SCORE_COLUMN: [r[2] for r in rows],
        }
    )
    realized = pl.DataFrame(
        {
            "instrument_id": [r[0] for r in rows],
            "session": [r[1] for r in rows],
            "risk_residual": [r[3] for r in rows],
            "open": [r[4] for r in rows],
            "adtv_20d": [r[5] for r in rows],
            "volatility_20d": [r[6] for r in rows],
        }
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=liquidity
    ).evaluate(scored, realized)
    assert evaluation.period_count == 1
    assert evaluation.observed_sessions == 3
    assert evaluation.active_cohort_count == 1
    assert evaluation.missing_realized_cohort_count == 0
    assert len(evaluation.period_net_returns) == 1
    assert evaluation.scored_sessions == 5
    assert evaluation.realized_sessions == 5
    assert evaluation.active_sessions == 5
    diagnostics = evaluation.replay_diagnostics()
    assert diagnostics["complete_cohorts"] == 1
    assert diagnostics["active_cohorts"] == 1
    assert diagnostics["missing_realized_cohorts"] == 0
    assert diagnostics["orders"] == len(evaluation.orders)


def test_evaluate_all_cash_cohort_is_observed_zero_return() -> None:
    sessions = [_session(f"2024-01-0{i}T00:00:00+00:00") for i in range(1, 4)]
    scored = _scored(
        [(f"KRX:{i:05d}", session, 0.0002) for i, session in enumerate(sessions)]
    )
    realized = _realized(
        [
            (f"KRX:{i:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for i, session in enumerate(sessions)
        ]
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    ).evaluate(scored, realized)
    assert evaluation.orders == ()
    assert evaluation.blocks == ()
    assert evaluation.period_net_returns == (0.0,)
    assert evaluation.period_count == 1
    assert evaluation.active_cohort_count == 0
    assert evaluation.eligible_sessions == 0
    assert evaluation.observed_sessions == 3


def test_evaluate_missing_realized_cohort_fails_closed_never_zero_filled() -> None:
    sessions = [_session(f"2024-01-0{i}T00:00:00+00:00") for i in range(1, 4)]
    scored = _scored(
        [
            (f"KRX:{instrument:05d}", session, 0.05)
            for session in sessions
            for instrument in range(2)
        ]
    )
    realized = _realized(
        [
            (f"KRX:{0:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for session in sessions
        ]
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    ).evaluate(scored, realized)
    assert evaluation.missing_realized_cohort_count == 1
    assert evaluation.period_net_returns == ()
    assert evaluation.period_count == 0
    assert evaluation.active_cohort_count == 0
    assert evaluation.blocks == ()


def test_evaluate_score_only_planning_has_no_period_evidence() -> None:
    session = _session("2024-01-01T00:00:00+00:00")
    scored = _scored([("KRX:00001", session, 0.05)])
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    ).evaluate(scored, realized=None)
    assert len(evaluation.orders) == 1
    assert evaluation.blocks == ()
    assert evaluation.period_count == 0
    assert evaluation.observed_sessions == 0
    assert evaluation.active_cohort_count == 0
    assert evaluation.period_net_returns == ()


def _segmented_panel() -> tuple[pl.DataFrame, pl.DataFrame, list[datetime]]:
    """Six sessions split into two OOF segments of three, horizon two."""
    sessions = [_session(f"2024-01-0{i}T00:00:00+00:00") for i in range(1, 7)]
    segment_ids = [0, 0, 0, 1, 1, 1]
    scored = pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i:05d}" for i in range(6)],
            "session": sessions,
            "oof_segment_id": segment_ids,
            SCORE_COLUMN: [0.05] * 6,
        }
    )
    realized = pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i:05d}" for i in range(6)],
            "session": sessions,
            "risk_residual": [0.03] * 6,
            "open": [100.0] * 6,
            "adtv_20d": [1.0e8] * 6,
            "volatility_20d": [0.02] * 6,
        }
    )
    return scored, realized, sessions


def test_evaluate_segments_never_share_a_cohort_and_partial_tails_are_counted() -> None:
    scored, realized, _sessions = _segmented_panel()
    evaluation = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    ).evaluate(scored, realized, segment_column="oof_segment_id")
    # Each three-session segment contributes floor(3 / 2) = 1 complete cohort
    # and one trailing partial cohort; the gap never merges segments.
    assert evaluation.period_count == 2
    assert evaluation.partial_cohort_count == 2
    assert evaluation.active_cohort_count == 2
    assert evaluation.missing_realized_cohort_count == 0
    assert evaluation.observed_sessions == 4
    segments = [meta[0] for meta in evaluation.cohort_metadata]
    assert segments == [0, 1]
    for _segment, start, end in evaluation.cohort_metadata:
        assert end - start == 2
    diagnostics = evaluation.replay_diagnostics()
    assert diagnostics["complete_cohorts"] == 2
    assert diagnostics["partial_cohorts"] == 2


def test_evaluate_single_segment_equivalent_without_segment_column() -> None:
    scored, realized, _sessions = _segmented_panel()
    replay = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    )
    segmented = replay.evaluate(scored, realized, segment_column="oof_segment_id")
    # Without the segment column the same six sessions form three complete
    # cohorts of two (positions 0-1, 2-3, 4-5) with no partial tail.
    plain = replay.evaluate(scored.drop("oof_segment_id"), realized)
    assert plain.period_count == 3
    assert plain.partial_cohort_count == 0
    assert segmented.period_count != plain.period_count


def test_evaluate_segment_missing_realized_cohort_fails_closed() -> None:
    scored, realized, _sessions = _segmented_panel()
    # Drop the realized row for one order in segment 1: the complete active
    # cohort in segment 1 must be excluded and counted as missing, never filled.
    missing_instrument = "KRX:00003"
    realized = realized.filter(pl.col("instrument_id") != missing_instrument)
    evaluation = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=stock_liquidity_model()
    ).evaluate(scored, realized, segment_column="oof_segment_id")
    assert evaluation.missing_realized_cohort_count == 1
    assert evaluation.active_cohort_count == 1
    assert len(evaluation.period_net_returns) == 1
    assert len(evaluation.blocks) == 1
    assert evaluation.cohort_metadata == ((0, 0, 2),)
    assert evaluation.partial_cohort_count == 2
