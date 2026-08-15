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
    session = datetime(2024, 1, 2, tzinfo=UTC)
    liquidity = stock_liquidity_model()
    scored = _scored(
        [("KRX:00001", session, 0.05), ("KRX:00002", session, 0.04)]
    )
    residual_1, residual_2 = 0.03, 0.02
    realized = _realized(
        [
            ("KRX:00001", session, residual_1, 100.0, 1.0e8, 0.02),
            ("KRX:00002", session, residual_2, 100.0, 1.0e8, 0.02),
        ]
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK,
        cost_schedule=default_base_schedule(),
        liquidity_model=liquidity,
    ).evaluate(scored, realized)
    assert len(evaluation.blocks) == 1
    point = default_base_schedule().cost_for(session)
    expected_costs: list[float] = []
    for order in evaluation.orders:
        slippage_bps = liquidity.slippage_bps(
            notional=order.order_size,
            adtv_20d=1.0e8,
            daily_volatility=0.02,
            reference_price=100.0,
            effective_time=session,
        )
        expected_costs.append(
            2.0 * point.commission_rate + point.tax_rate + 2.0 * slippage_bps / 10_000.0
        )
    expected_growth = (residual_1 - expected_costs[0] + residual_2 - expected_costs[1]) / 2.0
    assert evaluation.blocks[0].block_log_excess == pytest.approx(expected_growth, abs=1e-12)


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
