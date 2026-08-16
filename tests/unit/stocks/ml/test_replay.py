"""NetAlphaPolicyReplay vintage/maturity semantics and segment diagnostics."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from src.core.costs import default_base_schedule
from src.stocks.ml.contracts import PortfolioSettings, RiskSettings
from src.stocks.ml.models import SCORE_COLUMN
from src.stocks.ml.replay import NetAlphaPolicyReplay, ReplaySegmentDiagnostic
from tests.fixtures.stocks.helpers import stock_liquidity_model

_PORTFOLIO = PortfolioSettings()
_RISK = RiskSettings()
_LIQUIDITY = stock_liquidity_model()


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


def _session(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _span(n: int, start: str = "2024-01-01T00:00:00+00:00") -> list[datetime]:
    return [
        datetime.fromisoformat(start) + __import__("datetime").timedelta(days=i)
        for i in range(n)
    ]


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
    assert evaluation.segment_diagnostics == ()


def test_evaluate_realized_none_is_score_only_planning() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored(
        [("KRX:00001", session, 0.02), ("KRX:00002", session, 0.01)]
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
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
            3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
        ).evaluate(scored, broken)


def test_evaluate_rejects_non_finite_realized() -> None:
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored([("KRX:00001", session, 0.02)])
    bad = _realized([("KRX:00001", session, float("nan"), 100.0, 1.0e8, 0.02)])
    with pytest.raises(ValueError, match="non-finite outcome/cost columns"):
        NetAlphaPolicyReplay(
            3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
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
            3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
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
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
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
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized)
    assert evaluation.orders == ()
    assert evaluation.blocks == ()


def test_evaluate_lower_bound_only_profile_never_orders_non_positive_lower_bound() -> None:
    """A zero-overlay profile must still never order a name with lower bound <= 0."""
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored(
        [
            ("KRX:00001", session, 0.001),
            ("KRX:00002", session, 0.0),
            ("KRX:00003", session, -0.002),
        ]
    )
    risk = RiskSettings(no_trade_band_bps=0.0)
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, risk, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized=None)
    instruments = [order.instrument_id for order in evaluation.orders]
    assert instruments == ["KRX:00001"]
    assert all(order.predicted_net_alpha > 0.0 for order in evaluation.orders)


def test_evaluate_legacy_5bps_band_preserves_baseline_filter() -> None:
    """The 5-bps overlay filters names below five basis points, as before."""
    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored(
        [
            ("KRX:00001", session, 0.0004),
            ("KRX:00002", session, 0.0006),
            ("KRX:00003", session, 0.0010),
        ]
    )
    risk = RiskSettings(no_trade_band_bps=5.0)
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, risk, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized=None)
    instruments = [order.instrument_id for order in evaluation.orders]
    assert instruments == ["KRX:00003", "KRX:00002"]


def test_evaluate_block_growth_uses_decimal_dynamic_cost() -> None:
    sessions = _span(6)
    scored = _scored(
        [(f"KRX:{i:05d}", session, 0.05) for i, session in enumerate(sessions)]
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
        liquidity_model=_LIQUIDITY,
    ).evaluate(scored, realized)
    # Horizon three: the first three vintages mature, the trailing three are
    # partial. Single-name exposure never binds the concurrent cap, so every
    # matured vintage is active.
    assert evaluation.period_count == 3
    assert evaluation.active_cohort_count == 3
    assert evaluation.matured_vintage_count == 3
    assert evaluation.partial_vintage_count == 3
    assert len(evaluation.orders) == 6
    point = default_base_schedule().cost_for(sessions[0])
    expected_growth: list[float] = []
    for order in evaluation.orders:
        slippage_bps = _LIQUIDITY.slippage_bps(
            notional=order.order_size,
            adtv_20d=1.0e8,
            daily_volatility=0.02,
            reference_price=100.0,
            effective_time=order.decision_session,
        )
        expected_growth.append(
            0.03
            - (2.0 * point.commission_rate + point.tax_rate + 2.0 * slippage_bps / 10_000.0)
        )
    assert evaluation.blocks[0].net_return == pytest.approx(
        expected_growth[0], abs=1e-12
    )
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
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized=None)
    instruments = [order.instrument_id for order in evaluation.orders]
    assert instruments == ["KRX:00001", "KRX:00002"]
    assert np.all([order.predicted_net_alpha > 0.0 for order in evaluation.orders])


def test_evaluate_matured_vintage_diagnostics_and_accounting() -> None:
    sessions = _span(6)
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
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized)
    assert evaluation.period_count == 3
    assert evaluation.observed_sessions == 3
    assert evaluation.active_cohort_count == 3
    assert evaluation.missing_realized_vintage_count == 0
    assert evaluation.partial_vintage_count == 3
    assert len(evaluation.period_net_returns) == 3
    assert evaluation.scored_sessions == 6
    assert evaluation.calibration_ready_sessions == 6
    assert evaluation.realized_sessions == 6
    assert evaluation.active_sessions == 6
    diagnostics = evaluation.replay_diagnostics()
    assert diagnostics["matured_vintages"] == 3
    assert diagnostics["active_vintages"] == 3
    assert diagnostics["missing_realized_vintages"] == 0
    assert diagnostics["partial_vintages"] == 3
    assert diagnostics["orders"] == len(evaluation.orders)


def test_evaluate_all_cash_vintages_are_observed_zero_returns() -> None:
    sessions = _span(6)
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
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized)
    assert evaluation.orders == ()
    assert evaluation.blocks == ()
    assert evaluation.period_net_returns == (0.0, 0.0, 0.0)
    assert evaluation.period_count == 3
    assert evaluation.cash_vintage_count == 3
    assert evaluation.matured_vintage_count == 0
    assert evaluation.eligible_sessions == 0
    assert evaluation.observed_sessions == 3
    assert evaluation.partial_vintage_count == 3


def test_evaluate_missing_realized_vintage_fails_closed_never_zero_filled() -> None:
    sessions = _span(6)
    scored = _scored(
        [
            (f"KRX:{pos:05d}", session, 0.05)
            for pos, session in enumerate(sessions)
        ]
    )
    realized = _realized(
        [
            (f"KRX:{pos:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for pos, session in enumerate(sessions)
            if pos != 0
        ]
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized)
    # Session zero's vintage has an order whose realized row is absent; it is
    # excluded and counted as missing, never zero-filled.
    assert evaluation.missing_realized_vintage_count == 1
    assert evaluation.period_count == 2
    assert evaluation.matured_vintage_count == 2
    assert len(evaluation.blocks) == 2
    assert len(evaluation.period_net_returns) == 2


def test_evaluate_score_only_planning_has_no_period_evidence() -> None:
    session = _session("2024-01-01T00:00:00+00:00")
    scored = _scored([("KRX:00001", session, 0.05)])
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized=None)
    assert len(evaluation.orders) == 1
    assert evaluation.blocks == ()
    assert evaluation.period_count == 0
    assert evaluation.observed_sessions == 0
    assert evaluation.period_net_returns == ()


def _segmented_panel(names_per_session: int = 1):
    """Six sessions split into two OOF segments of three, horizon two."""
    sessions = _span(6)
    segment_ids = [0, 0, 0, 1, 1, 1]
    rows = [
        (
            f"KRX:{pos * names_per_session + t:05d}",
            session,
            segment_ids[pos],
            0.05,
        )
        for pos, session in enumerate(sessions)
        for t in range(names_per_session)
    ]
    scored = pl.DataFrame(
        {
            "instrument_id": [r[0] for r in rows],
            "session": [r[1] for r in rows],
            "oof_segment_id": [r[2] for r in rows],
            SCORE_COLUMN: [r[3] for r in rows],
        }
    )
    realized = pl.DataFrame(
        {
            "instrument_id": [r[0] for r in rows],
            "session": [r[1] for r in rows],
            "risk_residual": [0.03] * len(rows),
            "open": [100.0] * len(rows),
            "adtv_20d": [1.0e8] * len(rows),
            "volatility_20d": [0.02] * len(rows),
        }
    )
    return scored, realized, sessions


def test_evaluate_segments_never_share_a_cohort_and_partial_tails_are_counted() -> None:
    scored, realized, _sessions = _segmented_panel()
    evaluation = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized, segment_column="oof_segment_id")
    # Each three-session segment matures one vintage and holds two partial
    # tails; no vintage or maturity crosses a segment boundary.
    assert evaluation.period_count == 2
    assert evaluation.matured_vintage_count == 2
    assert evaluation.partial_vintage_count == 4
    assert evaluation.missing_realized_vintage_count == 0
    assert evaluation.observed_sessions == 2
    assert evaluation.vintage_segment_ids == (0, 1)
    assert [block.segment_id for block in evaluation.blocks] == [0, 1]
    diagnostics = evaluation.replay_diagnostics()
    assert diagnostics["matured_vintages"] == 2
    assert diagnostics["partial_vintages"] == 4


def test_evaluate_segment_diagnostics_accounting_invariant_holds() -> None:
    scored, realized, _sessions = _segmented_panel()
    evaluation = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized, segment_column="oof_segment_id")
    assert all(
        isinstance(diag, ReplaySegmentDiagnostic) for diag in evaluation.segment_diagnostics
    )
    for diag in evaluation.segment_diagnostics:
        assert diag.vintage_count == diag.scored_sessions
        assert (
            diag.matured_vintage_count
            + diag.cash_vintage_count
            + diag.missing_realized_vintage_count
            + diag.partial_vintage_count
        ) == diag.scored_sessions
        assert diag.base_active_fraction == diag.stress_active_fraction
    segments = [diag.segment_id for diag in evaluation.segment_diagnostics]
    assert segments == [0, 1]
    segment = evaluation.segment_diagnostics[0]
    assert segment.matured_vintage_count == 1
    assert segment.partial_vintage_count == 2
    assert segment.scored_sessions == 3


def test_evaluate_single_segment_equivalent_without_segment_column() -> None:
    scored, realized, _sessions = _segmented_panel()
    replay = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    )
    segmented = replay.evaluate(scored, realized, segment_column="oof_segment_id")
    plain = replay.evaluate(scored.drop("oof_segment_id"), realized)
    # Without the segment column the six sessions form one continuous segment:
    # horizon two leaves two partial tails (positions 4 and 5).
    assert plain.partial_vintage_count == 2
    assert plain.segment_diagnostics[0].scored_sessions == 6
    assert len(plain.segment_diagnostics) == 1
    assert segmented.vintage_segment_ids == (0, 1)
    assert plain.vintage_segment_ids == (0, 0, 0, 0)


def test_evaluate_segment_missing_realized_vintage_fails_closed() -> None:
    scored, realized, _sessions = _segmented_panel()
    # Drop the realized row for the only order in segment 1: the matured
    # vintage in segment 1 must be excluded and counted as missing, never filled.
    missing_instrument = "KRX:00003"
    realized = realized.filter(pl.col("instrument_id") != missing_instrument)
    evaluation = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized, segment_column="oof_segment_id")
    assert evaluation.missing_realized_vintage_count == 1
    assert evaluation.matured_vintage_count == 1
    assert len(evaluation.period_net_returns) == 1
    assert len(evaluation.blocks) == 1
    assert evaluation.vintage_segment_ids == (0,)
    assert evaluation.partial_vintage_count == 4
    segment_one = next(
        diag
        for diag in evaluation.segment_diagnostics
        if diag.segment_id == 1
    )
    assert segment_one.missing_realized_vintage_count == 1
    assert segment_one.matured_vintage_count == 0


def test_evaluate_concurrent_exposure_never_exceeds_cap() -> None:
    """Many names make each vintage want full exposure; the cap must bind."""
    sessions = _span(9)
    names = 20
    rows = [
        (f"KRX:{t:05d}", session, 0.05)
        for session in sessions
        for t in range(names)
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
            "risk_residual": [0.03] * len(rows),
            "open": [100.0] * len(rows),
            "adtv_20d": [1.0e8] * len(rows),
            "volatility_20d": [0.02] * len(rows),
        }
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    ).evaluate(scored, realized)
    exposure_by_session: dict[object, float] = {}
    for order in evaluation.orders:
        exposure_by_session[order.decision_session] = (
            exposure_by_session.get(order.decision_session, 0.0) + order.weight
        )
    for exposure in exposure_by_session.values():
        assert exposure <= _PORTFOLIO.max_exposure + 1e-12
    # Single-session exposure is bounded and at least one vintage deploys the cap.
    assert max(exposure_by_session.values()) == pytest.approx(
        _PORTFOLIO.max_exposure, rel=1e-6
    )
    # The deterministic exposure cap leaves cash vintages between deployments.
    assert evaluation.cash_vintage_count > 0
    assert evaluation.matured_vintage_count > 0


def test_evaluate_deterministic_for_identical_inputs() -> None:
    scored, realized, _sessions = _segmented_panel(names_per_session=4)
    replay = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    )
    first = replay.evaluate(scored, realized, segment_column="oof_segment_id")
    second = replay.evaluate(scored, realized, segment_column="oof_segment_id")
    assert first.orders == second.orders
    assert first.period_net_returns == second.period_net_returns
    assert first.segment_diagnostics == second.segment_diagnostics


def _status_projection(
    scored: pl.DataFrame,
    realized: pl.DataFrame,
    *,
    unresolved: set[tuple[str, object]] | None = None,
    tail: set[tuple[str, object]] | None = None,
) -> pl.DataFrame:
    """All-REALIZED status projection, with optional typed unresolved/tail keys."""
    rows: list[dict[str, object]] = []
    for row in scored.select("instrument_id", "session").unique().iter_rows(named=True):
        key = (str(row["instrument_id"]), row["session"])
        if tail is not None and key in tail:
            state = "PARTIAL_TAIL"
        elif unresolved is not None and key in unresolved:
            state = "MISSING_EXIT_PRICE"
        elif str(key[0]) in set(realized["instrument_id"].to_list()) or (
            row["session"] in realized["session"].to_list()
            and str(row["instrument_id"]) in set(realized["instrument_id"].to_list())
        ):
            state = "REALIZED"
        else:
            state = "MISSING_EXIT_PRICE"
        rows.append(
            {
                "instrument_id": row["instrument_id"],
                "session": row["session"],
                "outcome_status": state,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "instrument_id": pl.Utf8,
            "session": pl.Datetime("us", "UTC"),
            "outcome_status": pl.Utf8,
        },
    )


def test_evaluate_status_all_realized_preserves_replay_timeline() -> None:
    """Acceptance 1: an all-REALIZED status panel changes no order or block."""
    scored, realized, _sessions = _segmented_panel(names_per_session=2)
    replay = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    )
    baseline = replay.evaluate(scored, realized, segment_column="oof_segment_id")
    status = _status_projection(scored, realized)
    with_status = replay.evaluate(
        scored, realized, segment_column="oof_segment_id", status=status
    )
    assert with_status.orders == baseline.orders
    assert with_status.blocks == baseline.blocks
    assert with_status.period_net_returns == baseline.period_net_returns
    assert with_status.vintage_segment_ids == baseline.vintage_segment_ids
    assert with_status.segment_diagnostics == baseline.segment_diagnostics
    assert with_status.missing_realized_vintage_count == 0
    assert with_status.unresolved_outcome_counts == ()


def test_evaluate_selected_missing_exit_price_is_never_zero_filled() -> None:
    """SCENARIO_REPLAY_NEVER_ZERO_FILLS: unresolved stays outside arithmetic."""
    sessions = _span(6)
    scored = _scored(
        [(f"KRX:{pos:05d}", session, 0.05) for pos, session in enumerate(sessions)]
    )
    realized = _realized(
        [
            (f"KRX:{pos:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for pos, session in enumerate(sessions)
            if pos != 0
        ]
    )
    missing_key = ("KRX:00000", sessions[0])
    status = _status_projection(scored, realized, unresolved={missing_key})
    replay = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    )
    evaluation = replay.evaluate(scored, realized, status=status)
    assert evaluation.missing_realized_vintage_count == 1
    assert evaluation.unresolved_outcome_counts == (("MISSING_EXIT_PRICE", 1),)
    assert evaluation.period_count == 2
    assert 0.0 not in evaluation.period_net_returns
    segment = evaluation.segment_diagnostics[0]
    assert segment.unresolved_outcome_counts == (("MISSING_EXIT_PRICE", 1),)
    assert segment.missing_realized_vintage_count == 1


def test_evaluate_multi_order_unresolved_vintage_counts_once_by_signature() -> None:
    """Two unresolved orders invalidate one vintage, not two vintages."""
    sessions = _span(6)
    scored = _scored(
        [
            (f"KRX:{instrument:05d}", session, 0.05 - instrument * 0.001)
            for session in sessions
            for instrument in range(2)
        ]
    )
    realized = _realized(
        [
            (f"KRX:{instrument:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for session in sessions
            for instrument in range(2)
        ]
    )
    unresolved = {
        (f"KRX:{instrument:05d}", sessions[0]) for instrument in range(2)
    }
    status = _status_projection(scored, realized, unresolved=unresolved)
    replay = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    )

    evaluation = replay.evaluate(scored, realized, status=status)

    assert evaluation.missing_realized_vintage_count == 1
    assert evaluation.unresolved_outcome_counts == (("MISSING_EXIT_PRICE", 1),)
    segment = evaluation.segment_diagnostics[0]
    assert segment.missing_realized_vintage_count == 1
    assert segment.unresolved_outcome_counts == (("MISSING_EXIT_PRICE", 1),)


def test_evaluate_mixed_unresolved_states_preserves_one_vintage_signature() -> None:
    """A missing entry is unfilled, so only the exit-side cause invalidates."""
    sessions = _span(6)
    scored = _scored(
        [
            (f"KRX:{instrument:05d}", session, 0.05 - instrument * 0.001)
            for session in sessions
            for instrument in range(2)
        ]
    )
    realized = _realized(
        [
            (f"KRX:{instrument:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for session in sessions
            for instrument in range(2)
        ]
    )
    unresolved = {
        (f"KRX:{instrument:05d}", sessions[0]) for instrument in range(2)
    }
    status = _status_projection(scored, realized, unresolved=unresolved).with_columns(
        pl.when(
            (pl.col("instrument_id") == "KRX:00001")
            & (pl.col("session") == sessions[0])
        )
        .then(pl.lit("MISSING_ENTRY_PRICE"))
        .otherwise(pl.col("outcome_status"))
        .alias("outcome_status")
    )
    replay = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    )

    evaluation = replay.evaluate(scored, realized, status=status)

    # The MISSING_ENTRY_PRICE order is an unfilled entry (no exposure, no
    # return), so it never enters the vintage signature; the remaining
    # MISSING_EXIT_PRICE order still invalidates the vintage.
    assert evaluation.missing_realized_vintage_count == 1
    assert evaluation.unresolved_outcome_counts == (("MISSING_EXIT_PRICE", 1),)


def test_evaluate_segment_tail_is_partial_and_mature_tail_raises() -> None:
    """Acceptance 3: chronological tail is PARTIAL_TAIL; mature-segment tail is error."""
    scored, realized, _sessions = _segmented_panel()
    replay = NetAlphaPolicyReplay(
        2, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    )
    # Every key is realized except the trailing position of the last segment.
    tail_key = ("KRX:00005", scored["session"].to_list()[-1])
    status = _status_projection(scored, realized, tail={tail_key})
    evaluation = replay.evaluate(
        scored, realized, segment_column="oof_segment_id", status=status
    )
    # The tail key lives in a segment-local tail position (position 2, horizon 2)
    # and is therefore a chronological partial vintage, not a mature failure.
    assert evaluation.partial_vintage_count == 4
    assert evaluation.missing_realized_vintage_count == 0

    # The same sidecar PARTIAL_TAIL in a mature position (first session of
    # segment 0) must raise ValueError: only the segment-local maturity rule
    # may classify a tail.
    mature_tail = ("KRX:00000", scored["session"].to_list()[0])
    bad_status = _status_projection(scored, realized, tail={mature_tail})
    with pytest.raises(ValueError, match="PARTIAL_TAIL inside a mature segment"):
        replay.evaluate(
            scored, realized, segment_column="oof_segment_id", status=bad_status
        )


def test_evaluate_realized_status_without_inputs_raises() -> None:
    """A REALIZED status whose replay inputs are absent is a contract violation."""
    sessions = _span(6)
    scored = _scored(
        [(f"KRX:{pos:05d}", session, 0.05) for pos, session in enumerate(sessions)]
    )
    realized = _realized(
        [
            (f"KRX:{pos:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for pos, session in enumerate(sessions)
            if pos != 0
        ]
    )
    # The missing key (session zero) is declared REALIZED even though its
    # realized inputs are absent from the panel.
    status = _status_projection(scored, realized)
    status = status.with_columns(
        pl.when(pl.col("session") == sessions[0])
        .then(pl.lit("REALIZED"))
        .otherwise(pl.col("outcome_status"))
        .alias("outcome_status")
    )
    replay = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    )
    with pytest.raises(ValueError, match="REALIZED status"):
        replay.evaluate(scored, realized, status=status)


def test_replay_policy_is_pinned_and_exposed() -> None:
    from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1

    replay = NetAlphaPolicyReplay(3, _PORTFOLIO, _RISK)
    assert replay.policy == SCHEDULED_OPEN_V1
    assert replay.policy_hash == SCHEDULED_OPEN_V1.canonical_hash


def test_evaluate_rejects_evidence_pinned_under_foreign_policy() -> None:
    from src.stocks.domain.execution_policy import (
        SCHEDULED_OPEN_V1,
        ExecutionOutcomePolicy,
    )

    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored([("KRX:00001", session, 0.02)])
    foreign = ExecutionOutcomePolicy(
        policy_id="first_tradable_open_v1", max_entry_delay_sessions=1
    )
    evidence = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "session": [session],
            "policy_hash": [foreign.canonical_hash],
            "outcome_status": ["REALIZED"],
        }
    )
    replay = NetAlphaPolicyReplay(3, _PORTFOLIO, _RISK, policy=SCHEDULED_OPEN_V1)
    with pytest.raises(ValueError, match="foreign execution policy"):
        replay.evaluate(scored, realized=None, evidence=evidence)


def test_evaluate_uses_evidence_as_status_projection_when_status_omitted() -> None:
    from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1

    sessions = _span(6)
    scored = _scored(
        [(f"KRX:{pos:05d}", session, 0.05) for pos, session in enumerate(sessions)]
    )
    realized = _realized(
        [
            (f"KRX:{pos:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for pos, session in enumerate(sessions)
            if pos != 0
        ]
    )
    evidence = pl.DataFrame(
        {
            "instrument_id": ["KRX:00000"],
            "session": [sessions[0]],
            "policy_hash": [SCHEDULED_OPEN_V1.canonical_hash],
            "outcome_status": ["MISSING_EXIT_PRICE"],
        }
    )
    evaluation = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY, policy=SCHEDULED_OPEN_V1
    ).evaluate(scored, realized=realized, evidence=evidence)
    assert len(evaluation.orders) == 6
    assert evaluation.missing_realized_vintage_count == 1
    assert evaluation.unresolved_outcome_counts == (("MISSING_EXIT_PRICE", 1),)


def test_evaluate_rejects_duplicate_evidence_keys() -> None:
    from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1

    session = datetime(2024, 1, 2, tzinfo=UTC)
    scored = _scored([("KRX:00001", session, 0.02)])
    evidence = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001", "KRX:00001"],
            "session": [session, session],
            "policy_hash": [SCHEDULED_OPEN_V1.canonical_hash] * 2,
            "outcome_status": ["REALIZED", "REALIZED"],
        }
    )
    replay = NetAlphaPolicyReplay(3, _PORTFOLIO, _RISK, policy=SCHEDULED_OPEN_V1)
    with pytest.raises(ValueError, match="duplicate instrument/session keys"):
        replay.evaluate(scored, realized=None, evidence=evidence)

def _typed_status_projection(
    scored: pl.DataFrame,
    *,
    missing_entry: set[tuple[str, object]] | None = None,
    missing_exit: set[tuple[str, object]] | None = None,
) -> pl.DataFrame:
    """All-REALIZED status projection with explicit typed missing keys."""
    missing_entry = missing_entry or set()
    missing_exit = missing_exit or set()
    rows: list[dict[str, object]] = []
    for row in scored.select("instrument_id", "session").unique().iter_rows(named=True):
        key = (str(row["instrument_id"]), row["session"])
        if key in missing_entry:
            state = "MISSING_ENTRY_PRICE"
        elif key in missing_exit:
            state = "MISSING_EXIT_PRICE"
        else:
            state = "REALIZED"
        rows.append(
            {
                "instrument_id": row["instrument_id"],
                "session": row["session"],
                "outcome_status": state,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "instrument_id": pl.Utf8,
            "session": pl.Datetime("us", "UTC"),
            "outcome_status": pl.Utf8,
        },
    )


def test_missing_entry_is_unfilled_and_missing_exit_invalidates_vintage() -> None:
    """SCENARIO_REPLAY_NEVER_ZERO_FILLS: unfilled entries leave cash; missing exits invalidate."""
    sessions = _span(6)
    scored = _scored(
        [
            (f"KRX:{instrument:05d}", session, 0.05 - instrument * 0.001)
            for session in sessions
            for instrument in range(2)
        ]
    )
    realized = _realized(
        [
            (f"KRX:{instrument:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for session in sessions
            for instrument in range(2)
        ]
    )
    # Session zero's two orders fail to fill; session two's KRX:00001 has an
    # entry but no exit. Every other order is REALIZED.
    missing_entry = {(f"KRX:{i:05d}", sessions[0]) for i in range(2)}
    missing_exit = {("KRX:00001", sessions[2])}
    status = _typed_status_projection(
        scored, missing_entry=missing_entry, missing_exit=missing_exit
    )
    replay = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY
    )

    evaluation = replay.evaluate(scored, realized, status=status)

    # The unfilled-entry vintage deploys nothing and is observed as cash (0.0),
    # never a missing-realized vintage; the missing-exit vintage is invalidated.
    assert evaluation.missing_realized_vintage_count == 1
    assert evaluation.cash_vintage_count == 1
    assert evaluation.matured_vintage_count == 1
    assert evaluation.period_count == 2
    assert evaluation.period_net_returns[0] == 0.0
    assert len(evaluation.blocks) == 1
    assert evaluation.unresolved_outcome_counts == (("MISSING_EXIT_PRICE", 1),)
    assert 0.0 in evaluation.period_net_returns
    assert evaluation.observed_sessions == 2

    # A panel whose only non-realized orders are unfilled entries never
    # increments the missing-realized count (contract python_assertion).
    only_unfilled = replay.evaluate(
        scored, realized,
        status=_typed_status_projection(scored, missing_entry=missing_entry),
    )
    assert only_unfilled.missing_realized_vintage_count == 0
    assert only_unfilled.cash_vintage_count == 1
    assert only_unfilled.matured_vintage_count == 2
    assert only_unfilled.period_count == 3
    assert only_unfilled.period_net_returns[0] == 0.0


def test_scenario_replay_never_zero_fills_blocked_vintage_with_evidence() -> None:
    """SCENARIO_REPLAY_NEVER_ZERO_FILLS: blocked exit gets a named record, no cash."""
    from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1

    sessions = _span(6)
    scored = _scored(
        [(f"KRX:{pos:05d}", session, 0.05) for pos, session in enumerate(sessions)]
    )
    realized = _realized(
        [
            (f"KRX:{pos:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for pos, session in enumerate(sessions)
            if pos != 0
        ]
    )
    missing_key = ("KRX:00000", sessions[0])
    status = _status_projection(scored, realized, unresolved={missing_key})
    evidence = pl.DataFrame(
        {
            "instrument_id": ["KRX:00000"],
            "session": [sessions[0]],
            "policy_hash": [SCHEDULED_OPEN_V1.canonical_hash],
            "outcome_status": ["MISSING_EXIT_PRICE"],
            "resolution_kind": ["CONFIRMED_NO_BAR"],
            "scheduled_entry_session": [sessions[0] + __import__("datetime").timedelta(days=1)],
            "scheduled_exit_session": [sessions[0] + __import__("datetime").timedelta(days=4)],
            "entry_disposition": ["FILLED"],
            "exit_disposition": ["NO_BAR"],
        }
    )
    replay = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY, policy=SCHEDULED_OPEN_V1
    )
    evaluation = replay.evaluate(scored, realized, status=status, evidence=evidence)
    assert evaluation.blocked_vintage_count == 1
    assert len(evaluation.blocked_vintages) == 1
    blocked = evaluation.blocked_vintages[0]
    assert blocked.instrument_id == "KRX:00000"
    assert blocked.decision_session == sessions[0]
    assert blocked.outcome_status == "MISSING_EXIT_PRICE"
    assert blocked.resolution_kind == "CONFIRMED_NO_BAR"
    assert blocked.entry_disposition == "FILLED"
    assert blocked.exit_disposition == "NO_BAR"
    assert evaluation.missing_realized_vintage_count == 1
    assert evaluation.cash_vintage_count == 0
    assert evaluation.period_count == 2
    assert 0.0 not in evaluation.period_net_returns
    assert evaluation.replay_diagnostics()["selected_blocked_exits"] == 1


def test_scenario_replay_never_zero_fills_missing_entry_stays_unfilled() -> None:
    """SCENARIO_REPLAY_NEVER_ZERO_FILLS: unfilled entry keeps a cash vintage."""
    from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1

    sessions = _span(6)
    scored = _scored(
        [(f"KRX:{pos:05d}", session, 0.05) for pos, session in enumerate(sessions)]
    )
    realized = _realized(
        [
            (f"KRX:{pos:05d}", session, 0.05, 100.0, 1.0e8, 0.02)
            for pos, session in enumerate(sessions)
            if pos != 0
        ]
    )
    status = _status_projection(
        scored, realized, unresolved={("KRX:00000", sessions[0])}
    ).with_columns(
        pl.when(
            (pl.col("instrument_id") == "KRX:00000")
            & (pl.col("session") == sessions[0])
        )
        .then(pl.lit("MISSING_ENTRY_PRICE"))
        .otherwise(pl.col("outcome_status"))
        .alias("outcome_status")
    )
    evidence = pl.DataFrame(
        {
            "instrument_id": ["KRX:00000"],
            "session": [sessions[0]],
            "policy_hash": [SCHEDULED_OPEN_V1.canonical_hash],
            "outcome_status": ["MISSING_ENTRY_PRICE"],
            "resolution_kind": ["CONFIRMED_NO_BAR"],
            "scheduled_entry_session": [sessions[0] + __import__("datetime").timedelta(days=1)],
            "scheduled_exit_session": [sessions[0] + __import__("datetime").timedelta(days=4)],
            "entry_disposition": ["NO_BAR"],
            "exit_disposition": ["NO_BAR"],
        }
    )
    replay = NetAlphaPolicyReplay(
        3, _PORTFOLIO, _RISK, liquidity_model=_LIQUIDITY, policy=SCHEDULED_OPEN_V1
    )
    evaluation = replay.evaluate(scored, realized, status=status, evidence=evidence)
    assert evaluation.blocked_vintage_count == 0
    assert evaluation.missing_realized_vintage_count == 0
    assert evaluation.cash_vintage_count == 1
    assert evaluation.period_count == 3
    assert 0.0 in evaluation.period_net_returns
