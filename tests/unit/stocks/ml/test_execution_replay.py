"""Execution-equivalent replay adapter contract tests."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1
from src.stocks.backtesting.engine import BacktestLedgerRow
from src.stocks.ml.contracts import (
    CompoundingCertificationSettings,
    NetAlphaTrainingRequest,
)
from src.stocks.ml.execution_replay import (
    ExecutionEquivalentReplayRequest,
    ExecutionReplayContext,
    ExecutionReplayEvidence,
    _ledger_growth_and_exposure,
    replay_execution_equivalent,
)
from src.stocks.ml.training import (
    _coverage_failure_reason,
    _evidence_from_execution,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from tests.fixtures.stocks.helpers import stock_liquidity_model

_SESSION_COLUMN = "session"
_SEGMENT_COLUMN = "oof_segment_id"
_SCORE_COLUMN = "predicted_net_alpha"


def _session_for(segment: int, index: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=segment * 12 + index)


def _market_frame(
    n_segments: int = 2, sessions_per_segment: int = 12, n_tickers: int = 3
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for segment in range(n_segments):
        for index in range(sessions_per_segment):
            session = _session_for(segment, index)
            for ticker in range(n_tickers):
                price = 100.0 + ticker + index
                rows.append(
                    {
                        "instrument_id": f"KRX:0000{ticker + 1}",
                        "session": session,
                        "observation_time": session.replace(hour=15, minute=30),
                        "available_time": session.replace(hour=15, minute=31),
                        "open": price,
                        "close": price * 1.01,
                        "volume": 1_000_000.0,
                        "trading_value": price * 1_000_000.0,
                        "sector": f"S{ticker % 2}",
                        "adtv": price * 1_000_000.0,
                    }
                )
    return pl.DataFrame(rows)


def test_sparse_telemetry_projection_is_bounded() -> None:
    """SPARSE_GROWTH_06_BOUNDED_TELEMETRY_AND_ARTIFACT_PARITY."""
    evidence = ExecutionReplayEvidence(
        base_log_growth=(0.001,),
        stress_log_growth=(0.0005,),
        segment_ids=(0,),
        planned_cycles=1,
        filled_orders=1,
        cash_session_fraction=0.0,
        turnover=0.1,
        observed_interval_count=1,
        invested_interval_count=1,
        invested_interval_fraction=1.0,
        filled_cycle_count=1,
        unfilled_order_reason_counts=(),
        action_diagnostics=(("replacement_count", 1), ("turnover_ratio", 0.5)),
    )
    diagnostics = evidence.diagnostics()
    assert diagnostics["action_diagnostics"] == {
        "replacement_count": 1,
        "turnover_ratio": 0.5,
    }
    assert diagnostics["invested_interval_fraction"] == 1.0
    assert diagnostics["invested_interval_count"] == 1
    assert diagnostics["filled_cycle_count"] == 1


def _score_frame(
    n_segments: int = 2, sessions_per_segment: int = 12, n_tickers: int = 3
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for segment in range(n_segments):
        for index in range(sessions_per_segment):
            session = _session_for(segment, index)
            for ticker in range(n_tickers):
                score = 0.01 + ticker * 0.001
                rows.append(
                    {
                        "instrument_id": f"KRX:0000{ticker + 1}",
                        "session": session,
                        _SEGMENT_COLUMN: segment,
                        _SCORE_COLUMN: score,
                        "expected_active_alpha": score,
                        "alpha_lower_bound": score,
                        "expected_net_alpha": score,
                        "net_alpha_lower_bound": score,
                        "exit_cost_rate": 0.001,
                    }
                )
    return pl.DataFrame(rows)


def _context(market: pl.DataFrame) -> ExecutionReplayContext:
    instruments = {
        str(instrument_id): Instrument(
            str(instrument_id), AssetKind.STOCK, "KRX",
            str(instrument_id).split(":")[-1], "KRW", lot_size=1,
        )
        for instrument_id in sorted(market["instrument_id"].unique().to_list())
    }
    sessions = sorted(market[_SESSION_COLUMN].unique().to_list())
    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="h",
        provider_version="p",
        universe_policy_version="u",
        universe_policy_hash="u",
        feature_set="stock_net_alpha_v1",
        feature_set_hash="f",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        time_start=sessions[0],
        time_end=sessions[-1],
        generated_time=sessions[-1],
        row_count=market.height,
    )
    return ExecutionReplayContext(
        registry=ModelArtifactRegistry(Path("mem://execution-replay")),
        manifest=manifest,
        instruments=instruments,
        artifact_id="na_replay_test",
        strategy_id="na_replay_test",
        initial_portfolio=PortfolioSnapshot(
            account_snapshot_id="oof",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=100_000_000.0,
            unsettled_cash=0.0,
            positions=(),
        ),
        risk_policy=StockRiskPolicy(
            top_k=3, gross_cap=0.9, single_name_cap=0.3, sector_cap=0.5,
            participation_limit=0.01, no_trade_band_bps=0.0,
        ),
        base_cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        liquidity_model=stock_liquidity_model(),
        stress_liquidity_model=stock_liquidity_model(stress_multiplier=1.5),
        execution_policy=SCHEDULED_OPEN_V1,
        seed=42,
    )


def _request(
    market: pl.DataFrame,
    scores: pl.DataFrame,
    segments: dict[int, tuple[datetime, ...]],
    context: ExecutionReplayContext,
) -> ExecutionEquivalentReplayRequest:
    return ExecutionEquivalentReplayRequest(
        context=context,
        market_frame=market,
        score_frame=scores,
        segment_column=_SEGMENT_COLUMN,
        decision_sessions_by_segment=segments,
        horizon_sessions=5,
    )


def _decision_sessions(market: pl.DataFrame) -> dict[int, tuple[datetime, ...]]:
    return {
        segment: tuple(
            session
            for index, session in enumerate(
                sorted(market[_SESSION_COLUMN].unique().to_list())
            )
            if index // 12 == segment
        )
        for segment in (0, 1)
    }


def test_prepared_allocations_reject_overlay_length_mismatch() -> None:
    """Duplicate/unmatched/non-finite score keys and invalid markets fail closed."""
    market = _market_frame()
    context = _context(market)
    scores = _score_frame()
    segments = _decision_sessions(market)

    duplicate = pl.concat(
        [scores, scores.head(1).with_columns(pl.col("session").cast(pl.Datetime("us", "UTC")))]
    )
    with pytest.raises(ValueError, match="duplicate instrument/session"):
        replay_execution_equivalent(_request(market, duplicate, segments, context))

    unmatched = scores.with_columns(
        pl.when(pl.col("session") == _session_for(0, 0))
        .then(pl.lit(datetime(2024, 6, 1, tzinfo=UTC)))
        .otherwise(pl.col("session"))
        .alias("session")
    )
    with pytest.raises(ValueError, match="without a market row"):
        replay_execution_equivalent(_request(market, unmatched, segments, context))

    non_finite = scores.with_columns(
        pl.when(pl.col(_SCORE_COLUMN) == pl.col(_SCORE_COLUMN).max())
        .then(pl.lit(float("inf")))
        .otherwise(pl.col(_SCORE_COLUMN))
        .alias(_SCORE_COLUMN)
    )
    with pytest.raises(ValueError, match="null or non-finite"):
        replay_execution_equivalent(_request(market, non_finite, segments, context))

    missing_executable = market.drop("close")
    with pytest.raises(ValueError, match="must carry"):
        replay_execution_equivalent(_request(missing_executable, scores, segments, context))

    naive = market.with_columns(
        pl.col("available_time").dt.replace_time_zone(None)
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        replay_execution_equivalent(_request(naive, scores, segments, context))


def test_replay_builds_one_prepared_market_per_segment_with_clean_segment_ids() -> None:
    """Each segment starts from independent initial cash and never mixes ids."""
    market = _market_frame()
    context = _context(market)
    scores = _score_frame()
    segments = _decision_sessions(market)

    evidence = replay_execution_equivalent(
        _request(market, scores, segments, context)
    )
    assert isinstance(evidence, ExecutionReplayEvidence)
    assert len(evidence.base_log_growth) == len(evidence.segment_ids)
    assert len(evidence.stress_log_growth) == len(evidence.segment_ids)
    first = evidence.segment_ids[0]
    transition = next(
        (i for i, segment in enumerate(evidence.segment_ids) if segment != first), None
    )
    assert transition is not None
    assert evidence.segment_ids[:transition] == (first,) * transition
    assert evidence.segment_ids[transition:] == (first + 1,) * (
        len(evidence.segment_ids) - transition
    )
    assert evidence.planned_cycles > 0
    assert evidence.filled_orders > 0
    assert evidence.filled_orders >= evidence.planned_cycles


def test_replay_supplies_prepared_adtv_to_the_scored_cycle() -> None:
    """The planner receives the engine's rolling ADTV, not a feature alias."""
    market = _market_frame().drop("adtv")
    context = _context(market)
    scores = _score_frame()

    evidence = replay_execution_equivalent(
        _request(market, scores, _decision_sessions(market), context)
    )

    assert evidence.planned_cycles > 0
    assert evidence.filled_orders > 0


def test_identical_scores_produce_identical_schedule_across_horizons() -> None:
    """horizon affects only the bootstrap block floor, never the decision path."""
    market = _market_frame()
    context = _context(market)
    scores = _score_frame()
    segments = _decision_sessions(market)

    low = _request(market, scores, segments, context)
    low = ExecutionEquivalentReplayRequest(
        context=low.context,
        market_frame=low.market_frame,
        score_frame=low.score_frame,
        segment_column=low.segment_column,
        decision_sessions_by_segment=low.decision_sessions_by_segment,
        horizon_sessions=5,
    )
    high = ExecutionEquivalentReplayRequest(
        context=low.context,
        market_frame=low.market_frame,
        score_frame=low.score_frame,
        segment_column=low.segment_column,
        decision_sessions_by_segment=low.decision_sessions_by_segment,
        horizon_sessions=40,
    )
    low_evidence = replay_execution_equivalent(low)
    high_evidence = replay_execution_equivalent(high)
    assert low_evidence.base_log_growth == high_evidence.base_log_growth
    assert low_evidence.stress_log_growth == high_evidence.stress_log_growth
    assert low_evidence.segment_ids == high_evidence.segment_ids
    assert low_evidence.planned_cycles == high_evidence.planned_cycles
    assert low_evidence.filled_orders == high_evidence.filled_orders


DELTA_COST_UTILITY_05 = "DELTA_COST_UTILITY_05_REPLAY_TELEMETRY"
SPARSE_GROWTH_V5_07_REPLAY_TELEMETRY = "SPARSE_GROWTH_V5_07_REPLAY_TELEMETRY"


def test_delta_cost_utility_05_replay_telemetry() -> None:
    """Replay under delta_cost_aware_v1 returns finite non-negative diagnostics."""
    market = _market_frame()
    instruments = {
        str(instrument_id): Instrument(
            str(instrument_id), AssetKind.STOCK, "KRX",
            str(instrument_id).split(":")[-1], "KRW", lot_size=1,
        )
        for instrument_id in sorted(market["instrument_id"].unique().to_list())
    }
    sessions = sorted(market["session"].unique().to_list())
    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="h",
        provider_version="p",
        universe_policy_version="u",
        universe_policy_hash="u",
        feature_set="stock_net_alpha_v1",
        feature_set_hash="f",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        time_start=sessions[0],
        time_end=sessions[-1],
        generated_time=sessions[-1],
        row_count=market.height,
    )
    context = ExecutionReplayContext(
        registry=ModelArtifactRegistry(Path("mem://replay-telemetry")),
        manifest=manifest,
        instruments=instruments,
        artifact_id="telemetry_test",
        strategy_id="telemetry_test",
        initial_portfolio=PortfolioSnapshot(
            account_snapshot_id="oof",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=100_000_000.0,
            unsettled_cash=0.0,
            positions=(),
        ),
        risk_policy=StockRiskPolicy(
            top_k=3, gross_cap=0.9, single_name_cap=0.3, sector_cap=0.5,
            participation_limit=0.01, no_trade_band_bps=0.0,
            execution_utility_mode="delta_cost_aware_v1",
        ),
        base_cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        liquidity_model=stock_liquidity_model(),
        stress_liquidity_model=stock_liquidity_model(stress_multiplier=1.5),
        execution_policy=SCHEDULED_OPEN_V1,
        seed=42,
    )
    scores = _score_frame()
    segments = _decision_sessions(market)
    request = ExecutionEquivalentReplayRequest(
        context=context,
        market_frame=market,
        score_frame=scores,
        segment_column=_SEGMENT_COLUMN,
        decision_sessions_by_segment=segments,
        horizon_sessions=5,
    )
    evidence = replay_execution_equivalent(request)
    assert isinstance(evidence, ExecutionReplayEvidence)
    assert evidence.planned_cycles > 0
    assert evidence.filled_orders > 0
    assert evidence.turnover >= 0.0
    assert 0.0 <= evidence.cash_session_fraction <= 1.0
    assert len(evidence.base_log_growth) == len(evidence.stress_log_growth)
    assert all(np.isfinite(g) for g in evidence.base_log_growth)
    assert all(np.isfinite(g) for g in evidence.stress_log_growth)


def _ledger_rows(positions_flags: list[float]) -> tuple[BacktestLedgerRow, ...]:
    rows: list[BacktestLedgerRow] = []
    equity = 100.0
    for index, flag in enumerate(positions_flags):
        equity += 1.0
        rows.append(
            BacktestLedgerRow(
                session=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index),
                settled_cash=50.0,
                unsettled_cash=0.0,
                positions_value=flag,
                accrued_costs=0.0,
                equity=equity,
            )
        )
    return tuple(rows)


def _v6_request() -> NetAlphaTrainingRequest:
    return NetAlphaTrainingRequest(
        artifact_id="exposure_test",
        candidate_horizon_sessions=(10,),
        compounding=CompoundingCertificationSettings(
            min_active_cohort_fraction=0.2,
            min_observed_sessions=1,
            bootstrap_alpha=0.05,
            bootstrap_resamples=50,
        ),
    )


def test_sparse_growth_v6_exposure_coverage() -> None:
    """SPARSE_GROWTH_V6_EXPOSURE_COVERAGE.

    Five complete intervals with a positive prior positions_value (a held
    position) report invested_interval_count == 5 and
    invested_interval_fraction == 1.0; a held-but-unfilled candidate still
    clears the exposure coverage gate. Five zero-position intervals report
    invested_interval_count == 0 and the candidate is rejected.
    """
    invested_ledger = _ledger_rows([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    growth, invested = _ledger_growth_and_exposure(invested_ledger)
    assert len(growth) == 5
    assert invested == 5

    base_evidence = ExecutionReplayEvidence(
        base_log_growth=growth,
        stress_log_growth=growth,
        segment_ids=(0,) * len(growth),
        planned_cycles=1,
        filled_orders=1,
        cash_session_fraction=0.0,
        turnover=0.1,
        observed_interval_count=len(growth),
        invested_interval_count=invested,
        invested_interval_fraction=invested / len(growth),
        filled_cycle_count=1,
        unfilled_order_reason_counts=(),
    )
    assert base_evidence.invested_interval_count == 5
    assert base_evidence.invested_interval_fraction == 1.0

    candidate = _evidence_from_execution(
        10, "lower_bound_only", "net_alpha_elastic_net",
        base_evidence, base_evidence, (0.1, 0.2, 0.3), 1,
    )
    assert _coverage_failure_reason(candidate, _v6_request()) == ""

    zero_ledger = _ledger_rows([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    zero_growth, zero_invested = _ledger_growth_and_exposure(zero_ledger)
    assert zero_invested == 0
    zero_evidence = ExecutionReplayEvidence(
        base_log_growth=zero_growth,
        stress_log_growth=zero_growth,
        segment_ids=(0,) * len(zero_growth),
        planned_cycles=1,
        filled_orders=1,
        cash_session_fraction=0.0,
        turnover=0.1,
        observed_interval_count=len(zero_growth),
        invested_interval_count=zero_invested,
        invested_interval_fraction=0.0,
        filled_cycle_count=1,
        unfilled_order_reason_counts=(),
    )
    zero_candidate = _evidence_from_execution(
        10, "lower_bound_only", "net_alpha_elastic_net",
        zero_evidence, zero_evidence, (0.1, 0.2, 0.3), 1,
    )
    reason = _coverage_failure_reason(zero_candidate, _v6_request())
    assert "active-coverage-insufficient" in reason


def test_interval_evidence_growth_recovery() -> None:
    """GROWTH_RECOVERY_INTERVAL_EVIDENCE_05.

    A horizon-locked ten-session replay with decision times at sessions 0, 2,
    and 4 exposes exactly two complete parallel base/stress interval log-growth
    observations (the incomplete terminal interval is excluded), matching the
    interval-count evidence admission contract.
    """
    import math

    from src.stocks.ml.execution_replay import _decision_interval_log_growth

    base = datetime(2024, 1, 1, tzinfo=UTC)
    times = [base + timedelta(days=d) for d in range(5)]

    base_equity = [100.0, 110.0, 121.0, 108.9, 119.79]
    stress_equity = [100.0, 109.0, 118.0, 110.0, 121.0]
    base_ledger = [
        BacktestLedgerRow(
            t, settled_cash=0.0, unsettled_cash=0.0,
            positions_value=1.0, accrued_costs=0.0, equity=base_equity[i],
        )
        for i, t in enumerate(times)
    ]
    stress_ledger = [
        BacktestLedgerRow(
            t, settled_cash=0.0, unsettled_cash=0.0,
            positions_value=1.0, accrued_costs=0.0, equity=stress_equity[i],
        )
        for i, t in enumerate(times)
    ]

    decisions = [times[0], times[2], times[4]]
    base_growth = _decision_interval_log_growth(base_ledger, decisions)
    stress_growth = _decision_interval_log_growth(stress_ledger, decisions)

    assert len(base_growth) == 2
    assert len(stress_growth) == 2
    assert all(math.isfinite(g) for g in base_growth)
    assert all(math.isfinite(g) for g in stress_growth)
    assert base_growth[0] == pytest.approx(math.log(121.0 / 100.0))
    assert base_growth[1] == pytest.approx(math.log(119.79 / 121.0))

    assert _decision_interval_log_growth(base_ledger, [times[0]]) == ()
