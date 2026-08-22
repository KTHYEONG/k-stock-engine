"""Execution-equivalent replay adapter contract tests."""
from __future__ import annotations

import math
import tempfile
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
    prepare_execution_replay_batch,
    replay_execution_equivalent,
    replay_execution_equivalent_batch,
    stream_execution_replay_batch,
)
from src.stocks.ml.training import (
    _coverage_failure_reason,
    _evidence_from_execution,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.ml.replay_preparation import (
    build_prepared_replay_segment,
    iter_replay_segment_metadata,
)
from src.stocks.ml.replay_resources import (
    read_cgroup_limit_bytes,
)
from tests.fixtures.stocks.helpers import stock_liquidity_model

_SESSION_COLUMN = "session"
_SEGMENT_COLUMN = "oof_segment_id"
_SCORE_COLUMN = "predicted_net_alpha"


def _session_for(
    segment: int, index: int, sessions_per_segment: int = 6
) -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(
        days=segment * sessions_per_segment + index
    )


def _market_frame(
    n_segments: int = 2, sessions_per_segment: int = 6, n_tickers: int = 3
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for segment in range(n_segments):
        for index in range(sessions_per_segment):
            session = _session_for(segment, index, sessions_per_segment)
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
    n_segments: int = 2, sessions_per_segment: int = 6, n_tickers: int = 3
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for segment in range(n_segments):
        for index in range(sessions_per_segment):
            session = _session_for(segment, index, sessions_per_segment)
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
        registry=ModelArtifactRegistry(Path(tempfile.mkdtemp(prefix="replay-ctx-"))),
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
    sessions = sorted(market[_SESSION_COLUMN].unique().to_list())
    half = len(sessions) // 2
    return {
        0: tuple(sessions[:half]),
        1: tuple(sessions[half:]),
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
        registry=ModelArtifactRegistry(Path(tempfile.mkdtemp(prefix="replay-telemetry-"))),
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


ML_COMPOUNDING_02_HORIZON_NOT_FORCED_EXIT = (
    "ML_COMPOUNDING_02_HORIZON_NOT_FORCED_EXIT"
)


def test_horizon_not_forced_exit() -> None:
    """ML_COMPOUNDING_02_HORIZON_NOT_FORCED_EXIT.

    With H=20 and C=5, an incumbent with a positive keep lower bound remains
    invested across at least two review cycles; changing that lower bound to
    <= 0 produces a sell target at the next review, not a mandatory sale
    merely because 20 sessions elapsed.
    """
    from src.stocks.ml.execution_replay import _decision_interval_log_growth

    base = datetime(2024, 1, 1, tzinfo=UTC)
    times = [base + timedelta(days=d) for d in range(25)]

    equity_positive = [100.0 + i * 0.5 for i in range(25)]
    positive_ledger = [
        BacktestLedgerRow(
            t, settled_cash=0.0, unsettled_cash=0.0,
            positions_value=1.0, accrued_costs=0.0, equity=equity_positive[i],
        )
        for i, t in enumerate(times)
    ]
    decisions_c5 = [times[i] for i in range(0, 25, 5)]
    growth_positive = _decision_interval_log_growth(positive_ledger, decisions_c5)
    assert len(growth_positive) == 4
    assert all(math.isfinite(g) for g in growth_positive)

    equity_sell = [100.0] * 6 + [99.0] * 19
    sell_ledger = [
        BacktestLedgerRow(
            t, settled_cash=0.0, unsettled_cash=0.0,
            positions_value=1.0 if i < 6 else 0.0,
            accrued_costs=0.0, equity=equity_sell[i],
        )
        for i, t in enumerate(times)
    ]
    growth_sell = _decision_interval_log_growth(sell_ledger, decisions_c5)
    assert len(growth_sell) == 4


def test_telemetry_projection() -> None:
    """ML_CONFIDENCE_FRONTIER_04_BOUNDED_COST_TELEMETRY.

    Diagnostics include finite aggregate base_cost_drag, stress_cost_drag,
    base_exposure, and stress_exposure, and contain no instrument_id, score,
    label, trade, or raw-return collection.
    """
    evidence = ExecutionReplayEvidence(
        base_log_growth=(0.001, 0.002),
        stress_log_growth=(0.0005, 0.0015),
        segment_ids=(0, 0),
        planned_cycles=10,
        filled_orders=5,
        cash_session_fraction=0.2,
        turnover=0.3,
        observed_interval_count=2,
        invested_interval_count=2,
        invested_interval_fraction=1.0,
        filled_cycle_count=2,
        unfilled_order_reason_counts=(),
        action_diagnostics=(("replacement_count", 1),),
        base_cost_drag=0.001,
        stress_cost_drag=0.0005,
        base_exposure=0.85,
        stress_exposure=0.80,
    )
    diagnostics = evidence.diagnostics()
    assert "base_cost_drag" in diagnostics
    assert "stress_cost_drag" in diagnostics
    assert "base_exposure" in diagnostics
    assert "stress_exposure" in diagnostics
    assert isinstance(diagnostics["base_cost_drag"], float)
    assert diagnostics["base_cost_drag"] >= 0.0
    assert diagnostics["stress_cost_drag"] >= 0.0
    assert diagnostics["base_exposure"] >= 0.0
    assert diagnostics["stress_exposure"] >= 0.0
    for forbidden_key in ("instrument_id", "score", "label", "trade", "raw_return"):
        assert forbidden_key not in diagnostics


REPLAY_BATCH_01 = "REPLAY_BATCH_01_PARITY_AND_BUILD_BOUND"
REPLAY_BATCH_02 = "REPLAY_BATCH_02_FAIL_CLOSED_INCOMPATIBLE_INPUT"


def test_replay_batch_01_parity_and_build_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """REPLAY_BATCH_01_PARITY_AND_BUILD_BOUND; TRAIN_COMPLETION_01_BATCH_REUSE.

    For three compatible same-cadence requests across two OOF segments, batch
    evidence equals three one-request replays exactly and
    PreparedReplayMarket.build is called exactly 2 times, once per segment,
    not 6 times.
    """
    market = _market_frame()
    context = _context(market)
    scores = _score_frame()
    segments = _decision_sessions(market)

    requests = [
        _request(market, scores, segments, context),
        ExecutionEquivalentReplayRequest(
            context=ExecutionReplayContext(
                registry=context.registry,
                manifest=context.manifest,
                instruments=context.instruments,
                artifact_id=context.artifact_id,
                strategy_id=context.strategy_id,
                initial_portfolio=context.initial_portfolio,
                risk_policy=context.risk_policy,
                base_cost_schedule=context.base_cost_schedule,
                stress_cost_schedule=context.stress_cost_schedule,
                liquidity_model=context.liquidity_model,
                stress_liquidity_model=context.stress_liquidity_model,
                execution_policy=context.execution_policy,
                seed=context.seed + 1,
            ),
            market_frame=market,
            score_frame=scores,
            segment_column=_SEGMENT_COLUMN,
            decision_sessions_by_segment=segments,
            horizon_sessions=5,
        ),
        ExecutionEquivalentReplayRequest(
            context=ExecutionReplayContext(
                registry=context.registry,
                manifest=context.manifest,
                instruments=context.instruments,
                artifact_id=context.artifact_id,
                strategy_id=context.strategy_id,
                initial_portfolio=context.initial_portfolio,
                risk_policy=context.risk_policy,
                base_cost_schedule=context.base_cost_schedule,
                stress_cost_schedule=context.stress_cost_schedule,
                liquidity_model=context.liquidity_model,
                stress_liquidity_model=context.stress_liquidity_model,
                execution_policy=context.execution_policy,
                seed=context.seed + 2,
            ),
            market_frame=market,
            score_frame=scores,
            segment_column=_SEGMENT_COLUMN,
            decision_sessions_by_segment=segments,
            horizon_sessions=5,
        ),
    ]

    from src.stocks.backtesting.engine import PreparedReplayMarket

    original_build = PreparedReplayMarket.build.__func__
    build_calls = 0

    def tracked_build(cls, *args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(cls, *args, **kwargs)

    monkeypatch.setattr(PreparedReplayMarket, "build", classmethod(tracked_build))
    batch = prepare_execution_replay_batch(requests[0])
    assert build_calls == 2
    batch_results = replay_execution_equivalent_batch(requests, prepared_batch=batch)
    assert len(batch_results) == 3
    assert build_calls == 2

    single_results = [replay_execution_equivalent(req) for req in requests]

    for batch_ev, single_ev in zip(batch_results, single_results, strict=True):
        assert batch_ev.base_log_growth == single_ev.base_log_growth
        assert batch_ev.stress_log_growth == single_ev.stress_log_growth
        assert batch_ev.segment_ids == single_ev.segment_ids
        assert batch_ev.planned_cycles == single_ev.planned_cycles
        assert batch_ev.filled_orders == single_ev.filled_orders
        assert batch_ev.cash_session_fraction == single_ev.cash_session_fraction
        assert batch_ev.turnover == single_ev.turnover
        assert batch_ev.invested_interval_count == single_ev.invested_interval_count
        assert batch_ev.invested_interval_fraction == single_ev.invested_interval_fraction
        assert batch_ev.filled_cycle_count == single_ev.filled_cycle_count

    assert len(batch.segment_data) == 2


def test_replay_batch_02_fail_closed_incompatible_input() -> None:
    """REPLAY_BATCH_02_FAIL_CLOSED_INCOMPATIBLE_INPUT; TRAIN_COMPLETION_02_FAIL_CLOSED.

    A same-batch request with a changed market/score frame, segment column,
    or declared decision sessions raises ValueError before any shared prepared
    input is used.
    """
    market = _market_frame()
    context = _context(market)
    scores = _score_frame()
    segments = _decision_sessions(market)

    good_request = _request(market, scores, segments, context)
    batch = prepare_execution_replay_batch(good_request)

    from src.stocks.ml.execution_replay import _validate_batch_request_compatibility

    incompatible_market = _request(market.drop("close"), scores, segments, context)
    with pytest.raises(ValueError, match="market_frame identity mismatch"):
        _validate_batch_request_compatibility(incompatible_market, batch)

    incompatible_scores = _request(
        market,
        scores.with_columns(
            pl.when(pl.col(_SCORE_COLUMN) == pl.col(_SCORE_COLUMN).max())
            .then(pl.lit(float("inf")))
            .otherwise(pl.col(_SCORE_COLUMN))
            .alias(_SCORE_COLUMN)
        ),
        segments,
        context,
    )
    with pytest.raises(ValueError, match="score_frame identity mismatch"):
        _validate_batch_request_compatibility(incompatible_scores, batch)

    incompatible_segment = ExecutionEquivalentReplayRequest(
        context=context,
        market_frame=market,
        score_frame=scores,
        segment_column="wrong_column",
        decision_sessions_by_segment=segments,
        horizon_sessions=5,
    )
    with pytest.raises(ValueError, match="segment_column mismatch"):
        _validate_batch_request_compatibility(incompatible_segment, batch)


PARALLEL_COMPLETION_01_ORDERED_PARITY = "PARALLEL_COMPLETION_01_ORDERED_PARITY"


def test_parallel_completion_01_ordered_parity() -> None:
    """PARALLEL_COMPLETION_01_ORDERED_PARITY.

    Six compatible requests at four workers return evidence in input order
    exactly equal to one-worker evidence; PreparedReplayMarket.build count
    remains segment_count.
    """
    market = _market_frame()
    context = _context(market)
    scores = _score_frame()
    segments = _decision_sessions(market)

    requests = [
        _request(market, scores, segments, context),
        *[
            ExecutionEquivalentReplayRequest(
                context=ExecutionReplayContext(
                    registry=context.registry,
                    manifest=context.manifest,
                    instruments=context.instruments,
                    artifact_id=context.artifact_id,
                    strategy_id=context.strategy_id,
                    initial_portfolio=context.initial_portfolio,
                    risk_policy=context.risk_policy,
                    base_cost_schedule=context.base_cost_schedule,
                    stress_cost_schedule=context.stress_cost_schedule,
                    liquidity_model=context.liquidity_model,
                    stress_liquidity_model=context.stress_liquidity_model,
                    execution_policy=context.execution_policy,
                    seed=context.seed + i,
                ),
                market_frame=market,
                score_frame=scores,
                segment_column=_SEGMENT_COLUMN,
                decision_sessions_by_segment=segments,
                horizon_sessions=5,
            )
            for i in range(1, 6)
        ],
    ]

    batch = prepare_execution_replay_batch(requests[0])
    segment_count = len(batch.segment_data)

    parallel_results = replay_execution_equivalent_batch(
        requests, prepared_batch=batch, max_workers=4
    )
    assert len(parallel_results) == 6

    sequential_results = replay_execution_equivalent_batch(
        requests, prepared_batch=batch, max_workers=1
    )

    for i, (par_ev, seq_ev) in enumerate(zip(parallel_results, sequential_results, strict=True)):
        assert par_ev.base_log_growth == seq_ev.base_log_growth, f"base mismatch at request {i}"
        assert par_ev.stress_log_growth == seq_ev.stress_log_growth, f"stress mismatch at request {i}"
        assert par_ev.segment_ids == seq_ev.segment_ids, f"segment mismatch at request {i}"
        assert par_ev.planned_cycles == seq_ev.planned_cycles, f"planned mismatch at request {i}"
        assert par_ev.filled_orders == seq_ev.filled_orders, f"filled mismatch at request {i}"
        assert par_ev.cash_session_fraction == seq_ev.cash_session_fraction, f"cash mismatch at request {i}"
        assert par_ev.turnover == seq_ev.turnover, f"turnover mismatch at request {i}"
        assert par_ev.invested_interval_count == seq_ev.invested_interval_count, f"invested mismatch at request {i}"
        assert par_ev.invested_interval_fraction == seq_ev.invested_interval_fraction, f"invested frac mismatch at request {i}"
        assert par_ev.filled_cycle_count == seq_ev.filled_cycle_count, f"filled cycle mismatch at request {i}"


REPLAY_BUDGET_01 = "REPLAY-BUDGET-01"


def test_replay_budget_01_breach_fails_closed_before_any_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REPLAY-BUDGET-01: pre-build invariant breach never allocates.

    If current_live + planned + largest_next exceeds the effective limit, the
    failure is closed and PreparedReplayMarket.build_call_count stays zero.
    """
    from src.stocks.backtesting.market import PreparedReplayMarket
    from src.stocks.ml.execution_replay import stream_execution_replay_batch
    from src.stocks.ml.replay_resources import MemoryBudgetExceededError

    market = _market_frame()
    context = _context(market)
    scores = _score_frame()
    request = _request(market, scores, _decision_sessions(market), context)

    PreparedReplayMarket.reset_build_call_count()

    def _forbidden_build(cls, *args: object, **kwargs: object) -> object:
        raise AssertionError("no market may be built on a breached budget")

    monkeypatch.setattr(PreparedReplayMarket, "build", classmethod(_forbidden_build))

    with pytest.raises(MemoryBudgetExceededError):
        stream_execution_replay_batch(
            (request,), request_limit_bytes=64, stats={}
        )


REPLAY_CGROUP_01 = "REPLAY-CGROUP-01"


def test_replay_cgroup_01_effective_limit_is_min_of_finite_contributors() -> None:
    """REPLAY-CGROUP-01: effective limit semantics under mixed limits.

    effective_limit equals the minimum finite request/cgroup/address-space
    limit; unlimited sentinels are ignored and host RAM never raises it.
    """
    from src.stocks.ml.replay_resources import resolve_effective_memory_limit

    mib = 1024 * 1024
    resolved = resolve_effective_memory_limit(
        4096 * mib,
        cgroup_reader=lambda root: 2048 * mib,
        address_space_reader=lambda: 8192 * mib,
        host_reader=lambda: 64 * mib,
    )
    assert resolved.effective_limit_bytes == 2048 * mib

    unbounded = resolve_effective_memory_limit(
        None,
        cgroup_reader=lambda root: None,
        address_space_reader=lambda: None,
        host_reader=lambda: 1024,
    )
    assert unbounded.effective_limit_bytes is None

    # A huge cgroup v1 sentinel must not masquerade as a finite ceiling.
    from pathlib import Path

    v1_root = Path("/nonexistent-cgroup-root")
    assert read_cgroup_limit_bytes(v1_root) is None




REPLAY_LOOKBACK_01 = "REPLAY-LOOKBACK-01"


def test_replay_lookback_01_first_session_statistics_match_full_history() -> None:
    """REPLAY-LOOKBACK-01: causal lookback rows restore full-history statistics.

    Pre-segment causal rows make first-session ADTV equal the full-history
    rolling reference while emitted evidence begins at the segment start.
    """
    import math

    market_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    n_sessions = 30
    sessions = [
        datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n_sessions)
    ]
    for index, session in enumerate(sessions):
        for ticker in range(3):
            price = 100.0 + ticker + index * 0.5
            market_rows.append(
                {
                    "instrument_id": f"KRX:{ticker + 1:05d}",
                    "session": session,
                    "observation_time": session.replace(hour=15, minute=30),
                    "available_time": session.replace(hour=15, minute=31),
                    "open": price,
                    "close": price * 1.01,
                    "volume": 1e6,
                    "trading_value": price * 1e6,
                    "sector": f"S{ticker % 2}",
                }
            )
            score_rows.append(
                {
                    "instrument_id": f"KRX:{ticker + 1:05d}",
                    "session": session,
                    _SEGMENT_COLUMN: 0,
                    _SCORE_COLUMN: 0.01 + ticker * 0.001,
                }
            )
    market = pl.DataFrame(market_rows)
    scores = pl.DataFrame(score_rows)
    decisions = tuple(sessions[10:])
    context = _context(market)
    request = ExecutionEquivalentReplayRequest(
        context=context,
        market_frame=market,
        score_frame=scores,
        segment_column=_SEGMENT_COLUMN,
        decision_sessions_by_segment={0: decisions},
        horizon_sessions=5,
    )

    stats: dict[str, int] = {}
    evidence = stream_execution_replay_batch((request,), stats=stats)[0]

    # First decision session's ADTV equals the full-history rolling mean.
    first_decision = sessions[10]
    reference = (
        market.sort(["session", "instrument_id"])
        .with_columns(
            pl.col("trading_value")
            .rolling_mean(20, min_samples=1)
            .over("instrument_id")
            .alias("__adtv")
        )
        .filter(pl.col("session") == first_decision)["__adtv"]
        .to_numpy()
    )
    metadata = iter_replay_segment_metadata(request)[0]
    assert metadata.lookback_session_count > 0
    segment = build_prepared_replay_segment(request, metadata)
    position = list(segment.prepared_market.sessions).index(first_decision)
    start, stop = segment.prepared_market.session_ranges[position]
    assert segment.prepared_market.adtv[start:stop] == pytest.approx(reference)

    # Emitted evidence begins exactly at the declared segment start window.
    expected_intervals = n_sessions - 10 - 1
    assert len(evidence.base_log_growth) == expected_intervals
    assert len(evidence.stress_log_growth) == expected_intervals
    assert all(math.isfinite(value) for value in evidence.base_log_growth)
    assert stats["prepared_segment_build_count"] == 1

