"""Execution-equivalent replay adapter contract tests."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1
from src.stocks.ml.execution_replay import (
    ExecutionEquivalentReplayRequest,
    ExecutionReplayContext,
    ExecutionReplayEvidence,
    replay_execution_equivalent,
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
