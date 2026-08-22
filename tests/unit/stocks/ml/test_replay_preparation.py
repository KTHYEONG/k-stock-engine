"""Segment metadata and one-segment preparation tests."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.market import PreparedReplayMarket
from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.ml.execution_replay import ExecutionEquivalentReplayRequest, ExecutionReplayContext, instruments_from_frame
from src.stocks.ml.replay_preparation import (
    ExecutionReplayBatchRequest,
    build_prepared_replay_segment,
    iter_prepared_replay_segments,
    iter_replay_segment_metadata,
)
from src.stocks.ml.replay_resources import (
    EffectiveMemoryLimit,
    MemoryBudgetExceededError,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy


def _fixture(n_sessions: int = 40, n_tickers: int = 2):
    market_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    sessions = [
        datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n_sessions)
    ]
    for session in sessions:
        for t in range(n_tickers):
            price = 100.0 + 0.1 * sessions.index(session)
            market_rows.append(
                {
                    "instrument_id": f"KRX:{t + 1:05d}",
                    "session": session,
                    "observation_time": session.replace(hour=15, minute=30),
                    "available_time": session.replace(hour=15, minute=31),
                    "open": price,
                    "close": price * 1.01,
                    "volume": 1_000_000.0,
                    "trading_value": price * 1_000_000.0,
                }
            )
            score_rows.append(
                {
                    "instrument_id": f"KRX:{t + 1:05d}",
                    "session": session,
                    "oof_segment_id": 0,
                    "predicted_net_alpha": 0.01,
                }
            )
    market = pl.DataFrame(market_rows)
    scores = pl.DataFrame(score_rows)
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
    request = NetAlphaTrainingRequest(artifact_id="t", candidate_horizon_sessions=(10,))
    context = ExecutionReplayContext(
        registry=ModelArtifactRegistry(Path("mem://prep")),
        manifest=manifest,
        instruments=instruments_from_frame(market),
        artifact_id="t",
        strategy_id="t",
        initial_portfolio=PortfolioSnapshot(
            account_snapshot_id="oof",
            as_of=sessions[0],
            settled_cash=100_000_000.0,
            unsettled_cash=0.0,
            positions=(),
        ),
        risk_policy=request.policy_profiles[0].__class__ and StockRiskPolicy(
            top_k=3, gross_cap=0.9, single_name_cap=0.3, sector_cap=0.5,
            participation_limit=0.01, no_trade_band_bps=0.0,
        ),
        base_cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        liquidity_model=None,
        stress_liquidity_model=None,
        execution_policy=SCHEDULED_OPEN_V1,
        seed=42,
    )
    replay_request = ExecutionEquivalentReplayRequest(
        context=context,
        market_frame=market,
        score_frame=scores,
        segment_column="oof_segment_id",
        decision_sessions_by_segment={0: tuple(sessions[20:])},
        horizon_sessions=10,
    )
    return replay_request


def test_metadata_includes_causal_lookback_window() -> None:
    replay_request = _fixture()
    metadata = iter_replay_segment_metadata(replay_request)[0]
    assert metadata.lookback_session_count >= 20 - 1 or metadata.lookback_session_count > 0
    assert metadata.window_sessions[0] < min(metadata.decision_sessions)
    assert max(metadata.decision_sessions) <= metadata.window_sessions[-1]
    assert metadata.row_estimate == len(replay_request.market_frame.filter(
        pl.col("session").is_in(list(metadata.window_sessions))
    ))


def test_prepared_market_adtv_matches_full_history_reference() -> None:
    replay_request = _fixture()
    metadata = iter_replay_segment_metadata(replay_request)[0]
    segment = build_prepared_replay_segment(replay_request, metadata)
    first_decision = min(metadata.decision_sessions)
    market = replay_request.market_frame
    reference = (
        market.sort(["session", "instrument_id"])
        .with_columns(
            pl.col("trading_value").rolling_mean(20, min_samples=1).over("instrument_id").alias("__adtv")
        )
        .filter(pl.col("session") == first_decision)
    )["__adtv"].to_numpy()
    position = list(segment.prepared_market.sessions).index(first_decision)
    range_start, range_stop = segment.prepared_market.session_ranges[position]
    prepared_first_adtv = segment.prepared_market.adtv[range_start:range_stop]
    assert prepared_first_adtv == pytest.approx(reference)


def test_budget_breach_fails_before_any_market_build() -> None:
    replay_request = _fixture()
    batch = ExecutionReplayBatchRequest(requests=(replay_request,))
    PreparedReplayMarket.reset_build_call_count()
    tiny_limit = EffectiveMemoryLimit(
        request_limit_bytes=64,
        cgroup_limit_bytes=None,
        address_space_limit_bytes=None,
        host_total_bytes=1024**4,
        effective_limit_bytes=64,
    )
    with pytest.raises(MemoryBudgetExceededError):
        list(iter_prepared_replay_segments(batch, tiny_limit))
    assert PreparedReplayMarket.build_call_count == 0
