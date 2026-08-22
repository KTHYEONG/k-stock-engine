"""Streaming execution replay contract tests.

Scenarios: STREAMING_REPLAY_01, RESOURCE_PLAN_01.
"""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind
from src.stocks.ml.contracts import (
    NetAlphaTrainingRequest,
)
from src.stocks.ml.execution_replay import (
    ExecutionEquivalentReplayRequest,
    ExecutionReplayContext,
    ReplayResourcePlan,
    instruments_from_frame,
    plan_execution_replay_resources,
    prepare_execution_replay_batch,
    replay_execution_equivalent_batch,
    stream_execution_replay_batch,
)


def _make_fixture(n_segments: int = 2, sessions_per_seg: int = 6, n_tickers: int = 3):
    """Build a deterministic two-segment market/score fixture."""
    market_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    decision_sessions: dict[int, list[datetime]] = {}
    for seg in range(n_segments):
        for idx in range(sessions_per_seg):
            session = datetime(2024, 1, 1 + seg * 12 + idx, tzinfo=UTC)
            decision_sessions.setdefault(seg, []).append(session)
            for t in range(n_tickers):
                price = 100.0 + t + idx * 0.1
                market_rows.append({
                    "instrument_id": f"KRX:{t + 1:05d}",
                    "session": session,
                    "observation_time": session.replace(hour=15, minute=30),
                    "available_time": session.replace(hour=15, minute=31),
                    "open": price,
                    "close": price * 1.01,
                    "volume": 1_000_000.0,
                    "trading_value": price * 1_000_000.0,
                    "sector": f"S{t % 2}",
                    "adtv": price * 1_000_000.0,
                })
                score_rows.append({
                    "instrument_id": f"KRX:{t + 1:05d}",
                    "session": session,
                    "oof_segment_id": seg,
                    "predicted_net_alpha": 0.01 + t * 0.001,
                    "expected_active_alpha": 0.01 + t * 0.001,
                    "alpha_lower_bound": 0.0,
                    "expected_net_alpha": 0.01 + t * 0.001,
                    "net_alpha_lower_bound": 0.0,
                    "exit_cost_rate": 0.001,
                })

    market = pl.DataFrame(market_rows)
    scores = pl.DataFrame(score_rows)
    segments = {s: tuple(sessions) for s, sessions in decision_sessions.items()}

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
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 1, 6, tzinfo=UTC),
        generated_time=datetime(2024, 1, 6, tzinfo=UTC),
        row_count=market.height,
    )
    return market, scores, segments, manifest


class TestStreamingReplay:
    """STREAMING_REPLAY_01."""

    def test_streaming_reuses_supplied_prepared_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RSS_REPLAY_01: supplied batch is reused without another preparation."""
        market, scores, segments, manifest = _make_fixture()
        request = NetAlphaTrainingRequest(
            artifact_id="test_streaming",
            candidate_horizon_sessions=(10,),
        )
        profile = request.policy_profiles[0]

        context = ExecutionReplayContext(
            registry=__import__("src.stocks.research.artifacts", fromlist=["ModelArtifactRegistry"]).ModelArtifactRegistry(
                __import__("pathlib").Path("mem://test")
            ),
            manifest=manifest,
            instruments=instruments_from_frame(market),
            artifact_id="test_streaming",
            strategy_id="test_streaming",
            initial_portfolio=__import__("src.core.portfolio", fromlist=["PortfolioSnapshot"]).PortfolioSnapshot(
                account_snapshot_id="oof",
                as_of=min(segments[0]),
                settled_cash=request.portfolio.initial_cash,
                unsettled_cash=0.0,
                positions=(),
            ),
            risk_policy=__import__("src.stocks.ml.training", fromlist=["_risk_policy_for_profile"])._risk_policy_for_profile(
                request, profile, 10,
                rebalance_frequency_sessions=5,
                top_k=20,
            ),
            base_cost_schedule=__import__("src.core.costs", fromlist=["default_base_schedule"]).default_base_schedule(),
            stress_cost_schedule=__import__("src.core.costs", fromlist=["default_stress_schedule"]).default_stress_schedule(),
            liquidity_model=None,
            stress_liquidity_model=None,
            execution_policy=__import__("src.stocks.domain.execution_policy", fromlist=["SCHEDULED_OPEN_V1"]).SCHEDULED_OPEN_V1,
            seed=42,
        )

        replay_request = ExecutionEquivalentReplayRequest(
            context=context,
            market_frame=market,
            score_frame=scores,
            segment_column="oof_segment_id",
            decision_sessions_by_segment=segments,
            horizon_sessions=10,
        )

        batch = prepare_execution_replay_batch(replay_request)
        legacy_evidences = replay_execution_equivalent_batch(
            (replay_request,), prepared_batch=batch
        )

        def fail_if_prepared(*args: object, **kwargs: object) -> object:
            raise AssertionError("streaming must reuse the supplied prepared batch")

        monkeypatch.setattr(
            "src.stocks.ml.execution_replay.prepare_execution_replay_batch",
            fail_if_prepared,
        )

        stream_results = list(stream_execution_replay_batch(
            (replay_request,), ReplayResourcePlan(
                max_workers=1, max_prepared_segments=1, projected_peak_bytes=0
            ), prepared_batch=batch,
        ))
        assert len(stream_results) == 1

        legacy = legacy_evidences[0]
        streaming = stream_results[0]
        assert len(legacy.base_log_growth) == len(streaming.base_log_growth)
        for lb, ls in zip(legacy.base_log_growth, streaming.base_log_growth, strict=True):
            assert lb == pytest.approx(ls, abs=1e-12)
        assert legacy.filled_orders == streaming.filled_orders
        assert legacy.turnover == pytest.approx(streaming.turnover, abs=1e-12)


class TestResourcePlan:
    """RESOURCE_PLAN_01."""

    def test_one_worker_when_headroom_sufficient(self) -> None:
        """700 MiB headroom, 600 MiB prepared segments, requested=2 -> chooses 1 worker."""
        plan = plan_execution_replay_resources(
            available_bytes=700 * 1024 * 1024,
            prepared_segment_bytes=600 * 1024 * 1024,
            requested_workers=2,
        )
        assert plan.max_workers == 1
        assert plan.max_prepared_segments == 1

    def test_unavailable_when_insufficient_headroom(self) -> None:
        """RSS_REPLAY_02: less than one segment plus reserve admits no worker."""
        plan = plan_execution_replay_resources(
            available_bytes=100 * 1024 * 1024,
            prepared_segment_bytes=600 * 1024 * 1024,
            requested_workers=2,
        )
        assert plan.max_workers == 0
        assert plan.max_prepared_segments == 0
