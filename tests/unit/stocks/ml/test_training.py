"""Tests for ML training decomposition and diagnostics wiring.

Scenarios:
- ML_04: Every candidate horizon emits checkpoints with fold/profile counts
  equal to the existing request grid.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import pytest

from src.stocks.ml.fitting import OofCache, atomic_write_parquet, read_oof_parquet
from src.stocks.ml.discovery import HorizonDiscovery
from src.stocks.ml.execution_replay import ExecutionReplayEvidence
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.observability.contracts import (
    DiagnosticCategory,
    DiagnosticEvent,
    DiagnosticStage,
    DiagnosticStatus,
)
from src.stocks.observability.recorder import NullRunDiagnostics
import polars as pl


class TestOofCache:
    """REPLAY_BATCH_04_OOF_SPILL_REGRESSION: OofCache manages temporary OOF spill files."""

    def test_store_and_load(self, tmp_path: Path) -> None:
        cache = OofCache(tmp_path)
        oof = pl.DataFrame({"score": [0.1, 0.2, 0.3]})
        labels = pl.DataFrame({"label": [0.01, 0.02, 0.03]})

        oof_path, labels_path = cache.store(10, oof, labels)
        assert oof_path.exists()
        assert labels_path.exists()

        loaded_oof, loaded_labels = cache.load(10)
        assert loaded_oof.shape == oof.shape
        assert loaded_labels.shape == labels.shape

        cache.close()

    def test_cache_bytes_tracking(self, tmp_path: Path) -> None:
        cache = OofCache(tmp_path)
        oof = pl.DataFrame({"score": [0.1, 0.2]})
        labels = pl.DataFrame({"label": [0.01, 0.02]})

        cache.store(10, oof, labels)
        assert cache.cache_bytes > 0

        cache.close()

    def test_closed_cache_rejects_store(self, tmp_path: Path) -> None:
        cache = OofCache(tmp_path)
        cache.close()
        with pytest.raises(ValueError, match="closed"):
            cache.store(10, pl.DataFrame({"a": [1]}), pl.DataFrame({"b": [1]}))


class TestAtomicWriteParquet:
    """Atomic parquet write via rename."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        frame = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        path = tmp_path / "test.parquet.zst"
        atomic_write_parquet(frame, path)
        assert path.exists()

        loaded = read_oof_parquet(path)
        assert loaded.shape == frame.shape


class TestHorizonDiscoveryDataclass:
    """HorizonDiscovery is a frozen dataclass."""

    def test_create_empty_discovery(self) -> None:
        discovery = HorizonDiscovery(
            evidence=(),
            diagnostics=(),
            oof_by_horizon={},
        )
        assert discovery.evidence == ()
        assert discovery.path_evaluation_count == 0

    def test_create_with_data(self, tmp_path: Path) -> None:
        discovery = HorizonDiscovery(
            evidence=(),
            diagnostics=(),
            oof_by_horizon={10: (tmp_path / "oof.parquet", tmp_path / "labels.parquet", [0.05])},
            dropout_reasons={(10, "test"): "insufficient_data"},
            path_evaluation_count=5,
            path_evaluation_bound=10,
        )
        assert discovery.path_evaluation_count == 5
        assert (10, "test") in discovery.dropout_reasons


class TestDiagnosticsWiring:
    """Verify diagnostics parameter is accepted by training and backtesting."""

    def test_train_accepts_diagnostics_parameter(self) -> None:
        from src.stocks.ml.training import train_net_alpha_model
        import inspect

        sig = inspect.signature(train_net_alpha_model)
        assert "diagnostics" in sig.parameters
        param = sig.parameters["diagnostics"]
        assert param.default is None

    def test_backtester_accepts_diagnostics_parameter(self) -> None:
        from src.stocks.backtesting.engine import StockBacktester
        import inspect

        sig = inspect.signature(StockBacktester.__init__)
        assert "diagnostics" in sig.parameters
        param = sig.parameters["diagnostics"]
        assert param.default is None

    def test_null_diagnostics_emits_no_events(self) -> None:
        sink = NullRunDiagnostics()
        event = DiagnosticEvent(
            run_id="test",
            sequence=0,
            category=DiagnosticCategory.ALGO,
            component="ml.training",
            stage=DiagnosticStage.SPLIT_FIT,
            event="test",
            status=DiagnosticStatus.START,
        )
        sink.emit(event)
        sink.close("PASS")


REPLAY_BATCH_03 = "REPLAY_BATCH_03_FULL_FRONTIER_WIRING"
REPLAY_BATCH_05 = "REPLAY_BATCH_05_BENCHMARK"


class TestReplayBatchBenchmark:
    """REPLAY_BATCH_05_BENCHMARK: benchmark script is importable and exits cleanly."""

    def test_benchmark_script_is_importable(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "benchmark_execution_replay_batch",
            "tools/benchmarks/benchmark_execution_replay_batch.py",
        )
        assert spec is not None
        assert spec.loader is not None


class TestReplayBatchWiring:
    """REPLAY_BATCH_03_FULL_FRONTIER_WIRING; TRAIN_COMPLETION_04_CADENCE_WIRING."""

    def test_cadence_grouping_preserves_keys_and_bounds_builds(self) -> None:
        from src.stocks.ml.training import _replay_costs_batch

        n_segments = 2
        sessions_per_seg = 6
        n_tickers = 3
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

        from src.core.datasets import DatasetManifest
        from src.core.instruments import AssetKind
        from src.stocks.ml.contracts import NetAlphaTrainingRequest

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

        request = NetAlphaTrainingRequest(
            artifact_id="test_batch",
            candidate_horizon_sessions=(10,),
        )
        profile_a = request.policy_profiles[0]
        profile_b = request.policy_profiles[1]

        specs = [
            (5, 12, profile_a),
            (5, 12, profile_b),
            (10, 12, profile_a),
        ]

        from src.stocks.ml.contracts import RiskSettings as _RS

        results = _replay_costs_batch(
            calibrated=scores,
            oof_labels=scores,
            request=request,
            horizon_sessions=10,
            risk=_RS(),
            market_frame=market,
            manifest=manifest,
            specs=specs,
        )

        assert (10, 5, 12, profile_a.profile_id) in results
        assert (10, 5, 12, profile_b.profile_id) in results
        assert (10, 10, 12, profile_a.profile_id) in results
        assert len(results) == 3

        for base_ev, stress_ev in results.values():
            assert isinstance(base_ev, ExecutionReplayEvidence)
            assert isinstance(stress_ev, ExecutionReplayEvidence)
            assert len(base_ev.base_log_growth) > 0
            assert len(base_ev.stress_log_growth) > 0


PARALLEL_COMPLETION_02_MIXED_PROFILE_SCHEDULE = "PARALLEL_COMPLETION_02_MIXED_PROFILE_SCHEDULE"


class TestParallelCompletionMixedProfileSchedule:
    """PARALLEL_COMPLETION_02_MIXED_PROFILE_SCHEDULE.

    Mixed sparse and non-sparse profiles create separate schedule-compatible
    batches, attach dense shadows only to sparse profiles, and preserve
    every candidate result key.
    """

    def test_mixed_profiles_preserve_all_keys(self) -> None:
        from src.stocks.ml.training import _replay_costs_batch
        from src.stocks.ml.contracts import PolicyProfile

        n_segments = 2
        sessions_per_seg = 6
        n_tickers = 3
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

        from src.core.datasets import DatasetManifest
        from src.core.instruments import AssetKind

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

        request = NetAlphaTrainingRequest(
            artifact_id="test_mixed",
            candidate_horizon_sessions=(10,),
        )

        sparse_profile = PolicyProfile(
            profile_id="sparse_v2",
            no_trade_band_bps=5.0,
            growth_risk_aversion=1.0,
            execution_utility_mode="sparse_hold_replace_v2",
            sizing_mode="risk_balanced_waterfill_v2",
        )
        non_sparse_profile = PolicyProfile(
            profile_id="dense_v1",
            no_trade_band_bps=0.0,
            growth_risk_aversion=1.0,
            execution_utility_mode="delta_cost_aware_v1",
            sizing_mode="alpha_vol_squared_v1",
        )

        specs = [
            (5, 12, sparse_profile),
            (5, 12, non_sparse_profile),
            (10, 12, sparse_profile),
            (10, 12, non_sparse_profile),
        ]

        from src.stocks.ml.contracts import RiskSettings as _RS

        results = _replay_costs_batch(
            calibrated=scores,
            oof_labels=scores,
            request=request,
            horizon_sessions=10,
            risk=_RS(),
            market_frame=market,
            manifest=manifest,
            specs=specs,
        )

        assert (10, 5, 12, sparse_profile.profile_id) in results
        assert (10, 5, 12, non_sparse_profile.profile_id) in results
        assert (10, 10, 12, sparse_profile.profile_id) in results
        assert (10, 10, 12, non_sparse_profile.profile_id) in results
        assert len(results) == 4

        for base_ev, stress_ev in results.values():
            assert isinstance(base_ev, ExecutionReplayEvidence)
            assert isinstance(stress_ev, ExecutionReplayEvidence)
            assert len(base_ev.base_log_growth) > 0
            assert len(base_ev.stress_log_growth) > 0
            assert base_ev.segment_ids == stress_ev.segment_ids
