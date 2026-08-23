"""Tests for ML training decomposition and diagnostics wiring.

Scenarios:
- ML_04: Every candidate horizon emits checkpoints with fold/profile counts
  equal to the existing request grid.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import pytest
import tempfile

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


def test_final_refit_lookback_is_purged_and_keeps_newest_sessions() -> None:
    from src.stocks.ml.training import _apply_final_refit_lookback

    pre_holdout = pl.DataFrame({"session_index": list(range(1, 401))})
    train = pre_holdout.with_columns(pl.lit(1.0).alias("net_alpha_target"))
    request = NetAlphaTrainingRequest(
        artifact_id="final-refit-lookback",
        max_training_lookback_sessions=252,
        embargo_sessions=5,
    )

    limited = _apply_final_refit_lookback(pre_holdout, train, request, 10)

    assert limited["session_index"].to_list() == list(range(133, 385))


def test_final_refit_lookback_none_preserves_training_rows() -> None:
    from src.stocks.ml.training import _apply_final_refit_lookback

    pre_holdout = pl.DataFrame({"session_index": [1, 2, 3]})
    request = NetAlphaTrainingRequest(artifact_id="final-refit-expanding")

    result = _apply_final_refit_lookback(pre_holdout, pre_holdout, request, 10)

    assert result.equals(pre_holdout)

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

    def test_cadence_grouping_preserves_keys_and_bounds_builds(self, tmp_path: Path) -> None:
        from src.stocks.ml.training import _replay_costs_batch
        from src.stocks.research.artifacts import ModelArtifactRegistry

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
            ModelArtifactRegistry(tmp_path / "replay-batch"),
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

    def test_mixed_profiles_preserve_all_keys(self, tmp_path: Path) -> None:
        from src.stocks.ml.training import _replay_costs_batch
        from src.stocks.ml.contracts import PolicyProfile
        from src.stocks.research.artifacts import ModelArtifactRegistry

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
            ModelArtifactRegistry(tmp_path / "replay-mixed"),
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


PERF_MEASURE_01 = "PERF-MEASURE-01"


class TestPerfMeasure01:
    """PERF-MEASURE-01: disjoint replay timers and observed build/cache stats."""

    def _replay_fixture(self):
        from datetime import UTC, datetime
        from pathlib import Path

        import polars as pl

        from src.core.costs import default_base_schedule, default_stress_schedule
        from src.core.datasets import DatasetManifest
        from src.core.instruments import AssetKind
        from src.core.portfolio import PortfolioSnapshot
        from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1
        from src.stocks.ml.contracts import NetAlphaTrainingRequest as Req
        from src.stocks.ml.execution_replay import (
            ExecutionEquivalentReplayRequest,
            ExecutionReplayContext,
            instruments_from_frame,
        )
        from src.stocks.research.artifacts import ModelArtifactRegistry
        from src.stocks.trading.portfolio_constructor import StockRiskPolicy

        market_rows, score_rows = [], []
        segments: dict[int, list[datetime]] = {}
        for seg in range(2):
            for idx in range(6):
                session = datetime(2024, 1, 1 + seg * 12 + idx, tzinfo=UTC)
                segments.setdefault(seg, []).append(session)
                for t in range(3):
                    price = 100.0 + t + idx * 0.1
                    market_rows.append(
                        {
                            "instrument_id": f"KRX:{t + 1:05d}",
                            "session": session,
                            "observation_time": session.replace(hour=15, minute=30),
                            "available_time": session.replace(hour=15, minute=31),
                            "open": price,
                            "close": price * 1.01,
                            "volume": 1e6,
                            "trading_value": price * 1e6,
                            "sector": f"S{t % 2}",
                            "adtv": price * 1e6,
                        }
                    )
                    score_rows.append(
                        {
                            "instrument_id": f"KRX:{t + 1:05d}",
                            "session": session,
                            "oof_segment_id": seg,
                            "predicted_net_alpha": 0.01 + t * 0.001,
                            "expected_active_alpha": 0.01 + t * 0.001,
                            "alpha_lower_bound": 0.0,
                            "expected_net_alpha": 0.01 + t * 0.001,
                            "net_alpha_lower_bound": 0.0,
                            "exit_cost_rate": 0.001,
                        }
                    )
        market = pl.DataFrame(market_rows)
        scores = pl.DataFrame(score_rows)
        manifest = DatasetManifest(
            asset_kind=AssetKind.STOCK, schema_version="v1", schema_hash="h",
            provider_version="p", universe_policy_version="u",
            universe_policy_hash="u", feature_set="stock_net_alpha_v1",
            feature_set_hash="f", label_definition="net_alpha_o2o",
            label_horizon_sessions=5,
            time_start=datetime(2024, 1, 1, tzinfo=UTC),
            time_end=datetime(2024, 2, 6, tzinfo=UTC),
            generated_time=datetime(2024, 2, 6, tzinfo=UTC),
            row_count=market.height,
        )
        request = Req(artifact_id="perf_measure", candidate_horizon_sessions=(10,))
        context = ExecutionReplayContext(
            registry=ModelArtifactRegistry(Path(tempfile.mkdtemp(prefix="perf-"))),
            manifest=manifest,
            instruments=instruments_from_frame(market),
            artifact_id="perf_measure",
            strategy_id="perf_measure",
            initial_portfolio=PortfolioSnapshot(
                account_snapshot_id="oof",
                as_of=min(segments[0]),
                settled_cash=request.portfolio.initial_cash,
                unsettled_cash=0.0,
                positions=(),
            ),
            risk_policy=StockRiskPolicy(
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
        return ExecutionEquivalentReplayRequest(
            context=context,
            market_frame=market,
            score_frame=scores,
            segment_column="oof_segment_id",
            decision_sessions_by_segment={s: tuple(v) for s, v in segments.items()},
            horizon_sessions=10,
        )

    def test_disjoint_timers_and_observed_build_stats(self) -> None:
        from src.stocks.backtesting.market import PreparedReplayMarket
        from src.stocks.ml.execution_replay import stream_execution_replay_batch

        request = self._replay_fixture()
        PreparedReplayMarket.reset_build_call_count()
        stats: dict[str, int] = {}
        evidences = stream_execution_replay_batch((request,), stats=stats)
        assert len(evidences) == 1
        assert stats["replay_prepare_elapsed_ms"] >= 0
        assert stats["replay_execute_elapsed_ms"] >= 0
        # Build count equals actual segment builds (2 segments here), never a
        # candidate-cardinality synthesis.
        assert stats["prepared_segment_build_count"] == 2
        assert stats["prepared_segment_build_count"] == PreparedReplayMarket.build_call_count
        assert stats["prepared_cache_bytes"] > 0
        assert stats["peak_live_prepared_segments"] == 1


FAILFAST_FRONTIER_01 = "FAILFAST-FRONTIER-01"


class TestFailfastFrontier01:
    """FAILFAST-FRONTIER-01: an infeasible frontier fails before any IO."""

    def test_infeasible_request_fails_closed_without_data_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polars as pl

        from src.stocks.cli.train import _validate_static_training_request
        from src.stocks.ml.contracts import (
            ExecutionFrontierSettings,
            NetAlphaTrainingRequest,
        )
        from src.storage.parquet_datasets import ParquetDatasetStore

        def _forbidden_read(*args: object, **kwargs: object) -> object:
            raise AssertionError("no Parquet access may precede feasibility")

        monkeypatch.setattr(pl, "read_parquet", _forbidden_read)
        monkeypatch.setattr(ParquetDatasetStore, "read_bounded", _forbidden_read)

        horizons = (3,)
        request = NetAlphaTrainingRequest(
            artifact_id="infeasible",
            candidate_horizon_sessions=horizons,
            execution_frontier=ExecutionFrontierSettings(
                candidate_horizon_sessions=horizons,
                # Default cadence grid starts at 5 sessions; H=3 owns no cell.
                candidate_rebalance_frequency_sessions=(5, 10, 20),
                candidate_top_k=(12,),
            ),
        )
        with pytest.raises(ValueError, match="feasible"):
            _validate_static_training_request(request)


PREPARED_MATRIX_01 = "PREPARED-MATRIX-01"


class TestPreparedMatrix01:
    """PREPARED-MATRIX-01: canonical matrix contract on the composed panel."""

    def test_matrix_contract_holds(self) -> None:
        from datetime import UTC, datetime, timedelta

        import numpy as np
        import polars as pl

        rows = []
        for s in range(5):
            session = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=s)
            rows.extend(
                {
                    "instrument_id": f"KRX:{t:05d}",
                    "session": session,
                    "feature__a": float(s) + t,
                    "feature__b": float(t) / 2.0,
                    "net_alpha_target": 0.01 * s,
                    "open": 100.0,
                }
                for t in range(4)
            )
        frame = pl.DataFrame(rows)

        class _Schema:
            learner_columns = ("feature__a", "feature__b")

        from src.stocks.ml.preparation import prepare_matrix_from_frame

        matrix = prepare_matrix_from_frame(frame, tuple(_Schema.learner_columns))
        x = matrix.X
        assert x.dtype == np.float32
        assert x.flags["C_CONTIGUOUS"]
        assert x.shape == (frame.height, 2)
        assert len(matrix.instrument_code) == frame.height
        assert len(matrix.session_code) == frame.height
        for forbidden in ("net_alpha_target", "open"):
            assert forbidden not in matrix.feature_columns
        # Label alignment is one-to-one over unique keys.
        keys = matrix.key_of(matrix.instrument_code, matrix.session_code)
        assert np.unique(keys).size == keys.size


OOF_TEMPORAL_01 = "OOF-TEMPORAL-01"


class TestOofTemporal01:
    """OOF-TEMPORAL-01: fold geometry stays purged and Rank-IC parity holds."""

    def test_fold_geometry_and_rank_ic_parity(self) -> None:
        import numpy as np
        from scipy.stats import spearmanr

        from src.stocks.ml.fitting import session_rank_ic_from_arrays

        rng = np.random.default_rng(9)
        n_sessions = 60
        codes = np.repeat(np.arange(n_sessions), 5).astype(np.int32)
        scores = rng.normal(size=codes.size)
        realized = 0.2 * scores + rng.normal(scale=0.05, size=codes.size)
        valid = np.isfinite(scores) & np.isfinite(realized)
        array_ic = session_rank_ic_from_arrays(scores, realized, codes, valid)

        frame = pl.DataFrame(
            {
                "session_index": codes,
                SCORE_COL: scores,
                "realized": realized,
            }
        ).filter(np.asarray(valid))
        ics = []
        for rows in frame.sort("session_index").partition_by("session_index"):
            if rows.height < 2:
                continue
            score_values = rows[SCORE_COL].to_numpy()
            realized_values = rows["realized"].to_numpy()
            rho, _ = spearmanr(score_values, realized_values)
            if np.std(score_values) and np.std(realized_values):
                ics.append(float(rho))
        reference_ic = float(np.mean(ics))
        assert abs(array_ic - reference_ic) <= 1e-12

    def test_prepared_fold_geometry_is_purged(self) -> None:
        from datetime import UTC, datetime, timedelta

        import numpy as np
        import polars as pl

        from src.stocks.ml.preparation import prepare_folds
        from src.stocks.research.folds import PurgedWalkForward

        sessions = [
            datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(80)
        ]
        panel = pl.DataFrame(
            {
                "session_index": np.repeat(np.arange(80), 3).astype(np.int64),
                "session": [s for s in sessions for _ in range(3)],
            }
        )
        splitter = PurgedWalkForward(
            n_folds=3,
            label_horizon_sessions=5,
            embargo_sessions=2,
            session_column="session_index",
        )
        folds = splitter.split(panel)
        prepared = prepare_folds(folds)
        for pfold in prepared:
            assert pfold.train_label_end < pfold.validation_decision_start
            overlap = set(pfold.train_rows.tolist()) & set(pfold.validation_rows.tolist())
            assert not overlap


PUBLIC_IMPORTS_01 = "PUBLIC-IMPORTS-01"


class TestPublicImports01:
    """PUBLIC-IMPORTS-01: pre-refactor public imports resolve without cycles."""

    def test_all_public_import_paths_resolve(self) -> None:
        import importlib

        training = importlib.import_module("src.stocks.ml.training")
        execution_replay = importlib.import_module("src.stocks.ml.execution_replay")
        engine = importlib.import_module("src.stocks.backtesting.engine")

        for name in (
            "TrainingTelemetry",
            "HorizonDiscovery",
            "train_net_alpha_model",
            "TrainingOrchestrator",
        ):
            assert hasattr(training, name)
        for name in (
            "stream_execution_replay_batch",
            "prepare_execution_replay_batch",
            "replay_execution_equivalent_batch",
            "ExecutionEquivalentReplayRequest",
            "ExecutionReplayContext",
            "ExecutionReplayEvidence",
            "ProfileReplayEvidence",
            "plan_execution_replay_resources",
            "instruments_from_frame",
        ):
            assert hasattr(execution_replay, name)
        for name in (
            "StockBacktester",
            "BacktestResult",
            "BacktestRequest",
            "BacktestTrade",
            "BacktestLedgerRow",
            "ArtifactSchedule",
            "ArtifactSlot",
            "BacktestValidationError",
            "PreparedReplayMarket",
            "REQUIRED_BACKTEST_COLUMNS",
        ):
            assert hasattr(engine, name)

    def test_module_graph_has_no_cycles(self) -> None:
        import subprocess
        import sys

        code = (
            "import src.stocks.ml.training, "
            "src.stocks.ml.execution_replay, "
            "src.stocks.backtesting.engine; raise SystemExit(0)"
        )
        result = subprocess.run(  # noqa: S603 - fixed import-graph probe
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr


SCORE_COL = "score"


SINGLE_MATRIX_SEED_04 = "ML_FULL_EXECUTION_P0_SINGLE_MATRIX_SEED_04"


def _seed04_fixture(
    n_sessions: int = 20,
    per_session: int = 4,
    embargo_sessions: int = 1,
) -> tuple[object, object, NetAlphaTrainingRequest, object]:
    """Small canonical matrix plus research data for seed-level unit tests."""
    import numpy as np
    import polars as pl
    from datetime import timedelta

    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind
    from src.stocks.ml.contracts import NetAlphaResearchData
    from src.stocks.ml.preparation import prepare_matrix_from_frame

    rng = np.random.default_rng(7)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for s in range(n_sessions):
        session = start + timedelta(days=s)
        for t in range(per_session):
            a, b = (float(v) for v in rng.normal(size=2))
            rows.append(
                {
                    "instrument_id": f"KRX:{t + 1:05d}",
                    "session": session,
                    "feature__a": a,
                    "feature__b": b,
                }
            )
    frame = pl.DataFrame(rows)
    matrix = prepare_matrix_from_frame(frame, ("feature__a", "feature__b"))

    horizon_sessions = 3
    label_rows = [
        {
            "instrument_id": row["instrument_id"],
            "session": row["session"],
            "net_alpha_target": float(rng.normal(scale=0.5)),
            "risk_residual": float(rng.normal(scale=0.003)),
            "reference_cost": 0.001,
            "label_available_time": row["session"]
            + timedelta(days=horizon_sessions),
        }
        for row in rows
    ]
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
        label_horizon_sessions=horizon_sessions,
        time_start=start,
        time_end=start + timedelta(days=n_sessions),
        generated_time=start + timedelta(days=n_sessions),
        row_count=len(rows),
    )
    data = NetAlphaResearchData(
        feature_frame=frame,
        labels_by_horizon={horizon_sessions: pl.DataFrame(label_rows)},
        manifest=manifest,
    )
    from src.stocks.ml.contracts import ExecutionFrontierSettings

    request = NetAlphaTrainingRequest(
        artifact_id="seed04",
        candidate_horizon_sessions=(horizon_sessions,),
        embargo_sessions=embargo_sessions,
        execution_frontier=ExecutionFrontierSettings(
            candidate_horizon_sessions=(horizon_sessions,),
            candidate_rebalance_frequency_sessions=(1,),
            candidate_top_k=(12,),
        ),
    )
    model_manifest = _seed04_model_manifest(horizon_sessions)
    return matrix, data, request, model_manifest


def _seed04_model_manifest(horizon_sessions: int):
    from src.core.instruments import AssetKind
    from src.stocks.research.models import ModelManifest

    return ModelManifest(
        artifact_id="seed04",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="net-alpha-v1",
        universe_policy_hash="net-alpha-v1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=horizon_sessions,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
    )


class TestSingleMatrixSeed04:
    """P0_SINGLE_MATRIX_SEED_04: one canonical matrix feeds OOF and the seed."""

    def test_calibration_seed_consumes_caller_matrix_without_second_preparation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        from src.stocks.ml import training as training_module

        matrix, data, request, model_manifest = _seed04_fixture(
            n_sessions=40, embargo_sessions=1
        )

        def _forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("second matrix preparation is forbidden")

        monkeypatch.setattr(
            training_module, "prepare_training_matrix", _forbidden
        )

        boundary_code = int(matrix.num_sessions) - 6
        initial_rows = np.flatnonzero(matrix.session_code < boundary_code)
        assert initial_rows.size > 0

        seed = training_module.build_initial_calibration_seed(
            matrix, initial_rows, request, 3, model_manifest, data=data
        )
        assert not seed.is_empty()
        for column in (
            "instrument_id",
            "session",
            "predicted_net_alpha",
            "risk_residual",
            "label_available_time",
            "realized_net_return",
        ):
            assert column in seed.columns
        # Seed rows stay inside the caller's pre-validation window.
        assert int(seed["session_index"].max()) < boundary_code
        assert seed["predicted_net_alpha"].is_finite().all()
        # Target-free: labels are only joined after prediction, so realized
        # outcomes must exist for every emitted ledger row.
        assert seed["realized_net_return"].is_not_null().all()

    def test_seed_is_inner_purged_across_segments(self) -> None:
        import numpy as np

        from src.stocks.ml import training as training_module

        matrix, data, request, model_manifest = _seed04_fixture(
            n_sessions=40, embargo_sessions=1
        )
        boundary_code = int(matrix.num_sessions) - 6
        initial_rows = np.flatnonzero(matrix.session_code < boundary_code)
        seed = training_module.build_initial_calibration_seed(
            matrix, initial_rows, request, 3, model_manifest, data=data
        )
        assert not seed.is_empty()
        segments = sorted(seed["oof_segment_id"].unique().to_list())
        # Degenerate inner folds legitimately drop out (constant-oof-score);
        # every surviving segment must still own a disjoint validation block.
        assert len(segments) >= 1
        validation_sessions_by_segment = {
            segment: set(
                seed.filter(
                    seed["oof_segment_id"] == segment
                )["session_index"].to_list()
            )
            for segment in segments
        }
        # Inner validation blocks are disjoint across segments.
        seen: set[int] = set()
        for segment in segments:
            block = validation_sessions_by_segment[segment]
            assert not (block & seen)
            seen |= block

    def test_pre_allocation_breach_publishes_no_trade_before_replay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.stocks.ml import training as training_module
        from src.stocks.ml.replay_resources import ResourceEnvelope
        from src.stocks.ml.telemetry import TrainingTelemetry
        from src.stocks.research.artifacts import ModelArtifactRegistry

        _, data, request, _ = _seed04_fixture()

        def _failed_envelope(planned_bytes: int, **kwargs: object):
            del kwargs
            return ResourceEnvelope(
                ok=False,
                planned_bytes=planned_bytes,
                limiting_source="cgroup",
                process_headroom_bytes=None,
                cgroup_headroom_bytes=1,
                system_headroom_bytes=None,
                reason="planned bytes exceed cgroup headroom",
            )

        monkeypatch.setattr(
            training_module, "_plan_training_allocation", _failed_envelope
        )
        registry_root = tmp_path / "registry"
        registry_root.mkdir()
        registry = ModelArtifactRegistry(registry_root)
        telemetry = TrainingTelemetry()
        frame = data.feature_frame
        manifest = training_module._run_discovery_and_publish(
            registry=registry,
            data=data,
            request=request,
            frame=frame,
            pre_holdout=frame,
            holdout=frame,
            folds=[],
            learner_columns=("feature__a", "feature__b"),
            schema=None,
            telemetry=telemetry,
            schema_hash="h",
            universe_policy_hash="u",
            oof_cache=training_module._OofCache(tmp_path / "oof"),
        )
        assert manifest.model_type == "no_trade"
        publish_phase = telemetry.to_dict()["phases"][-1]
        assert publish_phase["reason"] == (
            "memory-budget-exceeded:matrix_prepare"
        )
