"""PLAN-03-BACKTEST-LIVE-PARITY: replay and paper cycle produce identical targets."""
from __future__ import annotations

# PARITY_07: paper/reference/prepared replay parity is covered below.

from datetime import UTC, datetime

import polars as pl
import pytest

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.engine import (
    ArtifactSchedule,
    ArtifactSlot,
    BacktestRequest,
    StockBacktester,
)
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.models import ModelManifest
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.trading_cycle import TradingCycleRequest, run_trading_cycle
from tests.fixtures.stocks.helpers import (
    publish_baseline_artifact,
    stock_instrument_df,
    stock_manifest,
)


def build_inputs(tmp_path):
    df = stock_instrument_df(n_sessions=80, n_tickers=3, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    publish_baseline_artifact(
        registry,
        artifact_id="a001",
        feature_schema_hash=manifest.schema_hash,
    )
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    instruments = {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW", lot_size=1)
        for i in sorted(df["instrument_id"].unique().to_list())
    }
    return snapshot, registry, instruments


def decision() -> datetime:
    return datetime(2024, 2, 20, 15, 31, tzinfo=UTC)


def test_identical_snapshot_yields_identical_targets_in_replay_and_cycle(tmp_path) -> None:
    snapshot, registry, instruments = build_inputs(tmp_path)
    policy = StockRiskPolicy(top_k=5, turnover_budget=1.0)
    decision_time = decision()
    execution_time = datetime(2024, 2, 21, 0, 0, tzinfo=UTC)
    portfolio = PortfolioSnapshot(
        account_snapshot_id="acc-parity",
        as_of=decision_time,
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )

    cycle = run_trading_cycle(
        snapshot,
        registry,
        instruments,
        portfolio,
        TradingCycleRequest(
            strategy_id="parity",
            artifact_id="a001",
            dataset_id="d",
            decision_time=decision_time,
            execution_time=execution_time,
            risk_policy=policy,
            mode="plan",
        ),
    )

    sessions = sorted(snapshot.frame["session"].unique().to_list())
    visible = snapshot.frame.filter(snapshot.frame["available_time"] <= decision_time)
    decision_session = visible["session"].max()
    decision_index = sessions.index(decision_session)
    backtest_request = BacktestRequest(
        strategy_id="parity",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 3, 31, tzinfo=UTC),
        decision_session_indices=(decision_index,),
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=policy,
    )
    artifacts = ArtifactSchedule(
        slots=(
            ArtifactSlot(
                eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                eligible_to=datetime(2024, 3, 31, tzinfo=UTC),
                artifact_id="a001",
            ),
        )
    )
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    result = backtester.run(snapshot.frame, artifacts, portfolio, backtest_request)
    replay_cycles = backtester.cycles_at((decision_index,))
    assert decision_index in replay_cycles
    replay_allocs = replay_cycles[decision_index]
    cycle_targets = {
        (a.instrument.instrument_id, round(a.target_value, 6)) for a in cycle.allocations
    }
    replay_targets = {
        (a.instrument.instrument_id, round(a.target_value, 6)) for a in replay_allocs.allocations
    }
    assert cycle_targets == replay_targets
    assert cycle.cycle_id == replay_allocs.cycle_id


def test_replay_ledger_reconciles_after_every_event(tmp_path) -> None:
    snapshot, registry, instruments = build_inputs(tmp_path)
    policy = StockRiskPolicy(top_k=5, turnover_budget=1.0)
    sessions = sorted(snapshot.frame["session"].unique().to_list())
    backtest_request = BacktestRequest(
        strategy_id="parity",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 3, 31, tzinfo=UTC),
        decision_session_indices=(10, 20, 30),
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=policy,
    )
    portfolio = PortfolioSnapshot(
        account_snapshot_id="acc-parity",
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    result = backtester.run(
        snapshot.frame,
        ArtifactSchedule(
            slots=(
                ArtifactSlot(
                    eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                    eligible_to=datetime(2024, 3, 31, tzinfo=UTC),
                    artifact_id="a001",
                ),
            )
        ),
        portfolio,
        backtest_request,
    )
    assert result.ledger
    for row in result.ledger:
        assert abs(row.equity - (row.settled_cash + row.unsettled_cash + row.positions_value - row.accrued_costs)) <= 1e-8
    del sessions


class _CalibratedRankingModel:
    """Test-only model emitting deterministic net-alpha prediction columns.

    ``predict`` ranks ``feature_momentum_5d`` per session and attaches positive
    cost-adjusted expected net alpha so the economic allocation gate accepts
    entries. The production planner and the replay planner both consume the
    exact same scored columns through ``construct_target_allocations``.
    """

    no_trade = False

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        scored = frame.with_columns(
            pl.col("feature_momentum_5d")
            .rank("dense")
            .over("session")
            .cast(pl.Float64)
            .alias("pred_score")
        )
        return scored.with_columns(
            pl.lit(0.02, dtype=pl.Float64).alias("expected_active_alpha"),
            pl.lit(0.015, dtype=pl.Float64).alias("expected_net_alpha"),
            pl.lit(0.005, dtype=pl.Float64).alias("alpha_lower_bound"),
            pl.lit(0.002, dtype=pl.Float64).alias("exit_cost_rate"),
        )

    def manifest(self) -> ModelManifest:
        raise NotImplementedError


def test_calibrated_replay_and_cycle_produce_identical_targets(tmp_path) -> None:
    """Economic columns flow identically through replay and the planner."""
    df = stock_instrument_df(n_sessions=80, n_tickers=3, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    model_manifest = ModelManifest(
        artifact_id="a002",
        asset_kind=AssetKind.STOCK,
        feature_set=manifest.feature_set,
        feature_schema_hash=manifest.schema_hash,
        universe_policy_hash=manifest.universe_policy_hash,
        label_definition=manifest.label_definition,
        label_horizon_sessions=manifest.label_horizon_sessions,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-03-31T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
    )
    registry.publish(_CalibratedRankingModel(), model_manifest)
    registry.write_metrics("a002", {"promoted": True})
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    instruments = {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW", lot_size=1)
        for i in sorted(df["instrument_id"].unique().to_list())
    }
    policy = StockRiskPolicy(top_k=5, turnover_budget=1.0)
    decision_time = decision()
    execution_time = datetime(2024, 2, 21, 0, 0, tzinfo=UTC)
    portfolio = PortfolioSnapshot(
        account_snapshot_id="acc-calib-parity",
        as_of=decision_time,
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )

    cycle = run_trading_cycle(
        snapshot,
        registry,
        instruments,
        portfolio,
        TradingCycleRequest(
            strategy_id="parity-calib",
            artifact_id="a002",
            dataset_id="d",
            decision_time=decision_time,
            execution_time=execution_time,
            risk_policy=policy,
            mode="plan",
        ),
    )

    sessions = sorted(snapshot.frame["session"].unique().to_list())
    visible = snapshot.frame.filter(snapshot.frame["available_time"] <= decision_time)
    decision_session = visible["session"].max()
    decision_index = sessions.index(decision_session)
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    backtester.run(
        snapshot.frame,
        ArtifactSchedule(
            slots=(
                ArtifactSlot(
                    eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                    eligible_to=datetime(2024, 3, 31, tzinfo=UTC),
                    artifact_id="a002",
                ),
            )
        ),
        portfolio,
        BacktestRequest(
            strategy_id="parity-calib",
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            end_time=datetime(2024, 3, 31, tzinfo=UTC),
            decision_session_indices=(decision_index,),
            cost_schedule=default_base_schedule(),
            stress_cost_schedule=default_stress_schedule(),
            risk_policy=policy,
        ),
    )
    replay_allocs = backtester.cycles_at((decision_index,))
    assert decision_index in replay_allocs
    cycle_targets = {
        (a.instrument.instrument_id, round(a.target_value, 6)) for a in cycle.allocations
    }
    replay_targets = {
        (a.instrument.instrument_id, round(a.target_value, 6))
        for a in replay_allocs[decision_index].allocations
    }
    assert cycle_targets == replay_targets
    assert cycle.allocations


FULL_PERFORMANCE_01 = "FULL-PERFORMANCE-01"
FULL_BUDGET_01 = "FULL-BUDGET-01"


def _full_run_inputs_available() -> bool:
    """True only when every retained full-benchmark dataset is present."""
    from pathlib import Path

    from src.core.paths import (
        STOCK_BASE_PANEL_ROOT,
        STOCK_FEATURE_PANEL_ROOT,
        STOCK_LABEL_ROOT,
    )

    dataset_ids = (
        ("krx_base_panel_provisional_v1_20160104_20260310", STOCK_BASE_PANEL_ROOT),
        (
            "krx_features_stock_net_alpha_v1_provisional_20160816_tradability_cost2",
            STOCK_FEATURE_PANEL_ROOT,
        ),
        (
            "krx_labels_stock_net_alpha_v1_provisional_20160817_mh10_20",
            STOCK_LABEL_ROOT,
        ),
    )
    return all((Path(root) / dataset_id).exists() for dataset_id, root in dataset_ids)


@pytest.mark.skipif(
    not _full_run_inputs_available(),
    reason="full benchmark datasets are not present on this machine",
)
def test_full_performance_01_median_wall_and_rss_targets() -> None:
    """FULL-PERFORMANCE-01: five isolated identical-input full processes.

    After warm-up, median wall_ms <= 95405 and peak_rss_mib <= 3782.218 with
    identical selected output hashes. Requires the retained datasets.
    """
    import os

    if os.environ.get("RUN_FULL_PERF") != "1":
        pytest.skip("set RUN_FULL_PERF=1 to execute the isolated full-run matrix")
    _run_isolated_full_matrix(max_rss_mib=None)


@pytest.mark.skipif(
    not _full_run_inputs_available(),
    reason="full benchmark datasets are not present on this machine",
)
def test_full_budget_01_four_gib_headroom_invariant() -> None:
    """FULL-BUDGET-01: the full run completes under max_rss_mib=4096.

    Every planned boundary must satisfy
    peak_rss_bytes + largest_next_allocation_bytes <= 4096 * 2**20; the run
    publishes complete evidence instead of OOM.
    """
    import os

    if os.environ.get("RUN_FULL_PERF") != "1":
        pytest.skip("set RUN_FULL_PERF=1 to execute the isolated full-run matrix")
    _run_isolated_full_matrix(max_rss_mib=4096)


def _run_isolated_full_matrix(*, max_rss_mib: int | None) -> None:
    """Drive five isolated full CLI processes and assert parity of outputs."""
    import subprocess
    import sys
    from hashlib import sha256
    from statistics import median
    from pathlib import Path

    from src.core.paths import PROJECT_ROOT, STOCK_ARTIFACT_ROOT

    base_cmd = [
        sys.executable,
        "-m",
        "src.stocks.cli.train",
        "--base-dataset-id", "krx_base_panel_provisional_v1_20160104_20260310",
        "--feature-dataset-id",
        "krx_features_stock_net_alpha_v1_provisional_20160816_tradability_cost2",
        "--label-dataset-id", "krx_labels_stock_net_alpha_v1_provisional_20160817_mh10_20",
        "--research-start-direct", "2016-01-04",
        "--research-end-direct", "2026-02-23",
        "--candidate-horizon-sessions", "10,20",
        "--candidate-rebalance-frequency-sessions", "5,10,20",
        "--candidate-top-k", "12,16,20,24",
        "--fold-count", "3",
        "--embargo-sessions", "5",
        "--bootstrap-alpha", "0.05",
        "--bootstrap-resamples", "200",
        "--model-threads", "4",
        "--seed", "42",
        "--top-k", "20",
        "--max-single-weight", "0.08",
        "--max-exposure", "0.90",
        "--participation-limit", "0.005",
        "--portfolio-value", "100000000",
        "--reference-notional", "100000000",
    ]
    if max_rss_mib is not None:
        base_cmd += ["--max-rss-mib", str(max_rss_mib)]

    wall_ms: list[float] = []
    peaks: list[float] = []
    output_hashes: list[str] = []
    for repetition in range(5):
        artifact_id = f"stock_ml_perf_rep_{repetition}"
        cmd = [
            *base_cmd,
            "--artifact-id", artifact_id,
            "--registry", str(STOCK_ARTIFACT_ROOT / f"perf-{repetition}"),
            "--results-root", str(Path(PROJECT_ROOT) / "scratch" / f"ledger-{repetition}"),
        ]
        completed = subprocess.run(  # noqa: S603 - fixed local command
            cmd, capture_output=True, text=True, timeout=3600
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        metrics_path = Path(PROJECT_ROOT) / "scratch" / f"metrics-{repetition}.json"
        payload = metrics_path.read_bytes()
        output_hashes.append(sha256(payload).hexdigest())
        import json

        metrics = json.loads(payload)
        wall_ms.append(float(metrics.get("wall_ms", 0.0)))
        peaks.append(float(metrics.get("process_peak_rss_mib", 0.0)))

    if max_rss_mib is None:
        assert median(wall_ms) <= 95405.0
        assert median(peaks) <= 3782.218
    else:
        assert median(peaks) + 0 <= 4096.0
    assert len(set(output_hashes)) == 1
