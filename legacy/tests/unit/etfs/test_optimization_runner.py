"""ETF optimization runner contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

from src.core.datasets import make_manifest
from src.core.instruments import AssetKind
from legacy.etfs.data.contracts import EtfDataset
from legacy.etfs.domain.universe import KOSPI_ETF_UNIVERSE
from legacy.etfs.research.optimization_runner import OptimizationRequest, run_optimization
from legacy.etfs.research.search_space import SearchSpace
from tests.fixtures.etfs.helpers import make_etf_fixture


def make_dataset(n_days: int = 120) -> EtfDataset:
    index_df, etf_df = make_etf_fixture(n_days=n_days)
    manifest = make_manifest(
        asset_kind=AssetKind.ETF,
        columns=["ticker", "date", "OPNPRC_IDX", "HGPRC_IDX", "LWPRC_IDX", "CLSPRC_IDX"],
        feature_set="etf_switch_v1",
        label_definition="etf_switch_v1",
        label_horizon_sessions=5,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 6, 30, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=index_df.height,
    )
    return EtfDataset(manifest=manifest, index_frame=index_df, etf_frame=etf_df)


class TestOptimizationRunner:
    def test_optimization_report_carries_etf_asset_kind(self) -> None:
        dataset = make_dataset(n_days=120)
        request = OptimizationRequest(search_space=SearchSpace(), n_folds=2, max_trials=5)
        report = run_optimization(dataset, KOSPI_ETF_UNIVERSE, request)
        assert report.asset_kind is AssetKind.ETF
        assert report.trials == 5

    def test_optimization_uses_walk_forward_folds(self) -> None:
        dataset = make_dataset(n_days=120)
        request = OptimizationRequest(search_space=SearchSpace(), n_folds=3, max_trials=3)
        report = run_optimization(dataset, KOSPI_ETF_UNIVERSE, request)
        assert report.best_mean_mdd_pct is not None
