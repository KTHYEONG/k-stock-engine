"""ETF walk-forward evaluation contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.datasets import make_manifest
from src.core.instruments import AssetKind
from legacy.etfs.backtesting.engine import EtfSimulationConfig
from legacy.etfs.data.contracts import EtfDataset
from legacy.etfs.domain.universe import KOSPI_ETF_UNIVERSE
from legacy.etfs.research.walk_forward import build_walk_forward_folds, run_walk_forward
from legacy.etfs.strategies.index_switch_v1 import IndexSwitchParams
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


class TestWalkForward:
    def test_folds_split_index_frame_deterministically(self) -> None:
        dataset = make_dataset(n_days=60)
        folds = build_walk_forward_folds(dataset, n_folds=3)
        assert len(folds) == 3
        assert folds[0].validation_start <= folds[-1].validation_end

    def test_run_walk_forward_returns_per_fold_results(self) -> None:
        dataset = make_dataset(n_days=120)
        folds = build_walk_forward_folds(dataset, n_folds=3)
        report = run_walk_forward(
            dataset, KOSPI_ETF_UNIVERSE, IndexSwitchParams(), folds,
            config=EtfSimulationConfig(),
        )
        assert report.results
        assert report.mean_return_pct is not None

    def test_insufficient_sessions_fail_closed(self) -> None:
        dataset = make_dataset(n_days=2)
        with pytest.raises(ValueError, match="folds"):
            build_walk_forward_folds(dataset, n_folds=3)
