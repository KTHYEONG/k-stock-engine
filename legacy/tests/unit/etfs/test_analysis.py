"""ETF stability analysis contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.datasets import make_manifest
from src.core.instruments import AssetKind
from legacy.etfs.data.contracts import EtfDataset
from legacy.etfs.domain.universe import KOSPI_ETF_UNIVERSE
from legacy.etfs.research.analysis import assess_stability, summarize_result
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


class TestAnalysis:
    def test_stability_report_checks_invariants(self) -> None:
        dataset = make_dataset(n_days=120)
        folds = build_walk_forward_folds(dataset, n_folds=2)
        report = run_walk_forward(dataset, KOSPI_ETF_UNIVERSE, IndexSwitchParams(), folds)
        stability = assess_stability(report)
        assert set(stability.reasons) == {
            "out_of_sample_growth",
            "max_drawdown_bounded",
            "profit_factor_above_floor",
            "sufficient_trades",
        }

    def test_summarize_result_extracts_scalar_inputs(self) -> None:
        dataset = make_dataset(n_days=120)
        folds = build_walk_forward_folds(dataset, n_folds=2)
        report = run_walk_forward(dataset, KOSPI_ETF_UNIVERSE, IndexSwitchParams(), folds)
        assert report.results
        summary = summarize_result(report.results[0].result)
        assert {"total_return_pct", "mdd_pct", "profit_factor", "total_trades"} <= set(summary)

    def test_stability_report_must_match_invariant_set(self) -> None:
        from legacy.etfs.research.analysis import StabilityReport

        with pytest.raises(ValueError, match="passed"):
            StabilityReport(
                mean_return_pct=-1.0,  # growth invariant fails
                mean_mdd_pct=1.0,
                mean_profit_factor=2.0,
                total_trades=10,
                passed=True,  # must be False when growth < 0
                reasons={},
            )
