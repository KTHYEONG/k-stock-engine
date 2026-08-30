"""Build-research CLI: net-alpha-only materialization."""
from __future__ import annotations

import pytest

from src.stocks.cli import build_research


def test_build_research_cli_defaults_to_net_alpha() -> None:
    parser = build_research.build_parser()
    args = parser.parse_args(
        [
            "--source-snapshot-id",
            "source_v1",
            "--feature-dataset-id",
            "features_na",
            "--label-dataset-id",
            "labels_na",
            "--snapshot-id",
            "snap_na",
            "--train-start",
            "2024-01-01",
            "--train-end",
            "2024-01-31",
            "--validation-start",
            "2024-02-01",
            "--validation-end",
            "2024-02-29",
            "--test-start",
            "2024-03-01",
            "--test-end",
            "2024-03-31",
        ]
    )
    assert args.pipeline == "net-alpha"
    assert args.candidate_horizon_sessions == "10,20"
    assert args.raw_bar_dataset_id is None


def test_build_research_cli_rejects_legacy_pipeline() -> None:
    with pytest.raises(SystemExit):
        build_research.main(
            [
                "--pipeline",
                "multi_horizon",
                "--source-snapshot-id",
                "s",
                "--feature-dataset-id",
                "f",
                "--label-dataset-id",
                "l",
                "--snapshot-id",
                "snap",
                "--train-start",
                "2024-01-01",
                "--train-end",
                "2024-01-31",
                "--validation-start",
                "2024-02-01",
                "--validation-end",
                "2024-02-29",
                "--test-start",
                "2024-03-01",
                "--test-end",
                "2024-03-31",
            ]
        )


def test_build_research_cli_rejects_missing_required_ids() -> None:
    with pytest.raises(SystemExit):
        build_research.main(["--pipeline", "net-alpha"])


def test_build_research_materializes_without_active_policy_mutation(monkeypatch, tmp_path) -> None:
    from src.stocks.data.catalog import CatalogStore
    from src.stocks.data.materialization import NetAlphaMaterializationRequest, NetAlphaMaterializationResult
    from src.core.datasets import DatasetCertification

    # patch materialize to return dummy result without touching catalog
    dummy_result = NetAlphaMaterializationResult(snapshot_id="snap_new", feature_dataset_id="feat_new", label_dataset_id="label_new", feature_content_hash="h1", label_content_hash="h2", feature_row_count=10, label_row_count=10, min_coverage=0.75, certification=DatasetCertification.PROVISIONAL)

    def fake_materialize(req: NetAlphaMaterializationRequest):
        assert isinstance(req, NetAlphaMaterializationRequest)
        return dummy_result

    monkeypatch.setattr("src.stocks.cli.build_research.materialize_net_alpha_snapshot", fake_materialize)
    monkeypatch.setattr("src.stocks.data.materialization.materialize_net_alpha_snapshot", fake_materialize)
    # ensure save_active_policy not called
    called = {"save": 0}
    orig_save = CatalogStore.save_active_policy

    def counted_save(self, policy):
        called["save"] += 1
        return orig_save(self, policy)

    monkeypatch.setattr(CatalogStore, "save_active_policy", counted_save)

    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir(parents=True, exist_ok=True)
    # need minimal catalog store to avoid errors in build_research's internal handling (but we patched materialize, so no catalog needed)
    code = build_research.main(
        [
            "--source-snapshot-id",
            "src_v1",
            "--feature-dataset-id",
            "feat_new",
            "--label-dataset-id",
            "label_new",
            "--snapshot-id",
            "snap_new",
            "--catalog-root",
            str(catalog_root),
            "--train-start",
            "2024-01-01",
            "--train-end",
            "2024-01-31",
            "--validation-start",
            "2024-02-01",
            "--validation-end",
            "2024-02-29",
            "--test-start",
            "2024-03-01",
            "--test-end",
            "2024-03-31",
        ]
    )
    assert code == 0
    assert called["save"] == 0
