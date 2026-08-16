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
    assert args.candidate_horizon_sessions == "3,5,8,10,15,20"
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
