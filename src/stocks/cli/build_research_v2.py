"""Stock build-research-v2 CLI: materialize a v2 snapshot from a source snapshot.

Requires explicit ids for the source snapshot, new feature dataset, new label
dataset, and new snapshot. There is no implicit ``latest`` selection. The CLI
prints the new feature id, label id, snapshot id, content hashes, coverage
threshold, row counts, and certification.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from src.core.datasets import DatasetCertification
from src.core.paths import (
    STOCK_BASE_PANEL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_FEATURE_PANEL_ROOT,
    STOCK_LABEL_ROOT,
)
from src.stocks.data.contracts import CoverageRange, ResearchWindows
from src.stocks.data.research_v2 import (
    StockAlphaV2MaterializationRequest,
    materialize_stock_alpha_v2_snapshot,
    materialize_stock_alpha_v3_snapshot,
)

logger = logging.getLogger("stocks.cli.build_research_v2")

LABEL_HORIZON_MODES = ("five_day", "multi_horizon")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize an immutable stock_alpha_v2 research snapshot"
    )
    parser.add_argument("--source-snapshot-id", required=True)
    parser.add_argument("--feature-dataset-id", required=True)
    parser.add_argument("--label-dataset-id", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--catalog-root", type=Path, default=STOCK_CATALOG_ROOT)
    parser.add_argument("--base-root", type=Path, default=STOCK_BASE_PANEL_ROOT)
    parser.add_argument("--feature-root", type=Path, default=STOCK_FEATURE_PANEL_ROOT)
    parser.add_argument("--label-root", type=Path, default=STOCK_LABEL_ROOT)
    parser.add_argument("--calendar-path", type=Path)
    parser.add_argument("--train-start", required=True, type=date.fromisoformat)
    parser.add_argument("--train-end", required=True, type=date.fromisoformat)
    parser.add_argument("--validation-start", required=True, type=date.fromisoformat)
    parser.add_argument("--validation-end", required=True, type=date.fromisoformat)
    parser.add_argument("--test-start", required=True, type=date.fromisoformat)
    parser.add_argument("--test-end", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--certification",
        type=DatasetCertification,
        choices=list(DatasetCertification),
        default=DatasetCertification.PROVISIONAL,
    )
    parser.add_argument("--generated-time", type=datetime.fromisoformat, default=None)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument(
        "--label-horizon-mode",
        choices=LABEL_HORIZON_MODES,
        default="five_day",
        help=(
            "five_day publishes the legacy single-horizon label dataset; "
            "multi_horizon publishes the immutable 5/10/15-session label panel "
            "so training can select a cost-amortizing route."
        ),
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(args)

    request = StockAlphaV2MaterializationRequest(
        source_snapshot_id=parsed.source_snapshot_id,
        feature_dataset_id=parsed.feature_dataset_id,
        label_dataset_id=parsed.label_dataset_id,
        snapshot_id=parsed.snapshot_id,
        catalog_root=parsed.catalog_root,
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
        generated_time=parsed.generated_time or datetime.now(UTC),
        windows=ResearchWindows(
            train=CoverageRange(parsed.train_start, parsed.train_end),
            validation=CoverageRange(parsed.validation_start, parsed.validation_end),
            test=CoverageRange(parsed.test_start, parsed.test_end),
        ),
        certification=parsed.certification,
        min_coverage=parsed.min_coverage,
        calendar_path=parsed.calendar_path,
    )
    if parsed.label_horizon_mode == "multi_horizon":
        result = materialize_stock_alpha_v3_snapshot(request)
    else:
        result = materialize_stock_alpha_v2_snapshot(request)
    sys.stdout.write(
        f"label_horizon_mode={parsed.label_horizon_mode}\n"
        f"feature_dataset_id={result.feature_dataset_id}\n"
        f"label_dataset_id={result.label_dataset_id}\n"
        f"snapshot_id={result.snapshot_id}\n"
        f"feature_content_hash={result.feature_content_hash}\n"
        f"label_content_hash={result.label_content_hash}\n"
        f"feature_rows={result.feature_row_count}\n"
        f"label_rows={result.label_row_count}\n"
        f"min_coverage={result.min_coverage}\n"
        f"certification={result.certification.value}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
