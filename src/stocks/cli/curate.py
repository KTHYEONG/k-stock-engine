"""Stock curate CLI: migrate legacy feature files into a canonical dataset.

The destination roots default to ``src.core.paths`` repository-local paths and
never come from ``.env``. ``--dataset-id`` is explicit and versioned; an
existing identifier is rejected so a rewrite always creates a new dataset.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from src.core.datasets import DatasetCertification
from src.core.paths import STOCK_DATASET_ROOT, STOCK_FEATURE_SOURCE_ROOT
from src.stocks.data.curation import StockCurationRequest, curate_legacy_feature_panel

logger = logging.getLogger("stocks.cli.curate")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy *_feat.parquet files into a canonical curated dataset"
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--feature-allowlist-version", default="v1")
    parser.add_argument("--source-root", type=Path, default=STOCK_FEATURE_SOURCE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=STOCK_DATASET_ROOT)
    parser.add_argument(
        "--certification",
        type=DatasetCertification,
        choices=list(DatasetCertification),
        default=DatasetCertification.PROVISIONAL,
    )
    parser.add_argument("--calendar-hash", default="")
    parser.add_argument("--corporate-action-hash", default="")
    parser.add_argument("--cost-source-hash", default="")
    parsed = parser.parse_args(args)

    request = StockCurationRequest(
        dataset_id=parsed.dataset_id,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        feature_allowlist_version=parsed.feature_allowlist_version,
        certification=parsed.certification,
        calendar_hash=parsed.calendar_hash,
        corporate_action_hash=parsed.corporate_action_hash,
        cost_source_hash=parsed.cost_source_hash,
    )
    result = curate_legacy_feature_panel(
        parsed.source_root, parsed.dataset_root, request
    )
    logger.info(
        "curated %s: %s rows, %s source files, manifest at %s",
        result.dataset_id,
        result.row_count,
        result.source_file_count,
        result.content_manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
