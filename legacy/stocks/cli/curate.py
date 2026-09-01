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
from legacy.stocks.paths import STOCK_DATASET_ROOT, STOCK_FEATURE_SOURCE_ROOT
from legacy.stocks.data.curation import StockCurationRequest, curate_legacy_feature_panel
from legacy.stocks.data.evidence import (
    AvailabilityPolicy,
    feature_availability_from_disclosures,
    load_corporate_action_snapshot,
    load_disclosure_availability_records,
    load_instrument_master_snapshot,
    load_krx_calendar_snapshot,
)
from legacy.stocks.data.quality import FeatureAvailabilityRecord

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
    parser.add_argument("--instrument-master-path", type=Path)
    parser.add_argument("--calendar-path", type=Path)
    parser.add_argument("--corporate-actions-path", type=Path)
    parser.add_argument("--disclosure-availability-path", type=Path)
    parsed = parser.parse_args(args)

    instrument_master = (
        load_instrument_master_snapshot(parsed.instrument_master_path)
        if parsed.instrument_master_path
        else None
    )
    calendar = (
        load_krx_calendar_snapshot(parsed.calendar_path) if parsed.calendar_path else None
    )
    corporate_actions = None
    if parsed.corporate_actions_path:
        if calendar is None:
            raise ValueError("--corporate-actions-path requires --calendar-path")
        corporate_actions = load_corporate_action_snapshot(parsed.corporate_actions_path, calendar)
    feature_availability: tuple[FeatureAvailabilityRecord, ...] = ()
    if parsed.disclosure_availability_path:
        records = load_disclosure_availability_records(parsed.disclosure_availability_path)
        feature_availability = feature_availability_from_disclosures(
            records, AvailabilityPolicy(calendar=calendar)
        )

    if parsed.certification is not DatasetCertification.PROVISIONAL:
        missing = [
            name
            for name, value in (
                ("--instrument-master-path", instrument_master),
                ("--calendar-path", calendar),
                ("--corporate-actions-path", corporate_actions),
                ("--disclosure-availability-path", feature_availability),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"{parsed.certification.value} curation requires evidence artifacts: {missing}"
            )

    request = StockCurationRequest(
        dataset_id=parsed.dataset_id,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        feature_allowlist_version=parsed.feature_allowlist_version,
        certification=parsed.certification,
        calendar_hash=parsed.calendar_hash or (calendar.content_hash if calendar else ""),
        corporate_action_hash=parsed.corporate_action_hash
        or (corporate_actions.content_hash if corporate_actions else ""),
        cost_source_hash=parsed.cost_source_hash,
        instrument_master=instrument_master,
        corporate_actions=corporate_actions,
        calendar=calendar,
        feature_availability=feature_availability,
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
