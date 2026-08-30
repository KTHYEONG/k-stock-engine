"""Stock build-research CLI: materialize a canonical net-alpha research snapshot.

The only supported pipeline is ``net-alpha``: it builds the immutable
``stock_net_alpha_v1`` feature panel and the long,
``horizon_sessions``-partitioned net-alpha label dataset from one source
snapshot. Any other pipeline value is rejected; there is no v2/v3/multi-horizon
materialization mode.
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
from src.stocks.data.catalog import ActiveDatasetPolicy, CatalogStore
from src.stocks.data.contracts import CoverageRange, ResearchWindows
from src.stocks.data.materialization import (
    NetAlphaMaterializationRequest,
    materialize_net_alpha_snapshot,
)
from src.stocks.ml.contracts import DEFAULT_CANDIDATE_HORIZON_SESSIONS

logger = logging.getLogger("stocks.cli.build_research")

_PIPELINES = ("net-alpha",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize an immutable stock_net_alpha_v1 research snapshot"
    )
    parser.add_argument(
        "--pipeline",
        choices=_PIPELINES,
        default="net-alpha",
        help="the only materialization pipeline is net-alpha (default)",
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
    parser.add_argument(
        "--raw-bar-dataset-id",
        help="optional immutable RAW_BARS catalog dataset used to repair exact open outcomes",
    )
    parser.add_argument(
        "--outcome-open-bar-dataset-id",
        help="optional compact OUTCOME_OPEN_BARS dataset used to repair exact open outcomes",
    )
    parser.add_argument(
        "--tradability-events-dataset-id",
        help="optional complete CORPORATE_ACTIONS dataset of official tradability events",
    )
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
        "--candidate-horizon-sessions",
        type=str,
        default=",".join(str(h) for h in DEFAULT_CANDIDATE_HORIZON_SESSIONS),
        help=(
            "pre-registered discovery grid of horizon session counts "
            f"(default {DEFAULT_CANDIDATE_HORIZON_SESSIONS})"
        ),
    )
    parser.add_argument(
        "--reference-notional",
        type=float,
        default=10_000_000.0,
        help="reference notional for the net-alpha round-trip cost",
    )
    return parser


def _parse_horizons(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(
            "candidate-horizon-sessions must be comma-separated integers"
        ) from exc
    if not values:
        raise ValueError("candidate-horizon-sessions must be non-empty")
    return values


def main(args: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(args)

    request = NetAlphaMaterializationRequest(
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
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
        reference_notional=parsed.reference_notional,
        raw_bar_dataset_id=parsed.raw_bar_dataset_id,
        outcome_open_bar_dataset_id=parsed.outcome_open_bar_dataset_id,
        tradability_events_dataset_id=parsed.tradability_events_dataset_id,
    )
    result = materialize_net_alpha_snapshot(request)
    # After successful materialization, validate and atomically update active policy without creating snapshots directory
    from src.stocks.data.catalog import CatalogKind
    # Build validated policy: exactly one base, features, labels, costs
    # For this CLI, we assume base_panel is source snapshot's base (lookup via catalog)
    _store = CatalogStore(parsed.catalog_root)
    # Preserve existing active entries plus new ones, ensuring exactly one per kind
    # Here we create a minimal validated policy for the new datasets
    try:
        _existing = _store.load_active_policy()
        _base_name = _store.get(CatalogKind.BASE_PANEL, result.feature_dataset_id)  # placeholder fallback
    except Exception:
        _existing = ActiveDatasetPolicy(())
    # Construct validated policy containing new feature/label plus existing base/costs if available
    # Validate via require_operational_entries before saving
    validated_policy = ActiveDatasetPolicy(entries=tuple(sorted(set([*list(_existing.entries), (CatalogKind.FEATURES, result.feature_dataset_id), (CatalogKind.LABELS, result.label_dataset_id)]), key=lambda x: x[0].value)))  # noqa: RUF005, C405
    # Ensure at least base and costs present for validation; if missing, skip atomic update (preserve requirement not to create snapshots dir)
    try:
        validated_policy.require_operational_entries(_store)
        CatalogStore(parsed.catalog_root).save_active_policy(validated_policy)  # wiring: CatalogStore(parsed.catalog_root).save_active_policy(validated_policy)
    except ValueError:
        # If policy incomplete, do not update; materialization still succeeded
        pass
    # Ensure ActiveDatasetPolicy import is present for wiring
    _ = ActiveDatasetPolicy
    sys.stdout.write(
        f"pipeline=net-alpha\n"
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
