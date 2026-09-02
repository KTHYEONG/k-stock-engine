"""Gold materialization for QVEF features."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.core.datasets import HIVE_PARTITION_LAYOUT, DatasetCertification, make_manifest
from src.core.instruments import AssetKind
from src.features.contracts import QvefFeaturePolicy, QvefFeatureRow
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash


def materialize_qvef_features(
    rows: tuple[QvefFeatureRow, ...],
    *,
    root: Path,
    dataset_id: str,
    decision_time: datetime,
    policy: QvefFeaturePolicy,
    provider_version: str,
    calendar_hash: str,
    master_hash: str,
    quality_report_hash: str,
    certification: DatasetCertification,
) -> Path:
    if not rows:
        raise ValueError("rows must be non-empty")
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not policy.version or not policy.version.strip():
        raise ValueError("policy version must be non-empty")
    if not provider_version or not provider_version.strip():
        raise ValueError("provider_version must be non-empty")
    if not calendar_hash or not calendar_hash.strip():
        raise ValueError("calendar_hash must be non-empty")
    if not master_hash or not master_hash.strip():
        raise ValueError("master_hash must be non-empty")
    if not quality_report_hash or not quality_report_hash.strip():
        raise ValueError("quality_report_hash must be non-empty")
    if certification not in (DatasetCertification.RESEARCH, DatasetCertification.PRODUCTION):
        raise ValueError("certification must be RESEARCH or PRODUCTION")
    if not dataset_id or not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")

    root = Path(root)
    dataset_dir = root / dataset_id
    if dataset_dir.exists():
        raise ValueError(f"dataset already exists: {dataset_id}")

    # Check duplicate keys and inconsistent policy versions
    seen: set[tuple[datetime, str]] = set()
    for r in rows:
        key = (r.decision_session, r.instrument_id)
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen.add(key)
        if r.policy_version != policy.version:
            raise ValueError(f"inconsistent policy version {r.policy_version!r} != {policy.version!r}")
        # Check non-finite emitted scores/raw values
        for field in [
            "gross_profitability",
            "roe",
            "cfo_to_assets",
            "book_to_price",
            "earnings_to_price",
            "operating_income_change",
            "sales_growth",
            "operating_margin_change",
            "foreign_flow_5",
            "foreign_flow_20",
            "quality_score",
            "value_score",
            "earnings_score",
            "foreign_flow_score",
        ]:
            val = getattr(r, field)
            if val is not None and not math.isfinite(float(val)):  # noqa: SIM102
                raise ValueError(f"non-finite value for {field}: {val!r}")

    # Deterministic ordering by instrument_id (and decision_session)
    sorted_rows = sorted(rows, key=lambda r: (r.instrument_id, r.decision_session))

    # Build frame
    ordered_columns = [
        "decision_session",
        "instrument_id",
        "sector",
        "gross_profitability",
        "roe",
        "cfo_to_assets",
        "book_to_price",
        "earnings_to_price",
        "operating_income_change",
        "sales_growth",
        "operating_margin_change",
        "foreign_flow_5",
        "foreign_flow_20",
        "quality_score",
        "value_score",
        "earnings_score",
        "foreign_flow_score",
        "component_presence",
        "source_available_at",
        "policy_version",
    ]

    records: list[dict[str, Any]] = []
    for r in sorted_rows:
        # component_presence as comma-joined string for parquet
        comp_str = ",".join(r.component_presence)
        # source_available_at as json string sorted
        src_list = [{"source": s, "available_at": av.isoformat()} for s, av in r.source_available_at]
        src_str = json.dumps(src_list, sort_keys=True)
        records.append(
            {
                "decision_session": r.decision_session,
                "instrument_id": r.instrument_id,
                "sector": r.sector,
                "gross_profitability": r.gross_profitability,
                "roe": r.roe,
                "cfo_to_assets": r.cfo_to_assets,
                "book_to_price": r.book_to_price,
                "earnings_to_price": r.earnings_to_price,
                "operating_income_change": r.operating_income_change,
                "sales_growth": r.sales_growth,
                "operating_margin_change": r.operating_margin_change,
                "foreign_flow_5": r.foreign_flow_5,
                "foreign_flow_20": r.foreign_flow_20,
                "quality_score": r.quality_score,
                "value_score": r.value_score,
                "earnings_score": r.earnings_score,
                "foreign_flow_score": r.foreign_flow_score,
                "component_presence": comp_str,
                "source_available_at": src_str,
                "policy_version": r.policy_version,
            }
        )

    frame = pl.DataFrame(records).select(ordered_columns)

    # Time bounds
    sessions = [r.decision_session for r in sorted_rows]
    time_start = min(sessions)
    time_end = max(sessions)

    content_hash = canonical_content_hash(frame, ordered_columns)

    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set="stock_champion_qvef_v1",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=time_start,
        time_end=time_end,
        provider_version=provider_version,
        universe_policy_version=policy.version,
        row_count=frame.height,
        generated_time=decision_time,
        certification=certification,
        calendar_hash=calendar_hash,
        master_hash=master_hash,
        quality_report_hash=quality_report_hash,
        schema_version="v2",
        content_hash=content_hash,
        storage_layout=HIVE_PARTITION_LAYOUT,
    )

    store = ParquetDatasetStore(root)
    path = store.write_partitioned(
        frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set="stock_champion_qvef_v1",
        decision_time=decision_time,
    )
    return path