"""Single certified investor-flow Silver union (LS native coverage plus KIS gap fill)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_partition_paths,
    dataset_reference,
    load_manifest,
    publish_dataset,
)
from src.data.schemas import PITDataError

POLICY_VERSION = "investor-flow-ls-kis-union-v1"
_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "ticker": pl.String,
    "provider": pl.String,
    "individual_net_shares": pl.Int64,
    "foreign_net_shares": pl.Int64,
    "institution_net_shares": pl.Int64,
    "other_net_shares": pl.Int64,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "source_hash": pl.String,
    "policy_version": pl.String,
}
_COMMON_COLUMNS: tuple[str, ...] = tuple(_SCHEMA)


@dataclass(frozen=True, slots=True)
class InvestorFlowUnionResult:
    dataset_path: Path
    dataset_id: str
    ls_dataset_id: str
    kis_supplement_dataset_id: str
    rows: int


def _read_input(dataset_dir: Path, label: str, *, expected_kind: str) -> tuple[str, pl.DataFrame]:
    paths = dataset_partition_paths(dataset_dir, allow_legacy=False)
    if not paths:
        return dataset_dir.name, pl.DataFrame(schema=pl.Schema(_SCHEMA))
    try:
        manifest = load_manifest(dataset_dir)
    except PITDataError:
        manifest = None
    if manifest is not None and manifest.kind != expected_kind:
        raise PITDataError(f"invalid {label} kind: {dataset_dir}")
    try:
        frame = (
            pl.scan_parquet([str(path) for path in paths])
            .select(list(_COMMON_COLUMNS))
            .collect()
            .cast(pl.Schema(_SCHEMA))
        )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise PITDataError(f"invalid {label} partitions: {dataset_dir}") from exc
    return dataset_dir.name, frame


def materialize_investor_flow_union(
    *, ls_flow_silver_path: Path, kis_supplement_silver_path: Path, silver_root: Path
) -> InvestorFlowUnionResult:
    """Publish the identity-bound union of certified LS and KIS flow rows."""

    ls_id, ls_frame = _read_input(
        Path(ls_flow_silver_path), "investor-flow-ls", expected_kind="investor_flow_ls"
    )
    kis_id, kis_frame = _read_input(
        Path(kis_supplement_silver_path),
        "investor-flow-kis-supplement",
        expected_kind="investor_flow_kis_supplement",
    )
    conflict = (
        ls_frame.select("session", "ticker")
        .unique()
        .join(kis_frame.select("session", "ticker").unique(), on=["session", "ticker"], how="inner")
    )
    if conflict.height:
        key = conflict.sort(["session", "ticker"]).row(0)
        raise PITDataError(
            f"investor-flow union key present in both inputs: session={key[0]!r} ticker={key[1]!r}"
        )
    if ls_frame.height == 0:
        frame = kis_frame
    elif kis_frame.height == 0:
        frame = ls_frame
    else:
        frame = pl.concat([ls_frame, kis_frame], how="vertical")
    frame = frame.sort(["session", "ticker"])
    partitions: dict[str, pl.DataFrame] = {}
    for year in sorted(frame["session"].dt.year().unique().to_list()) if frame.height else []:
        partitions[f"year={year}/part.parquet"] = frame.filter(pl.col("session").dt.year() == year)
    identity = DatasetIdentity(
        kind="investor_flow",
        layer=DatasetLayer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={
            "ls": dataset_reference(ls_id, kind="investor_flow_ls"),
            "kis_supplement": dataset_reference(kis_id, kind="investor_flow_kis_supplement"),
        },
        params={},
    )
    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions=partitions,
        details={
            "ls_dataset_id": ls_id,
            "kis_supplement_dataset_id": kis_id,
            "partitions": [
                {"path": path, "rows": part.height} for path, part in sorted(partitions.items())
            ],
        },
    )
    return InvestorFlowUnionResult(
        dataset_path=published.path,
        dataset_id=published.dataset_id,
        ls_dataset_id=ls_id,
        kis_supplement_dataset_id=kis_id,
        rows=frame.height,
    )
