"""Single certified investor-flow Silver union (LS native coverage plus KIS gap fill)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.core.pit import PITDataError
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    PartitionSpool,
    dataset_partition_paths,
    dataset_reference,
    load_manifest,
    publish_dataset,
)

REVISION = "investor-flow-ls-kis-union-v1"
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


def _input_paths(dataset_dir: Path, label: str, *, expected_kind: str) -> tuple[str, tuple[Path, ...]]:
    """Return the dataset id and verified partition paths without reading row data."""
    paths = dataset_partition_paths(dataset_dir, allow_legacy=False)
    if not paths:
        return dataset_dir.name, ()
    try:
        manifest = load_manifest(dataset_dir)
    except PITDataError:
        manifest = None
    if manifest is not None and manifest.kind != expected_kind:
        raise PITDataError(f"invalid {label} kind: {dataset_dir}")
    try:
        pl.scan_parquet([str(path) for path in paths]).select(list(_COMMON_COLUMNS)).cast(
            pl.Schema(_SCHEMA)
        ).collect_schema()
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise PITDataError(f"invalid {label} partitions: {dataset_dir}") from exc
    return dataset_dir.name, paths


def _years_of(paths: tuple[Path, ...], dataset_dir: Path, label: str) -> list[int]:
    if not paths:
        return []
    try:
        years = (
            pl.scan_parquet([str(path) for path in paths])
            .select(pl.col("session").cast(pl.Date).dt.year().alias("year"))
            .unique()
            .sort("year")
            .collect()
            .get_column("year")
            .to_list()
        )
        if None in years:
            raise ValueError("investor-flow sessions must not be null")
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise PITDataError(f"invalid {label} partitions: {dataset_dir}") from exc
    return [int(year) for year in years]


def _read_year(
    paths: tuple[Path, ...], year: int, dataset_dir: Path, label: str
) -> pl.DataFrame:
    """Read one year's rows projected to the union schema."""
    if not paths:
        return pl.DataFrame(schema=pl.Schema(_SCHEMA))
    try:
        return (
            pl.scan_parquet([str(path) for path in paths])
            .filter(pl.col("session").cast(pl.Date).dt.year() == year)
            .select(list(_COMMON_COLUMNS))
            .collect()
            .cast(pl.Schema(_SCHEMA))
        )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise PITDataError(f"invalid {label} partitions: {dataset_dir}") from exc


def _read_input(dataset_dir: Path, label: str, *, expected_kind: str) -> tuple[str, pl.DataFrame]:
    dataset_id, paths = _input_paths(dataset_dir, label, expected_kind=expected_kind)
    if not paths:
        return dataset_id, pl.DataFrame(schema=pl.Schema(_SCHEMA))
    try:
        frame = (
            pl.scan_parquet([str(path) for path in paths])
            .select(list(_COMMON_COLUMNS))
            .collect()
            .cast(pl.Schema(_SCHEMA))
        )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise PITDataError(f"invalid {label} partitions: {dataset_dir}") from exc
    return dataset_id, frame


def materialize_investor_flow_union(
    *, ls_flow_silver_path: Path, kis_supplement_silver_path: Path, silver_root: Path
) -> InvestorFlowUnionResult:
    """Publish the identity-bound union of certified LS and KIS flow rows."""

    ls_id, ls_paths = _input_paths(
        Path(ls_flow_silver_path), "investor-flow-ls", expected_kind="investor_flow_ls"
    )
    kis_id, kis_paths = _input_paths(
        Path(kis_supplement_silver_path),
        "investor-flow-kis-supplement",
        expected_kind="investor_flow_kis_supplement",
    )
    ls_dir = Path(ls_flow_silver_path)
    kis_dir = Path(kis_supplement_silver_path)
    years = sorted(
        set(_years_of(ls_paths, ls_dir, "investor-flow-ls"))
        | set(_years_of(kis_paths, kis_dir, "investor-flow-kis-supplement"))
    )
    total_rows = 0
    partition_details: list[dict[str, object]] = []
    with PartitionSpool() as spool:
        for year in years:
            ls_year = _read_year(ls_paths, year, ls_dir, "investor-flow-ls")
            kis_year = _read_year(kis_paths, year, kis_dir, "investor-flow-kis-supplement")
            conflict = (
                ls_year.select("session", "ticker")
                .unique()
                .join(kis_year.select("session", "ticker").unique(), on=["session", "ticker"], how="inner")
            )
            if conflict.height:
                key = conflict.sort(["session", "ticker"]).row(0)
                raise PITDataError(
                    f"investor-flow union key present in both inputs: session={key[0]!r} ticker={key[1]!r}"
                )
            if ls_year.height == 0:
                frame = kis_year
            elif kis_year.height == 0:
                frame = ls_year
            else:
                frame = pl.concat([ls_year, kis_year], how="vertical")
            del ls_year, kis_year
            frame = frame.sort(["session", "ticker"])
            relative_path = f"year={year}/part.parquet"
            rows = spool.add(relative_path, frame)
            partition_details.append({"path": relative_path, "rows": rows})
            total_rows += rows
            del frame
        identity = DatasetIdentity(
            kind="investor_flow",
            layer=DatasetLayer.SILVER,
            policy_version=REVISION,
            inputs={
                "ls": dataset_reference(ls_id, kind="investor_flow_ls"),
                "kis_supplement": dataset_reference(kis_id, kind="investor_flow_kis_supplement"),
            },
            params={},
        )
        published = publish_dataset(
            layer_root=Path(silver_root),
            identity=identity,
            partitions=dict(spool.partitions),
            details={
                "ls_dataset_id": ls_id,
                "kis_supplement_dataset_id": kis_id,
                "partitions": partition_details,
            },
        )
    return InvestorFlowUnionResult(
        dataset_path=published.path,
        dataset_id=published.dataset_id,
        ls_dataset_id=ls_id,
        kis_supplement_dataset_id=kis_id,
        rows=total_rows,
    )
