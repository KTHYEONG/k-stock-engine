"""Single certified investor-flow Silver union (LS native coverage plus KIS gap fill)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.data.investor_flow_gap import _read_verified_partitions
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


def materialize_investor_flow_union(
    *, ls_flow_silver_path: Path, kis_supplement_silver_path: Path, silver_root: Path
) -> InvestorFlowUnionResult:
    """Build the single certified ``investor_flow_<hash16>`` dataset.

    This produces the one investor-flow table downstream Gold/engine work is
    meant to treat as *the* investor-flow table, combining LS's native
    coverage with the KIS-sourced supplement that was deliberately restricted
    to exactly LS's gap.

    Only the four columns common to both inputs (``individual_net_shares``,
    ``foreign_net_shares``, ``institution_net_shares``, ``other_net_shares``)
    are carried into the union alongside the shared keys and provenance
    columns. LS-only breakdown columns (``tjj0000_net_shares``..
    ``tjj0011_net_shares``, ``ls_close``, ``ls_volume``, ``ls_value_mkrw``)
    stay queryable in the original LS dataset for audit but are not projected
    here, since no consumer needs them and carrying a partially-null wide
    schema across two providers is not a real invariant to maintain.

    Args:
        ls_flow_silver_path: Certified LS Silver ``investor_flow_<id>``
            directory (native coverage).
        kis_supplement_silver_path: Certified KIS supplement
            ``investor_flow_kis_supplement_<id>`` directory (exactly the LS
            gap, never overlapping it).
        silver_root: Scope Silver root receiving ``investor_flow_<hash16>/``.

    Returns:
        The union result; ``rows`` equals the summed row count of both
        inputs.

    Raises:
        PITDataError: a ``(session, ticker)`` key is present in both inputs
            (must never happen given the supplement's exact-gap construction;
            a genuine invariant violation, not defensive bloat), either
            input's manifest or partition hashes fail verification, or a
            dataset already exists at the computed id whose content differs
            from what this run would produce.
    """
    ls_dir = Path(ls_flow_silver_path)
    kis_dir = Path(kis_supplement_silver_path)
    ls_id, ls_files = _read_verified_partitions(ls_dir, label="investor-flow")
    kis_id, kis_files = _read_verified_partitions(kis_dir, label="investor-flow-kis-supplement")
    ls_frame = (
        pl.scan_parquet(ls_files)
        .select(list(_COMMON_COLUMNS))
        .collect()
        .cast(pl.Schema(_SCHEMA))
    )
    kis_frame = (
        pl.scan_parquet(kis_files)
        .select(list(_COMMON_COLUMNS))
        .collect()
        .cast(pl.Schema(_SCHEMA))
    )
    conflict = (
        ls_frame.select("session", "ticker")
        .unique()
        .join(kis_frame.select("session", "ticker").unique(), on=["session", "ticker"], how="inner")
    )
    if conflict.height:
        key = conflict.sort(["session", "ticker"]).row(0)
        raise PITDataError(f"investor-flow union key present in both inputs: session={key[0]!r} ticker={key[1]!r}")
    frame = pl.concat([ls_frame, kis_frame], how="vertical").sort(["session", "ticker"])
    silver_root = Path(silver_root)
    silver_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".investor-flow-union-", dir=silver_root))
    try:
        partition_hashes: list[str] = []
        partitions: list[dict[str, Any]] = []
        total_rows = 0
        years = sorted(frame["session"].dt.year().unique().to_list()) if frame.height else []
        for year in years:
            part = frame.filter(pl.col("session").dt.year() == year)
            rel = Path(f"year={year}") / "part.parquet"
            out_path = staging / rel
            out_path.parent.mkdir(parents=True)
            part.write_parquet(out_path)
            digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
            partition_hashes.append(digest)
            partitions.append({
                "path": str(rel),
                "row_count": part.height,
                "parquet_sha256": digest,
                "year": year,
            })
            total_rows += part.height
        dataset_id = "investor_flow_" + hashlib.sha256(
            "\n".join((POLICY_VERSION, ls_id, kis_id, *sorted(partition_hashes))).encode("utf-8")
        ).hexdigest()[:16]
        target_path = silver_root / dataset_id
        manifest = {
            "dataset_id": dataset_id,
            "policy_version": POLICY_VERSION,
            "ls_dataset_id": ls_id,
            "kis_supplement_dataset_id": kis_id,
            "rows": total_rows,
            "partitions": partitions,
        }
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (staging / "manifest.json").write_text(encoded, encoding="utf-8")
        if target_path.exists():
            try:
                current = (target_path / "manifest.json").read_text(encoding="utf-8")
            except OSError as exc:
                raise PITDataError(
                    f"existing investor-flow union dataset is unreadable: {target_path}"
                ) from exc
            if current != encoded:
                raise PITDataError(f"existing investor-flow union dataset differs: {target_path}")
            shutil.rmtree(staging, ignore_errors=True)
        else:
            os.rename(staging, target_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return InvestorFlowUnionResult(
        dataset_path=target_path,
        dataset_id=dataset_id,
        ls_dataset_id=ls_id,
        kis_supplement_dataset_id=kis_id,
        rows=total_rows,
    )
