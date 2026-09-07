"""Immutable Gold artifact bundle resolver (single dataset id, fail-closed)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from src.core.datasets import validate_dataset_manifest
from src.core.instruments import AssetKind
from src.data.schemas import PITDataError
from src.storage.parquet_datasets import ParquetDatasetStore

_UNIVERSE_FEATURE_SET = "stock_historical_eligible_universe_v1"
_QVEF_FEATURE_SET = "stock_champion_qvef_v1"
_SCORES_FEATURE_SET = "stock_champion_scores_v1"


@dataclass(frozen=True, slots=True)
class GoldArtifactBundle:
    dataset_id: str
    universe_path: Path
    qvef_path: Path
    champion_scores_path: Path
    universe_manifest_hash: str
    qvef_manifest_hash: str
    champion_scores_manifest_hash: str


def _check_dataset_id(dataset_id: str) -> None:
    if not dataset_id.strip() or "/" in dataset_id or "\\" in dataset_id or ".." in dataset_id:
        raise PITDataError(f"invalid dataset id {dataset_id!r}: must be a single directory name")


def resolve_gold_artifact_bundle(
    *, gold_root: Path, dataset_id: str, decision_time: datetime
) -> GoldArtifactBundle:
    _check_dataset_id(dataset_id)
    root = Path(gold_root)
    kinds = ("universe", "qvef", "champion_scores")
    for kind in kinds:
        if not (root / kind / dataset_id).is_dir():
            raise PITDataError(f"missing {kind}/{dataset_id}: selected Gold component is absent")
    expected = {
        "universe": _UNIVERSE_FEATURE_SET,
        "qvef": _QVEF_FEATURE_SET,
        "champion_scores": _SCORES_FEATURE_SET,
    }
    hashes: dict[str, str] = {}
    for kind in kinds:
        manifest = ParquetDatasetStore(root / kind).read_manifest(dataset_id)
        validate_dataset_manifest(manifest, AssetKind.STOCK, expected[kind], decision_time)
        hashes[kind] = manifest.content_hash
    return GoldArtifactBundle(
        dataset_id=dataset_id,
        universe_path=root / "universe" / dataset_id,
        qvef_path=root / "qvef" / dataset_id,
        champion_scores_path=root / "champion_scores" / dataset_id,
        universe_manifest_hash=hashes["universe"],
        qvef_manifest_hash=hashes["qvef"],
        champion_scores_manifest_hash=hashes["champion_scores"],
    )


def load_gold_artifact_frames(
    *, bundle: GoldArtifactBundle, decision_time: datetime
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    universe = ParquetDatasetStore(bundle.universe_path.parent).read(
        bundle.dataset_id, AssetKind.STOCK, _UNIVERSE_FEATURE_SET, decision_time
    )
    qvef = ParquetDatasetStore(bundle.qvef_path.parent).read(
        bundle.dataset_id, AssetKind.STOCK, _QVEF_FEATURE_SET, decision_time
    )
    scores = ParquetDatasetStore(bundle.champion_scores_path.parent).read(
        bundle.dataset_id, AssetKind.STOCK, _SCORES_FEATURE_SET, decision_time
    )
    return universe, qvef, scores
