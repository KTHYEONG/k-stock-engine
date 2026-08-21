"""OOF fitting infrastructure extracted from training.py.

``OofCache`` manages the per-run temporary spill cache for OOF Parquet files.
``atomic_write_parquet`` and ``read_oof_parquet`` are shared I/O helpers.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, cast

import polars as pl

from src.core.paths import PROJECT_ROOT


def default_oof_cache_base() -> Path:
    return PROJECT_ROOT / "tmp" / "training"


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Write a Zstandard Parquet file atomically via a same-dir rename."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temp_path, compression="zstd")
    os.replace(temp_path, path)


def read_oof_parquet(path: Path) -> pl.DataFrame:
    """Load a cached OOF file; missing/corrupt files raise ``ValueError``."""
    if not path.exists():
        raise ValueError(f"missing cached OOF file {path}")
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"corrupt cached OOF file {path}: {exc}") from exc


class OofCache:
    """Per-run temporary spill cache for OOF Parquet files.

    Admitted horizons write the calibrated OOF scores and the label join as
    separate Zstandard Parquet files and release the DataFrames; only the file
    paths and the small Rank-IC tuple stay in process memory.
    """

    def __init__(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(dir=base_dir, prefix="oof-")
        self._root = Path(self._temporary.name)
        self._cache_bytes = 0
        self._closed = False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    def store(
        self,
        horizon_sessions: int,
        calibrated: pl.DataFrame,
        labels: pl.DataFrame,
    ) -> tuple[Path, Path]:
        if self._closed:
            raise ValueError("OOF cache is closed")
        oof_path = self._root / f"horizon_{horizon_sessions}_oof.parquet.zst"
        labels_path = self._root / f"horizon_{horizon_sessions}_labels.parquet.zst"
        atomic_write_parquet(calibrated, oof_path)
        atomic_write_parquet(labels, labels_path)
        self._cache_bytes += oof_path.stat().st_size + labels_path.stat().st_size
        return oof_path, labels_path

    def load(self, horizon_sessions: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        oof_path = self._root / f"horizon_{horizon_sessions}_oof.parquet.zst"
        labels_path = self._root / f"horizon_{horizon_sessions}_labels.parquet.zst"
        return read_oof_parquet(oof_path), read_oof_parquet(labels_path)

    def close(self) -> None:
        if not self._closed:
            self._temporary.cleanup()
            self._closed = True


def fit_horizon_oof(context: object) -> object:
    """Run the existing fold-local OOF fitter from an explicit context."""
    from src.stocks.ml.training import _fit_oof

    ctx = cast(Any, context)
    return _fit_oof(
        ctx.pre_holdout,
        ctx.folds,
        ctx.data,
        ctx.request,
        ctx.manifest,
        ctx.learner_columns,
        ctx.horizon,
        ctx.seed_ledger,
        family=ctx.family,
    )
