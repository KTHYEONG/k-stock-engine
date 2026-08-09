"""Stock data-layer contracts.

``DatasetSnapshot`` is the validated input passed into stock workflows: it
bundles the manifest with the frame it describes, so a workflow never has to
read Parquet or manufacture a manifest.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.core.datasets import DatasetManifest


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """A validated stock dataset: manifest plus the frame it describes."""

    manifest: DatasetManifest
    frame: pl.DataFrame
