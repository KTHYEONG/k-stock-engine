"""ETF data-layer contracts: validated dataset snapshots for ETF workflows."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind


@dataclass(frozen=True, slots=True)
class EtfDataset:
    """A validated ETF dataset: manifest plus the frames it describes.

    ``index_frame`` carries the index OHLC used for signals; ``etf_frame``
    carries the bull/bear ETF OHLC used for fills.
    """

    manifest: DatasetManifest
    index_frame: pl.DataFrame
    etf_frame: pl.DataFrame

    @property
    def asset_kind(self) -> AssetKind:
        return AssetKind.ETF
