"""Provider collection boundary for ETF/index market data.

Collection adapters (KRX OpenAPI, etc.) implement this protocol; curated
storage belongs to ``repositories``. No provider parsing logic lives in the
workflows.
"""
from __future__ import annotations

from typing import Protocol

import polars as pl


class EtfDataProvider(Protocol):
    """External ETF/index data source producing raw frames for curation."""

    def collect_index_bars(self, *, start_date: str, end_date: str) -> pl.DataFrame:
        """Collect raw index bars in a provider-specific schema."""
        ...

    def collect_etf_bars(self, *, start_date: str, end_date: str) -> pl.DataFrame:
        """Collect raw ETF bars in a provider-specific schema."""
        ...
