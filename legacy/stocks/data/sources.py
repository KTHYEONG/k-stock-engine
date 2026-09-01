"""Provider collection boundary for stock market data.

Collection adapters (KRX, DART, etc.) implement this protocol; curated storage
belongs to ``repositories``. No provider parsing logic lives in ``core`` or in
the workflows.
"""
from __future__ import annotations

from typing import Protocol

import polars as pl


class StockDataProvider(Protocol):
    """External stock data source producing raw frames for curation."""

    def collect_bars(self, *, start_date: str, end_date: str) -> pl.DataFrame:
        """Collect raw stock bars in a provider-specific schema."""
        ...
