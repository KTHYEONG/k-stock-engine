"""Stock bounded-context settings (primitive fields only)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

# Pin omitted execution dates to the last complete market-data boundary.
REFERENCE_DATE = date(2026, 3, 10)
REFERENCE_DATETIME = datetime(2026, 3, 10, 6, 30, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class StockAlphaSettings:
    """Configuration artifact for the stock alpha baseline pipeline."""

    feature_set: str = "stock_alpha_v1"
    label_definition: str = "fwd_ret_5d"
    label_horizon_sessions: int = 5
    n_folds: int = 3
    embargo_sessions: int = 5
    top_k: int = 5
    max_single_weight: float = 0.2
    max_exposure: float = 1.0
    version: str = "v1"


DEFAULT_STOCK_ALPHA = StockAlphaSettings()
