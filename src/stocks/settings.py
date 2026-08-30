"""Stock bounded-context settings (primitive fields only)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

# Pin omitted execution dates to the last complete market-data boundary.
REFERENCE_DATE = date(2026, 3, 10)
REFERENCE_DATETIME = datetime(2026, 3, 10, 6, 30, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class StockAlphaSettings:
    """Compatibility projection of CanonicalResearchProfile.

    CanonicalResearchProfile is the single owner of default statistical and
    portfolio values; this compatibility artifact projects the same resolved
    values so callers using the legacy import retain byte-identical outputs.
    """

    feature_set: str = "stock_alpha_v1"
    label_definition: str = "fwd_ret_5d"
    label_horizon_sessions: int = 5
    n_folds: int = 3
    embargo_sessions: int = 5
    top_k: int = 20
    max_single_weight: float = 0.08
    max_exposure: float = 0.90
    version: str = "v1"
    participation_limit: float = 0.005


DEFAULT_STOCK_ALPHA = StockAlphaSettings()
