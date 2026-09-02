"""Point-in-time Q/V/E/F feature contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class QvefFeaturePolicy:
    version: str = "champion-v1-qvef-v1"
    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    minimum_sector_cohort: int = 10
    earnings_staleness_sessions: int = 60

    def __post_init__(self) -> None:
        if not self.version or not self.version.strip():
            raise ValueError("policy version must be non-empty")
        if not (0 <= self.winsor_lower_quantile < self.winsor_upper_quantile <= 1):
            raise ValueError("invalid quantile bounds")
        if self.winsor_lower_quantile < 0 or self.winsor_upper_quantile > 1:
            raise ValueError("quantiles must be within [0,1]")
        if self.minimum_sector_cohort < 2:
            raise ValueError("minimum_sector_cohort must be >=2")
        if self.earnings_staleness_sessions < 0:
            raise ValueError("earnings_staleness_sessions must be non-negative")


@dataclass(frozen=True, slots=True)
class QvefFeatureRow:
    decision_session: datetime
    instrument_id: str
    sector: str
    gross_profitability: float | None
    roe: float | None
    cfo_to_assets: float | None
    book_to_price: float | None
    earnings_to_price: float | None
    operating_income_change: float | None
    sales_growth: float | None
    operating_margin_change: float | None
    foreign_flow_5: float | None
    foreign_flow_20: float | None
    quality_score: float | None
    value_score: float | None
    earnings_score: float | None
    foreign_flow_score: float | None
    component_presence: tuple[str, ...]
    source_available_at: tuple[tuple[str, datetime], ...]
    policy_version: str
