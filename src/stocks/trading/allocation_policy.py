"""Score-to-target-portfolio policy."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation


@dataclass(frozen=True, slots=True)
class PortfolioPolicy:
    """Maps prediction scores to target allocations within exposure limits."""

    top_k: int
    max_single_weight: float = 0.2
    max_exposure: float = 1.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not (0 < self.max_single_weight <= 1):
            raise ValueError("max_single_weight must be in (0, 1]")
        if not (0 < self.max_exposure <= 1):
            raise ValueError("max_exposure must be in (0, 1]")

    def targets(
        self,
        scores: pl.DataFrame,
        asset_kind: AssetKind,
        instrument_column: str = "instrument_id",
        score_column: str = "pred_score",
    ) -> list[Allocation]:
        if scores.is_empty():
            return []
        ranked = scores.sort(score_column, descending=True).head(self.top_k)
        raw = ranked[score_column].to_list()
        total = sum(max(0.0, s) for s in raw) or 1.0
        allocations: list[Allocation] = []
        for row in ranked.iter_rows(named=True):
            instrument = Instrument(
                instrument_id=str(row[instrument_column]),
                asset_kind=asset_kind,
                exchange="KRX",
                symbol=str(row[instrument_column]).split(":")[-1],
                currency="KRW",
            )
            weight = max(0.0, float(row[score_column])) / total
            weight = min(weight, self.max_single_weight)
            allocations.append(
                Allocation(
                    instrument=instrument,
                    target_value=weight * self.max_exposure,
                    reason="score-rank-policy",
                )
            )
        return allocations
