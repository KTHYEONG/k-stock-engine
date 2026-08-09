"""Score-to-target-allocation policy.

The produced ``StockTargetWeight.target_weight`` is a target *weight* (a
fraction of portfolio value), never a currency amount. The research path keeps
the strict ``target_weight`` contract; conversion to KRW notional belongs only
to the execution boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.core.instruments import AssetKind

_MAX_PROJECTION_ITERATIONS = 64


@dataclass(frozen=True, slots=True)
class StockTargetWeight:
    """Long-only target weight for one instrument."""

    instrument_id: str
    target_weight: float
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AllocationPolicy:
    """Maps prediction scores to constrained long-only target weights.

    Inverse-volatility weights are projected onto single-name, sector, gross
    exposure, and ADTV-participation constraints. Infeasible demand is left in
    cash; no leverage and no shorting are ever produced.
    """

    top_k: int
    max_single_weight: float = 0.2
    max_exposure: float = 1.0
    max_sector_weight: float | None = None
    participation_limit: float = 0.0
    portfolio_value: float = 100_000_000.0
    volatility_column: str | None = "volatility"
    sector_column: str = "sector"
    adtv_column: str = "adtv"

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not (0 < self.max_single_weight <= 1):
            raise ValueError("max_single_weight must be in (0, 1]")
        if not (0 < self.max_exposure <= 1):
            raise ValueError("max_exposure must be in (0, 1]")
        if self.max_sector_weight is not None and not (0 < self.max_sector_weight <= 1):
            raise ValueError("max_sector_weight must be in (0, 1]")
        if self.participation_limit < 0:
            raise ValueError("participation_limit must be non-negative")
        if self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")

    def targets(
        self,
        scores: pl.DataFrame,
        asset_kind: AssetKind,
        instrument_column: str = "instrument_id",
        score_column: str = "pred_score",
    ) -> list[StockTargetWeight]:
        del asset_kind
        if scores.is_empty():
            return []
        if self.volatility_column is not None and self.volatility_column not in scores.columns:
            raise ValueError(
                f"missing volatility input {self.volatility_column!r}"
            )
        if self.max_sector_weight is not None and self.sector_column not in scores.columns:
            raise ValueError(f"missing sector input {self.sector_column!r}")
        if self.participation_limit > 0 and self.adtv_column not in scores.columns:
            raise ValueError(f"missing capacity input {self.adtv_column!r}")

        ranked = scores.sort(
            [pl.col(score_column).sort(descending=True), pl.col(instrument_column).sort()]
        ).head(self.top_k)

        raw_weights: list[float] = []
        sectors: list[object] = []
        for row in ranked.to_dicts():
            if self.volatility_column is not None:
                vol = row[self.volatility_column]
                if vol is None or vol <= 0:
                    raise ValueError("volatility must be positive and non-null")
                raw_weights.append(1.0 / float(vol))
            else:
                raw_weights.append(1.0)
            sectors.append(row.get(self.sector_column))

        instrument_ids = [str(row[instrument_column]) for row in ranked.to_dicts()]
        if not instrument_ids:
            return []

        weights = self._project(raw_weights, sectors, instrument_ids, ranked, instrument_column)
        return [
            StockTargetWeight(
                instrument_id=instrument_ids[i],
                target_weight=weights[i],
                reason="inverse-vol-constrained",
            )
            for i in range(len(instrument_ids))
            if weights[i] > 0.0
        ]

    def _project(
        self,
        raw_weights: list[float],
        sectors: list[object],
        instrument_ids: list[str],
        ranked: pl.DataFrame,
        instrument_column: str,
    ) -> list[float]:
        total_raw = sum(raw_weights) or 1.0
        weights = [w * self.max_exposure / total_raw for w in raw_weights]

        caps: list[float] = [self.max_single_weight] * len(weights)
        if self.max_sector_weight is not None:
            sector_sum: dict[object, float] = {}
            for i, sector in enumerate(sectors):
                sector_sum[sector] = sector_sum.get(sector, 0.0) + weights[i]
            for i, sector in enumerate(sectors):
                excess = sector_sum.get(sector, 0.0) - self.max_sector_weight
                if excess > 0 and weights[i] > 0:
                    caps[i] = min(caps[i], weights[i] - excess / sum(
                        1 for j in range(len(sectors)) if sectors[j] == sector
                    ))
        if self.participation_limit > 0 and self.adtv_column in ranked.columns:
            for i, instrument_id in enumerate(instrument_ids):
                match = ranked.filter(pl.col(instrument_column) == instrument_id)
                if match.is_empty():
                    continue
                adtv = float(match[self.adtv_column][0])
                if adtv <= 0:
                    raise ValueError("adtv must be positive")
                caps[i] = min(caps[i], self.participation_limit * adtv / self.portfolio_value)

        active = [w > 0.0 for w in weights]
        for _ in range(_MAX_PROJECTION_ITERATIONS):
            changed = False
            for i in range(len(weights)):
                if active[i] and weights[i] > caps[i]:
                    surplus = weights[i] - caps[i]
                    weights[i] = caps[i]
                    redistributable = [j for j in range(len(weights)) if active[j] and weights[j] < caps[j]]
                    if not redistributable:
                        break
                    share = surplus / len(redistributable)
                    for j in redistributable:
                        weights[j] = min(caps[j], weights[j] + share)
                        changed = True
            if not changed:
                break
        return [max(0.0, w) for w in weights]
