"""Constrained long-only target-weight policy tests."""
from __future__ import annotations

import polars as pl
import pytest

from src.core.instruments import AssetKind
from src.stocks.trading.allocation_policy import AllocationPolicy


def score_frame(top: int = 6) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i:05d}" for i in range(top)],
            "pred_score": [float(i) for i in reversed(range(top))],
            "volatility": [0.02 + i * 0.001 for i in range(top)],
            "adtv": [1.0e9] * top,
        }
    )


def test_targets_are_non_negative_and_bounded_by_exposure() -> None:
    policy = AllocationPolicy(top_k=5, max_single_weight=0.2, max_exposure=1.0)
    targets = policy.targets(score_frame(), AssetKind.STOCK)
    total = sum(t.target_weight for t in targets)
    assert 0.0 <= total <= policy.max_exposure
    assert all(t.target_weight >= 0.0 for t in targets)
    assert all(t.target_weight <= policy.max_single_weight for t in targets)


def test_targets_tie_break_deterministically() -> None:
    policy = AllocationPolicy(top_k=3, max_single_weight=0.5, max_exposure=1.0)
    first = policy.targets(score_frame(), AssetKind.STOCK)
    second = policy.targets(score_frame(), AssetKind.STOCK)
    assert [(t.instrument_id, t.target_weight) for t in first] == [
        (t.instrument_id, t.target_weight) for t in second
    ]


def test_targets_reject_missing_volatility() -> None:
    policy = AllocationPolicy(top_k=3)
    with pytest.raises(ValueError, match="volatility"):
        policy.targets(score_frame().drop("volatility"), AssetKind.STOCK)


def test_targets_reject_missing_capacity_input() -> None:
    policy = AllocationPolicy(top_k=3, participation_limit=0.01)
    with pytest.raises(ValueError, match="capacity input"):
        policy.targets(score_frame().drop("adtv"), AssetKind.STOCK)


def test_targets_reject_missing_sector_input() -> None:
    policy = AllocationPolicy(top_k=3, max_sector_weight=0.4)
    with pytest.raises(ValueError, match="sector"):
        policy.targets(score_frame(), AssetKind.STOCK)


def test_infeasible_allocation_stays_in_cash() -> None:
    tiny_cap = score_frame().with_columns(pl.lit(1_000.0).alias("adtv"))
    policy = AllocationPolicy(top_k=3, max_exposure=1.0, participation_limit=0.01)
    targets = policy.targets(tiny_cap, AssetKind.STOCK)
    assert sum(t.target_weight for t in targets) < policy.max_exposure
