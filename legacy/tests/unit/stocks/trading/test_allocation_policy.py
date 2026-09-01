"""Constrained long-only target-weight policy tests."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.core.instruments import AssetKind
from legacy.stocks.trading.allocation_policy import AllocationPolicy, rank_stock_candidate_indices


def score_frame(top: int = 6) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i:05d}" for i in range(top)],
            "pred_score": [float(i) for i in reversed(range(top))],
            "volatility": [0.02 + i * 0.001 for i in range(top)],
            "adtv": [1.0e9] * top,
        }
    )


def test_rank_stock_candidate_indices_orders_score_desc_id_asc() -> None:
    order = rank_stock_candidate_indices(
        np.array([0.1, 0.9, 0.4, 0.8]),
        np.array(["KRX:A", "KRX:B", "KRX:C", "KRX:D"], dtype=object),
    )
    assert order.dtype == np.int64
    assert order.tolist() == [1, 3, 2, 0]


def test_rank_stock_candidate_indices_resolves_ties_by_identifier_ascending() -> None:
    order = rank_stock_candidate_indices(
        np.array([0.5, 0.5, 0.5]),
        np.array(["KRX:B", "KRX:A", "KRX:C"], dtype=object),
    )
    assert order.tolist() == [1, 0, 2]


def test_rank_stock_candidate_indices_membership_invariant_to_permutation() -> None:
    scores = np.array([0.1, 0.9, 0.4, 0.8])
    ids = np.array(["KRX:A", "KRX:B", "KRX:C", "KRX:D"], dtype=object)
    reference = rank_stock_candidate_indices(scores, ids)
    perm = np.array([2, 0, 3, 1])
    permuted = rank_stock_candidate_indices(scores[perm], ids[perm])
    assert (ids[perm][permuted] == ids[reference]).all()


@pytest.mark.parametrize(
    ("scores", "ids", "match"),
    [
        (np.array([[1.0, 2.0]]), np.array(["A"]), "one-dimensional"),
        (np.array([1.0, 2.0]), np.array(["A"]), "equal length"),
        (np.array([np.nan]), np.array(["A"]), "finite"),
        (np.array([np.inf]), np.array(["A"]), "finite"),
        (np.array([1.0]), np.array([None], dtype=object), "null"),
        (np.array([1.0, 2.0]), np.array(["A", "A"], dtype=object), "unique"),
    ],
)
def test_rank_stock_candidate_indices_rejects_invalid_input(
    scores: np.ndarray, ids: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        rank_stock_candidate_indices(scores, ids)


def test_targets_select_descending_score_ids_then_ascending_ties() -> None:
    policy = AllocationPolicy(top_k=2, max_single_weight=0.5, max_exposure=1.0)
    frame = pl.DataFrame(
        {
            "instrument_id": ["KRX:A", "KRX:B", "KRX:C", "KRX:D"],
            "pred_score": [0.1, 0.9, 0.4, 0.8],
            "volatility": [0.1, 0.1, 0.1, 0.1],
            "adtv": [1.0e9] * 4,
        }
    )
    targets = policy.targets(frame, AssetKind.STOCK)
    assert [t.instrument_id for t in targets] == ["KRX:B", "KRX:D"]


def test_targets_membership_invariant_to_input_row_permutation() -> None:
    policy = AllocationPolicy(top_k=3, max_single_weight=0.5, max_exposure=1.0)
    frame = score_frame()
    permuted = frame.sample(fraction=1.0, seed=7, shuffle=True)
    original_ids = {t.instrument_id for t in policy.targets(frame, AssetKind.STOCK)}
    permuted_ids = {t.instrument_id for t in policy.targets(permuted, AssetKind.STOCK)}
    assert original_ids == permuted_ids


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
