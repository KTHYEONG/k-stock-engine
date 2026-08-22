"""Bounded-workspace bootstrap primitive parity tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.stocks.research.bootstrap import (
    _effective_block_length,
    moving_block_bootstrap_means,
    pooled_segment_bootstrap_means,
)


def test_moving_block_reference_matches_manual_indexing() -> None:
    values = np.arange(1, 21, dtype=float)
    means = moving_block_bootstrap_means(values, block_length=5, n_bootstrap=64, seed=7)
    rng = np.random.default_rng(7)
    starts = rng.integers(0, max(1, 20 - 5 + 1), size=(64, 4))
    manual = np.empty(64)
    for draw in range(64):
        sampled = [
            values[min(int(start) + int(offset), 19)]
            for start in starts[draw]
            for offset in range(5)
        ][:20]
        manual[draw] = float(np.mean(sampled))
    assert np.allclose(means, manual)


def test_pooled_means_match_materialized_within_error_bound() -> None:
    segments = (
        np.random.default_rng(1).normal(size=90),
        np.random.default_rng(2).normal(size=120) + 0.05,
    )
    pooled = pooled_segment_bootstrap_means(segments, 5, 128, seed=42)
    exact_parts = []
    weights = []
    for position, segment in enumerate(segments):
        block = _effective_block_length(segment.size, 5)
        exact_parts.append(
            moving_block_bootstrap_means(segment, block, 128, seed=42 + position)
        )
        weights.append(float(segment.size))
    total = sum(weights)
    reference = sum(w * d for w, d in zip(weights, exact_parts, strict=True)) / total
    assert pooled.shape == reference.shape
    # Prefix-sum kernel stays within the conservative float64 error bound.
    bound = 1e-9
    assert float(np.max(np.abs(pooled - reference))) <= bound


def test_effective_block_length_applies_cube_root_floor() -> None:
    assert _effective_block_length(1000, 1) == 10
    assert _effective_block_length(27, 5) == 5
    with pytest.raises(ValueError, match="at least one"):
        _effective_block_length(0, 5)
