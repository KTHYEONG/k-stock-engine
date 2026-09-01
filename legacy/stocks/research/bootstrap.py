"""Deterministic bounded-workspace moving-block bootstrap primitives.

Extracted from the prefix-sum kernel already certified in the economic-alpha
calibration path. ``moving_block_bootstrap_means`` is the exact materialized
reference; ``pooled_segment_bootstrap_means`` pools per-segment resample means
with the identical seeded draw order while keeping the workspace
``O(B * ceil(N / L) + N)`` instead of ``O(B * N)``.
"""
from __future__ import annotations

import numpy as np

from legacy.stocks.research.economic_alpha import (
    _bootstrap_error_bound,
    _prefix_sum_block_means,
)


def _segment_draw_geometry(n: int, block_length: int) -> tuple[int, int]:
    """Return ``(n_blocks, max_start)`` for one segment's resampling grid."""
    block = max(int(block_length), 1)
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block + 1)
    return n_blocks, max_start


def _effective_block_length(n: int, block_floor: int) -> int:
    """Deterministic per-segment block ``max(ceil(n ** (1/3)), floor)``."""
    if n < 1:
        raise ValueError("segment must contain at least one observation")
    cube_root = max(1, int(np.ceil(float(n) ** (1.0 / 3.0))))
    return max(cube_root, max(int(block_floor), 1))


def moving_block_bootstrap_means(
    values: np.ndarray,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Exact materialized moving-block bootstrap means for one segment.

    The seeded block starts, the row-major draw order, and the final-block
    truncation are identical to the historical reference kernel, so this is the
    parity oracle for every bounded-workspace variant.
    """
    arr = np.asarray(values, dtype=np.float64)
    n_bootstrap = int(n_bootstrap)
    if arr.size == 0 or n_bootstrap < 1:
        return np.zeros(max(n_bootstrap, 0), dtype=np.float64)
    block = max(int(block_length), 1)
    n_blocks, max_start = _segment_draw_geometry(arr.size, block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]).reshape(
        n_bootstrap, n_blocks * block
    )[:, : arr.size]
    return np.asarray(arr[index].mean(axis=1), dtype=np.float64)


def pooled_segment_bootstrap_means(
    segments: tuple[np.ndarray, ...],
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Pooled per-segment moving-block means with bounded workspace.

    Segment ``i`` is resampled with ``default_rng(seed + i)``, one shared
    block-length rule (``max(ceil(n_s ** (1/3)), block_length)``), and the same
    number/order of RNG draws as the materialized reference; each draw's final
    block is truncated at the segment length. Per-segment means come from the
    deterministic block-prefix-sum kernel so the workspace stays
    ``O(B * ceil(N / L) + N)``, and each segment mean is within the
    conservative float64 error bound of the exact kernel.

    When any pooled element sits inside that bound of the centered gate
    (``2 * observed``), the pooled distribution is recomputed with the exact
    materialized kernel so a near-zero decision never depends on prefix-sum
    rounding.
    """
    if not segments:
        raise ValueError("pooled bootstrap requires at least one segment")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        return np.zeros(0, dtype=np.float64)

    distributions: list[np.ndarray] = []
    weights: list[float] = []
    total_weight = 0.0
    for position, raw_values in enumerate(segments):
        values = np.asarray(raw_values, dtype=np.float64)
        if values.size == 0:
            raise ValueError("pooled bootstrap segments must be non-empty")
        block = _effective_block_length(values.size, block_length)
        n_blocks, max_start = _segment_draw_geometry(values.size, block)
        rng = np.random.default_rng(seed + position)
        starts = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))
        means = _prefix_sum_block_means(
            values,
            block,
            n_blocks,
            max_start,
            n_bootstrap,
            None,
            starts=starts,
        )
        distributions.append(means)
        weights.append(float(values.size))
        total_weight += float(values.size)
    if total_weight <= 0.0:
        raise ValueError("pooled bootstrap requires positive segment weights")

    pooled = np.zeros(n_bootstrap, dtype=np.float64)
    for weight, distribution in zip(weights, distributions, strict=True):
        pooled += weight * distribution
    pooled /= total_weight

    observed = float(
        sum(
            float(np.sum(np.asarray(raw, dtype=np.float64)))
            for raw in segments
        )
        / total_weight
    )
    gate = 2.0 * observed
    margin = max(_bootstrap_error_bound(np.asarray(raw, dtype=np.float64)) for raw in segments)
    if bool(np.any(np.abs(pooled - gate) <= margin)):
        exact = tuple(
            moving_block_bootstrap_means(
                np.asarray(raw, dtype=np.float64),
                _effective_block_length(np.asarray(raw, dtype=np.float64).size, block_length),
                n_bootstrap,
                seed + i,
            )
            for i, raw in enumerate(segments)
        )
        pooled = np.zeros(n_bootstrap, dtype=np.float64)
        for weight, distribution in zip(weights, exact, strict=True):
            pooled += weight * distribution
        pooled /= total_weight
    return pooled
