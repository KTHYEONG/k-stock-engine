# mypy: ignore-errors
"""Pure lower-bound, drawdown, bootstrap helpers."""
# ruff: noqa: PERF401
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def segmented_moving_block_lower_bound(values: np.ndarray, segment_ids: np.ndarray, *, alpha: float, resamples: int, minimum_tail_draws: int, block_length: int, seed: int) -> float:
    arr = np.asarray(values, dtype=np.float64)
    seg = np.asarray(segment_ids)
    if arr.ndim != 1 or seg.ndim != 1 or arr.shape[0] != seg.shape[0]:
        raise ValueError("values and segment_ids must be aligned 1-D arrays")
    if arr.size == 0:
        raise ValueError("values empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if resamples < 2:
        raise ValueError("resamples must be at least 2")
    if minimum_tail_draws < 1:
        raise ValueError("minimum_tail_draws must be positive")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    effective = int(resamples)
    required = math.ceil(minimum_tail_draws / alpha) if alpha > 0 else resamples
    if effective * alpha < minimum_tail_draws:
        effective = int(required)
    unique = np.unique(seg)
    blocks: list[np.ndarray] = []
    for sid in unique:
        mask = seg == sid
        idxs = np.where(mask)[0]
        seg_vals = arr[idxs]
        n_seg = seg_vals.size
        if n_seg == 0:
            continue
        if n_seg < block_length:
            blocks.append(seg_vals.copy())
        else:
            for start in range(n_seg - block_length + 1):
                blocks.append(seg_vals[start:start + block_length].copy())
    if not blocks:
        raise ValueError("no blocks formed")
    rng = np.random.default_rng(int(seed))
    n = arr.size
    replicate_means = np.empty(effective, dtype=np.float64)
    num_blocks_needed_max = math.ceil(n / block_length) + 1
    for r in range(effective):
        block_choices = rng.integers(0, len(blocks), size=num_blocks_needed_max)
        assembled = np.empty(n, dtype=np.float64)
        pos = 0
        for choice in block_choices:
            block = blocks[int(choice)]
            take = min(block.size, n - pos)
            assembled[pos:pos + take] = block[:take]
            pos += take
            if pos >= n:
                break
        replicate_means[r] = float(np.mean(assembled))
    return float(np.quantile(replicate_means, float(alpha)))

def log_growth_max_drawdown(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    cum = np.cumsum(arr)
    peaks = np.maximum.accumulate(cum)
    drawdowns = peaks - cum
    return float(np.max(drawdowns))
