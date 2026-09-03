"""Dependence-preserving bootstrap for log returns."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class BootstrapMethod(StrEnum):
    MOVING_BLOCK = "moving_block"
    STATIONARY = "stationary"


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    method: BootstrapMethod
    resamples: int
    block_length_sessions: int
    seed: int
    promotion_run: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.method, BootstrapMethod):
            raise ValueError("method must be BootstrapMethod")
        # resamples validation: must be integral finite positive
        if isinstance(self.resamples, bool) or not isinstance(self.resamples, int):
            raise ValueError("resamples must be integer")
        if isinstance(self.block_length_sessions, bool) or not isinstance(self.block_length_sessions, int):
            raise ValueError("block_length_sessions must be integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be integer")
        if not isinstance(self.promotion_run, bool):
            raise ValueError("promotion_run must be bool")
        if not math.isfinite(float(self.resamples)) or self.resamples < 1:
            raise ValueError("resamples must be at least 1")
        if self.block_length_sessions < 20 or self.block_length_sessions > 60:
            raise ValueError("block_length_sessions must be between 20 and 60")
        if self.promotion_run and self.resamples < 5000:
            raise ValueError("promotion_run requires resamples >= 5000")


@dataclass(frozen=True, slots=True)
class BootstrapDistribution:
    values: tuple[float, ...]
    config: BootstrapConfig

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise ValueError("values must be tuple")
        for v in self.values:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("values must be finite floats")
            if not math.isfinite(float(v)):
                raise ValueError("values must be finite")


def moving_block_bootstrap_indices(
    observation_count: int,
    config: BootstrapConfig,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise ValueError("observation_count must be integer")
    if observation_count < config.block_length_sessions:
        raise ValueError("observation_count must be at least block_length_sessions")
    if observation_count <= 0:
        raise ValueError("observation_count must be positive")
    rng = np.random.default_rng(int(config.seed))
    n = observation_count
    b = int(config.block_length_sessions)
    k = int(config.resamples)
    result: list[tuple[int, ...]] = []
    for _ in range(k):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            block = [(start + i) % n for i in range(b)]
            indices.extend(block)
        result.append(tuple(indices[:n]))
    return tuple(result)


def stationary_bootstrap_indices(
    observation_count: int,
    config: BootstrapConfig,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise ValueError("observation_count must be integer")
    if observation_count < config.block_length_sessions:
        raise ValueError("observation_count must be at least block_length_sessions")
    if observation_count <= 0:
        raise ValueError("observation_count must be positive")
    rng = np.random.default_rng(int(config.seed))
    n = observation_count
    k = int(config.resamples)
    p = 1.0 / float(config.block_length_sessions)
    result: list[tuple[int, ...]] = []
    for _ in range(k):
        indices: list[int] = []
        first = int(rng.integers(0, n))
        indices.append(first)
        for _ in range(1, n):
            if float(rng.random()) < p:  # noqa: SIM108
                nxt = int(rng.integers(0, n))
            else:
                nxt = (indices[-1] + 1) % n
            indices.append(nxt)
        result.append(tuple(indices))
    return tuple(result)


def bootstrap_annualized_log_growth(
    log_returns: tuple[float, ...],
    config: BootstrapConfig,
    *,
    sessions_per_year: int = 252,
) -> BootstrapDistribution:
    if not isinstance(log_returns, tuple):
        raise ValueError("log_returns must be tuple")
    if len(log_returns) == 0:
        raise ValueError("log_returns must be non-empty")
    for v in log_returns:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("log_returns must be finite")
        if not math.isfinite(float(v)):
            raise ValueError("log_returns must be finite")
    if isinstance(sessions_per_year, bool) or not isinstance(sessions_per_year, int):
        raise ValueError("sessions_per_year must be integer")
    if sessions_per_year <= 0:
        raise ValueError("sessions_per_year must be positive")
    n = len(log_returns)
    if n < config.block_length_sessions:
        raise ValueError("observation_count must be at least block_length_sessions")
    # Generate indices deterministic via config.method
    if config.method is BootstrapMethod.MOVING_BLOCK:
        paths = moving_block_bootstrap_indices(n, config)
    elif config.method is BootstrapMethod.STATIONARY:
        paths = stationary_bootstrap_indices(n, config)
    else:
        raise ValueError("unknown bootstrap method")
    values: list[float] = []
    for path in paths:
        resampled = [float(log_returns[i]) for i in path]
        s = sum(resampled)
        # annualized = sessions_per_year / n * sum
        annualized = float(sessions_per_year) / float(n) * float(s)
        if not math.isfinite(annualized):
            raise ValueError("annualized log growth must be finite")
        values.append(float(annualized))
    return BootstrapDistribution(values=tuple(values), config=config)
