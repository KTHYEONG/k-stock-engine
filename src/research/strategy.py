"""Declarative long-only top-N strategy specs and screening target construction."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from src.data.research_protocol import BenchmarkPolicy
from src.research.cube import ResearchCube
from src.research.features import FEATURES

__all__ = [
    "JunkFilter",
    "StrategySpec",
    "UniverseRule",
    "benchmark_targets",
    "load_strategy_spec",
    "rebalance_rows",
    "target_weights",
    "universe_mask",
]


class UniverseRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    min_adtv20_krw: int
    min_price_krw: int

    @field_validator("min_adtv20_krw", "min_price_krw")
    @classmethod
    def _non_negative_int(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"universe threshold must be a non-negative int, got {value!r}")
        return value


class JunkFilter(BaseModel):
    """Exclude names whose within-universe percentile of any listed feature is below ``min_rank``.

    Features are oriented so higher is better (e.g. ``lowturn``), so a low percentile marks a
    lottery/attention name whose long-side expected return is poor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    features: tuple[str, ...]
    min_rank: float

    @field_validator("min_rank")
    @classmethod
    def _rank_range(cls, value: float) -> float:
        rank = float(value)
        if not 0.0 <= rank < 1.0 or not math.isfinite(rank):
            raise ValueError(f"min_rank must satisfy 0 <= min_rank < 1, got {value!r}")
        return rank

    @field_validator("features")
    @classmethod
    def _known_features(cls, value: object) -> tuple[str, ...]:
        items = tuple(value) if isinstance(value, (list, tuple)) else ()
        for name in items:
            if name not in FEATURES:
                raise ValueError(f"unknown feature: {name!r}")
        return items


class StrategySpec(BaseModel):
    """Declarative long-only top-N strategy; its hash is the trial identity of the logic."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    family: str
    universe: UniverseRule
    junk: JunkFilter | None = None
    score: dict[str, float]
    n: int
    keep_rank_multiple: float
    rebalance: Literal["D", "W", "2W", "M", "Q"]

    @field_validator("family")
    @classmethod
    def _non_empty_family(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("family must be a non-empty string")
        return value

    @field_validator("n")
    @classmethod
    def _positive_n(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"n must be >= 1, got {value!r}")
        return value

    @field_validator("keep_rank_multiple")
    @classmethod
    def _keep_range(cls, value: float) -> float:
        multiple = float(value)
        if not math.isfinite(multiple) or multiple < 1.0:
            raise ValueError(f"keep_rank_multiple must be >= 1, got {value!r}")
        return multiple

    @field_validator("score")
    @classmethod
    def _valid_score(cls, value: dict[str, float]) -> dict[str, float]:
        if not isinstance(value, dict) or not value:
            raise ValueError("score must be a non-empty mapping")
        cleaned: dict[str, float] = {}
        for name, weight in value.items():
            if name not in FEATURES:
                raise ValueError(f"unknown feature: {name!r}")
            w = float(weight)
            if not math.isfinite(w) or w == 0.0:
                raise ValueError(f"score weight must be a finite non-zero number, got {weight!r}")
            cleaned[str(name)] = w
        return cleaned

    def canonical_json(self) -> str:
        """Canonical JSON of the spec with sorted keys and compact separators."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        """SHA-256 hex of the canonical JSON; the trial identity of the logic."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def feature_names(self) -> tuple[str, ...]:
        """Sorted union of junk and score feature names required by the spec."""
        names = set(self.score)
        if self.junk is not None:
            names.update(self.junk.features)
        return tuple(sorted(names))


def load_strategy_spec(path: Path) -> StrategySpec:
    """Load a TOML strategy spec file."""
    import tomllib

    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    return StrategySpec.model_validate(raw)


def _as_float(cube: ResearchCube, name: str) -> NDArray[np.float64]:
    return np.asarray(cube.arrays[name], dtype=np.float64)


def _as_bool(cube: ResearchCube, name: str) -> NDArray[np.bool_]:
    return np.asarray(cube.arrays[name], dtype=bool)


def universe_mask(cube: ResearchCube, rule: UniverseRule) -> NDArray[np.bool_]:
    """Universe mask for every row under the given liquidity and price floors."""
    present = _as_bool(cube, "present")
    eligible = _as_bool(cube, "eligible")
    blocked = _as_bool(cube, "entry_blocked")
    volume = _as_float(cube, "volume")
    adtv20 = _as_float(cube, "adtv20")
    close = _as_float(cube, "close")
    vol60 = _as_float(cube, "ret_vol60")
    mask: NDArray[np.bool_] = (
        present
        & eligible
        & ~blocked
        & (volume > 0)
        & (adtv20 >= float(rule.min_adtv20_krw))
        & (close >= float(rule.min_price_krw))
        & np.isfinite(vol60)
    )
    return mask


def rebalance_rows(sessions: Sequence[date], *, lo: int, hi: int, freq: str) -> tuple[int, ...]:
    """Decision rows ``t`` in ``[lo, hi)`` that trigger a rebalance."""
    days = list(sessions)
    total = len(days)
    if not isinstance(lo, int) or isinstance(lo, bool) or not isinstance(hi, int) or isinstance(hi, bool):
        raise ValueError(f"lo/hi must be ints, got {lo!r}, {hi!r}")
    if not 0 <= lo <= hi <= total:
        raise ValueError(f"window [{lo}, {hi}) is not within [0, {total})")
    if freq not in ("D", "W", "2W", "M", "Q"):
        raise ValueError(f"unknown rebalance frequency: {freq!r}")
    out: list[int] = []
    for t in range(lo, hi):
        if freq == "D":
            out.append(t)
            continue
        if t + 1 >= total:
            continue
        current, nxt = days[t], days[t + 1]
        if freq == "M":
            if current.month != nxt.month:
                out.append(t)
        elif freq == "Q":
            if current.month != nxt.month and current.month in (3, 6, 9, 12):
                out.append(t)
        else:
            cur_iso = current.isocalendar()
            nxt_iso = nxt.isocalendar()
            if (cur_iso[0], cur_iso[1]) != (nxt_iso[0], nxt_iso[1]) and (
                freq == "W" or cur_iso[1] % 2 == 0
            ):
                out.append(t)
    return tuple(out)


def _percentile_ranks(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Stable ascending percentile ranks ``(position + 0.5) / count`` within a mask."""
    ranks = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(np.asarray(values, dtype=np.float64))
    idx = np.flatnonzero(valid)
    count = idx.size
    if count < 5:
        return ranks
    order = np.argsort(np.asarray(values, dtype=np.float64)[idx], kind="stable")
    positions = np.empty(count, dtype=np.float64)
    positions[order] = (np.arange(count, dtype=np.float64) + 0.5) / float(count)
    ranks[idx] = positions
    return ranks


def _select_row(
    *,
    spec: StrategySpec,
    row_features: dict[str, NDArray[np.float64]],
    universe_row: NDArray[np.bool_],
    incumbents: set[int],
) -> list[int]:
    n_names = universe_row.size
    if spec.junk is not None and spec.junk.features:
        survivors = np.asarray(universe_row, dtype=bool).copy()
        for name in spec.junk.features:
            ranks = _percentile_ranks(row_features[name], np.asarray(universe_row, dtype=bool))
            keep = np.isfinite(ranks) & (ranks >= float(spec.junk.min_rank))
            survivors &= keep
    else:
        survivors = np.asarray(universe_row, dtype=bool).copy()
    if not bool(np.any(survivors)):
        return []
    weights = spec.score
    total_abs = sum(abs(w) for w in weights.values())
    score = np.full(n_names, np.nan, dtype=np.float64)
    rank_cols: dict[str, NDArray[np.float64]] = {}
    for name in weights:
        rank_cols[name] = _percentile_ranks(row_features[name], survivors)
    for n in np.flatnonzero(survivors):
        acc = 0.0
        ok = True
        for name, weight in weights.items():
            rank = float(rank_cols[name][n])
            if not math.isfinite(rank):
                ok = False
                break
            adj = rank if weight > 0 else 1.0 - rank
            acc += abs(weight) * adj
        if ok:
            score[n] = acc / total_abs
    ranked = sorted(
        (n for n in np.flatnonzero(survivors) if math.isfinite(float(score[n]))),
        key=lambda n: (-float(score[int(n)]), int(n)),
    )
    if not ranked:
        return []
    position_of = {int(n): pos for pos, n in enumerate(ranked)}
    keep_limit = int(float(spec.n) * float(spec.keep_rank_multiple))
    kept: list[int] = []
    for n in ranked:
        key = int(n)
        if key in incumbents and position_of[key] < keep_limit and len(kept) < spec.n:
            kept.append(key)
    kept_set = set(kept)
    for n in ranked:
        if len(kept) >= spec.n:
            break
        if int(n) not in kept_set:
            kept.append(int(n))
    return kept


def target_weights(
    spec: StrategySpec,
    cube: ResearchCube,
    features: Mapping[str, NDArray[np.float64]],
    *,
    lo: int,
    hi: int,
) -> dict[int, NDArray[np.float64]]:
    """Target weights (fractions of NAV) on each decision row in ``[lo, hi)``.

    Returns:
        Mapping decision row -> length-N weights summing to 1 (equal weights over the chosen names)
        or all zeros when nothing qualifies.
    """
    sessions = list(cube.sessions)
    total = len(sessions)
    if not 0 <= lo <= hi <= total:
        raise ValueError(f"window [{lo}, {hi}) is not within [0, {total})")
    n_names = len(cube.instrument_ids)
    for name, arr in features.items():
        if np.asarray(arr).shape != (total, n_names):
            raise ValueError(f"feature array has wrong shape: {name}")
    needed = spec.feature_names()
    for name in needed:
        if name not in features:
            raise KeyError(name)
    universe = np.asarray(universe_mask(cube, spec.universe), dtype=bool)
    rows = rebalance_rows(sessions, lo=lo, hi=hi, freq=spec.rebalance)
    row_set = set(rows)
    out: dict[int, NDArray[np.float64]] = {}
    incumbents: set[int] = set()
    for t in rows:
        per_row: dict[str, NDArray[np.float64]] = {}
        for name in needed:
            col = np.asarray(features[name][t], dtype=np.float64)
            arr1 = np.full(n_names, np.nan, dtype=np.float64)
            arr1[:] = col
            per_row[name] = arr1
        chosen = _select_row(
            spec=spec, row_features=per_row, universe_row=np.asarray(universe[t], dtype=bool),
            incumbents=set(incumbents),
        )
        weights = np.zeros(n_names, dtype=np.float64)
        if chosen:
            share = 1.0 / len(chosen)
            for n in chosen:
                weights[int(n)] = share
            incumbents = {int(n) for n in chosen}
        else:
            incumbents = set()
        out[int(t)] = np.ascontiguousarray(weights)
    for t in sorted(row_set):
        out[t] = np.ascontiguousarray(np.asarray(out[t], dtype=np.float64))
    return out


def benchmark_targets(
    cube: ResearchCube,
    policy: BenchmarkPolicy,
    *,
    lo: int,
    hi: int,
) -> dict[str, dict[int, NDArray[np.float64]]]:
    """Protocol benchmarks: ``U_EW`` (equal weight over the benchmark universe) and ``CW``.

    ``CW`` is the market-cap weight over present, eligible, unblocked, traded names.
    Both use monthly rows.
    """
    sessions = list(cube.sessions)
    total = len(sessions)
    if not 0 <= lo <= hi <= total:
        raise ValueError(f"window [{lo}, {hi}) is not within [0, {total})")
    n_names = len(cube.instrument_ids)
    rows = rebalance_rows(sessions, lo=lo, hi=hi, freq=policy.rebalance)
    rule = UniverseRule(min_adtv20_krw=policy.universe_min_adtv20_krw, min_price_krw=policy.universe_min_price_krw)
    universe = np.asarray(universe_mask(cube, rule), dtype=bool)
    present = _as_bool(cube, "present")
    eligible = _as_bool(cube, "eligible")
    blocked = _as_bool(cube, "entry_blocked")
    if "traded" in cube.arrays:
        traded = _as_bool(cube, "traded")
    else:
        volume = _as_float(cube, "volume")
        open_px = _as_float(cube, "open")
        traded = np.asarray(present & (volume > 0) & (open_px > 0), dtype=bool)
    mcap = _as_float(cube, "market_cap")
    uew: dict[int, NDArray[np.float64]] = {}
    cap: dict[int, NDArray[np.float64]] = {}
    for t in rows:
        urow = np.asarray(universe[t], dtype=bool)
        weights = np.zeros(n_names, dtype=np.float64)
        idx = np.flatnonzero(urow)
        if idx.size:
            weights[idx] = 1.0 / float(idx.size)
        uew[int(t)] = np.ascontiguousarray(weights)
        cw_mask = (
            np.asarray(present[t], dtype=bool)
            & np.asarray(eligible[t], dtype=bool)
            & ~np.asarray(blocked[t], dtype=bool)
            & np.asarray(traded[t], dtype=bool)
            & np.isfinite(np.asarray(mcap[t], dtype=np.float64))
            & (np.asarray(mcap[t], dtype=np.float64) > 0)
        )
        cw = np.zeros(n_names, dtype=np.float64)
        cidx = np.flatnonzero(cw_mask)
        if cidx.size:
            caps = np.asarray(mcap[t], dtype=np.float64)[cidx]
            cw[cidx] = caps / float(np.sum(caps))
        cap[int(t)] = np.ascontiguousarray(cw)
    return {"U_EW": uew, "CW": cap}
