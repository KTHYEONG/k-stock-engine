"""Walk-forward LightGBM horizon-ensemble scorer over a causal feature panel."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.data.research_protocol import WindowAuthorization, WindowError
from src.research.panel import FEATURE_SETS, FeaturePanel, feature_names_for

__all__ = ["ScoreMatrix", "ScorerConfig", "_member_seeds", "walk_forward_scores"]

_LOG = logging.getLogger(__name__)


class ScorerConfig(BaseModel):
    """Frozen hyper-parameters of the ensemble scorer; its canonical JSON is part of the strategy identity.

    Why frozen: a scorer change is a new strategy identity that must win a champion/challenger comparison;
    silently re-tuning in place would make every stored run irreproducible.

    Fields (defaults): horizons=(5, 10, 21); num_boost_round=300; learning_rate=0.03; num_leaves=15;
    min_data_in_leaf=800; feature_fraction=0.7; bagging_fraction=0.7; bagging_freq=1; lambda_l2=50.0;
    seed=11; num_threads=8; min_cross_section=50; winsor_low_pct=1.0; winsor_high_pct=99.0;
    purge_extra_sessions=2; min_train_rows=20000; first_test_year=2018; feature_set="full62" — name in
    ``FEATURE_SETS``; part of the identity; seeds=() — ensemble member seeds; empty means the single `seed`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizons: tuple[int, ...] = (5, 10, 21)
    num_boost_round: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_data_in_leaf: int = 800
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    lambda_l2: float = 50.0
    seed: int = 11
    num_threads: int = 8
    min_cross_section: int = 50
    winsor_low_pct: float = 1.0
    winsor_high_pct: float = 99.0
    purge_extra_sessions: int = 2
    min_train_rows: int = 20000
    first_test_year: int = 2018
    feature_set: str = "full62"
    seeds: tuple[int, ...] = ()

    @field_validator("seeds", mode="before")
    @classmethod
    def _seeds_members(cls, value: object) -> tuple[int, ...]:
        if isinstance(value, tuple) and len(value) == 0:
            return ()
        if isinstance(value, list):
            items: tuple[object, ...] = tuple(value)
        elif isinstance(value, tuple):
            items = tuple(value)
        else:
            raise ValueError(f"seeds must be empty or hold >= 2 distinct ints >= 0, got {value!r}")
        if len(items) == 0:
            return ()
        if len(items) < 2:
            raise ValueError(f"seeds must be empty or hold >= 2 distinct ints >= 0, got {value!r}")
        seen: set[int] = set()
        out: list[int] = []
        for entry in items:
            if isinstance(entry, bool):
                raise ValueError(f"seeds must be empty or hold >= 2 distinct ints >= 0, got {value!r}")
            if not isinstance(entry, int):
                raise ValueError(f"seeds must be empty or hold >= 2 distinct ints >= 0, got {value!r}")
            if entry < 0:
                raise ValueError(f"seeds must be empty or hold >= 2 distinct ints >= 0, got {value!r}")
            if entry in seen:
                raise ValueError(f"seeds must hold distinct values, got {value!r}")
            seen.add(entry)
            out.append(entry)
        return tuple(out)

    @model_validator(mode="after")
    def _seed_identity(self) -> ScorerConfig:
        if self.seeds and int(self.seed) != int(self.seeds[0]):
            raise ValueError(f"seed must equal seeds[0] when seeds is set, got {self.seed!r}")
        return self

    @field_validator("feature_set")
    @classmethod
    def _known_feature_set(cls, value: object) -> str:
        if not isinstance(value, str) or value not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}, got {value!r}")
        return value

    @field_validator("horizons")
    @classmethod
    def _non_empty_horizons(cls, value: object) -> tuple[int, ...]:
        items = tuple(value) if isinstance(value, (list, tuple)) else ()
        if not items or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in items):
            raise ValueError(f"horizons must be a non-empty tuple of positive ints, got {value!r}")
        return tuple(int(h) for h in items)

    @field_validator("winsor_low_pct", "winsor_high_pct")
    @classmethod
    def _pct_range(cls, value: object) -> float:
        pct = float(value)  # type: ignore[arg-type]
        if not 0.0 <= pct <= 100.0:
            raise ValueError(f"winsor percentile must satisfy 0 <= p <= 100, got {value!r}")
        return pct

    def canonical_json(self) -> str:
        """Canonical JSON of the config with sorted keys and compact separators.

        ``feature_set`` is emitted only when non-default so pre-set specs keep their hash.
        """
        payload = self.model_dump(mode="json")
        if self.feature_set == "full62":
            del payload["feature_set"]
        if not self.seeds:
            del payload["seeds"]
        else:
            payload["seeds"] = list(self.seeds)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        """SHA-256 hex of the canonical JSON; part of the strategy identity."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _member_seeds(config: ScorerConfig) -> tuple[int, ...]:
    """Seeds of the ensemble members: ``config.seeds`` when set, else ``(config.seed,)``."""
    if config.seeds:
        return tuple(int(s) for s in config.seeds)
    return (int(config.seed),)


@dataclass(frozen=True, slots=True)
class ScoreMatrix:
    """Ensemble scores (S' x N float32) valid only on universe rows of the requested test years.

    scores: NaN outside the universe, outside the test years, or where the cross-section was too thin.
    test_years: years scored. config_hash/last_row: provenance for cache validation.
    """

    scores: NDArray[np.float32]
    test_years: tuple[int, ...]
    config_hash: str
    last_row: int


def _weekly_rows(sessions: Sequence[date], *, lo: int, hi: int) -> tuple[int, ...]:
    days = list(sessions)
    total = len(days)
    if not 0 <= lo <= hi <= total:
        raise ValueError(f"window [{lo}, {hi}) is not within [0, {total})")
    out: list[int] = []
    for t in range(lo, hi):
        if t + 1 >= total:
            continue
        cur_iso = days[t].isocalendar()
        nxt_iso = days[t + 1].isocalendar()
        if (cur_iso[0], cur_iso[1]) != (nxt_iso[0], nxt_iso[1]):
            out.append(t)
    return tuple(out)


def _rank_stack(
    feat_mats: Sequence[NDArray[np.float32]],
    universe: NDArray[np.bool_],
    rows: Sequence[int],
    min_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Cross-sectionally rank-normalize features to (0, 1) per date within the universe."""
    uni = np.asarray(universe, dtype=bool)
    n_feat = len(feat_mats)
    blocks: list[NDArray[np.float64]] = []
    row_parts: list[NDArray[np.int64]] = []
    col_parts: list[NDArray[np.int64]] = []
    for r in rows:
        idx = np.flatnonzero(uni[int(r)])
        if idx.size == 0:
            continue
        cols = np.full((idx.size, n_feat), np.nan, dtype=np.float64)
        for j, mat in enumerate(feat_mats):
            v = np.asarray(mat[int(r), idx], dtype=np.float64)
            finite = np.isfinite(v)
            count = int(finite.sum())
            if count >= min_count:
                order = np.argsort(np.argsort(v[finite], kind="stable"), kind="stable")
                col = np.full(idx.size, np.nan, dtype=np.float64)
                col[finite] = (order.astype(np.float64) + 0.5) / float(count)
                cols[:, j] = col
        blocks.append(cols)
        row_parts.append(np.full(idx.size, int(r), dtype=np.int64))
        col_parts.append(idx.astype(np.int64))
    if not blocks:
        return (
            np.zeros((0, n_feat), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return (np.vstack(blocks), np.concatenate(row_parts), np.concatenate(col_parts))


def _winsor_z_by_date(
    labels: NDArray[np.float32],
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    low_pct: float,
    high_pct: float,
    min_count: int,
) -> NDArray[np.float64]:
    """Per-date winsorized z-score of labels at the stacked sample positions."""
    target = np.full(rows.shape, np.nan, dtype=np.float64)
    if rows.size == 0:
        return target
    for r in np.unique(rows):
        sel = np.flatnonzero(rows == r)
        y = np.asarray(labels[int(r), cols[sel]], dtype=np.float64)
        finite = np.isfinite(y)
        if int(finite.sum()) < min_count:
            continue
        yy = y[finite]
        lo, hi = np.percentile(yy, [low_pct, high_pct])
        yw = np.clip(yy, lo, hi)
        mu = float(yw.mean())
        sd = float(yw.std())
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        target[sel[finite]] = (yw - mu) / sd
    return target


def _lgbm_params(config: ScorerConfig, seed: int) -> dict[str, Any]:
    return {
        "objective": "regression",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "lambda_l2": config.lambda_l2,
        "verbose": -1,
        "num_threads": config.num_threads,
        "seed": seed,
        "bagging_seed": (seed + 1) % 2147483647,
        "feature_fraction_seed": (seed + 2) % 2147483647,
        "data_random_seed": (seed + 3) % 2147483647,
        "deterministic": True,
        "force_row_wise": True,
    }


def _train_booster(
    train_x: NDArray[np.float64],
    train_y: NDArray[np.float64],
    params: dict[str, Any],
    num_boost_round: int,
    seed: int,
    rows: NDArray[np.int64] | None = None,
) -> lgb.Booster:
    """Train one LightGBM regressor; ``rows`` carries the session row per sample for test doubles."""
    _ = (seed, rows)
    dataset = lgb.Dataset(np.asarray(train_x, dtype=np.float64), np.asarray(train_y, dtype=np.float64))
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def _fold_workers(threads_per_booster: int) -> int:
    """Bound concurrent fits while preserving each booster's configured thread count."""
    if threads_per_booster <= 0:
        return 1  # LightGBM uses the OpenMP default thread count for non-positive values.
    return max(1, (os.cpu_count() or 1) // threads_per_booster)


def _zscore_rows(mat: NDArray[np.float64], min_count: int) -> NDArray[np.float64]:
    out = np.full(mat.shape, np.nan, dtype=np.float64)
    for r in range(mat.shape[0]):
        row = np.asarray(mat[r], dtype=np.float64)
        mask = np.isfinite(row)
        if int(mask.sum()) < min_count:
            continue
        vals = row[mask]
        mu = float(vals.mean())
        sd = float(vals.std())
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        out[r, mask] = (vals - mu) / sd
    return out


def _ensemble_average(parts: Sequence[NDArray[np.float64]], min_count: int) -> NDArray[np.float64]:
    scored = [_zscore_rows(np.asarray(part, dtype=np.float64), min_count) for part in parts]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.stack(scored, axis=0), axis=0)
    return np.asarray(mean, dtype=np.float64)


def walk_forward_scores(
    panel: FeaturePanel,
    universe: NDArray[np.bool_],
    sessions: Sequence[date],
    config: ScorerConfig,
    *,
    test_years: Sequence[int],
    authorization: WindowAuthorization,
) -> ScoreMatrix:
    """Train one LightGBM regressor per horizon and year on strictly earlier data; score each test year.

    Training rows: weekly decision rows (last session of each ISO week) inside the universe whose label
    window ended before the test year starts, i.e. row + horizon + purge_extra_sessions < first row of the
    test year. Features are cross-sectionally rank-normalized to (0, 1) per date within the universe
    (NaN kept; a date with fewer than ``min_cross_section`` finite values yields all-NaN for that feature).
    Target: per-date winsorized z-score of the horizon label within the universe. Prediction rows: every
    universe row of the test year (daily), so any rebalance phase can be evaluated.
    Ensemble: per-horizon scores are z-scored within each universe row, then averaged with NaN-mean.

    Why ranked features + z target: the probe showed the alpha sits in the return tail; ordinal losses
    (LambdaRank, XENDCG, top-decile classification) and robust losses (L1, Huber) destroyed it.

    Memory: feature matrices are read in the panel's float32 dtype. Only the rank-normalised design rows are
    float64, built one session row at a time. A float32-to-float64 cast is exact, so scores are bitwise
    identical to casting the whole panel first.

    Raises:
        WindowError: a test year ends after ``authorization.end``.
        ValueError: a fold has fewer than ``min_train_rows`` training rows, ``test_years`` is empty, is
            not ascending, or begins before ``config.first_test_year``.
    """
    years = list(test_years)
    if not years:
        raise ValueError("test_years must be non-empty")
    if any(y < config.first_test_year for y in years):
        raise ValueError(f"test_years {years} begins before first_test_year {config.first_test_year}")
    if list(years) != sorted(years) or len(set(years)) != len(years):
        raise ValueError(f"test_years must be strictly ascending, got {years}")
    feat_mats = [np.asarray(panel.features[name]) for name in feature_names_for(config.feature_set)]
    shape = feat_mats[0].shape
    if any(mat.shape != shape for mat in feat_mats):
        raise ValueError("panel feature arrays have mismatched shapes")
    n_rows, n_inst = shape
    uni = np.asarray(universe, dtype=bool)
    if uni.shape != (n_rows, n_inst):
        raise ValueError(f"universe shape {uni.shape} != panel shape {(n_rows, n_inst)}")
    if panel.last_row >= len(sessions):
        raise ValueError("panel last_row is outside the sessions timeline")
    sess = list(sessions)[:n_rows]
    year_rows: dict[int, list[int]] = {}
    for year in years:
        members = [r for r, day in enumerate(sess) if day.year == year]
        if not members:
            raise ValueError(f"test year {year} has no sessions in the panel")
        if max(sess[r] for r in members) > authorization.end:
            raise WindowError(f"test year {year} ends after the authorization end {authorization.end}")
        year_rows[year] = members
    weekly = list(_weekly_rows(sess, lo=0, hi=n_rows))
    train_x, train_r, train_i = _rank_stack(feat_mats, uni, weekly, config.min_cross_section)
    horizons = tuple(int(h) for h in config.horizons)
    targets: dict[int, NDArray[np.float64]] = {}
    for horizon in horizons:
        lab = np.asarray(panel.labels[int(horizon)])
        if lab.shape != (n_rows, n_inst):
            raise ValueError(f"label array for horizon {horizon} has wrong shape {lab.shape}")
        targets[horizon] = _winsor_z_by_date(
            lab, train_r, train_i, config.winsor_low_pct, config.winsor_high_pct, config.min_cross_section
        )
    member_seeds = _member_seeds(config)
    if len(member_seeds) == 1:
        per_horizon: list[NDArray[np.float64]] = [
            np.full((n_rows, n_inst), np.nan, dtype=np.float64) for _ in horizons
        ]
        workers = min(len(horizons), _fold_workers(config.num_threads))
        for year in years:
            first = year_rows[year][0]
            end = year_rows[year][-1] + 1
            pred_x, pred_r, pred_i = _rank_stack(
                feat_mats, uni, range(first, end), config.min_cross_section
            )
            train_args: list[tuple[int, int, NDArray[np.float64], NDArray[np.float64], dict[str, Any], NDArray[np.int64]]] = []
            for hi, horizon in enumerate(horizons):
                target = targets[horizon]
                mask = (train_r + int(horizon) + config.purge_extra_sessions < first) & np.isfinite(target)
                n_train = int(mask.sum())
                _LOG.info("[ALGO] fold year=%d horizon=%d rows=%d", year, int(horizon), n_train)
                if n_train < config.min_train_rows:
                    raise ValueError(
                        f"fold year={year} horizon={horizon} has {n_train} rows below min_train_rows"
                    )
                seed = (int(member_seeds[0]) + int(horizon) * 100003 + int(year) * 101) % 2147483647
                train_args.append(
                    (
                        hi,
                        seed,
                        np.ascontiguousarray(train_x[mask]),
                        np.ascontiguousarray(target[mask]),
                        _lgbm_params(config, seed),
                        train_r[mask],
                    )
                )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _train_booster, tx, ty, params, config.num_boost_round, seed, rows
                    )
                    for hi, seed, tx, ty, params, rows in train_args
                ]
                boosters = [future.result() for future in futures]
            if pred_r.size:
                for (hi, _seed, _tx, _ty, _params, _rows), booster in zip(train_args, boosters, strict=True):
                    preds = np.asarray(booster.predict(pred_x), dtype=np.float64)
                    per_horizon[hi][pred_r, pred_i] = preds
            del pred_x, pred_r, pred_i, train_args, futures, boosters
        ensemble = _ensemble_average(per_horizon, config.min_cross_section)
        ensemble = np.where(uni, ensemble, np.nan)
        _LOG.info("[ALGO] member seed=%d done", int(member_seeds[0]))
    else:
        n_members = len(member_seeds)
        member_horizon: list[list[NDArray[np.float64]]] = [
            [np.full((n_rows, n_inst), np.nan, dtype=np.float64) for _ in horizons]
            for _ in range(n_members)
        ]
        workers = min(len(horizons) * n_members, _fold_workers(config.num_threads))
        for year in years:
            first = year_rows[year][0]
            end = year_rows[year][-1] + 1
            pred_x, pred_r, pred_i = _rank_stack(
                feat_mats, uni, range(first, end), config.min_cross_section
            )
            fold_design: list[tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]] = []
            for hi, horizon in enumerate(horizons):
                target = targets[horizon]
                mask = (train_r + int(horizon) + config.purge_extra_sessions < first) & np.isfinite(target)
                n_train = int(mask.sum())
                _LOG.info("[ALGO] fold year=%d horizon=%d rows=%d", year, int(horizon), n_train)
                if n_train < config.min_train_rows:
                    raise ValueError(
                        f"fold year={year} horizon={horizon} has {n_train} rows below min_train_rows"
                    )
                fold_design.append(
                    (
                        hi,
                        np.ascontiguousarray(train_x[mask]),
                        np.ascontiguousarray(target[mask]),
                        train_r[mask],
                    )
                )
            train_jobs: list[tuple[int, int, int, NDArray[np.float64], NDArray[np.float64], dict[str, Any], NDArray[np.int64]]] = []
            for mi, member_seed in enumerate(member_seeds):
                for hi, tx, ty, rows in fold_design:
                    horizon = horizons[hi]
                    seed = (int(member_seed) + int(horizon) * 100003 + int(year) * 101) % 2147483647
                    train_jobs.append((mi, hi, seed, tx, ty, _lgbm_params(config, seed), rows))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _train_booster, tx, ty, params, config.num_boost_round, seed, rows
                    )
                    for mi, hi, seed, tx, ty, params, rows in train_jobs
                ]
                boosters = [future.result() for future in futures]
            if pred_r.size:
                for (mi, hi, _seed, _tx, _ty, _params, _rows), booster in zip(train_jobs, boosters, strict=True):
                    preds = np.asarray(booster.predict(pred_x), dtype=np.float64)
                    member_horizon[mi][hi][pred_r, pred_i] = preds
            del pred_x, pred_r, pred_i, fold_design, train_jobs, futures, boosters
        member_scores: list[NDArray[np.float32]] = []
        for mi, member_seed in enumerate(member_seeds):
            part = _ensemble_average(member_horizon[mi], config.min_cross_section)
            part = np.where(uni, part, np.nan)
            member_scores.append(np.ascontiguousarray(part, dtype=np.float32))
            _LOG.info("[ALGO] member seed=%d done", int(member_seed))
        del member_horizon
        ensemble = _ensemble_average(
            [np.asarray(part, dtype=np.float64) for part in member_scores],
            config.min_cross_section,
        )
        ensemble = np.where(uni, ensemble, np.nan)
    scores32 = np.ascontiguousarray(ensemble, dtype=np.float32)
    scores32.flags.writeable = False
    _LOG.info("[ALGO] scores years=%s horizons=%s", tuple(years), tuple(config.horizons))
    return ScoreMatrix(
        scores=scores32,
        test_years=tuple(years),
        config_hash=config.config_hash,
        last_row=panel.last_row,
    )
