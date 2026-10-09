"""Causal feature panel and open-to-open forward labels from a research cube."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.research.cube import ResearchCube

__all__ = ["FEATURE_NAMES", "FEATURE_SETS", "HORIZONS", "FeaturePanel", "build_panel", "feature_names_for"]

_LOG = logging.getLogger(__name__)

HORIZONS: Final[tuple[int, ...]] = (5, 10, 21)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "ret_1d",
    "ret_5",
    "ret_10",
    "ret_21",
    "mom_63_21",
    "mom_126_21",
    "mom_252_21",
    "hi52",
    "lo52",
    "dist_hi20",
    "dev_ma20",
    "dev_ma60",
    "dev_ma120",
    "vol20",
    "vol60",
    "vol_ratio",
    "pvol20",
    "range20",
    "max21",
    "min21",
    "skew60",
    "dvol60",
    "beta60",
    "ivol60",
    "on20",
    "id20",
    "on60",
    "id60",
    "gap1",
    "absgap20",
    "size",
    "lnadtv",
    "turn20",
    "vol_surge5_60",
    "tv_z1",
    "amihud20",
    "upvol20",
    "fl_for5",
    "fl_for20",
    "fl_for60",
    "fl_ins5",
    "fl_ins20",
    "fl_ins60",
    "fl_ind5",
    "fl_ind20",
    "fl_ind60",
    "bm",
    "ep",
    "roe",
    "opa",
    "gpa",
    "sue_op",
    "sue_ni",
    "asset_g",
    "accrual",
    "sales_g",
    "sue_op_e",
    "sue_ni_e",
    "sue_sales_e",
    "earn_age",
    "ear",
    "post_ear",
)

# dedup52: greedy removal of features with mean cross-sectional rank correlation
# >= 0.80 to an already retained feature, families ordered by ablation importance.
# dedup52_v1 is the legacy alias of the same tuple so stored specs keep loading;
# new files must use dedup52.
_DEDUP52_V1: Final[tuple[str, ...]] = (
    "upvol20",
    "turn20",
    "lnadtv",
    "size",
    "vol_surge5_60",
    "tv_z1",
    "ret_5",
    "ret_1d",
    "dev_ma20",
    "ret_21",
    "gap1",
    "dist_hi20",
    "fl_ins5",
    "fl_ind60",
    "fl_for5",
    "fl_ins20",
    "fl_for20",
    "fl_for60",
    "fl_ind20",
    "fl_ind5",
    "fl_ins60",
    "sue_ni_e",
    "sue_op_e",
    "post_ear",
    "earn_age",
    "sue_sales_e",
    "ear",
    "absgap20",
    "on20",
    "on60",
    "id20",
    "id60",
    "max21",
    "skew60",
    "ivol60",
    "pvol20",
    "min21",
    "vol_ratio",
    "beta60",
    "lo52",
    "dev_ma60",
    "mom_63_21",
    "mom_126_21",
    "hi52",
    "mom_252_21",
    "ep",
    "opa",
    "bm",
    "gpa",
    "accrual",
    "sales_g",
    "asset_g",
)

_FEATURE_SETS_RAW: Final[dict[str, tuple[str, ...]]] = {
    "full62": FEATURE_NAMES,
    "dedup52": _DEDUP52_V1,
    "dedup52_v1": _DEDUP52_V1,
}

def _validate_feature_sets(sets: Mapping[str, tuple[str, ...]]) -> None:
    for _set_name, _set_names in sets.items():
        if len(set(_set_names)) != len(_set_names):
            raise ValueError(f"feature set {_set_name!r} contains duplicate names")
        unknown = set(_set_names) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"feature set {_set_name!r} has names outside FEATURE_NAMES: {sorted(unknown)}")


_validate_feature_sets(_FEATURE_SETS_RAW)

FEATURE_SETS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(dict(_FEATURE_SETS_RAW))


def feature_names_for(feature_set: str) -> tuple[str, ...]:
    """Ordered feature names the scorer trains on for a named, frozen feature set.

    Why named sets instead of free lists: a feature set is part of the strategy identity; a name that maps to one frozen
    ordered tuple keeps stored champion specs reproducible and makes every change an explicit new set.
    Order matters: LightGBM column sampling is seeded per column position, so a reordering is a different model.

    Raises:
        ValueError: ``feature_set`` is not a key of ``FEATURE_SETS``.
    """
    try:
        return FEATURE_SETS[feature_set]
    except KeyError:
        raise ValueError(f"unknown feature_set {feature_set!r}") from None


_PRICE_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "ret_1d",
        "ret_5",
        "ret_10",
        "ret_21",
        "mom_63_21",
        "mom_126_21",
        "mom_252_21",
        "hi52",
        "lo52",
        "dist_hi20",
        "dev_ma20",
        "dev_ma60",
        "dev_ma120",
        "vol20",
        "vol60",
        "vol_ratio",
        "pvol20",
        "range20",
        "max21",
        "min21",
        "skew60",
        "dvol60",
        "beta60",
        "ivol60",
        "on20",
        "id20",
        "on60",
        "id60",
        "gap1",
        "absgap20",
        "size",
        "lnadtv",
        "turn20",
        "vol_surge5_60",
        "tv_z1",
        "amihud20",
        "upvol20",
    }
)


@dataclass(frozen=True, slots=True)
class FeaturePanel:
    """Session x instrument feature and label arrays derived from one research cube.

    Attributes:
        features: name -> float32 (S', N) array; NaN = undefined. Row t depends only on cube rows <= t.
        labels: horizon h -> float32 (S', N) open-to-open total-return label. Row t is the return of a position
            entered at the session t+1 open and exited at the session t+1+h open, so it is observable only
            after row t+1+h. NaN when the entry/exit row is beyond ``last_row`` or the entry price is unknown.
        last_row: last cube row included (inclusive); S' = last_row + 1.
    """

    features: dict[str, NDArray[np.float32]]
    labels: dict[int, NDArray[np.float32]]
    last_row: int


def _fill(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(np.isfinite(x), x, 0.0)


def _roll_sum(x: NDArray[np.float64], w: int) -> NDArray[np.float64]:
    c = np.cumsum(_fill(x), axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return np.asarray(out, dtype=np.float64)


def _roll_cnt(x: NDArray[np.float64], w: int) -> NDArray[np.float64]:
    return _roll_sum(np.isfinite(x).astype(np.float64), w)


def _roll_mean(x: NDArray[np.float64], w: int, minp: int | None = None) -> NDArray[np.float64]:
    mp = minp if minp is not None else max(2, w // 2)
    s = _roll_sum(x, w)
    c = _roll_cnt(x, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = s / c
    m[c < mp] = np.nan
    return np.asarray(m, dtype=np.float64)


def _roll_std(x: NDArray[np.float64], w: int, minp: int | None = None) -> NDArray[np.float64]:
    mp = minp if minp is not None else max(3, w // 2)
    m1 = _roll_mean(x, w, mp)
    m2 = _roll_mean(x * x, w, mp)
    with np.errstate(invalid="ignore"):
        v = m2 - m1 * m1
    v = np.where(v < 0, 0.0, v)
    return np.sqrt(v)


def _roll_max(x: NDArray[np.float64], w: int, minp: int | None = None) -> NDArray[np.float64]:
    mp = minp if minp is not None else max(2, w // 2)
    return np.asarray(pd.DataFrame(x).rolling(w, min_periods=mp).max().to_numpy(dtype=np.float64))


def _roll_min(x: NDArray[np.float64], w: int, minp: int | None = None) -> NDArray[np.float64]:
    mp = minp if minp is not None else max(2, w // 2)
    return np.asarray(pd.DataFrame(x).rolling(w, min_periods=mp).min().to_numpy(dtype=np.float64))


def _shift(x: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return np.asarray(out, dtype=np.float64)


def build_panel(cube: ResearchCube, *, last_row: int) -> FeaturePanel:
    """Compute features and labels from cube rows ``0..last_row`` only.

    Why last_row: research segments are sealed; slicing before any computation makes it structurally
    impossible for a sealed row to influence a feature or label, regardless of caller discipline.

    Raises:
        ValueError: ``last_row`` outside ``[0, len(cube.sessions) - 1]``.
        KeyError: a cube array required by ``FEATURE_NAMES`` is missing.

    Memory: each feature is stored as float32 as soon as it is final, while every intermediate a later feature
    reads stays a float64 local. The output is therefore bitwise identical to computing everything in float64
    and casting at the end, but the full float64 feature set never exists at once.
    """
    n_sessions = len(cube.sessions)
    if last_row < 0 or last_row >= n_sessions:
        raise ValueError(f"last_row {last_row} outside [0, {n_sessions - 1}]")
    n_out = last_row + 1
    n_inst = len(cube.instrument_ids)

    def _f(name: str) -> NDArray[np.float64]:
        arr = cube.arrays[name]
        return np.asarray(arr, dtype=np.float64)[:n_out]

    pres = np.asarray(cube.arrays["present"], dtype=bool)[:n_out]
    tr = np.where(pres, _f("adj_tr"), np.nan)
    px = _f("adj_px")
    rcc = np.where(pres, _f("ret_cc"), np.nan)
    ron = np.where(pres, _f("r_on"), np.nan)
    rid = np.where(pres, _f("r_id"), np.nan)
    hi = _f("high")
    lo = _f("low")
    cl = _f("close")
    tv = np.where(pres, _f("trading_value"), np.nan)
    mcap_raw = np.where(pres, _f("market_cap"), np.nan)
    mcap = np.where(mcap_raw > 0, mcap_raw, np.nan)

    def _lag_ratio(a: NDArray[np.float64], k: int) -> NDArray[np.float64]:
        lag = _shift(a, k)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(a) & np.isfinite(lag) & (lag > 0), a / lag - 1.0, np.nan)

    w_lag = np.where(np.isfinite(_shift(mcap, 1)) & np.isfinite(rcc), _shift(mcap, 1), 0.0)
    mret = (w_lag * _fill(rcc)).sum(axis=1) / np.maximum(w_lag.sum(axis=1), 1.0)
    mret_col = mret[:, None] * np.ones((1, n_inst))
    del w_lag, mret, mcap_raw

    out: dict[str, NDArray[np.float32]] = {}

    def _keep(name: str, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Store a final feature as float32; return the float64 array for later features to read."""
        src = np.where(pres, arr, np.nan) if name in _PRICE_FEATURES else arr
        out[name] = np.ascontiguousarray(src, dtype=np.float32)
        return arr

    _keep("ret_1d", rcc)
    _keep("ret_5", _lag_ratio(tr, 5))
    _keep("ret_10", _lag_ratio(tr, 10))
    _keep("ret_21", _lag_ratio(tr, 21))
    _keep(
        "mom_63_21",
        np.where(
            np.isfinite(_shift(tr, 21)) & np.isfinite(_shift(tr, 63)),
            _shift(tr, 21) / _shift(tr, 63) - 1.0,
            np.nan,
        ),
    )
    _keep(
        "mom_126_21",
        np.where(
            np.isfinite(_shift(tr, 21)) & np.isfinite(_shift(tr, 126)),
            _shift(tr, 21) / _shift(tr, 126) - 1.0,
            np.nan,
        ),
    )
    _keep(
        "mom_252_21",
        np.where(
            np.isfinite(_shift(tr, 21)) & np.isfinite(_shift(tr, 252)),
            _shift(tr, 21) / _shift(tr, 252) - 1.0,
            np.nan,
        ),
    )
    pk = _roll_max(px, 252, 126)
    _keep("hi52", px / pk)
    del pk
    _keep("lo52", px / _roll_min(px, 252, 126))
    _keep("dist_hi20", px / _roll_max(px, 20, 10))
    for wnd in (20, 60, 120):
        _keep(f"dev_ma{wnd}", px / _roll_mean(px, wnd) - 1.0)
    del px
    vol20 = _keep("vol20", _roll_std(rcc, 20, 10))
    vol60 = _keep("vol60", _roll_std(rcc, 60, 30))
    _keep("vol_ratio", vol20 / vol60)
    with np.errstate(divide="ignore", invalid="ignore"):
        park = np.log(np.where((hi > 0) & (lo > 0), hi / lo, np.nan)) ** 2 / (4 * np.log(2))
    park = np.where(pres, park, np.nan)
    _keep("pvol20", np.sqrt(_roll_mean(park, 20, 10)))
    del park
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = np.where(cl > 0, (hi - lo) / cl, np.nan)
    _keep("range20", _roll_mean(np.where(pres, rng, np.nan), 20, 10))
    del rng, hi, lo, cl
    _keep("max21", _roll_max(rcc, 21, 10))
    _keep("min21", _roll_min(rcc, 21, 10))
    m1 = _roll_mean(rcc, 60, 30)
    m2 = _roll_mean(rcc**2, 60, 30)
    m3 = _roll_mean(rcc**3, 60, 30)
    sd = np.sqrt(np.maximum(m2 - m1**2, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        _keep("skew60", (m3 - 3 * m1 * m2 + 2 * m1**3) / sd**3)
    del m1, m2, m3, sd
    neg = np.where(rcc < 0, rcc, 0.0)
    neg = np.where(pres, neg, np.nan)
    _keep("dvol60", np.sqrt(_roll_mean(neg**2, 60, 30)))
    del neg
    xm = _roll_mean(rcc * mret_col, 60, 30) - _roll_mean(rcc, 60, 30) * _roll_mean(mret_col, 60, 30)
    vm = _roll_mean(mret_col**2, 60, 30) - _roll_mean(mret_col, 60, 30) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = xm / np.where(vm > 1e-12, vm, np.nan)
    _keep("beta60", beta)
    resid_var = np.maximum(vol60**2 - beta**2 * vm, 0)
    _keep("ivol60", np.sqrt(resid_var))
    del xm, vm, beta, resid_var, vol20, vol60, mret_col
    _keep("on20", _roll_mean(ron, 20, 10))
    _keep("id20", _roll_mean(rid, 20, 10))
    _keep("on60", _roll_mean(ron, 60, 30))
    _keep("id60", _roll_mean(rid, 60, 30))
    del rid
    _keep("gap1", ron)
    _keep("absgap20", _roll_mean(np.abs(ron), 20, 10))
    with np.errstate(divide="ignore", invalid="ignore"):
        _keep("size", np.log(mcap))
    tv20 = _roll_mean(tv, 20, 10)
    with np.errstate(divide="ignore", invalid="ignore"):
        _keep("lnadtv", np.log(np.where(tv20 > 0, tv20, np.nan)))
        _keep("turn20", tv20 / mcap)
        _keep("vol_surge5_60", _roll_mean(tv, 5, 3) / _roll_mean(tv, 60, 30))
    with np.errstate(divide="ignore", invalid="ignore"):
        _keep("tv_z1", tv / tv20)
    with np.errstate(divide="ignore", invalid="ignore"):
        _keep("amihud20", _roll_mean(np.where(tv > 0, np.abs(rcc) / (tv / 1e9), np.nan), 20, 10))
    del tv20
    upv = np.where(rcc > 0, tv, 0.0)
    upv = np.where(pres, upv, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        _keep("upvol20", _roll_sum(upv, 20) / np.maximum(_roll_sum(tv, 20), 1.0))
    del upv, tv, rcc

    for nm, key in (("for", "flow_for_krw"), ("ins", "flow_ins_krw"), ("ind", "flow_ind_krw")):
        fl = np.where(pres, _f(key), np.nan)
        for wnd in (5, 20, 60):
            with np.errstate(divide="ignore", invalid="ignore"):
                val = _roll_sum(fl, wnd) / mcap
            cnt = _roll_cnt(fl, wnd)
            val[cnt < wnd * 0.6] = np.nan
            _keep(f"fl_{nm}{wnd}", val)
            del cnt
        del fl, val

    fresh = np.isfinite(_f("f_age_q")) & (_f("f_age_q") <= 2)
    eq = _f("f_equity")
    ni_ttm = _f("f_net_income_ttm")
    assets = _f("f_assets")
    opq = _f("f_operating_profit_q")
    opq_ly = _f("f_operating_profit_q_ly")
    niq = _f("f_net_income_q")
    niq_ly = _f("f_net_income_q_ly")
    with np.errstate(divide="ignore", invalid="ignore"):
        _keep("bm", np.where(fresh & (eq > 0), eq / mcap, np.nan))
        _keep("ep", np.where(fresh, ni_ttm / mcap, np.nan))
        _keep("roe", np.where(fresh & (eq > 0), ni_ttm / eq, np.nan))
        _keep("opa", np.where(fresh & (assets > 0), _f("f_operating_profit_ttm") / assets, np.nan))
        _keep("gpa", np.where(fresh & (assets > 0), _f("f_gross_profit_ttm") / assets, np.nan))
        _keep("sue_op", np.where(fresh, (opq - opq_ly) / mcap, np.nan))
        _keep("sue_ni", np.where(fresh, (niq - niq_ly) / mcap, np.nan))
        _keep(
            "asset_g",
            np.where(fresh & (_f("f_assets_ly") > 0), assets / _f("f_assets_ly") - 1.0, np.nan),
        )
        _keep(
            "accrual",
            np.where(
                fresh & (assets > 0), (ni_ttm - _f("f_operating_cash_flow_ttm")) / assets, np.nan
            ),
        )
        _keep(
            "sales_g",
            np.where(
                fresh & (_f("f_sales_ttm_ly") > 0), _f("f_sales_ttm") / _f("f_sales_ttm_ly") - 1.0, np.nan
            ),
        )
    del fresh, eq, ni_ttm, assets, opq, opq_ly, niq, niq_ly

    sess_qk = (
        np.asarray(
            [day.year * 4 + (day.month - 1) // 3 for day in list(cube.sessions)[:n_out]],
            dtype=np.float64,
        )[:, None]
        * np.ones((1, n_inst))
    )
    eqk = _f("earn_qk")
    e_fresh = np.isfinite(eqk) & ((sess_qk - eqk) <= 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        _keep(
            "sue_op_e",
            np.where(
                e_fresh, (_f("earn_operating_profit_q") - _f("earn_operating_profit_q_ly")) / mcap, np.nan
            ),
        )
        _keep(
            "sue_ni_e",
            np.where(
                e_fresh, (_f("earn_net_income_q") - _f("earn_net_income_q_ly")) / mcap, np.nan
            ),
        )
        _keep(
            "sue_sales_e",
            np.where(
                e_fresh,
                (_f("earn_sales_q") - _f("earn_sales_q_ly")) / np.abs(_f("earn_sales_q_ly")),
                np.nan,
            ),
        )
    del sess_qk, eqk, e_fresh, mcap
    ea = _f("earn_avail_t")
    idx = np.arange(n_out, dtype=np.float64)[:, None] * np.ones((1, n_inst))
    age = np.where(np.isfinite(ea), idx - ea, np.nan)
    _keep("earn_age", age)
    ok = np.isfinite(ea) & (age >= 0) & (age <= 60)
    ea_i = np.where(ok, ea, 0).astype(int)
    cols = np.arange(n_inst)[None, :] * np.ones((n_out, 1), dtype=int)
    rows_lo = np.clip(ea_i - 1, 0, n_out - 1)
    rows_hi = np.clip(np.minimum(ea_i + 1, idx.astype(int)), 0, n_out - 1)
    t_b = tr[rows_lo, cols]
    t_c = tr[rows_hi, cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        _keep(
            "ear",
            np.where(ok & np.isfinite(t_b) & np.isfinite(t_c) & (t_b > 0), t_c / t_b - 1.0, np.nan),
        )
        _keep("post_ear", np.where(ok & np.isfinite(t_b) & (t_b > 0), tr / t_b - 1.0, np.nan))
    del ea, idx, age, ok, ea_i, cols, rows_lo, rows_hi, t_b, t_c

    tr_ff = pd.DataFrame(tr).ffill().to_numpy(dtype=np.float64)
    del tr
    with np.errstate(invalid="ignore"):
        uo = _shift(tr_ff, 1) * (1.0 + np.where(np.isfinite(ron), ron, 0.0))
    del tr_ff, ron
    labels32: dict[int, NDArray[np.float32]] = {}
    for h in HORIZONS:
        num = _shift(uo, -(1 + h))
        den = _shift(uo, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(np.isfinite(num) & np.isfinite(den) & (den > 0), num / den - 1.0, np.nan)
        labels32[h] = np.ascontiguousarray(y, dtype=np.float32)
    del uo, num, den, y

    features32: dict[str, NDArray[np.float32]] = {name: out[name] for name in FEATURE_NAMES}
    for arr in list(features32.values()) + list(labels32.values()):
        arr.flags.writeable = False

    _LOG.info("[DATA] panel rows=%d features=%d horizons=%s", n_out, len(FEATURE_NAMES), tuple(HORIZONS))
    return FeaturePanel(features=features32, labels=labels32, last_row=last_row)
