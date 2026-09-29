"""Cross-sectional feature registry computed causally from a research cube."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.research.cube import ResearchCube

__all__ = ["FEATURES", "FeatureDef", "compute_features"]

_ALLOWED_FAMILIES = frozenset({"price", "risk", "value", "quality", "earnings", "flow", "size"})


@dataclass(frozen=True, slots=True)
class FeatureDef:
    """One cross-sectional feature; higher values mean 'more attractive' by construction.

    Attributes:
        name: Registry key.
        family: One of ``price``, ``risk``, ``value``, ``quality``, ``earnings``, ``flow``, ``size``.
        lookback_sessions: Rows needed before the first defined value.
        description: Formula and economic rationale.
    """

    name: str
    family: str
    lookback_sessions: int
    description: str


def _def(name: str, family: str, lookback: int, description: str) -> FeatureDef:
    return FeatureDef(name=name, family=family, lookback_sessions=lookback, description=description)


FEATURES: Mapping[str, FeatureDef] = {
    "mom_12_1": _def("mom_12_1", "price", 252, "tr[t-21]/tr[t-252]-1 on total-return index; skips the last month."),
    "mom_6_1": _def("mom_6_1", "price", 126, "tr[t-21]/tr[t-126]-1; medium-horizon momentum without the last month."),
    "mom_3_1": _def("mom_3_1", "price", 63, "tr[t-21]/tr[t-63]-1; short-horizon momentum without the last month."),
    "rev_1m": _def("rev_1m", "price", 21, "-(tr[t]/tr[t-21]-1); one-month reversal, higher means stronger reversal."),
    "rev_1w": _def("rev_1w", "price", 5, "-(tr[t]/tr[t-5]-1); one-week reversal."),
    "hi52": _def("hi52", "price", 251, "px[t]/max(px[t-251..t]); proximity to the 52-week high."),
    "lowvol": _def("lowvol", "risk", 60, "-ret_vol60; low-volatility exclusion filter."),
    "max21": _def("max21", "risk", 20, "-max(ret_cc[t-20..t]); lottery-demand exclusion filter."),
    "turn": _def("turn", "risk", 19, "mean(trading_value[t-19..t])/mcap; attention filter, higher means hotter."),
    "lowturn": _def("lowturn", "risk", 19, "-turn; illiquidity exclusion filter."),
    "size": _def("size", "size", 0, "-ln(mcap); higher means smaller capitalization."),
    "bm": _def("bm", "value", 0, "equity/mcap when fresh and equity>0; book-to-market."),
    "ep": _def("ep", "value", 0, "ni_ttm/mcap when fresh; earnings yield."),
    "sp": _def("sp", "value", 0, "sales_ttm/mcap when fresh; sales yield."),
    "cfp": _def("cfp", "value", 0, "ocf_ttm/mcap when fresh; cash-flow yield."),
    "op_ev": _def("op_ev", "value", 0, "op_ttm/(mcap+debt-cash) when fresh and EV>0."),
    "dy": _def("dy", "value", 0, "dps_ttm/close when fresh and close>0; trailing dividend yield."),
    "gpa": _def("gpa", "quality", 0, "gp_ttm/assets when fresh and assets>0; gross profitability."),
    "opa": _def("opa", "quality", 0, "op_ttm/assets when fresh and assets>0; operating profitability."),
    "roe": _def("roe", "quality", 0, "ni_ttm/equity when fresh and equity>0; return on equity."),
    "accrual": _def("accrual", "quality", 0, "-(ni_ttm-ocf_ttm)/assets when fresh; conservative accruals score."),
    "asset_growth": _def("asset_growth", "quality", 0, "-(assets/assets_ly-1) when fresh; shrinking assets score."),
    "sue_op": _def("sue_op", "earnings", 0, "(f_op_q-f_op_q_ly)/mcap when fresh; periodic standardized surprise."),
    "sue_ni": _def("sue_ni", "earnings", 0, "(f_ni_q-f_ni_q_ly)/mcap when fresh."),
    "op_growth": _def("op_growth", "earnings", 0, "(op_ttm-op_ttm_ly)/mcap when fresh."),
    "sales_growth": _def("sales_growth", "earnings", 0, "sales_ttm/sales_ttm_ly-1 when fresh."),
    "sue_op_early": _def(
        "sue_op_early", "earnings", 0, "(earn_op_q-earn_op_q_ly)/mcap when early-fresh; earliest surprise."
    ),
    "sue_ni_early": _def("sue_ni_early", "earnings", 0, "(earn_ni_q-earn_ni_q_ly)/mcap when early-fresh."),
    "sue_sales_early": _def("sue_sales_early", "earnings", 0, "(earn_sales_q-earn_sales_q_ly)/mcap when early-fresh."),
    "earn_age": _def("earn_age", "earnings", 0, "t-earn_avail_t; sessions since the newest quarter appeared."),
    "flow_for_5": _def("flow_for_5", "flow", 4, "sum(flow_for_krw[t-4..t])/mcap; foreign pressure 5d."),
    "flow_for_20": _def("flow_for_20", "flow", 19, "sum(flow_for_krw[t-19..t])/mcap; foreign pressure 20d."),
    "flow_for_60": _def("flow_for_60", "flow", 59, "sum(flow_for_krw[t-59..t])/mcap; foreign pressure 60d."),
    "flow_ins_5": _def("flow_ins_5", "flow", 4, "sum(flow_ins_krw[t-4..t])/mcap; institutional pressure 5d."),
    "flow_ins_20": _def("flow_ins_20", "flow", 19, "sum(flow_ins_krw[t-19..t])/mcap; institutional pressure 20d."),
    "flow_ins_60": _def("flow_ins_60", "flow", 59, "sum(flow_ins_krw[t-59..t])/mcap; institutional pressure 60d."),
    "flow_ind_5": _def("flow_ind_5", "flow", 4, "sum(flow_ind_krw[t-4..t])/mcap; individual pressure 5d."),
    "flow_ind_20": _def("flow_ind_20", "flow", 19, "sum(flow_ind_krw[t-19..t])/mcap; individual pressure 20d."),
    "flow_ind_60": _def("flow_ind_60", "flow", 59, "sum(flow_ind_krw[t-59..t])/mcap; individual pressure 60d."),
    "flow_smart_5": _def("flow_smart_5", "flow", 4, "flow_for_5+flow_ins_5; smart-money pressure 5d."),
    "flow_smart_20": _def("flow_smart_20", "flow", 19, "flow_for_20+flow_ins_20; smart-money pressure 20d."),
    "flow_smart_60": _def("flow_smart_60", "flow", 59, "flow_for_60+flow_ins_60; smart-money pressure 60d."),
}


def _get(cube: ResearchCube, name: str) -> NDArray[np.float64]:
    arr = cube.arrays[name]
    return np.asarray(arr, dtype=np.float64)


def _present(cube: ResearchCube) -> NDArray[np.bool_]:
    return np.asarray(cube.arrays["present"], dtype=bool)


def _present_count(present: NDArray[np.bool_], window: int) -> NDArray[np.float64]:
    kernel = np.ones(window, dtype=np.float64)
    counts = np.apply_along_axis(
        lambda col: np.convolve(col.astype(np.float64), kernel, mode="full")[: col.size], 0, present
    )
    return np.asarray(counts, dtype=np.float64)


def _rolling_mean_skip_absent(values: NDArray[np.float64], present: NDArray[np.bool_], window: int) -> NDArray[np.float64]:
    masked = np.where(present & np.isfinite(values), values, 0.0)
    counts = _present_count(present & np.isfinite(values), window)
    sums = np.apply_along_axis(
        lambda col: np.convolve(col, np.ones(window, dtype=np.float64), mode="full")[: col.size],
        0,
        masked,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / np.maximum(counts, 1.0)
    half = window / 2.0
    full_counts = _present_count(present, window)
    return np.where((full_counts >= half) & (counts > 0), mean, np.nan)


def _rolling_max_skip_absent(values: NDArray[np.float64], present: NDArray[np.bool_], window: int) -> NDArray[np.float64]:
    n_s, n_n = values.shape
    out = np.full((n_s, n_n), np.nan, dtype=np.float64)
    full_counts = _present_count(present, window)
    for t in range(n_s):
        lo = max(0, t - window + 1)
        block = values[lo : t + 1, :]
        mask = present[lo : t + 1, :] & np.isfinite(block)
        with np.errstate(invalid="ignore"):
            best = np.where(mask, block, -np.inf).max(axis=0)
        out[t, :] = np.where(full_counts[t, :] >= window / 2.0, np.where(best == -np.inf, np.nan, best), np.nan)
    return out


def _rolling_sum_strict(values: NDArray[np.float64], present: NDArray[np.bool_], window: int) -> NDArray[np.float64]:
    n_s, n_n = values.shape
    out = np.full((n_s, n_n), np.nan, dtype=np.float64)
    full_counts = _present_count(present, window)
    for t in range(n_s):
        lo = max(0, t - window + 1)
        block = values[lo : t + 1, :]
        mask = present[lo : t + 1, :]
        finite = np.isfinite(block)
        ok = (full_counts[t, :] >= window / 2.0) & np.all(np.where(mask, finite, True), axis=0)
        sums = np.where(np.where(mask, finite, False), np.where(finite, block, 0.0), 0.0).sum(axis=0)
        out[t, :] = np.where(ok, sums, np.nan)
    return out


def compute_features(cube: ResearchCube, names: Sequence[str]) -> dict[str, NDArray[np.float64]]:
    """Compute registry features as S x N float64 arrays (NaN = undefined).

    Row ``t`` depends only on cube rows ``<= t``; this is the causal contract every screening,
    gate and simulator step relies on.

    Raises:
        KeyError: an unknown feature name.
    """
    requested = list(names)
    for name in requested:
        if name not in FEATURES:
            raise KeyError(name)
    shape = (len(cube.sessions), len(cube.instrument_ids))
    present = _present(cube)
    tr = _get(cube, "adj_tr")
    px = _get(cube, "adj_px")
    ret_cc = _get(cube, "ret_cc")
    ret_vol60 = _get(cube, "ret_vol60")
    trading_value = _get(cube, "trading_value")
    market_cap = _get(cube, "market_cap")
    close = _get(cube, "close")
    mcap = np.where(np.isfinite(market_cap) & (market_cap > 0), market_cap, np.nan)
    fresh = np.isfinite(_get(cube, "f_age_q")) & (_get(cube, "f_age_q") <= 2)
    sess_qk = np.asarray(
        [day.year * 4 + (day.month - 1) // 3 for day in cube.sessions], dtype=np.float64
    )[:, None]
    earn_qk = _get(cube, "earn_qk")
    earn_fresh = np.isfinite(earn_qk) & ((sess_qk - earn_qk) <= 2)
    out: dict[str, NDArray[np.float64]] = {}

    def _final(key: str, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        lookback = FEATURES[key].lookback_sessions
        result = np.asarray(arr, dtype=np.float64)
        if lookback > 0:
            result[:lookback, :] = np.nan
        return np.where(present, result, np.nan)

    for name in requested:
        if name == "mom_12_1":
            with np.errstate(divide="ignore", invalid="ignore"):
                num = np.roll(tr, 21, axis=0)
                den = np.roll(tr, 252, axis=0)
                arr = num / den - 1.0
                arr[:252, :] = np.nan
                arr = np.where(np.isfinite(num) & np.isfinite(den) & (den > 0), arr, np.nan)
            out[name] = _final(name, arr)
        elif name == "mom_6_1":
            with np.errstate(divide="ignore", invalid="ignore"):
                num = np.roll(tr, 21, axis=0)
                den = np.roll(tr, 126, axis=0)
                arr = num / den - 1.0
                arr[:126, :] = np.nan
                arr = np.where(np.isfinite(num) & np.isfinite(den) & (den > 0), arr, np.nan)
            out[name] = _final(name, arr)
        elif name == "mom_3_1":
            with np.errstate(divide="ignore", invalid="ignore"):
                num = np.roll(tr, 21, axis=0)
                den = np.roll(tr, 63, axis=0)
                arr = num / den - 1.0
                arr[:63, :] = np.nan
                arr = np.where(np.isfinite(num) & np.isfinite(den) & (den > 0), arr, np.nan)
            out[name] = _final(name, arr)
        elif name == "rev_1m":
            with np.errstate(divide="ignore", invalid="ignore"):
                den = np.roll(tr, 21, axis=0)
                arr = -(tr / den - 1.0)
                arr[:21, :] = np.nan
                arr = np.where(np.isfinite(tr) & np.isfinite(den) & (den > 0), arr, np.nan)
            out[name] = _final(name, arr)
        elif name == "rev_1w":
            with np.errstate(divide="ignore", invalid="ignore"):
                den = np.roll(tr, 5, axis=0)
                arr = -(tr / den - 1.0)
                arr[:5, :] = np.nan
                arr = np.where(np.isfinite(tr) & np.isfinite(den) & (den > 0), arr, np.nan)
            out[name] = _final(name, arr)
        elif name == "hi52":
            peak = _rolling_max_skip_absent(px, present, 252)
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = px / peak
                arr = np.where(np.isfinite(px) & np.isfinite(peak) & (peak > 0), arr, np.nan)
            out[name] = _final(name, arr)
        elif name == "lowvol":
            out[name] = _final(name, -ret_vol60)
        elif name == "max21":
            peak_ret = _rolling_max_skip_absent(ret_cc, present, 21)
            out[name] = _final(name, -peak_ret)
        elif name in ("turn", "lowturn"):
            mean_tv = _rolling_mean_skip_absent(trading_value, present, 20)
            with np.errstate(divide="ignore", invalid="ignore"):
                turn = mean_tv / mcap
            if name == "lowturn":
                turn = -turn
            out[name] = _final(name, turn)
        elif name == "size":
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = -np.log(mcap)
            out[name] = _final(name, arr)
        elif name == "bm":
            equity = _get(cube, "f_equity")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(equity) & (equity > 0), equity / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "ep":
            ni_ttm = _get(cube, "f_net_income_ttm")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(ni_ttm), ni_ttm / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "sp":
            sales_ttm = _get(cube, "f_sales_ttm")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(sales_ttm), sales_ttm / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "cfp":
            ocf_ttm = _get(cube, "f_operating_cash_flow_ttm")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(ocf_ttm), ocf_ttm / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "op_ev":
            op_ttm = _get(cube, "f_operating_profit_ttm")
            debt = _get(cube, "f_debt")
            cash = _get(cube, "f_cash")
            with np.errstate(divide="ignore", invalid="ignore"):
                ev = mcap + np.where(np.isfinite(debt), debt, np.nan) - np.where(np.isfinite(cash), cash, np.nan)
                arr = np.where(fresh & np.isfinite(op_ttm) & np.isfinite(ev) & (ev > 0), op_ttm / ev, np.nan)
            out[name] = _final(name, arr)
        elif name == "dy":
            dps = _get(cube, "dps_ttm")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(dps) & np.isfinite(close) & (close > 0), dps / close, np.nan)
            out[name] = _final(name, arr)
        elif name == "gpa":
            gp_ttm = _get(cube, "f_gross_profit_ttm")
            assets = _get(cube, "f_assets")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(gp_ttm) & np.isfinite(assets) & (assets > 0), gp_ttm / assets, np.nan)
            out[name] = _final(name, arr)
        elif name == "opa":
            op_ttm = _get(cube, "f_operating_profit_ttm")
            assets = _get(cube, "f_assets")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(op_ttm) & np.isfinite(assets) & (assets > 0), op_ttm / assets, np.nan)
            out[name] = _final(name, arr)
        elif name == "roe":
            ni_ttm = _get(cube, "f_net_income_ttm")
            equity = _get(cube, "f_equity")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(ni_ttm) & np.isfinite(equity) & (equity > 0), ni_ttm / equity, np.nan)
            out[name] = _final(name, arr)
        elif name == "accrual":
            ni_ttm = _get(cube, "f_net_income_ttm")
            ocf_ttm = _get(cube, "f_operating_cash_flow_ttm")
            assets = _get(cube, "f_assets")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(
                    fresh & np.isfinite(ni_ttm) & np.isfinite(ocf_ttm) & np.isfinite(assets) & (assets > 0),
                    -((ni_ttm - ocf_ttm) / assets),
                    np.nan,
                )
            out[name] = _final(name, arr)
        elif name == "asset_growth":
            assets = _get(cube, "f_assets")
            assets_ly = _get(cube, "f_assets_ly")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(
                    fresh & np.isfinite(assets) & np.isfinite(assets_ly) & (assets_ly > 0),
                    -((assets / assets_ly) - 1.0),
                    np.nan,
                )
            out[name] = _final(name, arr)
        elif name == "sue_op":
            op_q = _get(cube, "f_operating_profit_q")
            op_ly = _get(cube, "f_operating_profit_q_ly")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(op_q) & np.isfinite(op_ly), (op_q - op_ly) / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "sue_ni":
            ni_q = _get(cube, "f_net_income_q")
            ni_ly = _get(cube, "f_net_income_q_ly")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(ni_q) & np.isfinite(ni_ly), (ni_q - ni_ly) / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "op_growth":
            op_ttm = _get(cube, "f_operating_profit_ttm")
            op_ttm_ly = _get(cube, "f_operating_profit_ttm_ly")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(fresh & np.isfinite(op_ttm) & np.isfinite(op_ttm_ly), (op_ttm - op_ttm_ly) / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "sales_growth":
            sales_ttm = _get(cube, "f_sales_ttm")
            sales_ly = _get(cube, "f_sales_ttm_ly")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(
                    fresh & np.isfinite(sales_ttm) & np.isfinite(sales_ly) & (sales_ly != 0),
                    sales_ttm / sales_ly - 1.0,
                    np.nan,
                )
            out[name] = _final(name, arr)
        elif name == "sue_op_early":
            cur = _get(cube, "earn_operating_profit_q")
            prv = _get(cube, "earn_operating_profit_q_ly")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(earn_fresh & np.isfinite(cur) & np.isfinite(prv), (cur - prv) / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "sue_ni_early":
            cur = _get(cube, "earn_net_income_q")
            prv = _get(cube, "earn_net_income_q_ly")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(earn_fresh & np.isfinite(cur) & np.isfinite(prv), (cur - prv) / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "sue_sales_early":
            cur = _get(cube, "earn_sales_q")
            prv = _get(cube, "earn_sales_q_ly")
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.where(earn_fresh & np.isfinite(cur) & np.isfinite(prv), (cur - prv) / mcap, np.nan)
            out[name] = _final(name, arr)
        elif name == "earn_age":
            earn_avail = _get(cube, "earn_avail_t")
            idx = np.arange(shape[0], dtype=np.float64)[:, None] * np.ones((1, shape[1]))
            with np.errstate(invalid="ignore"):
                arr = np.where(np.isfinite(earn_avail), idx - earn_avail, np.nan)
            out[name] = _final(name, arr)
    flow_defs = {
        "flow_for_5": ("flow_for_krw", 5),
        "flow_for_20": ("flow_for_krw", 20),
        "flow_for_60": ("flow_for_krw", 60),
        "flow_ins_5": ("flow_ins_krw", 5),
        "flow_ins_20": ("flow_ins_krw", 20),
        "flow_ins_60": ("flow_ins_krw", 60),
        "flow_ind_5": ("flow_ind_krw", 5),
        "flow_ind_20": ("flow_ind_krw", 20),
        "flow_ind_60": ("flow_ind_krw", 60),
    }
    for name, (source, window) in flow_defs.items():
        if name in requested:
            flow = _get(cube, source)
            summed = _rolling_sum_strict(flow, present, window)
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = summed / mcap
            out[name] = _final(name, arr)
    for name, window in (("flow_smart_5", 5), ("flow_smart_20", 20), ("flow_smart_60", 60)):
        if name in requested:
            for_key, ins_key = f"flow_for_{window}", f"flow_ins_{window}"
            if for_key not in out:
                flow = _get(cube, "flow_for_krw")
                summed = _rolling_sum_strict(flow, present, window)
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[for_key] = _final(for_key, summed / mcap)
            if ins_key not in out:
                flow = _get(cube, "flow_ins_krw")
                summed = _rolling_sum_strict(flow, present, window)
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[ins_key] = _final(ins_key, summed / mcap)
            base_for, base_ins = out[for_key], out[ins_key]
            with np.errstate(invalid="ignore"):
                arr = np.where(np.isfinite(base_for) & np.isfinite(base_ins), base_for + base_ins, np.nan)
            out[name] = _final(name, arr)
    for name in requested:
        arr = np.ascontiguousarray(np.asarray(out[name], dtype=np.float64))
        out[name] = arr
    return {name: out[name] for name in requested}
