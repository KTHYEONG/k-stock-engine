# mypy: ignore-errors
"""Vectorized economic eligibility gates."""
from __future__ import annotations

from typing import Literal

import numpy as np


def economic_gate_masks(*, expected_active_alpha: np.ndarray, expected_net_alpha: np.ndarray, alpha_lower_bound: np.ndarray, net_alpha_lower_bound: np.ndarray, alpha_standard_error: np.ndarray, exit_cost_rate: np.ndarray, incumbent_mask: np.ndarray, no_trade_band_bps: float, economic_gate_mode: Literal['lower_bound_v1', 'finite_mean_v1']) -> tuple[np.ndarray, np.ndarray]:
    n = expected_active_alpha.size
    e_active = np.asarray(expected_active_alpha, dtype=float)
    e_net = np.asarray(expected_net_alpha, dtype=float)
    lb = np.asarray(alpha_lower_bound, dtype=float)
    net_lb = np.asarray(net_alpha_lower_bound, dtype=float)
    se = np.asarray(alpha_standard_error, dtype=float)
    exit_c = np.asarray(exit_cost_rate, dtype=float)
    inc = np.asarray(incumbent_mask, dtype=bool)
    band = float(no_trade_band_bps) / 10000.0
    keep = np.zeros(n, dtype=bool)
    enter = np.zeros(n, dtype=bool)
    if economic_gate_mode == "finite_mean_v1":
        finite_pos_net = np.isfinite(e_net) & (e_net > 0.0)
        finite_se = np.isfinite(se) & (se >= 0.0)
        finite_active_pos = np.isfinite(e_active) & (e_active > 0.0)
        base_ok = finite_pos_net & finite_se & finite_active_pos
        keep = inc & base_ok
        enter = (~inc) & base_ok
    elif economic_gate_mode == "lower_bound_v1":
        keep_ok = (e_active - exit_c > 0.0) & (e_active > 0.0) & (lb > 0.0) & np.isfinite(e_active) & np.isfinite(lb) & np.isfinite(exit_c)
        enter_ok = (e_net > 0.0) & (e_active > 0.0) & (lb > 0.0) & ((net_lb - band) > 0.0) & np.isfinite(e_net) & np.isfinite(e_active) & np.isfinite(lb) & np.isfinite(net_lb)
        keep = inc & keep_ok
        enter = (~inc) & enter_ok
    else:
        raise ValueError(f"unknown economic_gate_mode {economic_gate_mode!r}")
    keep = keep & np.isfinite(e_active) & np.isfinite(e_net)
    enter = enter & np.isfinite(e_active) & np.isfinite(e_net)
    return keep, enter
