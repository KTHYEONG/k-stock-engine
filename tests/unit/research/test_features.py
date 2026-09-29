"""Feature registry causality and contract invariants."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.research.cube import ResearchCube
from src.research.features import FEATURES, compute_features

INSTS = ("KRX:000001", "KRX:000002")


def _sessions(n: int = 300) -> tuple[date, ...]:
    start = date(2020, 1, 2)
    return tuple(start + timedelta(days=i) for i in range(n))


def _synthetic_cube(n_s: int = 300, n_n: int = 2) -> ResearchCube:
    rng = np.random.default_rng(7)
    sessions = _sessions(n_s)
    shape = (n_s, n_n)
    close = 5000 + np.cumsum(rng.standard_normal(shape) * 20, axis=0)
    close = np.maximum(close, 1000.0)
    base = np.roll(close, 1, axis=0)
    base[0, :] = close[0, :]
    open_px = base * (1 + rng.standard_normal(shape) * 0.005)
    ret_cc = close / base - 1.0
    adj_px = np.cumprod(1 + ret_cc, axis=0)
    adj_px[0, :] = 1.0
    adj_tr = adj_px.copy()
    arrays: dict[str, np.ndarray] = {
        "adj_tr": adj_tr,
        "adj_px": adj_px,
        "ret_cc": ret_cc,
        "ret_vol60": np.full(shape, 0.02),
        "trading_value": np.full(shape, 1e8),
        "market_cap": np.full(shape, 1e11),
        "close": close,
        "open": open_px,
        "base_price": base,
        "present": np.ones(shape, dtype=bool),
        "f_age_q": np.full(shape, 1.0),
        "f_avail_t": np.zeros(shape),
        "f_qk": np.full(shape, 2020 * 4 + 1.0),
        "f_equity": np.full(shape, 1e10),
        "f_assets": np.full(shape, 2e10),
        "f_assets_ly": np.full(shape, 1.9e10),
        "f_debt": np.full(shape, 5e9),
        "f_cash": np.full(shape, 1e9),
        "f_sales_ttm": np.full(shape, 1e10),
        "f_sales_ttm_ly": np.full(shape, 9e9),
        "f_gross_profit_ttm": np.full(shape, 4e9),
        "f_operating_profit_ttm": np.full(shape, 2e9),
        "f_operating_profit_ttm_ly": np.full(shape, 1.8e9),
        "f_net_income_ttm": np.full(shape, 1e9),
        "f_net_income_ttm_ly": np.full(shape, 9e8),
        "f_operating_cash_flow_ttm": np.full(shape, 1.2e9),
        "f_sales_q": np.full(shape, 2e9),
        "f_sales_q_ly": np.full(shape, 1.8e9),
        "f_operating_profit_q": np.full(shape, 5e8),
        "f_operating_profit_q_ly": np.full(shape, 4e8),
        "f_net_income_q": np.full(shape, 3e8),
        "f_net_income_q_ly": np.full(shape, 2.5e8),
        "dps_ttm": np.full(shape, 100.0),
        "earn_qk": np.full(shape, 2020 * 4 + 1.0),
        "earn_avail_t": np.full(shape, 5.0),
        "earn_sales_q": np.full(shape, 2e9),
        "earn_sales_q_ly": np.full(shape, 1.8e9),
        "earn_operating_profit_q": np.full(shape, 5e8),
        "earn_operating_profit_q_ly": np.full(shape, 4e8),
        "earn_net_income_q": np.full(shape, 3e8),
        "earn_net_income_q_ly": np.full(shape, 2.5e8),
        "flow_for_krw": np.full(shape, 1e7),
        "flow_ins_krw": np.full(shape, 2e7),
        "flow_ind_krw": np.full(shape, -1e7),
    }
    return ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=sessions,
        instrument_ids=list(INSTS[:n_n]),
        arrays=arrays,
        exit_at=np.full(n_n, -1, dtype=np.int64),
        exit_halted=np.zeros(n_n, dtype=np.bool_),
    )


def _equal_with_nan(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(
        np.array_equal(np.isnan(first), np.isnan(second))
        and np.array_equal(first[~np.isnan(first)], second[~np.isnan(second)])
    )


def test_causal_prefix_invariance() -> None:
    """Rows at or before the cut never depend on later rows."""
    cube = _synthetic_cube()
    cut = 200
    noisy: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(1234)
    for name, arr in cube.arrays.items():
        if arr.dtype == np.bool_:
            noisy[name] = arr.copy()
            continue
        perturbed = np.asarray(arr, dtype=np.float64).copy()
        perturbed[cut + 1 :, :] = rng.standard_normal((perturbed.shape[0] - cut - 1, perturbed.shape[1])) * 999
        noisy[name] = perturbed
    altered = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays=noisy,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    names = sorted(FEATURES)
    base = compute_features(cube, names)
    other = compute_features(altered, names)
    for name in names:
        assert _equal_with_nan(base[name][: cut + 1, :], other[name][: cut + 1, :])


def test_momentum_skips_last_month() -> None:
    """A rally confined to the last 21 rows leaves 12-1 momentum unchanged."""
    cube = _synthetic_cube(n_s=300, n_n=1)
    tr = np.ones((300, 1))
    tr[21:279] = 1.0 + np.arange(21, 279)[:, None] * 0.001
    tr[279:] = tr[278] * 1.05
    px = tr.copy()
    arrays = dict(cube.arrays)
    arrays["adj_tr"] = tr
    arrays["adj_px"] = px
    rallied = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays=arrays,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    out = compute_features(rallied, ["mom_12_1"])
    assert out["mom_12_1"][299, 0] == pytest.approx(tr[278, 0] / tr[47, 0] - 1.0)


def test_stale_fundamentals_suppressed() -> None:
    """Stale quarters suppress value/quality while price features survive."""
    cube = _synthetic_cube(n_s=300, n_n=1)
    arrays = dict(cube.arrays)
    arrays["f_age_q"] = np.full((300, 1), 3.0)
    stale = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays=arrays,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    out = compute_features(stale, ["bm", "gpa", "mom_12_1"])
    assert bool(np.isnan(out["bm"][299, 0]))
    assert bool(np.isnan(out["gpa"][299, 0]))
    assert bool(np.isfinite(out["mom_12_1"][299, 0]))


def test_early_sue_uses_release_before_filing() -> None:
    """Early SUE is defined from the release while periodic SUE waits for the filing."""
    cube = _synthetic_cube(n_s=60, n_n=1)
    arrays = dict(cube.arrays)
    arrays["f_operating_profit_q"] = np.full((60, 1), np.nan)
    arrays["f_operating_profit_q_ly"] = np.full((60, 1), np.nan)
    arrays["f_operating_profit_q"][50:, :] = 5e8
    arrays["f_operating_profit_q_ly"][50:, :] = 4e8
    arrays["f_age_q"] = np.full((60, 1), 5.0)
    arrays["f_age_q"][50:, :] = 1.0
    sess_qk = np.array([day.year * 4 + (day.month - 1) // 3 for day in cube.sessions], dtype=np.float64)
    arrays["earn_qk"] = np.full((60, 1), sess_qk[10])
    arrays["earn_avail_t"] = np.full((60, 1), 10.0)
    early = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays=arrays,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    out = compute_features(early, ["sue_op_early", "sue_op"])
    assert bool(np.isfinite(out["sue_op_early"][20, 0]))
    assert bool(np.isnan(out["sue_op"][20, 0]))
    assert bool(np.isfinite(out["sue_op"][55, 0]))


def test_negative_equity_excluded() -> None:
    """Negative book equity suppresses book-to-market and ROE."""
    cube = _synthetic_cube(n_s=300, n_n=1)
    arrays = dict(cube.arrays)
    arrays["f_equity"] = np.full((300, 1), -1.0)
    stressed = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays=arrays,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    out = compute_features(stressed, ["bm", "roe"])
    assert bool(np.isnan(out["bm"][299, 0]))
    assert bool(np.isnan(out["roe"][299, 0]))


def test_unknown_feature_rejected() -> None:
    """Unknown registry names raise KeyError."""
    cube = _synthetic_cube(n_s=10, n_n=1)
    with pytest.raises(KeyError):
        compute_features(cube, ["nope"])


def test_registry_metadata_complete() -> None:
    """Every registry entry carries a valid family and rationale."""
    allowed = {"price", "risk", "value", "quality", "earnings", "flow", "size"}
    assert len(FEATURES) >= 30
    for item in FEATURES.values():
        assert item.family in allowed
        assert item.description.strip() != ""


def test_smart_flow_standalone_request() -> None:
    """Requesting only smart-money flows materializes their legs internally."""
    cube = _synthetic_cube(n_s=70, n_n=1)
    out = compute_features(cube, ["flow_smart_5"])
    assert bool(np.isfinite(out["flow_smart_5"][69, 0]))
