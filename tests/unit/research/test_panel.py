"""Panel causality, label and contract invariants."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.research.cube import ResearchCube
from src.research.panel import FEATURE_NAMES, HORIZONS, FeaturePanel, build_panel

INSTS = ("KRX:000001", "KRX:000002")


def _sessions(n: int) -> tuple[date, ...]:
    start = date(2020, 1, 2)
    return tuple(start + timedelta(days=i) for i in range(n))


def _synthetic_cube(n_s: int = 300, n_n: int = 2, insts: list[str] | None = None) -> ResearchCube:
    rng = np.random.default_rng(7)
    shape = (n_s, n_n)
    close = 5000 + np.cumsum(rng.standard_normal(shape) * 20, axis=0)
    close = np.maximum(close, 1000.0)
    base = np.roll(close, 1, axis=0)
    base[0, :] = close[0, :]
    open_px = base * (1 + rng.standard_normal(shape) * 0.005)
    high = np.maximum(open_px, close) * 1.002
    low = np.minimum(open_px, close) * 0.998
    ret_cc = close / base - 1.0
    r_on = open_px / base - 1.0
    r_id = close / open_px - 1.0
    growth = 1.0 + ret_cc
    adj_px = np.cumprod(growth, axis=0)
    adj_px[0, :] = 1.0
    if n_s > 1:
        adj_px[1:, :] = np.cumprod(growth[1:, :], axis=0)
    adj_tr = adj_px.copy()
    arrays: dict[str, np.ndarray] = {
        "adj_tr": adj_tr,
        "adj_px": adj_px,
        "ret_cc": ret_cc,
        "r_on": r_on,
        "r_id": r_id,
        "high": high,
        "low": low,
        "close": close,
        "open": open_px,
        "trading_value": np.full(shape, 1e8),
        "market_cap": np.full(shape, 1e11),
        "present": np.ones(shape, dtype=bool),
        "f_age_q": np.full(shape, 1.0),
        "f_equity": np.full(shape, 1e10),
        "f_assets": np.full(shape, 2e10),
        "f_assets_ly": np.full(shape, 1.9e10),
        "f_net_income_ttm": np.full(shape, 1e9),
        "f_operating_profit_ttm": np.full(shape, 2e9),
        "f_gross_profit_ttm": np.full(shape, 4e9),
        "f_operating_cash_flow_ttm": np.full(shape, 1.2e9),
        "f_operating_profit_q": np.full(shape, 5e8),
        "f_operating_profit_q_ly": np.full(shape, 4e8),
        "f_net_income_q": np.full(shape, 3e8),
        "f_net_income_q_ly": np.full(shape, 2.5e8),
        "f_sales_ttm": np.full(shape, 1e10),
        "f_sales_ttm_ly": np.full(shape, 9e9),
        "earn_qk": np.full(shape, 2020 * 4 + 1.0),
        "earn_avail_t": np.full(shape, 5.0),
        "earn_operating_profit_q": np.full(shape, 5e8),
        "earn_operating_profit_q_ly": np.full(shape, 4e8),
        "earn_net_income_q": np.full(shape, 3e8),
        "earn_net_income_q_ly": np.full(shape, 2.5e8),
        "earn_sales_q": np.full(shape, 2e9),
        "earn_sales_q_ly": np.full(shape, 1.8e9),
        "flow_for_krw": np.full(shape, 1e7),
        "flow_ins_krw": np.full(shape, 2e7),
        "flow_ind_krw": np.full(shape, -1e7),
    }
    return ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=list(_sessions(n_s)),
        instrument_ids=list(insts) if insts is not None else list(INSTS[:n_n]),
        arrays=arrays,
        exit_at=np.full(n_n, -1, dtype=np.int64),
        exit_halted=np.zeros(n_n, dtype=np.bool_),
    )


def _equal_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))


def test_feature_names_are_frozen_contract() -> None:
    assert len(FEATURE_NAMES) == 62
    assert len(set(FEATURE_NAMES)) == 62
    assert HORIZONS == (5, 10, 21)
    panel = build_panel(_synthetic_cube(), last_row=299)
    assert set(panel.features) == set(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert panel.features[name].shape == (300, 2)
        assert panel.features[name].dtype == np.float32


def test_future_rows_cannot_change_past_features() -> None:
    cube = _synthetic_cube()
    cut = 200
    rng = np.random.default_rng(1234)
    noisy: dict[str, np.ndarray] = {}
    for name, arr in cube.arrays.items():
        if arr.dtype == np.bool_:
            noisy[name] = np.asarray(arr).copy()
            continue
        perturbed = np.asarray(arr, dtype=np.float64).copy()
        perturbed[cut + 1 :, :] = rng.standard_normal((perturbed.shape[0] - cut - 1, perturbed.shape[1]))
        noisy[name] = perturbed
    altered = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays=noisy,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    base = build_panel(cube, last_row=cut)
    other = build_panel(altered, last_row=cut)
    for name in FEATURE_NAMES:
        assert _equal_with_nan(base.features[name], other.features[name])


def test_last_row_slices_before_computing() -> None:
    cube = _synthetic_cube(n_s=100)
    with pytest.raises(ValueError, match="last_row"):
        build_panel(cube, last_row=100)
    with pytest.raises(ValueError, match="last_row"):
        build_panel(cube, last_row=-1)
    panel = build_panel(cube, last_row=49)
    assert panel.last_row == 49
    for arr in list(panel.features.values()) + list(panel.labels.values()):
        assert arr.shape == (50, 2)
    width = max(HORIZONS)
    for h, lab in panel.labels.items():
        assert bool(np.all(np.isnan(np.asarray(lab)[50 - h - 1 :, :])))
def test_label_is_open_to_open_from_next_open() -> None:
    cube = _synthetic_cube(n_s=30, n_n=2)
    arrays = {k: np.asarray(v, dtype=np.float64).copy() for k, v in cube.arrays.items()}
    tr = np.ones((30, 2))
    tr[:, 0] = np.cumprod(1 + np.full(30, 0.001))
    tr[:, 1] = np.cumprod(1 + np.full(30, -0.002))
    tr[0, :] = 1.0
    arrays["adj_tr"] = tr
    ron = np.zeros((30, 2))
    ron[:, 0] = 0.0005
    ron[:, 1] = -0.0003
    arrays["r_on"] = ron
    rebuilt = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays={**arrays, "present": np.ones((30, 2), dtype=bool)},
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    panel = build_panel(rebuilt, last_row=29)
    u = tr[:-1, :] * (1.0 + ron[1:, :])
    u_full = np.full((30, 2), np.nan)
    u_full[1:, :] = u
    t = 10
    h = 5
    expected = u_full[t + 1 + h] / u_full[t + 1] - 1.0
    got = np.asarray(panel.labels[h][t, :], dtype=np.float64)
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-7)


def test_delisted_name_keeps_last_price_in_labels() -> None:
    cube = _synthetic_cube(n_s=40, n_n=1)
    arrays = {k: np.asarray(v, dtype=np.float64).copy() for k, v in cube.arrays.items()}
    present = np.ones((40, 1), dtype=bool)
    present[30:, 0] = False
    arrays["present"] = present
    rebuilt = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays=arrays,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    panel = build_panel(rebuilt, last_row=39)
    lab = np.asarray(panel.labels[5])
    assert bool(np.isfinite(lab[24, 0]))
    assert bool(np.isfinite(lab[28, 0]))


def test_warmup_rows_are_undefined() -> None:
    panel = build_panel(_synthetic_cube(n_s=300, n_n=1), last_row=299)
    mom = np.asarray(panel.features["mom_252_21"][:, 0])
    assert bool(np.all(np.isnan(mom[:252])))
    assert bool(np.isfinite(mom[280]))
    hi = np.asarray(panel.features["hi52"][:, 0])
    assert bool(np.all(np.isnan(hi[:125])))
    assert bool(np.isfinite(hi[200]))


def test_cube_is_not_mutated() -> None:
    cube = _synthetic_cube(n_s=60, n_n=1)
    before = {k: (np.asarray(v).tobytes(), np.asarray(v).shape) for k, v in cube.arrays.items()}
    first = build_panel(cube, last_row=59)
    second = build_panel(cube, last_row=59)
    for k, v in cube.arrays.items():
        assert (np.asarray(v).tobytes(), np.asarray(v).shape) == before[k]
    for name in FEATURE_NAMES:
        assert _equal_with_nan(first.features[name], second.features[name])


def test_absent_rows_are_nan() -> None:
    cube = _synthetic_cube(n_s=60, n_n=1)
    arrays = dict(cube.arrays)
    present = np.ones((60, 1), dtype=bool)
    present[30, 0] = False
    arrays["present"] = present
    rebuilt = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(cube.sessions),
        instrument_ids=list(cube.instrument_ids),
        arrays=arrays,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    panel = build_panel(rebuilt, last_row=59)
    for name in ("ret_1d", "vol20", "hi52", "size", "turn20", "gap1"):
        assert bool(np.isnan(np.asarray(panel.features[name])[30, 0]))


def _panel_digest(panel: FeaturePanel) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in FEATURE_NAMES:
        h.update(np.ascontiguousarray(panel.features[name]).tobytes())
    for horizon in HORIZONS:
        h.update(np.ascontiguousarray(panel.labels[horizon]).tobytes())
    return h.hexdigest()


def test_panel_bitwise_golden() -> None:
    assert _panel_digest(build_panel(_synthetic_cube(), last_row=299)) == (
        "46d19a38028c7d7b6430f78b6409d2f23ba2eb2100d5a51416a8479cb8379283"
    )


def test_panel_peak_memory_bounded() -> None:
    import gc
    import tracemalloc

    cube = _synthetic_cube(n_s=300, n_n=200, insts=[f"KRX:{i:06d}" for i in range(200)])
    probe = build_panel(cube, last_row=299)
    output_bytes = sum(
        arr.nbytes for arr in list(probe.features.values()) + list(probe.labels.values())
    )
    del probe
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    build_panel(cube, last_row=299)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak - before <= 2.5 * output_bytes


def test_panel_outputs_are_read_only_float32() -> None:
    panel = build_panel(_synthetic_cube(n_s=60, n_n=2), last_row=59)
    for arr in list(panel.features.values()) + list(panel.labels.values()):
        assert arr.dtype == np.float32
        assert bool(arr.flags["C_CONTIGUOUS"])
        assert not arr.flags.writeable
    assert list(panel.features) == list(FEATURE_NAMES)
    assert list(panel.labels) == list(HORIZONS)
