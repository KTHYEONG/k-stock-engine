"""KOSPI 200 trend overlay invariants."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core.pit import PITDataError
from src.research.trend_overlay import TrendOverlay, TrendOverlaySpec, trend_derivative_config


def _spec(**overrides: Any) -> TrendOverlaySpec:
    values: dict[str, Any] = {
        "ma_sessions": 100,
        "long_fraction": 1.0,
        "short_fraction": 0.5,
        "rebalance_every_sessions": 1,
        "contract_multiplier_krw": 50000,
        "initial_margin_rate": 0.2,
        "margin_buffer_rate": 0.1,
        "margin_topup_trigger_fraction": 0.75,
        "futures_cost_rate": 0.0003,
        "futures_tax_rate": 0.11,
        "futures_annual_deduction_krw": 2500000,
    }
    values.update(overrides)
    return TrendOverlaySpec(**values)


def _state(idx: int, nav: int = 100_000_000, level: float = 1000.0) -> Any:
    from src.backtest.overlay import OverlayState

    return OverlayState(
        session_idx=idx,
        nav=nav,
        stock_book_nav=nav,
        stock_book_returns=np.zeros(idx + 1),
        index_returns=np.zeros(idx + 1),
        index_level=level,
        contracts=0,
        inverse_units=0,
    )


def _drive(spec: TrendOverlaySpec, levels: np.ndarray, nav: int = 100_000_000, **kw: Any) -> list[Any]:
    overlay = TrendOverlay(spec, index_level=np.ascontiguousarray(levels, dtype=np.float64), **kw)
    return [overlay.target(_state(i, nav=nav)) for i in range(len(levels))]


def test_long_above_moving_average() -> None:
    levels = 1000.0 + np.arange(120, dtype=np.float64)
    spec = _spec()
    out = _drive(spec, levels)[110]
    assert out is not None
    assert out.inverse_value_krw == 0
    assert out.contracts == -2


def test_short_below_moving_average() -> None:
    levels = 2000.0 - np.arange(120, dtype=np.float64)
    spec = _spec()
    out = _drive(spec, levels)[110]
    assert out is not None
    assert out.contracts > 0
    assert out.contracts == 1
    assert out.inverse_value_krw == 0


def test_insufficient_history_is_flat() -> None:
    levels = 1000.0 + np.arange(120, dtype=np.float64)
    out = _drive(_spec(), levels)[50]
    assert out is not None
    assert (out.contracts, out.inverse_value_krw) == (0, 0)
    gapped = levels.copy()
    gapped[51] = np.nan
    outs = _drive(_spec(rebalance_every_sessions=5), gapped)
    assert outs[110] is not None
    assert (outs[110].contracts, outs[110].inverse_value_krw) == (0, 0)


def test_causal_future_levels_cannot_change_decision() -> None:
    rng = np.random.default_rng(0)
    base = 1000.0 + np.cumsum(rng.normal(0.5, 1.0, 120))
    alt = base.copy()
    alt[80:] *= 1.5
    first = _drive(_spec(), base)
    second = _drive(_spec(), alt)
    for i in range(80):
        assert first[i] == second[i]


def test_execution_delay_lags_signal() -> None:
    rng = np.random.default_rng(1)
    levels = 1000.0 + np.cumsum(rng.normal(0.3, 1.0, 120))
    spec = _spec()
    plain = TrendOverlay(spec, index_level=levels, rebalance_offset=0)
    lagged = TrendOverlay(spec, index_level=levels, rebalance_offset=0, execution_delay=1)
    got_plain: list[Any] = [plain.target(_state(i)) for i in range(120)]
    got_lagged: list[Any] = [lagged.target(_state(i)) for i in range(120)]
    for t in range(1, 120):
        assert got_lagged[t] == got_plain[t - 1]


def test_zero_fractions_are_always_flat() -> None:
    levels = 1000.0 + np.arange(120, dtype=np.float64)
    for out in _drive(_spec(long_fraction=0.0, short_fraction=0.0), levels):
        assert out is not None
        assert (out.contracts, out.inverse_value_krw) == (0, 0)


def test_missing_level_at_rebalance_fails_closed() -> None:
    levels = 1000.0 + np.arange(120, dtype=np.float64)
    levels[110] = np.nan
    overlay = TrendOverlay(_spec(), index_level=levels, rebalance_offset=0)
    for i in range(110):
        overlay.target(_state(i))
    with pytest.raises(PITDataError):
        overlay.target(_state(110))


def test_rebalance_grid_and_guards() -> None:
    levels = 1000.0 + np.arange(30, dtype=np.float64)
    grid = TrendOverlay(_spec(rebalance_every_sessions=5, ma_sessions=5), index_level=levels, rebalance_offset=2)
    for i in range(15):
        assert (grid.target(_state(i)) is not None) == (i % 5 == 2)
    flat_nav = TrendOverlay(_spec(ma_sessions=5), index_level=1000.0 + np.arange(30, dtype=np.float64))
    for i in range(10):
        assert flat_nav.target(_state(i, nav=0)) is not None
    assert flat_nav.target(_state(10, nav=0)).contracts == 0  # type: ignore[union-attr]
    short_zero = TrendOverlay(
        _spec(ma_sessions=5, short_fraction=0.0),
        index_level=2000.0 - np.arange(30, dtype=np.float64),
    )
    outs = [short_zero.target(_state(i)) for i in range(30)]
    assert outs[29] is not None
    assert outs[29].contracts == 0
    with pytest.raises(ValueError, match="rebalance_offset"):
        TrendOverlay(_spec(), index_level=levels, rebalance_offset=5)
    with pytest.raises(ValueError, match="execution_delay"):
        TrendOverlay(_spec(), index_level=levels, execution_delay=-1)
    rewind = TrendOverlay(_spec(ma_sessions=5), index_level=levels)
    rewind.target(_state(5))
    with pytest.raises(ValueError, match="before the run start"):
        rewind.target(_state(2))


def test_derivative_config_has_no_inverse_terms() -> None:
    config = trend_derivative_config(_spec())
    assert config.contract_multiplier_krw == 50000
    assert config.inverse_cost_rate == 0.0
    assert str(config.inverse_tax_rate) == "0"


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="ma_sessions"):
        _spec(ma_sessions=1)
    with pytest.raises(ValueError, match="fraction"):
        _spec(long_fraction=-0.1)
    with pytest.raises(ValueError, match="margin"):
        _spec(initial_margin_rate=0.9, margin_buffer_rate=0.9)
    assert _spec().canonical_json() != _spec(long_fraction=0.5).canonical_json()


@pytest.mark.parametrize(
    "field",
    [
        {"signal": "invalid"},
        {"signal": "tsmom", "ma_sessions": 10},
        {"signal": "ma", "tsmom_horizons": (21, 63)},
        {"signal": "tsmom", "tsmom_horizons": ()},
        {"signal": "tsmom", "tsmom_horizons": "not_a_tuple"},
        {"signal": "tsmom", "tsmom_horizons": (10, 5)},
        {"signal": "tsmom", "tsmom_horizons": (10, 10)},
        {"signal": "tsmom", "tsmom_horizons": (0, 10)},
        {"signal": "tsmom", "tsmom_horizons": (True, 10)},
        {"vol_cap": -0.5, "vol_window_sessions": 20},
        {"vol_cap": float("nan"), "vol_window_sessions": 20},
        {"vol_cap": 0.25, "vol_window_sessions": 1},
        {"vol_cap": 0.25, "vol_window_sessions": True},
        {"ma_sessions": True},
        {"long_fraction": float("nan")},
        {"short_fraction": float("inf")},
        {"rebalance_every_sessions": 0},
        {"rebalance_every_sessions": 1.5},
        {"rebalance_every_sessions": True},
        {"contract_multiplier_krw": 0},
        {"contract_multiplier_krw": 1.5},
        {"contract_multiplier_krw": True},
        {"initial_margin_rate": 0.0},
        {"initial_margin_rate": 1.0},
        {"margin_buffer_rate": 0.0},
        {"margin_buffer_rate": -0.1},
        {"margin_topup_trigger_fraction": 0.0},
        {"margin_topup_trigger_fraction": 1.5},
        {"futures_cost_rate": -0.1},
        {"futures_cost_rate": 1.0},
        {"futures_tax_rate": float("nan")},
        {"futures_annual_deduction_krw": -1},
        {"futures_annual_deduction_krw": 1.5},
        {"futures_annual_deduction_krw": True},
    ],
)
def test_spec_rejects_invalid_fields(field: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"."):
        _spec(**field)


def test_constructor_rejects_bad_types() -> None:
    levels = 1000.0 + np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="rebalance_offset"):
        TrendOverlay(_spec(), index_level=levels, rebalance_offset=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="execution_delay"):
        TrendOverlay(_spec(), index_level=levels, execution_delay=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="execution_delay"):
        TrendOverlay(_spec(), index_level=levels, execution_delay=1.5)  # type: ignore[arg-type]


def test_decision_past_the_series_end_fails_closed() -> None:
    levels = 1000.0 + np.arange(30, dtype=np.float64)
    overlay = TrendOverlay(_spec(ma_sessions=5), index_level=levels)
    for i in range(30):
        overlay.target(_state(i))
    with pytest.raises(PITDataError):
        overlay.target(_state(30))


@pytest.mark.parametrize(("rising", "sign"), [(True, -1), (False, 1)])
@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.5, 0), (1.5, 1), (2.5, 2), (2.5000000001, 3), (2.4999999999, 2)],
)
def test_contract_rounding_ties_toward_zero(rising: bool, sign: int, fraction: float, expected: int) -> None:
    levels = np.array([900.0 if rising else 1100.0, 1000.0])
    spec = _spec(ma_sessions=2, long_fraction=fraction, short_fraction=fraction)
    overlay = TrendOverlay(spec, index_level=levels)
    assert overlay.target(_state(1, nav=50_000_000)).contracts == sign * expected


def test_finite_extreme_levels_preserve_trend_and_sizing() -> None:
    levels = np.array([1e308, 1.5e308])
    spec = _spec(ma_sessions=2, long_fraction=1e308)
    target = TrendOverlay(spec, index_level=levels).target(_state(1))
    assert target.contracts == -1333
    assert target.inverse_value_krw == 0


def test_index_level_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="1-D"):
        TrendOverlay(_spec(), index_level=np.ones((5, 2)))


def test_legacy_ma_spec_is_byte_identical() -> None:
    from pathlib import Path
    from src.research.pipeline import load_strategy_spec

    champion = load_strategy_spec(Path("config/research/strategies/ml_growth_t85_b50_k200.toml"))
    assert champion.spec_hash == "8c2aad2c8c3c5a133e7a3c6cef3134f43b89749dd87fa3d2f57c223118e4676c"
    spec = champion.trend_overlay
    assert spec is not None
    # Canonical JSON does not contain "signal" or None fields
    assert '"signal"' not in spec.canonical_json()
    assert '"tsmom_horizons"' not in spec.canonical_json()
    assert '"vol_cap"' not in spec.canonical_json()
    assert '"vol_window_sessions"' not in spec.canonical_json()

    levels = 1000.0 + np.arange(120, dtype=np.float64)
    out = _drive(spec, levels)[110]
    assert out is not None
    assert out.contracts == -2


def test_tsmom_all_up_is_full_long() -> None:
    # 300 sessions rising path
    levels = 100.0 + np.arange(300, dtype=np.float64)
    spec = _spec(
        signal="tsmom",
        ma_sessions=None,
        tsmom_horizons=(21, 63, 126, 252),
        long_fraction=1.0,
    )
    overlay = TrendOverlay(spec, index_level=levels)
    target = overlay.target(_state(280, nav=100_000_000, level=levels[280]))
    assert target is not None
    # s = 1.0, contracts = -round(nav / (mult * level_t))
    # notional = 100_000_000 / (50000 * 380) = 5.263 -> round 5
    expected_magnitude = round(100_000_000 / (50000 * levels[280]))
    assert target.contracts == -expected_magnitude


def test_tsmom_mixed_gives_fractional_size() -> None:
    # Path where only the 252-session return is negative, shorter returns positive
    # Horizons: (21, 63, 126, 252). 3 positive, 1 negative -> mean is (1 + 1 + 1 - 1)/4 = 0.5
    horizons = (21, 63, 126, 252)
    levels = np.ones(300, dtype=np.float64) * 200.0
    t = 280
    levels[t - 252] = 250.0  # past level higher -> return negative
    levels[t - 126] = 150.0  # past level lower -> return positive
    levels[t - 63] = 160.0   # past level lower -> return positive
    levels[t - 21] = 170.0   # past level lower -> return positive
    levels[t] = 200.0

    spec = _spec(
        signal="tsmom",
        ma_sessions=None,
        tsmom_horizons=horizons,
        long_fraction=1.0,
    )
    overlay = TrendOverlay(spec, index_level=levels)
    target = overlay.target(_state(t, nav=100_000_000, level=levels[t]))
    assert target is not None
    # s = 0.5, fraction = 1.0 -> m = 0.5 * 1.0 = 0.5
    # notional = 0.5 * 100_000_000 / (50000 * 200) = 5.0 -> contracts = -5
    assert target.contracts == -5


def test_missing_horizon_history_is_flat() -> None:
    # 200 finite levels and horizon 252 -> idx 199 - 252 < 0 -> flat
    levels = 1000.0 + np.arange(200, dtype=np.float64)
    spec = _spec(
        signal="tsmom",
        ma_sessions=None,
        tsmom_horizons=(21, 63, 126, 252),
    )
    overlay = TrendOverlay(spec, index_level=levels)
    target = overlay.target(_state(199))
    assert target is not None
    assert (target.contracts, target.inverse_value_krw) == (0, 0)


def _alternating_path(n: int, annual_vol: float, *, drift: float = 0.0) -> np.ndarray:
    step = annual_vol / np.sqrt(252)
    moves = np.where(np.arange(n - 1) % 2 == 0, step, -step) + drift
    return 1000.0 * np.exp(np.concatenate(([0.0], np.cumsum(moves))))


def test_cap_inactive_in_calm_regimes() -> None:
    levels = _alternating_path(100, 0.15, drift=0.002)
    window = np.diff(np.log(levels[30:51]))
    assert np.isclose(np.std(window) * np.sqrt(252), 0.15)
    uncapped = TrendOverlay(_spec(ma_sessions=10), index_level=levels).target(_state(50))
    capped = TrendOverlay(_spec(ma_sessions=10, vol_cap=0.25, vol_window_sessions=20), index_level=levels).target(
        _state(50)
    )
    assert uncapped is not None
    assert capped is not None
    assert uncapped.contracts != 0
    assert capped.contracts == uncapped.contracts


def test_cap_scales_in_turbulent_regimes() -> None:
    from fractions import Fraction

    levels = 1_000_000.0 / _alternating_path(21, 0.50)  # inverted path: the last move is up, so MA(2) is long
    sigma = float(np.std(np.diff(np.log(levels)))) * np.sqrt(252)
    assert np.isclose(sigma, 0.50)
    nav = 10_000_000_000
    uncapped = TrendOverlay(_spec(ma_sessions=2), index_level=levels).target(_state(20, nav=nav))
    capped = TrendOverlay(_spec(ma_sessions=2, vol_cap=0.25, vol_window_sessions=20), index_level=levels).target(
        _state(20, nav=nav)
    )
    assert uncapped is not None
    assert capped is not None
    notional = Fraction(nav) / (50000 * Fraction(str(float(levels[20]))))
    assert uncapped.contracts == -round(notional)
    assert capped.contracts == -round(notional * Fraction(str(0.25)) / Fraction(str(sigma)))
    assert abs(capped.contracts) == pytest.approx(abs(uncapped.contracts) / 2, abs=1)


def test_cap_requires_a_window() -> None:
    with pytest.raises(ValueError, match="vol_cap and vol_window_sessions must be set together"):
        _spec(vol_cap=0.25, vol_window_sessions=None)
    with pytest.raises(ValueError, match="vol_cap and vol_window_sessions must be set together"):
        _spec(vol_cap=None, vol_window_sessions=20)


def test_causality_with_vol_cap_and_tsmom() -> None:
    rng = np.random.default_rng(42)
    base = 1000.0 + np.cumsum(rng.normal(0.5, 1.0, 150))
    alt = base.copy()
    alt[100:] *= 2.0  # drastically alter future after row 100

    spec = _spec(
        signal="tsmom",
        ma_sessions=None,
        tsmom_horizons=(5, 10, 20),
        vol_cap=0.25,
        vol_window_sessions=20,
    )
    first = _drive(spec, base)
    second = _drive(spec, alt)
    for i in range(100):
        assert first[i] == second[i]


def test_tsmom_and_vol_cap_boundary_conditions() -> None:
    # 1. Non-finite or non-positive past level in tsmom -> flat
    levels = 1000.0 + np.arange(50, dtype=np.float64)
    levels[10] = np.nan
    spec = _spec(
        signal="tsmom",
        ma_sessions=None,
        tsmom_horizons=(5, 10),
    )
    # at row 20, past_lvl for horizon 10 is row 10 (nan)
    target = TrendOverlay(spec, index_level=levels).target(_state(20))
    assert target is not None
    assert (target.contracts, target.inverse_value_krw) == (0, 0)

    # 2. Perfectly flat levels -> tsmom signs are 0 -> s = 0 -> flat
    flat_levels = np.ones(50, dtype=np.float64) * 1000.0
    target_flat = TrendOverlay(spec, index_level=flat_levels).target(_state(20))
    assert target_flat is not None
    assert (target_flat.contracts, target_flat.inverse_value_krw) == (0, 0)

    # 3. vol_cap with insufficient history (< vol_window_sessions) -> flat
    spec_cap = _spec(
        ma_sessions=2,
        vol_cap=0.25,
        vol_window_sessions=30,
    )
    levels_short = 1000.0 + np.arange(20, dtype=np.float64)
    target_cap_short = TrendOverlay(spec_cap, index_level=levels_short).target(_state(15))
    assert target_cap_short is not None
    assert (target_cap_short.contracts, target_cap_short.inverse_value_krw) == (0, 0)

    # 4. vol_cap with non-finite level in vol window -> flat
    levels_gap = 1000.0 + np.arange(50, dtype=np.float64)
    levels_gap[25] = np.nan
    target_cap_gap = TrendOverlay(spec_cap, index_level=levels_gap).target(_state(40))
    assert target_cap_gap is not None
    assert (target_cap_gap.contracts, target_cap_gap.inverse_value_krw) == (0, 0)

    # 5. Zero fraction on active signal -> flat
    spec_zero_frac = _spec(
        signal="tsmom",
        ma_sessions=None,
        tsmom_horizons=(5, 10),
        long_fraction=0.0,
    )
    levels_rise = 1000.0 + np.arange(50, dtype=np.float64)
    target_zero = TrendOverlay(spec_zero_frac, index_level=levels_rise).target(_state(20))
    assert target_zero is not None
    assert (target_zero.contracts, target_zero.inverse_value_krw) == (0, 0)


def test_spec_invalid_combinations() -> None:
    with pytest.raises(ValueError, match="signal='tsmom' requires non-empty tsmom_horizons"):
        _spec(signal="tsmom", ma_sessions=None, tsmom_horizons=None)
    raw_dict = _spec().model_dump()
    raw_dict["ma_sessions"] = None
    with pytest.raises(ValueError, match="signal='ma' requires ma_sessions >= 2"):
        TrendOverlaySpec.model_validate(raw_dict)
    raw_tsmom = _spec(signal="tsmom", ma_sessions=None, tsmom_horizons=(5, 10)).model_dump()
    raw_tsmom["ma_sessions"] = 10
    with pytest.raises(ValueError, match="signal='tsmom' requires ma_sessions is None"):
        TrendOverlaySpec.model_validate(raw_tsmom)


def test_tsmom_third_signal_rounds_exactly() -> None:
    levels = np.full(30, 1000.0)
    levels[29 - 5] = 900.0
    levels[29 - 10] = 1100.0
    levels[29 - 20] = 1000.0
    spec = _spec(signal="tsmom", ma_sessions=None, tsmom_horizons=(5, 10, 20), long_fraction=0.75, short_fraction=0.75)
    target = TrendOverlay(spec, index_level=levels).target(_state(29, nav=200_000_000))
    assert target is not None
    assert target.contracts == 0  # s = 0: one up, one down, one unchanged

    levels[29 - 20] = 950.0  # s = 1/3; notional = 0.75 * 1/3 * 2e8 / (5e4 * 1e3) = 1.0 exactly
    target = TrendOverlay(spec, index_level=levels).target(_state(29, nav=200_000_000))
    assert target is not None
    assert target.contracts == -1



def test_shared_signal_helpers_reject_incomplete_history() -> None:
    from fractions import Fraction

    from src.research.trend_overlay import realised_vol, tsmom_signal

    levels = np.array([100.0, 110.0, 90.0, np.nan, 95.0], dtype=np.float64)
    assert tsmom_signal(levels, 2, (1, 2)) == Fraction(-1)
    assert tsmom_signal(levels, 3, (1,)) is None
    assert tsmom_signal(levels, 4, (1,)) is None
    assert tsmom_signal(levels, 1, (2,)) is None
    assert realised_vol(levels, 2, 2) is not None
    assert realised_vol(levels, 4, 2) is None
    assert realised_vol(levels, 1, 2) is None
