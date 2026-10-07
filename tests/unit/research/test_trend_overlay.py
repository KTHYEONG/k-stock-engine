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
