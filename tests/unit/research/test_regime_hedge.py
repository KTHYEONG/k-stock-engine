"""Unit tests for KOSDAQ 150 regime hedge leg and CompositeOverlay."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from src.backtest.overlay import (
    DerivativeConfig,
    OverlayState,
    OverlayTarget,
)
from src.core.pit import PITDataError
from src.research.trend_overlay import realised_vol
from src.research.regime_hedge import (
    CompositeOverlay,
    RegimeHedgeLeg,
    RegimeHedgeSpec,
    regime_hedge_derivative_config,
)


def _spec(**kwargs: Any) -> RegimeHedgeSpec:
    defaults: dict[str, Any] = {
        "tsmom_horizons": (5, 10, 15, 20),
        "target_vol": 0.10,
        "max_fraction": 1.5,
        "vol_window_sessions": 20,
        "rebalance_every_sessions": 5,
        "contract_multiplier_krw": 10000,
        "initial_margin_rate": 0.2,
        "margin_buffer_rate": 0.1,
        "margin_topup_trigger_fraction": 0.75,
        "futures_cost_rate": 0.0003,
        "futures_tax_rate": 0.11,
        "futures_annual_deduction_krw": 2500000,
    }
    defaults.update(kwargs)
    return RegimeHedgeSpec(**defaults)


def _round_half_down(value: Fraction) -> int:
    whole, remainder = divmod(value.numerator, value.denominator)
    return whole + int(2 * remainder > value.denominator)


def _state(
    session_idx: int,
    *,
    nav: int = 100_000_000,
    contracts: int = 0,
    leg_contracts: tuple[tuple[str, int], ...] = (),
    index_level: float = 1000.0,
) -> OverlayState:
    return OverlayState(
        session_idx=session_idx,
        nav=nav,
        stock_book_nav=nav,
        stock_book_returns=np.zeros(session_idx + 1, dtype=np.float64),
        index_returns=np.zeros(session_idx + 1, dtype=np.float64),
        index_level=index_level,
        contracts=contracts,
        inverse_units=0,
        leg_contracts=leg_contracts,
    )


def test_spec_validation() -> None:
    spec = _spec()
    assert spec.tsmom_horizons == (5, 10, 15, 20)
    assert spec.target_vol == 0.10
    assert "target_vol" in spec.canonical_json()

    with pytest.raises(ValueError, match="tsmom_horizons"):
        _spec(tsmom_horizons="not-a-tuple")
    with pytest.raises(ValueError, match="tsmom_horizons"):
        _spec(tsmom_horizons=(0, 5))
    with pytest.raises(ValueError, match="tsmom_horizons"):
        _spec(tsmom_horizons=(True, 5))
    with pytest.raises(ValueError, match="tsmom_horizons"):
        _spec(tsmom_horizons=())
    with pytest.raises(ValueError, match="strictly ascending"):
        _spec(tsmom_horizons=(10, 5))
    with pytest.raises(ValueError, match="target_vol"):
        _spec(target_vol=0.0)
    with pytest.raises(ValueError, match="max_fraction"):
        _spec(max_fraction=-0.1)
    with pytest.raises(ValueError, match="vol_window_sessions"):
        _spec(vol_window_sessions=1)
    with pytest.raises(ValueError, match="vol_window_sessions"):
        _spec(vol_window_sessions=True)
    with pytest.raises(ValueError, match="rebalance_every_sessions"):
        _spec(rebalance_every_sessions=0)
    with pytest.raises(ValueError, match="rebalance_every_sessions"):
        _spec(rebalance_every_sessions=True)
    with pytest.raises(ValueError, match="contract_multiplier_krw"):
        _spec(contract_multiplier_krw=0)
    with pytest.raises(ValueError, match="contract_multiplier_krw"):
        _spec(contract_multiplier_krw=True)
    with pytest.raises(ValueError, match="margin rate"):
        _spec(initial_margin_rate=1.0)
    with pytest.raises(ValueError, match="initial_margin_rate \\+ margin_buffer_rate"):
        _spec(initial_margin_rate=0.6, margin_buffer_rate=0.5)
    with pytest.raises(ValueError, match="margin_topup_trigger_fraction"):
        _spec(margin_topup_trigger_fraction=1.5)
    with pytest.raises(ValueError, match="rate"):
        _spec(futures_cost_rate=-0.01)
    with pytest.raises(ValueError, match="rate"):
        _spec(futures_tax_rate=1.5)
    with pytest.raises(ValueError, match="futures_annual_deduction_krw"):
        _spec(futures_annual_deduction_krw=-10)
    with pytest.raises(ValueError, match="futures_annual_deduction_krw"):
        _spec(futures_annual_deduction_krw=True)


def test_regime_hedge_leg_constructor_validation() -> None:
    spec = _spec(rebalance_every_sessions=5)
    levels = np.linspace(100.0, 200.0, 30)
    with pytest.raises(ValueError, match="rebalance_offset"):
        RegimeHedgeLeg(spec, index_level=levels, rebalance_offset=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rebalance_offset"):
        RegimeHedgeLeg(spec, index_level=levels, rebalance_offset=5)
    with pytest.raises(ValueError, match="rebalance_offset"):
        RegimeHedgeLeg(spec, index_level=levels, rebalance_offset=-1)
    with pytest.raises(ValueError, match="execution_delay"):
        RegimeHedgeLeg(spec, index_level=levels, execution_delay=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="execution_delay"):
        RegimeHedgeLeg(spec, index_level=levels, execution_delay=-1)
    with pytest.raises(ValueError, match="index_level must be a 1-D array"):
        RegimeHedgeLeg(spec, index_level=levels.reshape(6, 5))


def test_regime_hedge_derivative_config() -> None:
    spec = _spec()
    cfg = regime_hedge_derivative_config(spec)
    assert isinstance(cfg, DerivativeConfig)
    assert cfg.contract_multiplier_krw == 10000
    assert cfg.initial_margin_rate == 0.2
    assert cfg.margin_buffer_rate == 0.1
    assert cfg.futures_cost_rate == 0.0003
    assert cfg.inverse_cost_rate == 0.0
    assert cfg.futures_tax_rate == Decimal("0.11")
    assert cfg.futures_annual_deduction_krw == 2500000


def test_uptrend_is_flat() -> None:
    """Invariant: Given a rising KQ150 path; target 0 at every rebalance row."""
    spec = _spec(rebalance_every_sessions=5, tsmom_horizons=(5, 10))
    levels = np.linspace(100.0, 200.0, 60)
    leg = RegimeHedgeLeg(spec, index_level=levels, rebalance_offset=0)

    for row in range(len(levels)):
        state = _state(row)
        res = leg.target_contracts(state)
        if row % 5 == 0:
            assert res == 0, f"row {row} expected 0 but got {res}"
        else:
            assert res is None, f"row {row} expected None on non-rebalance session"


def test_downtrend_shorts_a_vol_scaled_size() -> None:
    """Invariant: Given a falling path (s = -1) with realised vol 20%, target_vol 0.10, max 1.5;
    contracts = round(0.5·nav/(mult·L)) > 0.
    """
    spec = _spec(
        tsmom_horizons=(5, 10),
        target_vol=0.10,
        max_fraction=1.5,
        vol_window_sessions=20,
        rebalance_every_sessions=1,
        contract_multiplier_krw=10000,
    )
    # Construct a downward path over 30 days where annualised vol of log returns is 0.20
    # daily vol = 0.20 / sqrt(252)
    daily_vol = 0.20 / math.sqrt(252)
    daily_drift = -0.01  # steadily downward so all horizons (5, 10) have level_t < level_{t-h} -> s = -1
    # Create alternating +- noise around negative drift to hit exactly 0.20 realised vol
    n = 35
    log_rets = np.full(n, daily_drift)
    # Add zero-mean alternating perturbation of magnitude daily_vol
    for i in range(1, n):
        log_rets[i] += daily_vol * (1 if i % 2 == 0 else -1)
    # Force exact sample std over the last 20 returns to match 0.20 annualised vol
    window_rets = log_rets[-20:].copy()
    window_std = np.std(window_rets, ddof=0)
    scaled_window = (window_rets - np.mean(window_rets)) * ((0.20 / math.sqrt(252)) / window_std) + daily_drift
    log_rets[-20:] = scaled_window

    sigma_actual = float(np.std(log_rets[-20:], ddof=0)) * math.sqrt(252)
    assert sigma_actual == pytest.approx(0.20, rel=1e-6)

    levels = 1000.0 * np.exp(np.cumsum(np.r_[0.0, log_rets]))
    row = len(levels) - 1
    level_t = float(levels[row])
    # Verify s = -1
    assert levels[row] < levels[row - 5]
    assert levels[row] < levels[row - 10]

    leg = RegimeHedgeLeg(spec, index_level=levels, rebalance_offset=0)
    nav = 100_000_000
    state = _state(row, nav=nav, index_level=level_t)
    contracts = leg.target_contracts(state)
    assert contracts is not None

    expected_notional = Fraction("0.10") / Fraction(str(sigma_actual)) * nav
    expected_contracts = round(float(expected_notional) / (10000 * level_t))
    assert contracts == expected_contracts
    assert contracts > 0


def test_partial_downtrend_scales_by_abs_s() -> None:
    """Invariant: Given s = -0.5; half the s = -1 notional (before rounding)."""
    # 4 horizons: 3 down, 1 up -> s = (-1 - 1 - 1 + 1) / 4 = -0.5
    horizons = (2, 4, 6, 8)
    spec = _spec(tsmom_horizons=horizons, rebalance_every_sessions=1, vol_window_sessions=5, max_fraction=1.0, target_vol=1.0)

    # Construct levels array ending at row 15 where:
    # level_15 < level_13, level_15 < level_11, level_15 < level_9, but level_15 > level_7
    levels = np.full(20, 1000.0)
    levels[15 - 2] = 1100.0  # past h=2
    levels[15 - 4] = 1100.0  # past h=4
    levels[15 - 6] = 1100.0  # past h=6
    levels[15 - 8] = 900.0   # past h=8 -> level_15 (1000) > level_7 (900)
    levels[15] = 1000.0

    levels_full = levels.copy()
    levels_full[15 - 8] = 1100.0  # s = -1; row 7 is outside the vol window, so sigma is shared
    nav = 10_000_000_000
    half = RegimeHedgeLeg(spec, index_level=levels, rebalance_offset=0).target_contracts(_state(15, nav=nav))
    full = RegimeHedgeLeg(spec, index_level=levels_full, rebalance_offset=0).target_contracts(_state(15, nav=nav))

    sigma = realised_vol(levels, 15, 5)
    assert sigma is not None
    assert sigma == realised_vol(levels_full, 15, 5)
    scale = min(Fraction(1), Fraction(1) / Fraction(str(sigma)))
    notional_full = scale * nav / (10000 * Fraction(1000))
    assert full == _round_half_down(notional_full)
    assert half == _round_half_down(notional_full / 2)
    assert half is not None
    assert 0 < half < full


def test_never_long() -> None:
    """Invariant: Given any path; every target >= 0."""
    spec = _spec(rebalance_every_sessions=2)
    rng = np.random.default_rng(42)

    for _ in range(10):
        path = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=50)))
        leg = RegimeHedgeLeg(spec, index_level=path, rebalance_offset=0)
        for t in range(len(path)):
            state = _state(t, nav=int(rng.integers(-1000, 1000000)))
            target = leg.target_contracts(state)
            if target is not None:
                assert target >= 0


def test_causal() -> None:
    """Invariant: Given two arrays equal through t; identical targets at rows <= t."""
    spec = _spec(rebalance_every_sessions=1)
    rng = np.random.default_rng(123)
    base = 1000.0 * np.exp(np.cumsum(rng.normal(-0.005, 0.01, size=60)))
    alt = base.copy()
    t_cut = 35
    alt[t_cut + 1 :] = alt[t_cut + 1 :] * 2.0  # diverge after t_cut

    leg1 = RegimeHedgeLeg(spec, index_level=base, rebalance_offset=0)
    leg2 = RegimeHedgeLeg(spec, index_level=alt, rebalance_offset=0)

    for row in range(t_cut + 1):
        state = _state(row, nav=50_000_000)
        assert leg1.target_contracts(state) == leg2.target_contracts(state)


def test_composite_keeps_a_silent_leg() -> None:
    """Invariant: Given a primary returning a target and the leg returning None;
    the composite target carries the leg's current contracts from state.leg_contracts.
    """
    class StubPrimary:
        def __init__(self, target_to_return: OverlayTarget | None) -> None:
            self._target = target_to_return

        def target(self, state: OverlayState) -> OverlayTarget | None:
            return self._target

    class StubLeg:
        def __init__(self, count_to_return: int | None) -> None:
            self._count = count_to_return

        def target_contracts(self, state: OverlayState) -> int | None:
            return self._count

    # Primary returns target, leg returns None -> composite keeps leg from state.leg_contracts
    composite = CompositeOverlay(
        StubPrimary(OverlayTarget(contracts=-10, inverse_value_krw=0)),
        legs={"kq150": StubLeg(None)},  # type: ignore[arg-type]
    )
    state = _state(10, contracts=-8, leg_contracts=(("kq150", 4),))
    target = composite.target(state)
    assert target is not None
    assert target.contracts == -10
    assert target.inverse_value_krw == 0
    assert target.legs == (("kq150", 4),)

    # Primary returns None, leg returns target -> composite keeps primary from state.contracts
    composite2 = CompositeOverlay(
        StubPrimary(None),
        legs={"kq150": StubLeg(7)},  # type: ignore[arg-type]
    )
    target2 = composite2.target(state)
    assert target2 is not None
    assert target2.contracts == -8
    assert target2.inverse_value_krw == 0
    assert target2.legs == (("kq150", 7),)

    # Both return None -> composite returns None
    composite3 = CompositeOverlay(
        StubPrimary(None),
        legs={"kq150": StubLeg(None)},  # type: ignore[arg-type]
    )
    target3 = composite3.target(state)
    assert target3 is None

    held_inverse = OverlayState(
        session_idx=10, nav=100_000_000, stock_book_nav=100_000_000,
        stock_book_returns=np.zeros(11), index_returns=np.zeros(11), index_level=1000.0,
        contracts=0, inverse_units=3, leg_contracts=(),
    )
    with pytest.raises(ValueError, match="inverse-ETF"):
        composite2.target(held_inverse)


def test_missing_or_nonfinite_level_raises_pit_data_error() -> None:
    spec = _spec(rebalance_every_sessions=1)
    levels = np.array([100.0, np.nan, 105.0])
    leg = RegimeHedgeLeg(spec, index_level=levels, rebalance_offset=0)

    # row 0 valid
    assert leg.target_contracts(_state(0)) == 0
    # row 1 non-finite raises PITDataError
    with pytest.raises(PITDataError, match="index level missing"):
        leg.target_contracts(_state(1))
    # row 3 past end raises PITDataError
    with pytest.raises(PITDataError, match="index level missing"):
        leg.target_contracts(_state(3))


def test_zero_vol_uses_max_fraction() -> None:
    spec = _spec(
        tsmom_horizons=(3,),
        target_vol=0.10,
        max_fraction=1.2,
        vol_window_sessions=2,
        rebalance_every_sessions=1,
    )
    levels = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])  # down over 3 rows, flat over the 2-return window
    nav = 900_000_000
    target = RegimeHedgeLeg(spec, index_level=levels, rebalance_offset=0).target_contracts(_state(5, nav=nav))
    assert target == _round_half_down(Fraction("1.2") * nav / (10000 * Fraction(90)))  # 1,200


def test_regime_hedge_leg_edge_cases() -> None:
    spec = _spec(
        tsmom_horizons=(5,),
        vol_window_sessions=10,
        rebalance_every_sessions=1,
    )
    levels = np.linspace(100.0, 50.0, 30)

    # 1. session_idx before run start
    leg = RegimeHedgeLeg(spec, index_level=levels)
    _ = leg.target_contracts(_state(10))  # initializes run start at 10
    with pytest.raises(ValueError, match="is before the run start"):
        leg.target_contracts(_state(5))

    # 2. execution_delay > session_idx -> idx < 0
    leg_delayed = RegimeHedgeLeg(spec, index_level=levels, execution_delay=5)
    assert leg_delayed.target_contracts(_state(2)) == 0

    # 3. Non-positive or non-finite NAV
    leg_nav = RegimeHedgeLeg(spec, index_level=levels)
    assert leg_nav.target_contracts(_state(15, nav=0)) == 0
    assert leg_nav.target_contracts(_state(15, nav=-1000)) == 0

    # 4. Non-finite level in tsmom past horizon -> signal is None -> returns 0
    bad_past = levels.copy()
    bad_past[10] = np.nan  # horizon 5 from 15 is 10
    leg_bad_past = RegimeHedgeLeg(spec, index_level=bad_past)
    assert leg_bad_past.target_contracts(_state(15)) == 0

    # 5. idx < vol_window_sessions -> vol_scale is None -> returns 0
    leg_short = RegimeHedgeLeg(spec, index_level=levels)
    assert leg_short.target_contracts(_state(7)) == 0  # 7 < 10

    # 6. Non-finite or <= 0 level in vol window -> vol_scale is None
    bad_vol = levels.copy()
    bad_vol[12] = 0.0
    leg_bad_vol = RegimeHedgeLeg(spec, index_level=bad_vol)
    assert leg_bad_vol.target_contracts(_state(18)) == 0
