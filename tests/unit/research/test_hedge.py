from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.core.pit import PITDataError
from src.data.research_protocol import WindowAuthorization, WindowError
from src.research.hedge import (
    HedgeInputs,
    HedgeSpec,
    hedge_inputs_from_frame,
    rolling_beta,
    simulate_hedged_book,
)

N = 400
CAPITAL = 100_000_000


def _days(n: int = N + 1) -> list[date]:
    return [date(2018, 1, 1) + timedelta(days=i) for i in range(n)]


def _spec(**overrides: Any) -> HedgeSpec:
    values: dict[str, Any] = {
        "hedge_ratio": 1.0,
        "beta_window_sessions": 120,
        "beta_min_sessions": 40,
        "beta_cap": 2.0,
        "rebalance_every_sessions": 5,
        "use_futures": True,
        "contract_multiplier_krw": 10_000,
        "initial_margin_rate": 0.2175,
        "margin_buffer_rate": 0.10,
        "margin_topup_trigger_fraction": 0.75,
        "futures_cost_rate": 0.0003,
        "inverse_cost_rate": 0.0007,
        "resize_sell_cost_rate": 0.0025,
        "resize_buy_cost_rate": 0.0005,
        "futures_tax_rate": 0.11,
        "futures_annual_deduction_krw": 2_500_000,
        "inverse_tax_rate": 0.154,
    }
    values.update(overrides)
    return HedgeSpec(**values)


def _free(**overrides: Any) -> HedgeSpec:
    return _spec(
        futures_cost_rate=0.0,
        inverse_cost_rate=0.0,
        resize_sell_cost_rate=0.0,
        resize_buy_cost_rate=0.0,
        futures_tax_rate=0.0,
        inverse_tax_rate=0.0,
        **overrides,
    )


def _world(seed: int = 1) -> tuple[list[date], np.ndarray, HedgeInputs, np.ndarray]:
    rng = np.random.default_rng(seed)
    days = _days()
    idx_ret = rng.normal(0.0, 0.012, N)
    level = 1000.0 * np.cumprod(np.r_[1.0, 1.0 + idx_ret])
    inverse = 10_000.0 * np.cumprod(np.r_[1.0, 1.0 - idx_ret])
    stock = 0.7 * idx_ret + rng.normal(0.0, 0.008, N)
    return days, stock, HedgeInputs(sessions=tuple(days), index_level=level, inverse_close=inverse), idx_ret


def _auth(days: list[date]) -> WindowAuthorization:
    return WindowAuthorization(start=days[1], end=days[-1])


def _run(spec: HedgeSpec, seed: int = 1, **kwargs: Any) -> Any:
    days, stock, inputs, _ = _world(seed)
    return simulate_hedged_book(stock, days[1:], inputs, spec, capital_krw=CAPITAL, authorization=_auth(days), **kwargs)


def test_zero_hedge_is_identity() -> None:
    _, stock, _, _ = _world()
    result = _run(_spec(hedge_ratio=0.0))
    assert np.allclose(result.log_returns, np.log1p(stock))
    assert result.cost_krw == 0.0
    assert result.tax_krw == 0.0
    assert not result.futures_contracts.any()


def test_rolling_beta_recovers_slope_and_is_causal() -> None:
    rng = np.random.default_rng(0)
    index = rng.normal(0, 0.01, 300)
    stock = 0.6 * index
    beta = rolling_beta(stock, index, window=120, min_sessions=40, cap=2.0)
    assert np.all(beta[:40] == 0.0)
    assert beta[200] == pytest.approx(0.6)
    changed = stock.copy()
    changed[150:] *= -3.0
    assert np.array_equal(rolling_beta(changed, index, window=120, min_sessions=40, cap=2.0)[:151], beta[:151])


def test_rolling_beta_clips_and_handles_zero_variance() -> None:
    index = np.random.default_rng(1).normal(0, 0.01, 200)
    assert rolling_beta(-index, index, window=120, min_sessions=40, cap=2.0)[150] == 0.0
    assert rolling_beta(5.0 * index, index, window=120, min_sessions=40, cap=2.0)[150] == 2.0
    assert not rolling_beta(index, np.zeros(200), window=120, min_sessions=40, cap=2.0).any()


def test_hedge_removes_index_exposure_and_contracts_are_integers() -> None:
    days, stock, _, idx_ret = _world()
    result = _run(_free())
    hedged = np.expm1(result.log_returns)
    assert abs(np.corrcoef(hedged, idx_ret)[0, 1]) < abs(np.corrcoef(stock, idx_ret)[0, 1]) / 2
    assert result.futures_contracts.dtype == np.int64
    assert result.futures_contracts.max() > 0
    assert np.all(np.isfinite(result.nav_krw))
    assert result.sessions == tuple(days[1:])


def test_inverse_only_uses_no_futures_and_covers_full_notional() -> None:
    result = _run(_free(use_futures=False))
    assert not result.futures_contracts.any()
    assert result.inverse_notional_krw.max() > 0


def test_costs_and_taxes_reduce_nav() -> None:
    free = _run(_free())
    taxed = _run(_spec())
    assert taxed.cost_krw > 0
    assert taxed.tax_krw > 0
    assert taxed.nav_krw[-1] < free.nav_krw[-1]
    stressed = _run(_spec(), extra_cost_rate=0.001)
    assert stressed.cost_krw > taxed.cost_krw


def _tax_world(index_year_drifts: list[float]) -> tuple[list[date], np.ndarray, HedgeInputs]:
    days = _days(len(index_year_drifts) * 252 + 1)
    rng = np.random.default_rng(5)
    year_of = np.array([d.year for d in days[1:]])
    drift = np.array([index_year_drifts[y - days[1].year] for y in year_of])
    index_ret = drift + rng.normal(0.0, 0.004, len(year_of))
    level = 1000.0 * np.cumprod(np.r_[1.0, 1.0 + index_ret])
    stock = 0.9 * index_ret + rng.normal(0.0, 0.002, len(year_of))
    inputs = HedgeInputs(sessions=tuple(days), index_level=level, inverse_close=np.full(len(days), 10_000.0))
    return days, stock, inputs


def _futures_pnl_by_year(result: Any, days: list[date], inputs: HedgeInputs, multiplier: int) -> dict[int, float]:
    level = np.asarray(inputs.index_level)
    pnl: dict[int, float] = {}
    for i in range(1, len(result.sessions)):
        gain = -float(result.futures_contracts[i - 1]) * multiplier * (level[i + 1] - level[i])
        pnl[days[i + 1].year] = pnl.get(days[i + 1].year, 0.0) + gain
    return pnl


def test_futures_tax_has_deduction_and_no_carry_forward() -> None:
    days, stock, inputs = _tax_world([0.002, -0.002, 0.0])
    spec_kwargs = {
        "futures_cost_rate": 0.0,
        "inverse_cost_rate": 0.0,
        "resize_sell_cost_rate": 0.0,
        "resize_buy_cost_rate": 0.0,
        "inverse_tax_rate": 0.0,
        "margin_buffer_rate": 0.0,
    }
    results = {}
    for deduction in (0, 2_500_000):
        results[deduction] = simulate_hedged_book(
            stock,
            days[1:],
            inputs,
            _spec(futures_annual_deduction_krw=deduction, **spec_kwargs),
            capital_krw=CAPITAL * 10,
            authorization=_auth(days),
        )
    pnl = _futures_pnl_by_year(results[0], days, inputs, 10_000)
    assert pnl[days[1].year] < 0
    assert pnl[days[1].year + 1] > 0
    expected_no_deduction = 0.11 * sum(max(v, 0.0) for v in pnl.values())
    assert results[0].tax_krw == pytest.approx(expected_no_deduction, rel=1e-6)
    expected_deducted = 0.11 * sum(max(v - 2_500_000, 0.0) for v in pnl.values())
    assert results[2_500_000].tax_krw == pytest.approx(expected_deducted, rel=0.05)


def test_losing_futures_year_pays_no_tax_and_rally_triggers_margin_topup() -> None:
    days, stock, inputs = _tax_world([0.01])
    result = simulate_hedged_book(
        stock,
        days[1:],
        inputs,
        _spec(inverse_tax_rate=0.0, margin_topup_trigger_fraction=1.0, rebalance_every_sessions=20),
        capital_krw=CAPITAL,
        authorization=_auth(days),
    )
    assert result.tax_krw == 0.0
    assert result.margin_topups > 0
    assert np.all(result.nav_krw > 0)


def test_exhausted_stock_sleeve_fails_closed() -> None:
    days, stock, inputs = _tax_world([0.002])
    with pytest.raises(PITDataError, match="exceed the stock sleeve"):
        simulate_hedged_book(
            stock, days[1:], inputs, _spec(futures_cost_rate=5.0), capital_krw=CAPITAL, authorization=_auth(days)
        )
    squeeze_ret = np.where(np.arange(len(stock)) >= 100, 0.3, 0.001)
    squeeze_level = 1000.0 * np.cumprod(np.r_[1.0, 1.0 + squeeze_ret])
    crash = np.where(np.arange(len(stock)) >= 100, -0.6, 0.3 * squeeze_ret)
    squeezed = HedgeInputs(inputs.sessions, squeeze_level, inputs.inverse_close)
    with pytest.raises(PITDataError, match="margin shortfall"):
        simulate_hedged_book(crash, days[1:], squeezed, _spec(), capital_krw=CAPITAL, authorization=_auth(days))


def test_inverse_gain_is_taxed_pro_rata_and_loss_is_not_refunded() -> None:
    days = _days(300)
    n = len(days) - 1
    rng = np.random.default_rng(3)
    index_ret = rng.normal(0.0, 0.01, n)
    level = 1000.0 * np.cumprod(np.r_[1.0, 1.0 + index_ret])
    stock = 0.7 * index_ret + rng.normal(0.0, 0.003, n)
    trend_up = 10_000.0 * np.cumprod(np.r_[1.0, np.full(n, 1.002)])
    trend_down = 10_000.0 * np.cumprod(np.r_[1.0, np.full(n, 0.998)])
    kwargs = {"use_futures": False, "futures_tax_rate": 0.0}
    gain = simulate_hedged_book(
        stock,
        days[1:],
        HedgeInputs(tuple(days), level, trend_up),
        _spec(**kwargs),
        capital_krw=CAPITAL,
        authorization=_auth(days),
    )
    loss = simulate_hedged_book(
        stock,
        days[1:],
        HedgeInputs(tuple(days), level, trend_down),
        _spec(**kwargs),
        capital_krw=CAPITAL,
        authorization=_auth(days),
    )
    assert gain.tax_krw > 0.0
    assert loss.tax_krw == 0.0


def test_determinism_causality_and_offsets() -> None:
    days, stock, inputs, _ = _world()
    base = _run(_spec())
    again = _run(_spec())
    assert np.array_equal(base.log_returns, again.log_returns)
    cut = 250
    stock2 = stock.copy()
    stock2[cut + 1 :] *= 1.5
    level2 = np.asarray(inputs.index_level).copy()
    level2[cut + 2 :] *= 1.2
    inputs2 = HedgeInputs(sessions=inputs.sessions, index_level=level2, inverse_close=inputs.inverse_close)
    other = simulate_hedged_book(stock2, days[1:], inputs2, _spec(), capital_krw=CAPITAL, authorization=_auth(days))
    assert np.array_equal(base.futures_contracts[: cut + 1], other.futures_contracts[: cut + 1])
    assert np.array_equal(base.nav_krw[: cut + 1], other.nav_krw[: cut + 1])
    shifted = _run(_spec(), rebalance_offset=2)
    assert not np.array_equal(base.log_returns, shifted.log_returns)


def test_execution_delay_makes_beta_stale() -> None:
    base = _run(_free())
    delayed = _run(_free(), execution_delay=2)
    assert not np.array_equal(base.beta, delayed.beta)
    assert np.all(delayed.beta[:42] == 0.0)


def test_run_window_outside_authorization_rejected() -> None:
    days, stock, inputs, _ = _world()
    narrow = WindowAuthorization(start=days[1], end=days[100])
    with pytest.raises(WindowError):
        simulate_hedged_book(stock, days[1:], inputs, _spec(), capital_krw=CAPITAL, authorization=narrow)


def test_fail_closed_on_bad_inputs() -> None:
    days, stock, inputs, _ = _world()
    auth = _auth(days)
    bad_level = np.asarray(inputs.index_level).copy()
    bad_level[10] = np.nan
    with pytest.raises(PITDataError):
        simulate_hedged_book(
            stock,
            days[1:],
            HedgeInputs(inputs.sessions, bad_level, inputs.inverse_close),
            _spec(),
            capital_krw=CAPITAL,
            authorization=auth,
        )
    with pytest.raises(PITDataError):
        simulate_hedged_book(
            stock, [days[2], days[1], *days[3:]], inputs, _spec(), capital_krw=CAPITAL, authorization=auth
        )
    with pytest.raises(PITDataError):
        simulate_hedged_book(
            stock,
            days[:-1],
            inputs,
            _spec(),
            capital_krw=CAPITAL,
            authorization=WindowAuthorization(start=days[0], end=days[-1]),
        )
    nan_inverse = np.asarray(inputs.inverse_close).copy()
    nan_inverse[:] = np.nan
    with pytest.raises(PITDataError):
        simulate_hedged_book(
            stock,
            days[1:],
            HedgeInputs(inputs.sessions, inputs.index_level, nan_inverse),
            _spec(use_futures=False),
            capital_krw=CAPITAL,
            authorization=auth,
        )
    worst = stock.copy()
    worst[5] = -1.0
    with pytest.raises(ValueError, match="finite and > -1"):
        simulate_hedged_book(worst, days[1:], inputs, _spec(), capital_krw=CAPITAL, authorization=auth)
    with pytest.raises(ValueError, match="1-D, non-empty"):
        simulate_hedged_book(stock[:-1], days[1:], inputs, _spec(), capital_krw=CAPITAL, authorization=auth)
    with pytest.raises(ValueError, match="rebalance_offset"):
        simulate_hedged_book(
            stock, days[1:], inputs, _spec(), capital_krw=CAPITAL, rebalance_offset=5, authorization=auth
        )


def test_frame_alignment_and_duplicates() -> None:
    days = _days(4)
    frame = pl.DataFrame(
        {
            "session": [days[0], days[1], days[3]],
            "index_level": [1000.0, 1010.0, 1030.0],
            "inverse_close": [None, 9900, 9800],
        },
        schema={"session": pl.Date, "index_level": pl.Float64, "inverse_close": pl.Int64},
    )
    aligned = hedge_inputs_from_frame(frame, sessions=days)
    assert np.isnan(aligned.index_level[2])
    assert np.isnan(aligned.inverse_close[0])
    assert aligned.index_level[3] == 1030.0
    with pytest.raises(PITDataError):
        hedge_inputs_from_frame(pl.concat([frame, frame.head(1)]), sessions=days)


def test_spec_identity_and_validation() -> None:
    assert _spec().canonical_json() != _spec(hedge_ratio=0.5).canonical_json()
    with pytest.raises(ValueError, match="beta_min"):
        _spec(beta_min_sessions=500)
    with pytest.raises(ValueError, match="margin"):
        _spec(initial_margin_rate=1.0)
    with pytest.raises(ValueError, match=r"finite|>= 0"):
        _spec(futures_cost_rate=-0.1)


@pytest.mark.parametrize(
    "field",
    [
        {"hedge_ratio": float("nan")},
        {"hedge_ratio": -0.1},
        {"beta_window_sessions": 1},
        {"beta_window_sessions": True},
        {"beta_min_sessions": 1},
        {"beta_cap": 0.0},
        {"beta_cap": float("inf")},
        {"rebalance_every_sessions": 0},
        {"use_futures": "maybe"},
        {"contract_multiplier_krw": 0},
        {"initial_margin_rate": 0.0},
        {"margin_buffer_rate": -0.1},
        {"margin_topup_trigger_fraction": 0.0},
        {"margin_topup_trigger_fraction": 1.5},
        {"inverse_cost_rate": float("nan")},
        {"resize_sell_cost_rate": -1.0},
        {"resize_buy_cost_rate": -1.0},
        {"futures_tax_rate": 1.0},
        {"inverse_tax_rate": -0.1},
        {"futures_annual_deduction_krw": -1},
    ],
)
def test_spec_rejects_invalid_fields(field: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"."):
        _spec(**field)


def test_frictionless_split_conserves_nav_and_hedges_the_sleeve() -> None:
    from src.research.hedge import _frictionless_split

    nav0, ratio, notional, reserve = 157_930_935.0, 0.3025, 15_134_123.0, 0.3175
    stock, contracts, inverse = _frictionless_split(nav0, ratio, notional, reserve, True)
    assert contracts == 2
    assert stock + reserve * contracts * notional + inverse == pytest.approx(nav0)
    assert contracts * notional + inverse == pytest.approx(ratio * stock)
    assert 0.0 <= inverse < notional
    only_inverse = _frictionless_split(nav0, ratio, notional, reserve, False)
    assert only_inverse[1] == 0
    assert only_inverse[0] + only_inverse[2] == pytest.approx(nav0)
    assert _frictionless_split(nav0, 0.0, notional, reserve, True) == (nav0, 0, 0.0)


def test_rolling_beta_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        rolling_beta(np.zeros(3), np.zeros(4), window=2, min_sessions=2, cap=1.0)
    with pytest.raises(ValueError, match="equal-length"):
        rolling_beta(np.zeros(3), np.zeros(3), window=0, min_sessions=2, cap=1.0)


def test_inverse_history_gap_while_held_fails_closed() -> None:
    days, stock, inputs, _ = _world()
    gap = np.asarray(inputs.inverse_close).copy()
    gap[150:] = np.nan
    with pytest.raises(PITDataError, match="inverse close"):
        simulate_hedged_book(
            stock,
            days[1:],
            HedgeInputs(inputs.sessions, inputs.index_level, gap),
            _spec(),
            capital_krw=CAPITAL,
            authorization=_auth(days),
        )


def test_frame_validation_errors() -> None:
    days = _days(3)
    schema = {"session": pl.Date, "index_level": pl.Float64, "inverse_close": pl.Int64}
    with pytest.raises(PITDataError, match="missing columns"):
        hedge_inputs_from_frame(pl.DataFrame({"session": days}), sessions=days)
    with pytest.raises(PITDataError, match="index level"):
        hedge_inputs_from_frame(
            pl.DataFrame({"session": days[:1], "index_level": [0.0], "inverse_close": [None]}, schema=schema),
            sessions=days,
        )
    with pytest.raises(PITDataError, match="inverse close"):
        hedge_inputs_from_frame(
            pl.DataFrame({"session": days[:1], "index_level": [1.0], "inverse_close": [-5]}, schema=schema),
            sessions=days,
        )


def test_extra_cost_rate_must_be_non_negative_and_margin_total_below_one() -> None:
    days, stock, inputs, _ = _world()
    with pytest.raises(ValueError, match="extra_cost_rate"):
        simulate_hedged_book(
            stock, days[1:], inputs, _spec(), capital_krw=CAPITAL, extra_cost_rate=-0.1, authorization=_auth(days)
        )
    with pytest.raises(ValueError, match="must be < 1"):
        _spec(initial_margin_rate=0.6, margin_buffer_rate=0.5)


def _overlay_state(idx: int, stock: np.ndarray, index: np.ndarray, **overrides: object) -> object:
    from src.backtest.overlay import OverlayState

    params: dict[str, object] = {
        "session_idx": idx,
        "nav": 100_000_000,
        "stock_book_nav": 100_000_000,
        "stock_book_returns": stock[: idx + 1],
        "index_returns": index[: idx + 1],
        "index_level": 1500.0,
        "contracts": 0,
        "inverse_units": 0,
    }
    params.update(overrides)
    return OverlayState(**params)  # type: ignore[arg-type]


def test_beta_neutral_overlay_uses_session_own_return() -> None:
    from src.research.hedge import BetaNeutralOverlay, _frictionless_split

    rng = np.random.default_rng(0)
    index = rng.normal(0, 0.01, 60)
    stock = 0.5 * index
    overlay = BetaNeutralOverlay(_spec(beta_window_sessions=60, beta_min_sessions=40), rebalance_offset=4)
    target = None
    for i in range(60):
        out = overlay.target(_overlay_state(i, stock, index))
        if i == 59:
            target = out
    assert target is not None
    _, contracts, inverse = _frictionless_split(100_000_000, 0.5, 10_000 * 1500.0, 0.3175, True)
    assert target.contracts == contracts
    assert target.inverse_value_krw == int(np.floor(inverse))


def test_beta_neutral_overlay_guards() -> None:
    from src.backtest.overlay import OverlayState
    from src.core.pit import PITDataError
    from src.research.hedge import BetaNeutralOverlay, derivative_config

    rng = np.random.default_rng(1)
    index = rng.normal(0, 0.01, 30)
    stock = 0.5 * index
    few = BetaNeutralOverlay(_spec(beta_min_sessions=40), rebalance_offset=0)
    out = None
    for i in range(30):
        got = few.target(_overlay_state(i, stock, index))
        if i == 25:
            out = got
    assert out is not None
    assert (out.contracts, out.inverse_value_krw) == (0, 0)

    grid = BetaNeutralOverlay(_spec(rebalance_every_sessions=5), rebalance_offset=2)
    for i in range(15):
        got = grid.target(_overlay_state(i, stock, stock))
        assert (got is not None) == (i % 5 == 2)

    zero = BetaNeutralOverlay(_spec(hedge_ratio=0.0), rebalance_offset=0)
    assert zero.target(_overlay_state(0, stock, index)).contracts == 0  # type: ignore[union-attr]

    stale_base = np.concatenate([rng.normal(0, 0.01, 59), [0.05]])
    stale_stock = np.concatenate([0.5 * stale_base[:59], [0.2]])
    plain = BetaNeutralOverlay(
        _spec(beta_window_sessions=60, beta_min_sessions=2, rebalance_every_sessions=100),
        rebalance_offset=59,
    )
    delayed = BetaNeutralOverlay(
        _spec(beta_window_sessions=60, beta_min_sessions=2, rebalance_every_sessions=100),
        rebalance_offset=59,
        execution_delay=1,
    )
    last_plain = last_delayed = None
    for i in range(60):
        last_plain = plain.target(_overlay_state(i, stale_stock, stale_base))
        last_delayed = delayed.target(_overlay_state(i, stale_stock, stale_base))
    assert last_plain != last_delayed

    only_inverse = BetaNeutralOverlay(_spec(use_futures=False), rebalance_offset=0)
    held = only_inverse.target(_overlay_state(5, stock, index))
    assert held is not None
    assert held.contracts == 0

    bad_level = BetaNeutralOverlay(_spec(beta_min_sessions=2), rebalance_offset=0)
    with pytest.raises(PITDataError):
        bad_level.target(_overlay_state(29, stock, index, index_level=float("nan")))
    flat = BetaNeutralOverlay(_spec(), rebalance_offset=0)
    assert flat.target(_overlay_state(0, np.zeros(3), np.zeros(3))).contracts == 0  # type: ignore[union-attr]
    from src.research.hedge import _ols_beta_tail

    assert _ols_beta_tail(stock, np.zeros_like(stock), window=30, min_sessions=2, cap=2.0) == 0.0
    empty_nav = BetaNeutralOverlay(_spec(beta_min_sessions=2), rebalance_offset=0)
    assert empty_nav.target(_overlay_state(29, stock, index, nav=0)).contracts == 0  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="equal-length"):
        flat.target(
            OverlayState(
                session_idx=100,
                nav=1,
                stock_book_nav=1,
                stock_book_returns=np.zeros(3),
                index_returns=np.zeros(2),
                index_level=1.0,
                contracts=0,
                inverse_units=0,
            )
        )
    with pytest.raises(ValueError, match="rebalance_offset"):
        BetaNeutralOverlay(_spec(), rebalance_offset=5)
    with pytest.raises(ValueError, match="rebalance_offset"):
        BetaNeutralOverlay(_spec(), rebalance_offset=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="execution_delay"):
        BetaNeutralOverlay(_spec(), rebalance_offset=0, execution_delay=-1)
    with pytest.raises(ValueError, match="execution_delay"):
        BetaNeutralOverlay(_spec(), rebalance_offset=0, execution_delay=False)  # type: ignore[arg-type]
    rewind = BetaNeutralOverlay(_spec(), rebalance_offset=0)
    rewind.target(_overlay_state(5, stock, index))
    with pytest.raises(ValueError, match="before the run start"):
        rewind.target(_overlay_state(2, stock, index))

    config = derivative_config(_spec())
    assert config.contract_multiplier_krw == 10_000
    assert str(config.futures_tax_rate) == "0.11"
    assert str(config.inverse_tax_rate) == "0.154"


def test_regime_filter_flattens_hedge_in_uptrend_and_fails_safe() -> None:
    from src.research.hedge import BetaNeutralOverlay

    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.002, 80)
    up = 0.004 + noise
    down = -0.004 + noise
    spec = _spec(beta_window_sessions=60, beta_min_sessions=20, regime_ma_sessions=50, rebalance_every_sessions=100)

    def last_target(index: np.ndarray, s: HedgeSpec) -> Any:
        overlay = BetaNeutralOverlay(s, rebalance_offset=79)
        out = None
        for i in range(80):
            out = overlay.target(_overlay_state(i, 0.5 * index, index))
        return out

    assert last_target(up, spec).contracts == 0
    assert last_target(down, spec).contracts > 0
    assert last_target(up, _spec(beta_window_sessions=60, beta_min_sessions=20, rebalance_every_sessions=100)).contracts > 0
    short = _spec(beta_window_sessions=60, beta_min_sessions=20, regime_ma_sessions=200, rebalance_every_sessions=100)
    assert last_target(up, short).contracts > 0
    gap = up.copy()
    gap[-3] = np.nan
    assert last_target(gap, spec).contracts > 0


def test_regime_filter_in_simulator_is_causal_and_validated() -> None:
    days, stock, inputs, idx_ret = _world()
    base = simulate_hedged_book(stock, days[1:], inputs, _spec(), capital_krw=CAPITAL, authorization=_auth(days))
    gated = simulate_hedged_book(
        stock, days[1:], inputs, _spec(regime_ma_sessions=50), capital_krw=CAPITAL, authorization=_auth(days)
    )
    assert not np.array_equal(base.beta, gated.beta)
    assert bool(np.all((gated.beta == 0.0) | (gated.beta == base.beta)))
    cut = 250
    future = inputs.index_level.copy()
    future[cut + 2 :] *= 1.5
    moved = HedgeInputs(sessions=inputs.sessions, index_level=future, inverse_close=inputs.inverse_close)
    perturbed = simulate_hedged_book(
        stock, days[1:], moved, _spec(regime_ma_sessions=50), capital_krw=CAPITAL, authorization=_auth(days)
    )
    assert np.array_equal(perturbed.beta[:cut], gated.beta[:cut])
    for bad in (1, 0, True, 2.5):
        with pytest.raises(ValueError, match="regime_ma_sessions"):
            _spec(regime_ma_sessions=bad)
    assert _spec().canonical_json() != _spec(regime_ma_sessions=100).canonical_json()
    assert _spec(regime_ma_sessions=None).regime_ma_sessions is None
