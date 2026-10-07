"""Screening simulator execution invariants."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pytest

from src.data.research_protocol import WindowAuthorization, WindowError
from src.research.cube import ResearchCube
from src.research.simulator import SimConfig, simulate
from tests.fixtures.synthetic_panel import synthetic_cube, synthetic_sessions

INSTRS = ["KRX:000001", "KRX:000002", "KRX:000003"]


def _auth(start: date, end: date) -> WindowAuthorization:
    return WindowAuthorization(start=start, end=end)


def _config(**overrides: Any) -> SimConfig:
    base: dict[str, Any] = {
        "capital_krw": 10_000_000,
        "commission_rate": 0.0,
        "impact_k": 0.0,
        "max_participation": 1.0,
        "cash_buffer": 0.0,
    }
    base.update(overrides)
    return SimConfig.model_validate(base)


def _cube_with_returns(sessions: list[date]) -> ResearchCube:
    cube = synthetic_cube(sessions, INSTRS)
    arrays = dict(cube.arrays)
    r_on = np.zeros((len(sessions), 3))
    r_id = np.zeros((len(sessions), 3))
    r_on[:, 1] = 0.01
    r_id[:, 1] = 0.005
    arrays["r_on"] = r_on
    arrays["r_id"] = r_id
    arrays["tick_at_open"] = np.zeros((len(sessions), 3))
    return ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=list(sessions),
        instrument_ids=list(INSTRS),
        arrays=arrays,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )


def test_frictionless_buy_and_hold_identity() -> None:
    """Constant full weight reproduces the analytic return product."""
    sessions = synthetic_sessions(6)
    cube = _cube_with_returns(sessions)
    weights = {0: np.array([0.0, 1.0, 0.0])}
    result = simulate(
        cube, weights, start=sessions[0], end=sessions[-1],
        config=_config(), authorization=_auth(sessions[0], sessions[-1]),
    )
    r_on = np.asarray(cube.arrays["r_on"])[:, 1]
    r_id = np.asarray(cube.arrays["r_id"])[:, 1]
    expected = (1 + r_id[1]) * float(np.prod([(1 + r_on[t]) * (1 + r_id[t]) for t in range(2, 6)]))
    assert float(result.nav_krw[-1] / 10_000_000) == pytest.approx(expected, rel=1e-12)


def test_blocked_buy_stays_cash() -> None:
    """Entry-blocked execution leaves cash untouched."""
    sessions = synthetic_sessions(3)
    cube = synthetic_cube(sessions, INSTRS)
    arrays = dict(cube.arrays)
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[1, 0] = True
    arrays["entry_blocked"] = blocked
    arrays["tick_at_open"] = np.zeros((3, 3))
    locked = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=sessions, instrument_ids=INSTRS, arrays=arrays,
        exit_at=np.asarray(cube.exit_at), exit_halted=np.asarray(cube.exit_halted),
    )
    weights = {0: np.array([1.0, 0.0, 0.0])}
    result = simulate(
        locked, weights, start=sessions[0], end=sessions[-1],
        config=_config(), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert float(result.nav_krw[-1]) == pytest.approx(10_000_000.0)
    assert result.blocked_buy_share == pytest.approx(1.0)
    assert int(result.holdings[-1]) == 0


def test_lower_limit_open_blocks_sell() -> None:
    """Lower-limit opens keep the position."""
    sessions = synthetic_sessions(4)
    cube = synthetic_cube(sessions, INSTRS)
    arrays = dict(cube.arrays)
    arrays["tick_at_open"] = np.zeros((4, 3))
    staged = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=sessions, instrument_ids=INSTRS, arrays=arrays,
        exit_at=np.asarray(cube.exit_at), exit_halted=np.asarray(cube.exit_halted),
    )
    first = simulate(
        staged, {0: np.array([1.0, 0.0, 0.0])}, start=sessions[0], end=sessions[1],
        config=_config(), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert float(first.nav_krw[-1]) == pytest.approx(10_000_000.0)
    arrays2 = dict(staged.arrays)
    lower = np.zeros((4, 3), dtype=bool)
    lower[2, 0] = True
    arrays2["open_at_lower"] = lower
    held = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=sessions, instrument_ids=INSTRS, arrays=arrays2,
        exit_at=np.asarray(cube.exit_at), exit_halted=np.asarray(cube.exit_halted),
    )
    result = simulate(
        held, {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 0.0, 0.0])},
        start=sessions[0], end=sessions[2],
        config=_config(), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert int(result.holdings[2 - 0]) == 1


def test_participation_cap_binds() -> None:
    """Traded notional never exceeds the participation cap."""
    sessions = synthetic_sessions(3)
    cube = synthetic_cube(sessions, INSTRS, adtv20=1_000_000.0)
    arrays = dict(cube.arrays)
    arrays["tick_at_open"] = np.zeros((3, 3))
    capped = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=sessions, instrument_ids=INSTRS, arrays=arrays,
        exit_at=np.asarray(cube.exit_at), exit_halted=np.asarray(cube.exit_halted),
    )
    weights = {0: np.array([1.0, 0.0, 0.0])}
    result = simulate(
        capped, weights, start=sessions[0], end=sessions[2],
        config=_config(max_participation=0.01), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert float(result.turnover[1] * 10_000_000) <= 0.01 * 1_000_000.0 + 1e-6


def test_sell_tax_and_commission_charged() -> None:
    """One round trip loses exactly the analytic commission and tax."""
    sessions = synthetic_sessions(4)
    cube = synthetic_cube(sessions, INSTRS)
    arrays = dict(cube.arrays)
    arrays["tick_at_open"] = np.zeros((4, 3))
    arrays["sell_tax_rate"] = np.full((4, 3), 0.002)
    flat = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=sessions, instrument_ids=INSTRS, arrays=arrays,
        exit_at=np.asarray(cube.exit_at), exit_halted=np.asarray(cube.exit_halted),
    )
    weights = {0: np.array([1.0, 0.0, 0.0]), 2: np.array([0.0, 0.0, 0.0])}
    result = simulate(
        flat, weights, start=sessions[0], end=sessions[-1],
        config=_config(commission_rate=0.001), authorization=_auth(sessions[0], sessions[-1]),
    )
    capital = 10_000_000.0
    buy_cost = capital * 1.001
    bought = capital / buy_cost * capital
    proceeds = bought * (1 - 0.001 - 0.002)
    assert float(result.nav_krw[-1]) == pytest.approx(proceeds, rel=1e-9)


def test_tick_floor_applies() -> None:
    """Zero impact pays no tick with s=0 and one tick with s=1."""
    sessions = synthetic_sessions(3)
    cube = synthetic_cube(sessions, INSTRS, close=1000.0, tick=5.0)
    weights = {0: np.array([1.0, 0.0, 0.0])}
    result = simulate(
        cube, weights, start=sessions[0], end=sessions[-1],
        config=_config(), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert float(result.cost[1]) == pytest.approx(0.0, abs=1e-12)
    one_tick = simulate(
        cube, weights, start=sessions[0], end=sessions[-1],
        config=_config(auction_slippage_ticks=1.0), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert float(one_tick.cost[1] / one_tick.turnover[1]) == pytest.approx(0.005, abs=1e-12)


def test_halted_exit_policy() -> None:
    """Halted exits with zero recovery contribute no cash."""
    sessions = synthetic_sessions(4)
    cube = synthetic_cube(sessions, INSTRS)
    arrays = dict(cube.arrays)
    arrays["tick_at_open"] = np.zeros((4, 3))
    exit_at = np.array([2, -1, -1], dtype=np.int64)
    exit_halted = np.array([True, False, False])
    doomed = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=sessions, instrument_ids=INSTRS, arrays=arrays,
        exit_at=exit_at, exit_halted=exit_halted,
    )
    weights = {0: np.array([1.0, 0.0, 0.0])}
    result = simulate(
        doomed, weights, start=sessions[0], end=sessions[-1],
        config=_config(halted_exit_value=0.0), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert float(result.nav_krw[-1]) == pytest.approx(0.0, abs=1e-6)


def test_execution_delay_shifts_fills() -> None:
    """Delay 1 shows holdings one row later than delay 0."""
    sessions = synthetic_sessions(4)
    cube = synthetic_cube(sessions, INSTRS)
    arrays = dict(cube.arrays)
    arrays["tick_at_open"] = np.zeros((4, 3))
    flat = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=sessions, instrument_ids=INSTRS, arrays=arrays,
        exit_at=np.asarray(cube.exit_at), exit_halted=np.asarray(cube.exit_halted),
    )
    weights = {0: np.array([1.0, 0.0, 0.0])}
    fast = simulate(
        flat, weights, start=sessions[0], end=sessions[-1],
        config=_config(), authorization=_auth(sessions[0], sessions[-1]),
    )
    slow = simulate(
        flat, weights, start=sessions[0], end=sessions[-1],
        config=_config(execution_delay=1), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert int(fast.holdings[1]) == 1
    assert int(slow.holdings[1]) == 0
    assert int(slow.holdings[2]) == 1


def test_run_window_outside_authorization_rejected() -> None:
    """Runs outside the authorization fail closed."""
    sessions = synthetic_sessions(3)
    cube = synthetic_cube(sessions, INSTRS)
    auth = _auth(date(2023, 1, 1), date(2023, 12, 31))
    with pytest.raises(WindowError):
        simulate(
            cube, {}, start=sessions[0], end=date(2024, 1, 5),
            config=_config(), authorization=auth,
        )


def test_buys_scaled_to_available_cash() -> None:
    """Cost wedges never drive cash negative."""
    sessions = synthetic_sessions(3)
    cube = synthetic_cube(sessions, INSTRS, tick=500.0)
    weights = {0: np.array([0.5, 0.5, 0.0])}
    result = simulate(
        cube, weights, start=sessions[0], end=sessions[-1],
        config=_config(commission_rate=0.01), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert bool(np.all(np.isfinite(result.nav_krw)))
    assert float(np.min(result.nav_krw)) >= 0.0


def test_config_validation_rejects_bad_inputs(tmp_path) -> None:
    """SimConfig validators and engine TOML loading fail closed."""
    with pytest.raises(ValueError, match="positive int"):
        _config(capital_krw=0)
    with pytest.raises(ValueError, match="finite"):
        SimConfig.model_validate(
            {"capital_krw": 100, "commission_rate": float("nan"), "impact_k": 0.0,
             "max_participation": 0.1, "cash_buffer": 0.0}
        )
    with pytest.raises(ValueError, match="max_participation"):
        _config(max_participation=0.0)
    with pytest.raises(ValueError, match="cash_buffer"):
        _config(cash_buffer=1.0)
    with pytest.raises(ValueError, match="halted_exit_value"):
        _config(halted_exit_value=2.0)
    with pytest.raises(ValueError, match="execution_delay"):
        _config(execution_delay=-1)
    with pytest.raises(ValueError, match="finite and >= 0"):
        _config(commission_rate=float("nan"))
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        _config(max_participation=0.0)
    with pytest.raises(ValueError, match="in \\[0, 1\\)"):
        _config(cash_buffer=1.0)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        _config(halted_exit_value=2.0)
    assert _config().canonical_json()
    loaded = SimConfig.from_engine_toml(tmp_path / "missing.toml", capital_krw=100) if False else None
    _ = loaded

    assert SimConfig.from_engine_toml(
        __import__("pathlib").Path("config/backtest/default_engine.toml"), capital_krw=100
    ).capital_krw == 100
    nested = tmp_path / "nested.toml"
    nested.write_text(
        '[execution]\nscenario = "open_auction"\nmax_participation = 0.1\ncarry_unfilled = false\n'
        '[costs]\ncommission_rate = "0.001"\ndividend_withholding_rate = "0.1"\nimpact_k = 0.1\n'
        'cash_buffer = 0.0\n',
        encoding="utf-8",
    )
    assert SimConfig.from_engine_toml(nested, capital_krw=100).commission_rate == pytest.approx(0.001)
    bad = tmp_path / "bad.toml"
    bad.write_text('scenario = "open_auction"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        SimConfig.from_engine_toml(bad, capital_krw=100)


def test_simulate_validation_branches() -> None:
    """Bad windows and target rows fail closed."""
    sessions = synthetic_sessions(4)
    cube = synthetic_cube(sessions, INSTRS)
    auth = _auth(sessions[0], sessions[-1])
    with pytest.raises(WindowError):
        simulate(cube, {}, start=sessions[0], end=sessions[-1],
                 config=_config(), authorization=_auth(sessions[1], sessions[-1]))
    with pytest.raises(ValueError, match="within the cube"):
        simulate(cube, {}, start=sessions[-1], end=sessions[0],
                 config=_config(), authorization=auth)
    with pytest.raises(ValueError, match="must be an int"):
        simulate(cube, {True: np.zeros(3)}, start=sessions[0], end=sessions[-1],
                 config=_config(), authorization=auth)
    with pytest.raises(ValueError, match="outside the executable"):
        simulate(cube, {99: np.zeros(3)}, start=sessions[0], end=sessions[-1],
                 config=_config(), authorization=auth)
    with pytest.raises(ValueError, match="wrong shape"):
        simulate(cube, {0: np.zeros(2)}, start=sessions[0], end=sessions[-1],
                 config=_config(), authorization=auth)
    with pytest.raises(ValueError, match="invalid target"):
        simulate(cube, {0: np.array([-0.1, 0.0, 0.0])}, start=sessions[0], end=sessions[-1],
                 config=_config(), authorization=auth)
    with pytest.raises(ValueError, match="invalid target"):
        simulate(cube, {0: np.array([0.6, 0.6, 0.0])}, start=sessions[0], end=sessions[-1],
                 config=_config(), authorization=auth)


def test_auction_slippage_validated_and_loaded(tmp_path) -> None:
    """SimConfig validates the auction grid and reads the engine TOML key."""
    with pytest.raises(ValueError, match="finite"):
        _config(auction_slippage_ticks=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _config(auction_slippage_ticks=-0.5)
    assert _config(auction_slippage_ticks=0.5).auction_slippage_ticks == pytest.approx(0.5)
    engine = tmp_path / "engine.toml"
    engine.write_text(
        'scenario = "open_auction"\nmax_participation = 0.1\ncarry_unfilled = false\n'
        'commission_rate = "0.001"\ndividend_withholding_rate = "0.1"\nimpact_k = 0.0\n'
        'cash_buffer = 0.0\nauction_slippage_ticks = 1.5\n',
        encoding="utf-8",
    )
    from pathlib import Path as _Path

    assert SimConfig.from_engine_toml(_Path(engine), capital_krw=100).auction_slippage_ticks == pytest.approx(1.5)
    assert SimConfig.from_engine_toml(
        _Path("config/backtest/default_engine.toml"), capital_krw=100
    ).auction_slippage_ticks == pytest.approx(1.0)


def test_simulator_ledger_pricing_parity() -> None:
    """One-tick slippage prices the same fraction in both engines."""
    sessions = synthetic_sessions(3)
    cube = synthetic_cube(sessions, INSTRS, close=10000.0, tick=50.0)
    weights = {0: np.array([1.0, 0.0, 0.0])}
    one_tick = simulate(
        cube, weights, start=sessions[0], end=sessions[-1],
        config=_config(auction_slippage_ticks=1.0), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert float(one_tick.cost[1] / one_tick.turnover[1]) == pytest.approx(50.0 / 10000.0, rel=1e-9)
    half_tick = simulate(
        cube, weights, start=sessions[0], end=sessions[-1],
        config=_config(auction_slippage_ticks=0.5), authorization=_auth(sessions[0], sessions[-1]),
    )
    assert float(half_tick.cost[1] / half_tick.turnover[1]) == pytest.approx(25.0 / 10000.0, rel=1e-9)


def test_zero_band_is_bitwise_identity() -> None:
    sessions = synthetic_sessions(5)
    cube = synthetic_cube(sessions, INSTRS)
    targets = {
        0: np.array([0.5, 0.3, 0.0]),
        1: np.array([0.4, 0.4, 0.0]),
        2: np.array([0.0, 0.5, 0.3]),
    }
    auth = _auth(sessions[0], sessions[-1])
    cfg = _config()
    base = simulate(cube, targets, start=sessions[0], end=sessions[-1], config=cfg, authorization=auth)
    banded = simulate(
        cube, targets, start=sessions[0], end=sessions[-1], config=cfg, authorization=auth,
        rebalance_band=0.0,
    )
    for name in ("nav_krw", "log_returns", "turnover", "cost", "holdings"):
        assert np.array_equal(np.asarray(getattr(base, name)), np.asarray(getattr(banded, name)))
    assert base.blocked_buy_share == banded.blocked_buy_share


def test_band_reduces_turnover_without_touching_exits() -> None:
    sessions = synthetic_sessions(6)
    cube = _cube_with_returns(sessions)
    drifted = {
        0: np.array([0.5, 0.3, 0.0]),
        1: np.array([0.48, 0.32, 0.0]),
        2: np.array([0.0, 0.5, 0.3]),
        3: np.array([0.0, 0.48, 0.32]),
        4: np.zeros(3),
    }
    auth = _auth(sessions[0], sessions[-1])
    cfg = _config()
    plain = simulate(cube, drifted, start=sessions[0], end=sessions[-1], config=cfg, authorization=auth)
    banded = simulate(
        cube, drifted, start=sessions[0], end=sessions[-1], config=cfg, authorization=auth,
        rebalance_band=0.5,
    )
    assert float(np.sum(banded.turnover)) <= float(np.sum(plain.turnover))
    assert float(np.sum(banded.turnover[:3])) <= float(np.sum(plain.turnover[:3]))
    assert float(banded.turnover[3]) > 0.0
    assert banded.holdings[3] == 2
    assert banded.holdings[-1] == 0
    assert banded.turnover[-1] > 0.0
    with pytest.raises(ValueError, match="rebalance_band"):
        simulate(
            cube, drifted, start=sessions[0], end=sessions[-1], config=cfg, authorization=auth,
            rebalance_band=1.0,
        )
