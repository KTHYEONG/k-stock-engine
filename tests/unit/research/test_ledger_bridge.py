"""Ledger bridge alignment and parity invariants."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.backtest.engine import DelistPolicy
from src.core.market_rules import load_krx_market_rules
from src.data.research_protocol import LockboxAuthorization, Segment
from src.research.cube import ResearchCube
from src.research.ledger_bridge import run_ledger
from src.research.simulator import SimConfig, simulate
from src.core.pit import PITDataError
from tests.fixtures.synthetic_panel import (
    synthetic_cube,
    synthetic_panel_row,
    synthetic_sessions,
    write_synthetic_panel,
)

ENGINE_TOML = Path("config/backtest/default_engine.toml")
RULES_PATH = Path("config/market/krx_market_rules.toml")


def _auth(sessions: list[date]) -> LockboxAuthorization:
    return LockboxAuthorization(
        segment=Segment.DISCOVERY, start=sessions[0], end=sessions[-1], spec_hash=None, evidence=True,
    )


def test_cube_panel_alignment_enforced(tmp_path: Path) -> None:
    """Mismatched instrument ids fail closed."""
    sessions = synthetic_sessions(5)
    cube = synthetic_cube(sessions, ["KRX:000001", "KRX:000002"])
    rows = [synthetic_panel_row(day, inst) for day in sessions for inst in ["KRX:000001", "KRX:000003"]]
    panel = write_synthetic_panel(tmp_path / "gold", "market_panel_test", rows)
    with pytest.raises(PITDataError):
        run_ledger(
            cube=cube, targets={}, panel_dir=panel, dividends=pl.DataFrame(),
            engine_config_path=ENGINE_TOML, rules=load_krx_market_rules(RULES_PATH),
            market_cache_root=tmp_path / "cache", capital_krw=10_000_000,
            halted_exit_policy=DelistPolicy.LAST_CLOSE,
            start=sessions[0], end=sessions[-1], authorization=_auth(sessions),
        )


def test_parity_on_synthetic_panel(tmp_path: Path) -> None:
    """Fast and ledger annualized growth agree within 0.002."""
    sessions = synthetic_sessions(65, start=date(2020, 1, 6))
    instruments = [f"KRX:{i:06d}" for i in range(6)]
    rows = []
    for day in sessions:
        for n, inst in enumerate(instruments):
            close = 10000 + n * 100
            rows.append(
                synthetic_panel_row(
                    day, inst, open=close, high=close, low=close, close=close, base_price=close,
                )
            )
    panel = write_synthetic_panel(tmp_path / "gold", "market_panel_test", rows)
    cube = synthetic_cube(sessions, instruments, tick=50.0)
    arrays = dict(cube.arrays)
    arrays["tick_at_open"] = np.full((len(sessions), len(instruments)), 50.0)
    cube = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=sessions, instrument_ids=instruments, arrays=arrays,
        exit_at=np.asarray(cube.exit_at), exit_halted=np.asarray(cube.exit_halted),
    )
    size = np.zeros((len(sessions), len(instruments)))
    for n in range(len(instruments)):
        size[:, n] = float(n)
    targets = {
        row: np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        for row in range(0, len(sessions) - 1, 21)
    }
    auth = _auth(sessions)
    sim_config = SimConfig.from_engine_toml(ENGINE_TOML, capital_krw=10_000_000)
    fast = simulate(cube, targets, start=sessions[0], end=sessions[-1], config=sim_config, authorization=auth)
    outcome = run_ledger(
        cube=cube, targets=targets, panel_dir=panel, dividends=pl.DataFrame(),
        engine_config_path=ENGINE_TOML, rules=load_krx_market_rules(RULES_PATH),
        market_cache_root=tmp_path / "cache", capital_krw=10_000_000,
        halted_exit_policy=DelistPolicy.LAST_CLOSE,
        start=sessions[0], end=sessions[-1], authorization=auth,
    )
    fast_g = float(np.mean(fast.log_returns) * 252)
    ledger_g = float(np.mean(outcome.log_returns) * 252)
    assert abs(fast_g - ledger_g) <= 0.002
    assert len(outcome.log_returns) == len(sessions)
    assert isinstance(outcome.reject_counts, dict)
    assert len(outcome.ledger_hash) == 64


def test_ledger_validation_and_dividends(tmp_path: Path) -> None:
    """Bad engine configs, lockbox windows and dividend filtering are covered."""
    from src.core.pit import PITDataError as _PIT

    sessions = synthetic_sessions(6, start=date(2020, 1, 6))
    instruments = ["KRX:000001", "KRX:000002"]
    rows = [synthetic_panel_row(day, inst) for day in sessions for inst in instruments]
    panel = write_synthetic_panel(tmp_path / "gold", "market_panel_test", rows)
    cube = synthetic_cube(sessions, instruments)
    auth = _auth(sessions)
    rules = load_krx_market_rules(RULES_PATH)
    with pytest.raises(_PIT):
        run_ledger(
            cube=cube, targets={}, panel_dir=panel, dividends=pl.DataFrame(),
            engine_config_path=ENGINE_TOML, rules=rules,
            market_cache_root=tmp_path / "cache", capital_krw=1000,
            halted_exit_policy=DelistPolicy.LAST_CLOSE,
            start=sessions[0], end=date(2024, 1, 5), authorization=auth,
        )
    bad = tmp_path / "bad_engine.toml"
    bad.write_text('scenario = "open_auction"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        run_ledger(
            cube=cube, targets={}, panel_dir=panel, dividends=pl.DataFrame(),
            engine_config_path=bad, rules=rules,
            market_cache_root=tmp_path / "cache", capital_krw=1000,
            halted_exit_policy=DelistPolicy.LAST_CLOSE,
            start=sessions[0], end=sessions[-1], authorization=auth,
        )
    dividends = pl.DataFrame(
        [{"instrument_id": instruments[0], "ex_session": sessions[1],
          "pay_session": sessions[2], "dps_krw": 10}],
        schema={"instrument_id": pl.String, "ex_session": pl.Date,
                "pay_session": pl.Date, "dps_krw": pl.Int64},
    )
    nested_engine = tmp_path / "nested_engine.toml"
    nested_engine.write_text(
        '[execution]\nscenario = "open_auction"\nmax_participation = 0.1\ncarry_unfilled = false\n'
        '[costs]\ncommission_rate = "0.001"\ndividend_withholding_rate = "0.1"\nimpact_k = 0.1\n'
        'cash_buffer = 0.0\n',
        encoding="utf-8",
    )
    nested_outcome = run_ledger(
        cube=cube, targets={}, panel_dir=panel, dividends=dividends,
        engine_config_path=nested_engine, rules=rules,
        market_cache_root=tmp_path / "cache", capital_krw=1000,
        halted_exit_policy=DelistPolicy.LAST_CLOSE,
        start=sessions[0], end=sessions[-1], authorization=auth,
    )
    assert nested_outcome.capital_krw == 1000
    base_values: dict[str, str] = {
        "scenario": '"open_auction"',
        "max_participation": "0.01",
        "carry_unfilled": "false",
        "commission_rate": '"0.00015"',
        "dividend_withholding_rate": '"0.154"',
        "impact_k": "0.1",
        "cash_buffer": "0.005",
    }
    bad_values: dict[str, str] = {
        "max_participation": '"bad"',
        "carry_unfilled": '"yes"',
        "commission_rate": '"bad"',
        "dividend_withholding_rate": '"bad"',
        "impact_k": '"bad"',
        "cash_buffer": '"bad"',
    }
    for bad_key, bad_value in bad_values.items():
        broken = tmp_path / f"broken_{bad_key}.toml"
        lines = []
        for key, value in base_values.items():
            lines.append(f"{key} = {bad_value if key == bad_key else value}")
        broken.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"must be|invalid|missing"):
            run_ledger(
                cube=cube, targets={}, panel_dir=panel, dividends=pl.DataFrame(),
                engine_config_path=broken, rules=rules,
                market_cache_root=tmp_path / "cache", capital_krw=1000,
                halted_exit_policy=DelistPolicy.LAST_CLOSE,
                start=sessions[0], end=sessions[-1], authorization=auth,
            )
    outcome = run_ledger(
        cube=cube, targets={}, panel_dir=panel, dividends=dividends,
        engine_config_path=ENGINE_TOML, rules=rules,
        market_cache_root=tmp_path / "cache", capital_krw=1000,
        halted_exit_policy=DelistPolicy.ZERO,
        start=sessions[0], end=sessions[-1], authorization=auth,
    )
    assert outcome.capital_krw == 1000
    assert outcome.halted_exit_policy == "zero"
