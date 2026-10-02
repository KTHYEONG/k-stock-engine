"""Ledger bridge alignment and parity invariants."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.backtest.engine import DelistPolicy
from src.backtest.market import MarketArrays
from src.core.market_rules import load_krx_market_rules
from src.data.research_protocol import WindowAuthorization
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


def _auth(sessions: list[date]) -> WindowAuthorization:
    return WindowAuthorization(
        start=sessions[0],
        end=sessions[-1],
    )


def test_cube_panel_alignment_enforced(tmp_path: Path) -> None:
    """Mismatched instrument ids fail closed."""
    sessions = synthetic_sessions(5)
    cube = synthetic_cube(sessions, ["KRX:000001", "KRX:000002"])
    rows = [synthetic_panel_row(day, inst) for day in sessions for inst in ["KRX:000001", "KRX:000003"]]
    panel = write_synthetic_panel(tmp_path / "gold", "market_panel_test", rows)
    with pytest.raises(PITDataError):
        run_ledger(
            cube=cube,
            targets={},
            panel_dir=panel,
            dividends=pl.DataFrame(),
            engine_config_path=ENGINE_TOML,
            rules=load_krx_market_rules(RULES_PATH),
            market_cache_root=tmp_path / "cache",
            capital_krw=10_000_000,
            halted_exit_policy=DelistPolicy.LAST_CLOSE,
            start=sessions[0],
            end=sessions[-1],
            authorization=_auth(sessions),
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
                    day,
                    inst,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    base_price=close,
                )
            )
    panel = write_synthetic_panel(tmp_path / "gold", "market_panel_test", rows)
    cube = synthetic_cube(sessions, instruments, tick=50.0)
    arrays = dict(cube.arrays)
    arrays["tick_at_open"] = np.full((len(sessions), len(instruments)), 50.0)
    cube = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=sessions,
        instrument_ids=instruments,
        arrays=arrays,
        exit_at=np.asarray(cube.exit_at),
        exit_halted=np.asarray(cube.exit_halted),
    )
    size = np.zeros((len(sessions), len(instruments)))
    for n in range(len(instruments)):
        size[:, n] = float(n)
    targets = {row: np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]) for row in range(0, len(sessions) - 1, 21)}
    auth = _auth(sessions)
    sim_config = SimConfig.from_engine_toml(ENGINE_TOML, capital_krw=10_000_000)
    fast = simulate(cube, targets, start=sessions[0], end=sessions[-1], config=sim_config, authorization=auth)
    outcome = run_ledger(
        cube=cube,
        targets=targets,
        panel_dir=panel,
        dividends=pl.DataFrame(),
        engine_config_path=ENGINE_TOML,
        rules=load_krx_market_rules(RULES_PATH),
        market_cache_root=tmp_path / "cache",
        capital_krw=10_000_000,
        halted_exit_policy=DelistPolicy.LAST_CLOSE,
        start=sessions[0],
        end=sessions[-1],
        authorization=auth,
    )
    fast_g = float(np.mean(fast.log_returns) * 252)
    ledger_g = float(np.mean(outcome.log_returns) * 252)
    assert abs(fast_g - ledger_g) <= 0.002
    assert len(outcome.log_returns) == len(sessions)
    assert isinstance(outcome.reject_counts, dict)
    assert len(outcome.ledger_hash) == 64


def test_ledger_validation_and_dividends(tmp_path: Path) -> None:
    """Bad engine configs, window-guard windows and dividend filtering are covered."""
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
            cube=cube,
            targets={},
            panel_dir=panel,
            dividends=pl.DataFrame(),
            engine_config_path=ENGINE_TOML,
            rules=rules,
            market_cache_root=tmp_path / "cache",
            capital_krw=1000,
            halted_exit_policy=DelistPolicy.LAST_CLOSE,
            start=sessions[0],
            end=date(2024, 1, 5),
            authorization=auth,
        )
    bad = tmp_path / "bad_engine.toml"
    bad.write_text('scenario = "open_auction"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        run_ledger(
            cube=cube,
            targets={},
            panel_dir=panel,
            dividends=pl.DataFrame(),
            engine_config_path=bad,
            rules=rules,
            market_cache_root=tmp_path / "cache",
            capital_krw=1000,
            halted_exit_policy=DelistPolicy.LAST_CLOSE,
            start=sessions[0],
            end=sessions[-1],
            authorization=auth,
        )
    dividends = pl.DataFrame(
        [{"instrument_id": instruments[0], "ex_session": sessions[1], "pay_session": sessions[2], "dps_krw": 10}],
        schema={"instrument_id": pl.String, "ex_session": pl.Date, "pay_session": pl.Date, "dps_krw": pl.Int64},
    )
    nested_engine = tmp_path / "nested_engine.toml"
    nested_engine.write_text(
        '[execution]\nscenario = "open_auction"\nmax_participation = 0.1\ncarry_unfilled = false\n'
        '[costs]\ncommission_rate = "0.001"\ndividend_withholding_rate = "0.1"\nimpact_k = 0.1\n'
        "cash_buffer = 0.0\n",
        encoding="utf-8",
    )
    nested_outcome = run_ledger(
        cube=cube,
        targets={},
        panel_dir=panel,
        dividends=dividends,
        engine_config_path=nested_engine,
        rules=rules,
        market_cache_root=tmp_path / "cache",
        capital_krw=1000,
        halted_exit_policy=DelistPolicy.LAST_CLOSE,
        start=sessions[0],
        end=sessions[-1],
        authorization=auth,
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
                cube=cube,
                targets={},
                panel_dir=panel,
                dividends=pl.DataFrame(),
                engine_config_path=broken,
                rules=rules,
                market_cache_root=tmp_path / "cache",
                capital_krw=1000,
                halted_exit_policy=DelistPolicy.LAST_CLOSE,
                start=sessions[0],
                end=sessions[-1],
                authorization=auth,
            )
    outcome = run_ledger(
        cube=cube,
        targets={},
        panel_dir=panel,
        dividends=dividends,
        engine_config_path=ENGINE_TOML,
        rules=rules,
        market_cache_root=tmp_path / "cache",
        capital_krw=1000,
        halted_exit_policy=DelistPolicy.ZERO,
        start=sessions[0],
        end=sessions[-1],
        authorization=auth,
    )
    assert outcome.capital_krw == 1000
    assert outcome.halted_exit_policy == "zero"


def test_engine_parts_reads_optional_cost_keys(tmp_path: Path) -> None:
    """Auction ticks and yield tax default, parse, and fail closed."""
    from src.research.ledger_bridge import _engine_parts

    _, costs, _ = _engine_parts(ENGINE_TOML)
    assert costs.auction_slippage_ticks == 1.0
    assert str(costs.cash_yield_tax_rate) == "0.154"
    minimal = tmp_path / "minimal.toml"
    minimal.write_text(
        'scenario = "open_auction"\nmax_participation = 0.1\ncarry_unfilled = false\n'
        'commission_rate = "0.001"\ndividend_withholding_rate = "0.1"\nimpact_k = 0.1\n'
        "cash_buffer = 0.0\n",
        encoding="utf-8",
    )
    _, defaulted, _ = _engine_parts(minimal)
    assert defaulted.auction_slippage_ticks == 0.0
    assert str(defaulted.cash_yield_tax_rate) == "0.154"
    bad_slip = tmp_path / "bad_slip.toml"
    bad_slip.write_text(minimal.read_text(encoding="utf-8") + "auction_slippage_ticks = -1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"auction_slippage_ticks|cash_yield"):
        _engine_parts(bad_slip)
    bad_tax = tmp_path / "bad_tax.toml"
    bad_tax.write_text(minimal.read_text(encoding="utf-8") + 'cash_yield_tax_rate = "1.5"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cash_yield_tax_rate"):
        _engine_parts(bad_tax)
    nested = tmp_path / "nested_costs.toml"
    nested.write_text(
        '[execution]\nscenario = "open_auction"\nmax_participation = 0.1\ncarry_unfilled = false\n'
        '[costs]\ncommission_rate = "0.001"\ndividend_withholding_rate = "0.1"\nimpact_k = 0.1\n'
        'cash_buffer = 0.0\nauction_slippage_ticks = 2.0\ncash_yield_tax_rate = "0.10"\n',
        encoding="utf-8",
    )
    _, nested_costs, _ = _engine_parts(nested)
    assert nested_costs.auction_slippage_ticks == 2.0
    assert str(nested_costs.cash_yield_tax_rate) == "0.10"
    bad_nested_slip = tmp_path / "bad_nested_slip.toml"
    bad_nested_slip.write_text(
        '[execution]\nscenario = "open_auction"\nmax_participation = 0.1\ncarry_unfilled = false\n'
        '[costs]\ncommission_rate = "0.001"\ndividend_withholding_rate = "0.1"\nimpact_k = 0.1\n'
        'cash_buffer = 0.0\nauction_slippage_ticks = "lots"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"auction_slippage_ticks"):
        _engine_parts(bad_nested_slip)
    bad_nested_tax = tmp_path / "bad_nested_tax.toml"
    bad_nested_tax.write_text(
        '[execution]\nscenario = "open_auction"\nmax_participation = 0.1\ncarry_unfilled = false\n'
        '[costs]\ncommission_rate = "0.001"\ndividend_withholding_rate = "0.1"\nimpact_k = 0.1\n'
        'cash_buffer = 0.0\ncash_yield_tax_rate = "oops"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cash_yield_tax_rate"):
        _engine_parts(bad_nested_tax)


def test_run_ledger_cost_overrides(tmp_path: Path) -> None:
    """Stress overrides flow into CostConfig without changing the base TOML."""
    import numpy as _np

    sessions = synthetic_sessions(4, start=date(2020, 1, 6))
    instruments = ["KRX:000001", "KRX:000002"]
    rows = [synthetic_panel_row(day, inst) for day in sessions for inst in instruments]
    panel = write_synthetic_panel(tmp_path / "gold", "market_panel_override", rows)
    cube = synthetic_cube(sessions, instruments)
    auth = _auth(sessions)
    rules = load_krx_market_rules(RULES_PATH)
    base = run_ledger(
        cube=cube,
        targets={},
        panel_dir=panel,
        dividends=pl.DataFrame(),
        engine_config_path=ENGINE_TOML,
        rules=rules,
        market_cache_root=tmp_path / "cache",
        capital_krw=1000,
        halted_exit_policy=DelistPolicy.LAST_CLOSE,
        start=sessions[0],
        end=sessions[-1],
        authorization=auth,
    )
    stressed = run_ledger(
        cube=cube,
        targets={},
        panel_dir=panel,
        dividends=pl.DataFrame(),
        engine_config_path=ENGINE_TOML,
        rules=rules,
        market_cache_root=tmp_path / "cache",
        capital_krw=1000,
        halted_exit_policy=DelistPolicy.LAST_CLOSE,
        start=sessions[0],
        end=sessions[-1],
        authorization=auth,
        extra_slippage=0.01,
        auction_slippage_ticks=0.0,
        cash_returns=_np.zeros(len(sessions)),
    )
    assert base.ledger_hash
    assert stressed.ledger_hash


def test_market_arrays_bypass_the_panel_load(tmp_path: Path) -> None:
    """Injected arrays are used verbatim; the cube identity check still applies to them."""
    import numpy as _np

    from src.backtest.market import load_market_arrays

    sessions = synthetic_sessions(4, start=date(2020, 1, 6))
    instruments = ["KRX:000001", "KRX:000002"]
    panel = write_synthetic_panel(
        tmp_path / "gold", "market_panel_injected", [synthetic_panel_row(d, i) for d in sessions for i in instruments]
    )
    cube = synthetic_cube(sessions, instruments)
    auth = _auth(sessions)
    rules = load_krx_market_rules(RULES_PATH)
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    common = {
        "cube": cube,
        "targets": {0: _np.array([0.5, 0.5])},
        "panel_dir": panel,
        "dividends": pl.DataFrame(),
        "engine_config_path": ENGINE_TOML,
        "rules": rules,
        "market_cache_root": tmp_path / "cache",
        "capital_krw": 100_000_000,
        "halted_exit_policy": DelistPolicy.LAST_CLOSE,
        "start": sessions[0],
        "end": sessions[-1],
        "authorization": auth,
    }
    from_disk = run_ledger(**common)
    assert _np.any(np.asarray(from_disk.nav_krw) != from_disk.capital_krw)
    injected = run_ledger(**common, market_arrays=arrays)
    assert injected.ledger_hash == from_disk.ledger_hash

    doubled = MarketArrays(
        dataset_id=arrays.dataset_id,
        sessions=arrays.sessions,
        instrument_ids=arrays.instrument_ids,
        int_fields={
            name: block * 2 if name == "close" else block for name, block in arrays.int_fields.items()
        },
        float_fields=arrays.float_fields,
        bool_fields=arrays.bool_fields,
        market=arrays.market,
    )
    assert _np.any(_np.asarray(doubled.int_fields["close"]) != _np.asarray(arrays.int_fields["close"]))
    assert run_ledger(**common, market_arrays=doubled).ledger_hash != from_disk.ledger_hash

    shifted = MarketArrays(
        dataset_id=arrays.dataset_id,
        sessions=arrays.sessions[:-1],
        instrument_ids=arrays.instrument_ids,
        int_fields=arrays.int_fields,
        float_fields=arrays.float_fields,
        bool_fields=arrays.bool_fields,
        market=arrays.market,
    )
    with pytest.raises(PITDataError, match="sessions or instruments differ"):
        run_ledger(**common, market_arrays=shifted)


def test_overlay_market_preserves_nans() -> None:
    import numpy as _np

    from src.research.hedge import HedgeInputs
    from src.research.ledger_bridge import overlay_market_from_inputs

    days = synthetic_sessions(3)
    market = overlay_market_from_inputs(
        HedgeInputs(
            sessions=tuple(days),
            index_level=_np.array([1.0, float("nan"), 3.0]),
            inverse_close=_np.array([float("nan"), 2.0, 3.0]),
        )
    )
    assert _np.isnan(market.index_level[1])
    assert _np.isnan(market.inverse_close[0])


def test_cash_returns_gaps_and_validation() -> None:
    from datetime import timedelta

    import numpy as _np

    from src.core.pit import PITDataError as _PIT
    from src.research.ledger_bridge import cash_returns_from_frame

    base = date(2020, 1, 6)
    days = [base + timedelta(days=i) for i in range(4)]
    frame = pl.DataFrame(
        {"session": [days[0], days[1], days[3]], "cash_close": [10000, 10100, 10200]},
        schema={"session": pl.Date, "cash_close": pl.Int64},
    )
    out = cash_returns_from_frame(frame, sessions=days)
    assert _np.isnan(out[0])
    assert out[1] == 10100 / 10000 - 1.0
    assert _np.isnan(out[2])
    assert _np.isnan(out[3])
    nulled = pl.DataFrame(
        {"session": [days[0], days[1]], "cash_close": [10000, None]},
        schema={"session": pl.Date, "cash_close": pl.Int64},
    )
    assert _np.isnan(cash_returns_from_frame(nulled, sessions=days[:2])[1])
    with pytest.raises(_PIT, match="missing columns"):
        cash_returns_from_frame(pl.DataFrame({"session": days}), sessions=days)
    with pytest.raises(_PIT, match="duplicate"):
        cash_returns_from_frame(pl.concat([frame.head(1), frame.head(1)]), sessions=days)
    with pytest.raises(_PIT, match="non-positive"):
        cash_returns_from_frame(
            pl.DataFrame(
                {"session": days[:1], "cash_close": [0]},
                schema={"session": pl.Date, "cash_close": pl.Int64},
            ),
            sessions=days[:1],
        )


def test_ledger_outcome_diagnostics_and_overlay(tmp_path: Path) -> None:
    import numpy as _np

    sessions = synthetic_sessions(4, start=date(2020, 1, 6))
    instruments = ["KRX:000001", "KRX:000002"]
    rows = [synthetic_panel_row(day, inst) for day in sessions for inst in instruments]
    panel = write_synthetic_panel(tmp_path / "gold", "market_panel_diag", rows)
    cube = synthetic_cube(sessions, instruments)
    auth = _auth(sessions)
    rules = load_krx_market_rules(RULES_PATH)
    outcome = run_ledger(
        cube=cube,
        targets={0: _np.array([1.0, 0.0])},
        panel_dir=panel,
        dividends=pl.DataFrame(),
        engine_config_path=ENGINE_TOML,
        rules=rules,
        market_cache_root=tmp_path / "cache",
        capital_krw=10_000_000,
        halted_exit_policy=DelistPolicy.LAST_CLOSE,
        start=sessions[0],
        end=sessions[-1],
        authorization=auth,
    )
    assert outcome.stock_book_returns.shape == outcome.log_returns.shape
    assert outcome.journal_totals_krw["commission"] < 0
    assert outcome.avg_stock_exposure > 0.0
    assert outcome.avg_margin_share == 0.0
    assert outcome.avg_inverse_share == 0.0
    assert outcome.turnover_per_year > 0.0
    from src.research.ledger_bridge import LedgerOutcome

    legacy = LedgerOutcome(
        capital_krw=1,
        halted_exit_policy="last_close",
        sessions=tuple(sessions),
        log_returns=_np.zeros(4),
        ledger_hash="h",
    )
    assert legacy.reject_counts == {}
    assert legacy.journal_totals_krw == {}
    assert legacy.stock_book_returns.shape == (4,)

    from src.research.hedge import HedgeInputs, HedgeSpec, BetaNeutralOverlay, derivative_config
    from src.research.ledger_bridge import overlay_market_from_inputs

    rng = _np.random.default_rng(0)
    index = _np.linspace(1000.0, 1010.0, len(sessions))
    inverse = _np.linspace(9000.0, 8990.0, len(sessions))
    inputs = HedgeInputs(
        sessions=tuple(sessions),
        index_level=_np.ascontiguousarray(index),
        inverse_close=_np.ascontiguousarray(inverse),
    )
    hedge_spec = HedgeSpec(
        hedge_ratio=1.0,
        beta_window_sessions=2,
        beta_min_sessions=2,
        beta_cap=2.0,
        rebalance_every_sessions=1,
        use_futures=True,
        contract_multiplier_krw=10000,
        initial_margin_rate=0.2175,
        margin_buffer_rate=0.10,
        margin_topup_trigger_fraction=0.75,
        futures_cost_rate=0.0,
        inverse_cost_rate=0.0,
        resize_sell_cost_rate=0.0,
        resize_buy_cost_rate=0.0,
        futures_tax_rate=0.0,
        futures_annual_deduction_krw=2500000,
        inverse_tax_rate=0.0,
    )
    hedged = run_ledger(
        cube=cube,
        targets={0: _np.array([0.5, 0.5])},
        panel_dir=panel,
        dividends=pl.DataFrame(),
        engine_config_path=ENGINE_TOML,
        rules=rules,
        market_cache_root=tmp_path / "cache",
        capital_krw=10_000_000,
        halted_exit_policy=DelistPolicy.LAST_CLOSE,
        start=sessions[0],
        end=sessions[-1],
        authorization=auth,
        overlay=BetaNeutralOverlay(hedge_spec, rebalance_offset=0),
        overlay_market=overlay_market_from_inputs(inputs),
        derivatives=derivative_config(hedge_spec),
    )
    assert hedged.sessions == outcome.sessions
    assert rng is not None
