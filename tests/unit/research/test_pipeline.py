"""Account-engine evaluation pipeline invariants (protocol v4)."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.data.research_protocol import WindowError
from src.research.cube import ResearchCube

_NOW = datetime(2026, 9, 30, tzinfo=UTC)
INSTRS = [f"KRX:{i:06d}" for i in range(1, 7)]


def _sessions(n: int = 80) -> list[date]:
    from tests.fixtures.synthetic_panel import synthetic_sessions

    return synthetic_sessions(n, start=date(2019, 1, 1))


def _multi_year_sessions(n: int = 1300) -> list[date]:
    """Panel wide enough for a first test year that has a preceding test year, so the causal perturbation
    has cut rows to work with."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    return synthetic_sessions(n, start=date(2018, 1, 1))


def _full_cube(sessions: list[date], *, close: float = 10000.0) -> ResearchCube:
    from tests.fixtures.synthetic_panel import synthetic_cube

    base = synthetic_cube(sessions, INSTRS, close=close)
    arrays = dict(base.arrays)
    n_s, n_n = len(sessions), len(INSTRS)
    rng = np.random.default_rng(7)
    rets = rng.normal(loc=0.001, scale=0.004, size=(n_s, n_n))
    trend = np.cumprod(1.0 + rets, axis=0) / (1.0 + rets[0])
    arrays["adj_tr"] = np.ascontiguousarray(trend)
    arrays["adj_px"] = np.ascontiguousarray(trend * close)
    arrays["high"] = np.ascontiguousarray(np.full((n_s, n_n), close * 1.01))
    arrays["low"] = np.ascontiguousarray(np.full((n_s, n_n), close * 0.99))
    arrays["trading_value"] = np.ascontiguousarray(
        np.asarray(arrays["volume"], dtype=float) * np.asarray(arrays["close"], dtype=float)
    )
    arrays["ret_cc"] = np.ascontiguousarray(np.zeros((n_s, n_n)))
    for name in (
        "f_age_q",
        "f_equity",
        "f_net_income_ttm",
        "f_assets",
        "f_operating_profit_ttm",
        "f_gross_profit_ttm",
        "f_operating_profit_q",
        "f_operating_profit_q_ly",
        "f_net_income_q",
        "f_net_income_q_ly",
        "f_assets_ly",
        "f_operating_cash_flow_ttm",
        "f_sales_ttm",
        "f_sales_ttm_ly",
        "earn_qk",
        "earn_operating_profit_q",
        "earn_operating_profit_q_ly",
        "earn_net_income_q",
        "earn_net_income_q_ly",
        "earn_sales_q",
        "earn_sales_q_ly",
        "earn_avail_t",
        "flow_for_krw",
        "flow_ins_krw",
        "flow_ind_krw",
    ):
        if name == "f_age_q":
            arrays[name] = np.ascontiguousarray(np.zeros((n_s, n_n)))
        else:
            arrays[name] = np.ascontiguousarray(np.full((n_s, n_n), np.nan))
    drift_on = np.asarray(arrays["r_on"]).copy()
    drift_id = np.asarray(arrays["r_id"]).copy()
    drift_on[:, -1] = 0.01
    drift_id[:, -1] = 0.005
    arrays["r_on"] = np.ascontiguousarray(drift_on)
    arrays["r_id"] = np.ascontiguousarray(drift_id)
    return ResearchCube.from_arrays(
        cube_id=base.cube_id,
        sessions=list(sessions),
        instrument_ids=list(INSTRS),
        arrays=arrays,
        exit_at=np.asarray(base.exit_at),
        exit_halted=np.asarray(base.exit_halted),
    )


def _protocol_for() -> Any:
    from src.data.research_protocol import load_research_protocol
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = load_research_protocol(Path("config/research/protocol.toml"), scope)
    evaluation = base.evaluation.model_copy(update={"draws": 50, "block_sessions": 5, "horizon_sessions": 30})
    return base.model_copy(update={"evaluation": evaluation})


def _spec(**overrides: Any) -> Any:
    from src.research.book import BookSpec
    from src.research.hedge import HedgeSpec
    from src.research.pipeline import StrategySpec
    from src.research.policy import TrendCashPolicy
    from src.research.model import ScorerConfig

    policy_kw: dict[str, Any] = {"rebalance_every_sessions": 5, "min_units_per_slot": 0, "n": 2}
    scorer_kw: dict[str, Any] = {
        "first_test_year": 2019,
        "min_train_rows": 1,
        "min_cross_section": 1,
        "num_boost_round": 2,
        "min_data_in_leaf": 2,
        "num_threads": 1,
    }
    book_kw: dict[str, Any] = {"sleeves": 5, "stock_capital_fraction": 0.75}
    hedge_kw: dict[str, Any] = {
        "hedge_ratio": 1.0,
        "beta_window_sessions": 10,
        "beta_min_sessions": 2,
        "beta_cap": 2.0,
        "rebalance_every_sessions": 5,
        "use_futures": True,
        "contract_multiplier_krw": 10000,
        "initial_margin_rate": 0.2175,
        "margin_buffer_rate": 0.10,
        "margin_topup_trigger_fraction": 0.75,
        "futures_cost_rate": 0.0003,
        "inverse_cost_rate": 0.0007,
        "resize_sell_cost_rate": 0.0025,
        "resize_buy_cost_rate": 0.0005,
        "futures_tax_rate": 0.11,
        "futures_annual_deduction_krw": 2500000,
        "inverse_tax_rate": 0.154,
    }
    policy_kw.update(overrides.pop("policy", {}))
    scorer_kw.update(overrides.pop("scorer", {}))
    book_kw.update(overrides.pop("book", {}))
    hedge_kw.update(overrides.pop("hedge", {}))
    policy = TrendCashPolicy(**policy_kw)  # type: ignore[arg-type]
    scorer = ScorerConfig(**scorer_kw)  # type: ignore[arg-type]
    return StrategySpec(
        policy=policy,
        scorer=scorer,
        book=BookSpec(**book_kw),
        hedge=HedgeSpec(**hedge_kw),
        trend_overlay=overrides.get("trend_overlay"),
        regime_hedge=overrides.get("regime_hedge"),
    )  # type: ignore[arg-type]


def _panel_rows(sessions: list[date]) -> list[dict[str, Any]]:
    """A rising panel wide enough that a shifted close is a different market, not the same one shifted."""
    from tests.fixtures.synthetic_panel import synthetic_panel_row

    rows: list[dict[str, Any]] = []
    for pos, day in enumerate(sessions):
        price = 10_000 + 20 * pos
        rows.extend(
            synthetic_panel_row(
                day,
                inst,
                open=price,
                high=price + 100,
                low=price - 100,
                close=price,
                base_price=price,
                upper_limit=price * 2,
                lower_limit=price // 2,
            )
            for inst in INSTRS
        )
    return rows


def _context(
    tmp_path: Path,
    sessions: list[date],
    cube: ResearchCube | None = None,
    *,
    trending_index: bool = False,
    dividends: pl.DataFrame | None = None,
    dataset_ids: dict[str, str] | None = None,
) -> Any:
    from src.core.market_rules import load_krx_market_rules
    from src.data.research_protocol import WindowGuard
    from src.research.pipeline import Pipeline, PipelineContext
    from src.research.registry import RunRegistry
    from tests.fixtures.synthetic_panel import write_synthetic_panel

    protocol = _protocol_for()
    cube = cube if cube is not None else _full_cube(sessions)
    registry = RunRegistry(tmp_path / "runs")
    guard = WindowGuard(protocol=protocol, last_session=sessions[-1])
    from tests.fixtures.synthetic_panel import synthetic_hedge_inputs

    hedge_inputs = _trending_hedge_inputs(sessions) if trending_index else synthetic_hedge_inputs(sessions)
    panel = write_synthetic_panel(tmp_path / "gold", "market_panel_test", _panel_rows(sessions))
    ids = {
        "market_panel": "market_panel_aaaaaaaaaaaaaaa",
        "dividend_events": "dividend_events_bbbbbbbbbbbbbbb",
        "hedge_series": "hedge_series_cccccccccccccccc",
        "cash_series": "cash_series_dddddddddddddddd",
    }
    ctx = PipelineContext(
        protocol=protocol,
        cube=cube,
        registry=registry,
        guard=guard,
        panel_dir=panel,
        dividends=pl.DataFrame() if dividends is None else dividends,
        hedge_inputs=hedge_inputs,
        rules=load_krx_market_rules(Path("config/market/krx_market_rules.toml")),
        engine_config_path=Path("config/backtest/default_engine.toml"),
        market_cache_root=tmp_path / "mcache",
        reports_root=tmp_path / "reports",
        scores_root=tmp_path / "scores",
        ledger_runner=_ledger_ok(sessions),
        now=lambda: _NOW,
        dataset_ids=ids if dataset_ids is None else dataset_ids,
        cash_returns=np.zeros(len(sessions), dtype=np.float64),
    )
    return Pipeline(ctx)


def _ledger_ok(sessions: list[date]) -> Any:
    from src.research.ledger_bridge import LedgerOutcome

    def _fake(**kwargs: Any) -> LedgerOutcome:
        start, end = kwargs["start"], kwargs["end"]
        idx = {d: i for i, d in enumerate(sessions)}
        window = tuple(sessions[idx[start] : idx[end] + 1])
        drift = 0.0008
        if kwargs.get("extra_slippage"):
            drift -= 0.0002
        if kwargs.get("auction_slippage_ticks"):
            drift -= 0.0001 * float(kwargs["auction_slippage_ticks"])
        return LedgerOutcome(
            capital_krw=kwargs["capital_krw"],
            halted_exit_policy=kwargs["halted_exit_policy"].value,
            sessions=window,
            log_returns=np.full(len(window), drift),
            stock_book_returns=np.full(len(window), drift * 0.9),
            reject_counts={},
            journal_totals_krw={"commission": -1000},
            avg_stock_exposure=0.8,
            avg_margin_share=0.1,
            avg_inverse_share=0.05,
            turnover_per_year=2.0,
            ledger_hash="h",
        )

    return _fake


def _trending_hedge_inputs(sessions: list[date]) -> Any:
    """Random-walk index level and inverse-ETF close, so the overlay actually opens a hedge position.

    Aligned to the panel sessions exactly, like ``hedge_inputs_from_frame``; the engine overlay market must
    line up row for row with the panel it prices against.
    """
    from src.research.hedge import HedgeInputs

    rng = np.random.default_rng(11)
    steps = rng.normal(loc=0.0004, scale=0.01, size=len(sessions))
    level = 1500.0 * np.exp(np.cumsum(steps))
    inverse = 8000.0 * np.exp(-np.cumsum(steps))
    return HedgeInputs(
        sessions=tuple(sessions),
        index_level=np.ascontiguousarray(level),
        inverse_close=np.ascontiguousarray(inverse),
    )


def _causal_account(sessions: list[date], *, leak_from_index: bool = False) -> Any:
    """Fake account engine that actually prices the book and asks the overlay for a target every session.

    A target decided at row ``r`` moves NAV from row ``r + 1`` on and the overlay only ever reads the state
    up to its own row, so NAV and overlay decisions at rows ``<= cut`` cannot react to anything after ``cut``.

    ``leak_from_index`` reproduces an engine that reads the whole index series before the run: its NAV drifts
    once the market it was handed has been corrupted past the cut, and it hands the overlay the full series
    instead of the prefix — the leak the perturbation test has to catch.
    """
    from src.backtest.overlay import OverlayState
    from src.research.ledger_bridge import LedgerOutcome

    position = {day: pos for pos, day in enumerate(sessions)}
    multiplier = 10_000.0
    pristine: np.ndarray | None = None

    def _fake(**kwargs: Any) -> LedgerOutcome:
        nonlocal pristine
        window = tuple(sessions[position[kwargs["start"]] : position[kwargs["end"]] + 1])
        targets = {int(r): np.asarray(w, dtype=np.float64) for r, w in kwargs["targets"].items()}
        market = np.asarray(kwargs["overlay_market"].index_level, dtype=np.float64)
        if pristine is None:
            pristine = market.copy()
        leak = 0.001 if leak_from_index and not np.array_equal(market, pristine) else 0.0
        overlay = kwargs.get("overlay")
        rows = np.asarray([position[day] for day in window], dtype=np.int64)
        levels = np.asarray(market[rows], dtype=np.float64)
        market_returns = np.zeros(levels.shape[0], dtype=np.float64)
        market_returns[1:] = levels[1:] / levels[:-1] - 1.0
        breadth = np.asarray(
            [
                0.0 if (weights := targets.get(int(rows[pos]) - 1)) is None else float(np.count_nonzero(weights > 0.0))
                for pos in range(len(window))
            ],
            dtype=np.float64,
        )
        book_full = 0.0002 + 0.6 * market_returns + 0.0001 * breadth + leak
        stock_returns = np.zeros(len(window), dtype=np.float64)
        index_returns = np.zeros(len(window), dtype=np.float64)
        logs = np.empty(len(window), dtype=np.float64)
        navs = np.empty(len(window), dtype=np.float64)
        prev = float(kwargs["capital_krw"])
        contracts = 0
        for pos in range(len(window)):
            book = float(book_full[pos])
            pnl = -contracts * multiplier * (levels[pos] - levels[pos - 1]) if pos else 0.0
            nav = prev * (1.0 + book) + pnl
            stock_returns[pos] = book
            index_returns[pos] = market_returns[pos]
            logs[pos] = math.log(nav / prev)
            navs[pos] = nav
            if overlay is not None:
                decision = overlay.target(
                    OverlayState(
                        session_idx=int(rows[pos]),
                        nav=int(nav),
                        stock_book_nav=int(nav),
                        stock_book_returns=book_full if leak_from_index else stock_returns[: pos + 1],
                        index_returns=market_returns if leak_from_index else index_returns[: pos + 1],
                        index_level=float(levels[pos]),
                        contracts=contracts,
                        inverse_units=0,
                    )
                )
                if decision is not None:
                    contracts = int(decision.contracts)
            prev = nav
        return LedgerOutcome(
            capital_krw=kwargs["capital_krw"],
            halted_exit_policy=kwargs["halted_exit_policy"].value,
            sessions=window,
            log_returns=logs,
            stock_book_returns=logs * 0.9,
            reject_counts={},
            journal_totals_krw={"commission": -1000},
            avg_stock_exposure=0.8,
            avg_margin_share=0.1,
            avg_inverse_share=0.05,
            turnover_per_year=2.0,
            ledger_hash="h",
            nav_krw=navs,
        )

    return _fake


def _clean_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.pipeline as pipe

    def _fake(panel: Any, universe: Any, sessions: Any, config: Any, *, test_years: Any, authorization: Any) -> Any:
        from src.research.model import ScoreMatrix

        n_rows = next(iter(panel.features.values())).shape[0]
        n_n = next(iter(panel.features.values())).shape[1]
        uni = np.asarray(universe, dtype=bool)
        out = np.full((n_rows, n_n), np.nan)
        sess = list(sessions)[:n_rows]
        for r, day in enumerate(sess):
            if day.year in set(test_years) and bool(uni[r].any()):
                out[r] = np.array([float(n) for n in range(n_n)])
        arr = np.ascontiguousarray(out, dtype=np.float32)
        arr.flags.writeable = False
        return ScoreMatrix(
            scores=arr, test_years=tuple(test_years), config_hash=config.config_hash, last_row=panel.last_row
        )

    monkeypatch.setattr(pipe, "walk_forward_scores", _fake)


def _leaky_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.pipeline as pipe

    def _fake(panel: Any, universe: Any, sessions: Any, config: Any, *, test_years: Any, authorization: Any) -> Any:
        from src.research.model import ScoreMatrix

        n_rows = next(iter(panel.features.values())).shape[0]
        n_n = next(iter(panel.features.values())).shape[1]
        uni = np.asarray(universe, dtype=bool)
        dev = np.asarray(panel.features["dev_ma20"], dtype=float)
        out = np.full((n_rows, n_n), np.nan)
        sess = list(sessions)[:n_rows]
        for r, day in enumerate(sess):
            if day.year in set(test_years) and bool(uni[r].any()) and r + 1 < n_rows:
                out[r] = np.asarray(dev[r + 1], dtype=float)
        arr = np.ascontiguousarray(out, dtype=np.float32)
        arr.flags.writeable = False
        return ScoreMatrix(
            scores=arr, test_years=tuple(test_years), config_hash=config.config_hash, last_row=panel.last_row
        )

    monkeypatch.setattr(pipe, "walk_forward_scores", _fake)


def test_evaluate_runs_every_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    calls: list[dict[str, Any]] = []
    runner = pipe._ctx.ledger_runner

    def _spy(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return runner(**kwargs)

    object.__setattr__(pipe._ctx, "ledger_runner", _spy)
    run = pipe.evaluate(spec)
    ticks = list(pipe._ctx.protocol.scenarios.cost_grid_ticks)
    assert len(calls) == 3 + len(ticks) + 2
    assert all(call["cash_returns"] is not None for call in calls)
    assert all(call["overlay"] is not None for call in calls)
    assert any(float(call["extra_slippage"]) > 0.0 for call in calls)
    grid_slips = sorted(
        float(call["auction_slippage_ticks"]) for call in calls if call["auction_slippage_ticks"] is not None
    )
    assert grid_slips == sorted(float(t) for t in ticks)
    assert run.report.passed is True
    assert run.evidence.sessions[0] == next(d for d in sessions if d.year == 2019)


def test_delay_scenario_shifts_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    seen: dict[str, dict[int, Any]] = {}
    runner = pipe._ctx.ledger_runner

    def _spy(**kwargs: Any) -> Any:
        overlay = kwargs["overlay"]
        delay = int(getattr(overlay, "_delay", 0))
        extra = float(kwargs.get("extra_slippage") or 0.0)
        slip = kwargs.get("auction_slippage_ticks")
        if delay > 0:
            seen["delay"] = dict(kwargs["targets"])
        elif extra > 0.0 or slip is not None:
            pass
        else:
            seen["base"] = dict(kwargs["targets"])
        return runner(**kwargs)

    object.__setattr__(pipe._ctx, "ledger_runner", _spy)
    pipe.evaluate(spec)
    delay_n = int(pipe._ctx.protocol.scenarios.stress_delay_sessions)
    bound = max(seen["base"])
    assert set(seen["delay"]) == {r + delay_n for r in seen["base"] if r + delay_n <= bound}
    assert any(r not in seen["delay"] for r in seen["base"])


def test_placebo_is_seeded_and_model_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    counter = _score_counter(monkeypatch)
    spec = _spec()
    per_run: list[list[dict[int, Any]]] = []
    runner = pipe._ctx.ledger_runner

    def _spy(**kwargs: Any) -> Any:
        per_run[-1].append(dict(kwargs["targets"]))
        return runner(**kwargs)

    object.__setattr__(pipe._ctx, "ledger_runner", _spy)
    per_run.append([])
    pipe.evaluate(spec)
    per_run.append([])
    pipe.evaluate(spec)
    assert counter.calls == 2
    assert len(per_run[0]) == len(per_run[1])
    first_placebo, second_placebo = per_run[0][-1], per_run[1][-1]
    assert set(first_placebo) == set(second_placebo)
    for row in first_placebo:
        assert np.array_equal(np.asarray(first_placebo[row]), np.asarray(second_placebo[row]))


def _score_counter(monkeypatch: pytest.MonkeyPatch) -> Any:
    import src.research.pipeline as pipe_mod

    real = pipe_mod.Pipeline.scores

    class _Counter:
        calls = 0

    def _spy(self: Any, spec: Any, *, panel: Any = None) -> Any:
        _Counter.calls += 1
        return real(self, spec, panel=panel)

    monkeypatch.setattr(pipe_mod.Pipeline, "scores", _spy)
    _clean_scores(monkeypatch)
    return _Counter


def test_registry_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    run = pipe.evaluate(spec)
    records = pipe._ctx.registry.runs(run_id=run.report.run_id)
    ticks = list(pipe._ctx.protocol.scenarios.cost_grid_ticks)
    assert len(records) == 3 + len(ticks) + 2
    assert {r.scenario for r in records} == (
        {"base", "stress_slippage", "stress_delay", "unhedged", "placebo"} | {f"cost_{t}" for t in ticks}
    )
    assert all(r.report_digest == run.report.digest for r in records)
    assert all(r.protocol_hash == pipe._ctx.protocol.content_hash for r in records)
    lines_before = len((tmp_path / "runs" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines())
    rerun = pipe.evaluate(spec)
    assert rerun.report.run_id == run.report.run_id
    assert rerun.report.digest == run.report.digest
    lines_after = len((tmp_path / "runs" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines())
    assert lines_after == lines_before


def test_report_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    run = pipe.evaluate(spec)
    path = tmp_path / "reports" / f"{spec.spec_hash}_{run.report.run_id}.json"
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["digest"] == run.report.digest


def test_leaky_scorer_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _multi_year_sessions()
    pipe = _context(tmp_path, sessions)
    _leaky_scores(monkeypatch)
    run = pipe.evaluate(_spec())
    by = {c.name: c for c in run.report.integrity}
    assert by["perturbation_mismatches"].passed is False
    assert run.report.passed is False


def test_perturbation_detects_an_account_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _multi_year_sessions()
    pipe = _context(tmp_path, sessions, trending_index=True)
    _clean_scores(monkeypatch)
    object.__setattr__(pipe._ctx, "ledger_runner", _causal_account(sessions, leak_from_index=True))
    run = pipe.evaluate(_spec())
    assert run.evidence.perturbation_mismatches > 0
    by = {c.name: c for c in run.report.integrity}
    assert by["perturbation_mismatches"].passed is False
    assert run.report.passed is False


def test_causal_account_passes_the_perturbation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.pipeline as pipe_mod

    sessions = _multi_year_sessions()
    pipe = _context(tmp_path, sessions, trending_index=True)
    _clean_scores(monkeypatch)
    clean = pipe_mod.walk_forward_scores
    rescores: list[int] = []

    def _counting(*args: Any, **kwargs: Any) -> Any:
        if len(kwargs["test_years"]) == 1:
            rescores.append(kwargs["test_years"][0])
        return clean(*args, **kwargs)

    monkeypatch.setattr(pipe_mod, "walk_forward_scores", _counting)
    object.__setattr__(pipe._ctx, "ledger_runner", _causal_account(sessions))
    run = pipe.evaluate(_spec())
    assert rescores
    assert len(rescores) == int(pipe._ctx.protocol.scenarios.perturbation_cuts)
    assert run.evidence.perturbation_mismatches == 0
    assert {c.name: c for c in run.report.integrity}["perturbation_mismatches"].passed is True


def test_perturbation_never_passes_vacuously(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.pipeline as pipe_mod

    sessions = _multi_year_sessions()
    pipe = _context(tmp_path, sessions)
    counter = _score_counter(monkeypatch)
    clean = pipe_mod.walk_forward_scores

    def _rescore_fails(*args: Any, **kwargs: Any) -> Any:
        if len(kwargs["test_years"]) == 1:
            raise RuntimeError("rescore failed")
        return clean(*args, **kwargs)

    monkeypatch.setattr(pipe_mod, "walk_forward_scores", _rescore_fails)
    with pytest.raises(RuntimeError, match="rescore failed"):
        pipe.evaluate(_spec())
    assert counter.calls == 1
    assert not list((tmp_path / "reports").glob("*.json"))


def test_capital_is_part_of_the_run_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    small = pipe.evaluate(spec, capital_krw=10_000_000)
    large = pipe.evaluate(spec, capital_krw=100_000_000)
    assert small.report.run_id != large.report.run_id
    records = pipe._ctx.registry.runs()
    assert {record.run_id for record in records} == {small.report.run_id, large.report.run_id}
    assert {record.capital_krw for record in records} == {10_000_000, 100_000_000}
    ticks = list(pipe._ctx.protocol.scenarios.cost_grid_ticks)
    per_run = 3 + len(ticks) + 2
    assert len(pipe._ctx.registry.runs(run_id=small.report.run_id)) == per_run
    assert len(pipe._ctx.registry.runs(run_id=large.report.run_id)) == per_run
    written = sorted(path.name for path in (tmp_path / "reports").glob("*.json"))
    assert written == sorted(
        [
            f"{spec.spec_hash}_{large.report.run_id}.json",
            f"{spec.spec_hash}_{small.report.run_id}.json",
        ]
    )


def test_non_primary_capital_keeps_integrity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # close * min_units_per_slot > sleeve slot at 10M, so the affordability filter really binds there.
    sessions = _multi_year_sessions()
    pipe = _context(tmp_path, sessions, _full_cube(sessions, close=400_000.0), trending_index=True)
    _clean_scores(monkeypatch)
    object.__setattr__(pipe._ctx, "ledger_runner", _causal_account(sessions))
    run = pipe.evaluate(_spec(policy={"min_units_per_slot": 3}), capital_krw=10_000_000)
    assert run.evidence.perturbation_mismatches == 0
    assert pipe._sim_config(10_000_000).capital_krw == 10_000_000
    assert {r.capital_krw for r in pipe._ctx.registry.runs(run_id=run.report.run_id)} == {10_000_000}


def test_dataset_rebuild_is_a_new_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    spec = _spec()
    _clean_scores(monkeypatch)
    first = _context(tmp_path / "a", sessions)
    original = first.evaluate(spec)
    second = _context(
        tmp_path / "b",
        sessions,
        dataset_ids=dict(first._ctx.dataset_ids, cash_series="cash_series_eeeeeeeeeeeeeeee"),
    )
    refreshed = second.evaluate(spec)

    assert original.report.run_id != refreshed.report.run_id
    assert original.report.objective_j == refreshed.report.objective_j
    assert {r.run_id for r in first._ctx.registry.runs()} == {original.report.run_id}
    assert {r.run_id for r in second._ctx.registry.runs()} == {refreshed.report.run_id}
    assert sorted(path.name for path in (tmp_path / "a" / "reports").glob("*.json")) == [
        f"{spec.spec_hash}_{original.report.run_id}.json"
    ]


def test_run_identity_requires_every_dataset_id() -> None:
    from src.research.pipeline import _evaluation_run_id

    with pytest.raises(ValueError, match="dataset_ids are missing"):
        _evaluation_run_id(
            spec=_spec(),
            capital_krw=100_000_000,
            protocol_hash="p",
            cube_id="c",
            dataset_ids={"market_panel": "market_panel_0"},
            start=date(2020, 1, 2),
            end=date(2020, 2, 2),
            engine_config_bytes=b"engine",
        )


def test_perturbation_reaches_the_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An account that values row ``t`` with the engine close of row ``t+1`` must fail the integrity check.

    The bug is injected as a ledger runner that hands the real ``run_ledger`` a patched ``MarketArrays``, so the
    detection runs through the production account engine rather than a stand-in, and the corrupted rows the
    pipeline replays are the ones the engine actually reads.
    """
    from src.backtest.market import MarketArrays
    from src.research.ledger_bridge import run_ledger

    sessions = _multi_year_sessions(800)
    pipe = _context(tmp_path, sessions, trending_index=True)
    _clean_scores(monkeypatch)

    def _tomorrows_close(**kwargs: Any) -> Any:
        arrays = kwargs.pop("market_arrays")
        close = np.asarray(arrays.int_fields["close"], dtype=np.int64)
        shifted = close.copy()
        shifted[:-1] = close[1:]
        return run_ledger(
            **kwargs,
            market_arrays=MarketArrays(
                dataset_id=arrays.dataset_id,
                sessions=arrays.sessions,
                instrument_ids=arrays.instrument_ids,
                int_fields={**arrays.int_fields, "close": np.ascontiguousarray(shifted)},
                float_fields=arrays.float_fields,
                bool_fields=arrays.bool_fields,
                market=arrays.market,
            ),
        )

    object.__setattr__(pipe._ctx, "ledger_runner", _tomorrows_close)
    run = pipe.evaluate(_spec())
    assert run.evidence.perturbation_mismatches > 0
    assert run.report.passed is False


def test_engine_input_corruption_respects_the_cut() -> None:
    from src.backtest.market import MarketArrays
    from src.research.pipeline import _corrupt_dividends, _corrupt_market_arrays

    sessions = _sessions(6)
    shape = (len(sessions), 3)
    ints = {
        "close": np.full(shape, 10_000, dtype=np.int64),
        "volume": np.zeros(shape, dtype=np.int64),
    }
    floats = {"adtv20": np.full(shape, 1.0e12, dtype=np.float64)}
    present = np.zeros(shape, dtype=bool)
    present[4:] = True
    arrays = MarketArrays(
        dataset_id="market_panel_0",
        sessions=tuple(sessions),
        instrument_ids=("a", "b", "c"),
        int_fields=ints,
        float_fields=floats,
        bool_fields={"present": present, "entry_blocked": np.zeros(shape, dtype=bool)},
        market=np.ones(shape, dtype=np.int8),
    )
    corrupted = _corrupt_market_arrays(arrays, 2, np.random.default_rng(3))
    assert np.array_equal(corrupted.int_fields["close"][:3], ints["close"][:3])
    assert np.array_equal(corrupted.int_fields["volume"][:3], ints["volume"][:3])
    assert np.array_equal(corrupted.float_fields["adtv20"][:3], floats["adtv20"][:3])
    assert not np.array_equal(corrupted.int_fields["close"][3:], ints["close"][3:])
    assert np.all(corrupted.int_fields["close"][3:] > 0)
    assert np.array_equal(corrupted.int_fields["volume"][3:], ints["volume"][3:])
    assert np.array_equal(corrupted.bool_fields["present"][:3], present[:3])
    assert corrupted.bool_fields["present"][3:].all()
    assert not np.array_equal(corrupted.bool_fields["entry_blocked"][3:], arrays.bool_fields["entry_blocked"][3:])

    dividends = pl.DataFrame(
        {
            "instrument_id": ["a", "a", "a"],
            "ex_session": [sessions[0], sessions[2], sessions[4]],
            "pay_session": [sessions[1], sessions[3], sessions[5]],
            "dps_krw": [10, 20, 30],
        }
    )
    scaled = _corrupt_dividends(dividends, sessions[2], np.random.default_rng(4))
    assert scaled["dps_krw"].to_list()[0] == 10
    assert scaled["dps_krw"].to_list()[1] == 20
    assert scaled["dps_krw"].to_list()[2] > 30
    assert _corrupt_dividends(pl.DataFrame(), sessions[0], np.random.default_rng(5)).height == 0


def test_data_end_follows_the_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    cash = np.zeros(len(sessions), dtype=np.float64)
    cash[-3:] = np.nan
    object.__setattr__(pipe._ctx, "cash_returns", cash)
    run = pipe.evaluate(_spec())
    assert run.report.end == sessions[-4]
    assert run.evidence.sessions[-1] == sessions[-4]


def test_internal_gap_fails_before_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.pit import PITDataError

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    counter = _score_counter(monkeypatch)
    cash = np.zeros(len(sessions), dtype=np.float64)
    cash[30] = np.nan
    object.__setattr__(pipe._ctx, "cash_returns", cash)
    with pytest.raises(PITDataError, match=sessions[30].isoformat()):
        pipe.evaluate(_spec())
    assert counter.calls == 0
    assert not list((tmp_path / "reports").glob("*.json"))


def test_index_level_gap_fails_before_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An index-level hole inside the window names its session, together with any inverse-close gap."""
    from src.core.pit import PITDataError
    from src.research.hedge import HedgeInputs

    sessions = _sessions()
    pipe = _context(tmp_path, sessions, trending_index=True)
    counter = _score_counter(monkeypatch)
    inputs = pipe._ctx.hedge_inputs
    levels = np.asarray(inputs.index_level, dtype=np.float64).copy()
    inverse = np.asarray(inputs.inverse_close, dtype=np.float64).copy()
    levels[30] = np.nan
    inverse[45] = np.nan
    object.__setattr__(
        pipe._ctx,
        "hedge_inputs",
        HedgeInputs(
            sessions=inputs.sessions,
            index_level=np.ascontiguousarray(levels),
            inverse_close=np.ascontiguousarray(inverse),
        ),
    )
    with pytest.raises(PITDataError) as excinfo:
        pipe.evaluate(_spec())
    message = str(excinfo.value)
    assert sessions[30].isoformat() in message
    assert f"inverse close is also missing at {sessions[45].isoformat()}" in message
    assert counter.calls == 0
    assert not list((tmp_path / "reports").glob("*.json"))


def test_unusable_inputs_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.pit import PITDataError
    from src.research.hedge import HedgeInputs

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    inputs = pipe._ctx.hedge_inputs

    object.__setattr__(
        pipe._ctx,
        "hedge_inputs",
        HedgeInputs(
            sessions=inputs.sessions,
            index_level=np.ascontiguousarray(np.asarray(inputs.index_level)[:-1]),
            inverse_close=np.ascontiguousarray(np.asarray(inputs.inverse_close)),
        ),
    )
    with pytest.raises(PITDataError, match="differs from its sessions"):
        pipe.evaluate(_spec())

    object.__setattr__(pipe._ctx, "hedge_inputs", inputs)
    object.__setattr__(pipe._ctx, "cash_returns", np.zeros(len(sessions) - 1, dtype=np.float64))
    with pytest.raises(PITDataError, match="not aligned"):
        pipe.evaluate(_spec())

    object.__setattr__(pipe._ctx, "cash_returns", np.full(len(sessions), np.nan, dtype=np.float64))
    with pytest.raises(PITDataError, match="no session in"):
        pipe.evaluate(_spec())

    object.__setattr__(pipe._ctx, "cash_returns", None)
    with pytest.raises(PITDataError, match="cash return series"):
        pipe.evaluate(_spec())


def test_window_guard_rejects_out_of_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec(scorer={"first_test_year": 2030})
    with pytest.raises(ValueError, match="first_test_year"):
        pipe.evaluate(spec)
    with pytest.raises(ValueError, match="first_test_year"):
        pipe.scores(spec)
    with pytest.raises(ValueError, match="capital_krw must be positive"):
        pipe.evaluate(_spec(), capital_krw=0)
    with pytest.raises(WindowError):
        pipe._ctx.guard.authorize(start=date(2017, 1, 1), end=sessions[-1])


def test_index_benchmark_fails_closed_on_missing_hedge_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.pit import PITDataError
    from src.research.hedge import HedgeInputs

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    window = tuple(sessions)
    inputs = pipe._ctx.hedge_inputs

    def _with(new_inputs: object) -> None:
        object.__setattr__(pipe._ctx, "hedge_inputs", new_inputs)

    levels = np.asarray(inputs.index_level)
    inverse = np.asarray(inputs.inverse_close)

    _with(
        HedgeInputs(
            sessions=tuple(inputs.sessions),
            index_level=np.ascontiguousarray(levels),
            inverse_close=np.ascontiguousarray(inverse),
        )
    )
    assert np.allclose(pipe._index_log_returns(window), 0.0)

    dropped_day = np.concatenate([levels[:4], levels[5:]])
    _with(
        HedgeInputs(
            sessions=tuple(inputs.sessions[:4]) + tuple(inputs.sessions)[5:],
            index_level=np.ascontiguousarray(dropped_day),
            inverse_close=np.ascontiguousarray(inverse),
        )
    )
    with pytest.raises(PITDataError, match="missing"):
        pipe._index_log_returns(window)

    _with(
        HedgeInputs(
            sessions=tuple(inputs.sessions),
            index_level=np.ascontiguousarray(levels[:-2]),
            inverse_close=np.ascontiguousarray(inverse),
        )
    )
    with pytest.raises(PITDataError, match="differs from its sessions"):
        pipe._index_log_returns(window)

    broken = levels.copy()
    broken[6] = np.nan
    _with(
        HedgeInputs(
            sessions=tuple(inputs.sessions),
            index_level=np.ascontiguousarray(broken),
            inverse_close=np.ascontiguousarray(inverse),
        )
    )
    with pytest.raises(PITDataError, match="non-positive"):
        pipe._index_log_returns(window)

    _with(
        HedgeInputs(
            sessions=tuple(inputs.sessions)[1:],
            index_level=np.ascontiguousarray(levels[1:]),
            inverse_close=np.ascontiguousarray(inverse[1:]),
        )
    )
    assert pipe._index_log_returns(window)[0] == 0.0

    _with(
        HedgeInputs(
            sessions=tuple(inputs.sessions),
            index_level=np.ascontiguousarray(levels),
            inverse_close=np.ascontiguousarray(inverse),
        )
    )
    run = pipe.evaluate(_spec())
    assert np.isfinite(run.report.controls["index_g"])


def test_universe_benchmark_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec(policy={"universe": {"min_adtv20_krw": 10**15, "min_price_krw": 0}})
    run = pipe.evaluate(spec)
    assert np.isfinite(run.report.controls["universe_ew_g"])
    assert len(run.evidence.universe_ew_log_returns) == len(run.evidence.sessions)

    n_rows = len(sessions)
    uni = np.ones((n_rows, len(INSTRS)), dtype=bool)
    flat = np.full((n_rows, len(INSTRS)), 10_000.0)
    assert np.all(pipe._universe_ew_log_returns(0, 1, uni, flat) == 0.0)

    one = np.zeros((n_rows, len(INSTRS)), dtype=bool)
    one[:, 0] = True
    nan_close = flat.copy()
    nan_close[4, 0] = np.nan
    assert np.all(pipe._universe_ew_log_returns(4, 5, one, nan_close) == 0.0)

    rising = flat * (1.001 ** np.arange(n_rows, dtype=float)[:, None])
    assert pipe._universe_ew_log_returns(1, 3, one, rising)[0] == pytest.approx(math.log(1.001))
    assert np.all(pipe._universe_ew_log_returns(1, 3, np.zeros_like(one), flat) == 0.0)


def test_scores_cache_validity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    first = pipe.scores(spec)
    cache_files = list((tmp_path / "scores").glob("*.npz"))
    assert cache_files
    with np.load(str(cache_files[0])) as store:
        bad_scores = np.asarray(store["scores"]).copy()
    bad_scores[:] = np.nan
    np.savez_compressed(
        cache_files[0],
        cube_id=np.array("wrong"),
        config_hash=np.array("x"),
        test_years=np.asarray([2019]),
        last_row=np.asarray(0),
        scores=bad_scores,
    )
    second = pipe.scores(spec)
    assert np.array_equal(np.asarray(first.scores), np.asarray(second.scores), equal_nan=True)


def test_scores_cache_corrupt_file_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    first = pipe.scores(spec)
    cache_files = list((tmp_path / "scores").glob("*.npz"))
    assert cache_files
    cache_files[0].write_bytes(b"not a npz")
    second = pipe.scores(spec)
    assert np.array_equal(np.asarray(first.scores), np.asarray(second.scores), equal_nan=True)


def test_panel_and_window_guards(tmp_path: Path) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    with pytest.raises(ValueError, match="last_row"):
        pipe.panel_for(-1)
    with pytest.raises(ValueError, match="last_row"):
        pipe.panel_for(len(sessions))
    with pytest.raises(ValueError, match="capital_krw must be positive"):
        pipe._sim_config(0)


def test_spec_identity(tmp_path: Path) -> None:
    from src.research.pipeline import load_strategy_spec

    good = tmp_path / "a.toml"
    good.write_text(
        '[policy]\nfamily="ml_trend_cash"\nn=20\nrebalance_every_sessions=5\n[policy.universe]\nmin_adtv20_krw=0\nmin_price_krw=0\n'
        "[scorer]\nfirst_test_year=2019\n"
        "[book]\nsleeves=5\nstock_capital_fraction=0.75\n"
        "[hedge]\nhedge_ratio=1.0\nbeta_window_sessions=10\nbeta_min_sessions=2\nbeta_cap=2.0\n"
        "rebalance_every_sessions=5\nuse_futures=true\ncontract_multiplier_krw=10000\ninitial_margin_rate=0.2175\n"
        "margin_buffer_rate=0.10\nmargin_topup_trigger_fraction=0.75\nfutures_cost_rate=0.0003\ninverse_cost_rate=0.0007\n"
        "resize_sell_cost_rate=0.0025\nresize_buy_cost_rate=0.0005\nfutures_tax_rate=0.11\n"
        "futures_annual_deduction_krw=2500000\ninverse_tax_rate=0.154\n",
        encoding="utf-8",
    )
    other = tmp_path / "b.toml"
    other.write_text(
        '[policy]\nfamily="ml_trend_cash"\nn=20\nrebalance_every_sessions=5\n[policy.universe]\nmin_adtv20_krw=0\nmin_price_krw=0\n'
        "[scorer]\nfirst_test_year=2019\nseed=99\n"
        "[book]\nsleeves=5\nstock_capital_fraction=0.75\n"
        "[hedge]\nhedge_ratio=1.0\nbeta_window_sessions=10\nbeta_min_sessions=2\nbeta_cap=2.0\n"
        "rebalance_every_sessions=5\nuse_futures=true\ncontract_multiplier_krw=10000\ninitial_margin_rate=0.2175\n"
        "margin_buffer_rate=0.10\nmargin_topup_trigger_fraction=0.75\nfutures_cost_rate=0.0003\ninverse_cost_rate=0.0007\n"
        "resize_sell_cost_rate=0.0025\nresize_buy_cost_rate=0.0005\nfutures_tax_rate=0.11\n"
        "futures_annual_deduction_krw=2500000\ninverse_tax_rate=0.154\n",
        encoding="utf-8",
    )
    assert load_strategy_spec(good).spec_hash != load_strategy_spec(other).spec_hash


def test_strategy_spec_from_canonical_json_round_trip() -> None:
    from src.research.pipeline import StrategySpec, strategy_spec_from_canonical_json

    spec = _spec(policy={"n": 7})
    restored = strategy_spec_from_canonical_json(spec.canonical_json())
    assert isinstance(restored, StrategySpec)
    assert restored == spec
    assert restored.spec_hash == spec.spec_hash
    for bad in ("{}", '{"policy": {}}', "not json"):
        with pytest.raises(ValueError, match="invalid strategy canonical JSON"):
            strategy_spec_from_canonical_json(bad)
    mismatched = json.loads(spec.canonical_json())
    mismatched["book"] = {"sleeves": 3, "stock_capital_fraction": 0.75}
    with pytest.raises(ValueError, match="invalid strategy canonical JSON"):
        strategy_spec_from_canonical_json(json.dumps(mismatched))


def _count_build_panel(monkeypatch: pytest.MonkeyPatch) -> Any:
    import src.research.pipeline as pipe_mod

    real = pipe_mod.build_panel
    calls = {"n": 0}

    def _counting(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pipe_mod, "build_panel", _counting)
    return calls


def test_panel_built_once_per_evaluate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    calls = _count_build_panel(monkeypatch)
    spec = _spec()
    pipe.evaluate(spec)
    assert calls["n"] == 1
    pipe.evaluate(spec)
    assert calls["n"] == 2


def test_scores_reject_a_foreign_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    _, hi, _ = pipe._window_rows(spec)
    with pytest.raises(ValueError, match="data-end row"):
        pipe.scores(spec, panel=pipe.panel_for(hi - 1))
    assert pipe.scores(spec, panel=pipe.panel_for(hi)).last_row == hi


def _causal_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str) -> Any:
    sessions = _multi_year_sessions()
    pipe = _context(tmp_path / tag, sessions, trending_index=True)
    _clean_scores(monkeypatch)
    object.__setattr__(pipe._ctx, "ledger_runner", _causal_account(sessions))
    return pipe, pipe.evaluate(_spec())


def test_perturbation_result_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, run = _causal_run(tmp_path, monkeypatch, "a")
    assert run.evidence.perturbation_mismatches == 0
    monkeypatch.setattr(type(pipe), "_perturbation_cuts", lambda self, **kwargs: ())
    _, unperturbed = _causal_run(tmp_path, monkeypatch, "b")
    assert run.report.canonical_json() == unperturbed.report.canonical_json()
    assert run.report.digest == unperturbed.report.digest
    assert len(pipe._ctx.registry.runs(run_id=run.report.run_id)) == 9


def test_cuts_do_not_overlap_in_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import gc
    import weakref

    import src.research.pipeline as pipe_mod

    real = pipe_mod.Pipeline._corrupt_cube
    seen: list[Any] = []

    def _spy(cube: Any, cut: int, seed: int) -> Any:
        # ResearchCube is a frozen slotted dataclass without weakref support; the copy-on-write
        # mapping it hands out is owned solely by the cube, so its lifetime is the cube's
        # lifetime for this check.
        gc.collect()
        if seen:
            assert seen[-1]() is None, "previous cut's corrupted cube is still alive"
        out = real(cube, cut, seed)
        seen.append(weakref.ref(out.arrays["close"]))
        return out

    monkeypatch.setattr(pipe_mod.Pipeline, "_corrupt_cube", staticmethod(_spy))
    pipe, run = _causal_run(tmp_path, monkeypatch, "b")
    assert len(seen) == int(pipe._ctx.protocol.scenarios.perturbation_cuts)
    assert run.evidence.perturbation_mismatches == 0
    gc.collect()
    assert all(ref() is None for ref in seen)


def test_memory_telemetry_emitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    import src.research.pipeline as pipe_mod

    caplog.set_level(logging.INFO, logger=pipe_mod.__name__)
    _, run = _causal_run(tmp_path, monkeypatch, "c")
    assert run.evidence.perturbation_mismatches == 0
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO and "[SYS] evaluate memory phase=" in record.getMessage()
    ]
    n_cuts = 3
    expected = ["inputs", "panel", "scores", "targets", "scenarios"] + [f"cut{i}" for i in range(n_cuts)] + ["done"]
    assert [line.split("phase=")[1].split(" ")[0] for line in lines] == expected


def _memmapped_cube(root: Path, arrays: dict[str, np.ndarray]) -> ResearchCube:
    """A cube whose arrays are read-only memory maps, exactly what a cache hit hands the pipeline."""
    root.mkdir(parents=True, exist_ok=True)
    mapped: dict[str, Any] = {}
    for name, arr in arrays.items():
        np.save(root / f"arr_{name}.npy", np.ascontiguousarray(arr), allow_pickle=False)
        mapped[name] = np.load(root / f"arr_{name}.npy", mmap_mode="r")
    n_s = next(iter(arrays.values())).shape[0]
    return ResearchCube(
        cube_id="research_cube_0123456789abcdef",
        sessions=tuple(_sessions(n_s)),
        instrument_ids=tuple(INSTRS),
        arrays=mapped,
        exit_at=np.full(len(INSTRS), -1, dtype=np.int64),
        exit_halted=np.zeros(len(INSTRS), dtype=bool),
    )


def _corrupt_cube(cube: ResearchCube, cut: int, seed: int) -> ResearchCube:
    from src.research.pipeline import Pipeline

    return Pipeline._corrupt_cube(cube, cut, seed)


def _probe_arrays(n_s: int) -> dict[str, np.ndarray]:
    n_n = len(INSTRS)
    return {
        "alpha": np.ascontiguousarray(np.arange(n_s * n_n, dtype=np.float64).reshape(n_s, n_n) + 1.0),
        "beta": np.ascontiguousarray(np.full((n_s, n_n), 7.0) + np.arange(n_s * n_n).reshape(n_s, n_n) % 3),
        "gamma": np.ascontiguousarray(np.arange(n_s * n_n).reshape(n_s, n_n) % 2 == 0),
        "delta": np.ascontiguousarray(np.arange(n_s * n_n, dtype=np.int64).reshape(n_s, n_n) % 11),
    }


def test_lazy_corruption_materialises_only_read_arrays(tmp_path: Path) -> None:
    """A consumer that reads one array pays for one array; the others stay untouched and match an eager read."""
    from collections.abc import Mapping

    cube = _memmapped_cube(tmp_path / "cube", _probe_arrays(12))
    lazy = _corrupt_cube(cube, 5, 3)
    eager = _corrupt_cube(cube, 5, 3)
    assert isinstance(lazy.arrays, Mapping)
    assert tuple(lazy.arrays) == tuple(cube.arrays)
    assert len(lazy.arrays) == len(cube.arrays)
    with pytest.raises(TypeError):
        lazy.arrays["delta"] = np.zeros((12, len(INSTRS)))  # type: ignore[index]
    assert lazy.arrays.materialised() == ()
    assert not lazy.arrays["alpha"].flags.writeable
    assert lazy.arrays.materialised() == ("alpha",)
    assert not lazy.arrays["beta"].flags.writeable
    for name in cube.arrays:
        assert np.array_equal(lazy.arrays[name], eager.arrays[name], equal_nan=True)
    assert lazy.cube_id == cube.cube_id
    assert lazy.sessions == cube.sessions
    assert lazy.instrument_ids == cube.instrument_ids
    assert np.array_equal(lazy.exit_at, cube.exit_at)
    assert np.array_equal(lazy.exit_halted, cube.exit_halted)


def test_corrupted_cube_keeps_the_prefix_and_changes_the_tail(tmp_path: Path) -> None:
    """Rows up to the cut are byte-identical to the source; the tail is corrupted per dtype."""
    cube = _memmapped_cube(tmp_path / "cube", _probe_arrays(12))
    out = _corrupt_cube(cube, 5, 11)
    for name in ("alpha", "beta"):
        corrupted = out.arrays[name]
        assert corrupted.flags.c_contiguous
        assert corrupted[:6].tobytes() == cube.arrays[name][:6].tobytes()
        assert corrupted[6:].tobytes() != cube.arrays[name][6:].tobytes()
    flags = out.arrays["gamma"]
    assert flags[:6].tobytes() == cube.arrays["gamma"][:6].tobytes()
    flipped = int(np.count_nonzero(flags[6:] != cube.arrays["gamma"][6:]))
    assert 0.2 < flipped / flags[6:].size < 0.8
    ints = out.arrays["delta"]
    assert ints[:6].tobytes() == cube.arrays["delta"][:6].tobytes()
    steps = ints[6:].astype(np.int64) - cube.arrays["delta"][6:].astype(np.int64)
    assert bool(np.any(steps != 0))
    assert int(np.abs(steps).max()) <= 2


def test_corruption_is_independent_of_access_order(tmp_path: Path) -> None:
    """Two cubes built with the same seed agree byte for byte however their arrays are read."""
    names = ("alpha", "beta", "gamma", "delta")
    cube = _memmapped_cube(tmp_path / "cube", _probe_arrays(12))
    forward = _corrupt_cube(cube, 4, 17)
    backward = _corrupt_cube(cube, 4, 17)
    for name in names:
        _ = forward.arrays[name]
    for name in reversed(names):
        _ = backward.arrays[name]
    for name in names:
        assert forward.arrays[name].tobytes() == backward.arrays[name].tobytes()
    assert forward.arrays.materialised() == names
    assert backward.arrays.materialised() == tuple(reversed(names))


def test_corruption_never_mutates_the_source(tmp_path: Path) -> None:
    """Reading every corrupted array leaves the memory-mapped source byte-identical and read-only."""
    import hashlib

    cube = _memmapped_cube(tmp_path / "cube", _probe_arrays(12))

    def _digest() -> str:
        return hashlib.sha256(
            b"".join(np.ascontiguousarray(cube.arrays[name]).tobytes() for name in sorted(cube.arrays))
        ).hexdigest()

    before = _digest()
    out = _corrupt_cube(cube, 5, 2)
    for name in sorted(out.arrays):
        _ = out.arrays[name]
    assert out.arrays.materialised() == tuple(sorted(out.arrays))
    assert _digest() == before
    assert all(not cube.arrays[name].flags.writeable for name in cube.arrays)


_SMAPS_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s")


def _private_dirty_kb(needle: str) -> int:
    """Sum of ``Private_Dirty`` over the VMAs backed by ``needle``: pages this process has written."""
    total = 0
    current = False
    with open("/proc/self/smaps", encoding="utf-8") as handle:
        for line in handle:
            if _SMAPS_HEADER.match(line):
                current = needle in line
            elif current and line.startswith("Private_Dirty:"):
                total += int(line.split()[1])
    return total


@pytest.mark.skipif(not Path("/proc/self/smaps").is_file(), reason="needs Linux procfs")
def test_copy_on_write_charges_only_the_corrupted_tail(tmp_path: Path) -> None:
    """Corrupting a mapped array dirties its tail, not the whole file: a full copy would cost 10x more."""
    n_s, n_n = 2_000, 500
    cube = _memmapped_cube(tmp_path / "cube", {"wide": np.ones((n_s, n_n), dtype=np.float64)})
    name = str((tmp_path / "cube" / "arr_wide.npy").resolve())
    before = _private_dirty_kb(name)
    corrupted = _corrupt_cube(cube, 1_800, 3).arrays["wide"]
    assert np.array_equal(corrupted[:1_801], cube.arrays["wide"][:1_801])
    tail_kb = (n_s - 1_801) * n_n * 8 / 1024
    full_kb = n_s * n_n * 8 / 1024
    assert _private_dirty_kb(name) - before <= 2 * tail_kb + 1024
    assert tail_kb < full_kb / 4


def test_corrupt_cube_rejects_a_cut_outside_the_panel(tmp_path: Path) -> None:
    """A cut that leaves no rows to corrupt, or points past the panel, is a caller error."""
    cube = _memmapped_cube(tmp_path / "cube", _probe_arrays(12))
    for cut in (-1, 11, 40):
        with pytest.raises(ValueError, match="perturbation cut"):
            _corrupt_cube(cube, cut, 1)
    assert _corrupt_cube(cube, 0, 1).arrays.materialised() == ()


def test_in_memory_source_is_copied_not_aliased() -> None:
    """A synthetic in-memory cube is copied per array, so a consumer can never write into the fixture."""
    sessions = _sessions(12)
    arrays = _probe_arrays(12)
    cube = ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=sessions,
        instrument_ids=INSTRS,
        arrays=arrays,
        exit_at=np.full(len(INSTRS), -1, dtype=np.int64),
        exit_halted=np.zeros(len(INSTRS), dtype=bool),
    )
    out = _corrupt_cube(cube, 5, 9)
    assert not isinstance(out.arrays["alpha"], np.memmap)
    assert out.arrays["alpha"][6:].tobytes() != arrays["alpha"][6:].tobytes()
    assert arrays["alpha"][6:].tobytes() == cube.arrays["alpha"][6:].tobytes()


def test_lazy_corruption_still_catches_a_panel_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A panel built from row t+1 of ``close`` is caught: the lazy copy still corrupts ``close`` on access."""
    import src.research.pipeline as pipe_mod
    from src.research.model import ScoreMatrix
    from src.research.panel import FeaturePanel

    sessions = _multi_year_sessions()
    pipe = _context(tmp_path, sessions)
    real_panel = pipe_mod.build_panel

    def _leaky_panel(cube: Any, *, last_row: int) -> Any:
        panel = real_panel(cube, last_row=last_row)
        n_rows, n_inst = panel.features["dev_ma20"].shape
        dev = np.asarray(panel.features["dev_ma20"], dtype=np.float64)
        close = np.asarray(cube.arrays["close"], dtype=np.float64)

        def _shifted(block: Any) -> Any:
            padded = np.full((n_rows, n_inst), np.nan)
            padded[:-1] = block[1:n_rows]
            return padded

        today = np.full((n_rows, n_inst), np.nan)
        today[:-1] = close[: n_rows - 1]
        # A deliberate leak: today's feature is next day's feature rescaled by tomorrow's close. The clean
        # panel is unchanged, so only a ``close`` corrupted past the cut can move a score at or before it.
        features = dict(panel.features)
        features["dev_ma20"] = np.ascontiguousarray(_shifted(dev) * (_shifted(close) / today), dtype=np.float32)
        features["dev_ma20"].flags.writeable = False
        return FeaturePanel(features=features, labels=dict(panel.labels), last_row=panel.last_row)

    def _scores_from_feature(
        panel: Any, universe: Any, panel_sessions: Any, config: Any, *, test_years: Any, authorization: Any
    ) -> Any:
        n_rows, n_inst = panel.features["dev_ma20"].shape
        uni = np.asarray(universe, dtype=bool)
        dev = np.asarray(panel.features["dev_ma20"], dtype=float)
        out = np.full((n_rows, n_inst), np.nan)
        for row, day in enumerate(list(panel_sessions)[:n_rows]):
            if day.year in set(test_years) and bool(uni[row].any()):
                out[row] = dev[row]
        arr = np.ascontiguousarray(out, dtype=np.float32)
        arr.flags.writeable = False
        return ScoreMatrix(
            scores=arr, test_years=tuple(test_years), config_hash=config.config_hash, last_row=panel.last_row
        )

    monkeypatch.setattr(pipe_mod, "build_panel", _leaky_panel)
    monkeypatch.setattr(pipe_mod, "walk_forward_scores", _scores_from_feature)
    run = pipe.evaluate(_spec())
    assert run.evidence.perturbation_mismatches > 0
    assert {c.name: c for c in run.report.integrity}["perturbation_mismatches"].passed is False
    assert run.report.passed is False


def test_band_reaches_every_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.pipeline as pipeline_mod

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec(book={"rebalance_band": 0.5})
    calls: list[dict[str, Any]] = []
    runner = pipe._ctx.ledger_runner
    simulator = pipeline_mod.simulate
    sim_bands: list[float] = []

    def _sim_spy(*args: Any, **kwargs: Any) -> Any:
        sim_bands.append(kwargs["rebalance_band"])
        return simulator(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "simulate", _sim_spy)

    def _spy(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return runner(**kwargs)

    object.__setattr__(pipe._ctx, "ledger_runner", _spy)
    real_cuts = type(pipe)._perturbation_cuts

    def _one_cut(self: Any, **kwargs: Any) -> Any:
        got = real_cuts(self, **kwargs)
        if got:
            return (got[0],)
        lo, hi = int(kwargs["lo"]), int(kwargs["hi"])
        return (lo + 1,) if hi > lo + 1 else ()

    monkeypatch.setattr(type(pipe), "_perturbation_cuts", _one_cut)
    pipe.evaluate(spec)
    ticks = list(pipe._ctx.protocol.scenarios.cost_grid_ticks)
    assert len(calls) == 3 + len(ticks) + 2 + 1
    assert calls
    assert all(float(call.get("rebalance_band", -1.0)) == 0.5 for call in calls)
    assert sim_bands == [0.5]


def _trend_spec(**overrides: Any) -> Any:
    from src.research.trend_overlay import TrendOverlaySpec

    trend_kw: dict[str, Any] = {
        "ma_sessions": 20,
        "long_fraction": 1.0,
        "short_fraction": 0.5,
        "rebalance_every_sessions": 5,
        "contract_multiplier_krw": 10000,
        "initial_margin_rate": 0.2,
        "margin_buffer_rate": 0.1,
        "margin_topup_trigger_fraction": 0.75,
        "futures_cost_rate": 0.0003,
        "futures_tax_rate": 0.11,
        "futures_annual_deduction_krw": 2500000,
    }
    trend_kw.update(overrides.pop("trend_overlay", {}))
    base = _spec(hedge={"hedge_ratio": 0.0}, **overrides)
    return base.model_copy(update={"trend_overlay": TrendOverlaySpec(**trend_kw)})


def _with_trend_inputs(pipe: Any, sessions: list[date]) -> Any:
    from src.research.hedge import HedgeInputs

    rng = np.random.default_rng(11)
    steps = rng.normal(loc=0.0004, scale=0.01, size=len(sessions))
    level = 1500.0 * np.exp(np.cumsum(steps))
    inverse = 8000.0 * np.exp(-np.cumsum(steps))
    inputs = HedgeInputs(
        sessions=tuple(sessions),
        index_level=np.ascontiguousarray(level),
        inverse_close=np.ascontiguousarray(inverse),
    )
    object.__setattr__(pipe._ctx, "trend_inputs", inputs)
    object.__setattr__(
        pipe._ctx,
        "dataset_ids",
        dict(pipe._ctx.dataset_ids, trend_series="trend_series_eeeeeeeeeeeeeeee"),
    )
    return inputs


def test_pre_overlay_spec_identity_unchanged() -> None:
    from src.research.pipeline import load_strategy_spec

    spec = load_strategy_spec(__import__("pathlib").Path("config/research/strategies/ml_sleeve_hedge.toml"))
    assert spec.spec_hash == "4ec4b1c58d8c722954a20cb7507dc9f7943bbd8d769fc040676696484f1c1dc9"


def test_one_overlay_per_account(tmp_path: Path) -> None:
    from src.research.pipeline import load_strategy_spec

    path = tmp_path / "trend.toml"
    path.write_text(
        '[policy]\nfamily="ml_trend_cash"\nn=20\nrebalance_every_sessions=5\n[policy.universe]\nmin_adtv20_krw=0\nmin_price_krw=0\n'
        "[scorer]\nfirst_test_year=2019\n"
        "[book]\nsleeves=5\nstock_capital_fraction=0.75\n"
        "[hedge]\nhedge_ratio=1.0\nbeta_window_sessions=10\nbeta_min_sessions=2\nbeta_cap=2.0\n"
        "rebalance_every_sessions=5\nuse_futures=true\ncontract_multiplier_krw=10000\ninitial_margin_rate=0.2175\n"
        "margin_buffer_rate=0.10\nmargin_topup_trigger_fraction=0.75\nfutures_cost_rate=0.0003\ninverse_cost_rate=0.0007\n"
        "resize_sell_cost_rate=0.0025\nresize_buy_cost_rate=0.0005\nfutures_tax_rate=0.11\n"
        "futures_annual_deduction_krw=2500000\ninverse_tax_rate=0.154\n"
        "[trend_overlay]\nma_sessions=50\nlong_fraction=1.0\nshort_fraction=0.5\nrebalance_every_sessions=5\n"
        "contract_multiplier_krw=10000\ninitial_margin_rate=0.2\nmargin_buffer_rate=0.1\n"
        "margin_topup_trigger_fraction=0.75\nfutures_cost_rate=0.0003\nfutures_tax_rate=0.11\n"
        "futures_annual_deduction_krw=2500000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="one overlay per account"):
        load_strategy_spec(path)
    path.write_text(path.read_text(encoding="utf-8").replace("hedge_ratio=1.0", "hedge_ratio=0.0"), encoding="utf-8")
    from src.research.pipeline import strategy_spec_from_canonical_json

    spec = load_strategy_spec(path)
    assert spec.trend_overlay is not None
    assert strategy_spec_from_canonical_json(spec.canonical_json()) == spec


def test_invalid_trend_overlay_table_rejected(tmp_path: Path) -> None:
    from src.research.pipeline import load_strategy_spec

    path = tmp_path / "bad_trend.toml"
    path.write_text(
        '[policy]\nfamily="ml_trend_cash"\nn=20\nrebalance_every_sessions=5\n[policy.universe]\nmin_adtv20_krw=0\nmin_price_krw=0\n'
        "[scorer]\nfirst_test_year=2019\n"
        "[book]\nsleeves=5\nstock_capital_fraction=0.75\n"
        "[hedge]\nhedge_ratio=0.0\nbeta_window_sessions=10\nbeta_min_sessions=2\nbeta_cap=2.0\n"
        "rebalance_every_sessions=5\nuse_futures=true\ncontract_multiplier_krw=10000\ninitial_margin_rate=0.2175\n"
        "margin_buffer_rate=0.10\nmargin_topup_trigger_fraction=0.75\nfutures_cost_rate=0.0003\ninverse_cost_rate=0.0007\n"
        "resize_sell_cost_rate=0.0025\nresize_buy_cost_rate=0.0005\nfutures_tax_rate=0.11\n"
        "futures_annual_deduction_krw=2500000\ninverse_tax_rate=0.154\n"
        "[trend_overlay]\nma_sessions=1\nlong_fraction=1.0\nshort_fraction=0.5\nrebalance_every_sessions=5\n"
        "contract_multiplier_krw=10000\ninitial_margin_rate=0.2\nmargin_buffer_rate=0.1\n"
        "margin_topup_trigger_fraction=0.75\nfutures_cost_rate=0.0003\nfutures_tax_rate=0.11\n"
        "futures_annual_deduction_krw=2500000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trend_overlay"):
        load_strategy_spec(path)


def test_trend_run_id_requires_trend_dataset_id() -> None:
    from datetime import date

    from src.research.pipeline import _evaluation_run_id

    spec = _trend_spec()
    with pytest.raises(ValueError, match="trend_series"):
        _evaluation_run_id(
            spec=spec,
            capital_krw=100_000_000,
            protocol_hash="p",
            cube_id="c",
            dataset_ids={
                "market_panel": "a",
                "dividend_events": "b",
                "hedge_series": "c",
                "cash_series": "d",
            },
            start=date(2020, 1, 2),
            end=date(2020, 2, 2),
            engine_config_bytes=b"engine",
        )


def test_trend_level_gap_fails_before_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.pit import PITDataError

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    counter = _score_counter(monkeypatch)
    _with_trend_inputs(pipe, sessions)
    inputs = pipe._ctx.trend_inputs
    assert inputs is not None
    levels = np.asarray(inputs.index_level, dtype=np.float64).copy()
    levels[30] = np.nan
    from src.research.hedge import HedgeInputs

    object.__setattr__(
        pipe._ctx,
        "trend_inputs",
        HedgeInputs(
            sessions=inputs.sessions,
            index_level=np.ascontiguousarray(levels),
            inverse_close=np.ascontiguousarray(np.asarray(inputs.inverse_close)),
        ),
    )
    with pytest.raises(PITDataError, match="trend index level is missing"):
        pipe.evaluate(_trend_spec())
    assert counter.calls == 0


def test_trend_scenario_overlay_requires_inputs(tmp_path: Path) -> None:
    from src.core.pit import PITDataError

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    with pytest.raises(PITDataError, match="trend return series"):
        pipe._trend_scenario_overlay(_trend_spec(), "base", delay_n=0, extra_cost=0.0)


def test_trend_spec_routes_every_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research.trend_overlay import TrendOverlay

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    trend_inputs = _with_trend_inputs(pipe, sessions)
    spec = _trend_spec()
    spec = spec.model_copy(update={"hedge": spec.hedge.model_copy(update={"inverse_cost_rate": 0.9999})})
    seen: list[dict[str, Any]] = []
    runner = pipe._ctx.ledger_runner

    def _spy(**kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        return runner(**kwargs)

    object.__setattr__(pipe._ctx, "ledger_runner", _spy)
    run = pipe.evaluate(spec)
    delay_n = int(pipe._ctx.protocol.scenarios.stress_delay_sessions)
    extra = float(pipe._ctx.protocol.scenarios.stress_hedge_extra_cost)
    names = ["base", "stress_slippage", "stress_delay"] + [
        f"cost_{tick}" for tick in pipe._ctx.protocol.scenarios.cost_grid_ticks
    ] + ["unhedged", "placebo"]
    assert len(seen) == len(names)
    for name, call in zip(names, seen, strict=True):
        overlay = call["overlay"]
        inner = overlay._inner if hasattr(overlay, "_inner") else overlay
        assert isinstance(inner, TrendOverlay)
        assert inner._delay == (delay_n if name == "stress_delay" else 0)
        assert inner._spec.long_fraction == (0.0 if name == "unhedged" else 1.0)
        assert inner._spec.short_fraction == (0.0 if name == "unhedged" else 0.5)
        assert call["derivatives"].futures_cost_rate == pytest.approx(0.0003 + (extra if name == "stress_slippage" else 0.0))
        assert call["derivatives"].inverse_cost_rate == 0.0
        np.testing.assert_array_equal(call["overlay_market"].index_level, trend_inputs.index_level)
        np.testing.assert_array_equal(inner._levels, trend_inputs.index_level)
    np.testing.assert_array_equal(run.evidence.index_log_returns, pipe._index_log_returns(tuple(sessions)))


def test_misaligned_trend_inputs_fail_before_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from src.core.pit import PITDataError

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    counter = _score_counter(monkeypatch)
    inputs = _with_trend_inputs(pipe, sessions)
    object.__setattr__(pipe._ctx, "trend_inputs", replace(inputs, sessions=tuple(reversed(sessions))))
    with pytest.raises(PITDataError, match="not aligned"):
        pipe.evaluate(_trend_spec())
    assert counter.calls == 0


def test_trend_overlay_replays_signed_positions_on_account_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research.hedge import HedgeInputs
    from src.research.ledger_bridge import run_ledger

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    inputs = _with_trend_inputs(pipe, sessions)
    levels = np.concatenate([np.linspace(1000.0, 1100.0, 40), np.linspace(1100.0, 1000.0, 40)])
    object.__setattr__(
        pipe._ctx, "trend_inputs", HedgeInputs(inputs.sessions, levels, np.full(len(sessions), np.nan))
    )
    base_decisions: list[Any] = []

    def _runner(**kwargs: Any) -> Any:
        outcome = run_ledger(**kwargs)
        if hasattr(kwargs["overlay"], "decisions"):
            base_decisions.extend(kwargs["overlay"].decisions)
        return outcome

    object.__setattr__(pipe._ctx, "ledger_runner", _runner)
    run = pipe.evaluate(_trend_spec(trend_overlay={"ma_sessions": 5, "rebalance_every_sessions": 1}))
    assert any(decision.contracts < 0 for decision in base_decisions)
    assert any(decision.contracts > 0 for decision in base_decisions)
    assert run.evidence.base.avg_margin_share > 0.0
    assert run.evidence.base.avg_inverse_share == 0.0
    assert np.all(np.isfinite(run.evidence.base.log_returns))
    assert run.evidence.unhedged.avg_margin_share == 0.0
    assert run.evidence.unhedged.avg_inverse_share == 0.0


def test_trend_spec_without_trend_inputs_fails_before_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.pit import PITDataError

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    counter = _score_counter(monkeypatch)
    spec = _trend_spec()
    with pytest.raises(PITDataError):
        pipe.evaluate(spec)
    assert counter.calls == 0


def test_perturbation_corrupts_trend_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.pipeline as pipe_mod

    sessions = _multi_year_sessions()
    pipe = _context(tmp_path, sessions, trending_index=True)
    _clean_scores(monkeypatch)
    _with_trend_inputs(pipe, sessions)
    spec = _trend_spec()
    object.__setattr__(pipe._ctx, "ledger_runner", _causal_account(sessions))
    run = pipe.evaluate(spec)
    assert run.evidence.perturbation_mismatches == 0

    other = _context(tmp_path / "leaky", sessions, trending_index=True)
    _with_trend_inputs(other, sessions)
    object.__setattr__(other._ctx, "ledger_runner", _causal_account(sessions))

    class _Leaky:
        def __init__(self, spec: Any, *, index_level: Any, rebalance_offset: int = 0, execution_delay: int = 0) -> None:
            self._levels = np.asarray(index_level, dtype=np.float64)
            self._start: int | None = None

        def target(self, state: Any) -> Any:
            from src.backtest.overlay import OverlayTarget

            if self._start is None:
                self._start = int(state.session_idx)
            idx = int(state.session_idx) + 1
            if idx >= len(self._levels):
                return OverlayTarget(contracts=0, inverse_value_krw=0)
            return OverlayTarget(contracts=int(float(self._levels[idx]) * 1e6) % 7 - 3, inverse_value_krw=0)

    monkeypatch.setattr(pipe_mod, "TrendOverlay", _Leaky)
    leaky = other.evaluate(spec)
    assert leaky.evidence.perturbation_mismatches > 0


def test_run_id_includes_trend_dataset_only_for_trend_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    _clean_scores(monkeypatch)
    champion = _context(tmp_path / "a", sessions)
    trend_spec = _trend_spec()
    _with_trend_inputs(champion, sessions)
    plain = _spec()
    champion_run = champion.evaluate(plain)
    trend_run = champion.evaluate(trend_spec)
    other_ids = dict(champion._ctx.dataset_ids, trend_series="trend_series_ffffffffffffffff")
    other = _context(tmp_path / "b", sessions)
    _with_trend_inputs(other, sessions)
    object.__setattr__(other._ctx, "dataset_ids", other_ids)
    assert other.evaluate(plain).report.run_id == champion_run.report.run_id
    assert other.evaluate(trend_spec).report.run_id != trend_run.report.run_id


def _regime_spec(**overrides: Any) -> Any:
    from src.research.regime_hedge import RegimeHedgeSpec

    regime_kw: dict[str, Any] = {
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
    regime_kw.update(overrides.pop("regime_hedge", {}))
    base = _trend_spec(**overrides)
    return base.model_copy(update={"regime_hedge": RegimeHedgeSpec(**regime_kw)})


def test_champion_identity_unchanged_with_regime_hedge_support() -> None:
    from src.research.pipeline import load_strategy_spec

    champion_path = Path("config/research/strategies/ml_growth_t85_b50_k200.toml")
    spec = load_strategy_spec(champion_path)
    assert spec.spec_hash == "8c2aad2c8c3c5a133e7a3c6cef3134f43b89749dd87fa3d2f57c223118e4676c"
    assert spec.regime_hedge is None


def test_regime_hedge_without_trend_overlay_rejected() -> None:
    from src.research.pipeline import StrategySpec
    from src.research.regime_hedge import RegimeHedgeSpec

    regime = RegimeHedgeSpec(
        tsmom_horizons=(5, 10),
        target_vol=0.10,
        max_fraction=1.5,
        vol_window_sessions=20,
        rebalance_every_sessions=5,
        contract_multiplier_krw=10000,
        initial_margin_rate=0.2,
        margin_buffer_rate=0.1,
        margin_topup_trigger_fraction=0.75,
        futures_cost_rate=0.0003,
        futures_tax_rate=0.11,
        futures_annual_deduction_krw=2500000,
    )
    with pytest.raises(ValueError, match="regime_hedge requires trend_overlay is not None"):
        _spec(regime_hedge=regime)

    base = _trend_spec()
    with pytest.raises(ValueError, match=r"hedge\.hedge_ratio == 0"):
        StrategySpec(
            policy=base.policy,
            scorer=base.scorer,
            book=base.book,
            hedge=base.hedge.model_copy(update={"hedge_ratio": 0.5}),
            trend_overlay=base.trend_overlay,
            regime_hedge=regime,
        )


def test_both_legs_reach_every_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.backtest.overlay import OverlayState
    from src.research.regime_hedge import CompositeOverlay, RegimeHedgeLeg
    from src.research.trend_overlay import TrendOverlay

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    _with_trend_inputs(pipe, sessions)
    spec = _regime_spec()

    seen: list[dict[str, Any]] = []
    runner = pipe._ctx.ledger_runner

    def _spy(**kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        return runner(**kwargs)

    object.__setattr__(pipe._ctx, "ledger_runner", _spy)
    pipe.evaluate(spec)

    delay_n = int(pipe._ctx.protocol.scenarios.stress_delay_sessions)
    extra = float(pipe._ctx.protocol.scenarios.stress_hedge_extra_cost)
    names = ["base", "stress_slippage", "stress_delay"] + [
        f"cost_{tick}" for tick in pipe._ctx.protocol.scenarios.cost_grid_ticks
    ] + ["unhedged", "placebo"]
    assert len(seen) == len(names)

    for name, call in zip(names, seen, strict=True):
        assert "leg_derivatives" in call
        leg_derivs = call["leg_derivatives"]
        assert leg_derivs is not None
        assert "kq150" in leg_derivs

        overlay = call["overlay"]
        inner = overlay._inner if hasattr(overlay, "_inner") else overlay
        assert isinstance(inner, CompositeOverlay)
        primary = inner._primary
        secondary = inner._legs["kq150"]
        assert isinstance(primary, TrendOverlay)
        assert isinstance(secondary, RegimeHedgeLeg)

        expected_delay = delay_n if name == "stress_delay" else 0
        assert primary._delay == expected_delay
        assert secondary._delay == expected_delay

        expected_cost = 0.0003 + (extra if name == "stress_slippage" else 0.0)
        assert call["derivatives"].futures_cost_rate == pytest.approx(expected_cost)
        assert leg_derivs["kq150"].futures_cost_rate == pytest.approx(expected_cost)

        if name == "unhedged":
            assert primary._spec.long_fraction == 0.0
            assert primary._spec.short_fraction == 0.0
            assert secondary._spec.max_fraction == 0.0
            dummy_state = OverlayState(
                session_idx=0,
                nav=100_000_000,
                stock_book_nav=100_000_000,
                stock_book_returns=np.zeros(1),
                index_returns=np.zeros(1),
                index_level=1000.0,
                contracts=0,
                inverse_units=0,
                leg_contracts=(),
            )
            target = inner.target(dummy_state)
            assert target is not None
            assert target.contracts == 0
            assert target.legs == (("kq150", 0),)


def test_perturbation_detects_a_leaking_regime_hedge_leg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.pipeline as pipe_mod

    sessions = _multi_year_sessions()
    pipe = _context(tmp_path / "clean", sessions, trending_index=True)
    _clean_scores(monkeypatch)
    _with_trend_inputs(pipe, sessions)
    spec = _regime_spec()
    object.__setattr__(pipe._ctx, "ledger_runner", _causal_account(sessions))

    run_clean = pipe.evaluate(spec)
    assert run_clean.evidence.perturbation_mismatches == 0

    other = _context(tmp_path / "leaky", sessions, trending_index=True)
    _with_trend_inputs(other, sessions)
    object.__setattr__(other._ctx, "ledger_runner", _causal_account(sessions))

    class _LeakyRegimeHedgeLeg:
        def __init__(self, spec: Any, *, index_level: Any, rebalance_offset: int = 0, execution_delay: int = 0) -> None:
            self._levels = np.asarray(index_level, dtype=np.float64)
            self._start: int | None = None

        def target_contracts(self, state: Any) -> int | None:
            idx = int(state.session_idx) + 1
            if idx >= len(self._levels):
                return 0
            return int(float(self._levels[idx]) * 1e6) % 5 + 1

    monkeypatch.setattr(pipe_mod, "RegimeHedgeLeg", _LeakyRegimeHedgeLeg)
    leaky_run = other.evaluate(spec)
    assert leaky_run.evidence.perturbation_mismatches > 0


def test_load_strategy_spec_invalid_regime_hedge_table(tmp_path: Path) -> None:
    from src.research.pipeline import load_strategy_spec

    base_toml = Path("config/research/strategies/ml_growth_t85_b50_k200.toml").read_text(encoding="utf-8")
    content = base_toml + "\n[regime_hedge]\ntarget_vol = 'invalid_not_a_float'\n"
    p = tmp_path / "bad.toml"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid \\[regime_hedge\\] table"):
        load_strategy_spec(p)


_STAGED_REGIME_SPECS = (
    "ml_regime_s1_kqhedge.toml",
    "ml_regime_s1_kqhedge_tv05.toml",
    "ml_regime_s1_kqhedge_tv15.toml",
    "ml_regime_s2_volcap.toml",
    "ml_regime_s2_volcap20.toml",
    "ml_regime_s2_volcap30.toml",
    "ml_regime_s3_tsmom.toml",
)


def test_staged_strategy_tomls_load_and_differ_only_in_targeted_knobs() -> None:
    """Invariant: every new staged strategy file loads cleanly, and neighbor variants differ
    from their challenger in only the targeted knob."""
    from src.research.champion import knob_changes
    from src.research.pipeline import load_strategy_spec

    strat_dir = Path("config/research/strategies")
    specs = {fname: load_strategy_spec(strat_dir / fname) for fname in _STAGED_REGIME_SPECS}

    # Stage 1: neighbors differ from challenger only in regime_hedge.target_vol
    s1_challenger = specs["ml_regime_s1_kqhedge.toml"]
    assert s1_challenger.regime_hedge is not None
    assert s1_challenger.regime_hedge.target_vol == pytest.approx(0.10)
    for neighbor_name, expected_tv in [
        ("ml_regime_s1_kqhedge_tv05.toml", 0.05),
        ("ml_regime_s1_kqhedge_tv15.toml", 0.15),
    ]:
        neighbor_spec = specs[neighbor_name]
        assert neighbor_spec.regime_hedge is not None
        assert neighbor_spec.regime_hedge.target_vol == pytest.approx(expected_tv)
        diff = knob_changes(neighbor_spec, s1_challenger)
        assert diff == ("regime_hedge.target_vol",)

    # Stage 2: neighbors differ from challenger only in trend_overlay.vol_cap
    s2_challenger = specs["ml_regime_s2_volcap.toml"]
    assert s2_challenger.trend_overlay is not None
    assert s2_challenger.trend_overlay.vol_cap == pytest.approx(0.25)
    for neighbor_name, expected_vc in [
        ("ml_regime_s2_volcap20.toml", 0.20),
        ("ml_regime_s2_volcap30.toml", 0.30),
    ]:
        neighbor_spec = specs[neighbor_name]
        assert neighbor_spec.trend_overlay is not None
        assert neighbor_spec.trend_overlay.vol_cap == pytest.approx(expected_vc)
        diff = knob_changes(neighbor_spec, s2_challenger)
        assert diff == ("trend_overlay.vol_cap",)

    # Stage 3: challenger replaces ma with tsmom
    s3_challenger = specs["ml_regime_s3_tsmom.toml"]
    assert s3_challenger.trend_overlay is not None
    assert s3_challenger.trend_overlay.signal == "tsmom"
    assert s3_challenger.trend_overlay.tsmom_horizons == (21, 63, 126, 252)
    assert s3_challenger.trend_overlay.ma_sessions is None
    assert knob_changes(s3_challenger, s2_challenger) == ("trend_overlay.ma_sessions",)
    assert knob_changes(s2_challenger, s1_challenger) == (
        "trend_overlay.vol_cap",
        "trend_overlay.vol_window_sessions",
    )
    champion = load_strategy_spec(strat_dir / "ml_growth_t85_b50_k200.toml")
    assert set(knob_changes(s1_challenger, champion)) == {
        f"regime_hedge.{name}"
        for name in json.loads(s1_challenger.regime_hedge.canonical_json())
        if name != "tsmom_horizons"
    }


def test_staged_strategy_leg_tax_terms_match() -> None:
    """Invariant: in every staged file, the [regime_hedge] tax terms equal the [trend_overlay] tax terms."""
    from src.research.pipeline import load_strategy_spec

    strat_dir = Path("config/research/strategies")
    for fname in _STAGED_REGIME_SPECS:
        spec = load_strategy_spec(strat_dir / fname)
        assert spec.trend_overlay is not None
        assert spec.regime_hedge is not None
        assert spec.regime_hedge.futures_tax_rate == pytest.approx(spec.trend_overlay.futures_tax_rate)
        assert spec.regime_hedge.futures_annual_deduction_krw == spec.trend_overlay.futures_annual_deduction_krw
        assert spec.regime_hedge.futures_cost_rate == pytest.approx(spec.trend_overlay.futures_cost_rate)


def test_f52_challenger_toml_round_trips_the_feature_set() -> None:
    """Invariant: the f52_n10 challenger TOML round-trips scorer.feature_set and policy.n."""
    from src.research.pipeline import _scores_identity, load_strategy_spec, strategy_spec_from_canonical_json

    strat_dir = Path("config/research/strategies")
    spec = load_strategy_spec(strat_dir / "ml_regime_s2_volcap_f52_n10.toml")
    restored = strategy_spec_from_canonical_json(spec.canonical_json())
    assert restored.scorer.feature_set == "dedup52_v1"
    assert restored.policy.n == 10
    assert restored == spec
    champion = load_strategy_spec(strat_dir / "ml_regime_s2_volcap.toml")
    assert _scores_identity(spec) != _scores_identity(champion)
    assert restored.policy.model_dump() != champion.policy.model_dump()
    champion_policy = champion.policy.model_dump()
    restored_policy = restored.policy.model_dump()
    assert {k: v for k, v in restored_policy.items() if k != "n"} == {
        k: v for k, v in champion_policy.items() if k != "n"
    }
    assert restored.scorer.feature_set != champion.scorer.feature_set
    assert restored.scorer.model_dump(exclude={"feature_set"}) == champion.scorer.model_dump(
        exclude={"feature_set"}
    )
    assert restored.book == champion.book
    assert restored.hedge == champion.hedge
    assert restored.trend_overlay == champion.trend_overlay
    assert restored.regime_hedge == champion.regime_hedge
    for fname, expected_n in (
        ("ml_regime_s2_volcap_f52_n8.toml", 8),
        ("ml_regime_s2_volcap_f52_n12.toml", 12),
        ("ml_regime_s2_volcap_f52_n20.toml", 20),
    ):
        neighbor = load_strategy_spec(strat_dir / fname)
        assert neighbor.scorer.feature_set == "dedup52_v1"
        assert neighbor.policy.n == expected_n
        expected = champion.model_copy(
            update={
                "scorer": champion.scorer.model_copy(update={"feature_set": "dedup52_v1"}),
                "policy": champion.policy.model_copy(update={"n": expected_n}),
            }
        )
        assert neighbor == expected
        assert strategy_spec_from_canonical_json(neighbor.canonical_json()) == neighbor
        assert _scores_identity(neighbor) == _scores_identity(spec)
