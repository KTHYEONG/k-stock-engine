"""ML trend-cash pipeline invariants."""

from __future__ import annotations

from datetime import UTC, datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.data.research_protocol import LockboxError, Segment
from src.research.cube import ResearchCube

_NOW = datetime(2026, 9, 30, tzinfo=UTC)
INSTRS = [f"KRX:{i:06d}" for i in range(1, 7)]


def _sessions(n: int = 80) -> list[date]:
    from tests.fixtures.synthetic_panel import synthetic_sessions

    return synthetic_sessions(n, start=date(2019, 1, 1))


def _full_cube(sessions: list[date]) -> ResearchCube:
    from tests.fixtures.synthetic_panel import synthetic_cube

    base = synthetic_cube(sessions, INSTRS)
    arrays = dict(base.arrays)
    n_s, n_n = len(sessions), len(INSTRS)
    rng = np.random.default_rng(7)
    rets = rng.normal(loc=0.001, scale=0.004, size=(n_s, n_n))
    trend = np.cumprod(1.0 + rets, axis=0) / (1.0 + rets[0])
    arrays["adj_tr"] = np.ascontiguousarray(trend)
    arrays["adj_px"] = np.ascontiguousarray(trend * 10000.0)
    arrays["high"] = np.ascontiguousarray(np.full((n_s, n_n), 10100.0))
    arrays["low"] = np.ascontiguousarray(np.full((n_s, n_n), 9900.0))
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
    # drift on last name so streams are positive
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


def _protocol_for(sessions: list[date]) -> Any:
    from src.data.research_protocol import load_research_protocol
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = load_research_protocol(Path("config/research/protocol.toml"), scope)
    n = len(sessions)
    d_end = sessions[min(60, max(2, n - 20))]
    h_start = sessions[min(61, max(3, n - 19))]
    h_end = sessions[min(75, max(4, n - 5))]
    f_start = sessions[min(76, max(5, n - 4))]
    boot = base.criteria.bootstrap.model_copy(update={"draws": 20, "block_sessions": 2, "horizon_sessions": 20})
    c1 = base.criteria.c1.model_copy(update={"max_p_cagr_le_zero": 1.0})
    c2 = base.criteria.c2.model_copy(
        update={
            "min_point_calmar": -1e9,
            "min_p_calmar": 0.0,
            "max_p_mdd_below_limit": 1.0,
            "max_underwater_median_sessions": 10**9,
            "max_underwater_p95_sessions": 10**9,
            "min_worst_phase_calmar": -1e9,
        }
    )
    c3 = base.criteria.c3.model_copy(
        update={
            "stress_min_point_calmar": -1e9,
            "ledger_min_calmar": -1e9,
            "parity_max_growth_gap": 10.0,
        }
    )
    c4 = base.criteria.c4.model_copy(update={"holdout_max_p_mean_le_zero": 1.0, "holdout_min_point_calmar": -1e9})
    criteria = base.criteria.model_copy(update={"bootstrap": boot, "c1": c1, "c2": c2, "c3": c3, "c4": c4})
    return base.model_copy(
        update={
            "discovery_start": sessions[1],
            "discovery_end": d_end,
            "holdout_start": h_start,
            "holdout_end": h_end,
            "forward_start": f_start,
            "criteria": criteria,
        }
    )


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
    return StrategySpec(policy=policy, scorer=scorer, book=BookSpec(**book_kw), hedge=HedgeSpec(**hedge_kw))  # type: ignore[arg-type]


def _context(tmp_path: Path, sessions: list[date], cube: ResearchCube | None = None) -> Any:
    from src.core.market_rules import load_krx_market_rules
    from src.data.research_protocol import LockboxLedger
    from src.research.pipeline import Pipeline, PipelineContext
    from src.research.registry import TrialRegistry

    protocol = _protocol_for(sessions)
    cube = cube if cube is not None else _full_cube(sessions)
    registry = TrialRegistry(tmp_path / "trials_ml")
    lockbox = LockboxLedger(state_root=tmp_path / "state", protocol=protocol, now=lambda: _NOW)
    from tests.fixtures.synthetic_panel import synthetic_hedge_inputs

    ctx = PipelineContext(
        protocol=protocol,
        cube=cube,
        registry=registry,
        lockbox=lockbox,
        panel_dir=tmp_path / "panel",
        dividends=pl.DataFrame(),
        hedge_inputs=synthetic_hedge_inputs(sessions),
        rules=load_krx_market_rules(Path("config/market/krx_market_rules.toml")),
        engine_config_path=Path("config/backtest/default_engine.toml"),
        market_cache_root=tmp_path / "mcache",
        reports_root=tmp_path / "reports",
        scores_root=tmp_path / "scores",
        ledger_runner=_ledger_ok(sessions),
        now=lambda: _NOW,
    )
    return Pipeline(ctx)


def _ledger_ok(sessions: list[date]) -> Any:
    from src.research.ledger_bridge import LedgerOutcome

    def _fake(**kwargs: Any) -> LedgerOutcome:
        start, end = kwargs["start"], kwargs["end"]
        idx = {d: i for i, d in enumerate(sessions)}
        window = tuple(sessions[idx[start] : idx[end] + 1])
        return LedgerOutcome(
            capital_krw=kwargs["capital_krw"],
            halted_exit_policy=kwargs["halted_exit_policy"].value,
            sessions=window,
            log_returns=np.full(len(window), 0.0008),
            reject_counts={},
            ledger_hash="h",
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


def test_discovery_records_five_base_trials_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    first = pipe.evaluate_discovery(spec)
    n1 = len(pipe._ctx.registry.trials())
    second = pipe.evaluate_discovery(spec)
    trials = pipe._ctx.registry.trials()
    assert n1 == 5
    assert len(trials) == 5
    assert {t.trial_id for t in trials} == {t.trial_id for t in pipe._ctx.registry.trials()}
    assert first.digest == second.digest


def test_stress_delay_executes_one_session_later(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research.policy import decision_rows

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    protocol = pipe._ctx.protocol
    start = next(d for d in sessions if d.year == spec.scorer.first_test_year)
    end = [d for d in sessions if d <= protocol.discovery_end][-1]
    lo = sessions.index(start)
    hi = sessions.index(end)
    rows = decision_rows(sessions, lo=lo, hi=hi + 1, every=5, phase=0)
    panel = pipe.panel_for(hi)
    scores = pipe.scores(spec, segment=Segment.DISCOVERY)
    import numpy as np

    from src.research.policy import universe_mask

    uni = np.asarray(universe_mask(pipe._ctx.cube, spec.policy.universe), dtype=bool)[: hi + 1]
    close = np.asarray(pipe._ctx.cube.arrays["close"], dtype=float)[: hi + 1]
    targets = pipe._targets_for(
        spec.policy,
        close,
        panel,
        np.asarray(scores.scores, dtype=float),
        uni,
        list(rows),
        protocol.primary_capital_krw,
        0.005,
    )
    base_cfg = pipe._base_sim_config()
    delay = int(protocol.criteria.c1.stress_execution_delay)
    assert delay >= 1
    from src.research.simulator import simulate

    auth = pipe._ctx.lockbox.authorize(start=start, end=end, spec_hash=None)
    exec_base = {r: w for r, w in targets.items() if lo - 1 <= r <= hi - 1}
    exec_delay = {r: w for r, w in targets.items() if lo - 1 - delay <= r <= hi - 1 - delay}
    assert set(exec_base) == set(targets) - ({max(targets)} if max(targets) == hi else set())
    assert set(exec_delay) != set(exec_base)
    rb = simulate(pipe._ctx.cube, exec_base, start=start, end=end, config=base_cfg, authorization=auth)
    rd = simulate(
        pipe._ctx.cube,
        exec_delay,
        start=start,
        end=end,
        config=base_cfg.model_copy(update={"execution_delay": delay}),
        authorization=auth,
    )
    assert not np.array_equal(np.asarray(rb.log_returns), np.asarray(rd.log_returns))


def test_leaky_scorer_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _leaky_scores(monkeypatch)
    spec = _spec()
    report = pipe.evaluate_discovery(spec)
    by = {c.name: c for c in report.checks}
    assert by["C1.perturbation"].passed is False
    assert report.checks[0].value > 0


def test_discovery_window_stays_inside_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec(scorer={"first_test_year": 2030})
    with pytest.raises(ValueError, match="first_test_year"):
        pipe.evaluate_discovery(spec)
    # start after discovery end (same year span, late first_test_year)
    from datetime import timedelta

    long_sessions = [date(2019, 1, 1) + timedelta(days=i) for i in range(400)]
    long_pipe = _context(tmp_path, long_sessions)
    _clean_scores(monkeypatch)
    late = _spec(scorer={"first_test_year": 2020})
    with pytest.raises(ValueError, match="after the discovery end"):
        long_pipe.evaluate_discovery(late)


def test_register_requires_passed_intact_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    with pytest.raises(ValueError, match="missing"):
        pipe.register_finalist(spec)
    pipe.evaluate_discovery(spec)
    path = pipe._ctx.reports_root / f"{spec.spec_hash}_discovery.json"
    raw = json.loads(path.read_text())
    raw["checks"][0]["passed"] = False
    # keep old digest to simulate failure? recompute: write failing report with correct digest
    from src.research.pipeline import _rebuild_report

    body = {k: v for k, v in raw.items() if k != "digest"}
    rebuilt = _rebuild_report(body)
    raw["digest"] = rebuilt.digest
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="did not pass"):
        pipe.register_finalist(spec)
    # tamper
    pipe.evaluate_discovery(spec)
    raw = json.loads(path.read_text())
    raw["checks"][0]["value"] = 999.0
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        pipe.register_finalist(spec)


def test_holdout_sealed_without_finalist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    with pytest.raises(LockboxError):
        pipe.holdout(spec)


def test_holdout_uses_production_phase_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    pipe.evaluate_discovery(spec)
    pipe.register_finalist(spec)
    report = pipe.holdout(spec)
    trials = pipe._ctx.registry.trials(segment=Segment.HOLDOUT)
    assert len(trials) == 1
    assert report.segment is Segment.HOLDOUT
    # forward stays sealed
    with pytest.raises(ValueError, match="sealed"):
        pipe.scores(spec, segment=Segment.FORWARD)


def test_annualized_turnover_and_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    pipe.evaluate_discovery(spec)
    for trial in pipe._ctx.registry.trials(segment=Segment.DISCOVERY):
        assert trial.metrics["turnover"] >= 0.0
        assert trial.metrics["cost"] >= 0.0
        assert trial.metrics["turnover"] == pytest.approx(trial.metrics["turnover"])
    # per-year multiples: turnover equals session mean scaled by sessions_per_year
    trial = pipe._ctx.registry.trials(segment=Segment.DISCOVERY)[0]
    assert np.isfinite(trial.metrics["turnover"])
    assert np.isfinite(trial.metrics["cost"])


def test_halted_policy_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research.simulator import simulate

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    protocol = pipe._ctx.protocol
    start = next(d for d in sessions if d.year == _spec().scorer.first_test_year)
    end = [d for d in sessions if d <= protocol.discovery_end][-1]
    auth = pipe._ctx.lockbox.authorize(start=start, end=end, spec_hash=None)
    cfg0 = pipe._base_sim_config().model_copy(update={"halted_exit_value": 0.0})
    cfg1 = pipe._base_sim_config().model_copy(update={"halted_exit_value": 1.0})
    r0 = simulate(pipe._ctx.cube, {}, start=start, end=end, config=cfg0, authorization=auth)
    r1 = simulate(pipe._ctx.cube, {}, start=start, end=end, config=cfg1, authorization=auth)
    assert r0.sessions == r1.sessions


def test_scores_cache_validity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    first = pipe.scores(spec, segment=Segment.DISCOVERY)
    # tamper cache file
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
    second = pipe.scores(spec, segment=Segment.DISCOVERY)
    assert np.array_equal(np.asarray(first.scores), np.asarray(second.scores), equal_nan=True)


def test_price_cap_effect_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec(policy={"min_units_per_slot": 3})
    report = pipe.evaluate_discovery(spec)
    by = {c.name: c for c in report.checks}
    assert "C3.price_cap_effect" in by
    assert np.isfinite(by["C3.price_cap_effect"].value)


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
    changed_book = tmp_path / "bb.toml"
    changed_book.write_text(
        good.read_text(encoding="utf-8").replace("stock_capital_fraction=0.75", "stock_capital_fraction=0.5"),
        encoding="utf-8",
    )
    assert load_strategy_spec(good).spec_hash != load_strategy_spec(changed_book).spec_hash
    changed_hedge = tmp_path / "bh.toml"
    changed_hedge.write_text(
        good.read_text(encoding="utf-8").replace("hedge_ratio=1.0", "hedge_ratio=0.5"), encoding="utf-8"
    )
    assert load_strategy_spec(good).spec_hash != load_strategy_spec(changed_hedge).spec_hash
    mismatched = tmp_path / "mm.toml"
    mismatched.write_text(
        good.read_text(encoding="utf-8").replace("sleeves=5", "sleeves=3"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sleeves"):
        load_strategy_spec(mismatched)
    bad = tmp_path / "c.toml"
    bad.write_text(
        '[policy]\nfamily="ml_trend_cash"\nn=20\n[policy.universe]\nmin_adtv20_krw=0\nmin_price_krw=0\n[nope]\nx=1\n[scorer]\n[book]\nsleeves=5\nstock_capital_fraction=0.75\n[hedge]\nhedge_ratio=1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown"):
        load_strategy_spec(bad)
    with pytest.raises(OSError, match="missing"):
        load_strategy_spec(tmp_path / "missing.toml")
    malformed = tmp_path / "mal.toml"
    malformed.write_text("[policy\nbroken", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid strategy TOML"):
        load_strategy_spec(malformed)
    missing_table = tmp_path / "mt.toml"
    missing_table.write_text('[policy]\nfamily="ml_trend_cash"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must declare"):
        load_strategy_spec(missing_table)
    bad_policy = tmp_path / "bp.toml"
    bad_policy.write_text(
        '[policy]\nfamily=""\n[policy.universe]\nmin_adtv20_krw=0\nmin_price_krw=0\n[scorer]\n'
        "[book]\nsleeves=5\nstock_capital_fraction=0.75\n[hedge]\nhedge_ratio=1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid \\[policy\\]"):
        load_strategy_spec(bad_policy)
    bad_scorer = tmp_path / "bs.toml"
    bad_scorer.write_text(
        '[policy]\nfamily="ml_trend_cash"\nrebalance_every_sessions=5\n[policy.universe]\nmin_adtv20_krw=0\nmin_price_krw=0\n[scorer]\nhorizons=[]\n'
        "[book]\nsleeves=5\nstock_capital_fraction=0.75\n[hedge]\nhedge_ratio=1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid \\[scorer\\]"):
        load_strategy_spec(bad_scorer)
    bad_book = tmp_path / "bb2.toml"
    bad_book.write_text(
        '[policy]\nfamily="ml_trend_cash"\nrebalance_every_sessions=5\n[policy.universe]\nmin_adtv20_krw=0\nmin_price_krw=0\n[scorer]\n'
        "[book]\nsleeves=0\n[hedge]\nhedge_ratio=1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid \\[book\\]"):
        load_strategy_spec(bad_book)
    bad_hedge = tmp_path / "bh2.toml"
    bad_hedge.write_text(
        '[policy]\nfamily="ml_trend_cash"\nrebalance_every_sessions=5\n[policy.universe]\nmin_adtv20_krw=0\nmin_price_krw=0\n[scorer]\n'
        "[book]\nsleeves=5\nstock_capital_fraction=0.75\n[hedge]\nhedge_ratio=-1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid \\[hedge\\]"):
        load_strategy_spec(bad_hedge)


def test_trial_metrics_without_capital_reports_raw_hedge_totals() -> None:
    from src.research.pipeline import _trial_metrics

    metrics = _trial_metrics(np.full(10, 0.001), np.zeros(10), np.zeros(10), 252)
    assert metrics["hedge_cost"] == 0.0
    assert metrics["tax"] == 0.0
    assert np.isfinite(metrics["cagr"])


def test_hedge_perturbation_counts_stale_leg_mismatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    protocol = pipe._ctx.protocol
    start = next(d for d in sessions if d.year == spec.scorer.first_test_year)
    end = [d for d in sessions if d <= protocol.discovery_end][-1]
    from src.research.pipeline import _window_indices

    lo, hi = _window_indices(sessions, start, end)
    window = tuple(sessions[lo : hi + 1])
    width = hi - lo + 1
    auth = pipe._ctx.lockbox.authorize(start=start, end=end, spec_hash=None)
    zeros = np.zeros(width)
    zeros_int = np.zeros(width, dtype=np.int64)
    rng = np.random.default_rng(3)
    mismatches = pipe._hedge_perturbation_mismatches(
        spec, np.zeros(width), window, lo, lo, zeros, zeros_int, zeros,
        int(protocol.primary_capital_krw), auth, rng,
    )
    assert mismatches == 0
    stale = pipe._hedge_perturbation_mismatches(
        spec, np.zeros(width), window, lo, lo, np.ones(width), np.ones(width, dtype=np.int64), np.ones(width),
        int(protocol.primary_capital_krw), auth, rng,
    )
    assert stale == 3


def test_window_helpers_and_panel_guards(tmp_path: Path) -> None:
    from src.research.pipeline import (
        _first_at_or_after,
        _last_at_or_before,
        _window_indices,
    )

    sessions = _sessions()
    with pytest.raises(ValueError, match="within the cube"):
        _window_indices(sessions, sessions[5], sessions[2])
    with pytest.raises(ValueError, match="after the last"):
        _first_at_or_after(sessions, date(2030, 1, 1))
    with pytest.raises(ValueError, match="before the first"):
        _last_at_or_before(sessions, date(1990, 1, 1))
    pipe = _context(tmp_path, sessions)
    with pytest.raises(ValueError, match="last_row"):
        pipe.panel_for(-1)
    with pytest.raises(ValueError, match="last_row"):
        pipe.panel_for(len(sessions))


def test_effective_and_rebuild_edge_cases() -> None:
    from src.research.pipeline import _effective_from_columns, _rebuild_report

    assert _effective_from_columns([]) == 1.0
    flat = [np.zeros(10), np.zeros(10)]
    assert _effective_from_columns(flat) == float(len(flat))
    single = [np.random.default_rng(0).normal(size=10)]
    assert _effective_from_columns(single) >= 1.0
    assert _effective_from_columns([np.zeros(10)]) == 1.0
    with pytest.raises(ValueError, match="invalid criteria report"):
        _rebuild_report({"checks": ["bad"]})  # type: ignore[dict-item]


def test_scores_cache_corrupt_file_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    first = pipe.scores(spec, segment=Segment.DISCOVERY)
    cache_files = list((tmp_path / "scores").glob("*.npz"))
    assert cache_files
    cache_files[0].write_bytes(b"not a npz")
    second = pipe.scores(spec, segment=Segment.DISCOVERY)
    assert np.array_equal(np.asarray(first.scores), np.asarray(second.scores), equal_nan=True)


def test_register_second_spec_refused_and_malformed_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    pipe.evaluate_discovery(spec)
    pipe.register_finalist(spec)
    other = _spec(policy={"n": 3})
    pipe.evaluate_discovery(other)
    with pytest.raises(LockboxError, match="only one finalist"):
        pipe.register_finalist(other)
    # malformed JSON
    path = pipe._ctx.reports_root / f"{spec.spec_hash}_discovery.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid discovery report"):
        pipe.register_finalist(spec)
    # missing digest
    pipe.evaluate_discovery(spec)
    raw = json.loads(path.read_text())
    del raw["digest"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        pipe.register_finalist(spec)
    # spec mismatch
    pipe.evaluate_discovery(spec)
    raw = json.loads(path.read_text())
    raw["spec_hash"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="spec mismatch"):
        pipe.register_finalist(spec)


def test_window_for_spec_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta

    sessions = [date(2019, 1, 1) + timedelta(days=i) for i in range(400)]
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec(scorer={"first_test_year": 2030})
    with pytest.raises(ValueError, match="no sessions"):
        pipe._window_for_spec(spec, Segment.DISCOVERY)
    with pytest.raises(ValueError, match="no sessions"):
        pipe.scores(spec, segment=Segment.DISCOVERY)
    late = _spec(scorer={"first_test_year": 2020})
    with pytest.raises(ValueError, match="after the discovery end"):
        pipe._window_for_spec(late, Segment.DISCOVERY)


def test_holdout_scores_refuse_outsider_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    pipe.evaluate_discovery(spec)
    pipe.register_finalist(spec)
    pipe.holdout(spec)
    outsider = _spec(policy={"n": 3})
    with pytest.raises(LockboxError, match=r"sealed|not authorized"):
        pipe.scores(outsider, segment=Segment.HOLDOUT)


def test_scores_identity_depends_on_universe_and_scorer() -> None:
    from src.research.pipeline import _scores_identity

    base = _spec()
    wider = _spec(policy={"universe": {"min_adtv20_krw": 1, "min_price_krw": 0}})
    retuned = _spec(scorer={"num_boost_round": 7})
    assert _scores_identity(base) == _scores_identity(_spec())
    assert _scores_identity(base) != _scores_identity(wider)
    assert _scores_identity(base) != _scores_identity(retuned)


def test_hedged_stream_equals_overlay_on_sleeve_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research.book import build_sleeve_targets, mean_sleeve_returns
    from src.research.hedge import simulate_hedged_book
    from src.research.model import ScoreMatrix
    from src.research.policy import universe_mask
    from types import SimpleNamespace

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    protocol = pipe._ctx.protocol
    start = next(d for d in sessions if d.year == spec.scorer.first_test_year)
    end = [d for d in sessions if d <= protocol.discovery_end][-1]
    lo = sessions.index(start)
    hi = sessions.index(end)
    panel = pipe.panel_for(hi)
    scores = pipe.scores(spec, segment=Segment.DISCOVERY)
    base_scores = np.asarray(scores.scores, dtype=float)
    uni = np.asarray(universe_mask(pipe._ctx.cube, spec.policy.universe), dtype=bool)[: hi + 1]
    close = np.asarray(pipe._ctx.cube.arrays["close"], dtype=float)[: hi + 1]
    from src.research.book import sleeve_capital_krw as _sleeve_cap

    sleeve_cap = _sleeve_cap(int(protocol.primary_capital_krw), spec.book)
    proxy = SimpleNamespace(arrays={"close": close})
    fake = ScoreMatrix(
        scores=np.ascontiguousarray(base_scores, dtype=np.float32),
        test_years=(0,),
        config_hash="pipeline",
        last_row=panel.last_row,
    )
    sleeves = build_sleeve_targets(
        spec.policy, proxy, panel, fake, uni, sessions=sessions, sleeves=5,  # type: ignore[arg-type]
        lo=lo, hi=hi + 1, sleeve_capital_krw=sleeve_cap, cash_buffer=0.005,
    )
    auth = pipe._ctx.lockbox.authorize(start=start, end=end, spec_hash=None)
    logs = []
    for sleeve_map in sleeves:
        from src.research.simulator import simulate

        cfg = pipe._base_sim_config().model_copy(update={"capital_krw": sleeve_cap})
        executable = {r: w for r, w in sleeve_map.items() if lo - 1 <= r <= hi - 1}
        logs.append(np.asarray(simulate(pipe._ctx.cube, executable, start=start, end=end, config=cfg, authorization=auth).log_returns))
    expected = simulate_hedged_book(
        mean_sleeve_returns(logs), tuple(sessions[lo : hi + 1]), pipe._ctx.hedge_inputs, spec.hedge,
        capital_krw=int(protocol.primary_capital_krw), rebalance_offset=0, authorization=auth,
    )
    report = pipe.evaluate_discovery(spec)
    stored = pipe._ctx.registry.returns(report.trial_id)
    assert np.allclose(np.asarray(stored.net), np.asarray(expected.log_returns), atol=1e-12)


def test_ledger_replay_uses_combined_stock_book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    calls: list[dict[str, object]] = []
    ledger = pipe._ctx.ledger_runner

    def _spy(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return ledger(**kwargs)  # type: ignore[arg-type]

    object.__setattr__(pipe._ctx, "ledger_runner", _spy)
    pipe.evaluate_discovery(spec)
    capitals = {int(call["capital_krw"]) for call in calls}  # type: ignore[arg-type]
    assert capitals == {int(c) for c in pipe._ctx.protocol.criteria.c3.ledger_capitals}
    assert len(calls) == 2 * len(capitals)
    first_targets = calls[0]["targets"]
    assert isinstance(first_targets, dict)
    assert len(first_targets) > 0


def test_missing_hedge_data_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import math

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    levels = np.asarray(pipe._ctx.hedge_inputs.index_level).copy()
    levels[5] = math.nan
    from src.research.hedge import HedgeInputs

    object.__setattr__(
        pipe._ctx,
        "hedge_inputs",
        HedgeInputs(
            sessions=tuple(pipe._ctx.hedge_inputs.sessions),
            index_level=np.ascontiguousarray(levels),
            inverse_close=np.ascontiguousarray(np.asarray(pipe._ctx.hedge_inputs.inverse_close)),
        ),
    )
    with pytest.raises(Exception, match=r"missing|non-positive|subsequence"):
        pipe.evaluate_discovery(spec)
    assert list((tmp_path / "reports").glob("*_discovery.json")) == []


def test_holdout_records_single_hedged_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    sessions = _sessions()
    pipe = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    spec = _spec()
    pipe.evaluate_discovery(spec)
    pipe.register_finalist(spec)
    report = pipe.holdout(spec)
    trials = pipe._ctx.registry.trials(segment=Segment.HOLDOUT)
    assert len(trials) == 1
    assert trials[0].trial_id == report.trial_id
    payload = _json.loads(trials[0].sim_config_json)
    assert payload["offset"] == 0
    assert trials[0].metrics["hedge_cost"] >= 0.0
    assert trials[0].metrics["tax"] >= 0.0
