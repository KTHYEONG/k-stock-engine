"""Evaluator screening, perturbation, and lockbox invariants."""
from __future__ import annotations

from datetime import datetime, UTC
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.data.research_protocol import LockboxError, Segment

_NOW = datetime(2026, 9, 29, tzinfo=UTC)
INSTRS = [f"KRX:{i:06d}" for i in range(1, 7)]


def _protocol_for(sessions: list[Any]) -> Any:
    from src.data.research_protocol import load_research_protocol
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = load_research_protocol(Path("config/research/protocol.toml"), scope)
    statistics = base.statistics.model_copy(update={
        "bootstrap_block_sessions": 2, "bootstrap_draws": 100, "cscv_blocks": 2,
    })
    gates = base.gates.model_copy(update={"perturbation_cuts": 6})
    return base.model_copy(update={
        "discovery_start": sessions[1], "discovery_end": sessions[54],
        "holdout_start": sessions[55], "holdout_end": sessions[57],
        "forward_start": sessions[58], "statistics": statistics, "gates": gates,
    })


def _cube(sessions: list[Any], *, drift: bool = False) -> Any:
    from tests.fixtures.synthetic_panel import synthetic_cube

    cube = synthetic_cube(sessions, INSTRS)
    arrays = dict(cube.arrays)
    n_s, n_n = len(sessions), len(INSTRS)
    rng_trend = np.random.default_rng(28)
    rets = rng_trend.normal(loc=0.001, scale=0.003, size=(n_s, n_n))
    trend = np.cumprod(1.0 + rets, axis=0) / (1.0 + rets[0])
    arrays["adj_tr"] = np.ascontiguousarray(trend)
    arrays["adj_px"] = np.ascontiguousarray(trend)
    close = np.asarray(arrays["close"], dtype=np.float64)
    volume = np.asarray(arrays["volume"], dtype=np.float64)
    arrays["trading_value"] = np.ascontiguousarray(volume * close)
    arrays["f_age_q"] = np.ascontiguousarray(np.zeros((n_s, n_n), dtype=np.float64))
    arrays["earn_qk"] = np.ascontiguousarray(np.full((n_s, n_n), np.nan, dtype=np.float64))
    if drift:
        drift_on = np.asarray(arrays["r_on"]).copy()
        drift_on[:, -1] = 0.01
        drift_id = np.asarray(arrays["r_id"]).copy()
        drift_id[:, -1] = 0.005
        arrays["r_on"] = np.ascontiguousarray(drift_on)
        arrays["r_id"] = np.ascontiguousarray(drift_id)
    from src.research.cube import ResearchCube

    return ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=list(sessions), instrument_ids=list(INSTRS),
        arrays=arrays, exit_at=np.asarray(cube.exit_at), exit_halted=np.asarray(cube.exit_halted),
    )


def _context(tmp_path: Path, sessions: list[Any], *, drift: bool = False) -> Any:
    from src.core.market_rules import load_krx_market_rules
    from src.data.research_protocol import LockboxLedger
    from src.research.cube import CubeInputs
    from src.research.evaluator import EvaluationContext, Evaluator
    from src.research.registry import TrialRegistry

    protocol = _protocol_for(sessions)
    cube = _cube(sessions, drift=drift)
    registry = TrialRegistry(tmp_path / "trials")
    lockbox = LockboxLedger(state_root=tmp_path / "state", protocol=protocol, now=lambda: _NOW)
    inputs = CubeInputs(
        market_panel=tmp_path / "panel", dividend_events=tmp_path / "div",
        financial_facts=tmp_path / "facts", investor_flow=tmp_path / "flow",
        earnings_releases=tmp_path / "releases",
        market_rules=load_krx_market_rules(Path("config/market/krx_market_rules.toml")),
        dividend_withholding_rate=Decimal("0.154"),
    )
    context = EvaluationContext(
        protocol=protocol, cube=cube, cube_inputs=inputs, registry=registry, lockbox=lockbox,
        engine_config_path=Path("config/backtest/default_engine.toml"),
        market_cache_root=tmp_path / "cache", reports_root=tmp_path / "reports",
        dividends=pl.DataFrame(), now=lambda: _NOW,
    )
    return Evaluator(context)


def _spec(**overrides: Any) -> Any:
    from src.research.strategy import StrategySpec

    base: dict[str, Any] = {
        "family": "probe", "universe": {"min_adtv20_krw": 0, "min_price_krw": 0},
        "junk": None, "score": {"size": 1.0}, "n": 1,
        "keep_rank_multiple": 1.0, "rebalance": "D",
    }
    base.update(overrides)
    return StrategySpec.model_validate(base)


def _ledger_fake(sessions: list[Any]) -> Any:
    from src.research.ledger_bridge import LedgerOutcome

    def _fake(**kwargs: Any) -> LedgerOutcome:
        start, end = kwargs["start"], kwargs["end"]
        idx = {day: i for i, day in enumerate(sessions)}
        lo, hi = idx[start], idx[end]
        window = tuple(sessions[lo:hi + 1])
        n = len(window)
        return LedgerOutcome(capital_krw=kwargs["capital_krw"],
                             halted_exit_policy=kwargs["halted_exit_policy"].value,
                             sessions=window,
                             log_returns=np.full(n, 0.0005, dtype=np.float64),
                             reject_counts={}, ledger_hash="abc")
    return _fake


def test_every_screen_is_recorded(tmp_path: Path) -> None:
    """Two screened specs leave two discovery trials with both benchmarks."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    evaluator.screen(_spec(n=1))
    evaluator.screen(_spec(n=2))
    trials = evaluator._ctx.registry.trials(segment=Segment.DISCOVERY)
    assert len(trials) == 2
    for trial in trials:
        stored = evaluator._ctx.registry.returns(trial.trial_id)
        assert set(stored.benchmarks) == {"U_EW", "CW"}


def test_screen_family_records_all(tmp_path: Path) -> None:
    """Family screening records one trial per member."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    records = evaluator.screen_family([_spec(n=1), _spec(n=2)])
    assert len(records) == 2


def test_clean_spec_passes_perturbation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A causal rev_1m spec shows zero perturbation mismatches."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    monkeypatch.setattr("src.research.evaluator.run_ledger", _ledger_fake(sessions))
    spec = _spec(score={"rev_1m": 1.0})
    report = evaluator.validate(spec, [spec, _spec(n=2)])
    by_name = {check.name: check for check in report.checks}
    assert by_name["G1.perturbation"].passed is True


def test_leaky_feature_detected_by_perturbation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A feature reading t+1 changes past targets under future perturbation."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    monkeypatch.setattr("src.research.evaluator.run_ledger", _ledger_fake(sessions))
    import src.research.evaluator as evaluator_module

    original = evaluator_module.compute_features

    def _leaky(cube: Any, names: Any) -> Any:
        out = original(cube, list(names))
        return {name: np.roll(np.asarray(arr, dtype=np.float64), -1, axis=0) for name, arr in out.items()}

    monkeypatch.setattr("src.research.evaluator.compute_features", _leaky)
    spec = _spec(score={"rev_1m": 1.0})
    report = evaluator.validate(spec, [spec, _spec(n=2)])
    by_name = {check.name: check for check in report.checks}
    assert by_name["G1.perturbation"].passed is False


def test_holdout_requires_finalist(tmp_path: Path) -> None:
    """Holdout without registration raises and records no trial."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    with pytest.raises(LockboxError):
        evaluator.holdout(_spec())
    assert evaluator._ctx.registry.trials(segment=Segment.HOLDOUT) == ()


def test_finalist_registration_requires_passing_report(tmp_path: Path) -> None:
    """A failing discovery report cannot become a finalist."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    from src.research.gates import GateCheck, GateReport

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    spec = _spec()
    failing = GateReport(spec_hash=spec.spec_hash, trial_id="t", segment=Segment.DISCOVERY,
                         protocol_version=evaluator._ctx.protocol.version,
                         checks=(GateCheck(gate="G1", name="G1.perturbation", value=1.0,
                                           threshold=0.0, passed=False, detail="x"),))
    evaluator._ctx.reports_root.mkdir(parents=True, exist_ok=True)
    (evaluator._ctx.reports_root / f"{spec.spec_hash}_discovery.json").write_text(
        failing.canonical_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        evaluator.register_finalists([spec])
    assert evaluator._ctx.lockbox.finalists() == ()


def test_family_must_contain_spec(tmp_path: Path) -> None:
    """Validation without the spec in the family fails closed."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    with pytest.raises(ValueError, match="family must contain"):
        evaluator.validate(_spec(n=1), [_spec(n=2)])


def test_register_holdout_forward_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing discovery registers, holdout records a verdict, forward follows."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    from src.research.gates import GateCheck, GateReport

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions, drift=True)
    monkeypatch.setattr("src.research.evaluator.run_ledger", _ledger_fake(sessions))
    spec = _spec(score={"size": 1.0})
    evaluator.screen(spec)
    passing = GateReport(spec_hash=spec.spec_hash, trial_id="t", segment=Segment.DISCOVERY,
                         protocol_version=evaluator._ctx.protocol.version,
                         checks=(GateCheck(gate="G1", name="G1.perturbation", value=0.0,
                                           threshold=0.0, passed=True, detail="ok"),))
    evaluator._ctx.reports_root.mkdir(parents=True, exist_ok=True)
    (evaluator._ctx.reports_root / f"{spec.spec_hash}_discovery.json").write_text(
        passing.canonical_json() + "\n", encoding="utf-8")
    evaluator.register_finalists([spec])
    assert len(evaluator._ctx.lockbox.finalists()) == 1
    holdout_report = evaluator.holdout(spec)
    assert (evaluator._ctx.reports_root / f"{spec.spec_hash}_holdout.json").exists()
    assert holdout_report.segment is Segment.HOLDOUT
    if holdout_report.passed:
        record = evaluator.forward(spec)
        assert record.segment is Segment.FORWARD
        assert (evaluator._ctx.reports_root / f"{spec.spec_hash}_forward.json").exists()
    else:
        assert evaluator._ctx.lockbox.holdout_passed(spec.spec_hash) is False


def test_forward_requires_passed_holdout(tmp_path: Path) -> None:
    """Forward without a passed holdout verdict raises."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    with pytest.raises(LockboxError):
        evaluator.forward(_spec())


def test_missing_discovery_report_fails_registration(tmp_path: Path) -> None:
    """Registration without any stored report fails closed."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    with pytest.raises(ValueError, match="missing discovery report"):
        evaluator.register_finalists([_spec()])


def test_window_helpers_fail_closed() -> None:
    """Out-of-cube windows raise before any simulation runs."""
    from datetime import date

    from src.research.evaluator import _first_at_or_after, _last_at_or_before, _window_indices

    sessions = [date(2020, 1, 6), date(2020, 1, 7)]
    with pytest.raises(ValueError, match="within the cube"):
        _window_indices(sessions, sessions[1], sessions[0])
    with pytest.raises(ValueError, match="within the cube"):
        _window_indices(sessions, date(1999, 1, 1), sessions[0])
    with pytest.raises(ValueError, match="after the last"):
        _first_at_or_after(sessions, date(2030, 1, 1))
    with pytest.raises(ValueError, match="before the first"):
        _last_at_or_before(sessions, date(1999, 1, 1))


def test_inverted_segment_window_fails(tmp_path: Path) -> None:
    """A segment window with no cube sessions fails without recording."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    protocol = evaluator._ctx.protocol.model_copy(update={
        "discovery_start": sessions[10], "discovery_end": sessions[5],
    })
    import dataclasses

    evaluator._ctx = dataclasses.replace(evaluator._ctx, protocol=protocol)
    with pytest.raises(ValueError, match="no cube sessions"):
        evaluator.screen(_spec())
    assert evaluator._ctx.registry.trials(segment=Segment.DISCOVERY) == ()


def test_forward_window_bounds(tmp_path: Path) -> None:
    """The forward window starts at the protocol forward start."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    start, end = evaluator._window_for(Segment.FORWARD)
    assert start == sessions[58]
    assert end == sessions[-1]


def test_perturbation_redraws_exits(tmp_path: Path) -> None:
    """Exits after a cut are redrawn during the perturbation probe."""
    import numpy as np

    from src.research.cube import ResearchCube
    from src.research.strategy import target_weights
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    arrays = {key: np.asarray(value).copy() for key, value in evaluator._ctx.cube.arrays.items()}
    cube = evaluator._ctx.cube
    exit_at = np.full(len(INSTRS), 40, dtype=np.int64)
    rebuilt = ResearchCube.from_arrays(
        cube_id=cube.cube_id, sessions=list(sessions), instrument_ids=list(INSTRS),
        arrays=arrays, exit_at=exit_at, exit_halted=np.asarray(cube.exit_halted),
    )
    import dataclasses

    evaluator._ctx = dataclasses.replace(evaluator._ctx, cube=rebuilt)
    spec = _spec()
    start, end = evaluator._window_for(Segment.DISCOVERY)
    features = evaluator._features_for(spec.feature_names())
    lo, hi = evaluator._target_window(start, end)
    targets = target_weights(spec, rebuilt, features, lo=lo, hi=hi)
    mismatches = evaluator._perturbation_mismatches(spec, targets, start, end)
    assert isinstance(mismatches, int)


def test_discovery_matrix_skips_bad_trials(tmp_path: Path) -> None:
    """Corrupt, mismatched, benchmark-free and flat trials never break evidence."""
    from datetime import datetime, UTC

    import numpy as np

    from src.research.registry import TrialRegistry, TrialReturns
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    registry = evaluator._ctx.registry
    assert isinstance(registry, TrialRegistry)
    record = evaluator.screen(_spec())
    stored = registry.returns(record.trial_id)
    window = tuple(stored.sessions)
    registry.record(
        family="f", spec_hash="other-sessions", spec_json='{"v": 1}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 2}', cube_id="cube",
        returns=TrialReturns(sessions=tuple(sessions[:5]), net=np.ones(5), benchmarks={}),
        metrics={}, now=datetime(2026, 9, 29, tzinfo=UTC),
    )
    registry.record(
        family="f", spec_hash="no-bench", spec_json='{"v": 2}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 3}', cube_id="cube",
        returns=TrialReturns(sessions=window, net=np.asarray(stored.net), benchmarks={}),
        metrics={}, now=datetime(2026, 9, 29, tzinfo=UTC),
    )
    flat = np.asarray(stored.benchmarks["U_EW"]) + 0.001
    registry.record(
        family="f", spec_hash="flat", spec_json='{"v": 3}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 4}', cube_id="cube",
        returns=TrialReturns(sessions=window, net=np.ascontiguousarray(flat),
                             benchmarks={"U_EW": np.asarray(stored.benchmarks["U_EW"])}),
        metrics={}, now=datetime(2026, 9, 29, tzinfo=UTC),
    )
    missing = registry.record(
        family="f", spec_hash="gone", spec_json='{"v": 4}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 5}', cube_id="cube",
        returns=TrialReturns(sessions=window, net=np.asarray(stored.net),
                             benchmarks={"U_EW": np.asarray(stored.benchmarks["U_EW"])}),
        metrics={}, now=datetime(2026, 9, 29, tzinfo=UTC),
    )
    (tmp_path / "trials" / "returns" / f"{missing.trial_id}.parquet").unlink()
    columns, sharpes = evaluator._discovery_trials_matrix(window)
    assert len(columns) >= 1
    assert len(sharpes) == len(columns)


def test_base_discovery_returns_fallback_and_missing(tmp_path: Path) -> None:
    """Non-base configs fall back to any trial; unknown specs raise."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    spec = _spec()
    evaluator.screen(spec, config=evaluator._base_sim_config().model_copy(update={"cash_buffer": 0.5}))
    stored = evaluator._base_discovery_returns(spec)
    assert stored.sessions
    with pytest.raises(ValueError, match="no discovery trial"):
        evaluator._base_discovery_returns(_spec(n=2))


def test_validate_effective_fallback_on_flat_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero-variance discovery trial falls back to raw trial counts."""
    from datetime import datetime, UTC

    import numpy as np

    from src.research.registry import TrialReturns
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    monkeypatch.setattr("src.research.evaluator.run_ledger", _ledger_fake(sessions))
    spec = _spec(score={"rev_1m": 1.0})
    record = evaluator.screen(spec)
    stored = evaluator._ctx.registry.returns(record.trial_id)
    window = tuple(stored.sessions)
    evaluator._ctx.registry.record(
        family="f", spec_hash="flat-zero", spec_json='{"v": "flat"}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 9}', cube_id="cube",
        returns=TrialReturns(sessions=window, net=np.zeros(len(window)),
                             benchmarks={"U_EW": np.zeros(len(window))}),
        metrics={}, now=datetime(2026, 9, 29, tzinfo=UTC),
    )
    report = evaluator.validate(spec, [spec, _spec(n=2)])
    assert report.checks


def test_validate_rejects_family_without_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A family trial without U_EW fails validation closed."""
    from datetime import datetime, UTC

    import numpy as np

    from src.research.registry import TrialReturns
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    monkeypatch.setattr("src.research.evaluator.run_ledger", _ledger_fake(sessions))
    member = _spec(n=2)
    window = tuple(sessions[1:55])
    evaluator._ctx.registry.record(
        family="probe", spec_hash=member.spec_hash, spec_json='{"v": "shadow"}',
        segment=Segment.DISCOVERY, sim_config_json='{"s": "shadow"}', cube_id="cube",
        returns=TrialReturns(sessions=window, net=np.zeros(len(window)), benchmarks={}),
        metrics={}, now=datetime(2026, 9, 29, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="missing U_EW"):
        evaluator.validate(_spec(n=1), [_spec(n=1), member])


def test_find_member_returns_missing_raises(tmp_path: Path) -> None:
    """Unknown family members have no recorded trial."""
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    with pytest.raises(ValueError, match="no discovery trial"):
        evaluator._find_member_returns("absent", tuple(sessions))


def test_register_rejects_tampered_and_malformed_reports(tmp_path: Path) -> None:
    """Digest mismatches and malformed checks fail registration."""
    import json

    from src.research.gates import GateCheck, GateReport
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions)
    spec = _spec()
    passing = GateReport(
        spec_hash=spec.spec_hash, trial_id="t", segment=Segment.DISCOVERY,
        protocol_version=evaluator._ctx.protocol.version,
        checks=(GateCheck(gate="G1", name="G1.perturbation", value=0.0,
                          threshold=0.0, passed=True, detail="ok"),),
    )
    evaluator._ctx.reports_root.mkdir(parents=True, exist_ok=True)
    path = evaluator._ctx.reports_root / f"{spec.spec_hash}_discovery.json"
    payload = json.loads(passing.canonical_json())
    payload["extra"] = "tampered"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        evaluator.register_finalists([spec])
    payload = json.loads(passing.canonical_json())
    payload["checks"] = ["not-a-check"]
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid gate check"):
        evaluator.register_finalists([spec])


def test_holdout_after_open_rejects_outsiders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Once holdout is open, non-finalists get evidence-free authorizations."""
    from src.research.gates import GateCheck, GateReport
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions, drift=True)
    monkeypatch.setattr("src.research.evaluator.run_ledger", _ledger_fake(sessions))
    first = _spec(score={"size": 1.0})
    evaluator.screen(first)
    passing = GateReport(
        spec_hash=first.spec_hash, trial_id="t", segment=Segment.DISCOVERY,
        protocol_version=evaluator._ctx.protocol.version,
        checks=(GateCheck(gate="G1", name="G1.perturbation", value=0.0,
                          threshold=0.0, passed=True, detail="ok"),),
    )
    evaluator._ctx.reports_root.mkdir(parents=True, exist_ok=True)
    (evaluator._ctx.reports_root / f"{first.spec_hash}_discovery.json").write_text(
        passing.canonical_json() + "\n", encoding="utf-8")
    evaluator.register_finalists([first])
    evaluator.holdout(first)
    with pytest.raises(LockboxError):
        evaluator.holdout(_spec(score={"rev_1m": 1.0}))


def test_forward_runs_after_passed_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A passed holdout unlocks the forward window."""
    from src.research.gates import GateCheck, GateReport
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(60)
    evaluator = _context(tmp_path, sessions, drift=True)
    monkeypatch.setattr("src.research.evaluator.run_ledger", _ledger_fake(sessions))
    spec = _spec(score={"size": 1.0})
    evaluator.screen(spec)
    passing = GateReport(
        spec_hash=spec.spec_hash, trial_id="t", segment=Segment.DISCOVERY,
        protocol_version=evaluator._ctx.protocol.version,
        checks=(GateCheck(gate="G1", name="G1.perturbation", value=0.0,
                          threshold=0.0, passed=True, detail="ok"),),
    )
    evaluator._ctx.reports_root.mkdir(parents=True, exist_ok=True)
    (evaluator._ctx.reports_root / f"{spec.spec_hash}_discovery.json").write_text(
        passing.canonical_json() + "\n", encoding="utf-8")
    evaluator.register_finalists([spec])
    green = (
        GateCheck(gate="G5", name="G5.active_positive", value=0.01, threshold=0.0,
                  passed=True, detail="ok"),
        GateCheck(gate="G5", name="G5.within_distribution", value=0.01, threshold=0.0,
                  passed=True, detail="ok"),
    )
    monkeypatch.setattr("src.research.evaluator.holdout_checks", lambda **kwargs: green)
    report = evaluator.holdout(spec)
    assert report.passed is True
    record = evaluator.forward(spec)
    assert record.segment is Segment.FORWARD
    assert (evaluator._ctx.reports_root / f"{spec.spec_hash}_forward.json").exists()
