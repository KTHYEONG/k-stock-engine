"""Champion/challenger promotion rule and champion-store invariants."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.data.research_protocol import ResearchProtocol
from src.research.champion import (
    ChallengeDecision,
    ChampionRecord,
    ChampionStore,
    decision_from_canonical_json,
    decide_challenge,
    knob_changes,
)
from src.research.evaluation import CheckResult, EvaluationEvidence, EvaluationPolicy, ReportCard
from src.research.ledger_bridge import LedgerOutcome
from src.research.pipeline import EvaluationRun, StrategySpec

_NOW = datetime(2026, 9, 30, tzinfo=UTC)
_SPY = 252


def _protocol(
    *,
    require_neighbors: bool = True,
    paired_horizon: str = "evaluation",
    multiplicity: str = "none",
) -> ResearchProtocol:
    from src.data.research_protocol import load_research_protocol
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    base = load_research_protocol(Path("config/research/protocol.toml"), scope)
    evaluation = base.evaluation.model_copy(update={"draws": 200, "block_sessions": 5, "horizon_sessions": 30})
    champion = base.champion.model_copy(
        update={
            "require_neighbors": require_neighbors,
            "paired_horizon": paired_horizon,
            "multiplicity": multiplicity,
        }
    )
    return base.model_copy(update={"evaluation": evaluation, "champion": champion})


def _policy(protocol: ResearchProtocol) -> EvaluationPolicy:
    return EvaluationPolicy.model_validate(protocol.evaluation.model_dump())


def _sessions(n: int = 60) -> list[date]:
    from tests.fixtures.synthetic_panel import synthetic_sessions

    return synthetic_sessions(n, start=date(2019, 1, 1))


def _spec(**overrides: Any) -> StrategySpec:
    from src.research.book import BookSpec
    from src.research.hedge import HedgeSpec
    from src.research.model import ScorerConfig
    from src.research.policy import TrendCashPolicy

    policy_kw: dict[str, Any] = {"rebalance_every_sessions": 5, "min_units_per_slot": 0, "n": 20}
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
    return StrategySpec(
        policy=TrendCashPolicy(**policy_kw),
        scorer=ScorerConfig(**scorer_kw),
        book=BookSpec(**book_kw),
        hedge=HedgeSpec(**hedge_kw),
    )


def _outcome(sessions: list[date], logs: np.ndarray) -> LedgerOutcome:
    return LedgerOutcome(
        capital_krw=100_000_000,
        halted_exit_policy="zero",
        sessions=tuple(sessions),
        log_returns=np.ascontiguousarray(logs, dtype=np.float64),
    )


def _evidence(
    sessions: list[date],
    *,
    drift: float,
    slip_drift: float | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> EvaluationEvidence:
    rng = np.random.default_rng(seed)
    slip = drift if slip_drift is None else slip_drift

    def _stream(level: float) -> np.ndarray:
        values = np.full(len(sessions), float(level))
        if noise > 0.0:
            values = values + rng.standard_normal(len(sessions)) * float(noise)
        return np.ascontiguousarray(values, dtype=np.float64)

    return EvaluationEvidence(
        sessions=tuple(sessions),
        base=_outcome(sessions, _stream(drift)),
        stress_slippage=_outcome(sessions, _stream(slip)),
        stress_delay=_outcome(sessions, _stream(drift - 0.0001)),
        cost_grid={0.0: _outcome(sessions, _stream(drift)), 1.0: _outcome(sessions, _stream(slip))},
        unhedged=_outcome(sessions, _stream(drift)),
        placebo=_outcome(sessions, _stream(-0.0005)),
        index_log_returns=_stream(0.0),
        universe_ew_log_returns=_stream(0.0),
        fast_sim_growth=float(drift) * _SPY,
        perturbation_mismatches=0,
    )


def _card(
    spec_hash: str,
    run_id: str,
    sessions: list[date],
    objective_j: float,
    *,
    passed: bool = True,
    stream: str = "stress_slippage",
) -> ReportCard:
    return ReportCard(
        spec_hash=spec_hash,
        run_id=run_id,
        protocol_version="research-protocol-v4",
        start=sessions[0],
        end=sessions[-1],
        objective_j=float(objective_j),
        objective_stream=stream,
        integrity=(CheckResult(name="perturbation_mismatches", value=0.0, threshold=0.0, passed=True),),
        guards=(CheckResult(name="p_growth_le_zero", value=0.0, threshold=0.05, passed=passed),),
        metrics={"g": float(objective_j), "mdd": -0.1},
        yearly_growth={2019: float(objective_j)},
        regime_growth={"index_up_years": float(objective_j)},
        cost_grid_growth={0.0: float(objective_j), 1.0: float(objective_j) - 0.05},
        breakeven_ticks=2.0,
        controls={"placebo_g": 0.0},
        recent={"g": float(objective_j), "mdd": -0.1, "sessions": float(len(sessions))},
    )


def _run(
    spec: StrategySpec,
    sessions: list[date],
    *,
    drift: float,
    slip_drift: float | None = None,
    objective_j: float | None = None,
    passed: bool = True,
    noise: float = 0.0,
    seed: int = 0,
    stream: str = "stress_slippage",
    run_id: str | None = None,
) -> EvaluationRun:
    j = float(drift) * _SPY if objective_j is None else float(objective_j)
    return EvaluationRun(
        report=_card(
            spec.spec_hash,
            run_id if run_id is not None else f"run-{spec.spec_hash[:12]}",
            sessions,
            j,
            passed=passed,
            stream=stream,
        ),
        evidence=_evidence(sessions, drift=drift, slip_drift=slip_drift, noise=noise, seed=seed),
        spec_json=spec.canonical_json(),
    )


def _decide(
    *,
    challenger: EvaluationRun,
    champion: EvaluationRun,
    neighbors: tuple[EvaluationRun, ...] = (),
    challenger_spec: StrategySpec,
    champion_spec: StrategySpec,
    protocol: ResearchProtocol | None = None,
    prior_decisions: int = 0,
) -> ChallengeDecision:
    protocol = protocol if protocol is not None else _protocol()
    return decide_challenge(
        challenger=challenger,
        champion=champion,
        neighbors=neighbors,
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
        protocol=protocol,
        policy=_policy(protocol),
        prior_decisions=prior_decisions,
    )


def test_paired_improvement_promotes() -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    champion = _run(champion_spec, sessions, drift=0.0008)
    challenger = _run(challenger_spec, sessions, drift=0.0012)
    neighbors = (
        _run(_spec(policy={"n": 15}), sessions, drift=0.0010),
        _run(_spec(policy={"n": 30}), sessions, drift=0.0009),
    )
    decision = _decide(
        challenger=challenger,
        champion=champion,
        neighbors=neighbors,
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert decision.promotable is True
    assert decision.reasons == ()
    assert decision.knob_changes == ("policy.n",)
    assert decision.window == (sessions[0], sessions[-1])
    assert decision.run_ids == (challenger.report.run_id, champion.report.run_id)
    assert decision.paired.mean > 0.0
    assert decision.paired.lower > 0.0
    assert {spec_hash for spec_hash, _ in decision.neighbors} == {
        _spec(policy={"n": 15}).spec_hash,
        _spec(policy={"n": 30}).spec_hash,
    }


def test_indistinguishable_is_not_promoted() -> None:
    sessions = _sessions()
    champion_spec = _spec()
    challenger_spec = _spec(scorer={"num_leaves": 31})
    champion = _run(champion_spec, sessions, drift=0.0008)
    challenger = _run(challenger_spec, sessions, drift=0.0008, noise=0.004, seed=3)
    decision = _decide(
        challenger=challenger,
        champion=champion,
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert decision.promotable is False
    assert decision.reasons == ("paired_lower_bound",)
    assert decision.knob_changes == ("scorer.num_leaves",)


def test_lower_objective_blocks() -> None:
    sessions = _sessions()
    champion_spec = _spec()
    challenger_spec = _spec(scorer={"num_leaves": 31})
    champion = _run(champion_spec, sessions, drift=0.0008, objective_j=0.20)
    challenger = _run(challenger_spec, sessions, drift=0.0012, objective_j=0.05)
    decision = _decide(
        challenger=challenger,
        champion=champion,
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert decision.paired.lower > 0.0
    assert decision.reasons == ("objective_j",)


def test_failed_report_blocks() -> None:
    sessions = _sessions()
    champion_spec = _spec()
    challenger_spec = _spec(scorer={"num_leaves": 31})
    champion = _run(champion_spec, sessions, drift=0.0008)
    challenger = _run(challenger_spec, sessions, drift=0.0012, passed=False)
    decision = _decide(
        challenger=challenger,
        champion=champion,
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert decision.reasons == ("report_failed",)


def test_missing_neighbor_blocks_knob_change() -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert decision.reasons == ("neighbors_missing:policy.n",)


def test_wrong_knob_neighbor_does_not_cover() -> None:
    """A neighbor probing another knob leaves this knob uncovered: coverage is per knob, not per neighbor."""
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25}, hedge={"hedge_ratio": 0.5})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        neighbors=(_run(_spec(policy={"n": 30}, hedge={"hedge_ratio": 0.5}), sessions, drift=0.0010),),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert decision.knob_changes == ("hedge.hedge_ratio", "policy.n")
    assert decision.reasons == ("neighbors_missing:hedge.hedge_ratio",)


def test_knife_edge_and_tied_neighbors_block() -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    champion = _run(champion_spec, sessions, drift=0.0008)
    worse = _spec(policy={"n": 15})
    tie = _spec(policy={"n": 35})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=champion,
        neighbors=(_run(worse, sessions, drift=0.0005), _run(tie, sessions, drift=0.0008)),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert decision.reasons == (f"neighbor_not_better:{worse.spec_hash}", f"neighbor_not_better:{tie.spec_hash}")


def test_scorer_only_change_skips_the_neighbor_rule() -> None:
    sessions = _sessions()
    champion_spec = _spec()
    challenger_spec = _spec(scorer={"num_leaves": 31})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert decision.promotable is True
    assert decision.neighbors == ()


def test_neighbors_optional_when_protocol_does_not_require_them() -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
        protocol=_protocol(require_neighbors=False),
    )
    assert decision.promotable is True


def test_bool_and_string_leaves_are_not_knobs() -> None:
    base = _spec()
    assert knob_changes(base, base) == ()
    assert knob_changes(base, _spec(hedge={"use_futures": False})) == ()
    assert knob_changes(base, _spec(policy={"family": "other"})) == ()
    assert knob_changes(base, _spec(policy={"n": 21})) == ("policy.n",)
    assert knob_changes(base, _spec(book={"stock_capital_fraction": 0.5})) == ("book.stock_capital_fraction",)
    assert knob_changes(base, _spec(policy={"universe": {"min_adtv20_krw": 1_000, "min_price_krw": 1_000}})) == (
        "policy.universe.min_adtv20_krw",
    )


def test_different_windows_are_refused() -> None:
    sessions = _sessions()
    other = _sessions(50)
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    champion = _run(champion_spec, sessions, drift=0.0008)
    with pytest.raises(ValueError, match="identical sessions"):
        _decide(
            challenger=_run(challenger_spec, other, drift=0.0012),
            champion=champion,
            challenger_spec=challenger_spec,
            champion_spec=champion_spec,
        )
    with pytest.raises(ValueError, match="champion sessions"):
        _decide(
            challenger=_run(challenger_spec, sessions, drift=0.0012),
            champion=champion,
            neighbors=(_run(_spec(policy={"n": 15}), other, drift=0.001),),
            challenger_spec=challenger_spec,
            champion_spec=champion_spec,
        )


def test_mismatched_run_or_stream_is_refused() -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    champion = _run(champion_spec, sessions, drift=0.0008)
    with pytest.raises(ValueError, match="challenger_spec"):
        _decide(
            challenger=_run(challenger_spec, sessions, drift=0.0012),
            champion=champion,
            challenger_spec=_spec(policy={"n": 26}),
            champion_spec=champion_spec,
        )
    with pytest.raises(ValueError, match="champion_spec"):
        _decide(
            challenger=_run(challenger_spec, sessions, drift=0.0012),
            champion=champion,
            challenger_spec=challenger_spec,
            champion_spec=_spec(policy={"n": 21}),
        )
    with pytest.raises(ValueError, match="objective stream"):
        _decide(
            challenger=_run(challenger_spec, sessions, drift=0.0012, stream="base"),
            champion=champion,
            challenger_spec=challenger_spec,
            champion_spec=champion_spec,
        )


def test_objective_stream_of_the_challenger_decides_the_paired_test() -> None:
    """Better on stress_delay and worse on stress_slippage: only the challenger's own objective stream decides."""
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    neighbor_spec = _spec(policy={"n": 30})
    champion = _run(champion_spec, sessions, drift=0.0008)

    def _challenger(stream: str) -> Any:
        return _run(
            challenger_spec,
            sessions,
            drift=0.0020,
            slip_drift=0.0005,
            objective_j=0.50,
            stream=stream,
        )

    neighbor = _run(neighbor_spec, sessions, drift=0.0019, slip_drift=0.0009, objective_j=0.48)
    late = _decide(
        challenger=_challenger("stress_delay"),
        champion=champion,
        neighbors=(neighbor,),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert late.promotable is True

    early = _decide(
        challenger=_challenger("stress_slippage"),
        champion=champion,
        neighbors=(neighbor,),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert early.reasons == ("paired_lower_bound",)


def test_decision_canonical_json_round_trip() -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        neighbors=(_run(_spec(policy={"n": 30}), sessions, drift=0.0010),),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    payload = decision.canonical_json()
    assert decision.digest == decision_from_canonical_json(payload).digest
    assert json.loads(payload)["promotable"] is True
    assert json.loads(payload)["window"] == [sessions[0].isoformat(), sessions[-1].isoformat()]
    restored = decision_from_canonical_json(payload)
    assert restored == decision
    with pytest.raises(ValueError, match="invalid challenge decision"):
        decision_from_canonical_json("{}")


def test_non_finite_objectives_survive_canonical_json() -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012, objective_j=float("nan")),
        champion=_run(champion_spec, sessions, drift=0.0008, objective_j=float("inf")),
        neighbors=(_run(_spec(policy={"n": 30}), sessions, drift=float("nan")),),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert "objective_j" in decision.reasons
    assert f"neighbor_not_better:{_spec(policy={'n': 30}).spec_hash}" in decision.reasons
    assert np.isnan(decision.neighbors[0][1].mean)
    restored = decision_from_canonical_json(decision.canonical_json())
    assert np.isnan(restored.challenger_j)
    assert restored.champion_j == float("inf")
    assert np.isnan(restored.neighbors[0][1].mean)


def _store_with_champion(tmp_path: Path) -> tuple[ChampionStore, StrategySpec, EvaluationRun, ChampionRecord]:
    sessions = _sessions()
    store = ChampionStore(tmp_path / "champion")
    champion_spec = _spec(policy={"n": 20})
    run = _run(champion_spec, sessions, drift=0.0008)
    record = store.bootstrap(
        run=run,
        spec=champion_spec,
        spec_path=Path("config/research/strategies/base.toml"),
        now=_NOW,
    )
    assert store.current() == record
    assert store.history() == (record,)
    assert record.reason == "bootstrap"
    assert record.decision_digest is None
    assert record.objective_j == pytest.approx(0.0008 * _SPY)
    assert record.promoted_at == _NOW
    assert record.spec_json == champion_spec.canonical_json()
    assert record.report_digest == run.report.digest
    return store, champion_spec, run, record


def test_bootstrap_once(tmp_path: Path) -> None:
    store, champion_spec, run, _ = _store_with_champion(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        store.bootstrap(run=run, spec=champion_spec, spec_path=Path("base.toml"), now=_NOW)
    assert len(store.history()) == 1


def test_bootstrap_refuses_a_failed_report(tmp_path: Path) -> None:
    sessions = _sessions()
    store = ChampionStore(tmp_path / "champion")
    spec = _spec()
    with pytest.raises(ValueError, match="did not pass"):
        store.bootstrap(
            run=_run(spec, sessions, drift=0.001, passed=False), spec=spec, spec_path=Path("s.toml"), now=_NOW
        )
    assert store.current() is None
    with pytest.raises(ValueError, match="not produced by this spec"):
        store.bootstrap(
            run=_run(spec, sessions, drift=0.001), spec=_spec(policy={"n": 21}), spec_path=Path("s.toml"), now=_NOW
        )
    assert store.current() is None


def _promotable(tmp_path: Path) -> tuple[ChampionStore, StrategySpec, EvaluationRun, ChallengeDecision]:
    sessions = _sessions()
    store, champion_spec, champion_run, _ = _store_with_champion(tmp_path)
    challenger_spec = _spec(policy={"n": 25})
    challenger_run = _run(challenger_spec, sessions, drift=0.0012)
    decision = _decide(
        challenger=challenger_run,
        champion=champion_run,
        neighbors=(_run(_spec(policy={"n": 30}), sessions, drift=0.0010),),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    assert decision.promotable is True
    return store, challenger_spec, challenger_run, decision


def test_promote_requires_the_saved_decision(tmp_path: Path) -> None:
    store, challenger_spec, challenger_run, decision = _promotable(tmp_path)
    before = store.current()
    with pytest.raises(ValueError, match="never saved"):
        store.promote(
            decision=decision,
            run=challenger_run,
            spec=challenger_spec,
            spec_path=Path("challenger.toml"),
            now=_NOW,
        )
    assert store.current() == before
    assert len(store.history()) == 1

    store.save_decision(decision)
    assert store.decisions() == (decision,)
    assert store.decision_path(decision.digest).is_file()

    blocked = replace(decision, reasons=("paired_lower_bound",))
    store.save_decision(blocked)
    with pytest.raises(ValueError, match="not promotable"):
        store.promote(decision=blocked, run=challenger_run, spec=challenger_spec, spec_path=Path("c.toml"), now=_NOW)
    assert store.current() == before
    assert len(store.history()) == 1


def test_promote_refuses_a_stale_champion_and_mismatched_run(tmp_path: Path) -> None:
    sessions = _sessions()
    store, challenger_spec, challenger_run, decision = _promotable(tmp_path)
    store.save_decision(decision)
    store.promote(decision=decision, run=challenger_run, spec=challenger_spec, spec_path=Path("c.toml"), now=_NOW)
    assert len(store.history()) == 2

    with pytest.raises(ValueError, match="current champion"):
        store.promote(decision=decision, run=challenger_run, spec=challenger_spec, spec_path=Path("c.toml"), now=_NOW)
    assert len(store.history()) == 2

    store = ChampionStore(tmp_path / "fresh")
    fresh_spec = _spec(policy={"n": 20})
    fresh_run = _run(fresh_spec, sessions, drift=0.0008)
    store.bootstrap(run=fresh_run, spec=fresh_spec, spec_path=Path("base.toml"), now=_NOW)
    store.save_decision(decision)
    with pytest.raises(ValueError, match="challenger run"):
        store.promote(
            decision=decision,
            run=_run(challenger_spec, sessions, drift=0.0012, run_id="other-run"),
            spec=challenger_spec,
            spec_path=Path("c.toml"),
            now=_NOW,
        )
    with pytest.raises(ValueError, match="challenger spec"):
        store.promote(
            decision=decision,
            run=challenger_run,
            spec=_spec(policy={"n": 26}),
            spec_path=Path("c.toml"),
            now=_NOW,
        )
    shifted = replace(decision, window=(sessions[0], date(2019, 3, 15)), reasons=())
    store.save_decision(shifted)
    with pytest.raises(ValueError, match="different window"):
        store.promote(decision=shifted, run=challenger_run, spec=challenger_spec, spec_path=Path("c.toml"), now=_NOW)
    assert store.current() == store.history()[0]
    assert len(store.history()) == 1


def test_promote_replaces_current_and_appends_history(tmp_path: Path) -> None:
    store, challenger_spec, challenger_run, decision = _promotable(tmp_path)
    store.save_decision(decision)
    record = store.promote(
        decision=decision,
        run=challenger_run,
        spec=challenger_spec,
        spec_path=Path("config/research/strategies/challenger.toml"),
        now=datetime(2026, 10, 1, tzinfo=UTC),
    )
    assert store.current() == record
    assert store.history()[-1] == record
    assert len(store.history()) == 2
    assert record.reason == "challenge"
    assert record.decision_digest == decision.digest
    assert record.spec_path == "config/research/strategies/challenger.toml"
    assert record.spec_json == challenger_spec.canonical_json()
    orphan = ChampionStore(tmp_path / "empty")
    orphan.save_decision(decision)
    with pytest.raises(ValueError, match="no champion to promote over"):
        orphan.promote(decision=decision, run=challenger_run, spec=challenger_spec, spec_path=Path("c.toml"), now=_NOW)
    assert orphan.current() is None
    assert orphan.history() == ()


def test_current_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, challenger_spec, challenger_run, decision = _promotable(tmp_path)
    store.save_decision(decision)
    before = store.current()
    real_replace = os.replace

    def _fail_on_current(src: Any, dst: Any) -> None:
        if str(dst).endswith("current.json"):
            raise OSError("injected failure before replace")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _fail_on_current)
    with pytest.raises(OSError, match="injected failure"):
        store.promote(
            decision=decision,
            run=challenger_run,
            spec=challenger_spec,
            spec_path=Path("c.toml"),
            now=_NOW,
        )
    monkeypatch.undo()
    assert store.current() == before
    assert len(store.history()) == 1


def test_empty_sessions_are_refused_everywhere(tmp_path: Path) -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    champion = _run(champion_spec, sessions, drift=0.0008)
    challenger = _run(challenger_spec, sessions, drift=0.0012)
    empty_champion = replace(champion, evidence=replace(champion.evidence, sessions=()))
    with pytest.raises(ValueError, match="must be non-empty"):
        _decide(
            challenger=challenger,
            champion=empty_champion,
            challenger_spec=challenger_spec,
            champion_spec=champion_spec,
        )

    store = ChampionStore(tmp_path / "champion")
    store.bootstrap(run=champion, spec=champion_spec, spec_path=Path("b.toml"), now=_NOW)
    decision = _decide(
        challenger=challenger,
        champion=champion,
        neighbors=(_run(_spec(policy={"n": 30}), sessions, drift=0.0010),),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    store.save_decision(decision)
    with pytest.raises(ValueError, match="must be non-empty"):
        store.promote(
            decision=decision,
            run=replace(challenger, evidence=replace(challenger.evidence, sessions=())),
            spec=challenger_spec,
            spec_path=Path("c.toml"),
            now=_NOW,
        )
    assert len(store.history()) == 1


def test_neighbor_without_a_usable_spec_json_covers_no_knob() -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    neighbor = _run(_spec(policy={"n": 30}), sessions, drift=0.0010)
    for spec_json in (None, "{not json"):
        decision = _decide(
            challenger=_run(challenger_spec, sessions, drift=0.0012),
            champion=_run(champion_spec, sessions, drift=0.0008),
            neighbors=(replace(neighbor, spec_json=spec_json),),
            challenger_spec=challenger_spec,
            champion_spec=champion_spec,
        )
        assert decision.reasons == ("neighbors_missing:policy.n",)


def test_save_decision_is_idempotent(tmp_path: Path) -> None:
    store, _, _, decision = _promotable(tmp_path)
    first = store.save_decision(decision)
    second = store.save_decision(decision)
    assert first == second
    assert len(list((tmp_path / "champion" / "decisions").glob("*.json"))) == 1
    assert store.decisions() == (decision,)


def test_unreadable_history_and_decision_files_are_pit_errors(tmp_path: Path) -> None:
    from src.core.pit import PITDataError

    root = tmp_path / "champion"
    store, _, _, decision = _promotable(tmp_path)
    path = store.save_decision(decision)
    assert path.is_file()
    assert store.decisions() == (decision,)

    with (root / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert len(store.history()) == 1

    path.unlink()
    path.mkdir()
    with pytest.raises(PITDataError, match="unreadable champion decision"):
        store.decisions()

    (root / "history.jsonl").unlink()
    (root / "history.jsonl").mkdir()
    with pytest.raises(PITDataError, match="unreadable champion history"):
        store.history()
    (root / "history.jsonl").rmdir()
    (root / "current.json").unlink()
    (root / "current.json").mkdir()
    with pytest.raises(PITDataError, match="unreadable champion record"):
        store.current()


def test_spec_path_outside_the_repo_is_stored_verbatim(tmp_path: Path) -> None:
    sessions = _sessions()
    store = ChampionStore(tmp_path / "champion")
    spec = _spec()
    record = store.bootstrap(
        run=_run(spec, sessions, drift=0.0008),
        spec=spec,
        spec_path=Path("/nowhere/strategies/spec.toml"),
        now=_NOW,
    )
    assert record.spec_path == "/nowhere/strategies/spec.toml"
    assert store.current() == record


def test_store_reads_corrupt_state_as_a_pit_error(tmp_path: Path) -> None:
    from src.core.pit import PITDataError

    root = tmp_path / "champion"
    store = ChampionStore(root)
    assert store.current() is None
    assert store.history() == ()
    assert store.decisions() == ()
    root.mkdir(parents=True, exist_ok=True)
    (root / "current.json").write_text("{not json", encoding="utf-8")
    (root / "history.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(PITDataError):
        store.current()
    with pytest.raises(PITDataError):
        store.history()
    (root / "history.jsonl").write_text('{"spec_hash": "x"}\n', encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid champion record"):
        store.history()
    (root / "decisions").mkdir(exist_ok=True)
    (root / "decisions" / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid challenge decision"):
        store.decisions()


def _trend_spec(**overrides: Any) -> StrategySpec:
    from src.research.trend_overlay import TrendOverlaySpec

    trend_kw: dict[str, Any] = {
        "ma_sessions": 50,
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


def test_trend_knobs_need_neighbors() -> None:
    sessions = _sessions()
    champion_spec = _trend_spec()
    challenger_spec = _trend_spec(trend_overlay={"ma_sessions": 60})
    decision = decide_challenge(
        challenger=_run(challenger_spec, sessions, drift=0.001),
        champion=_run(champion_spec, sessions, drift=0.0008),
        neighbors=(),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
        protocol=_protocol(),
        policy=_policy(_protocol()),
    )
    assert decision.knob_changes == ("trend_overlay.ma_sessions",)
    assert decision.reasons == ("neighbors_missing:trend_overlay.ma_sessions",)


def test_full_horizon_tightens_the_interval() -> None:
    sessions = _sessions(2000)
    champion_spec = _spec()
    challenger_spec = _spec(scorer={"num_leaves": 31})
    champion = _run(champion_spec, sessions, drift=0.0008, noise=0.004, seed=3)
    challenger = _run(challenger_spec, sessions, drift=0.0012, noise=0.004, seed=4)
    base_protocol = _protocol()
    base_protocol = base_protocol.model_copy(update={"evaluation": base_protocol.evaluation.model_copy(
        update={"horizon_sessions": 1260, "block_sessions": 63, "draws": 400}
    )})
    base_policy = _policy(base_protocol)
    narrow = _decide(
        challenger=challenger,
        champion=champion,
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
        protocol=base_protocol.model_copy(
            update={"champion": base_protocol.champion.model_copy(update={"paired_horizon": "full"})}
        ),
    )
    wide = _decide(
        challenger=challenger,
        champion=champion,
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
        protocol=base_protocol,
    )
    assert narrow.paired_horizon_sessions == len(sessions)
    assert wide.paired_horizon_sessions == base_policy.horizon_sessions
    assert narrow.paired.lower > wide.paired.lower
    assert narrow.paired.mean == pytest.approx(wide.paired.mean, abs=0.005)


def test_bonferroni_uses_prior_decisions() -> None:
    sessions = _sessions()
    champion_spec = _spec()
    challenger_spec = _spec(scorer={"num_leaves": 31})
    protocol = _protocol(multiplicity="bonferroni_decisions")
    challenger = _run(challenger_spec, sessions, drift=0.0012, noise=0.004, seed=4)
    champion = _run(champion_spec, sessions, drift=0.0008)
    assert protocol.champion.alpha == pytest.approx(0.05)
    decision = _decide(
        challenger=challenger,
        champion=champion,
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
        protocol=protocol,
        prior_decisions=4,
    )
    assert decision.alpha_effective == pytest.approx(0.01)
    assert decision.prior_decisions == 4
    assert decision_from_canonical_json(decision.canonical_json()) == decision
    rng = np.random.default_rng(protocol.evaluation.seed)
    difference = challenger.evidence.stress_slippage.log_returns - champion.evidence.stress_slippage.log_returns
    deltas = [
        np.concatenate([difference[start:start + 5] for start in rng.integers(0, 56, size=6)]).mean() * _SPY
        for _ in range(200)
    ]
    assert decision.paired.lower == pytest.approx(float(np.quantile(deltas, 0.01)))
    unadjusted = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
        protocol=protocol,
        prior_decisions=0,
    )
    assert unadjusted.alpha_effective == pytest.approx(0.05)
    assert decision.paired.lower <= unadjusted.paired.lower
    with pytest.raises(ValueError, match="prior_decisions"):
        _decide(
            challenger=_run(challenger_spec, sessions, drift=0.0012),
            champion=_run(champion_spec, sessions, drift=0.0008),
            challenger_spec=challenger_spec,
            champion_spec=champion_spec,
            protocol=protocol,
            prior_decisions=-1,
        )


def test_legacy_decision_files_still_load() -> None:
    sessions = _sessions()
    champion_spec = _spec(policy={"n": 20})
    challenger_spec = _spec(policy={"n": 25})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        neighbors=(_run(_spec(policy={"n": 30}), sessions, drift=0.0010),),
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
    )
    payload = json.loads(decision.canonical_json())
    del payload["alpha_effective"]
    del payload["paired_horizon_sessions"]
    del payload["prior_decisions"]
    restored = decision_from_canonical_json(json.dumps(payload))
    assert restored.challenger_hash == decision.challenger_hash
    assert restored.champion_hash == decision.champion_hash
    assert restored.run_ids == decision.run_ids
    assert restored.paired.mean == pytest.approx(decision.paired.mean)
    assert restored.alpha_effective == pytest.approx(0.05)
    assert restored.paired_horizon_sessions == 1260
    assert restored.prior_decisions == 0
    customized = decision_from_canonical_json(json.dumps(payload), alpha=0.02, horizon_sessions=30)
    assert customized.alpha_effective == 0.02
    assert customized.paired_horizon_sessions == 30


def test_ruin_guard_non_blocking_under_v5() -> None:
    from src.research.evaluation import build_report_card

    sessions = _sessions()
    protocol, _ = _v5_protocol_and_scope()
    assert protocol.evaluation.max_p_ruin == pytest.approx(1.0)
    evidence = _evidence(sessions, drift=0.001)
    logs = np.concatenate([np.full(10, -0.1), np.full(50, 0.03)])
    evidence = replace(evidence, stress_slippage=_outcome(sessions, logs), stress_delay=_outcome(sessions, logs))
    policy = _policy(protocol).model_copy(update={"block_sessions": 60, "horizon_sessions": 60, "draws": 2})
    relaxed = build_report_card(
        evidence, policy, spec_hash=_spec().spec_hash, run_id="run-ruin", protocol_version=protocol.version,
    )
    assert relaxed.guards[0].value == 1.0
    assert relaxed.passed is True
    strict = build_report_card(
        evidence, policy.model_copy(update={"max_p_ruin": 0.05}),
        spec_hash=_spec().spec_hash, run_id="run-ruin", protocol_version=protocol.version,
    )
    assert strict.passed is False


def _v5_protocol_and_scope() -> tuple[ResearchProtocol, Any]:
    from src.data.research_scope import load_research_scope

    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    from src.data.research_protocol import load_research_protocol

    return load_research_protocol(Path("config/research/protocol.toml"), scope), scope


@pytest.mark.parametrize("missing", [
    None, "policy.trend_fail_weight_fraction", "book.rebalance_band",
    "trend_overlay.ma_sessions", "trend_overlay.long_fraction", "trend_overlay.short_fraction",
])
def test_growth_runbook_neighbors_cover_strategy_knobs(missing: str | None) -> None:
    from src.research.pipeline import load_strategy_spec

    root = Path("config/research/strategies")
    champion_spec = load_strategy_spec(root / "ml_sleeve_hedge.toml")
    challenger_spec = load_strategy_spec(root / "ml_growth_t85_b50_k200.toml")
    sessions = _sessions()
    neighbor_specs = [
        load_strategy_spec(path) for path in sorted(root.glob("ml_growth_*.toml"))
        if path.name != "ml_growth_t85_b50_k200.toml"
    ]
    neighbors = tuple(
        _run(spec, sessions, drift=0.0011) for spec in neighbor_specs
        if missing not in knob_changes(challenger_spec, spec)
    )
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008), neighbors=neighbors,
        challenger_spec=challenger_spec, champion_spec=champion_spec,
        protocol=_protocol(paired_horizon="full", multiplicity="bonferroni_decisions"),
    )
    assert "hedge.hedge_ratio" in decision.knob_changes
    assert "trend_overlay.initial_margin_rate" in decision.knob_changes
    assert decision.reasons == (() if missing is None else (f"neighbors_missing:{missing}",))


def test_trend_contract_terms_are_audited_without_neighbor_requirement() -> None:
    sessions = _sessions()
    champion_spec = _trend_spec()
    challenger_spec = _trend_spec(trend_overlay={"initial_margin_rate": 0.15})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        challenger_spec=challenger_spec, champion_spec=champion_spec,
    )
    assert decision.knob_changes == ("trend_overlay.initial_margin_rate",)
    assert decision.promotable


def test_trend_cadence_change_still_requires_neighbor() -> None:
    sessions = _sessions()
    champion_spec = _trend_spec()
    challenger_spec = _trend_spec(trend_overlay={"rebalance_every_sessions": 1})
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        challenger_spec=challenger_spec, champion_spec=champion_spec,
    )
    assert decision.reasons == ("neighbors_missing:trend_overlay.rebalance_every_sessions",)


def test_overlay_removal_does_not_require_impossible_neighbors() -> None:
    sessions = _sessions()
    champion_spec = _trend_spec()
    challenger_spec = _spec()
    decision = _decide(
        challenger=_run(challenger_spec, sessions, drift=0.0012),
        champion=_run(champion_spec, sessions, drift=0.0008),
        challenger_spec=challenger_spec, champion_spec=champion_spec,
    )
    assert decision.promotable
