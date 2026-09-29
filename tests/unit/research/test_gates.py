"""Promotion gate invariants over synthetic evidence."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from src.data.research_protocol import ResearchProtocol, Segment, load_research_protocol
from src.data.research_scope import load_research_scope
from src.research.gates import DiscoveryEvidence, GateReport, discovery_checks, holdout_checks


def _protocol() -> ResearchProtocol:
    scope = load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    return load_research_protocol(Path("config/research/protocol.toml"), scope)


def _sessions(n: int = 800, start: date = date(2020, 1, 1)) -> tuple[date, ...]:
    return tuple(start + timedelta(days=i) for i in range(n))


def _strong_evidence(protocol: ResearchProtocol, seed: int = 7) -> DiscoveryEvidence:
    rng = np.random.default_rng(seed)
    n = 800
    sessions = _sessions(n)
    active = rng.normal(loc=0.003, scale=0.005, size=n)
    active_cw = rng.normal(loc=0.002, scale=0.005, size=n)
    delayed = active * 0.9
    stressed = active * 0.8
    halted = active * 0.85
    spy = protocol.sessions_per_year
    base_g = float(np.mean(active) * spy)
    bench_g = base_g - 0.05
    family_cols = [rng.normal(loc=0.001, scale=0.005, size=n) for _ in range(15)]
    dominant = rng.normal(loc=0.004, scale=0.005, size=n)
    family = np.column_stack([dominant, *family_cols])
    trial_sharpes = rng.normal(loc=0.0, scale=0.1, size=60)
    capitals = [int(c) for c in protocol.gates.ledger_capitals]
    return DiscoveryEvidence(
        sessions=sessions,
        active_uew=np.ascontiguousarray(active),
        active_cw=np.ascontiguousarray(active_cw),
        delayed_active_uew=np.ascontiguousarray(delayed),
        stressed_active_uew=np.ascontiguousarray(stressed),
        halted_zero_active_uew=np.ascontiguousarray(halted),
        ledger_growth=dict.fromkeys(capitals, base_g),
        ledger_benchmark_growth=dict.fromkeys(capitals, bench_g),
        fast_growth=dict.fromkeys(capitals, base_g),
        perturbation_mismatches=0,
        trial_active_sharpes=np.ascontiguousarray(trial_sharpes),
        effective_trials=3.0,
        family_active=np.ascontiguousarray(family),
    )


def _check_map(checks: object) -> dict[str, object]:
    return {check.name: check for check in checks}  # type: ignore[attr-defined]


def test_strong_evidence_passes_all_gates() -> None:
    """Strong synthetic evidence passes every discovery check."""
    protocol = _protocol()
    evidence = _strong_evidence(protocol)
    checks = discovery_checks(evidence, protocol)
    assert checks
    assert all(check.passed for check in checks), [c.name for c in checks if not c.passed]
    names = [check.name for check in checks]
    assert names == sorted(names, key=lambda n: (n.split(".")[0], names.index(n)))


def test_perturbation_mismatch_fails_report() -> None:
    """One perturbed decision fails G1 and the whole report."""
    protocol = _protocol()
    evidence = _strong_evidence(protocol)
    mutated = DiscoveryEvidence(
        sessions=evidence.sessions, active_uew=evidence.active_uew, active_cw=evidence.active_cw,
        delayed_active_uew=evidence.delayed_active_uew, stressed_active_uew=evidence.stressed_active_uew,
        halted_zero_active_uew=evidence.halted_zero_active_uew, ledger_growth=evidence.ledger_growth,
        ledger_benchmark_growth=evidence.ledger_benchmark_growth, fast_growth=evidence.fast_growth,
        perturbation_mismatches=1, trial_active_sharpes=evidence.trial_active_sharpes,
        effective_trials=evidence.effective_trials, family_active=evidence.family_active,
    )
    checks = discovery_checks(mutated, protocol)
    by_name = _check_map(checks)
    assert by_name["G1.perturbation"].passed is False  # type: ignore[attr-defined]
    report = GateReport(spec_hash="a" * 64, trial_id="t", segment=Segment.DISCOVERY,
                        protocol_version=protocol.version, checks=checks)
    assert report.passed is False


def test_delay_retention_with_nonpositive_base_fails() -> None:
    """Non-positive base growth fails delay retention without arithmetic errors."""
    protocol = _protocol()
    evidence = _strong_evidence(protocol)
    flat = np.full(800, -0.001)
    mutated = DiscoveryEvidence(
        sessions=evidence.sessions, active_uew=np.ascontiguousarray(flat), active_cw=evidence.active_cw,
        delayed_active_uew=np.ascontiguousarray(flat), stressed_active_uew=evidence.stressed_active_uew,
        halted_zero_active_uew=evidence.halted_zero_active_uew, ledger_growth=evidence.ledger_growth,
        ledger_benchmark_growth=evidence.ledger_benchmark_growth, fast_growth=evidence.fast_growth,
        perturbation_mismatches=0, trial_active_sharpes=evidence.trial_active_sharpes,
        effective_trials=evidence.effective_trials, family_active=evidence.family_active,
    )
    by_name = _check_map(discovery_checks(mutated, protocol))
    assert by_name["G1.delay_retention"].passed is False  # type: ignore[attr-defined]


def test_optimistic_simulator_fails_parity() -> None:
    """A 0.01 optimistic gap fails parity at both small and large capitals."""
    protocol = _protocol()
    evidence = _strong_evidence(protocol)
    fast = {c: float(v) + 0.01 for c, v in evidence.fast_growth.items()}
    mutated = DiscoveryEvidence(
        sessions=evidence.sessions, active_uew=evidence.active_uew, active_cw=evidence.active_cw,
        delayed_active_uew=evidence.delayed_active_uew, stressed_active_uew=evidence.stressed_active_uew,
        halted_zero_active_uew=evidence.halted_zero_active_uew, ledger_growth=evidence.ledger_growth,
        ledger_benchmark_growth=evidence.ledger_benchmark_growth, fast_growth=fast,
        perturbation_mismatches=0, trial_active_sharpes=evidence.trial_active_sharpes,
        effective_trials=evidence.effective_trials, family_active=evidence.family_active,
    )
    by_name = _check_map(discovery_checks(mutated, protocol))
    assert by_name["G2.parity.100000000"].passed is False  # type: ignore[attr-defined]
    assert by_name["G2.parity.10000000"].passed is False  # type: ignore[attr-defined]


def test_small_family_fails_plateau() -> None:
    """Five family columns fail the plateau breadth check."""
    protocol = _protocol()
    evidence = _strong_evidence(protocol)
    small = np.ascontiguousarray(np.asarray(evidence.family_active)[:, :5])
    mutated = DiscoveryEvidence(
        sessions=evidence.sessions, active_uew=evidence.active_uew, active_cw=evidence.active_cw,
        delayed_active_uew=evidence.delayed_active_uew, stressed_active_uew=evidence.stressed_active_uew,
        halted_zero_active_uew=evidence.halted_zero_active_uew, ledger_growth=evidence.ledger_growth,
        ledger_benchmark_growth=evidence.ledger_benchmark_growth, fast_growth=evidence.fast_growth,
        perturbation_mismatches=0, trial_active_sharpes=evidence.trial_active_sharpes,
        effective_trials=evidence.effective_trials, family_active=small,
    )
    by_name = _check_map(discovery_checks(mutated, protocol))
    assert by_name["G3.plateau"].passed is False  # type: ignore[attr-defined]


def test_year_concentration_fails_stability() -> None:
    """All positive active in one year fails the single-year share gate."""
    protocol = _protocol()
    sessions = _sessions(240, date(2021, 1, 1))
    active = np.zeros(240)
    active[:120] = 0.01
    evidence = _strong_evidence(protocol)
    mutated = DiscoveryEvidence(
        sessions=sessions, active_uew=np.ascontiguousarray(active), active_cw=np.ascontiguousarray(active),
        delayed_active_uew=np.ascontiguousarray(active * 0.9),
        stressed_active_uew=np.ascontiguousarray(active * 0.8),
        halted_zero_active_uew=np.ascontiguousarray(active * 0.85),
        ledger_growth=evidence.ledger_growth, ledger_benchmark_growth=evidence.ledger_benchmark_growth,
        fast_growth=evidence.fast_growth, perturbation_mismatches=0,
        trial_active_sharpes=evidence.trial_active_sharpes, effective_trials=evidence.effective_trials,
        family_active=np.ascontiguousarray(np.tile(active.reshape(-1, 1), (1, 10))),
    )
    by_name = _check_map(discovery_checks(mutated, protocol))
    assert by_name["G4.single_year_share"].passed is False  # type: ignore[attr-defined]


def test_more_prior_trials_lower_dsr() -> None:
    """Identical evidence with more prior trials gives a lower DSR."""
    from src.research.stats import deflated_sharpe_ratio

    protocol = _protocol()
    evidence = _strong_evidence(protocol)
    low = deflated_sharpe_ratio(evidence.active_uew, trial_sharpes=evidence.trial_active_sharpes,
                                n_trials=float(evidence.effective_trials) + 10.0)
    high = deflated_sharpe_ratio(evidence.active_uew, trial_sharpes=evidence.trial_active_sharpes,
                                 n_trials=float(evidence.effective_trials) + 1000.0)
    assert high < low


def test_holdout_below_distribution_fails() -> None:
    """Holdout active at the 1st percentile fails the distribution gate."""
    protocol = _protocol()
    rng = np.random.default_rng(3)
    discovery = rng.normal(loc=0.003, scale=0.005, size=600)
    holdout = np.full(60, -0.05)
    checks = holdout_checks(holdout_active_uew=holdout, discovery_active_uew=discovery, protocol=protocol)
    by_name = _check_map(checks)
    assert by_name["G5.within_distribution"].passed is False  # type: ignore[attr-defined]
    assert by_name["G5.active_positive"].passed is False  # type: ignore[attr-defined]


def test_report_digest_stable() -> None:
    """The same checks always give the same digest."""
    protocol = _protocol()
    checks = discovery_checks(_strong_evidence(protocol), protocol)
    first = GateReport(spec_hash="ab", trial_id="t1", segment=Segment.DISCOVERY,
                       protocol_version=protocol.version, checks=checks)
    second = GateReport(spec_hash="ab", trial_id="t1", segment=Segment.DISCOVERY,
                        protocol_version=protocol.version, checks=checks)
    assert first.digest == second.digest
    assert first.canonical_json() == second.canonical_json()


def test_single_year_share_empty_base_fails() -> None:
    """All-zero active years fail concentration without division errors."""
    protocol = _protocol()
    evidence = _strong_evidence(protocol)
    flat = np.zeros(800)
    mutated = DiscoveryEvidence(
        sessions=evidence.sessions, active_uew=np.ascontiguousarray(flat), active_cw=evidence.active_cw,
        delayed_active_uew=evidence.delayed_active_uew, stressed_active_uew=evidence.stressed_active_uew,
        halted_zero_active_uew=evidence.halted_zero_active_uew, ledger_growth=evidence.ledger_growth,
        ledger_benchmark_growth=evidence.ledger_benchmark_growth, fast_growth=evidence.fast_growth,
        perturbation_mismatches=0, trial_active_sharpes=evidence.trial_active_sharpes,
        effective_trials=evidence.effective_trials, family_active=evidence.family_active,
    )
    by_name = _check_map(discovery_checks(mutated, protocol))
    assert by_name["G4.single_year_share"].passed is False  # type: ignore[attr-defined]


def test_holdout_passes_on_matching_distribution() -> None:
    """Holdout near the discovery mean passes both G5 checks."""
    protocol = _protocol()
    rng = np.random.default_rng(5)
    discovery = rng.normal(loc=0.003, scale=0.005, size=600)
    holdout = rng.normal(loc=0.003, scale=0.005, size=60)
    checks = holdout_checks(holdout_active_uew=holdout, discovery_active_uew=discovery, protocol=protocol)
    assert checks[0].passed is True


def test_one_dimensional_family_treated_as_single_member() -> None:
    """A 1-D family array counts as one member and fails the breadth gate."""
    protocol = _protocol()
    evidence = _strong_evidence(protocol)
    flat = np.ascontiguousarray(np.asarray(evidence.family_active)[:, 0])
    mutated = DiscoveryEvidence(
        sessions=evidence.sessions, active_uew=evidence.active_uew, active_cw=evidence.active_cw,
        delayed_active_uew=evidence.delayed_active_uew, stressed_active_uew=evidence.stressed_active_uew,
        halted_zero_active_uew=evidence.halted_zero_active_uew, ledger_growth=evidence.ledger_growth,
        ledger_benchmark_growth=evidence.ledger_benchmark_growth, fast_growth=evidence.fast_growth,
        perturbation_mismatches=0, trial_active_sharpes=evidence.trial_active_sharpes,
        effective_trials=evidence.effective_trials, family_active=flat,
    )
    by_name = _check_map(discovery_checks(mutated, protocol))
    assert by_name["G3.plateau"].passed is False  # type: ignore[attr-defined]


def test_degenerate_family_shapes_fail_breadth() -> None:
    """3-D and empty families fail plateau without raising."""
    protocol = _protocol()
    evidence = _strong_evidence(protocol)

    def _mutated(family: object) -> object:
        return DiscoveryEvidence(
            sessions=evidence.sessions, active_uew=evidence.active_uew, active_cw=evidence.active_cw,
            delayed_active_uew=evidence.delayed_active_uew, stressed_active_uew=evidence.stressed_active_uew,
            halted_zero_active_uew=evidence.halted_zero_active_uew, ledger_growth=evidence.ledger_growth,
            ledger_benchmark_growth=evidence.ledger_benchmark_growth, fast_growth=evidence.fast_growth,
            perturbation_mismatches=0, trial_active_sharpes=evidence.trial_active_sharpes,
            effective_trials=evidence.effective_trials, family_active=family,  # type: ignore[arg-type]
        )

    n = len(evidence.sessions)
    weird = np.ascontiguousarray(np.zeros((n, 2, 2)))
    by_name = _check_map(discovery_checks(_mutated(weird), protocol))  # type: ignore[arg-type]
    assert by_name["G3.plateau"].passed is False  # type: ignore[attr-defined]
    assert by_name["G3.pbo"].passed is False  # type: ignore[attr-defined]
    empty = np.ascontiguousarray(np.zeros((n, 0)))
    by_name = _check_map(discovery_checks(_mutated(empty), protocol))  # type: ignore[arg-type]
    assert by_name["G3.plateau"].passed is False  # type: ignore[attr-defined]
