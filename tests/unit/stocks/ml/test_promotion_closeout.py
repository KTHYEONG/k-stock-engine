"""Promotion closeout scenarios: tail censoring, blend shadow, Holm scope, perf."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.stocks.ml.contracts import (
    CompoundingCertificationSettings,
    NetAlphaTrainingRequest,
)
from src.stocks.research.calibration_schedule import (
    SessionClusterCalibrationSchedule,
    _session_cluster_bootstrap_means,
)
from src.stocks.research.metrics import certify_compounded_holdout


# ---------------------------------------------------------------------------
# G1: tail censoring
# ---------------------------------------------------------------------------

def test_tail_censoring_allows_structural_shortfall() -> None:
    """SCENARIO_TAIL_CENSORING_PASS."""
    settings = CompoundingCertificationSettings(
        min_observed_sessions=252,
        bootstrap_resamples=200,
        allowed_tail_censoring_sessions=10,
    )
    rng = np.random.default_rng(5)
    returns = (0.001 + 0.01 * rng.standard_normal(239)).tolist()
    evidence = certify_compounded_holdout(
        returns, returns, 10, 239, 236, settings
    )
    assert "insufficient-observed-sessions" not in evidence.reasons

    short = returns[:220]
    evidence_short = certify_compounded_holdout(
        short, short, 10, 220, 216, settings
    )
    assert "insufficient-observed-sessions" in evidence_short.reasons


def test_zero_allowance_preserves_current_rule() -> None:
    """SCENARIO_TAIL_CENSORING_FLAG_OFF."""
    settings = CompoundingCertificationSettings(
        min_observed_sessions=252,
        bootstrap_resamples=200,
    )
    assert settings.allowed_tail_censoring_sessions == 0
    rng = np.random.default_rng(6)
    returns = (0.001 + 0.01 * rng.standard_normal(239)).tolist()
    evidence = certify_compounded_holdout(
        returns, returns, 10, 239, 236, settings
    )
    assert "insufficient-observed-sessions" in evidence.reasons

    with pytest.raises(ValueError, match="must be"):
        CompoundingCertificationSettings(
            min_observed_sessions=252,
            allowed_tail_censoring_sessions=-1,
        )
    with pytest.raises(ValueError, match="must be"):
        CompoundingCertificationSettings(
            min_observed_sessions=252,
            allowed_tail_censoring_sessions=252,
        )


# ---------------------------------------------------------------------------
# G3: Holm family scope
# ---------------------------------------------------------------------------

def _evidence(profile_id: str, mean: float, n: int = 60, sigma: float = 0.01, seed: int | None = None):
    from src.stocks.ml.horizons import HorizonOOFEvidence

    rng = np.random.default_rng((abs(hash(profile_id)) % 10000) if seed is None else seed)
    base = tuple(mean + sigma * float(v) for v in rng.standard_normal(n))
    return HorizonOOFEvidence(
        horizon_sessions=10,
        profile_id=profile_id,
        model_family="economic_rawnet_lgbm",
        base_log_growth=base,
        stress_log_growth=tuple(v * 0.9 for v in base),
        cohort_segment_ids=tuple(0 for _ in range(n)),
        complete_cohort_count=n,
        active_cohort_count=n,
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=1,
        fold_rank_ics=(0.2,),
        rebalance_frequency_sessions=5,
        top_k=12,
    )


def test_route_gatekeeping_demotes_cell_holm() -> None:
    """SCENARIO_HOLM_ROUTE_GATEKEEPING_SCOPE."""
    from src.stocks.ml.horizons import select_horizons

    # Find a weak-cell mean whose discrete bootstrap p sits above the Holm
    # threshold while its lower bound stays positive - the exact regime where
    # route_gatekeeping diverges from the frontier scope.
    weak_mean = None
    probe = tuple(
        [_evidence("strong_cell", 0.02, sigma=0.002)]
        + [_evidence(f"weak_cell_{i}", m, sigma=0.002, seed=99) for i in range(11)]
        for m in (0.00002, 0.00005, 0.0001, 0.00015, 0.0002)
    )
    for probe_set in probe:
        sel = select_horizons(probe_set, 0.05, 42, 300)
        line = next(
            (r for r in sel.selection_reasons if "10:weak_cell_0" in r), ""
        )
        if "rejected" in line and "base=" in line:
            base_token = next(t for t in line.split() if t.startswith("base="))
            if float(base_token.split("=")[1]) > 0:
                weak_mean = float(line.split("base_p=")[1].split()[0]) * 0 +                     float(base_token.split("=")[1])
                chosen = probe_set
                break
    assert weak_mean is not None, "fixture could not locate a divergence point"

    candidates = chosen
    frontier = select_horizons(candidates, 0.05, 42, 300)
    gatekeeping = select_horizons(
        candidates, 0.05, 42, 300, family_scope="route_gatekeeping"
    )

    # published statistics are identical between scopes (I2)
    assert gatekeeping.adjusted_lower_growth == frontier.adjusted_lower_growth
    assert gatekeeping.base_p_values == frontier.base_p_values
    assert gatekeeping.base_holm_thresholds == frontier.base_holm_thresholds
    assert gatekeeping.stress_p_values == frontier.stress_p_values

    # frontier scope: weak positive-LB cells rejected by Holm p-threshold;
    # gatekeeping scope: the same cells admitted on lower bounds alone.
    for weak_index in range(11):
        weak = f"weak_cell_{weak_index}"
        frontier_line = next(
            (r for r in frontier.selection_reasons if f"10:{weak}" in r), ""
        )
        assert "rejected" in frontier_line
        gate_line = next(
            (r for r in gatekeeping.selection_reasons if f"10:{weak}" in r), ""
        )
        assert "admissible" in gate_line


def test_request_field_default_and_validation() -> None:
    request = NetAlphaTrainingRequest(artifact_id="v1")
    assert request.holm_family_scope == "frontier"
    with pytest.raises(ValueError, match="must be"):
        NetAlphaTrainingRequest(artifact_id="v1", holm_family_scope="bogus")


# ---------------------------------------------------------------------------
# G2: blend shadow deferral
# ---------------------------------------------------------------------------

def test_blend_cells_admit_without_dense_shadow() -> None:
    """SCENARIO_BLEND_FRONTIER_NO_SHADOW_REQUIRED."""
    from src.stocks.ml.execution_replay import (
        ExecutionReplayEvidence,
        ProfileReplayEvidence,
    )

    pair = ProfileReplayEvidence(
        candidate=ExecutionReplayEvidence(
            base_log_growth=tuple(0.01 for _ in range(30)),
            stress_log_growth=tuple(0.008 for _ in range(30)),
            segment_ids=tuple(0 for _ in range(30)),
            planned_cycles=3,
            filled_orders=90,
            cash_session_fraction=0.0,
            turnover=1.5,
            observed_interval_count=30,
            invested_interval_count=30,
            invested_interval_fraction=1.0,
            filled_cycle_count=3,
        ),
        dense_shadow=None,
    )
    assert pair.candidate.filled_orders > 0
    assert pair.dense_shadow is None
    unpacked_candidate, unpacked_shadow = pair
    assert unpacked_shadow is unpacked_candidate


# ---------------------------------------------------------------------------
# R2: calibration memoization
# ---------------------------------------------------------------------------

def test_bucket_lower_bound_memoized() -> None:
    """SCENARIO_CALIB_MEMO_HIT."""
    schedule = SessionClusterCalibrationSchedule.__new__(
        SessionClusterCalibrationSchedule
    )
    schedule._lb_memo = {}
    schedule._block_length = 3
    schedule._calibrator = SimpleNamespace(
        n_bootstrap=200, seed=7, bootstrap_alpha=0.05
    )
    sums = np.array([0.01, -0.005, 0.02, 0.015, -0.008, 0.012, 0.004, 0.006], dtype=float)
    counts = np.full(sums.size, 3.0)
    state = SimpleNamespace(
        session_sums=sums,
        session_counts=counts,
        csum_sums=np.concatenate([[0.0], np.cumsum(sums)]),
        csum_counts=np.concatenate([[0.0], np.cumsum(counts)]),
        row_sum=float(sums.sum()),
        row_count=float(counts.sum()),
    )

    draws = {"n": 0}
    original = _session_cluster_bootstrap_means

    def counting(*args, **kwargs):
        draws["n"] += 1
        return original(*args, **kwargs)

    import src.stocks.research.calibration_schedule as cs

    cs._session_cluster_bootstrap_means = counting
    try:
        first = schedule._bucket_lower_bound(0, state)
        second = schedule._bucket_lower_bound(0, state)
        assert first == second
        assert draws["n"] == 1  # second call hit the memo

        grown = SimpleNamespace(
            session_sums=np.append(sums, 0.009),
            session_counts=np.append(counts, 3.0),
            csum_sums=np.concatenate([[0.0], np.cumsum(np.append(sums, 0.009))]),
            csum_counts=np.concatenate([[0.0], np.cumsum(np.append(counts, 3.0))]),
            row_sum=float(sums.sum()) + 0.009,
            row_count=float(counts.sum()) + 3.0,
        )
        third = schedule._bucket_lower_bound(0, grown)
        assert draws["n"] == 2  # new revealed state recomputed
        assert isinstance(third, float)
    finally:
        cs._session_cluster_bootstrap_means = original


# ---------------------------------------------------------------------------
# P3: discovery workers
# ---------------------------------------------------------------------------

def test_parallel_horizons_merge_deterministically() -> None:
    """SCENARIO_DISCOVERY_WORKERS_DETERMINISTIC."""
    from src.stocks.ml.training import _run_ordered_with_workers

    def task(value: int) -> int:
        if value == 13:
            raise RuntimeError("boom")
        return value * 2

    ordered = _run_ordered_with_workers([3, 1, 2], task, workers=2)
    assert ordered == [6, 2, 4]  # item order preserved regardless of completion

    failed = _run_ordered_with_workers([13], task, workers=2)
    assert failed is None  # caller falls back to sequential with recorded reason
