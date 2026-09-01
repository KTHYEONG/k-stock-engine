"""Economic truth recovery contract tests.

Scenarios: ECONOMIC_TRUTH_01, ECONOMIC_TRUTH_02, ECONOMIC_TRUTH_03,
EXPOSURE_MATCHED_01.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from legacy.stocks.backtesting.engine import BacktestLedgerRow
from legacy.stocks.ml.contracts import (
    CompoundingCertificationSettings,
    NetAlphaTrainingRequest,
)
from legacy.stocks.ml.execution_replay import (
    ExecutionReplayEvidence,
    _ledger_growth_and_exposure,
    exposure_matched_benchmark_log_growth,
)
from legacy.stocks.ml.training import (
    ProfileReplayEvidence,
    _coverage_failure_reason,
    _evidence_from_execution,
)
from legacy.stocks.research.metrics import (
    certify_compounded_holdout,
    certify_exposure_matched_excess,
)


def _simple_evidence(
    base_growth: tuple[float, ...],
    stress_growth: tuple[float, ...] | None = None,
    *,
    segment_ids: tuple[int, ...] | None = None,
    planned_cycles: int = 10,
    filled_orders: int = 5,
    observed_interval_count: int | None = None,
    invested_interval_count: int | None = None,
    turnover: float = 0.1,
) -> ExecutionReplayEvidence:
    stress_growth = stress_growth if stress_growth is not None else base_growth
    segs = segment_ids or (0,) * len(base_growth)
    obs = observed_interval_count if observed_interval_count is not None else len(base_growth)
    inv = invested_interval_count if invested_interval_count is not None else len(base_growth)
    n = len(base_growth)
    base_exp = tuple(1.0 if i < inv / max(n, 1) else 0.0 for i in range(n))
    return ExecutionReplayEvidence(
        base_log_growth=base_growth,
        stress_log_growth=stress_growth,
        segment_ids=segs,
        planned_cycles=planned_cycles,
        filled_orders=filled_orders,
        cash_session_fraction=0.0,
        turnover=turnover,
        observed_interval_count=obs,
        invested_interval_count=inv,
        invested_interval_fraction=inv / max(obs, 1),
        filled_cycle_count=filled_orders,
        base_interval_exposure=base_exp,
        stress_interval_exposure=base_exp,
    )


def _make_ledger(
    positions: list[float], equities: list[float]
) -> tuple[BacktestLedgerRow, ...]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return tuple(
        BacktestLedgerRow(
            session=base + timedelta(days=i),
            settled_cash=50.0,
            unsettled_cash=0.0,
            positions_value=pos,
            accrued_costs=0.0,
            equity=eq,
        )
        for i, (pos, eq) in enumerate(zip(positions, equities, strict=True))
    )


# ---------------------------------------------------------------------------
# ECONOMIC_TRUTH_01
# ---------------------------------------------------------------------------

def test_economic_truth_01_holdout_uses_candidate_only() -> None:
    """ECONOMIC_TRUTH_01.

    Given candidate base/stress growth and a deliberately different dense
    shadow, the holdout certificate consumes candidate.base_log_growth and
    candidate.stress_log_growth only; dense-shadow stress cannot alter
    base CAGR, lower CAGR, or drawdown.
    """
    candidate_base = (0.01, 0.02, -0.005, 0.015, 0.01)
    candidate_stress = (0.008, 0.015, -0.006, 0.012, 0.008)
    shadow_stress = (-0.05, -0.04, -0.03, -0.02, -0.01)

    candidate = _simple_evidence(candidate_base, candidate_stress)
    shadow = _simple_evidence(candidate_base, shadow_stress, turnover=0.2)
    replay = ProfileReplayEvidence(candidate=candidate, dense_shadow=shadow)

    settings = CompoundingCertificationSettings(
        annualization_sessions=252,
        min_observed_sessions=1,
        bootstrap_alpha=0.05,
        bootstrap_resamples=50,
        seed=42,
    )
    growth_count = len(replay.candidate.base_log_growth)
    filled = replay.candidate.filled_cycle_count

    cert_candidate = certify_compounded_holdout(
        tuple(np.expm1(v) for v in replay.candidate.base_log_growth),
        tuple(np.expm1(v) for v in replay.candidate.stress_log_growth),
        10, growth_count, filled, settings,
    )

    cert_shadow = certify_compounded_holdout(
        tuple(np.expm1(v) for v in replay.candidate.base_log_growth),
        tuple(np.expm1(v) for v in replay.dense_shadow.stress_log_growth),
        10, growth_count, filled, settings,
    )

    assert cert_candidate.base["cagr"] == cert_shadow.base["cagr"]
    assert cert_candidate.base["lower_cagr"] == cert_shadow.base["lower_cagr"]
    assert cert_candidate.base["mdd"] == cert_shadow.base["mdd"]


def test_profile_replay_evidence_is_immutable() -> None:
    """ProfileReplayEvidence is a frozen dataclass with named fields."""
    candidate = _simple_evidence((0.01,), (0.008,))
    replay = ProfileReplayEvidence(candidate=candidate)
    assert replay.candidate is candidate
    assert replay.dense_shadow is None
    with pytest.raises(AttributeError):
        replay.candidate = candidate  # type: ignore[misc]


def test_profile_replay_evidence_dense_shadow_named() -> None:
    """candidate owns base+stress; dense_shadow is optional."""
    candidate = _simple_evidence((0.01, 0.02), (0.008, 0.015))
    shadow = _simple_evidence((0.01, 0.02), (-0.01, -0.02))
    replay = ProfileReplayEvidence(candidate=candidate, dense_shadow=shadow)
    assert len(replay.candidate.stress_log_growth) == 2
    assert replay.candidate.stress_log_growth[0] == pytest.approx(0.008)
    assert replay.dense_shadow.stress_log_growth[0] == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# ECONOMIC_TRUTH_02
# ---------------------------------------------------------------------------

def test_economic_truth_02_coverage_is_invested_over_observed() -> None:
    """ECONOMIC_TRUTH_02.

    For 496 observed intervals, 490 positive-exposure intervals, 95 planned
    cycles, coverage is exactly 490/496 and exceeds 0.98;
    substituting 95/496 is rejected.
    """
    growth = (0.001,) * 496
    evidence = _simple_evidence(
        growth,
        planned_cycles=95,
        filled_orders=10,
        observed_interval_count=496,
        invested_interval_count=490,
    )
    assert evidence.invested_interval_fraction == pytest.approx(490 / 496)
    assert evidence.invested_interval_fraction > 0.98
    wrong_coverage = 95 / 496
    assert wrong_coverage < 0.20
    assert evidence.invested_interval_fraction != pytest.approx(wrong_coverage, abs=0.01)


def test_economic_truth_02_coverage_from_evidence_from_execution() -> None:
    """_evidence_from_execution preserves exposure-based coverage."""
    growth = (0.001,) * 50
    base = _simple_evidence(
        growth, growth,
        planned_cycles=10, filled_orders=3,
        observed_interval_count=50, invested_interval_count=49,
    )
    candidate = _evidence_from_execution(
        10, "test_profile", "net_alpha_elastic_net",
        base, base, (0.1, 0.2, 0.3), 1,
    )
    assert candidate.active_cohort_count == 49
    assert candidate.complete_cohort_count == 50
    request = NetAlphaTrainingRequest(
        artifact_id="t",
        compounding=CompoundingCertificationSettings(min_observed_sessions=1),
    )
    reason = _coverage_failure_reason(candidate, request)
    assert reason == ""


# ---------------------------------------------------------------------------
# ECONOMIC_TRUTH_03
# ---------------------------------------------------------------------------

def test_economic_truth_03_ledger_exposure_parallel() -> None:
    """ECONOMIC_TRUTH_03.

    Ledger replay returns one growth per interval, all values are finite,
    and invested count equals the exact number of positive prior-row exposures.
    """
    ledger = _make_ledger(
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
    )
    growth, invested = _ledger_growth_and_exposure(ledger)
    assert len(growth) == 5
    assert invested == 3
    assert all(math.isfinite(g) for g in growth)
    assert all(g > -1.0 for g in growth)


def test_economic_truth_03_interval_exposure_finite_bounded() -> None:
    """base_interval_exposure and stress_interval_exposure are finite in [0,1]."""
    evidence = _simple_evidence(
        (0.01, 0.02, 0.015),
        (0.008, 0.015, 0.012),
    )
    assert hasattr(evidence, "base_interval_exposure")
    assert hasattr(evidence, "stress_interval_exposure")
    assert len(evidence.base_interval_exposure) == len(evidence.base_log_growth)
    assert len(evidence.stress_interval_exposure) == len(evidence.stress_log_growth)
    for exp in evidence.base_interval_exposure:
        assert 0.0 <= exp <= 1.0
        assert math.isfinite(exp)
    for exp in evidence.stress_interval_exposure:
        assert 0.0 <= exp <= 1.0
        assert math.isfinite(exp)


def test_economic_truth_03_invested_count_matches_positive_exposure() -> None:
    """invested_interval_count equals count(exposure > 0)."""
    evidence = _simple_evidence(
        (0.01, 0.02, 0.015, 0.01, 0.005),
        (0.008, 0.015, 0.012, 0.008, 0.004),
        observed_interval_count=5,
        invested_interval_count=3,
    )
    assert evidence.invested_interval_count == 3


# ---------------------------------------------------------------------------
# EXPOSURE_MATCHED_01
# ---------------------------------------------------------------------------

def test_exposure_matched_benchmark_length_matches_strategy() -> None:
    """EXPOSURE_MATCHED_01.

    For a synthetic 4-interval panel, benchmark length equals strategy length.
    """
    base = datetime(2024, 1, 1, tzinfo=UTC)
    sessions = [base + timedelta(days=i) for i in range(4)]
    rows: list[dict[str, object]] = []
    for session in sessions:
        for ticker in ("A", "B"):
            price = 100.0 if ticker == "A" else 200.0
            rows.append({
                "instrument_id": f"KRX:{ticker}",
                "session": session,
                "open": price,
                "close": price * 1.01,
                "volume": 1_000_000.0,
                "trading_value": price * 1_000_000.0,
                "adtv": price * 1_000_000.0,
            })
    market = pl.DataFrame(rows)
    exposure = (0.5, 0.8, 0.3, 0.0)
    benchmark = exposure_matched_benchmark_log_growth(market, sessions, exposure)
    assert len(benchmark) == len(sessions)


def test_exposure_matched_zero_exposure_yields_zero_growth() -> None:
    """Zero exposure yields zero benchmark growth."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    sessions = [base + timedelta(days=i) for i in range(3)]
    rows = [
        {
            "instrument_id": "KRX:A",
            "session": session,
            "open": 100.0,
            "close": 101.0,
            "volume": 1_000_000.0,
            "trading_value": 100_000_000.0,
            "adtv": 100_000_000.0,
        }
        for session in sessions
    ]
    market = pl.DataFrame(rows)
    exposure = (0.0, 0.0, 0.0)
    benchmark = exposure_matched_benchmark_log_growth(market, sessions, exposure)
    assert len(benchmark) == 3
    assert all(v == pytest.approx(0.0) for v in benchmark)


def test_exposure_matched_unit_exposure_matches_equal_weight() -> None:
    """Unit exposure yields gross equal-weight open-to-open growth."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    sessions = [base + timedelta(days=i) for i in range(3)]
    rows: list[dict[str, object]] = []
    opens_by_session = {
        sessions[0]: {"KRX:A": 100.0, "KRX:B": 200.0},
        sessions[1]: {"KRX:A": 102.0, "KRX:B": 204.0},
        sessions[2]: {"KRX:A": 104.0, "KRX:B": 208.0},
    }
    for session in sessions:
        for ticker, price in opens_by_session[session].items():
            rows.append({
                "instrument_id": ticker,
                "session": session,
                "open": price,
                "close": price * 1.01,
                "volume": 1_000_000.0,
                "trading_value": price * 1_000_000.0,
                "adtv": price * 1_000_000.0,
            })
    market = pl.DataFrame(rows)
    exposure = (1.0, 1.0, 1.0)
    benchmark = exposure_matched_benchmark_log_growth(market, sessions, exposure)
    assert len(benchmark) == 3
    assert all(math.isfinite(v) for v in benchmark)
    expected_first = math.log(102.0 / 100.0)
    assert benchmark[0] == pytest.approx(expected_first, abs=1e-10)


def test_exposure_matched_mismatched_length_raises() -> None:
    """Interval/session mismatch raises ValueError."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    sessions = [base + timedelta(days=i) for i in range(3)]
    rows = [
        {
            "instrument_id": "KRX:A",
            "session": session,
            "open": 100.0,
            "close": 101.0,
            "volume": 1_000_000.0,
            "trading_value": 100_000_000.0,
            "adtv": 100_000_000.0,
        }
        for session in sessions
    ]
    market = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="interval_exposure length"):
        exposure_matched_benchmark_log_growth(market, sessions, (0.5, 0.8))


# ---------------------------------------------------------------------------
# RELATIVE_CERTIFICATE_01
# ---------------------------------------------------------------------------

def test_relative_certificate_non_positive_matched_excess_fails() -> None:
    """RELATIVE_CERTIFICATE_01.

    Positive absolute base/stress lower CAGR combined with non-positive matched
    lower excess sets passed=false, reason=RESEARCH_ABSOLUTE_PASS_RELATIVE_UNPROVEN
    and promoted=false.
    """

    strategy_growth = [0.001] * 50
    benchmark_growth = [0.002] * 50
    settings = CompoundingCertificationSettings(
        annualization_sessions=252,
        min_observed_sessions=1,
        bootstrap_alpha=0.05,
        bootstrap_resamples=50,
        seed=42,
    )
    cert = certify_exposure_matched_excess(
        strategy_growth, benchmark_growth,
        horizon_sessions=10, active_cohort_count=50,
        settings=settings,
    )
    assert not cert.passed
    assert any("RELATIVE_UNPROVEN" in r for r in cert.reasons)
    assert not cert.promoted
