"""Termination criteria C1-C4 over precomputed evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.data.research_protocol import ResearchProtocol, Segment
from src.research.stats import (
    BootstrapProfile,
    annualized_log_growth,
    bootstrap_profile,
    deflated_sharpe_ratio_from_dispersion,
    point_metrics,
)

__all__ = [
    "CriteriaReport",
    "CriterionCheck",
    "DiscoveryEvidence",
    "evaluate_discovery",
    "evaluate_holdout",
]

_HALTED_POLICIES: Final = ("zero", "last_close")
_NAN_PROFILE: Final = BootstrapProfile(
    p_cagr_ge_abs_mdd=math.nan,
    p_cagr_le_zero=math.nan,
    cagr_p5=math.nan,
    mdd_p5=math.nan,
    p_mdd_below_limit=math.nan,
    underwater_median_sessions=math.nan,
    underwater_p95_sessions=math.nan,
)
_EMPTY: Final = np.zeros(0, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CriterionCheck:
    """One named threshold check; only blocking checks decide the report."""

    criterion: str
    name: str
    value: float
    threshold: float
    passed: bool
    blocking: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CriteriaReport:
    """Threshold checks for one strategy on one segment."""

    spec_hash: str
    trial_id: str
    segment: Segment
    protocol_version: str
    checks: tuple[CriterionCheck, ...]

    @property
    def passed(self) -> bool:
        """True when every blocking check passes."""
        return bool(all(check.passed for check in self.checks if check.blocking))

    def canonical_json(self) -> str:
        """Key-sorted compact JSON; byte-identical for identical evidence."""
        payload = {
            "checks": [
                {
                    "blocking": bool(check.blocking),
                    "criterion": check.criterion,
                    "detail": check.detail,
                    "name": check.name,
                    "passed": bool(check.passed),
                    "threshold": float(check.threshold),
                    "value": float(check.value),
                }
                for check in self.checks
            ],
            "protocol_version": self.protocol_version,
            "segment": self.segment.value,
            "spec_hash": self.spec_hash,
            "trial_id": self.trial_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        """SHA-256 hex of the canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DiscoveryEvidence:
    """Precomputed per-phase streams feeding the discovery criteria."""

    sessions: tuple[date, ...]
    base: tuple[NDArray[np.float64], ...]
    stress_slippage: tuple[NDArray[np.float64], ...]
    stress_delay: tuple[NDArray[np.float64], ...]
    fast_growth: Mapping[tuple[int, str], float]
    ledger_returns: Mapping[tuple[int, str], NDArray[np.float64]]
    perturbation_mismatches: int
    price_cap_growth_delta: float
    effective_registry_trials: float


def _check(
    criterion: str, name: str, value: float, threshold: float, passed: bool, *, blocking: bool
) -> CriterionCheck:
    achievements = float(value)
    bar = float(threshold)
    return CriterionCheck(
        criterion=criterion,
        name=name,
        value=achievements,
        threshold=bar,
        passed=bool(passed),
        blocking=bool(blocking),
        detail=f"{name} value={achievements:.6f} threshold={bar:.6f}",
    )


def _lower_median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered or any(math.isnan(v) for v in ordered):
        return math.nan
    return float(ordered[(len(ordered) - 1) // 2])


def _minimum(values: list[float]) -> float:
    if not values or any(math.isnan(v) for v in values):
        return math.nan
    return float(min(values))


def _safe_growth(stream: NDArray[np.float64], sessions_per_year: int) -> float:
    try:
        return float(
            annualized_log_growth(np.asarray(stream, dtype=np.float64), sessions_per_year=sessions_per_year)
        )
    except ValueError:
        return math.nan


def _safe_calmar(stream: NDArray[np.float64], sessions_per_year: int) -> float:
    try:
        return float(
            point_metrics(np.asarray(stream, dtype=np.float64), sessions_per_year=sessions_per_year).calmar
        )
    except ValueError:
        return math.nan


def _safe_profile(
    stream: NDArray[np.float64],
    *,
    block: int,
    draws: int,
    seed: int,
    horizon: int,
    mdd_limit: float,
    sessions_per_year: int,
) -> BootstrapProfile:
    try:
        return bootstrap_profile(
            np.asarray(stream, dtype=np.float64),
            block=block,
            draws=draws,
            seed=seed,
            horizon=horizon,
            mdd_limit=mdd_limit,
            sessions_per_year=sessions_per_year,
        )
    except ValueError:
        return _NAN_PROFILE


def _phase_growths(streams: tuple[NDArray[np.float64], ...], sessions_per_year: int) -> list[float]:
    return [_safe_growth(stream, sessions_per_year) for stream in streams]


def _phase_calmars(streams: tuple[NDArray[np.float64], ...], sessions_per_year: int) -> list[float]:
    return [_safe_calmar(stream, sessions_per_year) for stream in streams]


def _phase_profiles(
    streams: tuple[NDArray[np.float64], ...],
    *,
    block: int,
    draws: int,
    seed: int,
    horizon: int,
    mdd_limit: float,
    sessions_per_year: int,
) -> list[BootstrapProfile]:
    return [
        _safe_profile(
            stream,
            block=block,
            draws=draws,
            seed=seed,
            horizon=horizon,
            mdd_limit=mdd_limit,
            sessions_per_year=sessions_per_year,
        )
        for stream in streams
    ]


def _phase_sharpe(stream: NDArray[np.float64], sessions_per_year: int) -> float:
    values = np.asarray(stream, dtype=np.float64)
    std = float(np.std(values, ddof=1)) if values.ndim == 1 and values.size >= 2 else math.nan
    if not math.isfinite(std) or std <= 1e-12:
        return -math.inf
    return float(np.mean(values) / std * math.sqrt(sessions_per_year))


def _safe_dsr(candidate: NDArray[np.float64], *, dispersion: float, n_trials: float) -> float:
    try:
        return float(
            deflated_sharpe_ratio_from_dispersion(
                np.asarray(candidate, dtype=np.float64),
                sharpe_std_per_period=dispersion,
                n_trials=n_trials,
            )
        )
    except ValueError:
        return math.nan


def _median_sharpe_candidate(streams: tuple[NDArray[np.float64], ...], sessions_per_year: int) -> NDArray[np.float64]:
    sharpes = [_phase_sharpe(stream, sessions_per_year) for stream in streams]
    keys = [s if math.isfinite(s) else -math.inf for s in sharpes]
    ranked = sorted(range(len(keys)), key=lambda i: (keys[i], i))
    return streams[ranked[(len(ranked) - 1) // 2]] if ranked else _EMPTY


def evaluate_discovery(
    evidence: DiscoveryEvidence, protocol: ResearchProtocol, *, spec_hash: str, trial_id: str
) -> CriteriaReport:
    """Evaluate C1-C3 over per-phase discovery evidence with medians across phases."""
    criteria = protocol.criteria
    spy = protocol.sessions_per_year
    boot = criteria.bootstrap
    scenarios = {"base": evidence.base, "slippage": evidence.stress_slippage, "delay": evidence.stress_delay}
    growths = {name: _phase_growths(streams, spy) for name, streams in scenarios.items()}
    calmars = {name: _phase_calmars(streams, spy) for name, streams in scenarios.items()}
    profiles = {
        name: _phase_profiles(
            streams,
            block=boot.block_sessions,
            draws=boot.draws,
            seed=boot.seed,
            horizon=boot.horizon_sessions,
            mdd_limit=boot.mdd_limit,
            sessions_per_year=spy,
        )
        for name, streams in scenarios.items()
    }
    checks: list[CriterionCheck] = []
    mismatches = int(evidence.perturbation_mismatches)
    checks.append(_check("C1", "C1.perturbation", float(mismatches), 0.0, mismatches == 0, blocking=True))
    for name in ("base", "slippage", "delay"):
        weakest = _minimum(growths[name])
        checks.append(_check("C1", f"C1.growth_positive.{name}", weakest, 0.0, weakest > 0.0, blocking=True))
        loss = _lower_median([p.p_cagr_le_zero for p in profiles[name]])
        limit = float(criteria.c1.max_p_cagr_le_zero)
        checks.append(
            _check("C1", f"C1.bootstrap_loss.{name}", loss, limit, math.isfinite(loss) and loss <= limit, blocking=True)
        )
    candidate = _median_sharpe_candidate(evidence.base, spy)
    n_trials = float(criteria.prior_effective_trials) + float(evidence.effective_registry_trials)
    dispersion = float(criteria.prior_trial_sharpe_std_annual) / math.sqrt(float(spy))
    dsr = _safe_dsr(candidate, dispersion=dispersion, n_trials=n_trials)
    checks.append(_check("C1", "C1.dsr_reported", dsr, 0.0, math.isfinite(dsr) and dsr > 0.0, blocking=False))
    base_calmars = calmars["base"]
    base_profiles = profiles["base"]
    point = _lower_median(base_calmars)
    checks.append(
        _check("C2", "C2.point_calmar", point, float(criteria.c2.min_point_calmar),
               not math.isnan(point) and point >= float(criteria.c2.min_point_calmar), blocking=True)
    )
    p_calmar = _lower_median([p.p_cagr_ge_abs_mdd for p in base_profiles])
    checks.append(
        _check("C2", "C2.p_calmar", p_calmar, float(criteria.c2.min_p_calmar),
               math.isfinite(p_calmar) and p_calmar >= float(criteria.c2.min_p_calmar), blocking=True)
    )
    tail = _lower_median([p.p_mdd_below_limit for p in base_profiles])
    checks.append(
        _check("C2", "C2.p_mdd_tail", tail, float(criteria.c2.max_p_mdd_below_limit),
               math.isfinite(tail) and tail <= float(criteria.c2.max_p_mdd_below_limit), blocking=True)
    )
    uw_med = _lower_median([p.underwater_median_sessions for p in base_profiles])
    checks.append(
        _check("C2", "C2.underwater_median", uw_med, float(criteria.c2.max_underwater_median_sessions),
               math.isfinite(uw_med) and uw_med <= float(criteria.c2.max_underwater_median_sessions), blocking=True)
    )
    uw_p95 = _lower_median([p.underwater_p95_sessions for p in base_profiles])
    checks.append(
        _check("C2", "C2.underwater_p95", uw_p95, float(criteria.c2.max_underwater_p95_sessions),
               math.isfinite(uw_p95) and uw_p95 <= float(criteria.c2.max_underwater_p95_sessions), blocking=True)
    )
    worst = _minimum(base_calmars)
    checks.append(
        _check("C2", "C2.worst_phase_calmar", worst, float(criteria.c2.min_worst_phase_calmar),
               not math.isnan(worst) and worst >= float(criteria.c2.min_worst_phase_calmar), blocking=True)
    )
    for name in ("slippage", "delay"):
        stressed = _lower_median(calmars[name])
        checks.append(
            _check("C3", f"C3.stress_calmar.{name}", stressed, float(criteria.c3.stress_min_point_calmar),
                   not math.isnan(stressed) and stressed >= float(criteria.c3.stress_min_point_calmar), blocking=True)
        )
    for capital in criteria.c3.ledger_capitals:
        cap = int(capital)
        for policy_name in _HALTED_POLICIES:
            ledger_calmar = _safe_calmar(evidence.ledger_returns.get((cap, policy_name), _EMPTY), spy)
            floor = float(criteria.c3.ledger_min_calmar)
            checks.append(
                _check("C3", f"C3.ledger_calmar.{cap}.{policy_name}", ledger_calmar, floor,
                       not math.isnan(ledger_calmar) and ledger_calmar >= floor, blocking=True)
            )
        fast = float(evidence.fast_growth.get((cap, "zero"), math.nan))
        ledger_growth = _safe_growth(evidence.ledger_returns.get((cap, "zero"), _EMPTY), spy)
        gap = abs(fast - ledger_growth)
        ceiling = float(criteria.c3.parity_max_growth_gap)
        checks.append(
            _check("C3", f"C3.parity.{cap}", gap, ceiling, math.isfinite(gap) and gap <= ceiling, blocking=True)
        )
    delta = float(evidence.price_cap_growth_delta)
    checks.append(_check("C3", "C3.price_cap_effect", delta, 0.0, math.isfinite(delta), blocking=False))
    return CriteriaReport(
        spec_hash=spec_hash,
        trial_id=trial_id,
        segment=Segment.DISCOVERY,
        protocol_version=protocol.version,
        checks=tuple(checks),
    )


def evaluate_holdout(
    holdout: NDArray[np.float64], protocol: ResearchProtocol, *, spec_hash: str, trial_id: str
) -> CriteriaReport:
    """Evaluate C4 over one sealed holdout stream with a bootstrap of its own length."""
    criteria = protocol.criteria
    spy = protocol.sessions_per_year
    boot = criteria.bootstrap
    series = np.asarray(holdout, dtype=np.float64)
    growth = _safe_growth(series, spy)
    calmar = _safe_calmar(series, spy)
    block = min(int(boot.block_sessions), int(series.size))
    profile = _safe_profile(
        series,
        block=block,
        draws=boot.draws,
        seed=boot.seed,
        horizon=int(series.size),
        mdd_limit=boot.mdd_limit,
        sessions_per_year=spy,
    )
    loss = float(profile.p_cagr_le_zero)
    ceiling = float(criteria.c4.holdout_max_p_mean_le_zero)
    floor = float(criteria.c4.holdout_min_point_calmar)
    checks = (
        _check("C4", "C4.growth_positive", growth, 0.0, growth > 0.0, blocking=True),
        _check("C4", "C4.p_mean_le_zero", loss, ceiling, math.isfinite(loss) and loss <= ceiling, blocking=True),
        _check("C4", "C4.point_calmar", calmar, floor, not math.isnan(calmar) and calmar >= floor, blocking=True),
    )
    return CriteriaReport(
        spec_hash=spec_hash,
        trial_id=trial_id,
        segment=Segment.HOLDOUT,
        protocol_version=protocol.version,
        checks=tuple(checks),
    )
