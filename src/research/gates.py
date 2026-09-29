"""Promotion gates G1-G5 over precomputed evaluation evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.data.research_protocol import ResearchProtocol, Segment
from src.research.stats import (
    active_t_stat,
    annualized_log_growth,
    block_bootstrap_annualized_means,
    cscv_pbo,
    deflated_sharpe_ratio,
    information_ratio,
    yearly_active_log,
)

__all__ = [
    "DiscoveryEvidence",
    "GateCheck",
    "GateReport",
    "discovery_checks",
    "holdout_checks",
]

_MIN_FAMILY_COLUMNS: Final = 8
_MIN_YEAR_SESSIONS: Final = 60


@dataclass(frozen=True, slots=True)
class GateCheck:
    gate: str
    name: str
    value: float
    threshold: float
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class GateReport:
    spec_hash: str
    trial_id: str
    segment: Segment
    protocol_version: str
    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return bool(all(check.passed for check in self.checks))

    def canonical_json(self) -> str:
        payload = {
            "checks": [
                {
                    "detail": check.detail,
                    "gate": check.gate,
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
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DiscoveryEvidence:
    sessions: tuple[date, ...]
    active_uew: NDArray[np.float64]
    active_cw: NDArray[np.float64]
    delayed_active_uew: NDArray[np.float64]
    stressed_active_uew: NDArray[np.float64]
    halted_zero_active_uew: NDArray[np.float64]
    ledger_growth: Mapping[int, float]
    ledger_benchmark_growth: Mapping[int, float]
    fast_growth: Mapping[int, float]
    perturbation_mismatches: int
    trial_active_sharpes: NDArray[np.float64]
    effective_trials: float
    family_active: NDArray[np.float64]


def _growth(x: NDArray[np.float64], sessions_per_year: int) -> float:
    return float(annualized_log_growth(np.asarray(x, dtype=np.float64), sessions_per_year=sessions_per_year))


def _loss_probability(x: NDArray[np.float64], protocol: ResearchProtocol) -> float:
    stats = protocol.statistics
    draws = block_bootstrap_annualized_means(
        np.asarray(x, dtype=np.float64),
        block=stats.bootstrap_block_sessions,
        draws=stats.bootstrap_draws,
        seed=stats.bootstrap_seed,
        sessions_per_year=protocol.sessions_per_year,
    )
    return float(np.mean(draws <= 0.0))


def _plateau_share(family_active: NDArray[np.float64], sessions_per_year: int) -> tuple[float, int]:
    matrix = np.asarray(family_active, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        return (0.0, 0)
    n_cols = int(matrix.shape[1])
    if n_cols == 0:
        return (0.0, 0)
    positive = sum(1 for k in range(n_cols) if information_ratio(matrix[:, k], sessions_per_year=sessions_per_year) > 0.0)
    return (float(positive) / float(n_cols), n_cols)


def _positive_year_share(sessions: tuple[date, ...], active: NDArray[np.float64]) -> tuple[float, int]:
    values = np.asarray(active, dtype=np.float64)
    days = list(sessions)
    counts: dict[int, int] = {}
    for day in days:
        counts[day.year] = counts.get(day.year, 0) + 1
    totals = yearly_active_log(days, values)
    eligible = [year for year, count in counts.items() if count >= _MIN_YEAR_SESSIONS]
    if not eligible:
        return (0.0, 0)
    positive = sum(1 for year in eligible if totals.get(year, 0.0) > 0.0)
    return (float(positive) / float(len(eligible)), len(eligible))


def _single_year_share(sessions: tuple[date, ...], active: NDArray[np.float64]) -> tuple[float, bool]:
    totals = yearly_active_log(list(sessions), np.asarray(active, dtype=np.float64))
    positives = [max(value, 0.0) for value in totals.values()]
    denominator = float(sum(positives))
    if denominator == 0.0:
        return (1.0, False)
    return (float(max(positives)) / denominator, True)


def discovery_checks(evidence: DiscoveryEvidence, protocol: ResearchProtocol) -> tuple[GateCheck, ...]:
    """Evaluate G1-G4 from precomputed evidence."""
    spy = protocol.sessions_per_year
    gates = protocol.gates
    stats = protocol.statistics
    active = np.asarray(evidence.active_uew, dtype=np.float64)
    base_g = _growth(active, spy)
    delayed_g = _growth(np.asarray(evidence.delayed_active_uew, dtype=np.float64), spy)
    stressed = np.asarray(evidence.stressed_active_uew, dtype=np.float64)
    halted = np.asarray(evidence.halted_zero_active_uew, dtype=np.float64)
    checks: list[GateCheck] = []

    mismatches = int(evidence.perturbation_mismatches)
    checks.append(
        GateCheck(gate="G1", name="G1.perturbation", value=float(mismatches), threshold=0.0,
                  passed=mismatches == 0, detail=f"mismatches={mismatches}")
    )
    if base_g <= 0.0:
        checks.append(
            GateCheck(gate="G1", name="G1.delay_retention", value=0.0,
                      threshold=float(gates.min_delay_retention), passed=False,
                      detail=f"base_g={base_g:.6f}<=0")
        )
    else:
        ratio = float(delayed_g / base_g)
        checks.append(
            GateCheck(gate="G1", name="G1.delay_retention", value=ratio,
                      threshold=float(gates.min_delay_retention), passed=ratio >= float(gates.min_delay_retention),
                      detail=f"delayed_g={delayed_g:.6f} base_g={base_g:.6f}")
        )

    stress_g = _growth(stressed, spy)
    checks.append(
        GateCheck(gate="G2", name="G2.stress_active", value=stress_g, threshold=0.0,
                  passed=stress_g > 0.0, detail=f"stress_g={stress_g:.6f}")
    )
    stress_prob = _loss_probability(stressed, protocol)
    checks.append(
        GateCheck(gate="G2", name="G2.stress_bootstrap", value=stress_prob,
                  threshold=float(gates.max_bootstrap_loss_probability),
                  passed=stress_prob <= float(gates.max_bootstrap_loss_probability),
                  detail=f"draws={stats.bootstrap_draws} block={stats.bootstrap_block_sessions}")
    )
    halted_g = _growth(halted, spy)
    checks.append(
        GateCheck(gate="G2", name="G2.halted_zero", value=halted_g, threshold=0.0,
                  passed=halted_g > 0.0, detail=f"halted_g={halted_g:.6f}")
    )
    for capital in gates.ledger_capitals:
        cap = int(capital)
        ledger = float(evidence.ledger_growth[cap])
        bench = float(evidence.ledger_benchmark_growth[cap])
        fast = float(evidence.fast_growth[cap])
        active_gap = float(ledger - bench)
        checks.append(
            GateCheck(gate="G2", name=f"G2.ledger_active.{cap}", value=active_gap, threshold=0.0,
                      passed=active_gap > 0.0, detail=f"ledger={ledger:.6f} bench={bench:.6f}")
        )
        gap = float(gates.parity_max_gap)
        if cap <= int(gates.parity_max_capital):
            parity_value = float(abs(fast - ledger))
            checks.append(
                GateCheck(gate="G2", name=f"G2.parity.{cap}", value=parity_value, threshold=gap,
                          passed=parity_value <= gap, detail=f"fast={fast:.6f} ledger={ledger:.6f}")
            )
        else:
            parity_value = float(fast - ledger)
            checks.append(
                GateCheck(gate="G2", name=f"G2.parity.{cap}", value=parity_value, threshold=gap,
                          passed=parity_value <= gap, detail=f"fast={fast:.6f} ledger={ledger:.6f}")
            )
    t_uew = float(active_t_stat(active))
    checks.append(
        GateCheck(gate="G3", name="G3.active_t.U_EW", value=t_uew, threshold=float(gates.min_active_t),
                  passed=t_uew >= float(gates.min_active_t), detail=f"t={t_uew:.4f}")
    )
    t_cw = float(active_t_stat(np.asarray(evidence.active_cw, dtype=np.float64)))
    checks.append(
        GateCheck(gate="G3", name="G3.active_t.CW", value=t_cw, threshold=float(gates.min_active_t),
                  passed=t_cw >= float(gates.min_active_t), detail=f"t={t_cw:.4f}")
    )
    loss_prob = _loss_probability(active, protocol)
    checks.append(
        GateCheck(gate="G3", name="G3.bootstrap_loss", value=loss_prob,
                  threshold=float(gates.max_bootstrap_loss_probability),
                  passed=loss_prob <= float(gates.max_bootstrap_loss_probability),
                  detail=f"draws={stats.bootstrap_draws} block={stats.bootstrap_block_sessions}")
    )
    n_trials = float(evidence.effective_trials) + float(protocol.prior_trials)
    try:
        dsr = float(
            deflated_sharpe_ratio(
                active,
                trial_sharpes=np.asarray(evidence.trial_active_sharpes, dtype=np.float64),
                n_trials=n_trials,
            )
        )
    except ValueError:
        dsr = 0.0
    checks.append(
        GateCheck(gate="G3", name="G3.dsr", value=dsr, threshold=float(gates.min_dsr),
                  passed=dsr >= float(gates.min_dsr), detail=f"n_trials={n_trials:.2f}")
    )
    try:
        pbo = float(cscv_pbo(np.asarray(evidence.family_active, dtype=np.float64), blocks=stats.cscv_blocks).pbo)
    except ValueError:
        pbo = 1.0
    checks.append(
        GateCheck(gate="G3", name="G3.pbo", value=pbo, threshold=float(gates.max_pbo),
                  passed=pbo <= float(gates.max_pbo), detail=f"blocks={stats.cscv_blocks}")
    )
    share, n_cols = _plateau_share(np.asarray(evidence.family_active, dtype=np.float64), spy)
    checks.append(
        GateCheck(gate="G3", name="G3.plateau", value=share,
                  threshold=float(gates.min_plateau_positive_share),
                  passed=n_cols >= _MIN_FAMILY_COLUMNS and share >= float(gates.min_plateau_positive_share),
                  detail=f"positive_share={share:.4f} members={n_cols}")
    )
    year_share, n_years = _positive_year_share(evidence.sessions, active)
    checks.append(
        GateCheck(gate="G4", name="G4.positive_years", value=year_share,
                  threshold=float(gates.min_positive_year_fraction),
                  passed=n_years > 0 and year_share >= float(gates.min_positive_year_fraction),
                  detail=f"years={n_years}")
    )
    concentration, has_base = _single_year_share(evidence.sessions, active)
    checks.append(
        GateCheck(gate="G4", name="G4.single_year_share", value=concentration,
                  threshold=float(gates.max_single_year_share),
                  passed=has_base and concentration <= float(gates.max_single_year_share),
                  detail="ok" if has_base else "denominator=0")
    )
    return tuple(checks)


def holdout_checks(
    *,
    holdout_active_uew: NDArray[np.float64],
    discovery_active_uew: NDArray[np.float64],
    protocol: ResearchProtocol,
) -> tuple[GateCheck, ...]:
    """Evaluate G5 from holdout active returns against the discovery distribution."""
    spy = protocol.sessions_per_year
    stats = protocol.statistics
    gates = protocol.gates
    holdout = np.asarray(holdout_active_uew, dtype=np.float64)
    discovery = np.asarray(discovery_active_uew, dtype=np.float64)
    holdout_g = _growth(holdout, spy)
    first = GateCheck(gate="G5", name="G5.active_positive", value=holdout_g, threshold=0.0,
                      passed=holdout_g > 0.0, detail=f"holdout_g={holdout_g:.6f}")
    draws = block_bootstrap_annualized_means(
        discovery,
        block=stats.bootstrap_block_sessions,
        draws=stats.bootstrap_draws,
        seed=stats.bootstrap_seed,
        sessions_per_year=spy,
        horizon=int(holdout.size),
    )
    threshold = float(np.quantile(draws, float(gates.holdout_min_percentile)))
    second = GateCheck(gate="G5", name="G5.within_distribution", value=holdout_g, threshold=threshold,
                       passed=holdout_g >= threshold,
                       detail=f"percentile={float(gates.holdout_min_percentile):.4f} draws={stats.bootstrap_draws}")
    return (first, second)
