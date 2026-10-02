"""Report card: robust-growth objective, survival guards, and diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from src.research.ledger_bridge import LedgerOutcome
from src.research.stats import breakeven_slippage_ticks, growth_profile, max_drawdown

__all__ = [
    "CheckResult",
    "EvaluationEvidence",
    "EvaluationPolicy",
    "ReportCard",
    "build_report_card",
]


class EvaluationPolicy(BaseModel):
    """Report-card parameters bound from protocol v4 (frozen, extra=forbid)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sessions_per_year: int
    block_sessions: int
    draws: int
    seed: int
    horizon_sessions: int
    objective_quantile: float
    ruin_mdd_limit: float
    max_p_ruin: float
    max_p_growth_le_zero: float
    report_mdd_limits: tuple[float, ...]
    recent_sessions: int

    @field_validator("sessions_per_year", "block_sessions", "draws", "horizon_sessions", "recent_sessions")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("evaluation policy session counts must be positive ints")
        return value

    @field_validator("seed")
    @classmethod
    def _non_negative_seed(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("evaluation policy seed must be a non-negative int")
        return value

    @field_validator("objective_quantile")
    @classmethod
    def _quantile_range(cls, value: float) -> float:
        amount = float(value)
        if not math.isfinite(amount) or not 0.0 < amount < 1.0:
            raise ValueError("objective_quantile must satisfy 0 < q < 1")
        return amount

    @field_validator("ruin_mdd_limit")
    @classmethod
    def _ruin_range(cls, value: float) -> float:
        amount = float(value)
        if not math.isfinite(amount) or not -1.0 < amount < 0.0:
            raise ValueError("ruin_mdd_limit must satisfy -1 < limit < 0")
        return amount

    @field_validator("max_p_ruin", "max_p_growth_le_zero")
    @classmethod
    def _probability_range(cls, value: float) -> float:
        amount = float(value)
        if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
            raise ValueError("guard probabilities must lie in [0, 1]")
        return amount

    @field_validator("report_mdd_limits")
    @classmethod
    def _report_limits(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        limits = tuple(float(v) for v in value)
        if not limits:
            raise ValueError("report_mdd_limits must be non-empty")
        for lim in limits:
            if not math.isfinite(lim) or not -1.0 < lim < 0.0:
                raise ValueError("report mdd limits must satisfy -1 < limit < 0")
        return limits


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    """Account-engine streams of one spec over one window (all arrays aligned to ``sessions``)."""

    sessions: tuple[date, ...]
    base: LedgerOutcome
    stress_slippage: LedgerOutcome
    stress_delay: LedgerOutcome
    cost_grid: Mapping[float, LedgerOutcome]
    unhedged: LedgerOutcome
    placebo: LedgerOutcome
    index_log_returns: NDArray[np.float64]
    universe_ew_log_returns: NDArray[np.float64]
    fast_sim_growth: float
    perturbation_mismatches: int


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    value: float
    threshold: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ReportCard:
    spec_hash: str
    run_id: str
    protocol_version: str
    start: date
    end: date
    objective_j: float
    objective_stream: str
    integrity: tuple[CheckResult, ...]
    guards: tuple[CheckResult, ...]
    metrics: Mapping[str, float]
    yearly_growth: Mapping[int, float]
    regime_growth: Mapping[str, float]
    cost_grid_growth: Mapping[float, float]
    breakeven_ticks: float
    controls: Mapping[str, float]
    recent: Mapping[str, float]

    @property
    def passed(self) -> bool:
        """True when every integrity and guard check passes."""
        return bool(
            all(check.passed for check in self.integrity)
            and all(check.passed for check in self.guards)
        )

    def canonical_json(self) -> str:
        """Key-sorted compact JSON; byte-identical for identical evidence."""
        payload = {
            "breakeven_ticks": _canon(self.breakeven_ticks),
            "controls": {k: _canon(v) for k, v in sorted(self.controls.items())},
            "cost_grid_growth": {str(k): _canon(v) for k, v in sorted(self.cost_grid_growth.items())},
            "end": self.end.isoformat(),
            "guards": [_canon_check(c) for c in self.guards],
            "integrity": [_canon_check(c) for c in self.integrity],
            "metrics": {k: _canon(v) for k, v in sorted(self.metrics.items())},
            "objective_j": _canon(self.objective_j),
            "objective_stream": self.objective_stream,
            "protocol_version": self.protocol_version,
            "recent": {k: _canon(v) for k, v in sorted(self.recent.items())},
            "regime_growth": {k: _canon(v) for k, v in sorted(self.regime_growth.items())},
            "run_id": self.run_id,
            "spec_hash": self.spec_hash,
            "start": self.start.isoformat(),
            "yearly_growth": {str(k): _canon(v) for k, v in sorted(self.yearly_growth.items())},
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        """SHA-256 hex of the canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _canon(value: float) -> Any:
    amount = float(value)
    if math.isnan(amount):
        return "nan"
    if math.isinf(amount):
        return "inf" if amount > 0 else "-inf"
    return amount


def _canon_check(check: CheckResult) -> dict[str, Any]:
    return {
        "name": check.name,
        "passed": bool(check.passed),
        "threshold": _canon(check.threshold),
        "value": _canon(check.value),
    }


def _finite_growth(values: NDArray[np.float64], *, sessions_per_year: int) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return math.nan
    return float(np.mean(finite) * sessions_per_year)


def _raw_growth(values: NDArray[np.float64], *, sessions_per_year: int) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        return math.nan
    return float(np.mean(x) * sessions_per_year)


def _point_stats(values: NDArray[np.float64], *, sessions_per_year: int) -> tuple[float, float, float, float, float, float]:
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    g = float(np.mean(finite) * sessions_per_year)
    try:
        cagr = float(math.exp(g) - 1.0) if math.isfinite(g) else math.nan
    except OverflowError:
        cagr = math.inf
    std = float(np.std(finite, ddof=1)) if finite.size >= 2 else 0.0
    vol = float(std * math.sqrt(sessions_per_year)) if math.isfinite(std) else math.nan
    sharpe = (
        float(np.mean(finite) / std * math.sqrt(sessions_per_year))
        if std > 0.0 and math.isfinite(std)
        else 0.0
    )
    mdd = float(max_drawdown(finite))
    calmar = math.inf if mdd == 0.0 else (float(cagr / abs(mdd)) if math.isfinite(cagr) else math.nan)
    return (g, cagr, vol, sharpe, mdd, calmar)


def _longest_underwater(values: NDArray[np.float64]) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return math.nan
    wealth = np.concatenate(([1.0], np.exp(np.cumsum(finite))))
    peak = np.maximum.accumulate(wealth)
    below = wealth[1:] < peak[1:]
    longest = 0
    run = 0
    for flag in below:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return float(longest)


def _ols_beta(strategy: NDArray[np.float64], index: NDArray[np.float64]) -> float:
    ys = np.asarray(strategy, dtype=np.float64).ravel()
    xs = np.asarray(index, dtype=np.float64).ravel()
    n = min(ys.size, xs.size)
    if n == 0:
        return math.nan
    ys = ys[:n]
    xs = xs[:n]
    mask = np.isfinite(ys) & np.isfinite(xs)
    if int(np.sum(mask)) < 2:
        return 0.0
    xd = xs[mask]
    yd = ys[mask]
    var = float(np.mean((xd - np.mean(xd)) ** 2))
    if var <= 0.0 or not math.isfinite(var):
        return 0.0
    cov = float(np.mean((xd - np.mean(xd)) * (yd - np.mean(yd))))
    return float(cov / var)


def _mean_nav(capital_krw: int, log_returns: NDArray[np.float64]) -> float:
    x = np.asarray(log_returns, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        return math.nan
    nav = float(capital_krw) * np.exp(np.cumsum(x))
    mean_nav = float(np.mean(nav))
    if not math.isfinite(mean_nav) or mean_nav <= 0.0:
        return math.nan
    return mean_nav


def _stock_only_growth(stock_returns: NDArray[np.float64], *, sessions_per_year: int) -> float:
    x = np.asarray(stock_returns, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return math.nan
    if np.any(finite <= -1.0):
        return math.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log1p(finite)
    return float(np.mean(logs) * sessions_per_year)


def _integrity_checks(evidence: EvaluationEvidence) -> tuple[CheckResult, ...]:
    checks: list[CheckResult] = []
    mismatches = int(evidence.perturbation_mismatches)
    checks.append(
        CheckResult(
            name="perturbation_mismatches",
            value=float(mismatches),
            threshold=0.0,
            passed=mismatches == 0,
        )
    )
    expected = len(evidence.sessions)
    outcomes: dict[str, LedgerOutcome] = {
        "base": evidence.base,
        "stress_slippage": evidence.stress_slippage,
        "stress_delay": evidence.stress_delay,
        "unhedged": evidence.unhedged,
        "placebo": evidence.placebo,
    }
    for tick in sorted(evidence.cost_grid):
        outcomes[f"cost_{tick}"] = evidence.cost_grid[tick]
    for name, outcome in outcomes.items():
        logs = np.asarray(outcome.log_returns, dtype=np.float64).ravel()
        aligned = logs.size == expected and bool(np.all(np.isfinite(logs)))
        checks.append(
            CheckResult(
                name=f"integrity_{name}",
                value=0.0 if aligned else 1.0,
                threshold=0.0,
                passed=aligned,
            )
        )
    return tuple(checks)


def build_report_card(
    evidence: EvaluationEvidence,
    policy: EvaluationPolicy,
    *,
    spec_hash: str,
    run_id: str,
    protocol_version: str,
) -> ReportCard:
    """Compute J on the stress stream with the lower objective quantile, guards on that same stream, and
    diagnostics from the base stream.

    Integrity checks (blocking): ``perturbation_mismatches == 0``; every outcome has
    ``len(log_returns) == len(sessions)`` and finite values.
    Guards (blocking): ``P(MDD < ruin_mdd_limit) ≤ max_p_ruin``; ``P(g ≤ 0) ≤ max_p_growth_le_zero``.
    """
    if not evidence.sessions:
        raise ValueError("evidence sessions must be non-empty")
    spy = int(policy.sessions_per_year)
    limits = tuple(float(v) for v in policy.report_mdd_limits)
    ruin = float(policy.ruin_mdd_limit)
    stress_limits = tuple(dict.fromkeys([*limits, ruin]))

    def _stress_profile(outcome: LedgerOutcome) -> Any | None:
        try:
            return growth_profile(
                np.asarray(outcome.log_returns, dtype=np.float64),
                block=int(policy.block_sessions),
                draws=int(policy.draws),
                seed=int(policy.seed),
                horizon=int(policy.horizon_sessions),
                sessions_per_year=spy,
                quantile=float(policy.objective_quantile),
                mdd_limits=stress_limits,
            )
        except ValueError:
            return None

    slip_profile = _stress_profile(evidence.stress_slippage)
    delay_profile = _stress_profile(evidence.stress_delay)
    slip_q = float(slip_profile.g_quantile) if slip_profile is not None else math.nan
    delay_q = float(delay_profile.g_quantile) if delay_profile is not None else math.nan
    if math.isnan(slip_q):
        chosen_name = "stress_slippage"
        chosen = slip_profile
    elif math.isnan(delay_q) or delay_q < slip_q:
        chosen_name = "stress_delay"
        chosen = delay_profile
    else:
        chosen_name = "stress_slippage"
        chosen = slip_profile
    if chosen is not None:
        objective_j = float(chosen.g_quantile)
        p_ruin = float(chosen.p_mdd_below.get(ruin, math.nan))
        p_zero = float(chosen.p_growth_le_zero)
    else:
        objective_j = math.nan
        p_ruin = math.nan
        p_zero = math.nan
    guards = (
        CheckResult(
            name=f"p_mdd_below_{ruin}",
            value=p_ruin,
            threshold=float(policy.max_p_ruin),
            passed=math.isfinite(p_ruin) and p_ruin <= float(policy.max_p_ruin),
        ),
        CheckResult(
            name="p_growth_le_zero",
            value=p_zero,
            threshold=float(policy.max_p_growth_le_zero),
            passed=math.isfinite(p_zero) and p_zero <= float(policy.max_p_growth_le_zero),
        ),
    )

    base_logs = np.asarray(evidence.base.log_returns, dtype=np.float64).ravel()
    g, cagr, vol, sharpe, mdd, calmar = _point_stats(base_logs, sessions_per_year=spy)
    underwater = _longest_underwater(base_logs)
    try:
        base_profile = growth_profile(
            base_logs,
            block=int(policy.block_sessions),
            draws=int(policy.draws),
            seed=int(policy.seed),
            horizon=int(policy.horizon_sessions),
            sessions_per_year=spy,
            quantile=float(policy.objective_quantile),
            mdd_limits=limits,
        )
        g_median = float(base_profile.g_median)
        p_mdd_map = {float(k): float(v) for k, v in base_profile.p_mdd_below.items()}
    except ValueError:
        g_median = math.nan
        p_mdd_map = {float(lim): math.nan for lim in limits}

    metrics: dict[str, float] = {
        "g": g,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "max_underwater_sessions": underwater,
        "g_median_5y": g_median,
        "avg_stock_exposure": float(evidence.base.avg_stock_exposure),
        "avg_margin_share": float(evidence.base.avg_margin_share),
        "avg_inverse_share": float(evidence.base.avg_inverse_share),
        "turnover_per_year": float(evidence.base.turnover_per_year),
        "beta_to_index": _ols_beta(
            base_logs, np.asarray(evidence.index_log_returns, dtype=np.float64).ravel()
        ),
        "fast_sim_gap": float(evidence.fast_sim_growth)
        - _stock_only_growth(
            np.asarray(evidence.unhedged.stock_book_returns, dtype=np.float64).ravel(),
            sessions_per_year=spy,
        ),
    }
    for lim, prob in p_mdd_map.items():
        metrics[f"p_mdd_below_{lim}"] = float(prob)
    mean_nav = _mean_nav(int(evidence.base.capital_krw), base_logs)
    journal = dict(evidence.base.journal_totals_krw or {})
    n_sessions = int(base_logs.size)
    for kind in sorted(journal):
        total = float(journal[kind])
        if math.isnan(mean_nav) or n_sessions <= 0:
            metrics[f"cost_{kind}_per_year"] = math.nan
        else:
            metrics[f"cost_{kind}_per_year"] = float(total / mean_nav * spy / n_sessions)

    sessions = list(evidence.sessions)
    usable = min(len(sessions), int(base_logs.size))
    yearly: dict[int, float] = {}
    counts: dict[int, int] = {}
    for idx in range(usable):
        counts[sessions[idx].year] = counts.get(sessions[idx].year, 0) + 1
    for year in sorted({day.year for day in sessions}):
        idxs = [i for i in range(usable) if sessions[i].year == year]
        if not idxs:
            yearly[year] = math.nan
        else:
            yearly[year] = _finite_growth(base_logs[idxs], sessions_per_year=spy)
    for year, count in counts.items():
        metrics[f"sessions_{year}"] = float(count)

    index_arr = np.asarray(evidence.index_log_returns, dtype=np.float64).ravel()
    index_usable = min(len(sessions), int(index_arr.size), usable)
    year_index_sum: dict[int, float] = {}
    for year in sorted({day.year for day in sessions[:index_usable]}):
        idxs = [i for i in range(index_usable) if sessions[i].year == year]
        year_index_sum[year] = float(np.nansum(index_arr[idxs]))
    up_idxs: list[int] = []
    down_idxs: list[int] = []
    for i in range(index_usable):
        if year_index_sum.get(sessions[i].year, 0.0) > 0.0:
            up_idxs.append(i)
        else:
            down_idxs.append(i)
    regime = {
        "index_up_years": _finite_growth(base_logs[up_idxs], sessions_per_year=spy)
        if up_idxs
        else math.nan,
        "index_down_years": _finite_growth(base_logs[down_idxs], sessions_per_year=spy)
        if down_idxs
        else math.nan,
    }

    cost_grid_growth = {
        float(tick): _raw_growth(
            np.asarray(outcome.log_returns, dtype=np.float64).ravel(), sessions_per_year=spy
        )
        for tick, outcome in evidence.cost_grid.items()
    }
    try:
        breakeven = float(breakeven_slippage_ticks(cost_grid_growth))
    except ValueError:
        breakeven = math.nan

    controls = {
        "placebo_g": _finite_growth(
            np.asarray(evidence.placebo.log_returns, dtype=np.float64).ravel(), sessions_per_year=spy
        ),
        "universe_ew_g": _finite_growth(
            np.asarray(evidence.universe_ew_log_returns, dtype=np.float64).ravel(),
            sessions_per_year=spy,
        ),
        "unhedged_g": _finite_growth(
            np.asarray(evidence.unhedged.log_returns, dtype=np.float64).ravel(),
            sessions_per_year=spy,
        ),
        "index_g": _finite_growth(index_arr, sessions_per_year=spy),
    }

    recent_n = min(int(policy.recent_sessions), int(base_logs.size))
    recent_slice = base_logs[-recent_n:] if recent_n > 0 else np.zeros(0, dtype=np.float64)
    recent_mdd = float(max_drawdown(recent_slice)) if recent_slice.size else math.nan
    recent = {
        "g": _finite_growth(recent_slice, sessions_per_year=spy) if recent_slice.size else math.nan,
        "mdd": recent_mdd,
        "sessions": float(recent_n),
    }

    return ReportCard(
        spec_hash=spec_hash,
        run_id=run_id,
        protocol_version=protocol_version,
        start=sessions[0],
        end=sessions[-1],
        objective_j=objective_j,
        objective_stream=chosen_name,
        integrity=_integrity_checks(evidence),
        guards=guards,
        metrics=metrics,
        yearly_growth=yearly,
        regime_growth=regime,
        cost_grid_growth=cost_grid_growth,
        breakeven_ticks=breakeven,
        controls=controls,
        recent=recent,
    )
