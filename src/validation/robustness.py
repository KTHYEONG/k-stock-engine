"""Champion v1 stress, ablation, and promotion verdicts."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from src.engine.fill_model import ExecutionScenario
from src.validation.bootstrap import BootstrapConfig
from src.validation.metrics import LedgerMetrics
from src.validation.runner import WalkForwardValidationArtifact

__all__ = [
    "FactorAblationEvidence",
    "FactorName",
    "IntegrityCheck",
    "IntegrityEvidence",
    "ParameterProbe",
    "PromotionEvidence",
    "PromotionGate",
    "PromotionGateResult",
    "PromotionMetricSnapshot",
    "PromotionRunMetadata",
    "PromotionStatus",
    "PromotionVerdict",
    "YearlyPerformance",
    "append_promotion_verdict",
    "assess_alpha_concentration",
    "evaluate_promotion",
    "summarize_walk_forward_validation",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_GRID: tuple[tuple[int, int], ...] = (
    (15, 4),
    (15, 5),
    (15, 10),
    (20, 4),
    (20, 5),
    (20, 10),
    (25, 4),
    (25, 5),
    (25, 10),
)


class IntegrityCheck(StrEnum):
    LOOK_AHEAD = "look_ahead"
    DUPLICATE = "duplicate"
    UNKNOWN_ACTION = "unknown_action"
    SURVIVORSHIP = "survivorship"
    FUTURE_FILING = "future_filing"
    LEDGER_MISMATCH = "ledger_mismatch"


class FactorName(StrEnum):
    Q = "Q"
    V = "V"
    E = "E"
    F = "F"


class PromotionGate(StrEnum):
    DATA_INTEGRITY = "data_integrity"
    OOS_PERFORMANCE = "oos_performance"
    YEAR_STABILITY = "year_stability"
    ALPHA_CONCENTRATION = "alpha_concentration"
    COST_STRESS = "cost_stress"
    PARAMETER_STABILITY = "parameter_stability"
    FACTOR_ABLATION = "factor_ablation"


class PromotionStatus(StrEnum):
    PASS = "pass"  # noqa: S105
    FAIL = "fail"
    INCOMPLETE = "incomplete"


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ValueError(f"{field} must be SHA-256 hex")


def _require_sha1(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA1_RE.match(value):
        raise ValueError(f"{field} must be 40-char hex")


def _require_hash_tuple(values: tuple[str, ...], field: str) -> None:
    if not isinstance(values, tuple) or len(values) == 0:
        raise ValueError(f"{field} must be non-empty tuple")
    seen: set[str] = set()
    for v in values:
        _require_sha256(v, field)
        if v in seen:
            raise ValueError(f"{field} must be unique")
        seen.add(v)
    if tuple(sorted(values)) != values:
        raise ValueError(f"{field} must be ordered")


@dataclass(frozen=True, slots=True)
class YearlyPerformance:
    year: int
    champion_return: float
    cap_weight_return: float
    equal_weight_return: float

    def __post_init__(self) -> None:
        if isinstance(self.year, bool) or not isinstance(self.year, int):
            raise ValueError("year must be integer")
        if self.year <= 0:
            raise ValueError("year must be positive")
        for name in ("champion_return", "cap_weight_return", "equal_weight_return"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class IntegrityEvidence:
    check: IntegrityCheck
    passed: bool
    artifact_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.check, IntegrityCheck):
            raise ValueError("check must be IntegrityCheck")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be bool")
        _require_sha256(self.artifact_hash, "artifact_hash")


@dataclass(frozen=True, slots=True)
class PromotionRunMetadata:
    run_id: str
    recorded_at: datetime
    git_commit: str
    dataset_ids: tuple[str, ...]
    hypothesis: str
    frozen_parameters: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not isinstance(self.recorded_at, datetime) or self.recorded_at.tzinfo is None:
            raise ValueError("recorded_at must be timezone-aware")
        _require_sha1(self.git_commit, "git_commit")
        if not isinstance(self.dataset_ids, tuple) or len(self.dataset_ids) == 0:
            raise ValueError("dataset_ids must be non-empty tuple")
        seen_ds: set[str] = set()
        for d in self.dataset_ids:
            if not isinstance(d, str) or not d.strip():
                raise ValueError("dataset_ids must be non-empty strings")
            if d in seen_ds:
                raise ValueError("dataset_ids must be unique")
            seen_ds.add(d)
        if tuple(sorted(self.dataset_ids)) != self.dataset_ids:
            raise ValueError("dataset_ids must be ordered")
        if not isinstance(self.hypothesis, str) or not self.hypothesis.strip():
            raise ValueError("hypothesis must be non-empty")
        if not isinstance(self.frozen_parameters, tuple):
            raise ValueError("frozen_parameters must be tuple")
        seen_fp: set[tuple[str, str]] = set()
        for pair in self.frozen_parameters:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("frozen_parameters must contain pairs")
            k, v = pair
            if not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip():
                raise ValueError("frozen_parameters must contain non-empty strings")
            if pair in seen_fp:
                raise ValueError("frozen_parameters must be unique")
            seen_fp.add(pair)
        if tuple(sorted(self.frozen_parameters)) != self.frozen_parameters:
            raise ValueError("frozen_parameters must be ordered")


@dataclass(frozen=True, slots=True)
class PromotionMetricSnapshot:
    base_metrics: LedgerMetrics
    stress_metrics: LedgerMetrics
    cap_weight_metrics: LedgerMetrics
    equal_weight_metrics: LedgerMetrics
    yearly: tuple[YearlyPerformance, ...]
    bootstrap_config: BootstrapConfig
    bootstrap_distribution_hash: str
    input_artifact_hashes: tuple[str, ...]
    execution_scenarios: tuple[ExecutionScenario, ...]

    def __post_init__(self) -> None:
        for name in ("base_metrics", "stress_metrics", "cap_weight_metrics", "equal_weight_metrics"):
            if not isinstance(getattr(self, name), LedgerMetrics):
                raise ValueError(f"{name} must be LedgerMetrics")
        if not isinstance(self.yearly, tuple) or len(self.yearly) == 0:
            raise ValueError("yearly must be non-empty tuple")
        years: list[int] = []
        for y in self.yearly:
            if not isinstance(y, YearlyPerformance):
                raise ValueError("yearly must contain YearlyPerformance")
            years.append(y.year)
        if len(set(years)) != len(years):
            raise ValueError("yearly must be unique")
        if tuple(sorted(years)) != tuple(years):
            raise ValueError("yearly must be ordered")
        if not isinstance(self.bootstrap_config, BootstrapConfig):
            raise ValueError("bootstrap_config must be BootstrapConfig")
        _require_sha256(self.bootstrap_distribution_hash, "bootstrap_distribution_hash")
        _require_hash_tuple(self.input_artifact_hashes, "input_artifact_hashes")
        if self.execution_scenarios != (
            ExecutionScenario.BASE,
            ExecutionScenario.STRESS,
            ExecutionScenario.BASE,
            ExecutionScenario.BASE,
        ):
            raise ValueError("execution_scenarios must be BASE, STRESS, BASE, BASE")


@dataclass(frozen=True, slots=True)
class ParameterProbe:
    portfolio_size: int
    rebalance_sessions: int
    snapshot: PromotionMetricSnapshot

    def __post_init__(self) -> None:
        for name in ("portfolio_size", "rebalance_sessions"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{name} must be integer")
            if val <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.snapshot, PromotionMetricSnapshot):
            raise ValueError("snapshot must be PromotionMetricSnapshot")


@dataclass(frozen=True, slots=True)
class FactorAblationEvidence:
    removed_factor: FactorName
    snapshot: PromotionMetricSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.removed_factor, FactorName):
            raise ValueError("removed_factor must be FactorName")
        if not isinstance(self.snapshot, PromotionMetricSnapshot):
            raise ValueError("snapshot must be PromotionMetricSnapshot")


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    metadata: PromotionRunMetadata
    baseline: PromotionMetricSnapshot | None
    integrity_checks: tuple[IntegrityEvidence, ...]
    parameter_probes: tuple[ParameterProbe, ...]
    factor_ablations: tuple[FactorAblationEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PromotionRunMetadata):
            raise ValueError("metadata must be PromotionRunMetadata")
        if self.baseline is not None and not isinstance(self.baseline, PromotionMetricSnapshot):
            raise ValueError("baseline must be PromotionMetricSnapshot or None")
        if not isinstance(self.integrity_checks, tuple):
            raise ValueError("integrity_checks must be tuple")
        for integrity_item in self.integrity_checks:
            if not isinstance(integrity_item, IntegrityEvidence):
                raise ValueError("integrity_checks must contain IntegrityEvidence")
        if not isinstance(self.parameter_probes, tuple):
            raise ValueError("parameter_probes must be tuple")
        for probe_item in self.parameter_probes:
            if not isinstance(probe_item, ParameterProbe):
                raise ValueError("parameter_probes must contain ParameterProbe")
        if not isinstance(self.factor_ablations, tuple):
            raise ValueError("factor_ablations must be tuple")
        for ablation_item in self.factor_ablations:
            if not isinstance(ablation_item, FactorAblationEvidence):
                raise ValueError("factor_ablations must contain FactorAblationEvidence")


@dataclass(frozen=True, slots=True)
class PromotionGateResult:
    gate: PromotionGate
    passed: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.gate, PromotionGate):
            raise ValueError("gate must be PromotionGate")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be bool")
        if not isinstance(self.reasons, tuple):
            raise ValueError("reasons must be tuple")
        for r in self.reasons:
            if not isinstance(r, str) or not r.strip():
                raise ValueError("reasons must be non-empty strings")


def _ledger_metrics_to_dict(m: LedgerMetrics) -> dict[str, object]:
    return {
        "annualized_log_growth": float(m.annualized_log_growth),
        "annualized_volatility": float(m.annualized_volatility),
        "cagr": float(m.cagr),
        "calmar": None if m.calmar is None else float(m.calmar),
        "max_drawdown": float(m.max_drawdown),
        "sortino": None if m.sortino is None else float(m.sortino),
    }


def _bootstrap_config_to_dict(c: BootstrapConfig) -> dict[str, object]:
    return {
        "block_length_sessions": int(c.block_length_sessions),
        "method": c.method.value,
        "promotion_run": bool(c.promotion_run),
        "resamples": int(c.resamples),
        "seed": int(c.seed),
    }


def _snapshot_to_dict(s: PromotionMetricSnapshot) -> dict[str, object]:
    return {
        "base_metrics": _ledger_metrics_to_dict(s.base_metrics),
        "bootstrap_config": _bootstrap_config_to_dict(s.bootstrap_config),
        "bootstrap_distribution_hash": s.bootstrap_distribution_hash,
        "cap_weight_metrics": _ledger_metrics_to_dict(s.cap_weight_metrics),
        "equal_weight_metrics": _ledger_metrics_to_dict(s.equal_weight_metrics),
        "input_artifact_hashes": list(s.input_artifact_hashes),
        "execution_scenarios": [scenario.value for scenario in s.execution_scenarios],
        "stress_metrics": _ledger_metrics_to_dict(s.stress_metrics),
        "yearly": [
            {
                "cap_weight_return": float(y.cap_weight_return),
                "champion_return": float(y.champion_return),
                "equal_weight_return": float(y.equal_weight_return),
                "year": int(y.year),
            }
            for y in s.yearly
        ],
    }


def _metadata_to_dict(m: PromotionRunMetadata) -> dict[str, object]:
    recorded = m.recorded_at.astimezone(UTC).isoformat()
    return {
        "dataset_ids": list(m.dataset_ids),
        "frozen_parameters": [list(pair) for pair in m.frozen_parameters],
        "git_commit": m.git_commit,
        "hypothesis": m.hypothesis,
        "recorded_at": recorded,
        "run_id": m.run_id,
    }


@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    schema_version: str
    status: PromotionStatus
    metadata: PromotionRunMetadata
    gate_results: tuple[PromotionGateResult, ...]
    input_artifact_hashes: tuple[str, ...]
    metrics: PromotionMetricSnapshot | None

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty")
        if not isinstance(self.status, PromotionStatus):
            raise ValueError("status must be PromotionStatus")
        if not isinstance(self.metadata, PromotionRunMetadata):
            raise ValueError("metadata must be PromotionRunMetadata")
        if not isinstance(self.gate_results, tuple) or len(self.gate_results) == 0:
            raise ValueError("gate_results must be non-empty tuple")
        for g in self.gate_results:
            if not isinstance(g, PromotionGateResult):
                raise ValueError("gate_results must contain PromotionGateResult")
        _require_hash_tuple(self.input_artifact_hashes, "input_artifact_hashes")
        if self.metrics is not None and not isinstance(self.metrics, PromotionMetricSnapshot):
            raise ValueError("metrics must be PromotionMetricSnapshot or None")

    def to_canonical_json(self) -> str:
        payload: dict[str, object] = {
            "gate_results": [
                {"gate": g.gate.value, "passed": bool(g.passed), "reasons": list(g.reasons)}
                for g in self.gate_results
            ],
            "input_artifact_hashes": list(self.input_artifact_hashes),
            "metadata": _metadata_to_dict(self.metadata),
            "metrics": None if self.metrics is None else _snapshot_to_dict(self.metrics),
            "schema_version": self.schema_version,
            "status": self.status.value,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()


def _yearly_nav_returns(
    navs: tuple[object, ...],
) -> dict[int, float]:
    from src.core.ledger import LedgerNav

    by_year: dict[int, list[LedgerNav]] = {}
    for nav in navs:
        assert isinstance(nav, LedgerNav)
        year = nav.as_of.astimezone(UTC).year
        by_year.setdefault(year, []).append(nav)
    out: dict[int, float] = {}
    for year in sorted(by_year):
        marks = sorted(by_year[year], key=lambda m: m.as_of)
        first = float(marks[0].nav)
        last = float(marks[-1].nav)
        if first <= 0 or last <= 0 or not math.isfinite(first) or not math.isfinite(last):
            raise ValueError("nav must be positive finite")
        out[year] = last / first - 1.0
    return out


def summarize_walk_forward_validation(
    artifact: WalkForwardValidationArtifact,
) -> PromotionMetricSnapshot:
    if not isinstance(artifact, WalkForwardValidationArtifact):
        raise ValueError("artifact must be WalkForwardValidationArtifact")
    champ = _yearly_nav_returns(artifact.champion_base_nav)
    cap = _yearly_nav_returns(artifact.cap_weight_base_nav)
    equal = _yearly_nav_returns(artifact.equal_weight_base_nav)
    years = sorted(set(champ) & set(cap) & set(equal))
    if not years:
        raise ValueError("no overlapping calendar years")
    yearly = tuple(
        YearlyPerformance(y, float(champ[y]), float(cap[y]), float(equal[y])) for y in years
    )
    dist_payload = json.dumps(
        {"values": [float(v) for v in artifact.bootstrap_distribution.values]},
        sort_keys=True,
        separators=(",", ":"),
    )
    dist_hash = hashlib.sha256(dist_payload.encode("utf-8")).hexdigest()
    nav_hashes: list[str] = []
    for seq in (
        artifact.champion_base_nav,
        artifact.champion_stress_nav,
        artifact.cap_weight_base_nav,
        artifact.equal_weight_base_nav,
    ):
        payload = json.dumps(
            [{"as_of": n.as_of.astimezone(UTC).isoformat(), "nav": float(n.nav)} for n in seq],
            sort_keys=True,
            separators=(",", ":"),
        )
        nav_hashes.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    input_hashes = tuple(sorted(set(nav_hashes)))
    return PromotionMetricSnapshot(
        artifact.champion_base_metrics,
        artifact.champion_stress_metrics,
        artifact.cap_weight_base_metrics,
        artifact.equal_weight_base_metrics,
        yearly,
        artifact.bootstrap_config,
        dist_hash,
        input_hashes,
        (ExecutionScenario.BASE, ExecutionScenario.STRESS, ExecutionScenario.BASE, ExecutionScenario.BASE),
    )


def assess_alpha_concentration(
    yearly: tuple[YearlyPerformance, ...],
) -> PromotionGateResult:
    gate = PromotionGate.ALPHA_CONCENTRATION
    if not isinstance(yearly, tuple) or len(yearly) == 0:
        return PromotionGateResult(gate, False, ("yearly must be non-empty",))
    years = [y.year for y in yearly]
    if any(not isinstance(y, YearlyPerformance) for y in yearly):
        return PromotionGateResult(gate, False, ("yearly must contain YearlyPerformance",))
    if len(set(years)) != len(years) or tuple(sorted(years)) != tuple(years):
        return PromotionGateResult(gate, False, ("yearly must be ordered unique",))
    cap_alphas: list[float] = []
    eq_alphas: list[float] = []
    for y in yearly:
        for v in (y.champion_return, y.cap_weight_return, y.equal_weight_return):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                return PromotionGateResult(gate, False, ("year values must be finite",))
            if float(v) <= -1.0:
                return PromotionGateResult(gate, False, ("year returns must exceed -100%",))
        try:
            a_cap = math.log1p(float(y.champion_return)) - math.log1p(float(y.cap_weight_return))
            a_eq = math.log1p(float(y.champion_return)) - math.log1p(float(y.equal_weight_return))
        except ValueError:
            return PromotionGateResult(gate, False, ("year returns must exceed -100%",))
        if not math.isfinite(a_cap) or not math.isfinite(a_eq):
            return PromotionGateResult(gate, False, ("annual alpha must be finite",))
        cap_alphas.append(a_cap)
        eq_alphas.append(a_eq)
    for label, alphas in (("cap weight", cap_alphas), ("equal weight", eq_alphas)):
        positive = [a for a in alphas if a > 0]
        total = sum(positive)
        if not math.isfinite(total) or total <= 0:
            return PromotionGateResult(
                gate, False, (f"total positive alpha against {label} is not positive",)
            )
        peak = max(positive)
        if not (peak < 0.5 * total):
            return PromotionGateResult(
                gate,
                False,
                (f"single year explains >=50% of total positive alpha against {label}",),
            )
    return PromotionGateResult(gate, True, ())


def _eval_data_integrity(
    checks: tuple[IntegrityEvidence, ...],
) -> tuple[PromotionGateResult, bool, bool]:
    gate = PromotionGate.DATA_INTEGRITY
    for item in checks:
        if not item.passed:
            return (
                PromotionGateResult(gate, False, (f"{item.check.value} failed",)),
                False,
                True,
            )
    expected = set(IntegrityCheck)
    counts: dict[IntegrityCheck, int] = {}
    for item in checks:
        counts[item.check] = counts.get(item.check, 0) + 1
    if set(counts) != expected or any(v != 1 for v in counts.values()) or len(checks) != len(expected):
        missing = sorted(f"missing {c.value}" for c in expected if counts.get(c, 0) != 1)
        reasons = tuple(missing) if missing else ("integrity evidence incomplete",)
        return (PromotionGateResult(gate, False, reasons), True, False)
    return (PromotionGateResult(gate, True, ()), False, False)


def _is_valid_bootstrap(config: BootstrapConfig) -> bool:
    return (
        config.promotion_run is True
        and config.resamples >= 5000
        and 20 <= config.block_length_sessions <= 60
    )


def _metrics_finite(m: LedgerMetrics) -> bool:
    for v in (m.annualized_log_growth, m.cagr, m.annualized_volatility, m.max_drawdown):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return False
    if m.calmar is not None:
        calmar = m.calmar
        if isinstance(calmar, bool) or not isinstance(calmar, (int, float)):
            return False
        if not math.isfinite(float(calmar)):
            return False
    sortino = m.sortino
    if sortino is None:
        return True
    if isinstance(sortino, bool) or not isinstance(sortino, (int, float)):
        return False
    return math.isfinite(float(sortino))


def _eval_oos(baseline: PromotionMetricSnapshot | None) -> tuple[PromotionGateResult, bool]:
    gate = PromotionGate.OOS_PERFORMANCE
    if baseline is None:
        return (PromotionGateResult(gate, False, ("missing baseline",)), True)
    if not _is_valid_bootstrap(baseline.bootstrap_config):
        return (PromotionGateResult(gate, False, ("missing bootstrap evidence",)), True)
    base, cap, equal = baseline.base_metrics, baseline.cap_weight_metrics, baseline.equal_weight_metrics
    if not (_metrics_finite(base) and _metrics_finite(cap) and _metrics_finite(equal)):
        return (PromotionGateResult(gate, False, ("missing benchmark metrics",)), True)
    excess_cap = float(base.cagr) - float(cap.cagr)
    excess_eq = float(base.cagr) - float(equal.cagr)
    if not (math.isfinite(excess_cap) and math.isfinite(excess_eq)):
        return (PromotionGateResult(gate, False, ("excess CAGR not finite",)), False)
    if not (excess_cap >= 0.03 and excess_eq >= 0.03):
        return (PromotionGateResult(gate, False, ("excess CAGR below 3pp",)), False)
    vol = float(base.annualized_volatility)
    growth = float(base.annualized_log_growth)
    if not math.isfinite(vol) or not math.isfinite(growth) or vol <= 0:
        return (PromotionGateResult(gate, False, ("sharpe undefined on zero volatility",)), False)
    sharpe = growth / vol
    if not math.isfinite(sharpe) or not (sharpe >= 0.8):
        return (PromotionGateResult(gate, False, ("sharpe below 0.8",)), False)
    if not math.isfinite(float(base.max_drawdown)) or not (float(base.max_drawdown) <= 0.25):
        return (PromotionGateResult(gate, False, ("mdd above 0.25",)), False)
    if base.calmar is None or not math.isfinite(float(base.calmar)) or not (float(base.calmar) >= 0.5):
        return (PromotionGateResult(gate, False, ("calmar missing or below 0.5",)), False)
    return (PromotionGateResult(gate, True, ()), False)


def _yearly_structurally_valid(yearly: tuple[YearlyPerformance, ...]) -> bool:
    if not isinstance(yearly, tuple) or len(yearly) == 0:
        return False
    years = [y.year for y in yearly]
    if len(set(years)) != len(years) or tuple(sorted(years)) != tuple(years):
        return False
    for y in yearly:
        for v in (y.champion_return, y.cap_weight_return, y.equal_weight_return):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                return False
    return True


def _eval_year_stability(baseline: PromotionMetricSnapshot | None) -> tuple[PromotionGateResult, bool]:
    gate = PromotionGate.YEAR_STABILITY
    if baseline is None or not _yearly_structurally_valid(baseline.yearly):
        return (PromotionGateResult(gate, False, ("missing yearly evidence",)), True)
    yearly = baseline.yearly
    n = len(yearly)
    positive = sum(1 for y in yearly if float(y.champion_return) > 0)
    out_cap = sum(1 for y in yearly if float(y.champion_return) > float(y.cap_weight_return))
    out_eq = sum(1 for y in yearly if float(y.champion_return) > float(y.equal_weight_return))
    if not (positive / n >= 0.7):
        return (PromotionGateResult(gate, False, ("positive years below 70%",)), False)
    if not (out_cap / n >= 0.6 and out_eq / n >= 0.6):
        return (PromotionGateResult(gate, False, ("benchmark outperformance years below 60%",)), False)
    return (PromotionGateResult(gate, True, ()), False)


def _eval_concentration(
    baseline: PromotionMetricSnapshot | None,
) -> tuple[PromotionGateResult, bool]:
    gate = PromotionGate.ALPHA_CONCENTRATION
    if baseline is None or not _yearly_structurally_valid(baseline.yearly):
        return (PromotionGateResult(gate, False, ("missing yearly evidence",)), True)
    result = assess_alpha_concentration(baseline.yearly)
    if result.passed:
        return (result, False)
    if any("must be" in r or "missing" in r for r in result.reasons):
        return (result, True)
    return (result, False)


def _eval_cost_stress(baseline: PromotionMetricSnapshot | None) -> tuple[PromotionGateResult, bool]:
    gate = PromotionGate.COST_STRESS
    if baseline is None:
        return (PromotionGateResult(gate, False, ("missing stress evidence",)), True)
    stress_cagr = float(baseline.stress_metrics.cagr)
    if not math.isfinite(stress_cagr):
        return (PromotionGateResult(gate, False, ("stress CAGR not finite",)), False)
    if not (stress_cagr > 0):
        return (PromotionGateResult(gate, False, ("stress CAGR not positive",)), False)
    return (PromotionGateResult(gate, True, ()), False)


def _probe_weaker(snapshot: PromotionMetricSnapshot) -> float | None:
    base, cap, equal = snapshot.base_metrics, snapshot.cap_weight_metrics, snapshot.equal_weight_metrics
    if not (_metrics_finite(base) and _metrics_finite(cap) and _metrics_finite(equal)):
        return None
    weaker = min(float(base.cagr) - float(cap.cagr), float(base.cagr) - float(equal.cagr))
    if not math.isfinite(weaker):
        return None
    return weaker


def _eval_parameter_stability(
    baseline: PromotionMetricSnapshot | None,
    probes: tuple[ParameterProbe, ...],
) -> tuple[PromotionGateResult, bool]:
    gate = PromotionGate.PARAMETER_STABILITY
    if baseline is None:
        return (PromotionGateResult(gate, False, ("missing parameter evidence",)), True)
    if not isinstance(probes, tuple) or len(probes) != len(_REQUIRED_GRID):
        return (PromotionGateResult(gate, False, ("missing required parameter pair",)), True)
    keys = [(p.portfolio_size, p.rebalance_sessions) for p in probes]
    if len(set(keys)) != len(keys) or set(keys) != set(_REQUIRED_GRID):
        return (PromotionGateResult(gate, False, ("missing required parameter pair",)), True)
    by_key = dict(zip(keys, probes, strict=True))
    ref = by_key.get((20, 5))
    if ref is None or tuple(ref.snapshot.input_artifact_hashes) != tuple(baseline.input_artifact_hashes):
        return (PromotionGateResult(gate, False, ("baseline parameter hashes mismatch",)), False)
    baseline_weaker = _probe_weaker(baseline)
    if baseline_weaker is None or not (baseline_weaker >= 0.03):
        return (PromotionGateResult(gate, False, ("baseline excess below 3pp",)), False)
    threshold = 0.7 * float(baseline_weaker)
    for key in sorted(set(_REQUIRED_GRID) - {(20, 5)}):
        weaker = _probe_weaker(by_key[key].snapshot)
        if weaker is None or not (weaker >= threshold):
            return (
                PromotionGateResult(gate, False, (f"parameter probe {key} below 70% retention",)),
                False,
            )
    return (PromotionGateResult(gate, True, ()), False)


def _eval_factor_ablation(
    ablations: tuple[FactorAblationEvidence, ...],
) -> tuple[PromotionGateResult, bool]:
    gate = PromotionGate.FACTOR_ABLATION
    if not isinstance(ablations, tuple) or len(ablations) != len(FactorName):
        return (PromotionGateResult(gate, False, ("missing ablation evidence",)), True)
    factors = [a.removed_factor for a in ablations]
    if len(set(factors)) != len(factors) or set(factors) != set(FactorName):
        return (PromotionGateResult(gate, False, ("missing ablation evidence",)), True)
    seen: set[str] = set()
    for item in ablations:
        hashes = item.snapshot.input_artifact_hashes
        if len(hashes) == 0:
            return (PromotionGateResult(gate, False, ("missing ablation evidence",)), True)
        for h in hashes:
            if h in seen:
                return (PromotionGateResult(gate, False, ("duplicate ablation hashes",)), True)
            seen.add(h)
    return (PromotionGateResult(gate, True, ()), False)


def _collect_input_hashes(evidence: PromotionEvidence) -> tuple[str, ...]:
    acc: set[str] = set()
    for item in evidence.integrity_checks:
        acc.add(item.artifact_hash)
    snapshots: list[PromotionMetricSnapshot] = []
    if evidence.baseline is not None:
        snapshots.append(evidence.baseline)
    snapshots.extend(p.snapshot for p in evidence.parameter_probes)
    snapshots.extend(a.snapshot for a in evidence.factor_ablations)
    for snap in snapshots:
        acc.add(snap.bootstrap_distribution_hash)
        acc.update(snap.input_artifact_hashes)
    return tuple(sorted(acc))


def evaluate_promotion(evidence: PromotionEvidence) -> PromotionVerdict:
    if not isinstance(evidence, PromotionEvidence):
        raise ValueError("evidence must be PromotionEvidence")
    integrity_result, integrity_incomplete, integrity_hard_fail = _eval_data_integrity(
        evidence.integrity_checks
    )
    oos_result, oos_incomplete = _eval_oos(evidence.baseline)
    year_result, year_incomplete = _eval_year_stability(evidence.baseline)
    conc_result, conc_incomplete = _eval_concentration(evidence.baseline)
    cost_result, cost_incomplete = _eval_cost_stress(evidence.baseline)
    param_result, param_incomplete = _eval_parameter_stability(evidence.baseline, evidence.parameter_probes)
    abl_result, abl_incomplete = _eval_factor_ablation(evidence.factor_ablations)
    results = (
        integrity_result,
        oos_result,
        year_result,
        conc_result,
        cost_result,
        param_result,
        abl_result,
    )
    incompletes = (
        integrity_incomplete,
        oos_incomplete,
        year_incomplete,
        conc_incomplete,
        cost_incomplete,
        param_incomplete,
        abl_incomplete,
    )
    if integrity_hard_fail:
        status = PromotionStatus.FAIL
    elif any(incompletes):
        status = PromotionStatus.INCOMPLETE
    elif any(not r.passed for r in results):
        status = PromotionStatus.FAIL
    else:
        status = PromotionStatus.PASS
    return PromotionVerdict(
        "promotion-verdict-v1",
        status,
        evidence.metadata,
        results,
        _collect_input_hashes(evidence),
        evidence.baseline,
    )


def _registry_has_data_artifacts(path: Path) -> bool:
    parts = path.parts
    return any(parts[i] == "data" and parts[i + 1] == "artifacts" for i in range(len(parts) - 1))


def append_promotion_verdict(registry_path: Path, verdict: PromotionVerdict) -> Path:
    if not isinstance(registry_path, Path):
        raise ValueError("registry_path must be Path")
    if not isinstance(verdict, PromotionVerdict):
        raise ValueError("verdict must be PromotionVerdict")
    if not _registry_has_data_artifacts(registry_path):
        raise ValueError("registry_path must be below data/artifacts")
    if registry_path.suffix != ".jsonl":
        raise ValueError("registry_path must be jsonl below data/artifacts")
    line = verdict.to_canonical_json()
    run_id = verdict.metadata.run_id
    parent = registry_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        if not registry_path.is_file():
            raise ValueError("registry_path must be a file below data/artifacts")
        existing = registry_path.read_text(encoding="utf-8")
        for raw in existing.splitlines():
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("registry row is not canonical JSON") from exc
            meta = record.get("metadata", {}) if isinstance(record, dict) else {}
            existing_id = meta.get("run_id") if isinstance(meta, dict) else None
            if existing_id == run_id:
                raise ValueError(f"duplicate run_id {run_id}")
    with registry_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
    return registry_path
