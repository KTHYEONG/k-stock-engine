"""Evaluator that screens specs, assembles gate evidence, and enforces the lockbox."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.backtest.engine import DelistPolicy
from src.core.pit import PITDataError
from src.data.research_protocol import LockboxError, ResearchProtocol, Segment
from src.research.cube import CubeInputs, ResearchCube
from src.research.features import compute_features
from src.research.gates import (
    DiscoveryEvidence,
    GateReport,
    discovery_checks,
    holdout_checks,
)
from src.research.ledger_bridge import run_ledger
from src.research.registry import TrialRecord, TrialRegistry, TrialReturns
from src.research.simulator import SimConfig, simulate
from src.research.stats import (
    annualized_log_growth,
    effective_trial_count,
    max_drawdown,
)
from src.research.strategy import StrategySpec, benchmark_targets, target_weights

__all__ = ["EvaluationContext", "Evaluator"]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    protocol: ResearchProtocol
    cube: ResearchCube
    cube_inputs: CubeInputs
    registry: TrialRegistry
    lockbox: Any
    engine_config_path: Path
    market_cache_root: Path
    reports_root: Path
    dividends: pl.DataFrame
    now: Callable[[], datetime]


def _sessions_of(cube: ResearchCube) -> list[date]:
    return list(cube.sessions)


def _window_indices(sessions: Sequence[date], start: date, end: date) -> tuple[int, int]:
    index_of = {day: idx for idx, day in enumerate(sessions)}
    if start not in index_of or end not in index_of or index_of[start] > index_of[end]:
        raise ValueError(f"run window [{start}, {end}] is not within the cube sessions")
    return (index_of[start], index_of[end])


def _first_at_or_after(sessions: Sequence[date], day: date) -> date:
    for session in sessions:
        if session >= day:
            return session
    raise ValueError(f"window start {day} is after the last cube session")


def _last_at_or_before(sessions: Sequence[date], day: date) -> date:
    for session in reversed(list(sessions)):
        if session <= day:
            return session
    raise ValueError(f"window end {day} is before the first cube session")


class Evaluator:
    """Runs specs on authorized windows, records every run as a trial, and assembles gate evidence."""

    def __init__(self, context: EvaluationContext) -> None:
        self._ctx = context
        self._feature_cache: dict[str, NDArray[np.float64]] = {}
        self._base_config: SimConfig | None = None

    def _base_sim_config(self) -> SimConfig:
        if self._base_config is None:
            self._base_config = SimConfig.from_engine_toml(
                self._ctx.engine_config_path, capital_krw=self._ctx.protocol.primary_capital_krw
            )
        return self._base_config

    def _window_for(self, segment: Segment) -> tuple[date, date]:
        sessions = _sessions_of(self._ctx.cube)
        protocol = self._ctx.protocol
        if segment is Segment.DISCOVERY:
            window = protocol.discovery_window()
        elif segment is Segment.HOLDOUT:
            window = protocol.holdout_window()
        else:
            window = (protocol.forward_start, sessions[-1])
        start = _first_at_or_after(sessions, window[0])
        end = _last_at_or_before(sessions, window[1])
        if start > end:
            raise ValueError(f"segment window has no cube sessions: {segment.value}")
        return (start, end)

    def _features_for(self, names: Sequence[str]) -> dict[str, NDArray[np.float64]]:
        missing = [name for name in names if name not in self._feature_cache]
        if missing:
            computed = compute_features(self._ctx.cube, missing)
            self._feature_cache.update(computed)
        return {name: self._feature_cache[name] for name in names}

    def _target_window(self, start: date, end: date) -> tuple[int, int]:
        sessions = _sessions_of(self._ctx.cube)
        lo_sim, hi_sim = _window_indices(sessions, start, end)
        lo = max(0, lo_sim - 1)
        return (lo, hi_sim)

    def _record_trial(
        self,
        *,
        spec: StrategySpec,
        segment: Segment,
        start: date,
        end: date,
        config: SimConfig,
        targets: Mapping[int, NDArray[np.float64]],
    ) -> tuple[TrialRecord, TrialReturns]:
        ctx = self._ctx
        authorization = ctx.lockbox.authorize(start=start, end=end, spec_hash=spec.spec_hash)
        lo, hi = self._target_window(start, end)
        executable = {row: np.asarray(weights, dtype=np.float64) for row, weights in targets.items()
                      if lo - 1 - int(config.execution_delay) <= int(row) <= hi - 1 - int(config.execution_delay)}
        result = simulate(ctx.cube, executable, start=start, end=end, config=config, authorization=authorization)
        bench_maps = benchmark_targets(ctx.cube, ctx.protocol.benchmarks, lo=lo, hi=hi)
        benches: dict[str, NDArray[np.float64]] = {}
        for name, bench_targets in bench_maps.items():
            bench_exec = {row: np.asarray(w, dtype=np.float64) for row, w in bench_targets.items()
                          if lo - 1 - int(config.execution_delay) <= int(row) <= hi - 1 - int(config.execution_delay)}
            bench_result = simulate(ctx.cube, bench_exec, start=start, end=end, config=config,
                                    authorization=authorization)
            benches[name] = np.ascontiguousarray(bench_result.log_returns)
        returns = TrialReturns(sessions=result.sessions, net=np.ascontiguousarray(result.log_returns),
                               benchmarks=benches)
        metrics = _trial_metrics(result.log_returns, benches, ctx.protocol.sessions_per_year,
                                 turnover=float(np.mean(result.turnover)) if result.turnover.size else 0.0,
                                 cost=float(np.mean(result.cost)) if result.cost.size else 0.0)
        record = ctx.registry.record(
            family=spec.family, spec_hash=spec.spec_hash, spec_json=spec.canonical_json(),
            segment=segment, sim_config_json=config.canonical_json(), cube_id=ctx.cube.cube_id,
            returns=returns, metrics=metrics, now=ctx.now(),
        )
        active_g = float(metrics.get("active_g_uew", 0.0))
        _LOG.info("[ALGO] trial id=%s family=%s g=%.6f active_g=%.6f",
                  record.trial_id, spec.family, float(metrics.get("g", 0.0)), active_g)
        return (record, returns)

    def screen(self, spec: StrategySpec, *, segment: Segment = Segment.DISCOVERY,
               config: SimConfig | None = None) -> TrialRecord:
        """Simulate one spec on its segment window and record the trial."""
        cfg = config if config is not None else self._base_sim_config()
        start, end = self._window_for(segment)
        features = self._features_for(spec.feature_names())
        lo, hi = self._target_window(start, end)
        targets = target_weights(spec, self._ctx.cube, features, lo=lo, hi=hi)
        record, _ = self._record_trial(spec=spec, segment=segment, start=start, end=end,
                                       config=cfg, targets=targets)
        return record

    def screen_family(self, specs: Sequence[StrategySpec]) -> tuple[TrialRecord, ...]:
        """Screen every family member on the discovery window."""
        return tuple(self.screen(spec) for spec in specs)

    def _simulate_unrecorded(
        self, *, targets: Mapping[int, NDArray[np.float64]], start: date, end: date, config: SimConfig,
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        ctx = self._ctx
        authorization = ctx.lockbox.authorize(start=start, end=end, spec_hash=None)
        lo, hi = self._target_window(start, end)
        executable = {row: np.asarray(w, dtype=np.float64) for row, w in targets.items()
                      if lo - 1 - int(config.execution_delay) <= int(row) <= hi - 1 - int(config.execution_delay)}
        result = simulate(ctx.cube, executable, start=start, end=end, config=config, authorization=authorization)
        benches: dict[str, NDArray[np.float64]] = {}
        for name, bench_targets in benchmark_targets(ctx.cube, ctx.protocol.benchmarks, lo=lo, hi=hi).items():
            bench_exec = {row: np.asarray(w, dtype=np.float64) for row, w in bench_targets.items()
                          if lo - 1 - int(config.execution_delay) <= int(row) <= hi - 1 - int(config.execution_delay)}
            bench_result = simulate(ctx.cube, bench_exec, start=start, end=end, config=config,
                                    authorization=authorization)
            benches[name] = np.ascontiguousarray(bench_result.log_returns)
        return (np.ascontiguousarray(result.log_returns), benches)

    def _perturbation_mismatches(self, spec: StrategySpec, base_targets: Mapping[int, NDArray[np.float64]],
                                 start: date, end: date) -> int:
        ctx = self._ctx
        sessions = _sessions_of(ctx.cube)
        lo_sim, hi_sim = _window_indices(sessions, start, end)
        cuts = int(ctx.protocol.gates.perturbation_cuts)
        seed = int(ctx.protocol.gates.perturbation_seed)
        raw_cuts = np.linspace(float(lo_sim), float(max(lo_sim, hi_sim - 1)), num=cuts)
        cut_rows = sorted({min(hi_sim, max(lo_sim, round(value))) for value in raw_cuts.tolist()})
        names = spec.feature_names()
        total = 0
        n_names = len(ctx.cube.instrument_ids)
        n_sessions = len(sessions)
        for position, cut in enumerate(cut_rows):
            rng = np.random.default_rng(seed + position)
            perturbed: dict[str, NDArray[Any]] = {}
            for name, arr in ctx.cube.arrays.items():
                copied = np.asarray(arr).copy()
                if copied.shape[0] == n_sessions and copied.ndim == 2:
                    for t in range(cut + 1, n_sessions):
                        perm = rng.permutation(n_names)
                        copied[t] = np.asarray(copied[t])[perm]
                perturbed[name] = copied
            exit_at = np.asarray(ctx.cube.exit_at, dtype=np.int64).copy()
            for n in range(n_names):
                if int(exit_at[n]) > cut and n_sessions - (cut + 1) > 0:
                    exit_at[n] = int(rng.integers(int(cut + 1), n_sessions + 1))
            halted = np.asarray(ctx.cube.exit_halted, dtype=bool).copy()
            pert_cube = ResearchCube.from_arrays(
                cube_id=ctx.cube.cube_id, sessions=list(sessions),
                instrument_ids=list(ctx.cube.instrument_ids), arrays=perturbed,
                exit_at=exit_at, exit_halted=halted,
            )
            pert_features = compute_features(pert_cube, list(names))
            lo, hi = self._target_window(start, end)
            pert_targets = target_weights(spec, pert_cube, pert_features, lo=lo, hi=hi)
            for row, weights in base_targets.items():
                if int(row) <= cut and int(row) in pert_targets and not np.array_equal(
                    np.asarray(weights, dtype=np.float64),
                    np.asarray(pert_targets[int(row)], dtype=np.float64),
                ):
                    total += 1
        return int(total)

    def _discovery_trials_matrix(self, sessions: tuple[date, ...]) -> tuple[list[NDArray[np.float64]], list[float]]:
        ctx = self._ctx
        sharpes: list[float] = []
        columns: list[NDArray[np.float64]] = []
        for trial in ctx.registry.trials(segment=Segment.DISCOVERY):
            try:
                stored = ctx.registry.returns(trial.trial_id)
            except PITDataError:
                continue
            if tuple(stored.sessions) != tuple(sessions):
                continue
            bench = stored.benchmarks.get("U_EW")
            if bench is None:
                continue
            active = np.asarray(stored.net, dtype=np.float64) - np.asarray(bench, dtype=np.float64)
            std = float(np.std(active, ddof=1)) if active.size >= 2 else 0.0
            sharpes.append(float(np.mean(active) / std) if std > 0.0 else 0.0)
            columns.append(np.asarray(stored.net, dtype=np.float64))
        return (columns, sharpes)

    def _base_discovery_returns(self, spec: StrategySpec) -> TrialReturns:
        ctx = self._ctx
        base_json = self._base_sim_config().canonical_json()
        candidates = [t for t in ctx.registry.trials(segment=Segment.DISCOVERY) if t.spec_hash == spec.spec_hash]
        for trial in candidates:
            if trial.sim_config_json == base_json:
                return ctx.registry.returns(trial.trial_id)
        if candidates:
            return ctx.registry.returns(candidates[0].trial_id)
        raise ValueError(f"no discovery trial for spec: {spec.spec_hash}")

    def validate(self, spec: StrategySpec, family: Sequence[StrategySpec]) -> GateReport:
        """Screen variants and family, assemble evidence, and write the discovery report."""
        ctx = self._ctx
        hashes = {member.spec_hash for member in family}
        if spec.spec_hash not in hashes:
            raise ValueError("family must contain the spec")
        start, end = self._window_for(Segment.DISCOVERY)
        base_config = self._base_sim_config()
        features = self._features_for(spec.feature_names())
        lo, hi = self._target_window(start, end)
        base_targets = target_weights(spec, ctx.cube, features, lo=lo, hi=hi)
        base_record, base_returns = self._record_trial(spec=spec, segment=Segment.DISCOVERY, start=start,
                                                       end=end, config=base_config, targets=base_targets)
        delay_config = base_config.model_copy(update={"execution_delay": 1})
        _, delay_returns = self._record_trial(spec=spec, segment=Segment.DISCOVERY, start=start, end=end,
                                              config=delay_config, targets=base_targets)
        stress_config = base_config.model_copy(
            update={"extra_slippage": float(ctx.protocol.gates.stress_extra_slippage)})
        _, stress_returns = self._record_trial(spec=spec, segment=Segment.DISCOVERY, start=start, end=end,
                                               config=stress_config, targets=base_targets)
        halted_config = base_config.model_copy(update={"halted_exit_value": 0.0})
        _, halted_returns = self._record_trial(spec=spec, segment=Segment.DISCOVERY, start=start, end=end,
                                               config=halted_config, targets=base_targets)
        for member in family:
            member_features = self._features_for(member.feature_names())
            member_targets = target_weights(member, ctx.cube, member_features, lo=lo, hi=hi)
            self._record_trial(spec=member, segment=Segment.DISCOVERY, start=start, end=end,
                               config=base_config, targets=member_targets)
        authorization = ctx.lockbox.authorize(start=start, end=end, spec_hash=spec.spec_hash)
        ledger_growth: dict[int, float] = {}
        ledger_bench_growth: dict[int, float] = {}
        fast_growth: dict[int, float] = {}
        for capital in ctx.protocol.gates.ledger_capitals:
            cap = int(capital)
            cap_config = base_config.model_copy(update={"capital_krw": cap})
            fast_net, fast_bench = self._simulate_unrecorded(targets=base_targets, start=start, end=end,
                                                             config=cap_config)
            spy = ctx.protocol.sessions_per_year
            fast_growth[cap] = float(annualized_log_growth(fast_net, sessions_per_year=spy))
            ledger_bench_growth[cap] = float(
                annualized_log_growth(np.asarray(fast_bench["U_EW"], dtype=np.float64), sessions_per_year=spy))
            outcome = run_ledger(
                cube=ctx.cube, targets=base_targets, panel_dir=Path(ctx.cube_inputs.market_panel),
                dividends=ctx.dividends, engine_config_path=Path(ctx.engine_config_path),
                rules=ctx.cube_inputs.market_rules, market_cache_root=Path(ctx.market_cache_root),
                capital_krw=cap, halted_exit_policy=DelistPolicy.ZERO, start=start, end=end,
                authorization=authorization,
            )
            ledger_growth[cap] = float(annualized_log_growth(np.asarray(outcome.log_returns, dtype=np.float64),
                                                             sessions_per_year=spy))
        mismatches = self._perturbation_mismatches(spec, base_targets, start, end)
        columns, sharpes = self._discovery_trials_matrix(base_returns.sessions)
        if len(columns) >= 1:
            matrix = np.column_stack(columns)
            try:
                effective = float(effective_trial_count(matrix))
            except ValueError:
                effective = 1.0
        else:
            effective = 1.0  # pragma: no cover - the base trial always contributes a column
        family_cols: list[NDArray[np.float64]] = []
        for member in family:
            stored = self._find_member_returns(member.spec_hash, base_returns.sessions)
            bench = stored.benchmarks.get("U_EW")
            if bench is None:
                raise ValueError(f"family trial is missing U_EW benchmark: {member.spec_hash}")
            family_cols.append(np.asarray(stored.net, dtype=np.float64) - np.asarray(bench, dtype=np.float64))
        family_active = np.column_stack(family_cols) if family_cols else np.zeros((len(base_returns.sessions), 0))
        base_uew = _active_of(base_returns, "U_EW")
        evidence = DiscoveryEvidence(
            sessions=tuple(base_returns.sessions),
            active_uew=np.ascontiguousarray(base_uew),
            active_cw=np.ascontiguousarray(_active_of(base_returns, "CW")),
            delayed_active_uew=np.ascontiguousarray(_active_of(delay_returns, "U_EW")),
            stressed_active_uew=np.ascontiguousarray(_active_of(stress_returns, "U_EW")),
            halted_zero_active_uew=np.ascontiguousarray(_active_of(halted_returns, "U_EW")),
            ledger_growth=dict(ledger_growth),
            ledger_benchmark_growth=dict(ledger_bench_growth),
            fast_growth=dict(fast_growth),
            perturbation_mismatches=int(mismatches),
            trial_active_sharpes=np.asarray(sharpes, dtype=np.float64),
            effective_trials=float(effective),
            family_active=np.ascontiguousarray(family_active),
        )
        checks = discovery_checks(evidence, ctx.protocol)
        report = GateReport(spec_hash=spec.spec_hash, trial_id=base_record.trial_id, segment=Segment.DISCOVERY,
                            protocol_version=ctx.protocol.version, checks=checks)
        ctx.reports_root.mkdir(parents=True, exist_ok=True)
        (ctx.reports_root / f"{spec.spec_hash}_discovery.json").write_text(
            report.canonical_json() + "\n", encoding="utf-8")
        failed = [check.name for check in checks if not check.passed]
        _LOG.info("[RISK] gates spec=%s passed=%s failed=%s", spec.spec_hash, report.passed, failed)
        return report

    def _find_member_returns(self, spec_hash: str, sessions: tuple[date, ...]) -> TrialReturns:
        ctx = self._ctx
        for trial in ctx.registry.trials(segment=Segment.DISCOVERY):
            if trial.spec_hash != spec_hash:
                continue
            stored = ctx.registry.returns(trial.trial_id)
            if tuple(stored.sessions) == tuple(sessions):
                return stored
        raise ValueError(f"no discovery trial for family member: {spec_hash}")

    def register_finalists(self, specs: Sequence[StrategySpec]) -> None:
        """Register specs with stored passing discovery reports."""
        from src.data.research_protocol import FinalistRecord

        ctx = self._ctx
        records: list[FinalistRecord] = []
        for spec in specs:
            path = ctx.reports_root / f"{spec.spec_hash}_discovery.json"
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except OSError as exc:
                raise ValueError(f"missing discovery report: {spec.spec_hash}") from exc
            trial_id = str(raw["trial_id"])
            checks = tuple(raw["checks"])
            rebuilt = GateReport(spec_hash=str(raw["spec_hash"]), trial_id=trial_id,
                                 segment=Segment(str(raw["segment"])),
                                 protocol_version=str(raw["protocol_version"]),
                                 checks=tuple(
                                     _gate_check_from_json(item) for item in checks
                                 ))
            if rebuilt.digest != _digest_of_canonical(path):
                raise ValueError(f"discovery report digest mismatch: {spec.spec_hash}")
            if not rebuilt.passed:
                raise ValueError(f"discovery report did not pass: {spec.spec_hash}")
            records.append(FinalistRecord(spec_hash=spec.spec_hash, family=spec.family,
                                          discovery_trial_id=trial_id,
                                          gate_report_digest=rebuilt.digest, registered_at=ctx.now()))
        ctx.lockbox.register_finalists(records)

    def holdout(self, spec: StrategySpec) -> GateReport:
        """Run the one-shot holdout test for a registered finalist."""
        ctx = self._ctx
        start, end = self._window_for(Segment.HOLDOUT)
        authorization = ctx.lockbox.authorize(start=start, end=end, spec_hash=spec.spec_hash)
        if not authorization.evidence:
            raise LockboxError(f"holdout is not authorized for spec: {spec.spec_hash}")
        base_config = self._base_sim_config()
        features = self._features_for(spec.feature_names())
        lo, hi = self._target_window(start, end)
        targets = target_weights(spec, ctx.cube, features, lo=lo, hi=hi)
        record, returns = self._record_trial(spec=spec, segment=Segment.HOLDOUT, start=start, end=end,
                                             config=base_config, targets=targets)
        holdout_active = _active_of(returns, "U_EW")
        discovery_returns = self._base_discovery_returns(spec)
        discovery_active = _active_of(discovery_returns, "U_EW")
        checks = holdout_checks(holdout_active_uew=np.ascontiguousarray(holdout_active),
                                discovery_active_uew=np.ascontiguousarray(discovery_active),
                                protocol=ctx.protocol)
        report = GateReport(spec_hash=spec.spec_hash, trial_id=record.trial_id, segment=Segment.HOLDOUT,
                            protocol_version=ctx.protocol.version, checks=checks)
        per_capital: dict[str, dict[str, float]] = {}
        for capital in ctx.protocol.gates.ledger_capitals:
            cap = int(capital)
            cap_config = base_config.model_copy(update={"capital_krw": cap})
            fast_net, _ = self._simulate_unrecorded(targets=targets, start=start, end=end, config=cap_config)
            spy = ctx.protocol.sessions_per_year
            per_capital[str(cap)] = {
                "cagr": float(np.exp(float(annualized_log_growth(fast_net, sessions_per_year=spy))) - 1.0),
                "g": float(annualized_log_growth(fast_net, sessions_per_year=spy)),
                "mdd": float(max_drawdown(fast_net)),
                "volatility": float(np.std(fast_net, ddof=1) * np.sqrt(spy)) if fast_net.size >= 2 else 0.0,
                "turnover": 0.0,
                "cost": 0.0,
            }
            for policy in (DelistPolicy.ZERO, DelistPolicy.LAST_CLOSE):
                outcome = run_ledger(
                    cube=ctx.cube, targets=targets, panel_dir=Path(ctx.cube_inputs.market_panel),
                    dividends=ctx.dividends, engine_config_path=Path(ctx.engine_config_path),
                    rules=ctx.cube_inputs.market_rules, market_cache_root=Path(ctx.market_cache_root),
                    capital_krw=cap, halted_exit_policy=policy, start=start, end=end,
                    authorization=authorization,
                )
                per_capital[str(cap)][f"ledger_g_{policy.value}"] = float(
                    annualized_log_growth(np.asarray(outcome.log_returns, dtype=np.float64),
                                          sessions_per_year=spy))
        ctx.lockbox.record_holdout_verdict(spec_hash=spec.spec_hash, passed=report.passed,
                                           report_digest=report.digest)
        ctx.reports_root.mkdir(parents=True, exist_ok=True)
        payload = json.loads(report.canonical_json())
        payload["per_capital"] = per_capital
        (ctx.reports_root / f"{spec.spec_hash}_holdout.json").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return report

    def forward(self, spec: StrategySpec) -> TrialRecord:
        """Run the forward window after a passed holdout."""
        ctx = self._ctx
        if not ctx.lockbox.holdout_passed(spec.spec_hash):
            raise LockboxError(f"forward requires a passed holdout: {spec.spec_hash}")
        start, end = self._window_for(Segment.FORWARD)
        base_config = self._base_sim_config()
        features = self._features_for(spec.feature_names())
        lo, hi = self._target_window(start, end)
        targets = target_weights(spec, ctx.cube, features, lo=lo, hi=hi)
        record, _ = self._record_trial(spec=spec, segment=Segment.FORWARD, start=start, end=end,
                                       config=base_config, targets=targets)
        payload = {
            "spec_hash": spec.spec_hash,
            "trial_id": record.trial_id,
            "segment": Segment.FORWARD.value,
            "metrics": dict(record.metrics),
        }
        ctx.reports_root.mkdir(parents=True, exist_ok=True)
        (ctx.reports_root / f"{spec.spec_hash}_forward.json").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return record


def _gate_check_from_json(item: Any) -> Any:
    from src.research.gates import GateCheck

    if not isinstance(item, dict):
        raise ValueError("invalid gate check payload")
    return GateCheck(gate=str(item["gate"]), name=str(item["name"]), value=float(item["value"]),
                     threshold=float(item["threshold"]), passed=bool(item["passed"]),
                     detail=str(item["detail"]))


def _digest_of_canonical(path: Path) -> str:
    import hashlib

    text = path.read_text(encoding="utf-8").strip()
    payload = json.loads(text)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _active_of(returns: TrialReturns, bench: str) -> NDArray[np.float64]:
    net = np.asarray(returns.net, dtype=np.float64)
    base = np.asarray(returns.benchmarks[bench], dtype=np.float64)
    return np.ascontiguousarray(net - base)


def _trial_metrics(net: NDArray[np.float64], benches: Mapping[str, NDArray[np.float64]],
                   sessions_per_year: int, *, turnover: float, cost: float) -> dict[str, float]:
    values = np.asarray(net, dtype=np.float64)
    growth = float(annualized_log_growth(values, sessions_per_year=sessions_per_year))
    metrics: dict[str, float] = {
        "g": growth,
        "cagr": float(np.exp(growth) - 1.0),
        "mdd": float(max_drawdown(values)),
        "turnover": float(turnover),
        "cost": float(cost),
    }
    for name, series in benches.items():
        active = values - np.asarray(series, dtype=np.float64)
        key = "U_EW" if name == "U_EW" else name
        metrics[f"active_g_{key}"] = float(annualized_log_growth(active, sessions_per_year=sessions_per_year))
    return metrics
