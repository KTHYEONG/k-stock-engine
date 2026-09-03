"""Walk-forward evaluation runner."""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.ledger import LedgerNav
from src.engine.fill_model import ExecutionScenario
from src.validation.bootstrap import BootstrapConfig, BootstrapDistribution, bootstrap_annualized_log_growth
from src.validation.metrics import LedgerMetrics, calculate_ledger_metrics
from src.validation.walk_forward import FoldReplay, build_walk_forward_folds, stitch_oos_ledger_nav


@dataclass(frozen=True, slots=True)
class WalkForwardValidationArtifact:
    champion_base_nav: tuple[LedgerNav, ...]
    champion_stress_nav: tuple[LedgerNav, ...]
    cap_weight_base_nav: tuple[LedgerNav, ...]
    equal_weight_base_nav: tuple[LedgerNav, ...]
    champion_base_metrics: LedgerMetrics
    champion_stress_metrics: LedgerMetrics
    cap_weight_base_metrics: LedgerMetrics
    equal_weight_base_metrics: LedgerMetrics
    bootstrap_distribution: BootstrapDistribution
    bootstrap_config: BootstrapConfig


def _check_provenance_parity(
    champion_base: tuple[FoldReplay, ...],
    champion_stress: tuple[FoldReplay, ...],
    cap_weight_base: tuple[FoldReplay, ...],
    equal_weight_base: tuple[FoldReplay, ...],
) -> None:
    for name, seq in (
        ("champion_base", champion_base),
        ("champion_stress", champion_stress),
        ("cap_weight_base", cap_weight_base),
        ("equal_weight_base", equal_weight_base),
    ):
        if not isinstance(seq, tuple):
            raise ValueError(f"{name} must be tuple")
        if len(seq) == 0:
            raise ValueError(f"{name} must be non-empty")
        for r in seq:
            if not isinstance(r, FoldReplay):
                raise ValueError(f"{name} must contain FoldReplay")

    # Same complete fold-id set
    def fold_ids(seq: tuple[FoldReplay, ...]) -> set[int]:
        return {r.fold.fold_id for r in seq}

    base_ids = fold_ids(champion_base)
    stress_ids = fold_ids(champion_stress)
    cap_ids = fold_ids(cap_weight_base)
    eq_ids = fold_ids(equal_weight_base)
    if not (base_ids == stress_ids == cap_ids == eq_ids):
        raise ValueError("fold coverage mismatch across scenarios: same complete fold-id set required")
    # Check contiguous (via stitch later but also here)
    sorted_ids = sorted(base_ids)
    for i in range(1, len(sorted_ids)):
        if sorted_ids[i] != sorted_ids[i - 1] + 1:
            raise ValueError("fold_ids must be contiguous")

    # Build map fold_id -> replay per scenario
    def map_by_fold(seq: tuple[FoldReplay, ...]) -> dict[int, FoldReplay]:
        m: dict[int, FoldReplay] = {}
        for r in seq:
            if r.fold.fold_id in m:
                raise ValueError(f"duplicate fold_id {r.fold.fold_id} in scenario")
            m[r.fold.fold_id] = r
        return m

    base_map = map_by_fold(champion_base)
    stress_map = map_by_fold(champion_stress)
    cap_map = map_by_fold(cap_weight_base)
    eq_map = map_by_fold(equal_weight_base)

    for fid in sorted_ids:
        rb = base_map[fid]
        rs = stress_map[fid]
        rc = cap_map[fid]
        re = eq_map[fid]
        # Session coverage: OOS sessions must be identical across scenarios for same fold
        if rb.fold.oos_sessions != rs.fold.oos_sessions or rb.fold.oos_sessions != rc.fold.oos_sessions or rb.fold.oos_sessions != re.fold.oos_sessions:
            raise ValueError(f"fold {fid} OOS session coverage mismatch")
        # For each fold, dataset_hash, universe_hash, execution_contract_hash must match across all four runs
        for hname in ("dataset_hash", "universe_hash", "execution_contract_hash"):
            vals = [getattr(rb, hname), getattr(rs, hname), getattr(rc, hname), getattr(re, hname)]
            if len(set(vals)) != 1:
                raise ValueError(f"{hname} mismatch at fold {fid}")
        # Scenario checks
        # champion_base, cap, equal must be BASE; stress must be STRESS
        if getattr(rb.result, "scenario") is not ExecutionScenario.BASE:  # noqa: B009
            raise ValueError(f"champion_base fold {fid} must report BASE, got {getattr(rb.result, 'scenario', None)}")
        if getattr(rc.result, "scenario") is not ExecutionScenario.BASE:  # noqa: B009
            raise ValueError(f"cap_weight_base fold {fid} must report BASE, got {getattr(rc.result, 'scenario', None)}")
        if getattr(re.result, "scenario") is not ExecutionScenario.BASE:  # noqa: B009
            raise ValueError(f"equal_weight_base fold {fid} must report BASE, got {getattr(re.result, 'scenario', None)}")
        if getattr(rs.result, "scenario") is not ExecutionScenario.STRESS:  # noqa: B009
            raise ValueError(f"champion_stress fold {fid} must report STRESS, got {getattr(rs.result, 'scenario', None)}")
        # Reject IDEAL explicitly (already covered but ensure)
        for r in (rb, rs, rc, re):
            if getattr(r.result, "scenario") is ExecutionScenario.IDEAL:  # noqa: B009
                raise ValueError(f"IDEAL scenario not allowed at fold {fid}")


def _validate_fold_schedules(replay_sets: tuple[tuple[FoldReplay, ...], ...]) -> None:
    for replays in replay_sets:
        for replay in replays:
            build_walk_forward_folds(
                replay.fold.research_sessions + replay.fold.oos_sessions,
                research_sessions=len(replay.fold.research_sessions),
                oos_sessions=len(replay.fold.oos_sessions),
            )


def evaluate_walk_forward(
    *,
    champion_base: tuple[FoldReplay, ...],
    champion_stress: tuple[FoldReplay, ...],
    cap_weight_base: tuple[FoldReplay, ...],
    equal_weight_base: tuple[FoldReplay, ...],
    bootstrap_config: BootstrapConfig,
) -> WalkForwardValidationArtifact:
    if not isinstance(bootstrap_config, BootstrapConfig):
        raise ValueError("bootstrap_config must be BootstrapConfig")

    _validate_fold_schedules((champion_base, champion_stress, cap_weight_base, equal_weight_base))
    # First verify provenance parity then scenarios
    _check_provenance_parity(champion_base, champion_stress, cap_weight_base, equal_weight_base)

    # Validate and stitch each scenario (also checks ledger_run_hash continuity and OOS marks)
    champion_base_nav = stitch_oos_ledger_nav(champion_base)
    champion_stress_nav = stitch_oos_ledger_nav(champion_stress)
    cap_weight_base_nav = stitch_oos_ledger_nav(cap_weight_base)
    equal_weight_base_nav = stitch_oos_ledger_nav(equal_weight_base)

    # Calculate metrics
    champion_base_metrics = calculate_ledger_metrics(champion_base_nav)
    champion_stress_metrics = calculate_ledger_metrics(champion_stress_nav)
    cap_weight_base_metrics = calculate_ledger_metrics(cap_weight_base_nav)
    equal_weight_base_metrics = calculate_ledger_metrics(equal_weight_base_nav)

    # Bootstrap distribution derived only from Champion Base stitched Ledger returns
    # Compute log returns from champion_base_nav
    log_returns: list[float] = []
    for i in range(1, len(champion_base_nav)):
        prev = float(champion_base_nav[i - 1].nav)
        cur = float(champion_base_nav[i].nav)
        if prev <= 0 or cur <= 0:
            raise ValueError("nav must be positive for bootstrap")
        r = math.log(cur / prev)
        if not math.isfinite(r):
            raise ValueError("log return must be finite")
        log_returns.append(r)
    bootstrap_distribution = bootstrap_annualized_log_growth(tuple(log_returns), bootstrap_config)

    return WalkForwardValidationArtifact(
        champion_base_nav=champion_base_nav,
        champion_stress_nav=champion_stress_nav,
        cap_weight_base_nav=cap_weight_base_nav,
        equal_weight_base_nav=equal_weight_base_nav,
        champion_base_metrics=champion_base_metrics,
        champion_stress_metrics=champion_stress_metrics,
        cap_weight_base_metrics=cap_weight_base_metrics,
        equal_weight_base_metrics=equal_weight_base_metrics,
        bootstrap_distribution=bootstrap_distribution,
        bootstrap_config=bootstrap_config,
    )
