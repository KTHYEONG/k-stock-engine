"""Application entry point for the unified event-driven backtester."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.engine.backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestSession,
    EventBacktester,
)
from src.engine.decision import StrategyDecisionPort

if TYPE_CHECKING:
    from src.validation.robustness import PromotionEvidence, PromotionVerdict

__all__ = ["run_backtest", "run_promotion_evaluation", "run_walk_forward_validation"]


def run_backtest(
    config: BacktestConfig,
    sessions: tuple[BacktestSession, ...],
    strategy: StrategyDecisionPort,
) -> BacktestResult:
    """Run a configured historical replay through the shared engine."""
    return EventBacktester(config).run(sessions, strategy)


def run_walk_forward_validation(
    *,
    champion_base: tuple[object, ...],
    champion_stress: tuple[object, ...],
    cap_weight_base: tuple[object, ...],
    equal_weight_base: tuple[object, ...],
    bootstrap_config: object,
) -> object:
    """Delegate walk-forward validation to the validation runner.

    Keeps engine wiring independent of validation internals while
    satisfying the wiring anchor ``return evaluate_walk_forward(``.
    """
    from src.validation.runner import evaluate_walk_forward

    return evaluate_walk_forward(
        champion_base=champion_base,  # type: ignore[arg-type]
        champion_stress=champion_stress,  # type: ignore[arg-type]
        cap_weight_base=cap_weight_base,  # type: ignore[arg-type]
        equal_weight_base=equal_weight_base,  # type: ignore[arg-type]
        bootstrap_config=bootstrap_config,  # type: ignore[arg-type]
    )


def run_promotion_evaluation(
    *,
    evidence: PromotionEvidence,
    registry_path: Path,
) -> PromotionVerdict:
    """Evaluate promotion gates then append the immutable verdict."""
    from src.validation.robustness import append_promotion_verdict, evaluate_promotion

    verdict = evaluate_promotion(evidence)
    append_promotion_verdict(registry_path, verdict)
    return verdict
