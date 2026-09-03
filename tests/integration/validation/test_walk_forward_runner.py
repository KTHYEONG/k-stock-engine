from __future__ import annotations


def test_evaluate_walk_forward_rejects_benchmark_provenance_mismatch() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.ledger import LedgerNav
    from src.engine.backtest import BacktestResult
    from src.engine.fill_model import ExecutionScenario
    from src.validation.bootstrap import BootstrapConfig, BootstrapMethod
    from src.validation.runner import evaluate_walk_forward
    from src.validation.walk_forward import FoldReplay, build_walk_forward_folds

    start = datetime(2020, 1, 1, tzinfo=UTC)
    sessions = tuple(start + timedelta(days=index) for index in range(6))
    fold = build_walk_forward_folds(sessions, research_sessions=4, oos_sessions=2)[0]
    nav = tuple(
        LedgerNav(f"mark-{index}", when, 100.0 + index, 100.0 + index, 0.0, 0.0)
        for index, when in enumerate(fold.oos_sessions)
    )
    result = BacktestResult((), (), (), nav, (), ExecutionScenario.BASE)
    champion = FoldReplay(fold, result, "champion-ledger", "dataset", "universe-a", "execution:base")
    mismatched_benchmark = FoldReplay(
        fold,
        result,
        "benchmark-ledger",
        "dataset",
        "universe-b",
        "execution:base",
    )
    config = BootstrapConfig(BootstrapMethod.MOVING_BLOCK, resamples=1, block_length_sessions=20, seed=1)

    with pytest.raises(ValueError, match="universe_hash"):
        evaluate_walk_forward(
            champion_base=(champion,),
            champion_stress=(champion,),
            cap_weight_base=(mismatched_benchmark,),
            equal_weight_base=(champion,),
            bootstrap_config=config,
        )
