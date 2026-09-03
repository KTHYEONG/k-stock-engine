from __future__ import annotations


def test_build_walk_forward_folds_keeps_oos_disjoint_and_after_research() -> None:
    from datetime import UTC, datetime, timedelta

    from src.validation.walk_forward import build_walk_forward_folds

    start = datetime(2020, 1, 1, tzinfo=UTC)
    sessions = tuple(start + timedelta(days=index) for index in range(8))

    folds = build_walk_forward_folds(sessions, research_sessions=4, oos_sessions=2)

    assert len(folds) == 2
    assert folds[0].research_sessions == sessions[:4]
    assert folds[0].oos_sessions == sessions[4:6]
    assert folds[1].research_sessions == sessions[2:6]
    assert folds[1].oos_sessions == sessions[6:8]
    assert set(folds[0].oos_sessions).isdisjoint(folds[1].oos_sessions)
    assert max(folds[0].research_sessions) < min(folds[0].oos_sessions)
    assert max(folds[1].research_sessions) < min(folds[1].oos_sessions)


def test_stitch_oos_ledger_nav_rejects_missing_or_duplicate_oos_mark() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.engine.backtest import BacktestResult
    from src.engine.fill_model import ExecutionScenario
    from src.core.ledger import LedgerNav
    from src.validation.walk_forward import FoldReplay, build_walk_forward_folds, stitch_oos_ledger_nav

    start = datetime(2020, 1, 1, tzinfo=UTC)
    sessions = tuple(start + timedelta(days=index) for index in range(6))
    fold = build_walk_forward_folds(sessions, research_sessions=4, oos_sessions=2)[0]
    mark = LedgerNav("m1", fold.oos_sessions[0], 100.0, 100.0, 0.0, 0.0)
    result = BacktestResult((), (), (), (mark,), (), ExecutionScenario.BASE)
    replay = FoldReplay(fold, result, "continuous-ledger", "dataset", "universe", "execution:base")

    with pytest.raises(ValueError, match="OOS"):
        stitch_oos_ledger_nav((replay,))


def test_benchmark_target_weights_are_pit_complete_and_sum_to_one() -> None:
    from datetime import UTC, datetime

    import pytest

    from src.validation.walk_forward import (
        BenchmarkConstituent,
        BenchmarkKind,
        EligibleUniverseSnapshot,
        benchmark_target_weights,
    )

    timestamp = datetime(2024, 1, 2, tzinfo=UTC)
    snapshot = EligibleUniverseSnapshot(
        timestamp,
        (
            BenchmarkConstituent("KRX:000001", 10.0, 100.0),
            BenchmarkConstituent("KRX:000002", 20.0, 300.0),
        ),
    )

    assert benchmark_target_weights(snapshot, BenchmarkKind.CAP_WEIGHT) == (
        ("KRX:000001", 0.25),
        ("KRX:000002", 0.75),
    )
    assert benchmark_target_weights(snapshot, BenchmarkKind.EQUAL_WEIGHT) == (
        ("KRX:000001", 0.5),
        ("KRX:000002", 0.5),
    )
    with pytest.raises(ValueError, match="market_cap"):
        EligibleUniverseSnapshot(timestamp, (BenchmarkConstituent("KRX:000003", 10.0, 0.0),))
