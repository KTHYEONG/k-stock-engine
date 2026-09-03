from __future__ import annotations


def test_evaluate_promotion_accepts_exact_threshold_boundaries() -> None:
    from datetime import UTC, datetime

    from src.validation.bootstrap import BootstrapConfig, BootstrapMethod
    from src.engine.fill_model import ExecutionScenario
    from src.validation.metrics import LedgerMetrics
    from src.validation.robustness import (
        FactorAblationEvidence, FactorName, IntegrityCheck, IntegrityEvidence,
        ParameterProbe, PromotionEvidence, PromotionMetricSnapshot,
        PromotionRunMetadata, PromotionStatus, YearlyPerformance, evaluate_promotion,
    )

    digest = 'a' * 64
    config = BootstrapConfig(BootstrapMethod.MOVING_BLOCK, 5_000, 20, 7, promotion_run=True)
    base = LedgerMetrics(0.10, 0.15, 0.125, 0.25, 0.5, None)
    benchmark = LedgerMetrics(0.01, 0.12, 0.10, 0.20, 0.6, None)
    stress = LedgerMetrics(0.01, 0.001, 0.10, 0.20, 0.1, None)
    yearly = tuple(
        YearlyPerformance(2020 + index, 0.10 if index < 7 else -0.01, 0.05 if index < 6 else (0.15 if index == 6 else 0.0), 0.05 if index < 6 else (0.15 if index == 6 else 0.0))
        for index in range(10)
    )
    scenarios = (ExecutionScenario.BASE, ExecutionScenario.STRESS, ExecutionScenario.BASE, ExecutionScenario.BASE)
    snapshot = PromotionMetricSnapshot(base, stress, benchmark, benchmark, yearly, config, digest, (digest,), scenarios)
    metadata = PromotionRunMetadata('run-boundary', datetime(2026, 1, 1, tzinfo=UTC), 'b' * 40, ('dataset-a',), 'frozen champion', (('portfolio_size', '20'), ('rebalance_sessions', '5')))
    integrity = tuple(IntegrityEvidence(check, True, digest) for check in IntegrityCheck)
    probes = tuple(ParameterProbe(n, cadence, snapshot) for n in (15, 20, 25) for cadence in (4, 5, 10))
    ablations = tuple(FactorAblationEvidence(factor, PromotionMetricSnapshot(base, stress, benchmark, benchmark, yearly, config, f'{index + 1:x}' * 64, (f'{index + 1:x}' * 64,), scenarios)) for index, factor in enumerate(FactorName))

    verdict = evaluate_promotion(PromotionEvidence(metadata, snapshot, integrity, probes, ablations))

    assert verdict.status is PromotionStatus.PASS
    assert all(result.passed for result in verdict.gate_results)


def test_evaluate_promotion_marks_missing_ablation_incomplete() -> None:
    from datetime import UTC, datetime

    from src.validation.robustness import (
        IntegrityCheck, IntegrityEvidence, PromotionEvidence, PromotionGate,
        PromotionRunMetadata, PromotionStatus, evaluate_promotion,
    )

    evidence = PromotionEvidence(
        PromotionRunMetadata('run-missing', datetime(2026, 1, 1, tzinfo=UTC), 'c' * 40, ('dataset-a',), 'test', (('portfolio_size', '20'),)),
        None,
        tuple(IntegrityEvidence(check, True, 'd' * 64) for check in IntegrityCheck),
        (),
        (),
    )

    verdict = evaluate_promotion(evidence)

    assert verdict.status is PromotionStatus.INCOMPLETE
    assert next(item for item in verdict.gate_results if item.gate is PromotionGate.FACTOR_ABLATION).passed is False


def test_evaluate_promotion_rejects_integrity_failure_before_performance() -> None:
    from datetime import UTC, datetime

    from src.validation.robustness import (
        IntegrityCheck, IntegrityEvidence, PromotionEvidence, PromotionGate,
        PromotionRunMetadata, PromotionStatus, evaluate_promotion,
    )

    metadata = PromotionRunMetadata('run-integrity', datetime(2026, 1, 1, tzinfo=UTC), 'e' * 40, ('dataset-a',), 'test', (('portfolio_size', '20'),))
    evidence = PromotionEvidence(metadata, None, (IntegrityEvidence(IntegrityCheck.LOOK_AHEAD, False, 'f' * 64),), (), ())

    verdict = evaluate_promotion(evidence)

    assert verdict.status is PromotionStatus.FAIL
    assert next(item for item in verdict.gate_results if item.gate is PromotionGate.DATA_INTEGRITY).passed is False


def test_assess_alpha_concentration_rejects_single_year_dominance() -> None:
    from src.validation.robustness import YearlyPerformance, assess_alpha_concentration

    yearly = (
        YearlyPerformance(2020, 0.20, 0.00, 0.00),
        YearlyPerformance(2021, 0.01, 0.00, 0.00),
    )

    result = assess_alpha_concentration(yearly)

    assert result.passed is False
    assert any('50%' in reason for reason in result.reasons)
