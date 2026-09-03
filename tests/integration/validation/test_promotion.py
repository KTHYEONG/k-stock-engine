from __future__ import annotations


def test_append_promotion_verdict_is_deterministic_and_append_only(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.validation.robustness import (
        IntegrityCheck, IntegrityEvidence, PromotionEvidence,
        PromotionRunMetadata, append_promotion_verdict, evaluate_promotion,
    )

    evidence = PromotionEvidence(
        PromotionRunMetadata('run-registry', datetime(2026, 1, 1, tzinfo=UTC), '1' * 40, ('dataset-a',), 'test', (('portfolio_size', '20'),)),
        None,
        (IntegrityEvidence(IntegrityCheck.LOOK_AHEAD, False, '2' * 64),),
        (),
        (),
    )
    verdict = evaluate_promotion(evidence)
    registry = tmp_path / 'data' / 'artifacts' / 'promotion_registry.jsonl'

    first = verdict.to_canonical_json()
    assert first == verdict.to_canonical_json()
    append_promotion_verdict(registry, verdict)
    original = registry.read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='run_id'):
        append_promotion_verdict(registry, verdict)

    assert registry.read_text(encoding='utf-8') == original
    assert original == first + '\n'
