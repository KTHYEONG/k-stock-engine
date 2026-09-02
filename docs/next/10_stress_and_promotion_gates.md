# 10 Stress, Ablation, and Promotion Gates

## Outcome

Produce a deterministic PASS/FAIL artifact covering data integrity, OOS
performance, stability, costs, parameter neighborhoods, and factor ablation.

## Dependencies

- Walk-forward and bootstrap artifacts for Champion Base and Stress scenarios.

## Planned production scope

- Add `src/validation/robustness.py` and a versioned promotion verdict schema.
- Evaluate the architecture-defined gates without changing strategy parameters.
- Run N `{15,20,25}`, rebalance `{4,5,10}`, and one-at-a-time Q/V/E/F ablations.
- Register every run with git commit, dataset IDs, hypothesis, parameters, metrics, and decision.

## Invariants

- Any data-integrity failure makes the overall verdict FAIL regardless of performance.
- Ideal costs cannot contribute to promotion.
- Missing scenario, benchmark, fold, or ablation evidence makes the verdict incomplete and non-promotable.
- Failed experiments remain append-only in the registry.
- A PASS artifact contains the exact hashes of every input artifact.

## Verification boundary

- Test every threshold boundary, mandatory-evidence omission, one-year alpha concentration, stability ratio, and deterministic verdict serialization.
- Command: `uv run pytest tests/unit/validation/test_robustness.py tests/integration/validation/test_promotion.py -q`.
