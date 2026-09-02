# 12 Challenger Factors

## Outcome

Evaluate reversal, price momentum, and short-interest hypotheses without
mutating Champion v1 or relaxing its promotion gates.

## Dependencies

- Champion promotion pipeline and append-only experiment registry.
- Additional PIT source evidence required by each hypothesis.

## Planned order

1. Short-term reversal.
2. Six-month residual and 12-minus-1 residual price momentum.
3. Short interest combined with foreign holdings/trading.

Each Challenger adds one versioned feature family and one explicit strategy
variant. It reuses identical universe, folds, portfolio constraints, fill model,
Base/Stress costs, and benchmarks.

## Invariants

- Challenger results cannot overwrite Champion artifacts.
- No hypothesis-specific retuning of unrelated portfolio or execution parameters.
- Missing historical availability or correction lineage blocks that Challenger.
- Promotion requires higher OOS excess and Stress growth without unacceptable MDD or turnover degradation.
- Multiple candidates are corrected for data snooping before selection.

## Verification boundary

- Test feature temporal boundaries, residualization isolation, fixed comparison inputs, and immutable Champion outputs.
- Command: `uv run pytest tests/unit/features/test_challengers.py tests/integration/validation/test_challenger_promotion.py -q`.
