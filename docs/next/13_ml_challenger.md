# 13 ML Challenger

## Outcome

Determine whether ElasticNet, then LightGBM, adds robust net growth beyond the
simple Champion under an identical economic and execution contract.

## Dependencies

- Promoted simple Champion and completed non-ML Challengers.
- PIT feature panel, purged folds, experiment registry, and promotion gates.

## Planned production scope

- Begin with ElasticNet; evaluate LightGBM only if ElasticNet establishes incremental value.
- Inputs are Q/V/E/F and risk components already available at decision time.
- Target is future 20-session sector-relative return net of execution costs.
- Fit preprocessing and models inside each training fold; purge overlapping labels and embargo boundaries.
- Convert model output through the unchanged portfolio, execution, cost, and Ledger path.

## Invariants

- No scaler, feature selection, label, or model state crosses from validation/OOS into training.
- Champion and ML use identical universe, folds, costs, capacity, and target construction.
- Model search space and trial budget are fixed before OOS evaluation.
- Higher prediction metrics alone cannot promote ML.
- ML is rejected unless OOS excess growth and Stress growth improve with no unacceptable MDD or turnover deterioration.

## Verification boundary

- Test purge/embargo boundaries, train-only fitting, deterministic seeds, model artifact hashes, and identical non-model inputs.
- Test promotion using Ledger metrics rather than IC or prediction loss.
- Command: `uv run pytest tests/unit/ml tests/integration/ml -q`.
