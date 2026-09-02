# 09 Walk-Forward and Bootstrap Validation

## Outcome

Build one immutable out-of-fold Ledger/equity curve and dependence-preserving
uncertainty estimates for the fixed Champion.

## Dependencies

- Unified Base/Stress backtest artifacts.
- At least ten years of certified data when available.

## Planned production scope

- Add `src/validation/walk_forward.py`, `bootstrap.py`, and `metrics.py`.
- Generate five-year research/one-year OOS folds without overlap leakage.
- Concatenate OOS folds exactly once and calculate metrics from Ledger NAV.
- Implement Moving Block or Stationary Bootstrap with explicit seed, 20-60-session block configuration, and at least 5,000 resamples for promotion runs.
- Build eligible-universe cap- and equal-weight benchmarks under the same costs and availability rules.

## Invariants

- OOS rows cannot influence a prior research window or fixed Champion constants.
- Every OOS session belongs to at most one fold and appears once in the stitched curve.
- Bootstrap sampling preserves contiguous dependence blocks and is deterministic for a seed.
- Missing benchmark or Ledger sessions fail closed.

## Verification boundary

- Test fold endpoints, zero overlap, stitched order, block membership, seed repeatability, and hand-computed metrics.
- Command: `uv run pytest tests/unit/validation/test_walk_forward.py tests/unit/validation/test_bootstrap.py tests/unit/validation/test_metrics.py -q`.
