# 07 Risk-Constrained Portfolio

## Outcome

Convert the selected set into long-only inverse-volatility targets subject to
security, sector, capacity, and market-exposure constraints.

## Dependencies

- Twenty-stock selection.
- PIT 60-session security volatility, ADTV20, market volatility, sectors, and reconciled NAV.

## Planned production scope

- Add `src/strategy/portfolio.py` for inverse-volatility weights and deterministic cap redistribution.
- Add market-volatility scaling at a fixed 15% target without leverage.
- Produce `Allocation`/target-position intents compatible with current execution contracts.
- Report binding constraints and residual cash.

## Invariants

- Security weights are nonnegative and at most 7.5%; sector sums are at most 25%.
- Gross exposure is at most `min(1, 0.15 / market_volatility)`.
- Target participation is at most 0.25% ADTV20 and hard-rejected above 0.50%.
- Invalid volatility, price, ADTV, sector, or NAV yields exclusion or `NO_TRADE`.
- Constraint projection never silently renormalizes above exposure.

## Verification boundary

- Test equal volatility, cap binding/redistribution, sector cap, exposure at 10/15/30% volatility, capacity, and residual cash.
- Add property tests for nonnegative finite weights and all aggregate caps.
- Command: `uv run pytest tests/unit/strategy/test_portfolio.py -q`.
