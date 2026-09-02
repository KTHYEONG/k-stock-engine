# 08 Unified Event-Driven Backtester

## Outcome

Run the same decision and target-position contracts through a deterministic
historical broker and the shared Ledger to produce net NAV.

## Dependencies

- Risk-constrained target portfolio.
- PIT datasets, corporate actions, costs, SessionCalendar, and shared Ledger.

## Planned production scope

- Add `src/engine/decision.py`, `backtest.py`, and `fill_model.py`.
- Implement the per-session order: settle, actions, fills, marks, Ledger check, decision.
- Add next-open fill pricing, tick/lot rules, spread, square-root participation impact, and Ideal/Base/Stress scenarios.
- Emit immutable fills, rejects, Ledger journal, daily NAV, and capacity diagnostics.

## Invariants

- T-close signals cannot fill before T+1.
- The backtest calls the production strategy decision contract without a backtest branch.
- Raw prices execute; adjusted prices research.
- Costs are effective-dated and sell tax is applied only to sells.
- Unsettled sale proceeds are unavailable until the configured KRX settlement session.
- Any Ledger mismatch, unknown action, price gap, or hard participation breach fails the run.

## Verification boundary

- Test T/T+1 separation, T+2 settlement, split/dividend, partial/unfilled order, tax side, slippage monotonicity, and exact NAV identity.
- Add a minimal multi-session integration replay with hand-calculated results.
- Command: `uv run pytest tests/unit/engine tests/integration/engine -q`.
