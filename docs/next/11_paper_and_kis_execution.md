# 11 Paper and KIS Execution

## Outcome

Run promoted Champion targets through paper and KIS broker adapters with signal,
target, fill, and Ledger reconciliation parity.

## Dependencies

- PASS promotion artifact.
- Unified strategy, execution intents, Ledger, and live-readiness gate.

## Planned production scope

- Extend current `src/execution` ports for cancel, reconcile, fill polling, and account snapshots.
- Add `src/live/runner.py`, `reconciliation.py`, and `safety.py`.
- Reuse `src/integrations/kis` transport; keep credentials and HTTP outside domain logic.
- Persist idempotent target, order, fill, and reconciliation artifacts.
- Implement Shadow, Paper, 10%, 25%, 50%, and 100% capital promotion states.

## Invariants

- Paper remains the default; live requires complete readiness evidence and explicit capital-stage authorization.
- Broker-confirmed account state dominates local order state.
- Duplicate intent keys, stale snapshots, unresolved open orders, or Ledger mismatch block submission.
- Actual fills and costs update the same Ledger used by backtest evaluation.
- Safety suspension cannot automatically resume or increase capital.

## Verification boundary

- Contract-test KIS request/response mapping at the network boundary.
- Integration-test retry idempotency, partial fill, cancel, stale snapshot, reconciliation mismatch, and restart recovery.
- Command: `uv run pytest tests/unit/live tests/integration/live tests/integration/execution -q`.
