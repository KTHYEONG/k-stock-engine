# 06 Twenty-Stock Selection and Hysteresis

## Outcome

Turn Champion ranks and current reconciled holdings into a deterministic
20-security target set with entry rank 20 and retention rank 40.

## Dependencies

- Fixed Champion scores.
- Reconciled `PortfolioSnapshot` from the shared Ledger/broker state.

## Planned production scope

- Add `src/strategy/selection.py` with immutable selection result and reason codes.
- Retain eligible current positions through rank 40, then fill from new entries through rank 20.
- Emit explicit exits for holdings that lose eligibility or exceed retention rank.
- Keep unfillable target slots as cash.

## Invariants

- At most 20 selected instruments; no duplicate identity.
- Current holdings never bypass the eligible-universe gate.
- Hysteresis depends only on current reconciled holdings, not stale local orders.
- Input row order cannot change the selected set.

## Verification boundary

- Test ranks 20/21 and 40/41, survivor priority, forced eligibility exit, fewer-than-20 coverage, and deterministic ties.
- Command: `uv run pytest tests/unit/strategy/test_selection.py -q`.
