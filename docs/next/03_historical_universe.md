# 03 Historical Eligible Universe

## Outcome

Build a session-specific KOSPI/KOSDAQ common-stock universe without
survivorship bias and with explicit exclusion reasons.

## Dependencies

- Research-certified Silver security master, daily market, status events, and calendar.

## Planned production scope

- Add `src/strategy/universe.py` with an immutable `UniversePolicy` and decision result.
- Resolve historical listing, delisting, share class, market, sector, financial-sector status, and trading status.
- Enforce 252-session listing age and 60-session median trading value.
- Persist eligible membership and reason-coded exclusions as a Gold dataset.

## Invariants

- Membership uses facts available at the requested decision time only.
- Current listings cannot backfill historical sessions.
- ETF, ETN, REIT, SPAC, preferred, financial, managed, suspended, and liquidation-trading rows are excluded.
- Missing sector, status, age, or liquidity evidence is an exclusion, not an inferred pass.
- The 2 billion KRW threshold is a fixed baseline and never tuned in this phase.

## Verification boundary

- Test delisted retention in past sessions, IPO age 251/252, liquidity 59/60 observations, and each asset/status exclusion.
- Test deterministic exclusion reasons and no duplicate `(session, instrument_id)` membership.
- Command: `uv run pytest tests/unit/strategy/test_universe.py tests/integration/strategy/test_historical_universe.py -q`.
