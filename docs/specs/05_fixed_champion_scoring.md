# 05 Fixed Champion Scoring

## Outcome

Convert eligible Q/V/E/F feature rows into a deterministic Champion v1 rank.

## Dependencies

- Gold Q/V/E/F feature dataset.

## Planned production scope

- Add `src/strategy/scoring.py` with versioned `ChampionScorePolicy`.
- Require all four factor scores and combine them at fixed 25% weights.
- Apply a deterministic total ordering using canonical instrument identity as the final tie-breaker.
- Persist scores, ranks, eligibility, and reason codes.

## Invariants

- Q, V, E, and F weights are exactly equal and are not runtime search parameters.
- Missing any factor excludes the security from Champion scoring.
- Input rows after decision time or from another strategy version fail closed.
- Scoring never reads broker state or execution prices.

## Verification boundary

- Test exact weighted scores, all four missing-factor cases, stable ties, and input order invariance.
- Test schema/version mismatch and temporal rejection.
- Command: `uv run pytest tests/unit/strategy/test_scoring.py tests/integration/strategy/test_champion_scores.py -q`.
