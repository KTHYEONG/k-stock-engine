# 04 Q/V/E/F Feature Engine

## Outcome

Produce versioned Point-in-Time Quality, Value, Earnings Momentum, Foreign Flow,
and risk components for every eligible session-security row.

## Dependencies

- Historical eligible universe.
- Silver financial facts, investor flow, daily market, sectors, and corporate actions.

## Planned production scope

- Add `src/features/profitability.py`, `value.py`, `earnings.py`, `foreign_flow.py`, and `risk.py`.
- Add one shared cross-sectional preprocessing pipeline for validity filtering, versioned winsorization, sector-relative ranking, and `[-1, 1]` scaling.
- Materialize a Gold feature dataset with source availability and component-presence flags.

## Invariants

- All divisions are explicit safe divisions; non-economic denominators produce missing components.
- Research returns use adjusted prices; execution prices remain raw.
- Corrected filings affect only decisions at or after the correction availability boundary.
- Earnings facts older than 60 sessions do not contribute.
- Foreign flow is normalized by positive ADTV20, never absolute size.
- No price momentum, technical indicator, or ML-derived feature is included.

## Verification boundary

- Use minimal synthetic cross-sections for ranks, ties, sector isolation, missing components, and correction timing.
- Verify exact Q/V/E/F formulas and no look-ahead at boundary timestamps.
- Command: `uv run pytest tests/unit/features tests/integration/features -q`.
