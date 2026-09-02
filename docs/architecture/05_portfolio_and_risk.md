# Portfolio Construction and Risk

## Selection state machine

Champion v1 targets 20 holdings and uses hysteresis:

1. Retain existing eligible holdings while rank is at most 40.
2. Remove holdings outside rank 40 or the eligible universe.
3. Fill vacancies from the highest-ranked non-held securities whose rank is at most 20.
4. If fewer than 20 eligible securities remain, hold the residual allocation as cash.

This rule reduces turnover caused by small cross-sectional rank changes.

## Raw weights

For eligible selected security \(i\):

$$
\tilde{w_i}=\frac{1}{\sigma_{i,60}}
$$

Volatility is annualized from PIT-adjusted research returns. Nonpositive,
nonfinite, or insufficient-history volatility makes the security ineligible.

## Constraint projection

| Constraint | Champion v1 bound |
| --- | --- |
| Long-only | \(w_i \ge 0\) |
| Leverage | none; gross exposure at most 100% |
| Single security | at most 7.5% NAV |
| Sector | at most 25% NAV |
| Target order participation | at most 0.25% of ADTV20 |
| Hard order participation | at most 0.50% of ADTV20 |

Projection is deterministic. Binding caps are redistributed only among
uncapped eligible holdings; any remainder stays in cash. Constraint failure
never produces an unconstrained portfolio.

## Market exposure

Let \(\sigma_{mkt,t}\) be annualized realized market volatility and the fixed
target be 15%:

$$
E_t=\min\left(1,\frac{0.15}{\sigma_{mkt,t}}\right)
$$

Final risky weights sum to at most \(E_t\). Missing, nonpositive, or nonfinite
market volatility yields `NO_TRADE`; leverage is never used when volatility is
below target.

## Target portfolio output

| Field | Constraint |
| --- | --- |
| `decision_time` | After source availability and before execution |
| `strategy_id` | Immutable Champion version |
| `account_snapshot_id` | Broker-reconciled state identity |
| `instrument_id` | Canonical KRX identity |
| `target_weight` | Finite, nonnegative, constraint-compliant |
| `target_value` | Derived from reconciled NAV and exposure |
| `reason` | Rank, survivor/new-entry status, and binding constraints |

Target positions are desired end states, not signed order deltas. Execution
derives buys and sells against the reconciled account snapshot.

## Excluded overlays

Drawdown-based exits, stop-losses, take-profit rules, market-direction regimes,
and per-stock Kelly sizing are outside Champion v1. Drawdown may trigger human
review or strategy suspension, but it does not mutate portfolio weights.
