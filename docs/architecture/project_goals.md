# Project Goals and Global Invariants

## Mission

Build a long-only Korean equity engine that maximizes robust out-of-sample
geometric growth for small capital while preserving capital through explicit
risk, capacity, cost, and data-integrity constraints.

The objective is not a preselected CAGR. A fixed return target would turn
research into parameter fitting against the desired answer.

## Objective

For net daily portfolio returns \(r_t^{net}\), primary growth is:

$$
g = \frac{252}{N}\sum_{t=1}^{N}\log(1+r_t^{net})
$$

where:

$$
r_t^{net}=r_t^{gross}-commission_t-tax_t-spread_t-slippage_t-impact_t
$$

All promotion decisions use Ledger-derived net returns. Factor IC, prediction
accuracy, and gross strategy returns are diagnostics only.

## Success constraints

| Dimension | Required outcome |
| --- | --- |
| Growth | Positive robust OOS geometric growth after all modeled costs |
| Relative | Outperform cap-weighted and equal-weight eligible-universe benchmarks |
| Drawdown | OOS MDD at or below the precommitted promotion boundary |
| Capacity | Order participation stays within the portfolio contract |
| Stress | Net growth remains positive under the stress cost model |
| Reproducibility | Dataset, code, parameters, and experiment decision are immutable and traceable |

## Global invariants

1. Point-in-Time: every consumed fact has an explicit real-world availability boundary.
2. Single Decision Engine: backtest, paper, and live use the same strategy decision contract.
3. Net PnL First: only shared-Ledger NAV is performance truth.
4. Minimal Degrees of Freedom: Champion v1 constants are fixed before OOS evaluation.
5. Fail Closed: missing provenance, unavailable facts, reconciliation failure, or ledger imbalance yields `NO_TRADE` or certification failure.
6. Long Only: no short positions and no leverage in Champion v1.
7. Execution Separation: research asks whether alpha exists; strategy asks which target portfolio is desired; execution reports what was actually obtained.

## Champion v1 scope

| Included | Excluded until promoted as a Challenger |
| --- | --- |
| KOSPI and KOSDAQ common stocks | ETF, ETN, REIT, SPAC, preferred shares |
| Quality, Value, Earnings Momentum, Foreign Flow | Price momentum, reversal, short interest |
| Inverse-volatility sizing and market-volatility scaling | Markowitz, drawdown timing, regime classifiers |
| Event-driven backtest, paper, and KIS adapters | ML, RL, deep learning, optimizer families |

## Non-goals

- Optimizing factor weights, portfolio size, or rebalance frequency for peak backtest CAGR.
- Predicting market direction or individual prices.
- Treating historical data repeatedly used in research as a pristine final holdout.
- Promoting a complex model that does not beat the fixed Champion under identical data, execution, and cost contracts.
