# Point-in-Time Data Architecture

## Storage tiers

```mermaid
flowchart LR
    Provider[Provider Payload] --> Bronze[Bronze: immutable source]
    Bronze --> Silver[Silver: normalized PIT tables]
    Silver --> Gold[Gold: features / snapshots / benchmarks]
    Gold --> Artifacts[Artifacts: experiments / promotion]
```

| Tier | Mutability | Contents | Eligibility |
| --- | --- | --- | --- |
| Bronze | Append-only | Raw KRX, DART, KIS payloads and retrieval metadata | Never consumed directly by strategy |
| Silver | Immutable dataset ID | Normalized certified tables | Snapshot and feature inputs |
| Gold | Immutable dataset ID | Features, universes, benchmarks, evaluation panels | Research/backtest only at required certification |
| Artifacts | Append-only | Parameters, folds, ledgers, metrics, verdicts | Audit and promotion evidence |

Every dataset directory carries a manifest and content hashes. An unknown
schema, incomplete coverage, hash mismatch, or insufficient certification
fails before table materialization.

## Time semantics

| Field | Meaning |
| --- | --- |
| `event_time` | Economic or market event occurrence |
| `published_at` | Provider publication time |
| `available_at` | Earliest instant the engine is allowed to consume the fact |
| `ingested_at` | Local durable-write completion time |
| `decision_time` | Frozen strategy decision instant |
| `execution_time` | Earliest permitted order/fill instant |

Required ordering is:

$$
event\_time \le published\_at \le available\_at \le decision\_time < execution\_time
$$

When historical intraday DART availability cannot be proven, a filing becomes
available on the next KRX session. No timestamp is inferred to improve coverage.

## Silver schemas

### Security master

| Key | Required fields |
| --- | --- |
| `(instrument_id, valid_from)` | ticker, company_id, market, sector, listing_date, delisting_date, share_class, status, valid_to, available_at |

### Daily market

| Key | Required fields |
| --- | --- |
| `(session, instrument_id)` | raw OHLC, volume, trading_value, market_cap, shares_outstanding, available_at |

### Investor flow

| Key | Required fields |
| --- | --- |
| `(session, instrument_id)` | foreign buy/sell/net value, institution net value, retail net value, available_at |

### Financial facts

| Key | Required fields |
| --- | --- |
| `(company_id, fiscal_period, filing_id, fact)` | published_at, available_at, value, unit, consolidated flag, restatement identity |

Required normalized facts are sales, gross profit, operating profit, net
income, assets, equity, cash, debt, operating cash flow, and capex.

### Corporate actions and disclosures

| Table | Key | Required fields |
| --- | --- | --- |
| Corporate actions | `(instrument_id, effective_date, action_id)` | type, factor, cash amount, source, available_at |
| Disclosures | `(company_id, filing_id)` | filing type, published_at, available_at, correction lineage |

## Price separation

- Execution uses raw open, high, low, and close prices.
- Research uses corporate-action-adjusted prices or a total-return index.
- An adjusted price is never accepted as an execution fill price.
- An unknown or unresolved action in the requested range blocks certification.

## Snapshot invariant

A snapshot at decision time \(d\) contains only rows satisfying
`available_at <= d`. Universe membership is reconstructed for each session;
the current listed-security snapshot is never used for historical membership.

## Required quality gates

Certification fails when any of the following is nonzero:

- duplicate primary keys;
- rows available after their consuming decision;
- unknown instrument mappings;
- missing sessions inside declared coverage;
- unresolved corporate actions;
- survivorship violations;
- content or lineage hash mismatches.
