# System Context and Boundaries

## Topology

```mermaid
flowchart TB
    Sources[KRX / DART / KIS] --> Adapters[Transport Adapters]
    Adapters --> Bronze[Immutable Bronze]
    Bronze --> Silver[Certified PIT Silver]
    Silver --> Features[Feature Engine]
    Features --> Decision[Single Decision Engine]
    Decision --> Target[TargetPortfolio]
    Target --> Portfolio[Portfolio Constructor]
    Portfolio --> Orders[Order Intents]
    Orders --> Backtest[Backtest Broker]
    Orders --> Paper[Paper Broker]
    Orders --> KIS[KIS Broker]
    Backtest --> Fills[Fills]
    Paper --> Fills
    KIS --> Fills
    Fills --> Ledger[Shared Ledger]
    Ledger --> Eval[Performance / Risk / Reconciliation]
```

## Stable boundaries

| Boundary | Input | Output | Must not own |
| --- | --- | --- | --- |
| Integrations | Provider request | Raw response plus retrieval metadata | Domain policy, ranking, portfolio sizing |
| Storage | Immutable dataset and manifest | Validated bounded read | Asset selection or trading logic |
| Data | Raw observations | PIT-certified market snapshot | Alpha weights or order state |
| Features | PIT market snapshot | Versioned feature rows | Portfolio construction |
| Strategy | Snapshot and reconciled portfolio | Target portfolio | Broker behavior or fills |
| Portfolio | Ranked candidates and risk state | Constrained target weights | Alpha discovery or provider I/O |
| Execution | Target positions and broker state | Orders and fills | Feature calculation |
| Ledger | Fills, settlements, actions, marks | Cash, positions, costs, NAV | Forecasts or desired weights |
| Validation | OOS Ledger and experiment artifacts | PASS/FAIL verdict | Strategy mutation |

## Decision contract

The conceptual contract is:

```python
TargetPortfolio = strategy.decide(
    market_snapshot,
    portfolio_state,
)
```

The same strategy object receives historical or live implementations of the
snapshot and portfolio inputs. Environment-specific code ends at adapter and
broker boundaries.

## Active repository mapping

The target design reuses the current active foundations instead of recreating
equivalent abstractions.

| Target responsibility | Current reusable owner |
| --- | --- |
| Domain identity, costs, PIT time, portfolio state | `src/core/` |
| Manifest-validated Parquet I/O | `src/storage/parquet_datasets.py` |
| Order intents, ports, paper broker, submission gate | `src/execution/` |
| Provider transport | `src/integrations/{krx,dart,kis}/` |

New strategy, feature, engine, validation, and live orchestration modules are
introduced only by ordered contracts. Modern code must not import `legacy`.

## Dependency rule

```mermaid
flowchart LR
    Integrations --> Application
    Storage --> Core
    Application --> Core
    Strategy --> Core
    Engine --> Strategy
    Engine --> ExecutionPorts
    Adapters --> ExecutionPorts
    Validation --> Engine
```

Dependencies point toward pure contracts. `core` imports no provider,
strategy, storage, or execution package. Adapters may depend on ports, but
ports never depend on adapters.
