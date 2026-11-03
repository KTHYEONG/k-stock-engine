---
paths:
  - "src/**"
---
# Performance & Optimization Directives (Physical Invariants)

> **Performance is an architectural invariant: never fake speed by truncating date ranges, universe constituents, or precision; correctness precedes speed.**

## 1. Memory Discipline
- **Bounded Resource Scaling:** Keep memory safely within host physical RAM, scaling predictably with workload; never accumulate intermediate state unboundedly.

## 2. Locality & Hot Paths
- **I/O Locality & Pushdown:** No repeated text parsing or deserialization in hot paths; prefer binary or columnar storage with column pruning and predicate pushdown.

## 3. Invariant Features vs. Dynamic State
- **One-Pass Invariant Materialization & Minimal Hot Paths:** Compute state-independent features once upstream (chunked chronologically when data exceeds the working set); keep inner loops to state-dependent transitions, never recomputing historical invariants inside simulation loops or optimization folds.

## 4. Parallelism & Telemetry
- **Balanced Parallelism:** After a parallelizable bottleneck is verified, saturate cores with coarse granularity that amortizes IPC; skip multiprocessing for lightweight work.
- **Continuous Progress Telemetry:** Long-running workloads emit deterministic progress heartbeats; a silent pipeline looks like a deadlock.
