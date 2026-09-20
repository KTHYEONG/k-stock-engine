---
trigger:
  - on_label: ["performance"]
  - on_file_path_regex: "src/.*"
  - on_file_path_glob: ["src/**/*.py"]
priority: 10
---

# Performance & Optimization Directives (Physical Invariants)

> **Performance is an architectural invariant, not an afterthought. Never fake speed by silently truncating date ranges, universe constituents, or numerical precision. Correctness strictly precedes speed. Maximize hardware throughput within four physical boundaries.**

## 1. Bounded Working Memory & Swap Prevention
- **Memory Ceiling:** Peak working memory must remain strictly bounded relative to available physical RAM. Working sets must scale $O(1)$ relative to total stream length through chunking, streaming, or localized views.
- **Zero Virtual Memory Spilling:** Eliminate unnecessary in-memory data duplication. Never allow processes to spill into OS swap space.
- **Deterministic Resource Reclamation:** Heavy memory buffers, file descriptors, and worker pools must be deterministically released when exiting their operational scope.

## 2. Algorithmic Locality & Sub-Quadratic Scaling
- **Zero Linear Scans in Iterations:** Never execute linear scans, collection filtering, or full traversals inside recurrent loops.
- **O(1) Keyed Lookups:** Access to historical or cross-sectional states inside hot iteration paths must be strictly $O(1)$ through pre-indexing, hashing, grouping, or cursor positioning.
- **I/O Locality & Pushdown:** Prohibit repetitive text deserialization in computational hot paths. Leverage binary columnar storage with column pruning and predicate pushdown.

## 3. Separation of Invariant Features and Dynamic State
- **One-Pass Invariant Materialization:** Compute all state-independent features and indicators once upstream across the complete historical timeline.
- **Minimal Hot Paths:** Confine inner sequential iteration strictly to state-dependent transitions. Never recompute static or historical invariants inside simulation loops or across optimization folds.

## 4. Hardware Saturation & Observable Throughput
- **Balanced Parallelism:** Saturate available CPU cores for embarrassingly parallel workloads while ensuring compute granularity heavily amortizes inter-process communication (IPC) overhead.
- **Continuous Progress Telemetry:** Long-running workloads must emit deterministic progress heartbeats. A silent pipeline is indistinguishable from a deadlocked system.