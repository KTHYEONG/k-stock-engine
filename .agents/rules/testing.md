---
paths:
  - "tests/**"
  - "tools/**"
---
# Testing Directives & Quality Standards

> **Verify observable contracts and invariants, not internals; keep tests fast, deterministic, and hermetic.**

## 1. Invariant-Driven Development
- **Pragmatic Invariant Verification:** Localized work gets targeted invariant guard tests via the project toolchain; complex interfaces and pipelines first get contracts and invariants in `docs/specs/*_spec.md`.
- **Observable Behavior & Invariants:** Assert return contracts, state mutations, error conditions, and conservation laws (balance reconciliation, allocation parity, idempotent outputs), not internals or mock call-counts.
- **In-Memory & Minimal Inputs:** Use the smallest deterministic synthetic data; never load multi-year disk datasets in unit tests.

## 2. Boundaries & Causality
- **Temporal & Numerical Boundaries:** Test zeros, NaNs, empty collections, missing keys, boundary timestamps, timezone transitions, and explicit `rtol`/`atol`, not just happy paths.
- **Causal Invariance:** Prove calculations never touch future time steps, uninitialized state, or unreleased data.

## 3. Latency, Isolation & Fixtures
- **Latency Budgets:** Fast unit tests stay in-memory; heavy end-to-end, retraining, or multi-year runs carry explicit boundary tags, isolated from the default run.
- **Process-Isolated Temp Workspaces:** Never wipe shared temp directories globally; each runner invocation gets its own partitioned workspace.
- **Hermetic Environment Invariant:** Unit tests are 100% hermetic: no live shell credentials, ambient tokens, or production env vars; declare isolated fixtures.
- **Tooling Artifact Awareness:** If tests pass alone but fail under coverage or profiling, suspect tracer overhead, timeouts, or fork interference first.
- **Pragmatic Fixtures & Mocking:** Mock only external boundaries (network APIs, clock, filesystem I/O); never mock internal domain calculations.

## 4. Coverage & Integrity
- **Diff-Coverage & Domain Focus:** Cover new production logic and critical boundaries; prune dead defensive branches instead of writing vacuous tests; use exclusion annotations sparingly.
- **Test Integrity:** Never weaken assertions, delete valid tests, or skip failing checks to pass CI; fix the root cause.
