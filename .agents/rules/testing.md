---
trigger:
  - on_label: ["testing"]
  - on_file_path_regex: "tests/.*\\.py"
  - on_file_path_glob: ["tests/**/*.py"]
priority: 8
---

# Testing Directives & Test Quality Standards

> **Verify observable behavior and interface contracts, not implementation internals. Maintain fast feedback loops with small deterministic inputs while isolating heavy integration runs. Prioritize failure modes, boundaries, and financial invariants over vanity coverage metrics.**

## 1. Test Architecture & Design
- **Observable Behavior & Contracts:** Verify return contracts, state mutations, and error handling rather than private implementation details.
- **Structure & Readability:** Keep tests readable with explicit setup, execution, and assertion phases (e.g., Given/When/Then, parametrized cases).
- **In-Memory & Minimal Inputs:** Use the smallest deterministic synthetic data sufficient to test the target logic; avoid unnecessary disk I/O or massive fixtures in unit tests.
- **Do Not Test Profitability:** Unit and integration tests verify correctness, edge cases, and schema transformations—never long-horizon profitability, market alpha, or model convergence.

## 2. Quantitative & Financial Invariant Testing
- **Financial Invariants:** Verify structural conservation laws: cash/position/NAV reconciliation, exposure and leverage limits, deterministic outputs for identical inputs, and deduplication of orders/fills.
- **Temporal & Numerical Boundaries:** Stress-test boundary conditions rather than happy paths alone: zeros, NaNs, empty universes, missing bars, market holidays, duplicate timestamps, timezone transitions, and floating-point tolerances (`pytest.approx`, `rtol`/`atol`).
- **No Look-Ahead Invariance:** Explicitly test that signal and execution calculations do not access future time steps or unreleased event timestamps.

## 3. Execution, Isolation & Environment Strategy
- **Fast Feedback Core Suite:** Default unit tests must run fast to maintain quick development cycles. Isolate heavy end-to-end simulations, full model retraining, or long-horizon replays into dedicated integration/slow suites.
- **Pragmatic Fixtures & State Isolation:** Scope fixtures appropriately to balance performance and test isolation; avoid sharing mutable state across tests via excessive fixture caching.
- **Boundary Mocking:** Mock external network boundaries, broker/exchange gateway APIs, hardware I/O, and non-deterministic clock/system interfaces. Never mock internal business logic or transform tests into meaningless mock-chains.
- **Robust Failure Verification:** Verify exception types, error codes, or structured payload attributes. Avoid brittle assertions on human-readable error string messages unless message formatting is an explicit contract.
- **Realistic Database & Storage Engines:** Use actual database engines or containerized equivalents when SQL dialect compatibility, query optimization, or transaction concurrency semantics are explicitly under test; use lightweight in-memory storage for pure repository logic.

## 4. Test Quality & Coverage Philosophy
- **Quality Over Vanity Coverage:** Treat test coverage as a diagnostic signal rather than a quota. Prioritize:
  1. Changed core execution paths
  2. Edge cases, failure modes, and boundary transitions
  3. High-risk regression paths
- **Test Integrity:** Never weaken assertions, delete valid test cases, or skip failing checks merely to satisfy CI runs. Diagnose and fix the root cause.