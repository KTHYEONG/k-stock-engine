---
trigger:
  - on_file_path_regex: "src/.*"
  - on_file_path_regex: "tests/.*"
priority: 9
---

# Code Style & Engineering Standards

> **Respect existing architecture and conventions. Maintain strong typed contracts at public boundaries without breaking scope. Reconcile code with specifications objectively, and apply equivalent engineering rigor across all languages.**

## 1. Architectural Integrity & Modularity
- **Layout & Scope Hygiene:** Respect the repository's existing source/test layout and architectural layer boundaries. Maintain high cohesion and avoid out-of-scope sweeping refactorings.
- **Contract & Spec Reconciliation:** Treat validated executable code and contracts as authoritative for current implementation, but actively investigate discrepancies with specifications or documentation rather than assuming either side is automatically correct.

## 2. Typing, Contracts & Configuration
- **Strong Typing at Boundaries:** Enforce strong static typing for public interfaces, domain models, and core production logic. Never weaken existing type guarantees without explicit justification.
- **Explicit Configuration:** Keep runtime settings externalized and validated at application boundaries; never hardcode environment-dependent secrets or endpoints.
- **Toolchain Alignment:** Use the repository's declared environment and quality tooling (e.g., `uv run` for `uv`-managed projects, with configured linter/type-checker/test suites); do not introduce competing toolchains unnecessarily.

## 3. Polyglot & Native Extensions
- **Equivalence of Rigor:** Any introduced non-Python component (e.g., Rust, C++) must meet equivalent quality baselines: static compiler checks, idiomatic linting, and risk-appropriate testing.
- **FFI Boundary & Safety:** Explicitly define memory ownership and error semantics across language boundaries. Safely translate recoverable native errors to host exceptions without crashing the process.
- **Contract Continuity:** Expose typed interface boundaries (e.g., type stubs, typed bindings) so end-to-end verification remains seamless from the host runtime.
