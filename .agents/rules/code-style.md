---
trigger:
  - on_file_path_regex: "src/.*"
  - on_file_path_regex: "tests/.*"
priority: 9
---

# Code Style & Engineering Standards

## 1. Universal Engineering Directives (Language-Agnostic)
- **Typing & Contracts:** Strict static typing is mandatory. Model invariants via contracts (schemas, protocols, interfaces). Code contracts strictly supersede external documentation.
- **Modularity & Scope Hygiene:** Maintain high cohesion and layer separation. Avoid out-of-scope sweeping refactorings.
- **Directory Isolation:** Production source resides in `src/`, test suites in `tests/`.

## 2. Default Runtime (Python)
- **Tooling & Env:** Managed via `uv`. Execute linting, typing, and tests with `uv run` (`ruff`, `mypy`, `pytest`).
- **Configuration:** Manage settings via environment variables and typed config schemas (`pydantic-settings`).

## 3. Polyglot & Native Extensions
- **Equivalence of Rigor:** Any introduced non-Python component must establish its own rigorous quality baseline (strict static type/compiler checks, idiomatic linter, comprehensive tests).
- **Runtime Safety & Isolation:** Extension modules must fail gracefully and translate errors safely into the host runtime without crashing the process.
- **Contract Continuity:** Expose typed interface boundaries (e.g. type stubs, typed FFI) so end-to-end type safety remains verifiable across the entire system.
