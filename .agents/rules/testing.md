# Testing Directives & Quality Standards

> **Verify observable behavior and interface contracts, not implementation internals. Maintain fast feedback loops with small deterministic inputs while isolating heavy runs. Enforce invariant verification on new logic, isolate failures with pinpoint precision, and prioritize economic correctness over vanity metrics.**

## 1. Development Paradigm: Invariant-Driven Development (IDD)
- **Contracts Over Premature Tests:** Prioritize strict typing (Pydantic, Enums, Mypy `strict`) and domain invariants over dogmatic test-first rituals. Define the interface and invariants before writing code; avoid premature test code that freezes internal APIs.
- **Two-Track Workflow (Escape Ceremony Trap):**
  - *Fast-Track (Surgical fixes, 1-2 files, bugfixes, config changes):* Direct implementation + Invariant Guard Test -> `lean_check.py` verification. Spec documents are not required for small localized changes.
  - *Standard Track (New modules, complex algorithms, pipeline changes):* Probe (scratch experiment) -> Spec (contract & invariant scenarios) -> Implement (logic & guard tests) -> Check.
- **Observable Behavior & Invariants:** Verify return contracts, state mutations, conservation laws, and error conditions rather than private implementation details or mock call-counts.
- **In-Memory & Minimal Inputs:** Use the smallest deterministic synthetic data sufficient to test target logic near-instantaneously; never load multi-year disk datasets in unit tests.
- **Do Not Test Profitability:** Unit and integration tests verify correctness, edge cases, and schema transformations—never long-horizon profitability, market alpha, or model convergence.

## 2. Quantitative & Financial Invariant Testing
- **Financial Invariants:** Verify structural conservation laws: cash/position/NAV reconciliation, exposure and leverage limits, deterministic outputs for identical inputs, and deduplication of orders/fills.
- **Temporal & Numerical Boundaries:** Stress-test boundary conditions rather than happy paths alone: zeros, NaNs, empty universes, missing bars, market holidays, duplicate timestamps, timezone transitions, and floating-point tolerances (`pytest.approx`, `rtol`/`atol`).
- **No Look-Ahead Invariance:** Explicitly test that signal and execution calculations do not access future time steps or unreleased event timestamps.

## 3. Execution, Latency Budgets & Failure Triage
- **Latency Budgets:** Fast unit tests must execute near-instantaneously using in-memory fixtures. Heavy end-to-end simulations, full model retraining, or multi-year backtests must be marked `@pytest.mark.slow` and isolated from default test runs.
- **Pinpoint Failure Isolation:** During localized TDD and debugging, test focused targets (`uv run pytest path/to/test.py -k <test_name> -q`) and inspect direct tracebacks for rapid feedback. For final task validation, run the designated spec suite and verify directly coupled integration tests.
- **Process-Isolated Temp Roots:** Never wipe shared temp directories globally. Each pytest run operates in its own partitioned `tmp/pytest/proc_{pid}_{uuid}/` directory to prevent race conditions and ghost crashes between concurrent agents and developer terminal commands.
- **Hermetic Environment Invariant:** Unit tests must be 100% hermetic and isolated from live developer shell credentials and environment variables (e.g. `LIVE_*`, broker/exchange API keys, data vendor tokens, or production order flags). Tests requiring specific configuration or credentials must configure safe mocks explicitly via `monkeypatch.setenv`.
- **Tooling Artifact Awareness:** If tests pass alone but fail strictly under instrumentation (`--cov`), diagnose tracer overhead, timeout expiration, or multiprocessing/fork interference before assuming a domain code regression.
- **Pragmatic Fixtures & Mocking:** Mock external boundaries (REST/WebSocket APIs, clock/system time, filesystem I/O). Never mock internal domain calculations or transform tests into meaningless mock-chains.

## 4. Diff-Coverage & Quality Philosophy
- **Diff-Coverage & Domain Invariant Focus:** Do not chase global percentage quotas across untouched legacy modules. Focus on thorough test coverage of newly-added production logic (`src/`), ensuring domain transformations and boundary scenarios are verified.
- **Uncovered Line Resolution Protocol:** If new lines are reported as uncovered by diff-coverage, evaluate:
  1. *Is it a genuine domain branch/exception?* -> Add a targeted scenario test exercising that boundary.
  2. *Is it speculative defensive code or unreachable dead branches?* -> **Do NOT write artificial tests; remove the defensive bloat and simplify the code.**
  3. *Is it abstract interface declarations, typing guards, or non-deterministic environment scaffolding where testing would require vacuous mocks?* -> Use `# pragma: no cover` sparingly rather than constructing fragile, artificial mock chains solely to satisfy coverage counters.
- **Test Integrity:** Never weaken assertions, delete valid tests, or skip failing checks to satisfy CI. Diagnose and fix the root cause.