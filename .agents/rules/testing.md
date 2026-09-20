---
trigger:
  - on_label: ["testing"]
  - on_file_path_regex: "tests/.*\\.py"
  - on_file_path_glob: ["tests/**/*.py"]
priority: 8
---

# Testing Directives & Quality Standards

> **Verify observable behavior and interface contracts, not implementation internals. Maintain fast feedback loops with small deterministic inputs while isolating heavy runs. Enforce diff-coverage on new logic, isolate failures with pinpoint precision, and prioritize economic correctness over vanity metrics.**

## 1. Development Paradigm: Invariant-Driven Development (IDD)
- **Contracts Over Premature Tests:** Prioritize strict typing (Pydantic, Enums, Mypy `strict`) and domain invariants over dogmatic test-first rituals. Define the interface and invariants before writing code; avoid premature test code that freezes internal APIs.
- **Two-Track Workflow (Escape Ceremony Trap):**
  - *Fast-Track (Surgical fixes, 1-2 files, bugfixes, config changes):* Direct implementation + Invariant Guard Test -> `lean_check.py` verification. Spec documents are not required for small localized changes.
  - *Standard Track (New modules, complex algorithms, pipeline changes):* Probe (scratch experiment) -> Spec (contract & invariant scenarios) -> Implement (logic & guard tests) -> Check.
- **Observable Behavior & Invariants:** Verify return contracts, state mutations, conservation laws, and error conditions rather than private implementation details or mock call-counts.
- **In-Memory & Minimal Inputs:** Use the smallest deterministic synthetic data sufficient to test target logic (< 0.1s); never load multi-year disk datasets in unit tests.
- **Do Not Test Profitability:** Unit and integration tests verify correctness, edge cases, and schema transformations—never long-horizon profitability, market alpha, or model convergence.

## 2. Quantitative & Financial Invariant Testing
- **Financial Invariants:** Verify structural conservation laws: cash/position/NAV reconciliation, exposure and leverage limits, deterministic outputs for identical inputs, and deduplication of orders/fills.
- **Temporal & Numerical Boundaries:** Stress-test boundary conditions rather than happy paths alone: zeros, NaNs, empty universes, missing bars, market holidays, duplicate timestamps, timezone transitions, and floating-point tolerances (`pytest.approx`, `rtol`/`atol`).
- **No Look-Ahead Invariance:** Explicitly test that signal and execution calculations do not access future time steps or unreleased event timestamps.

## 3. Execution, Latency Budgets & Failure Triage
- **Latency Budgets:** Fast unit tests must execute in < 0.1s per test (in-memory only). Heavy end-to-end simulations, full model retraining, or multi-year backtests must be marked `@pytest.mark.slow` and isolated from the default run.
- **Pinpoint Failure Isolation (No Full-Suite Retries):** During TDD and debugging, NEVER rerun entire test suites (e.g. 100+ tests) or perform blind sweeps. AI must test ONLY the specific modified file or test (`uv run pytest path/to/test.py -k <test_name> -q`) and inspect the direct traceback.
- **Process-Isolated Temp Roots:** Never wipe shared temp directories globally. Each pytest run operates in its own partitioned `tmp/pytest/proc_{pid}_{uuid}/` directory to prevent race conditions and ghost crashes between concurrent agents and developer terminal commands.
- **Hermetic Environment Invariant:** Unit tests must be 100% isolated from developer shell credentials and environment variables (`LIVE_*`, `BINANCE_*`, `UPBIT_*`). Tests requiring specific environment variables must configure them explicitly via `monkeypatch.setenv`.
- **Tooling Artifact Awareness:** If tests pass alone but fail strictly under instrumentation (`--cov`), diagnose tracer overhead, timeout expiration, or multiprocessing/fork interference before assuming a domain code regression.
- **Pragmatic Fixtures & Mocking:** Mock external boundaries (REST/WebSocket APIs, clock/system time, filesystem I/O). Never mock internal domain calculations or transform tests into meaningless mock-chains.

## 4. Diff-Coverage & Quality Philosophy
- **Diff-Coverage Over Vanity Metrics:** Do not chase global percentage quotas across untouched legacy modules. Focus strictly on 100% diff-coverage for newly-added production logic (`src/`) to ensure zero untested code.
- **Uncovered Line Resolution Protocol:** If new lines are reported as uncovered by diff-coverage, evaluate:
  1. *Is it a genuine domain branch/exception?* -> Add a targeted scenario test exercising that boundary.
  2. *Is it speculative defensive code (e.g. unrequested `try-except` or dead branches)?* -> **Do NOT write artificial tests; remove the defensive bloat and simplify the code.**
- **Escape Hatch for Non-Measurable Lines:** Use `# pragma: no cover` sparingly for genuine infrastructure edge cases (e.g. OS signal exits, fatal crash loggers) rather than contorting tests with complex mocks.
- **Test Integrity:** Never weaken assertions, delete valid tests, or skip failing checks to satisfy CI. Diagnose and fix the root cause.