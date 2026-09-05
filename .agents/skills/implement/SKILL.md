---
name: implement
description: Implement an approved spec mechanically with focused TDD and integration verification.
---

# Implement Protocol

Fast-execution protocol for mechanical code implementation based strictly on frozen spec contracts.

## Directives

1. **Zero Guesswork & Complete Implementation**:
   - Treat `contract.json` as absolute truth. Do not invent new parameters, change signatures, or create speculative abstraction layers.
   - **Anti-Stub Rule**: Never leave `pass`, `...`, `NotImplementedError`, `TODO`, or placeholder return values. Implement the full domain logic specified in `requirements`.
   - **1:1 Test Mapping**: Every entry in `contract.json` -> `scenarios` MUST be explicitly implemented as a concrete test function across all specified `target_test_file`s (matching `scenario_id`). When using `pytest.raises`, always specify `match=` or concrete exceptions (Ruff PT011).
   - **Zero-Search Context Loading**: Read only `target_file`, `target_test_file`, and files listed in `context_files` (if present) via targeted `view_file`. Do NOT run exploratory `rg` / `find` / `list_dir` commands across the repository.
   - **Translate `design_rationale` & `performance_budget` Into Code, Not Just `requirements`**: These are not reference-only fields.
     - Each entry in `design_rationale.failure_modes` MUST map to a concrete guard (branch, validation, or assertion) in `target_file` — if a listed failure mode has no corresponding code path, that's a missing requirement, not an optional note.
     - If `performance_budget` is present, its values are literal implementation choices: use `dtype_precision` for array/DataFrame construction, `storage_format` for any read/write I-O, and `chunking_strategy` for the iteration/batch structure. Do not substitute your own defaults.

2. **Phased Mechanical Workflow (Zero Invention)**:
   - **Phase A (TDD Scenarios Placement - Red)**:
     - Group `scenarios` by unique `target_test_file`.
     - Count total scenarios ($N$). Iterate through each `target_test_file`:
       1. View `target_test_file`.
       2. Append all $N$ test functions using `test_skeleton` directly from `contract.json`. **Do NOT rewrite assertions, redesign fixtures, or invent test structures. Paste skeletons faithfully.**
       3. Run **one** verification pass for the whole file, never one `-k`-filtered invocation per scenario: `uv run pytest <target_test_file> -q --tb=short && uv run ruff check <target_test_file>`. A single unfiltered run already reports every scenario's pass/fail individually, so N separate `-k` invocations for N scenarios in one file buy zero extra signal for N× the tool calls (and N× process-startup/collection overhead). Confirm the newly-added scenario tests fail (Red); pre-existing tests in the same file are expected to still pass.
     - **Gate Check**: DO NOT touch `target_file` (Phase B) until ALL $N$ scenario tests are physically present in `target_test_file`s.
   - **Phase B (Core Logic & Wiring - Green)**:
     - Implement logic in `target_file` according to `changes` and `requirements` to turn tests green.
     - Complete wiring in `caller_file` at `anchor` using `import_symbol` and `invocation_expression`.
     - Run: `uv run ruff check <target_file> <caller_file>`.

3. **Mandatory Final Verification & Hard Stop**:
   - **Hard Rule**: You MUST run `uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json`.
   - **DO NOT output completion response until `lean_check.py` returns `PASS` (0 errors).**
   - **Self-Healing on Missing Tests/Symbols**: If `lean_check` reports missing tests/symbols, read the diagnostic, implement the missing elements in designated files, and re-run `lean_check` until status is `PASS`. **General retry ceiling**: if the *same* diagnostic (same phase + same file) survives 3 consecutive fix attempts — not just ones you've labeled "spec-level" — stop retrying and escalate per below. A full Tier 1 run (ruff+mypy+pytest+coverage) is not free; an open-ended retry loop on a fix that isn't landing burns wall-clock without a cap otherwise.
   - **Escalation to `/spec`**: STOP immediately ONLY if:
     1) `contract.json` signature fundamentally conflicts with existing repository contracts.
     2) Tests reveal an architectural impossibility or circular dependency.
     3) 3 consecutive fix attempts fail on the same diagnostic (whether or not the cause is obviously spec-level).
     4) The actual data/workload measured during implementation breaches `performance_budget` (e.g. real memory use exceeds `memory_target_mb`) — this is a sizing error in the design, not something to silently patch by shrinking scope (never resolve this by adding a timeout or cutting the run short; re-spec the budget instead).

## Output

### 🔨 [IMPLEMENT] <Task Title>

- **Status**: ✅ COMPLETE (or ❌ ESCALATED TO /spec)
- **Modified**: <Count> files
- **Verification**:
  - 🧪 Pytest: <Passed>/<Total> passed
  - 🧹 Ruff / Mypy: <PASS/FAIL>
  - 📐 Spec Compliance: <PASS/FAIL>
