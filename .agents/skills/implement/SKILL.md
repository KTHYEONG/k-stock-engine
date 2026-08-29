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

2. **Phased Mechanical Workflow**:
   - **Phase A (TDD Scenarios - Per-File Sequential Implementation)**:
     - Group `scenarios` by unique `target_test_file`.
     - Iterate through each `target_test_file` one by one:
       1. View `target_test_file`.
       2. Append/update all test functions for the scenarios belonging to that specific file (matching `scenario_id` 1:1).
       3. Run verification immediately: `uv run pytest <target_test_file> -k "<scenario_id>" -q --tb=short && uv run ruff check <target_test_file>`.
     - DO NOT proceed to Phase B until all test scenarios across all `target_test_file`s are written and failing/passing as expected.
   - **Phase B (Core Logic)**: Implement complete source logic in `target_file`.
     - *Quick Syntax/Lint Check*: `uv run ruff check <target_file>`
   - **Phase C (Integration Wiring)**: Wire logic into `caller_file` at `anchor` location using `import_symbol` and `invocation_expression`.
     - *Quick Syntax/Lint Check*: `uv run ruff check <caller_file>`

3. **Surgical Modifications & Hygiene**:
   - Use targeted edits (`replace_file_content`) only. Preserve all surrounding unrelated code and imports.
   - Never embed ephemeral `docs/specs/*.md` file paths or section numbers into comments or docstrings.

4. **Mandatory Verification & Self-Healing Loop**:
   - **Mandatory Final Verification**: MUST run `uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json` before finishing.
   - **Self-Healing on Missing Tests/Symbols**: If `lean_check` reports any missing tests (e.g. `contract에 정의된 N개 시나리오 테스트가 미구현`) or missing symbols, DO NOT stop or request user action. Read the error diagnostic, implement the missing tests/symbols in their designated files, and re-run `lean_check` until status is `PASS` (0 errors).
   - **Local Bug Fix (Max 3 attempts)**: Fix straightforward implementation bugs (typos, off-by-one, type errors, imports) autonomously.
   - **Escalation to `/spec`**: STOP immediately and do NOT rewrite caller interfaces or invent new architectures ONLY if:
     1) `contract.json` signature/type fundamentally conflicts with existing caller/callee contracts.
     2) Tests reveal an architectural impossibility or circular dependency.
     3) 3 fix attempts fail due to underlying design flaws.

## Output

### 🔨 [IMPLEMENT] <Task Title>

- **Status**: ✅ COMPLETE (or ❌ ESCALATED TO /spec)
- **Modified**: <Count> files
- **Verification**:
  - 🧪 Pytest: <Passed>/<Total> passed
  - 🧹 Ruff / Mypy: <PASS/FAIL>
  - 📐 Spec Compliance: <PASS/FAIL>
