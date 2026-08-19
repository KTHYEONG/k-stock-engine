---
name: implement
description: Implement an approved spec mechanically with focused TDD and integration verification.
---

# Implement Protocol

Fast-execution protocol for mechanical code implementation based strictly on frozen spec contracts.

## Directives

1. **Zero Guesswork & Minimalist Implementation**:
   - Treat `contract.json` as absolute truth. Do not invent new parameters, change signatures, or create speculative abstraction layers.
   - Implement strictly what is specified in `requirements` and `scenarios`.
   - **Zero-Search Context Loading**: Read only `target_file`, `target_test_file`, and files listed in `context_files` (if present) via targeted `view_file`. Do NOT run exploratory `rg` / `find` / `list_dir` commands across the repository.

2. **Phased Mechanical Workflow**:
   - **Phase A (TDD Scenarios)**: Translate `scenarios` from `contract.json` into concrete `pytest` test cases in `target_test_file`.
     - *Quick TDD Check*: `uv run pytest <target_test_file> -k "<scenario_id>" -q --tb=short`
   - **Phase B (Core Logic)**: Implement source logic in `target_file`.
     - *Quick Syntax/Lint Check*: `uv run ruff check <target_file>`
   - **Phase C (Integration Wiring)**: Wire logic into `caller_file` at `anchor` location using `import_symbol` and `invocation_expression`.
     - *Quick Syntax/Lint Check*: `uv run ruff check <caller_file>`

3. **Surgical Modifications & Hygiene**:
   - Use targeted edits (`replace_file_content`) only. Preserve all surrounding unrelated code and imports.
   - Never embed ephemeral `docs/specs/*.md` file paths or section numbers into comments or docstrings.

4. **Final Verification & Adaptive Fix Loop**:
   - Run full pinpoint verification: `uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json`
   - **Local Bug Fix (Max 3 attempts)**: Fix straightforward implementation bugs (typos, off-by-one, type errors, imports) autonomously.
   - **Escalation to `/spec`**: STOP immediately and do NOT rewrite caller interfaces or invent new architectures if:
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
