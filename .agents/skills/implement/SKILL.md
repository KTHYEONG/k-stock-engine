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
   - **Phase B (Core Logic)**: Implement source logic in `target_file`.
     - *Checkpoint*: `uv run python tools/agent_skills/lean_check.py --fast --spec docs/specs/<feature>_contract.json`
   - **Phase C (Integration Wiring)**: Wire logic into `caller_file` at `anchor` location using `import_symbol` and `invocation_expression`.
     - *Checkpoint*: `uv run python tools/agent_skills/lean_check.py --fast --spec docs/specs/<feature>_contract.json`

3. **Surgical Modifications & Token Efficiency**:
   - Use targeted edits (`replace_file_content`) only. Preserve all surrounding unrelated code and imports.
   - Never embed ephemeral `docs/specs/*.md` file paths or section numbers into comments or docstrings.
   - **NO FLUFF**: Do NOT output intermediate phase explanations or duplicate modified code in chat text. Execute edits and commands immediately.
   - **Quiet Commands**: Run test commands with quiet/compact flags (e.g. `pytest -q --tb=short`).

4. **Verification & Adaptive Fix Loop**:
   - Run verification: `uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json`
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

