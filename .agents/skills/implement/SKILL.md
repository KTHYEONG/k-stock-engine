---
name: implement
description: Implement an approved spec mechanically with focused TDD and integration verification.
---

# Implement Protocol

Fast-execution model protocol for mechanical feature implementation based on frozen spec contracts.

## Directives

1. **Strict Contract Compliance (Zero Guesswork)**:
   - Treat `contract.json` as absolute input. Do not invent new parameters, change signatures, or alter thresholds.

2. **Phased Mechanical Implementation**:
   - **Phase A (Scenarios)**: Translate `scenarios` and `python_assertion` from `contract.json` into concrete `pytest` test cases in `target_test_file`.
   - **Phase B (Core Logic)**: Implement source logic at `target_file`.
     - *Checkpoint*: Run `uv run python tools/agent_skills/lean_check.py --fast --spec docs/specs/<feature>_contract.json`. Verify non-dummy implementation.
   - **Phase C (Wiring Integration)**: Integrate logic into `caller_file` at specified `anchor` location using `import_symbol` and `invocation_expression`.
     - *Checkpoint*: Re-run `uv run python tools/agent_skills/lean_check.py --fast --spec docs/specs/<feature>_contract.json`. Ensure no orphaned implementations remain.

3. **Surgical Code Modifications & Token Efficiency**:
   - MUST use targeted block/line edits (`replace_file_content` / `multi_replace_file_content`) to prevent code loss or unintended rewrites.
   - **NO FLUFF**: Do NOT output intermediate phase explanations or duplicate modified code in chat text. Execute edits and commands immediately.
   - **Quiet Commands**: Run test commands with quiet/compact flags (e.g. `pytest -q --tb=short`).

4. **Full Audit Verification & Escalation Loop**:
   - Run full verification via `uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json`.
   - If contract conflicts with codebase realities or tests fail due to bad spec logic, STOP and escalate to `/spec`.

## Output

Provide a clear, concise summary with emojis. Example:

### 🔨 [IMPLEMENT] <Task Title>

- **Status**: COMPLETE
- **Modified**: <Count> files (`file1.py`, `file2.py`)
- **Verification**: Pytest <Passed>/<Total> | Ruff <PASS/FAIL> | Mypy <PASS/FAIL>

