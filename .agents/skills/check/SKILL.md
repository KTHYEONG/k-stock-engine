---
name: check
description: Independently audit contract compliance, typing, regressions, coverage, and test validity.
---

# Check Protocol

Independent audit gate completing the main development loop (`spec` -> `implement` -> `check`). Performs code review, strict quality checks, and regression verification.

## Directives

1. **Identify Modified Scope & Active Spec**:
   - Inspect modified files using `git status --short`.
   - Identify active spec contract under `docs/specs/*_contract.json` if available.

2. **Standard Audit Execution**:
   - Run Smart Selective Verification runner (auto-detects modified `.py` files, executes static checks & pinpoint tests in seconds):
     ```bash
     uv run python tools/agent_skills/lean_check.py
     ```
   - If a specific contract is targeted, explicitly pass `--spec`:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json
     ```
   - **Fast Static Mode** (when testing is verified and checking lint/types/contract only):
     ```bash
     uv run python tools/agent_skills/lean_check.py --fast
     ```
   - Fallback (if script fails):
     - Code Style: `uv run ruff check <modified_files>`
     - Strict Typing: `uv run mypy <modified_files>`
     - Target Tests: `uv run pytest <target_test_files> -q --tb=line`

3. **Strict Audit Gate (No Code Mutation)**:
   - Perform auditing independently. Do NOT modify source code during the check pass.
   - Verify non-vacuous tests and contract compliance against `contract.json`.
   - **Pre-sync Housekeeping Exception**: If the *only* failure is `test_code_map.py` (due to newly added canonical modules not yet registered in `docs/code_map.json`), treat logic audit as **PASS** with a clear note to run `/sync` next to close out code_map registration. Do NOT waste tokens trying to debug code logic for this pre-sync gap.
   - If actual logic/type/test audit fails, report the exact failure diagnosis clearly for resolution in `/implement` or `/spec`.

## Output

Do NOT add any intro, preamble, sub-bullet checks, breakdown items, or extra explanations. Print EXACTLY one line for PASS:

- **PASS** (Strict 1-Line ONLY, No sub-bullets or details):
  ✅ PASS: <Audit Target>

- **FAIL** (Compact format):
  ❌ FAIL: <Audit Target> | Root: <Cause> | Impact: <Scope> | Fix: <Action>

