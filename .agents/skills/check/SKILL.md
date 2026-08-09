---
name: check
description: Independently audit contract compliance, typing, regressions, coverage, and test validity.
---

# Check Protocol

Independent audit gate completing the main development loop (`spec` -> `implement` -> `check`). Performs code review, strict quality checks, and regression verification.

## Directives

1. **Identify Modified Scope**:
   - Inspect modified files using `git status` or `git diff --name-only`.

2. **Standard Audit Execution**:
   - Code Style & Linter: `uv run ruff check .`
   - Strict Type Check: `uv run mypy .`
   - Test Suite Verification: `uv run pytest`


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

