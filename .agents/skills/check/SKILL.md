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

2. **Tier 1: Deterministic Audit Gate (Fast Script)**:
   - Run Smart Selective Verification runner (auto-detects modified `.py` files, executes static checks & pinpoint tests):
     ```bash
     uv run python tools/agent_skills/lean_check.py
     ```
   - If a specific contract is targeted, explicitly pass `--spec`:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json
     ```
   - **Immediate Stop on Tier 1 Failure**: If `lean_check.py` fails, immediately report `FAIL` with the root cause diagnostics without proceeding to Tier 2.

3. **Tier 2: Silent Semantic Defect Scan (Targeted Code Review)**:
   - If Tier 1 passes, perform a silent, token-efficient scan of the modified changes (`git diff`) focused ONLY on critical defects:
     1) **Test Realism & Exception Specificity**: Ensure tests are non-vacuous (no trivial `assert True`, mocks do not mask core logic, and `pytest.raises` specifies `match=` or precise exception types).
     2) **Contract & Invariant Integrity**: Verify core business invariants, division by zero / None handling, and boundary edge cases specified in requirements.
     3) **Production Wire-up & No Ghost Paths**: Verify new logic is actually invoked in the production pipeline/entry-point (replacing legacy callers) and no unhandled branches or orphaned dead code remain.
   - *Rule*: Do NOT nitpick purely subjective styling or propose cosmetic refactoring. Flag ONLY concrete bugs, spec omissions, or broken invariants.

4. **Strict Audit Gate (No Code Mutation)**:
   - Perform auditing independently. Do NOT modify source code during the check pass.
   - **Pre-sync Housekeeping Exception**: If the *only* failure is `test_code_map.py` (due to newly added canonical modules not yet registered in `docs/code_map.json`), treat logic audit as **PASS** with a clear note to run `/sync` next to close out code_map registration.
   - If any Tier 1 or Tier 2 defect is found, reject the check with a compact failure diagnosis for resolution in `/implement` or `/spec`.

## Output

Do NOT add any intro, preamble, sub-bullet checks, breakdown items, or extra explanations. Print EXACTLY one line for PASS:

- **PASS** (Strict 1-Line ONLY, No sub-bullets or details):
  ✅ PASS: <Audit Target>

- **FAIL** (Compact format):
  ❌ FAIL: <Audit Target> | Root: <Cause> | Impact: <Scope> | Fix: <Action>

