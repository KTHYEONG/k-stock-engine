---
name: check
description: Independently audit contract compliance, typing, regressions, coverage, and test validity.
---

# Check Protocol

Independent audit gate completing the main development loop (`probe` -> `spec` -> `implement` -> `check`). Performs code review, strict quality checks, and regression verification without mutating source code.

## Directives

1. **Identify Modified Scope & Active Spec**:
   - Inspect modified files using `git status --short`.
   - Identify active spec contract under `docs/specs/*_contract.json` if available.

2. **Tier 1: Deterministic Audit Gate (Fast Script)**:
   - Run Smart Selective Verification runner (auto-detects modified `.py` files; self-heals `docs/code_map.json`, then runs static checks, pinpoint tests, and a diff-scoped coverage gate):
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json
     ```
     (Omit `--spec` if auditing an un-specced patch or chore).
   - This gate includes: Ruff, Mypy, impact-scoped pytest, and **diff coverage** — every line the diff *adds* to a touched `src/` file must execute during the test run (not a flat %, the exact new lines).
   - **Immediate Stop on Tier 1 Failure**: If `lean_check.py` fails, immediately report `FAIL` with the root cause diagnostics without proceeding to Tier 2.

3. **Tier 2: Semantic Defect Scan (Targeted Code Review)**:
   - Correctness over speed here: Tier 1 already caught the mechanical failures, so spend the reasoning budget Tier 2 needs to actually catch what a script can't. Scan the modified changes (`git diff`) for:
     1) **Test Realism & Exception Specificity**: Ensure tests are non-vacuous (no trivial `assert True`, mocks do not mask core logic, and `pytest.raises` specifies `match=` or precise exception types).
     2) **Contract & Invariant Integrity**: Verify core business invariants, division by zero / None handling, and boundary edge cases specified in `requirements`. Verify each entry in `design_rationale.failure_modes` has a corresponding guard in the diff.
     3) **No Dead Defensive Code (Defensive Sprawl Audit)**: Flag unrequested `try-except Exception` catches, silent `except: return None`, or speculative null checks that hide bugs or skirt coverage.
     4) **Performance Budget Honored**: Confirm actual use of `dtype_precision`/`storage_format`/`chunking_strategy`, and flag any `timeout`, `max_iterations`/`n_epochs` cap, sample-size reduction, or shortened date-range introduced without technical justification (`.agents/rules/performance.md` §0).
     5) **Domain Principle Compliance**: Cross-check against `.agents/rules/quant.md`, `.agents/rules/performance.md`, and `.agents/rules/python.md`.
     6) **Production Wire-up & No Ghost Paths**: Verify new logic is actually invoked in the production pipeline/entry-point and no unhandled branches or orphaned dead code remain.

4. **Strict Audit Gate (Strictly No Code Mutation & Clean Routing)**:
   - **Zero Code Modification**: Perform auditing independently. **Do NOT edit code or tests during the check pass.** Modifying code in `check` causes workflow leakage and destroys accountability.
   - **Triage and Route Failures Cleanly**:
     - **Route to `/implement`**: Defect is purely implementation execution (unrequested dead defensive branch, broken syntax, failing assertion, incomplete wiring). Command: Provide exact root cause and instruction to simplify/fix.
     - **Route to `/spec`**: Defect is architectural or missing specification (missing wiring test scenario in contract, wrong invariant, contract signature mismatch, performance budget breach). Command: Provide exact contract amendment required.
     - **Route to `/probe`**: Defect is fundamental hypothesis invalidation, unexpected mathematical instability, or algorithmic failure under real data that requires re-experimentation before re-contracting.

## Output

Do NOT add any intro, preamble, sub-bullet checks, breakdown items, or conversational commentary.

- **PASS** (Strict 1-Line ONLY):
  ✅ PASS: <Audit Target>

- **FAIL** (Compact 1-2 Lines format):
  ❌ FAIL: <Audit Target> | Root: <Cause> | Impact: <Scope> | Fix: <Action> → `/implement`, `/spec`, or `/probe`
