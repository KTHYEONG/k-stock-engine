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
   - Run Smart Selective Verification runner (auto-detects modified `.py` files; self-heals `docs/code_map.json`, then runs static checks, pinpoint tests, and a diff-scoped coverage gate):
     ```bash
     uv run python tools/agent_skills/lean_check.py
     ```
   - If a specific contract is targeted, explicitly pass `--spec`:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json
     ```
   - This gate now includes: Ruff, Mypy, impact-scoped pytest, and **diff coverage** — every line the diff *adds* to a touched `src/` file must execute during the test run (not a flat %, the exact new lines). A failure here means the new code is untested, not just "under some threshold" — treat it the same as a failing test, not a style nit.
   - **Immediate Stop on Tier 1 Failure**: If `lean_check.py` fails, immediately report `FAIL` with the root cause diagnostics without proceeding to Tier 2.

3. **Tier 2: Semantic Defect Scan (Targeted Code Review)**:
   - Correctness over speed here: Tier 1 already caught the mechanical failures, so spend the reasoning budget Tier 2 needs to actually catch what a script can't. If a repeat check cycle is cheaper than a defect reaching `implement` a second time, take the repeat cycle. Scan the modified changes (`git diff`) for:
     1) **Test Realism & Exception Specificity**: Ensure tests are non-vacuous (no trivial `assert True`, mocks do not mask core logic, and `pytest.raises` specifies `match=` or precise exception types).
     2) **Contract & Invariant Integrity**: Verify core business invariants, division by zero / None handling, and boundary edge cases specified in `requirements`. If the spec's `design_rationale.failure_modes` is present, check off each listed failure mode against a corresponding guard in the diff — an unaddressed entry is a defect, not a note to skip.
     3) **Performance Budget Honored**: If `performance_budget` is present, confirm the diff actually uses its `dtype_precision`/`storage_format`/`chunking_strategy` rather than ad-hoc defaults, AND flag any `timeout`, `max_iterations`/`n_epochs` cap, sample-size reduction, or shortened date-range introduced without a stated technical justification tied to a real constraint — per `.agents/rules/performance.md` §0, silently truncated backtests/training runs are a fail-closed defect.
     4) **Domain Principle Compliance**: Cross-check the diff against **every** active rule file whose trigger paths match the touched files — `.agents/rules/quant.md`, `.agents/rules/performance.md`, `.agents/rules/python.md` — not just the contract's own `requirements`; the contract itself can be domain-wrong even if internally consistent. Don't skip a rule file to save a pass; a missed domain violation is more expensive than reading one more file.
     5) **Production Wire-up & No Ghost Paths**: Verify new logic is actually invoked in the production pipeline/entry-point (replacing legacy callers) and no unhandled branches or orphaned dead code remain.
   - *Rule*: Do NOT nitpick purely subjective styling or propose cosmetic refactoring unrelated to the diff. Flag concrete bugs, spec omissions, or broken invariants — thoroughness on those is the point of this tier, not something to trade away for brevity.

4. **Strict Audit Gate (No Code Mutation)**:
   - Perform auditing independently. Do NOT modify source code during the check pass.
   - If any Tier 1 or Tier 2 defect is found, reject the check with a compact failure diagnosis. Route the fix explicitly: implementation-only bugs (wiring, test logic, wrong branch) go to `/implement`; defects traceable to `requirements` or `design_rationale` itself (wrong invariant, missed domain rule, flawed assumption) go to `/spec`.

## Output

Do NOT add any intro, preamble, sub-bullet checks, breakdown items, or extra explanations. Print EXACTLY one line for PASS:

- **PASS** (Strict 1-Line ONLY, No sub-bullets or details):
  ✅ PASS: <Audit Target>

- **FAIL** (Compact format):
  ❌ FAIL: <Audit Target> | Root: <Cause> | Impact: <Scope> | Fix: <Action> → `/implement` or `/spec`

