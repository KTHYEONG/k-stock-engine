---
name: implement
description: Implement an approved spec mechanically with focused invariant guards and integration verification.
---

# Implement Protocol

Fast-execution protocol for mechanical code implementation based strictly on frozen specs (`_spec.md`).

## Execution Principles

Operate as a deterministic translator turning the specification into code and passing tests:
1. Implement clean production logic satisfying the spec's invariants and docstring across all targets.
2. Implement targeted invariant guard tests satisfying the spec's Invariant Scenarios.
3. Wire the caller at the specified anchor point(s).
4. Verify in one pass with `lean_check.py`.

## Directives

1. **Scaffolding Exclusion**:
   - Production code must contain only finalized code and docstrings.
   - Do not paste or leave temporary spec directives, step numbers, or placeholder comments in code or docstrings.

2. **Fidelity & Anti-Defensive Sprawl**:
   - Treat the spec as truth. Do not invent unrequested parameters or speculative abstraction layers.
   - Do not leave stubs (`pass`, `...`, `NotImplementedError`, placeholder returns).
   - Avoid speculative `try-except` blocks or unrequested null checks that are not required by spec invariants.

3. **Clean Test Naming Rule**:
   - Do NOT hardcode temporary spec/ticket IDs (e.g. `POLICY-01`, `SCENARIO-02`) into test function names.
   - Use idiomatic Pythonic names reflecting the target and behavior: `def test_<target_function>_<invariant_behavior>():`.
   - If scenario traceability is desired, add it optionally to the first line of the test docstring, not the function identifier.

4. **Streamlined Implementation Pipeline (One-Pass Gate)**:
   - **Phase 1 (Production Logic & Tests)**:
     - Implement clean production logic for all targets specified in the blueprint.
     - Implement corresponding invariant guard tests in designated test files.
     - Batch related target and test implementations without fracturing into unnecessary intermediate turns.
     - Do NOT run redundant Red-check runs (executing pytest before code is written) or intermediate ad-hoc `ruff check` commands.
   - **Phase 2 (Anchor Wiring)**:
     - Wire invocations into caller files at designated `- Anchor: <anchor>` points once target symbols are in place.
   - **Phase 3 (Single-Gate Verification)**:
     - Run the unified verification gate once across the entire scope:
       ```bash
       uv run python tools/agent_skills/lean_check.py --spec <spec_file>
       ```
     - `lean_check.py` executes Ruff, Mypy, Pytest (xdist + diff-coverage), and scaffolding guards in parallel.
   - **Phase 4 (Targeted Failure Isolation - Only on Failure)**:
     - If `lean_check.py` reports failures, isolate and fix only the flagged points:
       - Pytest failure: run only the failing test file (`uv run pytest <failed_test_file> -q --tb=short`) to debug and repair.
       - Lint/Type failure: fix the exact line reported in the diagnostic.
       - Re-run `lean_check.py` to confirm resolution.

5. **Diff Coverage Resolution (Pruning Over Bloat)**:
   - If diff coverage reports untested lines:
     1. Evaluate if it is speculative defensive code (unrequested `try-except`, unreachable branches): **Prune and delete the code**.
     2. If required domain logic lacks coverage, add the missing boundary scenario.

## Output

Keep chat output ultra-compact and token-efficient.
**Strictly Prohibited**: Do NOT write lengthy implementation prose, detailed code changes, verbose Problem / Root Cause / Impact explanations, or redundant lists of modified files (the spec and git already track them). Do NOT include "다음 단계" recommendations. Only output the minimal summary card below:

### 🔨 [IMPLEMENT] <Task Title>
> 📄 **구현 스펙**: [`<spec_filename>.md`](file:///path/to/docs/specs/<spec_filename>.md)  
> 🚦 **상태**: ✅ COMPLETE (<Count>개 파일 수정)

- 🧪 **검증 요약**: Pytest PASS · Ruff PASS · Mypy PASS · Diff Coverage 100%

*(On Failure / Escalation)*:
### 🔨 [IMPLEMENT] <Task Title>
> 📄 **구현 스펙**: [`<spec_filename>.md`](file:///path/to/docs/specs/<spec_filename>.md)  
> 🚦 **상태**: ❌ ESCALATED (또는 ❌ FAIL)

- 💥 **실패 지점**: [<Pytest | Ruff | Mypy | Diff Coverage | Anchor Wiring>] `<실패한 테스트명 또는 핵심 에러 1줄>`
