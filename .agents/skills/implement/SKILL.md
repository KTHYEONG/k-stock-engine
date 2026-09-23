---
name: implement
description: Implement an approved spec mechanically with focused invariant guards and integration verification.
---

# Implement Protocol

Fast-execution protocol for mechanical code implementation based strictly on frozen specs (`_spec.md`).

## Execution Principles

Execute the approved specification into production code and passing tests:
1. Implement clean production logic satisfying the spec's invariants and docstring across all targets.
2. Implement targeted invariant guard tests satisfying the spec's Invariant Scenarios.
3. Wire the caller at the specified anchor point(s).
4. Verify in one pass with `lean_check.py`.

## Directives

1. **Scaffolding Exclusion**:
   - Production code must contain only finalized code and docstrings.
   - Do not paste or leave temporary spec directives, step numbers, or placeholder comments in code or docstrings.

2. **Fidelity with Pragmatic Grounding**:
   - Treat the spec as the authoritative blueprint. Do not invent unrequested parameters, speculative abstraction layers, or dead defensive branches.
   - Do not leave stubs (`pass`, `...`, `NotImplementedError`, placeholder returns).
   - **System Truth Discrepancy Escalation**: If the spec conflicts with real codebase invariants, external type signatures, or existing contracts, do not force an incompatible implementation. Document the concrete discrepancy and escalate/adjust the invariant rather than guessing.

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
     - Run the unified verification gate for the target feature and spec scope:
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
     1. Evaluate if it is speculative defensive code (unrequested dead branches): **Prune and delete the bloat**.
     2. If required domain logic lacks coverage, add the missing boundary scenario.
     3. For non-testable infrastructure branches, use `# pragma: no cover` appropriately.

## Output

Keep chat output compact and token-efficient. Retain English keys/badges while writing descriptions in natural Korean (한국어):
- **On success**: Output only the minimal completion card below without redundant code dumps or conversational filler.
- **On failure or discrepancy**: Clearly report the issue (Problem → Root Cause → Fix).

### 🔨 [IMPLEMENT] <Task Title>
> 📄 **Spec**: [`<spec_filename>.md`](docs/specs/<spec_filename>.md)  
> 🚦 **Status**: ✅ COMPLETE (<Count> file(s) modified)

- 🧪 **Verification**: <실제 통과 내역 요약, e.g. Pytest PASS · Ruff PASS · Mypy PASS · Diff Coverage PASS>

*(On Failure / Escalation)*:
### 🔨 [IMPLEMENT] <Task Title>
> 📄 **Spec**: [`<spec_filename>.md`](docs/specs/<spec_filename>.md)  
> 🚦 **Status**: ❌ ESCALATED (or ❌ FAIL)

- 💥 **Failure Point**: [<Pytest | Ruff | Mypy | Diff Coverage | Anchor Wiring | Invariant Conflict>] `<실패한 테스트명 또는 핵심 에러 1줄>`
- 🎯 **Root Cause & Action**: `<불일치 원인 또는 해결을 위해 필요한 조치 1-2줄>`
