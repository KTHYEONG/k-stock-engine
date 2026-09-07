---
name: spec
description: Produce a machine-readable, zero-invention implementation contract from probed architecture rationale.
---

# Spec Protocol

Produce an unambiguous implementation plan and precision contract (`contract.json`) optimized for mechanical, zero-search, zero-invention downstream execution by low-reasoning models (`implement`).

## High-Reasoning Allocation Philosophy

The design rationale and empirical proof have already been resolved in `/probe`.
As the contract architect, your cognitive budget here is 100% dedicated to **synthesizing complete, executable test harnesses (Unit + Wiring) and exact function signatures**.
Downstream `implement` models have low reasoning capacity and cannot extrapolate missing test fixtures or invent complex mock contexts. Your specification must be so mechanically complete that `implement` only pastes skeletons and turns them green.

## Directives

1. **Prerequisite & Context Alignment**:
   - Input from `/probe`: Consume the chosen architecture, invariants, failure modes, and performance budget established in the probe phase.
   - Inspect target files and immediate callers (1-depth call-sites) to ensure signatures, imports, and AST anchors are exact.

2. **Ambiguity Gate & Invariant Specification**:
   - Translate probed invariants into explicit fail-closed requirements.
   - Forbid open-ended fallback catches (`try-except Exception`) or silent `None` swallows that create unreachable branches downstream.
   - **No Silent Scope-Shrinking**: Ensure full date ranges and required scales are preserved.

3. **Deliverables (Single Source of Truth - `docs/specs/<feature>_contract.json`)**:
   - `target_file`: Relative path to modify or create.
   - `context_files`: Minimal prerequisite paths for zero-search context loading.
   - `changes` (or `symbols`): Array of `{ name, signature, kind, target_file }`.
   - `wiring`: Array of `{ caller_file, anchor, import_symbol, invocation_expression }` ensuring entry-point hookup.
   - `requirements`: Explicit fail-closed boundary rules, invariant constraints, and complexity requirements.
   - `design_rationale`: `{ alternatives_considered, chosen_reason, failure_modes }` — carry over directly from `/probe`.
   - `performance_budget` (required when `target_file` touches backtesting, ML training, or bulk data I/O): `{ expected_data_scale, memory_target_mb, storage_format, dtype_precision, chunking_strategy, acceleration_candidate }`.
   - `scenarios`: Array of `{ scenario_id, target_test_file, execution_command, expected_behavior, test_skeleton }`.
     - **Dual-Scope Coverage Mandate (Unit + Wiring)**:
       1) **Unit Scenarios**: Scenarios exercising new symbols/functions in `target_file`.
       2) **Wiring Scenarios (MANDATORY when `wiring` modifies existing callers)**: If `wiring` touches a `caller_file` (e.g. pipeline, orchestrator, CLI), you MUST provide at least one scenario targeting that `caller_file` test suite (with all required mocks/fixtures 100% written out).
     - `scenario_id`: Valid pytest function name (e.g. `test_<func>_<condition>`).
     - `test_skeleton`: **Mandatory 100% executable Python test function** (Given/When/Then, imports, actual call, concrete assertions). NEVER leave `pass`, `...`, or empty body. Every branch required by `requirements` and `failure_modes` must have an explicit test skeleton.

4. **Self-Validation Gate**:
   - Validate contract schema, test_skeleton AST syntax, and caller anchors:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json --pre-impl
     ```
   - **Low-Reasoning Execution Feasibility Check**: Re-read the contract through the eyes of a low-reasoning model:
     - Will pasting these `test_skeleton`s and implementing `changes` + `wiring` yield 100% diff coverage on BOTH `target_file` and `caller_file`?
     - Are there any hidden fixtures, unprovided mocks, or unspecified exception branches that would force downstream guesswork? If yes, resolve them in `contract.json` before publishing.
   - **Domain Principle Self-Check**: Before finalizing, cross-check against `.agents/rules/performance.md`, `.agents/rules/quant.md`, and `.agents/rules/python.md`.

## Chat Output Format

Keep chat response strictly minimal, zero-redundancy, and actionable (do not repeat spec details contained inside the JSON):

### 📐 [SPEC] <Feature Name>

- **계약 파일**: `docs/specs/<feature>_contract.json` (Pre-impl check: ✅ PASS)
- **대상 파일**: `<target_file>` (+ wiring: `<caller_file>`)
- **테스트 시나리오**: 총 <N>개 (<Unit N개> + <Wiring N개> executable skeletons)

---
👉 `/implement docs/specs/<feature>_contract.json`
