---
name: spec
description: Produce a concise, evidence-based implementation blueprint and machine-readable contract.
---

# Spec Protocol

Produce an unambiguous implementation plan and precision contract (`contract.json`) optimized for mechanical, zero-search, zero-invention downstream execution by low-reasoning models (`implement`).

## High-Reasoning Allocation Philosophy

As the high-reasoning architect, spend your cognitive budget on **domain invariants, failure modes, and complete executable test harnesses**. Downstream `implement` models have low reasoning capacity and cannot extrapolate missing test fixtures or invent complex mock contexts. Your specification must be so mechanically complete that `implement` only pastes skeletons and turns them green.

## Directives

1. **Context & Verification**:
   - Collect domain references & historical ADRs:
     ```bash
     uv run python tools/agent_skills/spec_init.py --feature <feature_name> --domain <domain> --query <keyword>
     ```
   - Inspect target files, tests, AND immediate callers (1-depth call-sites) using `rg`/`view_file` to verify schemas, invariants, and type contracts.

2. **Ambiguity Gate & Assumptions**:
   - **Critical Gate**: If requirements leave core financial dynamics, trading risk, or public API breaking changes unstated, stop and ask clarifying questions.
   - **Autonomous Engineering**: For algorithmic/internal architecture details, state concrete **Assumptions & Invariants** in the Blueprint and proceed autonomously.
   - **No Silent Scope-Shrinking**: When a backtest/training run's real completion criterion (full date range, epoch count, convergence condition) is ambiguous, ask or state it as an explicit assumption — never default to a shorter run to make the spec "safer" or faster to satisfy. See `.agents/rules/performance.md` §0.

3. **Selective Empirical Proof**:
   - If algorithm correctness, numeric edge cases, or vectorization is uncertain, verify via a minimal script in `scratch/test_<topic>.py` using `uv run`.

4. **Deliverables (Single Source of Truth - `docs/specs/<feature>_contract.json`)**:
   - *No separate `.md` file*: All implementation specifications, boundary requirements, and tests live directly in `contract.json`.
   - `target_file`: Relative path to modify or create.
   - `context_files`: Minimal prerequisite paths for zero-search context loading.
   - `changes` (or `symbols`): Array of `{ name, signature, kind, target_file }`.
   - `wiring`: Array of `{ caller_file, anchor, import_symbol, invocation_expression }` ensuring entry-point hookup.
   - `requirements`: Explicit fail-closed boundary rules, invariant constraints, and complexity requirements.
     - **Fail-Fast by Design**: Mandate deterministic, fail-closed contracts. Forbid open-ended fallback catches (`try-except Exception`) or silent `None` swallows that create unreachable branches downstream.
   - `design_rationale`: `{ alternatives_considered, chosen_reason, failure_modes }` — record rejected alternatives, chosen rationale, and 2-3 concrete failure modes the contract must guard against.
   - `performance_budget` (required when `target_file` touches backtesting, ML training, or bulk data I/O): `{ expected_data_scale, memory_target_mb, storage_format, dtype_precision, chunking_strategy, acceleration_candidate }` — a *design target* for `implement` to size buffers/batches/dtypes against.
     - **Do NOT include a `timeout_s`, `max_iterations`, or any field that would truncate a backtest/training run early** — see `.agents/rules/performance.md` §0.
   - `scenarios`: Array of `{ scenario_id, target_test_file, execution_command, expected_behavior, test_skeleton }`.
     - **Dual-Scope Coverage Mandate (Unit + Wiring)**:
       1) **Unit Scenarios**: Scenarios exercising new symbols/functions in `target_file`.
       2) **Wiring Scenarios (MANDATORY when `wiring` modifies existing callers)**: If `wiring` touches a `caller_file` (e.g. pipeline, orchestrator, CLI), you MUST provide at least one scenario targeting that `caller_file`'s test suite (with all required mocks/fixtures 100% written out). Downstream low-reasoning models cannot invent mock pipelines — without wiring skeletons, `caller_file` edits will fail diff coverage.
     - `scenario_id`: Valid pytest function name (e.g. `test_<func>_<condition>`).
     - `test_skeleton`: **Mandatory 100% executable Python test function** (Given/When/Then, imports, actual call, concrete assertions). NEVER leave `pass`, `...`, or empty body. Every branch required by `requirements` and `failure_modes` must have an explicit test skeleton.

5. **Self-Validation Gate**:
   - Validate contract schema, test_skeleton AST syntax, and caller anchors:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json --pre-impl
     ```
   - **Low-Reasoning Execution Feasibility Check**: Re-read the contract through the eyes of a low-reasoning model:
     - Will pasting these `test_skeleton`s and implementing `changes` + `wiring` yield 100% diff coverage on BOTH `target_file` and `caller_file`?
     - Are there any hidden fixtures, unprovided mocks, or unspecified exception branches that would force downstream guesswork? If yes, resolve them in `contract.json` before publishing.
   - **Domain Principle Self-Check**: Before finalizing, re-read against `.agents/rules/performance.md`, `.agents/rules/quant.md`, and `.agents/rules/python.md`.

## Chat Output Format

Keep chat response concise, intuitive, and provide copy-pasteable execution command:

### 📐 [SPEC] <Task Title>
> **목표**: <1-line objective>

* **Before (현재)**
  * <현재 문제점 / 원인 또는 한계점 1-2줄>

* **After (개선)**
  * <개선 후 동작 / 해결 방식 및 기대효과 1-2줄>

* **Guards (방어 기준 & 불변식)**
  * <반드시 지켜져야 할 핵심 비즈니스 불변식 / Fail-closed 원칙>

---
👉 `/implement docs/specs/<feature>_contract.json`
