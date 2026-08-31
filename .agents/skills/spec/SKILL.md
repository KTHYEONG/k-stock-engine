---
name: spec
description: Produce a concise, evidence-based implementation blueprint and machine-readable contract.
---

# Spec Protocol

Produce an unambiguous implementation plan and precision contract (`contract.json`) optimized for mechanical, zero-search downstream execution (`implement`).

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

3. **Selective Empirical Proof**:
   - If algorithm correctness, numeric edge cases, or vectorization is uncertain, verify via a minimal script in `scratch/test_<topic>.py` using `uv run`.

4. **Deliverables (Single Source of Truth)**:
   - **Precision Contract (`docs/specs/<feature>_contract.json`) ONLY**:
     - *No separate `.md` file*: All implementation specifications, boundary requirements, and tests live directly in `contract.json`.
     - `target_file`: Relative path to modify or create.
     - `context_files`: Minimal prerequisite paths for zero-search context loading.
     - `changes` (or `symbols`): Array of `{ name, signature, kind, target_file }`.
     - `wiring`: Array of `{ caller_file, anchor, import_symbol, invocation_expression }` ensuring entry-point hookup.
     - `requirements`: Explicit fail-closed boundary rules, invariant constraints, and complexity requirements.
     - `scenarios`: Array of `{ scenario_id, target_test_file, execution_command, expected_behavior, test_skeleton }`.
       - `scenario_id`: Valid pytest function name (e.g. `test_<func>_<condition>`).
       - `test_skeleton`: **Mandatory 100% executable Python test function** (Given/When/Then, imports, actual call, concrete assertions). NEVER leave `pass`, `...`, or empty body. This enables 0-reasoning mechanical pasting by downstream low-cost models.

5. **Self-Validation Gate**:
   - Validate contract schema, test_skeleton AST syntax, and caller anchors:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json --pre-impl
     ```

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
