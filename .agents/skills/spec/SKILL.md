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

4. **Deliverables**:
   - **Blueprint (`docs/specs/<feature>.md`)**:
     - **Diagnosis & Invariants**: Mathematical logic, structural invariants, and time/space complexity constraints.
     - **Architecture & Mitigation**: Layer separation, fail-closed handling, and caller wiring strategy.
     - **Execution Target**: Exact reproduction command (e.g., `uv run pytest <target_test_file> -k <scenario_id>`).
   
   - **Precision Contract (`docs/specs/<feature>_contract.json`)**:
     - `target_file`: Relative path to modify or create (e.g., `src/core/...`).
     - `context_files`: Array of prerequisite paths for direct zero-search context loading.
     - `symbols` (or `changes`): Array of `{ name, signature, kind }` covering all modified/added symbols.
     - `wiring`: Array of `{ caller_file, anchor, import_symbol, invocation_expression }` to ensure caller integration.
     - `requirements`: Explicit fail-closed boundary rules, complexity, and immutable output rules.
     - `scenarios`: Array of `{ scenario_id, target_test_file, execution_command, expected_behavior }` where `scenario_id` MUST be a valid, descriptive pytest function name (e.g. `test_<func>_<condition>`) and `expected_behavior` MUST include explicit predicates or quantitative thresholds (no vague descriptive phrases).

5. **Self-Validation Gate**:
   - Validate contract schema, paths, and caller anchor existence before finishing:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json --pre-impl
     ```

## Chat Output Format

Keep chat response concise and provide copy-pasteable execution command:

### 📐 [SPEC] <Task Title>
- **Goal**: <1-line objective>
- **Diagnosis**: `[Component]` -> <1-line root cause or bottleneck>
- **Core Invariant**: <1-line mathematical or structural rule>
- **Artifacts**: [`<feature>.md`](file:///docs/specs/<feature>.md), [`<feature>_contract.json`](file:///docs/specs/<feature>_contract.json)
- **Next Command**: `/implement docs/specs/<feature>_contract.json`
