---
name: spec
description: Produce a concise, evidence-based implementation blueprint and machine-readable contract.
---

# Spec Protocol

Produce an unambiguous implementation plan and precision contract (`contract.json`) for downstream mechanical execution.

## Directives

1. **Context & Verification**:
   - Collect domain references:
     ```bash
     uv run python tools/agent_skills/spec_init.py --feature <feature_name> --domain <domain> --query <keyword>
     ```
   - Directly inspect target files and tests (`rg`, `view_file`) to verify exact signatures and behavior.

2. **Ambiguity Gate**:
   - If requirements leave critical architectural ambiguities or domain trade-offs unstated, ask the user clarifying questions. Otherwise, proceed autonomously.

3. **Selective Empirical Proof**:
   - If algorithm correctness, performance, or vectorization is uncertain, verify via a temporary script in `scratch/test_<topic>.py` using `uv run`.

4. **Deliverables**:
   - **Blueprint (`docs/specs/<feature>.md`)**: Core architecture, rationale, and failure-mode mitigations.
   - **Precision Contract (`docs/specs/<feature>_contract.json`)**:
     - `target_file`: Absolute path to modify or create.
     - `context_files`: Array of relative/absolute paths to prerequisite definitions, models, or utilities for direct zero-search context loading.
     - `symbol` & `signature`: Full type-hinted signature.
     - `scenarios`: Array of `{ scenario_id, target_test_file, expected_behavior }` (include edge/boundary cases).
     - `requirements`: Explicit fail-closed exceptions, vectorization, or performance rules.
     - `wiring`: Declarative caller hook (`caller_file`, `anchor` with exact target function/class name, `import_symbol`, `invocation_expression`).

5. **Self-Validation Gate**:
   - Verify contract integrity before completing:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec-only --spec docs/specs/<feature>_contract.json
     ```

## Chat Output Format

Keep chat response concise and provide copy-pasteable execution command:

### 📐 [SPEC] <Task Title>
- **Goal**: <1-line objective>
- **Diagnosis**: `[Component]` -> <1-line bottleneck>
- **Solution**: <1-line architecture decision>
- **Artifacts**: [`<feature>.md`](file:///docs/specs/<feature>.md), [`<feature>_contract.json`](file:///docs/specs/<feature>_contract.json)
- **Next Command**: `/implement docs/specs/<feature>_contract.json`
