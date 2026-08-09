---
name: spec
description: Produce a concise, evidence-based implementation blueprint and machine-readable contract.
---

# Spec Protocol

Frontier reasoning model protocol for specification engineering and architecture design. Uses high-reasoning autonomy to solve design ambiguities, establish empirical proofs, and produce **precision contracts executable by low-reasoning models without architectural guesswork**.

## Core Directives

1. **Context Aggregation & Initial Inspection**:
   - Run context aggregator helper to quickly collect domain ADRs and code map references:
     ```bash
     uv run python tools/agent_skills/spec_init.py --feature <feature_name> --domain <domain> --query <keyword>
     ```
   - **MANDATORY DEEP INSPECTION**: Do NOT rely solely on the aggregated summary. Perform direct codebase searches (`rg`) and file inspections (`view_file`) on target sources, callers, and tests to verify actual signatures, line numbers, and behavior.

2. **Ambiguity & Alignment Check**:
   - Assess if requirements contain open design choices, ambiguous quant trade-offs, or unstated boundary conditions.
   - If ambiguity exists, ask concise clarification questions. Proceed immediately if clear.

3. **Selective Empirical Sandbox**:
   - For complex algorithms, novel quantitative models, or uncertain performance-critical logic, write temporary scripts in `scratch/test_<topic>.py` and verify metrics via `uv run`.
   - Skip sandbox experimentation for straightforward refactoring, simple bug fixes, or minor API additions.

4. **Precision Contract Specification (`contract.json`)**:
   - High-reasoning model MUST produce a precise, deterministic contract so low-reasoning execution models can implement it mechanically.
   - Emit `docs/specs/<feature>_contract.json` with explicit declarations:
     - `target_file`: Absolute path to modify or create.
     - `symbol` & `signature`: Full Python type-hinted signature without parenthetical hints.
     - `python_assertion`: Directly executable Python assertion expression (e.g. `assert fee_calc(100) == 0.05`).
     - `requirements`: Explicit fail-closed exception types (e.g., raise `ValueError` on bad inputs) and performance/vectorization constraints (e.g., vectorized NumPy without `pd.apply`).
     - `scenarios`: Array of `{ scenario_id, target_test_file, expected_behavior }`.
     - `wiring`: Object or array containing explicit wiring declarations:
       - `caller_file`: Absolute path of the integration caller file.
       - `anchor`: Concrete line pattern, function signature, or class symbol in caller file.
       - `import_symbol`: Exact symbol name or import statement to be imported.
       - `invocation_expression`: Exact invocation snippet to be wired into caller.
   - Create main design blueprint in `docs/specs/<feature>.md`.

5. **Contract Self-Validation Gate**:
   - Immediately verify target paths, anchor symbols, and JSON syntax of the newly created contract:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec-only --spec docs/specs/<feature>_contract.json
     ```
   - If spec-only validation fails due to invalid target file paths or non-existent caller anchors, fix `contract.json` before completing the `/spec` phase.


## Output & Chat Notification Directives

Keep the chat response extremely concise (max 4-5 bullet points total, under 80 tokens).
NEVER dump detailed failure histories, nested function signatures, or multi-paragraph background analyses into the chat window. All deep rationale MUST reside strictly inside `docs/specs/<feature>.md`.

### Format Rule:
- **Goal**: 1-line core objective.
- **Diagnosis**: 1-line root cause in `[Component] -> [Bottleneck]` format.
- **Solution**: 1-line key architectural action.
- **Artifacts**: Clickable markdown links to the spec and contract files.

### 📐 [SPEC] <Task Title>

- **Goal**: <Core objective in 1 line>
- **Diagnosis**: `[Component]` -> `<Short bottleneck description in 1 line>`
- **Solution**: `<Key architecture change>` (Success: `<1-line verification criteria>`)
- **Artifacts**:
  - 📄 Specification: [`<feature>.md`](file:///docs/specs/<feature>.md)
  - ⚙️ Contract: [`<feature>_contract.json`](file:///docs/specs/<feature>_contract.json)
