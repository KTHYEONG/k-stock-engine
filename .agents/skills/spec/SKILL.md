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
   - **Input from `/probe` (`scratch/probe_<feature>.json`)**:
     - Check if `scratch/probe_<feature>.json` exists. If present, load it as primary input.
     - Carry over `alternatives_considered`, `chosen_reason`, `failure_modes`, `invariants`, and `performance_budget` directly without re-probing or guessing.
   - Inspect target files and immediate callers (1-depth call-sites) to ensure signatures, imports, and AST anchors are exact.

2. **Ambiguity Gate & Invariant Specification**:
   - Translate probed invariants (`scratch/probe_<feature>.json` -> `invariants`) into explicit fail-closed requirements.
   - Forbid open-ended fallback catches (`try-except Exception`) or silent `None` swallows that create unreachable branches downstream.
   - **No Silent Scope-Shrinking**: Ensure full date ranges and required scales are preserved.

3. **Deliverables (Single Source of Truth - `docs/specs/<feature>_contract.json`)**:
   - `target_file`: Relative path to modify or create.
   - `context_files`: Minimal prerequisite paths for zero-search context loading.
   - `changes` (or `symbols`): Array of `{ name, signature, kind, target_file }`.
   - `wiring`: Array of `{ caller_file, anchor, import_symbol, invocation_expression }` ensuring entry-point hookup.
   - `requirements`: Explicit fail-closed boundary rules, invariant constraints, and complexity requirements.
   - `design_rationale`: `{ alternatives_considered, chosen_reason, failure_modes }` — carry over directly from `/probe` (`scratch/probe_<feature>.json`).
   - `performance_budget` (required when `target_file` touches backtesting, ML training, or bulk data I/O): `{ expected_data_scale, memory_target_mb, storage_format, dtype_precision, chunking_strategy, acceleration_candidate }` — carry over from `scratch/probe_<feature>.json` if present.
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
   - **Domain Principle Self-Check**: Before finalizing, cross-check against `.agents/rules/performance.md`, `.agents/rules/quant.md`, and `.agents/rules/code-style.md`.

## Chat Output Format

Keep chat response structured, scannable, and hierarchical using progressive disclosure (Executive Summary -> Invariants -> Changes & Wiring -> Verification). Avoid flat symbol-by-symbol tables or text walls.
**Language Requirement:** All instructions and template fields below are written in English, but the actual rendered chat response to the user MUST be translated and presented in Korean (한국어).

### 📐 [SPEC] <Feature Name>
> 📄 **Contract**: `docs/specs/<feature>_contract.json`  
> 📊 **Scale**: <N> files · <N> changes · <N> wiring · <N> scenarios

#### 1. Executive Summary
- **Purpose**: <Summary of core problem and why this change is necessary>
- **Key Changes**: <Core architectural and logic changes (1~3 concise points)>
- ⚠️ **Breaking Changes & Impact**: <Schema/API incompatibilities, or "None">

#### 2. Core Invariants & Guardrails
- 🛡️ **<Invariant Name>** (<Fail-Closed / Rule>): <Boundary condition and behavior>
- 🚪 **<Chokepoint / Contract>**: <Parameter rules, signature constraints>
- 🚫 **<Gate / Rejection Rule>**: <Early-abort criteria>

#### 3. Changes & Wiring Summary
| Category | Target Module / File | Core Logic & Wiring Details |
| :--- | :--- | :--- |
| **Target** | `[<target_file>](file:///<target_file>)` | • `<symbol_1>`: <Core logic change><br>• `<symbol_2>`: <Core logic change> |
| **Wiring** | `[<caller_file>](file:///<caller_file>)` | • `<anchor>`: <Caller wiring and parameter pass-through> |

#### 4. Verification Plan
| Scope | Count | Key Test Cases | Success Criteria |
| :--- | :---: | :--- | :--- |
| **Unit** | <N> | • <Normal / boundary cases><br>• <Error / fail-closed cases> | <Assertion or exception criterion> |
| **Wiring** | <N> | • <Caller / pipeline integration cases><br>• <CLI / option routing validation> | <Pipeline run / zero unhandled exception> |
| **Gate** | - | `lean_check --pre-impl` AST & Static Verification | **PASS** (<N>/<N> skeleton AST valid, 0 diagnostics) |

#### 5. Out of Scope
- <Deferred items and rationale, or "None">

---
👉 Next Step: `/implement docs/specs/<feature>_contract.json`



