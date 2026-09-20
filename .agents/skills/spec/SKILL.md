---
name: spec
description: Produce a machine-readable, zero-invention implementation blueprint (Markdown Spec) from probed architecture rationale.
---

# Spec Protocol

Produce an unambiguous implementation blueprint (`docs/specs/<feature>_spec.md`) optimized for mechanical downstream execution.

## Allocation

The design rationale and empirical proof are resolved in `/probe`.
Your output in `/spec` focuses strictly on:
1. **Target Interfaces & Invariants**: Exact signatures, finalized production docstrings (explaining Why and constraints), and bulleted core invariants. Do not include temporary pseudo-code, code skeletons, or recipes in docstrings or invariants (prevents model anchoring and scaffolding leaks).
2. **Wiring Points**: Caller anchor and invocation snippet.
3. **Invariant Verification Scenarios**: Concrete scenarios (Given / When / Invariant) validating contracts without freezing internal test syntax prematurely.

## Directives

1. **Context Alignment**:
   - Infer feature context, invariants, and decisions directly from the preceding probe step.
   - Inspect target files and immediate callers for exact imports and AST anchors without scanning unrelated paths.

2. **Sizing & Modularity (Single-Bolt Rule)**:
   - **Sweet Spot**: A single spec typically targets 2–4 related production files, 1–2 test files, and 10–15 invariant scenarios (200–350 lines).
   - **Mandatory Split**: Split into sequential sub-specs (e.g. ordered by layer or dependency: `<feature>_<sub_layer>_spec.md`) if:
     1) Total invariant scenarios exceed 20 or spec exceeds 500 lines.
     2) Targets span distinct architectural layers (e.g., Domain Core Engine vs. CLI / External Adapters).

3. **Blueprint Structure (`docs/specs/<feature>_spec.md`)**:
   Organize targets with **component-centric co-location** (group Target, Wiring, and Invariant Scenarios together per component to prevent cross-referencing context loss). For multi-component specs, repeat this 3-part sequence for each component unit:

   ### A. Target Blueprint (`## Target: <relative_path>`)
   - Function/class signature with finalized production docstring (domain context, Args, Returns, Raises).
   - Core Invariants: bulleted preconditions, calculation sequence, negative constraints ("Do NOT..."), edge case handling, and postconditions.
   - **No Code Skeletons**: Specify strict types and invariants only; do not provide implementation code snippets or algorithms to avoid model overfitting.

   ### B. Wiring Blueprint (`## Wiring: <caller_file>`)
   - Anchor point (`- Anchor: <symbol_or_line>`) and invocation snippet.

   ### C. Invariant Scenarios (`## Invariant Scenarios: <target_test_file>`)
   - Bulleted test scenarios with descriptive domain titles:
     - `- **<Behavior/Invariant Name>** Given ...; When ...; Invariant: ...`
     - Keep titles clear so they translate directly into clean test function names (`test_<target>_<behavior>`).

## Chat Output Format

Keep chat output ultra-compact and token-efficient. Do NOT execute CLI commands or pre-impl validation loops.
Output only the minimal card below:

### 📐 [SPEC] <기능명>
> 📄 **청사진**: [`docs/specs/<feature>_spec.md`](file:///docs/specs/<feature>_spec.md)

- 🎯 **작업 요약**: <구현할 핵심 기능 1줄 요약> (<N>개 파일 대상 · <N>개 시나리오)

*(단일 스펙인 경우)*:
```bash
/implement docs/specs/<feature>_spec.md
```

*(다중 스펙으로 분할된 경우: 순차 실행할 /implement 명령어를 복사하기 쉽게 나열)*:
```bash
/implement docs/specs/<sub_spec_1>.md
```
```bash
/implement docs/specs/<sub_spec_2>.md
```
