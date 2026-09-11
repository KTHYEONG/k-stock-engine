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

Keep chat response ultra-compact, scannable, and contract-focused. Strictly avoid narrative walls of text, multi-line table cells (`<br>`), or repeating full code skeletons that already exist in `contract.json`.

**Output Directives:**
- **Terminal-Safe Tables**: Keep table cells to single-line values (no `<br>` or nested bullets).
- **Single Source of Truth**: Point directly to `docs/specs/<feature>_contract.json` for full skeletons and AST anchors.
- **Telegraphic Bullets**: Use concise, telegraphic bullets (명사형/종결형 축약, 최대 1-2줄).
- **Language Requirement**: All output rendered to the user MUST be written in Korean (한국어). Template titles and labels below MUST be presented in Korean as shown.

---

### 📐 [SPEC] <기능명>
> 📄 **계약 문서**: [`docs/specs/<feature>_contract.json`](file:///docs/specs/<feature>_contract.json)  
> 📊 **작업 규모**: <N>개 파일 · <N>개 변경점 · <N>개 배선 · <N>개 시나리오 (단위: <U>, 배선: <W>)  
> 🚦 **게이트 검증**: `lean_check --pre-impl` **PASS** (<N>/<N> AST 유효)

#### 1. 계획 요약 (Plan Summary)
- 🎯 **목표**: <구체화 대상 1줄 요약>
- ⚠️ **영향도/파괴적 변경**: <없음 또는 핵심 영향 1줄>
- 🚫 **범위 제외 (Out of Scope)**: <제외 또는 이연 항목 1줄>

#### 2. 변경 및 배선 매트릭스 (Changes & Wiring)
| 파일 경로 | 유형 | 대상 심볼 / 앵커 |
| :--- | :--- | :--- |
| `[<target_file>](file:///<target_file>)` | Target | `<symbol_1>`, `<symbol_2>` |
| `[<caller_file>](file:///<caller_file>)` | Wiring | `<anchor_symbol>` (호출부 주입) |

#### 3. 핵심 불변식 및 가드레일 (Invariants & Guardrails)
- 🛡️ **<INV-NAME>**: <Fail-Closed 조건 또는 경계 규칙 1줄 요약>
- 🚪 **<GATE-RULE>**: <파라미터 검증 또는 조기 중단 기준 1줄 요약>

#### 4. 검증 시나리오 (Verification Scenarios)
| 구분 | 건수 | 대상 테스트 스위트 | 주요 검증 초점 |
| :--- | :---: | :--- | :--- |
| **단위 (Unit)** | <U> | `[<test_unit_file>](file:///<test_unit_file>)` | `<정상 + 경계 + fail-closed 케이스>` |
| **배선 (Wiring)** | <W> | `[<test_caller_file>](file:///<test_caller_file>)` | `<호출부 통합 + 옵션 전달 케이스>` |

---
👉 다음 단계: `/implement docs/specs/<feature>_contract.json`
