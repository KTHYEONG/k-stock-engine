---
name: probe
description: Explore hypotheses, conduct empirical scratch experiments, and establish high-reasoning design rationale before contracting.
---

# Probe Protocol

High-reasoning exploratory protocol to formulate hypotheses, uncover failure modes, and empirically validate solutions via scratch experiments before freezing contracts.

## High-Reasoning Exploration Philosophy

As the high-reasoning architect, your cognitive budget here is 100% dedicated to **problem formulation, empirical validation, edge case discovery, and structural design choices**.
Do NOT write `contract.json` or full test suite skeletons in this phase.
Focus on: *Is the hypothesis sound? What does the real data/runtime look like? Which architecture handles failure modes best?*

## Directives

1. **Context & Prior ADR Lookup**:
   - Collect domain references and historical decisions:
     ```bash
     uv run python tools/agent_skills/spec_init.py --feature <feature_name> --domain <domain> --query <keyword>
     ```
   - Inspect existing schemas, calling pipelines, and invariant contracts using targeted `rg`/`view_file`.

2. **Autonomous Hypothesis & Exploration**:
   - Explore the solution space with full autonomy. Formulate hypotheses and compare design alternatives based strictly on problem complexity — whether a single verified hypothesis or multiple competing models.
   - Actively identify failure modes (e.g. division by zero, lookahead leak, missing corporate action adjustment, numerical instability, OOM on full window) without artificial ceilings.

3. **Mandatory Empirical Verification (Scratch Probing)**:
   - When numeric stability, data availability, schema compatibility, or runtime performance is non-trivial, run empirical probes:
     - Create a temporary probe script: `scratch/probe_<topic>.py`
     - Run via `uv run python scratch/probe_<topic>.py`
     - Measure real values, shape transformations, or execution bottlenecks directly on actual or synthetic datasets.
     - Never rely on speculative assumptions when an empirical command can prove or falsify them.

4. **Invariants & Performance Budget Formulation**:
   - Define strict Fail-Closed invariants and domain boundaries (.agents/rules/quant.md, performance.md, code-style.md).
   - If touching backtest, training, or bulk I/O, draft a realistic `performance_budget`:
     - `{ expected_data_scale, memory_target_mb, storage_format, dtype_precision, chunking_strategy }`
     - Do NOT introduce artificial truncation or shortened windows.

5. **Probe Summary Persistence & Transition Gate to `/spec`**:
   - To prevent context loss across agent sessions or token budget limits, **always persist probe conclusions** to `scratch/probe_<topic>.json`:
     ```json
     {
       "feature": "<feature_name>",
       "domain": "<domain>",
       "hypothesis": "<validated core hypothesis>",
       "empirical_proof": {
         "script": "scratch/probe_<topic>.py",
         "summary": "<real measurement/benchmark result>"
       },
       "alternatives_considered": ["<alt 1>", "<alt 2>"],
       "chosen_reason": "<why the selected architecture was chosen>",
       "failure_modes": ["<failure mode 1>", "<failure mode 2>"],
       "invariants": ["<fail-closed rule 1>", "<invariant rule 2>"],
       "performance_budget": null
     }
     ```
   - Once persisted, hand off directly to `/spec` which will consume `scratch/probe_<topic>.json` as input.

## Chat Output Format

Keep chat response ultra-compact, scannable, and evidence-focused. Strictly avoid conversational prose and narrative walls of text.
Detailed logs, benchmark payloads, and raw traces MUST be dumped to `scratch/probe_<topic>.json` and referenced via link, not pasted into chat.

**Output Directives:**
- **Terminal-Safe Tables**: Never put multiline descriptions or long code snippets inside Markdown tables. Keep columns short (`검증 항목 / 대상`, `실증 측정치 / 근거`, `판정`).
- **Concise Bullet Points**: Use concise, telegraphic bullets (명사형/종결형 축약, 최대 1-2줄).
- **Zero Redundancy**: Do not repeat explanations across Summary, Matrix, and Traps.
- **Language Requirement**: All output rendered to the user MUST be written in Korean (한국어). Template titles and labels below MUST be presented in Korean as shown.

---

### 🔬 [PROBE] <기능/토픽 제목>

#### 1. 판정 요약 (Triage)
- 🎯 **핵심 결론**: <발견된 근본 원인 및 채택 방향 1줄 요약>
- 📦 **작업 범위**: <포함 대상 및 제외/백로그 대상 명시>

#### 2. 실증 검증 매트릭스 (Empirical Matrix)
> 📁 상세 로그/페이로드: [`scratch/probe_<topic>.json`](file:///scratch/probe_<topic>.json)

| 검증 항목 / 대상 | 실증 측정치 / 근거 | 판정 (Verdict) |
| :--- | :--- | :--- |
| `<모듈 or 이슈>` | `<측정값, 실패 라인, exit code 등 컴팩트한 근거>` | `CONFIRMED / REJECTED / BUG` |

#### 3. 핵심 불변식 (Invariants & Boundaries)
- 🛡️ **<INV-NAME>**: <Fail-Closed 조건 또는 경계 규칙 1줄 요약>
- 🧩 **<STATE-RULE>**: <상태 전이 또는 스키마 규약 1줄 요약>

#### 4. 구현 함정 및 주의사항 (Traps)
*필요한 경우에만 최대 2개 이하의 콜아웃 박스 사용. 요약/매트릭스에 적은 내용 중복 금지.*
> [!CRITICAL]
> **<핵심 위험/주의 제목>**: <구현 시 주의점 또는 회귀 경고 1줄 요약>

---
👉 다음 단계: `/spec --feature <feature_name> --domain <domain>`
