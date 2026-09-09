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

Keep chat response structured, scannable, and evidence-focused using tables and clear bullet points (avoid dense wall-of-text paragraphs).
**Language Requirement:** All instructions and template fields below are written in English, but the actual rendered chat response to the user MUST be translated and presented in Korean (한국어) for intuitive review.

### 🔬 [PROBE] <Feature/Topic Title>

#### 1. 설계 결정 (Architecture)
| 구분 (Category) | 내용 (Details) |
| :--- | :--- |
| **채택 설계 (Chosen Architecture)** | <Summary of the selected approach in Korean> |
| **격리/범위 (Scope & Isolation)** | <Preservation of existing hot-paths or scope boundary in Korean> |
| **선택 이유 (Rationale)** | <Core rationale for choosing this approach over alternatives in Korean> |

#### 2. 실측 검증 (Empirical Benchmark)
- **스크립트 (Script)**: `scratch/probe_<topic>.py` (`.json` recorded)
- **실측 성능/처리량 (Throughput & Latency)**: <Observed benchmark figures in Korean>
- **검증 & 결함 검출 (Verification)**: <Tested edge cases and defect detection in Korean>
- **발견된 버그 & 사전 수정 (Preempted Bugs)**: <Bugs identified and fixed during probing in Korean>

#### 3. 핵심 불변식 & 주의점 (Invariants & Risks)
- **핵심 불변식 (Invariants)**: <Fail-closed rules, conservation laws, or critical boundaries in Korean>
- **다운스트림 주의 (Downstream Risks)**: <Dependencies, edge cases, or out-of-scope notes for spec/implement in Korean>

---
👉 다음 단계: `/spec --feature <feature_name> --domain <domain>`



