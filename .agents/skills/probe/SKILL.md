---
name: probe
description: Explore hypotheses, conduct empirical scratch experiments, and establish high-reasoning design rationale before contracting.
---

# Probe Protocol

High-reasoning exploratory protocol to formulate hypotheses, uncover failure modes, and empirically validate solutions via scratch experiments before freezing contracts.

## Allocation

Your cognitive budget is focused on **causal root-cause discovery, architectural trade-offs, and empirical stress testing**.
Focus on: *What is the fundamental causality behind this state? What competing architectures resolve it? Under what realistic boundary conditions does each approach break?*

## Directives

1. **Focused Context**:
   - Inspect the target module and its immediate 1-depth callers or fixtures using targeted `grep` / `view_file`.
   - Challenge historical assumptions when current empirical evidence contradicts past decisions.

2. **Empirical Probing**:
   - Create a lightweight probe script: `scratch/probe_<topic>.py`.
   - Run via `uv run python scratch/probe_<topic>.py`.
   - Measure real values, state transformations, or execution bottlenecks directly on actual or synthetic workloads.

3. **Invariants & Performance Budget**:
   - Define strict Fail-Closed invariants and domain boundaries.
   - If touching backtest, training, or bulk I/O, draft a realistic performance budget (memory, data scale, chunking).

4. **Seamless Transition**:
   - Diagnosis, invariants, and architectural decisions established here flow directly through the conversation context into `/spec`. Intermediate JSON files are not required.

## Chat Output Format

Keep chat response clear, intuitive for humans, and token-efficient. Use the structured summary card below:

### 🔬 [PROBE] <기능/토픽 제목>

- 🔍 **상황**: <어떤 결함이나 요구사항인지 직관적으로 1-2줄 요약>
- ⚙️ **근본 원인**: <데이터 흐름이나 시스템 제약상의 진짜 원인 1-2줄>
- 🛠️ **해결 방식**: <선택한 접근법과 핵심 기술적 근거 1-2줄>
- 🎯 **기대 효과**: <시스템과 사용자 관점의 개선 효과 1줄>
- ⚠️ **주의할 점**: <놓치기 쉬운 전제조건이나 경계 규칙 1-2줄>

---
👉 다음 단계: `/spec`
