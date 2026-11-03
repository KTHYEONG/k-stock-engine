---
paths:
  - "src/**"
  - "tests/**"
  - "docs/**"
---
# Documentation & Code Commenting Directives

> **Comments explain the non-obvious "why" behind contracts and constraints, never the mechanical "what"; keep code self-documenting.**

## 1. Commenting Principles
- **Self-Documenting Code & Rationale-Only Standard:** Names and strong types carry the "what"; near-zero inline comments, only for non-obvious trade-offs and external constraints. No paraphrase, procedural steps, or session metadata.

## 2. Interface Documentation
- **Caller Contracts, Not Spec Echoes:** Docstrings are 1-2 line caller contracts (inputs, returns, raised errors); never paste spec narratives, task metadata, or implementation steps.
- **Private Helpers:** Omit interface docs for private helpers unless their algorithm is complex or non-trivial.

## 3. Architecture Docs (`docs/architecture/`)
- **Stable Boundaries & Semantics:** Keep architecture docs on stable boundaries, data flows, formal models, and domain invariants, free of transient details; update them in place coherently, never append disconnected notes. Treat code-vs-doc discrepancies as issues to resolve; never assume either side is right.

## 4. Language
- **All Documentation & In-line Comments:** Concise Technical English.
