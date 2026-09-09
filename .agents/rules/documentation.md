---
trigger:
  - on_file_path_regex: "src/.*\\.py"
  - on_file_path_regex: "docs/.*\\.md"
priority: 8
---

# Documentation & Code Commenting Directives

> **Explain the non-obvious "Why" behind domain logic, math, and constraints, not the mechanical "What" visible in code. Keep comments and docs focused on stable architecture, invariants, and trade-offs. Never embed ephemeral spec paths or AI session logs into persistent code.**

## 1. Commenting Principles: Explain "Why", Not "What"
- **Focus on Rationale:** Document business context, mathematical derivations, domain constraints, timing semantics, units, and non-obvious trade-offs. Omit comments that merely paraphrase readable code.
- **Conciseness with Clarity:** Keep comments focused and compact. Use extended explanations only when the rationale cannot be clearly expressed briefly.
- **No Ephemeral Spec References:** NEVER reference temporary `docs/specs/*.md` or `contract.json` paths in code, docstrings, or comments. Specs are transient working files that get purged. Use persistent `ADR-XXXX` IDs or self-contained domain rationale.
- **No Diagnostic or Session Artifacts:** Never leave AI task logs, revision chronicles, or linter fix annotations (e.g., `# fix mypy error`) in production code or comments.

## 2. Docstring Standards
- **Contract-Driven Documentation:** Document public interfaces when behavior, side effects, units, timing assumptions, or failure conditions are not self-evident from type signatures and function names.
- **Consistent Convention:** When a structured docstring is required, follow the repository's Google-style convention (`Args:`, `Returns:`, `Raises:`). Avoid conversational prose or historical journaling.
- **Private Helpers:** Omit docstrings for private helper functions unless the algorithmic logic or mathematical derivation is non-trivial.

## 3. Architecture & Domain Documentation (`docs/architecture/`)
- **Stable Boundaries & Semantics:** Keep architecture documents focused on stable system boundaries, data flows, formal mathematical models, and domain semantics. Avoid cluttering with transient implementation details.
- **Coherent Structure:** Update existing tables, schemas, and topology diagrams coherently rather than accumulating disconnected append-only notes.
- **Contract & Spec Reconciliation:** Treat discrepancies between code contracts and architecture documentation as issues to investigate and resolve; never silently assume either side is automatically correct.

## 4. Repository Language Conventions
- **Docstrings & External Documentation:** Written in English for toolchain and global standard compatibility.
- **In-line Comments (`#`):** Korean preferred for immediate intuition and clarity among local maintainers.
