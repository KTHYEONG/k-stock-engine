---
name: commit
description: Execute fast, automated git commits with 1-line rationale and optional multi-commit splitting.
---

# Commit Protocol

Fast automated git execution protocol enforcing atomic commits, bisect-safe change grouping, and a concise 1-line rationale (`Why:`) without redundant code verification loops.

## Directives

1. **Task-Scoped Working-Tree Staging**:
   - Inspect `git status --short` and stage ONLY the files verified as part of the current task/spec scope (`git add <file1> <file2> ...`).
   - Do NOT perform blind blanket staging (`git add .` or `git add -A`) when unrelated working tree modifications exist.
   - Do not stage ephemeral artifacts: `scratch/`, `tmp/`, `.pytest_cache/`, `*.pyc`, or logs.

2. **Atomic & Bisect-Safe Grouping**:
   - **Bisect-Safe Invariant**: Code changes (`src/`) and their associated tests (`tests/`) are committed together in the same `feat:` or `fix:` commit so tests pass at any checkout point.
   - **Boundary Separation**:
     - `feat:` / `fix:` / `refactor:`: Core logic (`src/`) + relevant tests (`tests/`).
     - `docs:`: Documentation updates (`docs/`, task indexes).
     - `chore:`: Tooling, configs, dependencies (`pyproject.toml`, `.agents/`).
   - Group changes into 1 atomic commit (or 2 when documentation/tooling warrants separation).

3. **Standard Message Format**:
   - **Subject**: `<type>: <concise description <= 50 chars>`
   - **Body**: `- **Why:** <concrete problem or business rationale solved>`
   - **Prohibited Metadata**: Do not add `Co-authored-by:`, AI model names, or session trailers. Keep commits clean with only subject and rationale.

## Output

Return ONLY the minimal summary card below:

### 📌 [COMMIT] COMPLETE

- **Commit**: `[<short_hash>]` <subject>
- **Summary**: <commit_count> commit(s) | <total_files_changed> file(s) changed
