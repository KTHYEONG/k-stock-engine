---
name: commit
description: Execute fast, automated git commits with 1-line rationale and optional multi-commit splitting.
---

# Fast-Track Commit Protocol (Option B)

Ultra-fast automated git execution protocol. Includes a concise 1-line rationale (`Why:`) while avoiding heavy code verification loops and token waste.

## Directives

1. **Zero Working-Tree Modification (STRICT IMMUTABILITY)**:
   - Treat the current working tree state as 100% intentional by the user (As-Is).
   - NEVER modify, recreate, edit, or revert any files.
   - NEVER execute commands that alter the working tree state (e.g., `git restore`, `git checkout`, `git reset`, `git clean`, file writes/deletions).
   - Stage deleted (`D`), modified (`M`), and untracked (`??`) files exactly as they currently exist.

2. **Fast Inspection (No Full Diff / No Verification)**:
   - Inspect status using `git status --short`.
   - Do NOT read full diff or run linters (`ruff`), type checkers (`mypy`), or unit tests (`pytest`).

3. **Chained Execution**:
   - **Single Layer / Small Changes**: Stage and commit in a single chained shell command:
     ```bash
     git add -A && git commit -m "<type>: <Korean summary <= 50 chars>" -m "- **Why:** <Concise reason ending with 함.>" && git log -n 1 --oneline
     ```
   - **Multi-Layer / Large Changes**: If file paths cross distinct logical boundaries (e.g. `src/` vs `docs/` vs `tests/`), split by path boundaries and chain consecutive commits in one command:
     ```bash
     git add <files1> && git commit -m "<type1>: <summary1>" -m "- **Why:** <reason1>" && git add <files2> && git commit -m "<type2>: <summary2>" -m "- **Why:** <reason2>"
     ```
   - Do NOT output markdown approval drafts or lengthy reasoning.

3. **Message Standard**:
   - Subject: `<type>: <Korean summary <= 50 chars>`
   - Body: `- **Why:** <Concise 1-line reason ending with ~함.>`

## Output

Return ONLY the minimal summary card below:

### 📌 [COMMIT] COMPLETE

- **Commit**: `[<short_hash>]` <subject>
- **Summary**: <commit_count> commit(s) | <total_files_changed> file(s) changed