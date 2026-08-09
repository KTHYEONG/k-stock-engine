---
name: commit
description: Analyze modifications and execute git commits directly.
---

# Commit Protocol

Automated git execution protocol. Supports multi-commit splitting with minimal token churn.

## Directives

1. **One-Shot Inspection**:
   - Inspect status to identify file boundaries: `git status --short`

2. **Auto-Splitting & Chained Execution**:
   - **Single Layer / Small Changes**: Stage and commit in one chained command:
     ```bash
     git add <files> && git commit -m "<Subject>" -m "- **Why:** <Reason>" -m "- **What:** <Details>"
     ```
   - **Multi-Layer / Large Changes**: If files cross logical boundaries (e.g. `src/` logic vs `docs/` specs vs `tools/` scripts), partition files into logical groups and execute consecutive chained commits.
   - Do NOT output markdown approval drafts.

3. **Message Standard (Korean & Noun-form)**:
   - Subject: `<type>: <Korean summary <= 50 chars>`
   - Body:
     - `- **Why:** <Reason ending with ~함.>`
     - `- **What:** <Details ending with ~함.>`

## Output

Return ONLY the summary card below:

### 📌 [COMMIT] <Scope Title>

- **Status**: COMPLETE
- **Commit**: `[<short_hash>]` <subject>
- **Summary**: <commit_count> commit(s) | <total_files_changed> file(s) changed
