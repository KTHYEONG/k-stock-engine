---
name: commit
description: Execute fast, automated git commits with 1-line rationale and optional multi-commit splitting.
---

# Fast-Track Commit Protocol (Option B - Production Standard)

Ultra-fast automated git execution protocol. Enforces atomic commits, bisect-safe change grouping, and a concise 1-line rationale (`Why:`) while avoiding heavy code verification loops.

## Directives

1. **Zero Working-Tree Modification (STRICT IMMUTABILITY)**:
   - Treat the current working tree state as 100% intentional by the user (As-Is).
   - NEVER modify, recreate, edit, or revert any files.
   - NEVER execute commands that alter the working tree state (e.g., `git restore`, `git checkout`, `git reset`, `git clean`, file writes/deletions).
   - Stage deleted (`D`), modified (`M`), and untracked (`??`) files exactly as they currently exist.

2. **Exclusion of Ephemeral Paths (Zero Scratch/Tmp Commit)**:
   - NEVER stage ephemeral artifacts: `scratch/`, `tmp/`, `.pytest_cache/`, `*.pyc`, logs, or scratch dumps.
   - Stage by explicit paths or directories. Do NOT use blind `git add -A` when `scratch/` or `tmp/` files are present in `git status --short`.

3. **Atomic & Bisect-Safe Commit Grouping**:
   - **Bisect-Safe Invariant**: Code changes (`src/`) and their associated unit/integration tests (`tests/`) MUST be committed together in the same `feat:` or `fix:` commit so tests never break at checkout.
   - **Boundary Separation**:
     - `feat:` / `fix:` / `refactor:`: Core logic (`src/`) + relevant test updates (`tests/`).
     - `docs:`: Documentation only (`docs/`, `*.md`, task indexes).
     - `chore:`: Tooling, configs, dependencies (`pyproject.toml`, `.github/`, `.agents/`).
     - `test:`: Pure test additions/refactoring without functional production code change.
   - **Chained Multi-Commit**:
     ```bash
     git add src/ tests/ && git commit -m "<type>: <Korean summary <= 50 chars>" -m "- **Why:** <Problem or concrete objective solved, ending with ~함.>" && git add docs/ && git commit -m "docs: <summary>" -m "- **Why:** <reason>" && git log -n 2 --oneline
     ```

4. **Message Standard (Anti-Tautology & High-Signal Rationale)**:
   - **Subject**: `<type>: <Korean summary <= 50 chars>` (Imperative, clear target)
   - **Body**: `- **Why:** <Specific business/technical reason ending with ~함.>`
   - **Prohibited Tautologies**: Avoid generic tautologies like "정합성을 확보함", "구조를 반영함", "회귀 방지 범위를 확보함". State **what problem was solved** or **what business requirement was met** (e.g., "동시호가 체결가 괴리율 완화 및 슬리피지 과소평가 보정을 위함.").

## Output

Return ONLY the minimal summary card below:

### 📌 [COMMIT] COMPLETE

- **Commit**: `[<short_hash>]` <subject>
- **Summary**: <commit_count> commit(s) | <total_files_changed> file(s) changed
