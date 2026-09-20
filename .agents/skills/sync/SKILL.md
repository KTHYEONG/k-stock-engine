---
name: sync
description: Documentation Synchronization, Compact Decision Logging, and Full Cleanup.
---

# Sync Protocol

Post-development protocol for task finalization, compact decision recording, and complete temporary artifact cleanup.

## Directives

1. **Auto-Inference & Task Sync**:
   - Do NOT ask the user to provide command arguments.
   - Automatically extract `title`, `why`, `what`, `impact`, and `caveat` from the active spec (`docs/specs/*_spec.md`) or recent audit context.
   - Run task sync:
     ```bash
     uv run python tools/agent_skills/sync_task.py --task TASK_ID --title "<Title>" --why "<Why>" --what "<What>" --impact "<Impact>" --caveat "<Caveat>" --domain <domain>
     ```
   - Automatically updates single ledger: `docs/decisions/task_index.json` and `docs/code_map.json`.
   - Keep each field strictly to 1 concise sentence to preserve token efficiency for future sessions.

2. **Complete Artifact Cleanup**:
   - `sync_task.py` completely purges all temporary files: `docs/specs/*_spec.md`, `scratch/` probe files, `tmp/` test roots, and logs.
   - Zero archive files are left behind, ensuring 100% clean Git history and zero file sprawl.

## Output

Keep chat output ultra-compact (1-2 lines). The user only needs confirmation that the record was committed and workspace is clean:

### 🧹 [SYNC] <Task Title>
- **기록 완료**: `task_index.json` (<ADR_ID>)
- **정리 완료**: 임시 스펙 및 `scratch/`, `tmp/` 완전 소각
