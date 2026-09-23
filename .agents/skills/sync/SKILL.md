---
name: sync
description: Documentation Synchronization, Compact Decision Logging, and Full Cleanup.
---

# Sync Protocol

Post-development protocol for task finalization, compact decision recording, and complete temporary artifact cleanup.

## Directives

1. **Auto-Inference & Task Sync**:
   - Do NOT ask the user to provide command arguments.
   - Automatically extract `title`, `why`, `what`, `impact`, `caveat`, `source`, and `test` from the active spec (`docs/specs/*_spec.md`) or recent audit context.
   - Run task sync:
     ```bash
     uv run python tools/agent_skills/sync_task.py --task TASK_ID --title "<Title>" --why "<Why>" --what "<What>" --impact "<Impact>" --caveat "<Caveat>" --domain <domain> --source <source_file> --test <test_file>
     ```
   - Automatically updates single ledger: `docs/decisions/task_index.json` and `docs/code_map.json`.
   - Keep each field strictly to 1 concise sentence to preserve token efficiency for future sessions.

2. **Targeted Artifact Cleanup**:
   - `sync_task.py` purges ONLY the completed task's matching temporary spec (`docs/specs/*<TASK_ID>*_spec.md`), task-associated `scratch/` probe files, and task-associated `tmp/` test roots, while strictly preserving all other specs, scratch files, and persistent logs.
   - Updates decision records ready for the downstream commit phase without leaving temporary task clutter.

## Output

Keep chat output ultra-compact (1-2 lines). Retain English keys/badges while confirming status:

### 🧹 [SYNC] <Task Title>
- **Registry Updated**: `task_index.json` (<ADR_ID>) / `code_map.json`
- **Cleanup Completed**: 태스크 임시 스펙 및 scratch 산출물 정리 완료 (커밋 대기)
