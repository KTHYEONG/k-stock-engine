---
name: sync
description: Documentation Synchronization, ADR Logging, Cleanup.
---

# Sync Protocol

Post-development protocol for task finalization, ADR registration, index updating, and temporary artifact cleanup.

## Directives

1. **Task Sync & Index Registration**:
   - Run task sync script:
     ```bash
     uv run python tools/agent_skills/sync_task.py --task TASK_ID --title "<Title>" --why "<Context>" --what "<Resolution>" --impact "<Impact>" --source src/x.py --domain <domain>
     ```
   - Automatically updates `docs/decisions/task_index.json` and `docs/code_map.json`.

2. **Artifact Cleanup**:
   - Purge temporary `docs/specs/` files and `scratch/` test scripts.

## Output

Provide a clear, concise summary with emojis. Example:

### 🧹 [SYNC] <Task Title>

- **Status**: 🎉 COMPLETE
- **ADR Index**: <Registered ADR_ID>
