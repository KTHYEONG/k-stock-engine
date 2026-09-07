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
   - **Keep `--why`/`--what`/`--impact` to 1 sentence each.** These fields are echoed verbatim into every future `spec_init.py` match on this `domain`/keyword — a verbose entry today taxes every later spec run's context, not just this one. The script hard-caps each field at 300 chars as a backstop, but that's a truncation, not a substitute for writing tight in the first place.

2. **Artifact Cleanup**:
   - `sync_task.py` deletes `docs/specs/*_contract.json` files (design rationale already summarized into the `--why`/`--what`/`--impact` fields written to `task_index.json`) and purges `scratch/` scripts, `tmp/` test roots, and logs. Persistent architecture documents (`docs/architecture/`, `00_architecture.md`) are safely preserved.

## Output

Provide a clear, concise summary with emojis. Example:

### 🧹 [SYNC] <Task Title>

- **Status**: 🎉 COMPLETE
- **ADR Index**: <Registered ADR_ID>
