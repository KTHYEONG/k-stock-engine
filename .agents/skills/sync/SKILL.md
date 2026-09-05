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

2. **Artifact Cleanup & Memory Linkage**:
   - `sync_task.py` archives `docs/specs/*_contract.json` files to `docs/decisions/archive/<task_id>/` (preserving `design_rationale`/`performance_budget` for future `spec` reuse and audit traceability) and purges `scratch/` scripts, `tmp/` test roots, and logs. Persistent architecture documents (`docs/architecture/`, `00_architecture.md`) are safely preserved.
   - The archived contract's path is written back into the matching `task_index.json` entry as `archive_path`, and `spec_init.py` surfaces that path (not its content) on a future domain/keyword match — so a later `spec` run can `Read` the full prior design rationale on demand, without every retrieval paying for it upfront.

## Output

Provide a clear, concise summary with emojis. Example:

### 🧹 [SYNC] <Task Title>

- **Status**: 🎉 COMPLETE
- **ADR Index**: <Registered ADR_ID>
