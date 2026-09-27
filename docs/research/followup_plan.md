# Follow-up plan after refactor R1–R4 (2026-09-25)

State at start:
- `src/` is ~17.7k lines. The legacy stacks are gone and the receipt catalog is on SQLite.
- The dataset registry lists 11 verified datasets. The backtest runs end to end on real data.
- The uncommitted backtest fixes (dividend window filter, undefined-volatility reject) are part of F0.

## Order and gates

| Step | Spec | Depends on | Gate to start |
|---|---|---|---|
| F0 | `docs/specs/followup_f0_cleanup_spec.md` | – | none |
| R5 | `docs/specs/followup_r5_config_spec.md` | F0 | no collection job running |
| R6 | `docs/specs/followup_r6_transport_spec.md` | R5 | – |
| R7a | `docs/specs/followup_r7a_dart_jobs_spec.md` | R6 | – |
| R7d | `docs/specs/followup_r7d_dart_document_parser_spec.md` | R7a | – |
| R7b | `docs/specs/followup_r7b_krx_ls_collection_spec.md` | R7d | – |
| R7c | `docs/specs/followup_r7c_refresh_pipeline_spec.md` | R7b | – |
| R8 | `docs/specs/followup_r8_cli_split_spec.md` | R7c | – |
| R9 | `docs/specs/followup_r9_tests_docs_spec.md` | R8 | – |

Rules that apply to every step:
- **Memory safety:** every real-data run executes under a hard memory cap, `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 <cmd>` (or `prlimit --as=4294967296` where systemd is unavailable), and reports its peak RSS (`/usr/bin/time -v`). A step whose design needs more than the cap is redesigned to stream. Loading every Bronze page or every fact page into memory at once is forbidden. A past analysis script was OOM-killed that way, and the legacy-recovery dry run reached about 17 GB.
- The full suite and `lean_check.py` must be green.
- `verify-datasets` must pass on real data before and after the step.
- The registry's current ids must not change unless a spec says so. When one does change, the spec states the reason and the rebuild evidence.
- Nothing calls a provider API during tests. Real-data API calls happen only in the steps that explicitly schedule a pilot.

## Not in this plan (research or deferred decisions)

- **KIS supplement depends on the Gold `market_panel` (layer inversion).** Fixing it changes content, so it needs its own probe.
- **The baseline strategy (`equal_weight_liquid`) performs poorly, and capital-size scaling of `max_names` is unsolved.** This is strategy research, not infrastructure.
- **The 2026 forward-window data collection.** It becomes a routine run of `refresh-scope` once R7c lands.
