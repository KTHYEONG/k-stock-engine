# Refactor roadmap (2026-09-25)

Goals: maintainability, extensibility, and AI-agent efficiency (fewer lines an agent must read to find the live path; one obvious place per concept).
Baseline: `src/` 42.8k lines, `tests/` ~42.7k lines, full suite green. Evidence: an import-reachability scan from the two CLIs and `tools/`, four read-only module audits, and on-disk data inspection.

## 1. Findings

### F1. Two generations of code coexist (≈40% of `src/` is legacy)
- **Live path:**
  - scoped layout `data/{bronze,silver,gold,state}/kr_swing_2019_v1`
  - per-dataset Silver builders (`*_silver.py`, `ordinary_universe`, `industry_silver`, `investor_flow_*`, `incremental_normalization`, `financial_quality`, `dividend_events`)
  - Gold `market_panel` and `reference_benchmarks`
  - the new engine `src/backtest/*`
- **Legacy path:**
  - the `SilverTable` layout (`data/*/stocks`, which no longer exists on disk), `streaming_normalization.py` (2,495 lines), `pipeline.py`, `replay.py`, `operations.py`, `gold.py`, `gold_loader.py`, `gold_artifacts.py`, `rebase.py`, `data_reset.py`, `legacy_inventory.py`
  - the old engine: `src/engine/*`, `src/data/backtest_{runner,sessions,exclusions,run_manifest}.py`, `src/strategy/*_strategy|portfolio|selection`, `src/validation/*`, `src/core/{costs,ledger,portfolio}.py`
- **Size:** about 17.5k lines in candidate modules, plus the legacy halves of `cli.py`, `collection.py`, `collection_plan.py`, `silver.py` and `normalization.py`.
- **CLI:** 19 `src/data/cli.py` subcommands are legacy-layout, 19 are live-scoped and 8 are maintenance. `backtest` is scoped but cannot work, because it needs a `scope_hash` that no manifest carries.
- **Legacy pieces still used by the live path:**
  - `strategy/{universe,scoring}` and `features/qvef`, used by the legacy Gold build.
  - The corporate-action audit functions in `backtest_sessions.py`, used by legacy `normalize`.
- **Stubs still wired in:** KRX `KindLifecycleCollector.search_notices` and `KrxHistoricalCollector.fetch_corporate_actions` always raise.
- **Test-only or dead code:** `src/execution/*` (no importer outside the package), `src/domain/`, `core/market_data.py`, `strategy/pipeline.py`.
- **Dead scripts:** `tools/collect_krx_master.py` and `tools/backfill_{financial_facts,investor_flow,investor_flow_parallel}.py`. They write to the non-existent `stocks` layout and bypass the catalog and quota ledger.

### F2. Bronze receipt catalog does not scale
- Every `persist_many` writes a full JSON snapshot. One revision is now 55 MB, and there are 489 revisions, 23 GB in total, which is half of Bronze.
- Cost per batch is O(total receipts). This is why the dividend collection ran at about 1 filing/s with 20-filing chunks.
- Readers load the whole 55 MB file to answer `latest(source, keys)`.

### F3. Dataset contract is inconsistent
- **Two identity schemes.** Hash16 datasets use `<kind>_<hash16>/manifest.json`. `financial_facts` and `financial_quality` use `<kind>/<sha64>/{dataset_manifest,content_manifest}.json`.
- **Mixed identity basis.** Some ids hash the inputs, others hash the output (`investor_flow` union, `dividend_events`).
- **Ambiguous prefix.** `investor_flow_` names both the LS-only dataset and the union dataset.
- **Dangling lineage.** The LS flow points to `ordinary_universe_eeb927f1…`, which is not on disk.
- **Layer inversion.** The Silver KIS supplement depends on the Gold `market_panel`.
- **No registry of current dataset ids.** Every build and backtest command takes ids by hand. The scope config binds none.
- **Wrong layer lookup.** `src/backtest/cli.py` looks for dividends under `gold/`, but `dividend_events_*` is in `silver/`, and it skips hash verification.
- **Duplicated boilerplate.** Staging, rename and compare-manifest code appears 9 times. Partition sha verification appears 5 times, and `_load_universe_sessions` twice (byte-identical).
- **No uniform checks.** Nothing re-hashes partitions, checks that lineage ids exist and validates schemas across all datasets. Hash16 builders record anomaly counters but no pass/fail.
- **Retention gap.** Superseded hash16 datasets are not covered by retention.

### F4. Configuration is scattered and already drifting
- **Paths:**
  - About 50 argparse defaults point at `data/*/stocks` and `data/artifacts`.
  - There are three artifact roots: `data/artifacts`, `data/bronze/artifacts` and `data/gold/artifacts`.
  - Tools hardcode `/home/kth/k-stock-engine/data`, because `build_workspace` requires an absolute path.
- **Provider policy:**
  - DART pacing and workers have 5 sources: scope TOML, pydantic defaults, `DartApiClient` defaults with env overrides (`OPENDART_REQUEST_MIN_INTERVAL_SECONDS`, `OPENDART_MAX_WORKERS`), `DartXbrlCollector(max_workers=20)` and the VPS `POLICY`.
  - The KCA avoid-window exists only in `tools/vps_dart/sync.py`.
  - `ResearchScope.content_hash` ignores `dart_extra_keys`.
- **Market rules:**
  - The tick tables appear 3 times: `core/costs.py` (pre-2023 only), `kis/client.py` (post-2023 only) and `config/market/krx_market_rules.toml`, which is the only correct one.
  - The fixed sell tax of 0.0023 in `core/costs.py` disagrees with the dated tax table in the TOML.
- **Periods:** research periods and the fiscal floor are restated in 6 or more places, and some of them disagree with the scope TOML.
- **Environment:**
  - `DART_API_KEY` and `KIWOM_*` aliases.
  - `load_dotenv()` runs at import time of `src/data/cli.py`.
  - `pydantic-settings` is a dependency but is never used.

### F5. Provider integration has no shared transport
- Each client re-implements its own session, pacing, retry, Retry-After handling and OAuth token caching. KIS retries 4xx and business errors, and Kiwoom has no retry or pacing.
- There are two error trees: DART, KRX and quota raise RuntimeError subclasses, while LS and Kiwoom raise `PITDataError` for transport failures.
- `PITDataError` is imported via `src.data.schemas` in 106 files and via `src.core.pit` in 35.
- Seven integrations import `src.data`, mostly to write Bronze from inside a collector.
- `DartXbrlCollector` is part factory and part proxy, and it calls the client's private methods. Tools reach into `_client`.
- `quota.py` is not safe across processes: it has no flock and uses a fixed temp name.
- Two DART HTML table parsers and two member decoders exist (`legacy_filing.py`, `dividend_decision.py`).

### F6. `tools/` holds production logic
- Tools import each other's private helpers through `sys.path` hacks (for example `_eligible_tickers`, `_ticker_by_corp_code` and `_pending_identities`).
- Four scripts each re-implement the same headroom → health check → chunk → persist loop.
- The JSON-line `_emit` exists 5 times, the corp-code bridge loader 5 times and the universe locator 4 times.

### F7. Tests and agent ergonomics
- **Layout:** tests only partly mirror `src`. There are duplicate pairs such as `test_instrument`/`test_instruments` and `kis/test_client`/`test_kis_client`, and coverage-filler files.
- **Big file:** `test_cli.py` has 3,189 lines, about 20 of which are old-engine tests.
- **No shared fakes:** there are no shared provider or HTTP fakes, and the only conftest just pins a temp directory.
- **lean_check mapping** points at the wrong test for some modules.
- **Suppressions:** 178 `pragma: no cover`, 139 of them in `src/data`, and 28 pinned layering violations.
- **Test fixture in production code:** `silver.complete_minimal_fixture` also contains a 09:00-UTC timestamp slip.

## 2. Target shape

```
src/
  config/            typed loaders: runtime (data_root, scope), providers, market rules, engine, strategy
  core/              pure domain: time/calendar, pit errors, instruments, market_rules (single source)
  integrations/
    transport.py     shared HTTP: session, pacing, Retry-After/backoff, error taxonomy, quota hook, token cache
    dart/ krx/ kis/ ls/ kiwoom/   provider adapters only (no Bronze writes)
  data/
    bronze/          catalog v2, scoped writer, document store
    collect/         provider jobs (dart_facts, dart_disclosures, dividend_decisions, krx_daily, investor_flow, industry)
                     + one budgeted job runner (headroom, health check, breaker, chunked persist)
    silver/          one module per dataset builder
    gold/            market_panel, reference_benchmarks
    datasets.py      hashed-dataset writer/verifier + registry
    cli/             command registry split by area (collect, build, verify, maintain)
  backtest/          new engine only
tools/               thin wrappers or none; agent_skills stays
```

## 3. Phases

Each phase is independently shippable, keeps the suite green and ends with a commit. Ordered by risk and payoff.

| Phase | Work | Payoff | Risk / guard |
|---|---|---|---|
| **R0 Baseline** | Tag `archive/pre-refactor`. Snapshot the current dataset ids into a registry file (see R4). Record the full-suite result. | Safe rollback | none |
| **R1 Remove dead code** | Delete `core/market_data.py`, `domain/`, `strategy/pipeline.py`, the dead `cli.py` helpers (`_run_backtest_from_silver`, `_execute_backtest`, `_build_sessions`) and the 4 dead tools. Fix `devops/clean_logs.py` (wrong logs dir). Point the 2 `domain` tests at `core.pit`. Prune the ratchet. | ~1.5k lines out | none; import scan verifies |
| **R2 Retire legacy stack** | (a) Move the live corporate-action audit functions out of `backtest_sessions.py` into `data/corporate_action_audit.py`. (b) Remove `run-backtest`, `backtest` and all legacy-layout subcommands, the old engine, strategies, `validation`, `core/{costs,ledger,portfolio}`, `streaming_normalization`, `pipeline`, `replay`, `operations`, legacy `gold*`, `rebase`/`data_reset`/`legacy_inventory`, the KRX stubs and their tests. `universe`/`scoring`/`qvef`: keep and move under `features/` only if a factor strategy will be ported; otherwise delete (**decision D1**). `execution/`: **decision D2**. | ≈15–17k `src` lines and a similar amount of tests out. The agent's search space roughly halves. | Large deletion, but only behind CLIs that no longer run. The guard is the tag plus the full suite. Blocked while any legacy command is in use; none is today. |
| **R3 Catalog v2** | Replace full-snapshot revisions with an append-only store: SQLite, or a JSONL delta log plus periodic compaction. Keep the `ReceiptCatalog` read/publish API. Keep the fcntl lock. Migrate from the latest revision, verify counts and hashes, then delete old revisions. | 23 GB → <1 GB. Per-batch publish becomes O(batch), which makes collection several times faster. | Must not run while collection is active. Migrate on a copy and diff `latest()` answers for every source. |
| **R4 Dataset contract & registry** | Add one `write_hashed_dataset`/`verify_hashed_dataset` with a uniform manifest (`dataset_id, kind, policy_version, inputs{name:id}, partitions[{path,rows,sha256}], checks{}, certification`). Migrate the 11 builders. Give `financial_facts`/`financial_quality` the flat `<kind>_<hash16>` naming (keeping the sha64 as a field). Rename LS flow to `investor_flow_ls_`. Registry `state/<scope>/datasets.json` holds current ids; builders update it and consumers (`build-*`, `src/backtest/cli.py`) default to it. Add a `verify-datasets` command (re-hash, lineage exists, schema, check thresholds). Add retention for superseded hash16 datasets. Base the KIS supplement on Silver sessions instead of the Gold panel. Read dividends from Silver. | One place to know "what is current", one verifier, no ids pasted by hand | Renames change ids, so rebuild or re-register once. Verified by `verify-datasets` equality of row content hashes. |
| **R5 Config consolidation** | `config/runtime.toml` (data_root relative to the repo, scope, artifact/log roots, session times, TZ). `config/providers.toml` (per-provider and per-key budget/reserve/pace/workers/chunk/avoid-windows/breaker, env var *names* only). `krx_market_rules.toml` as the only tick/tax/limit source. Delete the old tick tables and the KIS tick function. A typed `src/config` loader using `pydantic-settings` for secrets. Remove import-time `load_dotenv`, env overrides and aliases. The scope hash covers the provider policy. Add `config/strategy/equal_weight_liquid.toml`. | No drift and no absolute paths | The provider policy values must be copied exactly. A test asserts the loaded values equal today's effective values. |
| **R6 Shared transport** | `integrations/transport.py`: session, pacing (thread + optional process lock), Retry-After/backoff, a `ProviderError` tree (retryable / terminal / quota), quota ledger hook, OAuth token cache. Port DART → KRX → LS → KIS → Kiwoom. Collectors return pages; Bronze writes move to `data/collect`, which removes the 7 integration→data violations. Add a `DartCollector` Protocol. Add flock to the quota ledger. Merge the two DART HTML table parsers. | One retry/limit behaviour to reason about. Safer 24h collection. | Behaviour change in retries. Existing client tests plus a fake-transport contract suite. |
| **R7 Budgeted job runner** | `data/collect/job.py`: a generic headroom → health → breaker → chunk → persist loop. Port the DART fact extension, dividend decisions, the VPS worker and legacy recovery. Move the corp bridge loader, universe locator and JSON-line emitter into `src`. Tools become `python -m` wrappers or CLI subcommands, with no `sys.path` hacks. | A new collection job becomes a small declaration | Low |
| **R8 Split remaining large live files** | After R2, `cli.py` becomes a `cli/` package with a command registry and lazy imports. `collection.py` splits by provider into `data/collect/*`. The live half of `collection_plan.py` becomes `scoped_plan.py`. `silver.py` keeps only the loaders it still needs. | Files an agent can read whole | Mechanical moves only, with re-export shims removed in the same commit |
| **R9 Tests & agent docs** | Mirror `tests/` to `src/`. Merge the duplicate test pairs. Delete coverage-filler files. Add shared fakes (HTTP transport, Bronze/Silver fixtures) in conftest. Fix the `lean_check` test mapping. Get the layering ratchet to 0. Review `pragma: no cover`. Add a one-page `docs/architecture/data_flow.md` (datasets → builders → commands → registry) and regenerate `code_map.json`. | Faster, more accurate agent work | Low |

Recommended order: R0 → R1 → R2 → R3 → R4 → R5 → R6 → R7 → R8 → R9.
- **R3 may move earlier.** It is independent, and it pays off before the next large collection.
- **R5 and R6 can run in parallel** once R2 is done.

## 4. Kept as is (large but cohesive)
- `src/backtest/*`: new, cohesive, recently specified.
- `integrations/dart/xbrl.py` and `client.py`: they only shrink once R6 moves the transport out.
- `incremental_normalization.py` and `normalization.py`: these are the live fact path. After R2 removes the legacy `normalize_stock_evidence` half (≈250 lines), they stay as they are.

## 5. Decisions (2026-09-25)
- D1: delete the old factor strategies (champion/core/compounding) together with `universe`/`scoring`/`qvef` and the whole `features/`, `strategy/`, `validation/`, `engine/` packages.
- D2: keep `src/execution/*` for future live trading; decouple it from the deleted `core/{costs,ledger}` (portfolio snapshot moves into `execution/domain`).
- D3: receipt catalog v2 on SQLite.
- D4: delete legacy code outright; recovery via tag `archive/pre-refactor`.

## 6. Execution plan
| Step | Spec | Gate |
|---|---|---|
| R0+R1 | `docs/specs/refactor_r1_dead_code_spec.md` | tag `archive/pre-refactor` exists first |
| R2a | `docs/specs/refactor_r2a_backtest_stack_spec.md` | R1 committed |
| R2b | `docs/specs/refactor_r2b_legacy_data_spec.md` | R2a committed; kept commands reproduce current Gold ids |
| R3 | `docs/specs/refactor_r3_catalog_sqlite_spec.md` | no collection running; may run any time after R1 |
| R4a | `docs/specs/refactor_r4a_dataset_contract_spec.md` | R2b committed |
| R4b | `docs/specs/refactor_r4b_dataset_migration_spec.md` | R4a committed; `verify-datasets` green on real data |
| R5–R9 | specs written after R4b lands | their anchors (cli.py, collection.py, config readers) change substantially in R2–R4, so specifying them now would pin stale line anchors |

R5–R9 scope stays as in §3; each gets its own spec at that time, in the order R5 (config) → R6 (transport) → R7 (job runner + tools) → R8 (file splits) → R9 (tests/docs).
