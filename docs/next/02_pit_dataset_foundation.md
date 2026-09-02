# 02 Point-in-Time Dataset Foundation

## Outcome

Import retained evidence into immutable Bronze and materialize certified Silver
calendar, security master, daily market, financial facts, investor flow,
corporate actions, disclosures, and historical costs.

## Dependencies

- Complete `01_domain_ledger_clock`.
- Approve and perform the removal set in `08_data_reuse_boundary.md`.
- Acquire the missing evidence listed in that document.

## Planned production scope

- Add stock schemas and PIT snapshot contracts under `src/data/`.
- Reuse `DatasetManifest`, `PointInTime`, `SessionCalendar`, and
  `ParquetDatasetStore`; do not create replacement manifest or storage types.
- Add deterministic evidence importers for each retained JSON source.
- Add content-addressed `data/bronze`, `data/silver`, and dataset manifests.
- Add certification reports for keys, coverage, hashes, and temporal ordering.

## Invariants

- Bronze writes are append-only and preserve source bytes and retrieval metadata.
- Silver primary keys are unique and all facts satisfy `available_at <= decision_time`.
- DART facts without proven intraday availability start on the next KRX session.
- Unknown actions, missing required sources, or incomplete declared coverage fail closed.
- No strategy consumes `data/evidence`, `canonical`, `derived`, or legacy catalogs directly.

## Verification boundary

- Unit tests: schema validation, timestamp ordering, DART delay, hash mismatch, duplicate keys.
- Integration tests: minimal retained-evidence import and bounded PIT snapshot.
- Command: `uv run pytest tests/unit/data tests/integration/data -q`.
- Exit evidence: one research-certified minimal Silver fixture and one deliberate certification failure.
