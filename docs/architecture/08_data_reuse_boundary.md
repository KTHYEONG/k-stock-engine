# Legacy Data Reuse Boundary

## Decision rule

Legacy data is reusable only when it contributes irreplaceable source evidence
to a required Champion v1 Silver table and carries enough provenance to be
revalidated. A dataset is not reused merely because it is expensive to rebuild.

Generated features, labels, provisional panels, experiment outputs, active
pointers, and mutable runtime state never cross into the new design.

## Retained import evidence

These files remain under `data/evidence/stocks/` as import-only inputs. They are
not certified Silver datasets and cannot feed a strategy directly.

| File | Reuse | Required recertification |
| --- | --- | --- |
| `calendar_20131213_20260311.json` | Historical KRX session seed | Verify session completeness and timezone semantics |
| `master_20160104_20260310_historical_v1.json` | Survivorship and common-share seed | Add company, market, sector, and status lineage |
| `krx-bars-20160104-20260310_backfill_v1.json` | Raw OHLCV and trading-value seed | Validate keys, availability, gaps, and provider provenance |
| `dart_disclosures_20160101_20260310_v1.json` | Filing identity and receipt-date index | Fetch/normalize XBRL facts; apply next-session availability |
| `corporate_actions_20160104_20260310_v2.json` | Adjustment/tradability evidence seed | Verify event classification and factor lineage; reject unknown actions |
| `costs/kis_lifetime_preferential_counterfactual_v1.json` | Tax/tick evidence and commission scenario | Separate statutory history from counterfactual commission assumption |

## Missing required evidence

The retained set does not provide a complete Champion input. The new data phase
must acquire or reconstruct:

- daily foreign, institution, and retail flow;
- DART XBRL financial facts and correction lineage;
- historical sector and company mappings;
- market capitalization and shares outstanding when absent from raw bars;
- authoritative suspension, management, and liquidation-trading status;
- complete raw corporate-action source provenance.

Missing evidence blocks certification; it is not imputed from legacy features.

## Removal set awaiting destructive-action approval

| Path | Reason | Approximate size before removal |
| --- | --- | ---: |
| `data/canonical/` | Provisional legacy base panel, labels, outcomes, and duplicate open-bar projections | 333 MB |
| `data/derived/` | Legacy `stock_net_alpha_v1` feature output | 161 MB |
| `data/catalog/` | Stale active/retention pointers to removed and missing datasets | 32 KB |
| `data/evidence/stocks/master_20260310.json` | Single current snapshot; historical master strictly dominates it for research | 918 KB |
| `data/trading_state.db` | Mutable legacy execution state; not a shared immutable Ledger | 28 KB |

These paths are Git-ignored and cannot be recovered with Git. Removal therefore
requires explicit approval of this exact list. Until then they are quarantined
by architecture: modern code and new manifests must not reference them.

## Target physical layout

After recertification, new data is written only to:

```text
data/
├── bronze/
├── silver/
├── gold/
└── artifacts/
```

Legacy evidence remains read-only until each source is imported into immutable
Bronze with content hashes. No in-place conversion or overwrite is permitted.
