# Industry classification source verification

Date: 2026-09-20.

## Result

KRX Data Marketplace is the candidate primary source. Its public issue-statistics
screens expose a per-security industry-name column (`IDX_IND_NM`), which confirms
that KRX publishes an industry label distinct from the membership field
`SECT_TP_NM` in the stock master.

The data request endpoint requires an authenticated Marketplace browser session.
A direct request to the documented page's JSON endpoint returned `LOGOUT`; it
therefore did not yield a single retrievable current or historical snapshot in
this environment. Historical availability, date parameters, and delisted-name
coverage remain unverified. No industry rows are collected or inferred.

## Collection contract after access is provisioned

1. Authenticate a dedicated Marketplace session outside source control and pass
   it only at runtime.
2. Probe KOSPI and KOSDAQ separately for two historical rebalancing sessions
   and one recent completed session. Persist the exact request parameters and
   response status even when a request fails.
3. Accept a page only when its response provides requested `as_of`, a six-digit
   ticker, and a nonempty industry label. Record whether delisted securities
   appear; a current-only response cannot backfill history.
4. Store original response bytes and an immutable receipt under
   `data/bronze/stocks/industry_classification/<sha256>/`. Receipt metadata must
   include provider, screen/endpoint, market, requested date, retrieval time,
   content hash, response status, and authentication-free request shape.
5. Normalize each accepted observation with key
   `(as_of, market, instrument_id, source_hash)` and fields
   `classification_system`, `industry_name`, optional `industry_code`,
   `available_at`, `retrieved_at`, and `evidence_status`. Resolve the ticker via
   the same-date KRX master and its ISIN; fail the row when the identity is
   ambiguous.

## Backtest use

Use industry data only on sessions with an accepted same-date snapshot whose
availability is before the decision cutoff. Do not forward-fill a classification
change unless KRX provides its effective date. Until the source passes the
probe, sector-relative features and sector caps remain disabled.

## Efficient coverage order

For monthly rebalancing, collect only each month-end decision session first:
two market requests per session, then deduplicate raw bytes by hash. Expand to
daily snapshots only if a strategy needs daily industry transitions. This keeps
the initial historical demand bounded while preserving point-in-time evidence.
