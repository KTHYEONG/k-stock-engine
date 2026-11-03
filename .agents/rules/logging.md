---
paths:
  - "src/**"
---
# Unified Logging & Diagnostic Directives

> **Logs stay concise, structured, and machine-parsable; never truncate precision, suppress tracebacks, log secrets, or delete evidence unilaterally.**

## 1. Diagnostic Integrity
- **Precision Preservation:** Keep full precision for fee ratios, funding rates, optimizer tolerances, and weights; never round small non-zero values to zero. Keep full tracebacks, root causes, and error payloads at `ERROR`/`CRITICAL`.
- **Correlation Identifiers:** Include stable session, execution, order, symbol, or stage IDs where concurrent or multi-stage pipelines make attribution ambiguous.
- **Strict Credential Redaction:** NEVER log broker/exchange API credentials, private secret keys, app keys, passphrase tokens, certificates, webhook secrets, or sensitive account/wallet identifiers.

## 2. Category Taxonomy
- **Fixed Domain Category Taxonomy:** Exactly one of six tags per operational log event:
  - `[SYS]`: infrastructure, configuration, external connections, host resources
  - `[DATA]`: ingestion, collection plans, validation, panel construction
  - `[ALGO]`: alpha signals, models, feature computation, scoring
  - `[PORTFOLIO]`: allocation, rebalancing, NAV reconciliation, target weights
  - `[RISK]`: exposure, leverage limits, volatility stops, kill-switches
  - `[EXEC]`: broker API calls, order intents, submission, fills, slippage
- **Consistent Log Format:** Pipeline transitions and lifecycle messages carry the category prefix: `logger.info("[<CATEGORY>] <event description>")` (as in `src/data/operations.py`); legacy untagged logs change only when touched.
- **Large Collection Summaries:** Never dump arrays, DataFrames, or raw depth; log count, shape, min/max, null count, or head/tail samples.

## 3. Output & Lifecycle
- **High-Frequency Streams:** Route ticks, order book depth, execution dumps, factor evaluations, and optimizer traces to dedicated diagnostic destinations, not primary logs.
- **INFO vs. DEBUG/TRACE:** `INFO` = phase transitions and milestones (about one line per phase); `DEBUG`/`TRACE` = structured troubleshooting events.
- **Configured Storage Paths:** Persistent service logs live in the project log path; temporary diagnostics use scratch locations.
- **Non-Destructive Cleanup:** Clear diagnostics only via maintenance scripts or explicit lifecycle commands; never unilaterally delete logs or evidence needed for unresolved investigations.
