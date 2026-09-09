---
trigger:
  - on_label: ["logging"]
  - on_file_path_regex: "src/.*(logging|engine|execution|api|pipeline|service|orchestration|strategy).*"
  - on_file_path_glob: ["src/**/logging/**/*.py", "src/**/execution/**/*.py", "src/**/engine/**/*.py"]
priority: 9
---

# Unified Logging & Diagnostic Directives

> **Logs must be concise, structured, and machine-parsable without sacrificing diagnostic correctness. Never truncate required numerical precision or suppress exception tracebacks to save tokens. Never log secrets or credentials, and never delete diagnostic evidence unilaterally.**

## 1. Information Integrity & Diagnostic Value
- **Correctness Over Token Economy:** Token thrift must never compromise the ability to diagnose bugs or reconstruct events. Preserve full exception tracebacks, root causes, and error payloads for `ERROR` and `CRITICAL` levels.
- **Precision Preservation:** Format numbers compactly, but retain sufficient precision to diagnose the underlying quantity (e.g., fee rates, funding rates, optimizer tolerances, and weights must not be rounded into meaningless zeros).
- **Correlation Identifiers:** Include stable identifiers (e.g., `run_id`, `order_id`, `trial_id`, `symbol`, or `stage`) when concurrent or multi-stage pipelines make event attribution ambiguous.
- **Strict Credential Redaction:** NEVER log broker API credentials, app keys, secret tokens, certificate passwords, or sensitive account numbers.

## 2. Structured Output & Collection Summaries
- **Structured Fields:** Prefer structured key-value pairs or machine-readable records (e.g., JSONL or standardized delimited fields) over long conversational sentences. Ensure keys are consistent and values are safely serialized.
- **Large Collection Summaries:** Never dump massive arrays, DataFrames, or raw market depth directly into logs. Summarize collections using concise descriptors: count, shape, min/max range, null/nan count, or representative head/tail samples.
- **Category Taxonomy Hygiene:** Maintain a small, stable set of category tags (e.g., `SYS`, `DATA`, `EXEC`, `ALGO`, `RISK`, `PORTFOLIO`). Reuse existing tags before introducing new ones, but do not force-fit unrelated events into arbitrary buckets.

## 3. Separation of Concerns & High-Frequency Output
- **Operational vs. High-Frequency Logs:** Keep high-frequency streaming data (e.g., quote ticks, order execution events, raw optimizer trials, detailed memory snapshots) isolated in dedicated log destinations (e.g., specific `.jsonl` or diagnostic artifacts) rather than spamming primary system logs.
- **INFO vs. DEBUG/TRACE:**
  - **`INFO`**: High-level, human-readable phase transitions and milestone summaries (typically 1 line per phase).
  - **`DEBUG` / `TRACE`**: Fine-grained, structured diagnostic events targeted for troubleshooting and programmatic inspection.

## 4. Lifecycle & Path Hygiene
- **Configured Storage Paths:** Persistent service logs must reside in the designated project log path. Temporary diagnostic logs should use distinct scratch locations.
- **Non-Destructive Cleanup:** Temporary diagnostic logs may be cleared via project-designated maintenance scripts or explicit lifecycle commands. The AI must never unilaterally delete diagnostic logs or historical evidence needed for unresolved investigations.


