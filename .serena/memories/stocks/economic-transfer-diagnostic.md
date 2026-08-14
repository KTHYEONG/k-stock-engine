# ML economic-transfer diagnostic (spec ml_backtest_economic_validity)

Implemented per `docs/specs/ml_backtest_economic_validity.md` / `..._contract.json`.

## Components
- `economic_transfer_attribution(scored, label_column, top_k)` in `src/stocks/research/metrics.py`.
  Pure per-session ranking→selected-tail attribution: cross-sectional Spearman Rank-IC,
  top-k/universe/active label means, membership turnover vs preceding retained session
  (first retained session turnover = 1.0). Null labels excluded (never zero-filled);
  non-finite score/label in a retained session raises ValueError; empty valid frame
  returns zero aggregates. Vectorized; only a bounded loop over sessions.
- `_economic_transfer_evidence(oos_scored, panel, label_column, top_k, replay)` in
  `src/stocks/workflows/train_model.py`. Emits JSON-safe dict with sections:
  `schema_version="economic-transfer-v1"`, `ranking` (per-session rank_ic, coverage,
  `excluded_unavailable_label_sessions`), `top_k_label`, `selection` (from
  `replay.compounding_overlay`: selected_count == overlay decision_count), `execution`
  (attempted/filled/unfilled + reason counts + cost_drag from replay), `compounding`
  (block-log series via extracted `_block_log_excess_series`; `bootstrap_lower_bound`/
  `dsr_probability` are `None` until injected).
- `_evaluate_economic_candidate` now takes optional `oos_scored`, `panel`, `top_k` and
  attaches `economic_transfer_evidence` (dataclass field, also in `to_json_safe()`),
  injecting the EXACT registered gate `bootstrap_lower_bound`/`dsr_probability`/blocks.
- `_select_economic_champion` passes `oos`, `tuning_panel`, `request.top_k` (skips when
  `oos` empty). Shortlist rows therefore carry the diagnostic into the artifact via
  `_selection_telemetry`'s `shortlist_candidate_evidence`.

## Semantics / gotchas
- "label available" = panel row's `label_column` non-null AND availability column
  non-null; availability column resolved by `_resolve_label_available_column`
  (`label_available_time` canonical, else `label_available_time_{N}d` by label suffix,
  else sole `label_available_time*`).
- Duplicate `(session, instrument_id)` keys in scored or panel raise ValueError.
- Diagnostic-only: does NOT change Optuna objective, shortlisting, multiplicity, gates,
  or holdout. `_block_log_excess_series` extraction from `_compounding_evidence` is a
  pure refactor (identical block construction).

## Tests
- `tests/unit/stocks/research/test_metrics.py` (new).
- `tests/unit/stocks/workflows/test_train_model.py`: added
  `test_economic_transfer_evidence_reconciles_replay_overlay_and_label_availability`,
  `test_economic_transfer_evidence_injects_registered_compounding_gate_values`,
  `test_prepared_replay_matches_reference_replay` (reference/prepared parity), plus
  helper `_transfer_evidence_panel`.
