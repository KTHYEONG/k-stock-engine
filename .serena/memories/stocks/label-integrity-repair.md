# Stock label integrity repair (spec stock_label_integrity_repair)

Implemented per `docs/specs/stock_label_integrity_repair_contract.json`.

## Semantics (critical)
- Net-alpha labels are SIMPLE decimal rates: `gross = exit_open / entry_open - 1` (was log diff). Applies to `build_net_alpha_label_dataset_with_status` only; older residual/cost-aware builders untouched.
- `classify_label_action_coverage(outcomes, calendar, CorporateActionSnapshot) -> outcomes + _action_unsupported` (src/stocks/data/labels.py). Vectorized O(n+a): calendar-position prefix counts via sorted `join_asof(by=instrument, allow_exact_matches=False, check_sortedness=False)`; blocked iff path [entry_pos, exit_pos) misses a covered pair or crosses `action_code != "no_action"`. Duplicate `(instrument, prev_pos)` in snapshot raises.
- Builder with `corporate_actions`: blocked price-complete paths emit exactly one `UNSUPPORTED_CORPORATE_ACTION`, no label row. Garbage opens map to MISSING_ENTRY/EXIT_PRICE. `corporate_actions=None` = legacy unverified mode (existing tests rely on it); publication rejects it.
- `materialize_net_alpha_snapshot` (research_v2): requires source.corporate_actions entry, loads via `load_corporate_action_snapshot`, fails `corporate-action-coverage-required` on absent/incompatible/no-no_action-intervals and on any UNSUPPORTED status post-build (before publish). Hash bound into label/status/evidence content_manifest as `corporate_actions_hash` (new optional kwarg on 3 publishers).

## ML tail objective
- `TailCaptureEvidence`/`SegmentTailEvidence` fields renamed `*_log_growth` -> `*_utility`; arithmetic decimal residual utility, no log1p anywhere, no `<= -1` domain check.
- `_decimal_utility` raises new `InvalidOofEconomicUtilityError(ValueError)` on null/non-finite ONLY on join-produced rows (join-first validation → locked holdout can't influence research).
- `evaluate_economic_window_candidate`: all-label guard removed; catches InvalidOofEconomicUtilityError -> window rejection token `invalid-oof-economic-utility`. Study maps it to `next_action=repair-label-integrity` when no winner (never no-label-capacity).

## Test gotchas
- lean_check greps full scenario ids (`LABEL_INTEGRITY_01_SIMPLE_RETURN_UNIT`) in target test file — docstrings carry them; function names keep uppercase prefix so contract `-k LABEL_INTEGRITY_xx` selects. N802 added to tests/* per-file-ignores in pyproject.
- research_v2 fixtures need ≥30 tickers/session (`_MIN_NET_ALPHA_ROWS_PER_SESSION`) or partitioned builder returns empty labels and publisher raises "cannot publish empty".
