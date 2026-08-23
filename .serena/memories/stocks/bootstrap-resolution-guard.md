# ML bootstrap resolution guard (spec ml_full_run_completion)

Implemented per ADR_20260823_ML_BOOTSTRAP_RESOLUTION_GUARD.

## Components
- `DEFAULT_BOOTSTRAP_RESAMPLES = 2000` in `src/stocks/ml/horizons.py` (was 200).
- `_holm_admission(..., n_bootstrap)` fail-closed guard: raises ValueError
  "n_bootstrap=... is below the resolvable minimum ..." when
  `n_bootstrap < ceil(m / bootstrap_alpha)` (m = registered path hypotheses =
  cells x (2 or 3 paths)). Mirrors the `stitch_prequential_growth_route`
  ceil(1/alpha) precedent. `select_horizons` passes n_bootstrap through.
- `NetAlphaTrainingRequest.bootstrap_resamples` and CLI `--bootstrap-resamples`
  defaults = 2000. `CompoundingCertificationSettings` (holdout certificate)
  deliberately untouched (200, own resolution rules).
- `tools/run_train_detached.sh`: nohup launcher pinning H10-only pre-registered
  scope (C 5,10 / K 12,16,20,24 -> 24 cells, m=72), lookback 1260, holdout 252,
  max-rss 4096, reserve 2048; writes pid/log under scratch/.

## Root cause context (why NO_TRADE dominated: 54/90 ledger runs)
- B=200 p-value grid {k/200} has min non-zero p = 0.005 while rank-1 Holm
  threshold was alpha/72 ~= 0.000694 -> 62/72 hypotheses passable only with
  exactly-zero draws (de facto "perfect separation" policy, unintended).
- Full-grid runs timed out interactively (900s observed); completed reduced runs
  take 136-406s, H10-only+lookback1260 ~172s at peak RSS 1.42GiB.
- Negative adjusted lower growth across candidates is genuine economics:
  weak signals still correctly resolve to NO_TRADE after this fix.

## Semantics / gotchas
- Resolution invariant: `n_bootstrap >= ceil(m_family / bootstrap_alpha)`;
  24-cell family needs B>=1440, full 48-cell grid (m=144) needs >=2880 explicit.
- Tests must size families accordingly: any select_horizons call with small
  n_bootstrap and >n_bootstrap*alpha hypotheses now raises (test_uses_request_
  controlled_resamples uses B=64 for m=3).
- Test fixture regression fixed separately: horizon (3,5)->(5,10) because
  require_feasible_horizons enforces C<=H against default cadences (5,10,20).

## Tests
- tests/unit/stocks/ml/test_horizons.py: guard rejection/admission, production
  family resolution (<5s, min threshold >= 1/B), strong-signal attainability,
  request-controlled resamples updated.
- tests/unit/stocks/workflows/test_train_model.py::test_request_and_cli_defaults_updated.
