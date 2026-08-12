# Stock Alpha v2 최신 81-Trial 결과

## 실행

| 항목 | 값 |
|---|---|
| 실행일 | 2026-08-12 |
| Snapshot | `research_provisional_20160104_20260812_cost_master_v3_mh2` |
| Artifact | `lambdarank_v2_20260812_sub10_81trial` |
| Mode | `research` |
| Selection policy | `economic-selection-v2-proxy-one-finalist` |
| Compute plan | `lgb_threads=4` |
| Command | `uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260812_sub10_81trial --snapshot-id research_provisional_20160104_20260812_cost_master_v3_mh2 --mode research --optuna-trials 81 --max-rss-mib 8000` |
| Exit status | `0` |

## 결과 판정

| 항목 | 값 |
|---|---:|
| `promoted` | `false` |
| `no_trade` | `true` |
| `selection_status` | `no_economically_eligible_candidate` |
| `n_folds_evaluated` | 0 |
| `n_terminal_trials` | 81 |
| `screened_trials` | 72 |
| `pruned_trials` | 9 |
| `shortlisted_trials` | 18 |
| `economically_eligible_trials` | 0 |
| `resolved_lgb_threads` | 4 |
| `best_screen_rank_ic` | 0.1034169899 |
| `gates.passed` | `false` |

## 소요시간

| 항목 | 값 |
|---|---:|
| Wall time | **12:29.05 (749.05 s)** |
| User time | 1,768.90 s |
| System time | 69.50 s |
| Screen total | 77.848 s |
| Full-refit total | 484.592 s |
| Economic replay total | 65.906 s |
| Early-rejected refits | 0 |
| Full-refit round cap | 900 |
| Full-refit patience | 100 |

## 메모리

| 항목 | 값 |
|---|---:|
| Baseline RSS | 1,504.188 MiB |
| Workflow peak RSS | **5,200.820 MiB** |
| External maximum RSS | **5,416.75 MiB** |
| Configured hard limit | 8,000 MiB |
| Replay operational limit | 7,000 MiB |
| Cache bytes | 340,529,292 bytes |
| Capacity failure | `null` |

## Route별 telemetry

| Route | Screen s | Context s | Refit train s | Refit predict s | Refit total s | Replay prepare s | Replay s | Peak RSS MiB | Cache bytes | Finalists |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 39.792 | 25.717 | 295.789 | 6.883 | 302.673 | 4.625 | 38.750 | 4,951.613 | 112,884,760 | 1 |
| 10 | 23.308 | 16.716 | 74.157 | 2.913 | 77.070 | 4.152 | 16.474 | 4,951.613 | 113,502,760 | 1 |
| 15 | 14.749 | 15.532 | 99.849 | 5.000 | 104.849 | 4.416 | 10.682 | 5,200.820 | 114,141,772 | 1 |

## Route별 refit rounds / best iteration

| Route | Refit actual rounds | Best iterations |
|---:|---|---|
| 5 | 900, 388, 600, 200, 200, 200, 288, 288 | 859, 288, 426, 1, 100, 1, 159, 141 |
| 10 | 326, 500, 311, 340, 243, 300, 511, 511 | 33, 398, 197, 162, 91, 183, 342, 323 |
| 15 | 340, 200, 400, 326, 249, 300, 700, 300 | 158, 74, 204, 176, 72, 106, 531, 121 |

## 백테스트 / 경제성 결과

| Route | Finalist trial | Median Rank-IC | Attempted orders | Filled orders | Planned cycles | Bootstrap lower bound | Strategy IR | Max drawdown | Turnover | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 sessions | 5 | 0.08661119 | 4,419 | 0 | 309 | -0.00040279 | 0.0 | 0.0 | 0.0 | `false` |
| 10 sessions | 26 | 0.10283674 | 2,302 | 0 | 160 | -0.00040279 | 0.0 | 0.0 | 0.0 | `false` |
| 15 sessions | 19 | 0.10164764 | 1,101 | 0 | 76 | -0.00040279 | 0.0 | 0.0 | 0.0 | `false` |
| **합계** | — | — | **7,822** | **0** | **545** | — | — | — | — | **NO_TRADE** |

### No-trade reason counts

| Route | Insufficient covariance | Non-finite scored frame | No feasible allocation |
|---:|---:|---:|---:|
| 5 | 59 | 90 | 35 |
| 10 | 24 | 46 | 17 |
| 15 | 18 | 31 | 40 |
| **합계** | **101** | **167** | **92** |

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260812_sub10_81trial/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260812_sub10_81trial/manifest.json)

---

# Execution recovery 재실행 결과

## 실행

| 항목 | 값 |
|---|---|
| 실행일 | 2026-08-13 |
| Snapshot | `research_provisional_20160104_20260812_cost_master_v3_mh2` |
| Artifact | `lambdarank_v2_20260813_execution_recovery` |
| Mode | `research` |
| Command | `uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260813_execution_recovery --snapshot-id research_provisional_20160104_20260812_cost_master_v3_mh2 --mode research --optuna-trials 81 --max-rss-mib 8000` |
| Exit status | `0` |

## 결과 판정

| 항목 | 값 |
|---|---:|
| `promoted` | `false` |
| `no_trade` | `true` |
| `selection_status` | `selected` (resource telemetry) |
| `n_folds_evaluated` | 3 |
| `n_terminal_trials` | 81 |
| `screened_trials` | 72 |
| `pruned_trials` | 9 |
| `shortlisted_trials` | 18 |
| `economically_eligible_trials` | 1 |
| `planned_cycles` | 134 |
| `attempted_orders` | 1,737 |
| `filled_orders` | **1,737** |
| `gates.passed` | `false` |

체결은 기존 0건에서 1,737건으로 회복되었다. 그러나 기존 fail-closed
정책에 따라 최종 승격은 거부되었다.

## 경제성 / 게이트 결과

| 항목 | 값 |
|---|---:|
| Gate 1 positive Rank-IC fraction | 1.0000 |
| Gate 2 excess bootstrap lower bound | **-0.00017155** |
| Gate 3 strategy IR / benchmark IR | 0.957496 / 0.095455 |
| Gate 4 stress excess | `true` |
| Gate 4 stress total / benchmark return | 0.82610756 / 0.15974089 |
| Gate 5 deflated Sharpe probability | 0.661712 |
| Gate 6 drawdown ratio | 0.2108 / 0.6174 |
| Gate 8 forward holdout | `false` (insufficient label-available sessions) |

## Route별 finalist evidence

| Route | Trial | Attempted | Filled | Bootstrap lower bound | Eligible |
|---:|---:|---:|---:|---:|---:|
| 5 sessions | 5 | 3,296 | 3,296 | -0.00002456 | `false` |
| 10 sessions | 26 | 1,823 | 1,823 | **0.00005287** | `true` |
| 15 sessions | 19 | 939 | 939 | -0.00022462 | `false` |

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260813_execution_recovery/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260813_execution_recovery/manifest.json)
- [Execution log](../../logs/scratch/execution_recovery_train_20260813.log)
