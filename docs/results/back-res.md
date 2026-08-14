# 2026-08-14 completion validation ML / 백테스트 결과

## 실행 및 자원

| 항목 | 값 |
|---|---|
| Artifact | `lambdarank_v2_20260814_completion_validation` |
| Snapshot | `research_provisional_20160104_20260812_cost_master_v3_mh2` |
| Command | `PYTHONPATH=. LOG_LEVEL=INFO uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260814_completion_validation --snapshot-id research_provisional_20160104_20260812_cost_master_v3_mh2 --mode research --optuna-trials 81 --max-rss-mib 8000` |
| 실행 구간 | `2026-08-14 13:33:17.969 ~ 14:00:16.143 KST` |
| Exit status | `0` |
| Wall time | **1,618.174 s (26분 58.174초)** |
| Screen / full refit / economic replay | **329.716 / 364.643 / 28.406 s** |
| Baseline / peak RSS | **1,749.66 / 5,757.00 MiB** |
| RSS limit | `8,000 MiB` |
| LGBM threads | `4` |

## ML 성능 및 탐색

| 항목 | 값 |
|---|---:|
| Terminal trials | **81** |
| Screened / pruned | **79 / 2** |
| Confirmation attempts / confirmed | **18 / 18** |
| Global multiplicity count | `81` |
| Confirmation Rank-IC | **0.05169472 ~ 0.13608934** |
| Confirmation Rank-IC 평균 / 중앙값 | **0.09452876 / 0.08837467** |
| 최고 fold DSR probability | **0.30378388** |
| Best screen proxy lower bound | **-0.00177818** |

## Route별 후보 및 백테스트

| Route | Screened | Pruned | Confirmed | Exact replay | Shortlist fills | Shortlist median Rank-IC | Bootstrap lower bound 범위 | Strategy IR 범위 | MDD 범위 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 sessions | 27 | 0 | 6 | 3 | 2,468 | 0.09073461 ~ 0.09388179 | -0.00190350 ~ -0.00160112 | 0.4223 ~ 0.6037 | 21.17% ~ 22.45% |
| 10 sessions | 27 | 0 | 6 | 3 | 1,393 | 0.09546132 ~ 0.10779940 | -0.00538307 ~ -0.00380743 | 0.3737 ~ 0.4503 | 19.10% ~ 24.13% |
| 15 sessions | 25 | 2 | 6 | 3 | 1,084 | 0.08460602 ~ 0.09439248 | -0.00732762 ~ -0.00405612 | 0.2934 ~ 0.5683 | 19.60% ~ 24.74% |
| **합계** | **79** | **2** | **18** | **9** | **4,945** | — | **모두 음수** | — | — |

모든 exact replay 후보에서 주문은 **4,945 / 4,945건 체결**됐다. Confirmation
fold 기준 CAGR은 **-2.0703% ~ +24.7657%**, MDD는 **5.0875% ~ 29.1199%**로
fold 간 편차가 컸다.

## 최종 판정

| 항목 | 값 |
|---|---|
| `promoted` / `no_trade` | `false / true` |
| `selection_status` | `no_economically_eligible_candidate` |
| Economically eligible trials | `0` |
| `n_folds_evaluated` | `0` (최종 champion 없음) |
| `promotion_reasons` | `no-champion-trial` |
| `ledger_metrics` / `stress_metrics` | `{}` / `null` |

Rank-IC 양수 신호와 체결 정상 여부는 확인됐지만, 9개 exact replay 후보의
bootstrap lower bound가 모두 음수이고 DSR도 승격 기준에 미달해 최종 승격하지
않았다.

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260814_completion_validation/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260814_completion_validation/manifest.json)
- 실행 로그: `scratch/lambdarank_v2_20260814_completion_validation.log`
