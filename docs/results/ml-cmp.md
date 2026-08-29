# ML 비교 결과 (최신 실행)

## 실행 개요

| 항목 | 값 |
|---|---:|
| 실행일 | 2026-08-29 |
| artifact | `exec-econ-verify-20260829` |
| snapshot | `research_stock_net_alpha_v1_exec_20260828_10m` |
| mode | `research-only-model-selection-study` |
| route | `unhedged_absolute` |
| horizon (H) | 10 sessions |
| rebalance cadence (C) | 10 sessions |
| Top-K | 12 |
| training lookback | 504 sessions |
| purged folds | 3 |
| embargo | 5 sessions |
| bootstrap resamples | 360 (alpha 보정 후) |
| adjusted bootstrap alpha | 0.002777777778 |
| screen / global budget | 270 / 300 sec |
| 결과 | `RESEARCH_ONLY`, `no-qualified-survivor` |
| artifact publish | `false` |

실행 명령:

```bash
uv run python src/stocks/cli/train.py --artifact-id exec-econ-verify-20260829 --snapshot-id research_stock_net_alpha_v1_exec_20260828_10m --mode research --research-only-model-selection-study --candidate-horizon-sessions 10 --candidate-rebalance-frequency-sessions 10 --candidate-top-k 12 --candidate-training-lookback-sessions 504 --fold-count 3 --embargo-sessions 5 --bootstrap-resamples 20 --model-selection-wall-clock-seconds 300 --model-selection-screen-phase-seconds 270
```

## 데이터 및 실행 자원

| 지표 | 값 |
|---|---:|
| input rows | 918,443 |
| candidate count | 18 |
| cache hits / folds | 3 / 3 |
| screen learner fit count | 108 |
| model fit count | 20 |
| attribution prediction count | 63 |
| full OOF fit count | 2 |
| replay count | 2 |
| 총 소요 시간 | 277.17 sec |
| screen 소요 시간 | 221.00 sec |
| OOF 소요 시간 | 56.17 sec |

## Family별 결과

| family | screen tail-excess LB | screen 판정 | OOF | replay | 최종 상태 |
|---|---:|---|---|---|---|
| `elastic_net_v2` | -0.000547 | 미자격 | 미진입 | 미진입 | `screen-not-qualified` |
| `huber_linear_v1` | -0.002387 | 미자격 | 미진입 | 미진입 | `screen-not-qualified` |
| `extra_trees_v1` | 약 0 | 미자격 | 미진입 | 미진입 | `screen-not-qualified` |
| `hist_gradient_quantile_v1` | -3.28e-18 | 자격 | 완료 | 완료 | `replay-no-fills` |
| `rawnet_lgbm_v2` | -3.73e-18 | 자격 | 완료 | 완료 | `replay-no-fills` |
| `tail_lambdarank_v2` | -1.0e12 | 미자격 | 미진입 | 미진입 | `screen-not-qualified` |

Screen absolute lower bound는 family별 약 `-0.047` 수준이었다. Oracle
tail-excess lower bound는 약 `-2.44e-18`이며, 모델 tail-excess lower bound가
양수가 아니어서 최종 champion을 선정하지 않았다.

## OOF 및 replay 결과

`hist_gradient_quantile_v1`와 `rawnet_lgbm_v2`만 full OOF 및 replay에 진입했다.
두 후보 모두 replay 자체는 완료됐지만 `filled_orders=0`으로 체결이 없어
`replay-no-fills`로 종료됐다. 따라서 승격 가능한 survivor는 없다.

실행 중 Huber screen fit에서 `max_iter=20` 도달 경고가 발생했으나 예외가 아닌
screen 반복 상한 경고이며 전체 연구 실행은 정상 완료됐다.

## 결론

- 파이프라인은 cache → screen → OOF → replay까지 예산 내 정상 완료됐다.
- 현재 snapshot의 H10/C10/K12 unhedged 조건에서는 양의 corrected tail lower bound와
  실제 replay 체결을 동시에 만족하는 모델이 없다.
- 결과는 artifact를 publish하지 않는 `RESEARCH_ONLY` 상태로 유지된다.
