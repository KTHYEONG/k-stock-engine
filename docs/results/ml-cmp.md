# ML 비교 결과 (최신 실행)

## 실행 개요

| 항목 | 값 |
|---|---:|
| 실행일 | 2026-08-29 |
| artifact | `ml-family-integrity-20260829` |
| snapshot | `research_stock_net_alpha_v1_exec_20260828_10m` |
| mode | `research-only-model-selection-study` |
| route | `unhedged_absolute` |
| horizon / cadence / Top-K | H10 / C10 / K12 |
| training lookback | 504 sessions |
| purged folds / embargo | 3 / 5 sessions |
| bootstrap resamples | 360 |
| adjusted bootstrap alpha | 0.002777777778 |
| screen / global budget | 450 / 600 sec |
| 결과 | `RESEARCH_ONLY`, `no-qualified-survivor` |
| artifact publish | `false` |

실행 명령:

```bash
uv run python src/stocks/cli/train.py --artifact-id ml-family-integrity-20260829 --snapshot-id research_stock_net_alpha_v1_exec_20260828_10m --mode research --research-only-model-selection-study --candidate-horizon-sessions 10 --candidate-rebalance-frequency-sessions 10 --candidate-top-k 12 --candidate-training-lookback-sessions 504 --fold-count 3 --embargo-sessions 5 --bootstrap-resamples 20 --model-selection-wall-clock-seconds 600 --model-selection-screen-phase-seconds 450
```

## 실행 자원

| 지표 | 값 |
|---|---:|
| input rows | 918,443 |
| candidate count | 18 |
| cache hits / folds | 3 / 3 |
| screen learner fits | 108 |
| attribution predictions | 63 |
| full OOF fits | 2 |
| replay count | 2 |
| 총 소요 시간 | 424.28 sec |
| screen 소요 시간 | 212.61 sec |
| OOF 소요 시간 | 211.67 sec |

## Family별 결과

| family | screen tail-excess LB | screen 판정 | OOF | replay / fills | 최종 상태 |
|---|---:|---|---|---|---|
| `elastic_net_v2` | -0.0005470756 | 미자격 | 미진입 | - | `screen-not-qualified` |
| `huber_linear_v1` | -0.0023869464 | 미자격 | 미진입 | - | `screen-not-qualified` |
| `extra_trees_v1` | -5.75e-18 | 미자격 | 미진입 | - | `screen-not-qualified` |
| `hist_gradient_quantile_v1` | -3.28e-18 | 자격 | 완료 | 3 profiles / 0 | `replay-no-fills` |
| `rawnet_lgbm_v2` | -3.73e-18 | 자격 | 완료 | 3 profiles / 0 | `replay-no-fills` |
| `tail_lambdarank_v2` | -1.0e12 | hard reject | 미진입 | - | `screen-not-qualified` |

Oracle tail-excess lower bound도 `-2.44e-18`로 양수가 아니었다. OOF에 진입한
두 family는 `legacy_overlay_5bps`, `lower_bound_only`,
`lower_bound_half_kelly` 세 profile 모두 `observed_interval_count=693`,
`invested_interval_count=0`, `filled_orders=0`이었다. 따라서 이번 실행에서는
승격 가능한 family/profile이 없다.

## 해석 및 후속 조치

- 수정한 canonical family fitting과 route-aligned calibration 경로는 제한 시간
  내 완주했지만, 현재 snapshot에서 경제적 signal은 확인되지 않았다.
- 0 fill은 5bp profile에 국한되지 않고 zero-band profile에서도 동일했다. 이는
  단순 no-trade band 완화로 해결할 수 있는 결과가 아니다.
- Huber screen에서 `max_iter=20` 수렴 경고가 반복됐지만 연구 전체는 정상
  완료됐다. 다음 개선은 Huber 반복 상한 자체보다 최근 regime별 label/feature
  signal과 absolute-route opportunity set을 별도로 검증해야 한다.
- artifact는 계속 publish하지 않으며, 양의 calibrated lower bound와 실제
  base/stress replay 체결이 동시에 확인될 때만 승격을 검토한다.
