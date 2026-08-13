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

---

# Compounding selection policy 동기화 후 검증 결과

## 실행 검증

| 항목 | 값 |
|---|---:|
| 실행일 | 2026-08-13 |
| 전체 테스트 | **643 passed** |
| 정책 버전 | `economic-selection-v3-compounding` |
| 경제 finalist 폭 | `2` |
| Ruff / mypy | `PASS / PASS` |
| ADR | `ADR_20260813_STOCK_COMPOUNDING_POLICY_SYNC` |

구버전 `economic-selection-v2-proxy-one-finalist`를 기대하던 81-trial 통합
테스트를 현재 v3 정책(`economic_finalist_width=2`)에 맞춰 동기화했다. 이
변경은 테스트·문서의 정책 식별자만 정정한 것이며, promotion threshold나
fail-closed 조건은 완화하지 않았다.

## 백테스트 판정

| 항목 | 값 |
|---|---:|
| 체결 | **1,737 / 1,737** |
| Base 누적수익률 | **82.90%** |
| Stress 누적수익률 | **82.61%** |
| Benchmark 누적수익률 | **15.97%** |
| `promoted` | `false` |
| `no_trade` | `true` |

### 쉽게 읽는 결론

1. **주문 실행 문제는 해결됐다.** 계획된 주문 1,737건이 모두 체결되어,
   “신호는 있으나 체결이 0건”인 이전 장애는 재현되지 않았다.
2. **겉보기 수익률은 좋다.** Base와 Stress 모두 benchmark보다 높고, 비용을
   반영한 Stress에서도 누적수익률이 82.61%였다.
3. **그래도 자동 승격은 거부됐다.** Gate 2는 비용·시계열 변동성을 반영한
   초과수익 bootstrap 하한을 검사하는데 `-0.00017155`로 0보다 작았다.
   관측된 평균 성과가 우연이 아니라고 확신할 수 없다는 뜻이다.
4. **모델의 순위 예측력과 자산증식은 다른 문제다.** 모든 outer fold의
   Rank-IC는 양수였지만, 개별 종목 순위를 실제 포트폴리오로 바꾸는 과정의
   회전율·상관위험·현금 구간을 거친 뒤 안정적인 복리 초과성장이 입증되지
   않았다.
5. **추가로 DSR도 부족하다.** Deflated Sharpe probability가 `0.661712`로
   요구치 `0.95`보다 낮아, 81개 후보를 탐색한 뒤 선택된 결과의
   multiple-testing 위험도 통과하지 못했다.

따라서 이번 결과는 “실행 가능한 전략이며 역사적 누적수익률은 높다”까지
확인한 것이고, “새 데이터에서도 자산을 안정적으로 증식할 전략으로
승격할 수 있다”는 증거는 아니다. 레이블 데이터가 `2026-02-10`에서 끝나
2026-03-10 이후 252개 label-available 세션을 아직 제공하지 못하므로,
forward holdout 역시 성숙할 때까지 `NO_TRADE`가 유지된다.

---

# Compounding stability telemetry 재실행 결과

## 실행

| 항목 | 값 |
|---|---|
| 실행일 | 2026-08-13 |
| Snapshot | `research_provisional_20160104_20260812_cost_master_v3_mh2` |
| Artifact | `lambdarank_v2_20260813_stability_telemetry` |
| Mode | `research` |
| Selection policy | `economic-selection-v3-compounding` |
| Compute plan | `lgb_threads` 기본 설정 |
| Command | `LOG_LEVEL=DEBUG PYTHONPATH=. timeout 1800 uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260813_stability_telemetry --snapshot-id research_provisional_20160104_20260812_cost_master_v3_mh2 --mode research --optuna-trials 81 --max-rss-mib 8000` |
| Exit status | `0` |

## 결과 판정

| 항목 | 값 |
|---|---:|
| `promoted` | `false` |
| `no_trade` | `true` |
| `selection_status` | `no_economically_eligible_candidate` |
| `n_terminal_trials` | 81 |
| `screened_trials` | 72 |
| `pruned_trials` | 9 |
| `shortlisted_trials` | 18 |
| `economically_eligible_trials` | **0** |
| `best_screen_rank_ic` | 0.10341699 |
| `gates.passed` | `false` |

이번 실행은 최종 promotion replay까지 도달하지 못했다. 18개 shortlist
후보를 6개 compounding policy와 조합해 평가했지만, 36개 `(trial, policy)`
모두 DSR gate를 통과하지 못해 최종 OOS ledger와 최종 수익률은 생성되지
않았다. fail-closed 동작으로 `NO_TRADE` artifact가 발행됐다.

## 실행 자원

| 항목 | 값 |
|---|---:|
| Screen | 37.27 s |
| Full refit | 299.03 s |
| Economic replay | 619.83 s |
| Peak RSS | 5,191.44 MiB |

## Compounding telemetry 분석

새 telemetry는 inner candidate evidence에 정상 저장됐다. 대표적으로 가장
높은 복리 bootstrap 하한은 10세션 route의 trial 14, policy
`5:ga2_tb0.2`에서 `0.00101887`이었지만 DSR은 `0.56010868`로 요구치
`0.95`에 미달했다.

| Route / trial | Policy | Bootstrap 하한 | DSR | 평균 scale | p10 scale | 평균 turnover lambda | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| 10 / 14 | `5:ga2_tb0.2` | **0.00101887** | 0.560109 | 0.980991 | 1.000000 | 0.386896 | DSR 실패 |
| 10 / 26 | `3:ga1_tb0.2` | 0.00092562 | 0.557029 | 0.994579 | 1.000000 | 0.381727 | DSR 실패 |
| 5 / 5 | `5:ga2_tb0.2` | -0.00005951 | — | 0.975481 | 1.000000 | 0.390777 | 하한 실패 |
| 15 / 19 | `5:ga2_tb0.2` | -0.00316764 | — | 0.891989 | 0.659710 | 0.417864 | 하한 실패 |

`cash_count`는 telemetry상 0이었고, `positive_scale_fraction`은 모든
후보에서 1.0이었다. 즉 이번 실행의 병목은 covariance/variance 때문에
현금화된 것이 아니라, 양수 edge를 가진 포지션을 구성한 뒤에도 81개 후보
탐색에 대한 통계적 확실성(DSR)이 부족했던 것이다. 특히 15세션 route는
평균 scale 0.892, p10 scale 0.660까지 축소되고 복리 하한도 크게 음수여서
현재 데이터에서는 우선순위가 낮다.

## 해석 및 다음 단계

1. **체결 경로는 정상이다.** 이번 실행은 후보 selection 단계에서 종료됐기
   때문에 최종 attempted/filled order 수를 산출하지 않았지만, 이전
   execution recovery 실행에서 1,737/1,737 전량 체결이 이미 확인됐다.
2. **telemetry 수집은 성공했다.** decision count, cash reason, confidence
   scale, gross exposure, turnover lambda가 후보별 JSON에 저장됐다.
3. **성과 기준 완화는 정당화되지 않는다.** 최고 하한도 DSR 0.56으로
   0.95에 크게 못 미치므로 Gate 5 또는 forward holdout 조건을 낮추면 안 된다.
4. **개선 방향은 모델·feature의 OOS 안정성 검증이다.** 현 단계에서 policy
   grid를 늘리거나 결과를 보고 policy를 수동 선택하지 않고, 동일 snapshot의
   block별 손실·feature 기여·forward label 성숙을 추가 확인해야 한다.

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260813_stability_telemetry/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260813_stability_telemetry/manifest.json)
