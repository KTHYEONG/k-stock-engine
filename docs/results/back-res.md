# Stock Alpha v2 재학습 결과

- 실행일: 2026-08-11
- Snapshot: `research_provisional_20160104_20260227_cost_master_v2_r1`
- Artifact: `lambdarank_v2_20260811_cost_master_r4_adtvfix`
- 결과: **NO_TRADE / 미승격**
- 검증 프로파일: outer walk-forward 3 folds, Optuna 3 trials

## 실행 범위와 한계

이번 실행은 production 설정의 Optuna 80 trials가 아니라 3 trials로 축소한
검증 프로파일이다. 80-trial 실행은 trial당 수십 초 이상 소요되어 제한 시간
내 최종 artifact 생성 전에 중단했다. 따라서 이번 결과는 수정된 다중 fold 및
event replay 경로의 실행 검증이며, 80-trial production 탐색 결과로 해석하면
안 된다.

## 핵심 성과

| 항목 | 결과 | 의미 |
|---|---:|---|
| Evaluated folds | 3 | 단일 fold 의존 제거 |
| Median Rank-IC | 0.09502298 | 세 fold 순위 예측력의 중앙값 |
| Positive fold IC fraction | 1.0000 | 세 fold 모두 양수 |
| CAGR | 0.06765591 | 연환산 전략 수익률 |
| Annualized volatility | 0.08574637 | 연환산 변동성 |
| Sharpe | 0.80689150 | base 비용 기준 위험조정 성과 |
| Max drawdown | 0.22505300 | 최대 낙폭 |
| Exposure | 0.68349673 | 평균 투자 노출 |
| Turnover | 4.16809787 | replay 누적 turnover |
| Cost drag | 0.00450877 | 비용이 equity에 미친 drag |
| Planned cycles | 269 | 정상 계획된 rebalance cycle |
| Attempted orders | 3,029 | 주문 시도 횟수 |
| Filled orders | 3,029 | 체결 횟수 |

이전 결과의 exposure·turnover·CAGR가 모두 0이었던 이유는 scored replay가
allocation을 만들고도 빈 `intents`를 반환했기 때문이다. 이번 결과에서는
실제 주문과 체결이 발생해 경제성 지표가 유효한 형태로 계산됐다.

## 비용 스트레스 비교

| 항목 | 결과 |
|---|---:|
| Base total return | 0.56212756 |
| Stress total return | 0.55858005 |
| Benchmark total return | -0.15048441 |
| Gate 4 | 통과 |

스트레스 비용에서도 benchmark보다 높은 누적 수익률이 기록됐다. 다만 이는
historical OOS replay 결과일 뿐, 미래 성과를 보장하지 않는다.

## 승격 게이트 결과

| 게이트 | 결과 | 판정 |
|---|---:|---|
| Positive fold IC fraction | 1.0000 | 통과 |
| Executed orders/fills | 3,029 / 3,029 | 통과 |
| Bootstrap excess lower bound | -0.00011480 | 실패 |
| Strategy IR vs benchmark IR | 0.764015 vs -0.101442 | 참고 |
| Stress cost excess | true | 통과 |
| Deflated Sharpe probability | 0.871202 | 실패 (`< 0.95`) |
| Forward holdout | 준비 안 됨 | 실패 |

Bootstrap 하한이 음수이므로 초과수익의 하방 안정성을 입증하지 못했다.
Deflated Sharpe도 후보 탐색 편향을 보정한 뒤 기준 0.95에 미달했다. 또한
현재 snapshot은 2026-02-27에서 끝나므로 2026-03-10 이후 label-available
252-session forward holdout을 구성할 수 없다.

## Replay 품질 진단

`no_trade_reason_counts`:

- `constraint:insufficient covariance data`: 74회
- `no-feasible-allocation`: 1회

초기 replay 구간은 공분산 lookback 데이터 부족으로 일부 cycle이 건너뛰어졌다.
그럼에도 269개 cycle이 계획됐고 3,029건이 모두 체결됐다.

## 이번 수정 사항

1. 요청된 `n_folds`와 `embargo_sessions`를 outer splitter에 적용했다.
2. 중첩 fold의 중복 학습행을 제거했다.
3. replay scored panel에 trading value 기반 ADTV를 연결했다.
4. allocation을 `TradeIntent`로 변환해 실제 backtester 주문 경로를 연결했다.
5. 체결 수가 0이면 promotion을 차단하도록 evidence gate를 강화했다.
6. stress return과 benchmark return을 직접 비교하도록 Gate 4를 수정했다.

## 최종 판정

모델은 양의 Rank-IC와 실제 체결 성과를 보였지만, bootstrap 안정성·Deflated
Sharpe·forward holdout을 모두 충족하지 못했다. 따라서 artifact는 의도대로
`NO_TRADE`이며 paper/live 사용을 금지한다.

## 다음 검증 작업

1. 장시간 실행 환경에서 동일 snapshot을 Optuna 80 trials로 재학습한다.
2. 2026-03-10 이후 252개 label-available session을 확보한다.
3. 고정 후보 fingerprint로 forward holdout을 단 한 번 평가한다.
4. covariance lookback 부족 cycle을 historical context 포함 replay로 재검증한다.
5. 모든 게이트 통과 전까지 `NO_TRADE`를 유지한다.

## 산출물

- Metrics: `data/artifacts/stocks/lambdarank_v2_20260811_cost_master_r4_adtvfix/metrics.json`
- Manifest: `data/artifacts/stocks/lambdarank_v2_20260811_cost_master_r4_adtvfix/manifest.json`
- Model: `data/artifacts/stocks/lambdarank_v2_20260811_cost_master_r4_adtvfix/model.joblib`

검증 결과: 대상 workflow 테스트 10 passed, `ruff check` PASS, `mypy` PASS.
