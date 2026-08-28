# ML 모델 비교 결과

실행일: 2026-08-28<br>
실행 모드: `--research-only-model-selection-study`<br>
초기자본/reference notional: **10,000,000 KRW**<br>
artifact: `mlcmp_full_report_unbounded`<br>
snapshot: `research_stock_net_alpha_v1_exec_20260828_10m`

## 실행 조건

- 후보 family: 6개 (`elastic_net_v2`, `huber_linear_v1`, `extra_trees_v1`, `hist_gradient_quantile_v1`, `rawnet_lgbm_v2`, `tail_lambdarank_v2`)
- horizon/lookback: H10 / 1260 sessions
- rebalance/top-K: 10 sessions / 12
- purged walk-forward: 3 folds, embargo 5 sessions
- feature rows: 918,443
- adjusted bootstrap alpha: `0.002777777778`
- bootstrap resamples: 360
- study는 read-only이며 artifact를 publish하지 않음

## 실행 자원 및 단계 결과

| 항목 | 결과 |
|---|---:|
| Wall-clock time | 11분 24.83초 |
| model-selection runtime ledger | 680.67초 |
| Peak RSS | 5,924,936KB (약 5.65GiB) |
| Screen fold count | 3 |
| Screen learner fits | 108 |
| Attribution predictions | 60 |
| Full OOF fits | 1 |
| Replay count | 0 (1회 시도 후 오류) |
| 최종 상태 | `RESEARCH_ONLY` |
| 다음 조치 | `no-qualified-survivor` |

## Screening 비교

`screen_lower_bound`는 비용 차감 후 session log-growth의 bootstrap lower bound이며 CAGR이 아니다. 값이 클수록 screening 단계의 경제적 근거가 강하다.

| 순위 | Family | Screen lower bound | SE | Full OOF | 주요 선택 source group |
|---:|---|---:|---:|---|---|
| 1 | `elastic_net_v2` | -0.008468 | 0.001598 | 진입 | `flow_intensity_20d` |
| 2 | `huber_linear_v1` | -0.010970 | 0.001341 | 미진입 | `flow_consensus` |
| 3 | `extra_trees_v1` | -0.011387 | 0.001646 | 미진입 | 20개 group 전체 |
| 4 | `hist_gradient_quantile_v1` | -0.011403 | 0.001077 | 미진입 | 8개 group |
| 5 | `tail_lambdarank_v2` | -0.011775 | 0.001226 | 미진입 | `bp_ratio` |
| 6 | `rawnet_lgbm_v2` | -0.012143 | 0.001642 | 미진입 | 15개 group |

Screening에서는 ElasticNet이 가장 높았고, one-SE/상위 family 제한에 따라 ElasticNet만 full OOF 대상으로 승격됐다. 각 family의 source-group attribution은 outer-fold training schema에서 계산됐으며 XGBoost는 독립 후보로 포함하지 않았다.

## Full OOF 및 replay 결과

ElasticNet에 대해 full OOF fit 1회가 수행됐으나, base/stress execution replay 단계에서 `ValueError`가 발생했다.

- `study_complete`: `false`
- `selected_family`: `null`
- `survivors`: `[]`
- `rejection_reason_counts`: `{ "replay-failed:ValueError": 1 }`
- base/stress lower CAGR, MDD, fill rate: **산출되지 않음**

따라서 이번 실행은 6개 family의 screening 비교에는 성공했지만, 실제 거래 ledger를 통과한 최종 ML이나 복리자산증식 성과를 확정한 결과는 아니다. ElasticNet을 최종 모델로 채택하려면 replay `ValueError`의 원인을 해결한 뒤 동일 snapshot에서 base/stress ledger를 재실행해야 한다.

## 판정

현재 데이터로 확정 가능한 결론은 **screening 1위가 `elastic_net_v2`라는 것뿐**이다. Replay가 실패했으므로 운영 승격, CAGR 비교, 앙상블 채택은 보류한다. 모든 최종 선택은 비용·유동성·체결·MDD 조건을 포함한 full execution ledger가 생성된 경우에만 허용된다.
