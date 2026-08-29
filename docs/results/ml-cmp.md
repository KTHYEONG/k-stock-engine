# ML 비교 결과 (최신)

## 실행 개요

| 항목 | 값 |
|---|---:|
| 실행일 | 2026-08-29 |
| artifact | `ml-selection-recovery-live` |
| snapshot | `research_stock_net_alpha_v1_exec_20260828_10m` |
| mode | `research-only-model-selection-study` |
| route | `unhedged_absolute` |
| horizon (H) | 10 sessions |
| rebalance cadence (C) | 10 sessions |
| Top-K | 12 |
| training lookback | 1,260 sessions |
| purged folds | 3 |
| embargo | 5 sessions |
| bootstrap resamples | 360 |
| adjusted bootstrap alpha | 0.002777777778 |
| screen / global budget | 720 / 900 sec |
| 결과 | `RESEARCH_ONLY`, `no-qualified-survivor` |
| artifact publish | false |

실행 명령:

```bash
uv run python src/stocks/cli/train.py --artifact-id ml-selection-recovery-live --snapshot-id research_stock_net_alpha_v1_exec_20260828_10m --mode research --research-only-model-selection-study --candidate-horizon-sessions 10 --candidate-rebalance-frequency-sessions 10 --candidate-top-k 12 --candidate-training-lookback-sessions 1260 --fold-count 3 --embargo-sessions 5 --bootstrap-resamples 360 --model-selection-wall-clock-seconds 900 --model-selection-screen-phase-seconds 720 --model-selection-debug-timing
```

## 데이터 및 비용 증거

| 지표 | 값 |
|---|---:|
| feature rows | 918,443 |
| feature sessions | 2,479 |
| instruments | 2,297 |
| screen design columns | 87 |
| reference cost mean | 0.0043643416 |
| target mean / std | 0.0876038048 / 2.2982172356 |
| target positive fraction | 0.4992073806 |
| 상위 결측 feature | `ret_21_60d`, `vol_regime`, `volatility_60d` (각 14.5840%) |
| 추가 결측 feature | `bp_ratio`, `ep_ratio` (각 6.4456%) |

unhedged screen utility는 모든 후보에 대해 `gross_return - reference_cost`를
정확히 한 번만 적용했다. Oracle은 같은 세션에서 실제 utility 기준 Top-12를
선택한 상한이며, 모델 점수의 사후 수익률을 의미하지 않는다.

## 모델별 screen 결과

`screen_lower_bound`는 비용 반영 session tail-excess bootstrap lower bound다.
모든 후보가 0 이하이므로 full OOF와 replay에 진입하지 않았다.

| 순위 | family | absolute LB | tail-excess LB (`screen_lower_bound`) | SE | oracle tail LB | 선택 prefix | 주요 source groups | 판정 |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `elastic_net_v2` | -0.027751 | **-0.004583** | 0.005876 | +0.134158 | 6 | `info_ratio_20d`, `flow_intensity_20d`, `bp_ratio`, `vpt_20d`, `intraday_ret`, `ep_ratio` | 탈락 |
| 2 | `huber_linear_v1` | -0.029357 | **-0.005801** | 0.006271 | +0.134158 | 6 | `flow_consensus`, `foreign_net_buy`, `institution_net_buy`, `ret_21_60d`, `bp_ratio`, `info_ratio_20d` | 탈락 |
| 3 | `rawnet_lgbm_v2` | -0.031875 | **-0.006986** | 0.005255 | +0.134158 | 6 | `ep_ratio`, `bp_ratio`, `vpt_20d`, `relative_trend_score`, `ret_21_60d`, `disparity_120d` | 탈락 |
| 4 | `extra_trees_v1` | -0.034348 | **-0.009791** | 0.007004 | +0.134158 | 1 | `ep_ratio` | 탈락 |
| 5 | `tail_lambdarank_v2` | -0.027907 | **-0.010453** | 0.006456 | +0.134158 | 11 | `bp_ratio`, `ep_ratio`, `disparity_120d`, `flow_intensity_20d`, `relative_trend_score`, `vpt_20d`, `ret_6_20d`, `vol_asymmetry_20d`, `ret_21_60d`, `info_ratio_20d`, `close_high_ratio_10d` | 탈락 |
| 6 | `hist_gradient_quantile_v1` | -0.031078 | **-0.011162** | 0.006282 | +0.134158 | 1 | `ret_6_20d` | 탈락 |

### Fold별 관측

- fold 0: LambdaRank 예시 prefix tail LB `-0.007 ~ -0.008`, oracle `+0.128`.
- fold 1: LambdaRank 예시 prefix tail LB `-0.004 ~ -0.015`, oracle `+0.120`.
- fold 2: LambdaRank 예시 prefix tail LB `-0.019 ~ -0.038`, oracle `+0.154`.
- 모든 fold에서 oracle tail bound가 양수였지만 모델 tail bound는 일관되게
  음수였다. 이는 기회집합 부재가 아니라 횡단면 ranking 복원 실패를 뜻한다.
- DEBUG 로그는 실제 `fold_id=0,1,2`를 기록하며, aggregate JSON은 물리적 fold가
  없으므로 `fold_id=null`, `aggregate_fold_id=null`로 기록된다.

## 실행 자원 및 승격 결과

| 지표 | 값 |
|---|---:|
| wall-clock (internal ledger) | 554.772004359 sec |
| screen fold count | 3 |
| screen learner fit count | 108 |
| model fit count | 18 |
| attribution prediction count | 63 |
| cache hits | 3 |
| full OOF fit count | 0 |
| replay count | 0 |
| next action | `no-qualified-survivor` |

## 결론 및 다음 실험

1. 현재 H10/C10/K12 조건에서 채택 가능한 모델은 없다. `elastic_net_v2`가
   최선이지만 corrected tail lower bound가 -0.004583으로 기준 0보다 낮다.
2. Oracle lower bound +0.134158은 데이터의 선택 가능 신호가 있음을 보인다.
   따라서 다음 개선의 우선순위는 scoring objective/feature alignment이며,
   단순히 더 많은 OOF나 replay를 실행하는 것이 아니다.
3. 다음 recovery run에서는 fold-local median imputation과 missing indicator를
   함께 사용하고, linear 계열에만 `flow_intensity × vol_regime`,
   `flow_consensus × relative_trend` rank interaction을 허용한다.
4. LambdaRank는 global median 대신 세션별 정확한 Top-K relevance를 사용해야
   한다. 이후 H5/C5/K12, H10/C10/K12, H20/C10/K12를 각각 독립 실행하고,
   완료된 `(family, horizon, profile)` 전체에 multiplicity correction을 적용한다.
5. 양수 corrected tail 및 oracle 증거를 동시에 만족하는 후보만 full OOF와
   base/stress exact replay 대상으로 승격한다.

상세 DEBUG 실행 로그: `scratch/ml_selection_latest.log`
