# ML 모델 비교 결과 (최신 실행)

실행일: **2026-08-29**
실행 모드: `--research-only-model-selection-study`
artifact: `ml-selection-runtime-live` (read-only, publish 안 함)
snapshot: `research_stock_net_alpha_v1_exec_20260828_10m`
판정: **채택 모델 없음 (`no-qualified-survivor`)**

## 1. 재현 조건

| 항목 | 값 |
|---|---:|
| horizon | H10 |
| rebalance frequency | 10 sessions |
| top-K | 12 |
| training lookback | 1,260 sessions |
| purged walk-forward folds | 3 |
| embargo | 5 sessions |
| bootstrap resamples | 360 |
| adjusted bootstrap alpha | 0.002777777778 |
| global wall budget | 900.0초 |
| screen phase budget | 720.0초 |
| 후보 family | 6개 |

실행 명령:

```bash
uv run python src/stocks/cli/train.py --artifact-id ml-selection-runtime-live --snapshot-id research_stock_net_alpha_v1_exec_20260828_10m --mode research --research-only-model-selection-study --candidate-horizon-sessions 10 --candidate-rebalance-frequency-sessions 10 --candidate-top-k 12 --candidate-training-lookback-sessions 1260 --fold-count 3 --embargo-sessions 5 --bootstrap-resamples 360 --model-selection-wall-clock-seconds 900 --model-selection-screen-phase-seconds 720 --model-selection-debug-timing
```

## 2. 데이터 규모 및 결측 현황

| 지표 | 값 |
|---|---:|
| feature rows | 918,443 |
| feature sessions | 2,479 |
| instruments | 2,297 |
| canonical feature columns | 29 |
| H10 label rows | 797,987 |
| H10 label sessions | 2,479 |
| label available-after-session 비율 | 1.000000 (100%) |
| target mean | 0.0876038048 |
| target std | 2.2982172356 |
| target positive fraction | 0.4992073806 (49.9207%) |
| reference cost mean | 0.0043643416 |

feature 결측률 상위 항목:

| feature | null fraction |
|---|---:|
| `ret_21_60d` | 14.5840% |
| `vol_regime` | 14.5840% |
| `volatility_60d` | 14.5840% |
| `bp_ratio` | 6.4456% |
| `ep_ratio` | 6.4456% |

## 3. 실행 자원 및 단계별 시간

`/usr/bin/time -v`와 study 내부 `runtime_ledger`를 함께 기록했다.

| 지표 | 외부 측정 | 내부 ledger |
|---|---:|---:|
| wall-clock | 338.76초 (5분 38.76초) | 334.591986916초 |
| 측정 차이 (CLI 초기화/정리 등) | 4.168013084초 | - |
| user CPU | 1,028.35초 | - |
| system CPU | 26.99초 | - |
| peak RSS | 4,620,320 KiB | - |
| peak RSS 환산 | 약 4.41 GiB (약 4.73 GB) | - |
| screen/cache elapsed | - | 334.591986916초 |
| screen fold count | - | 3 |
| screen learner fit count | - | 108 |
| attribution prediction count | - | 60 |
| model fit count | - | 18 |
| full OOF fit count | - | 0 |
| replay count | - | 0 |
| processed row count | - | 918,443 |
| cache hits | - | 3 |

이번 실행은 모든 후보가 screen admission에서 탈락했기 때문에 full OOF와 execution replay를 수행하지 않았다. 따라서 338.76초는 screen/cache 단계의 실제 비용이며, 900초 budget을 **561.24초 남겨 두고** 종료했다.

## 4. 모델별 screen 비교

`screen_lower_bound`는 비용을 차감한 session log-growth의 bootstrap lower bound이며 CAGR이 아니다. 최종 채택 기준과 동일하게 경제적 하한이 0보다 커야 full OOF에 진입할 수 있다.

| 순위 | family | lower bound | SE | lower bound + SE | 주요 선택 source group | full OOF |
|---:|---|---:|---:|---:|---|---|
| 1 | `elastic_net_v2` | -0.0106266646 | 0.0018385571 | -0.0087881075 | `flow_intensity_20d` | 미진입 |
| 2 | `huber_linear_v1` | -0.0123344078 | 0.0015930503 | -0.0107413575 | `flow_consensus` | 미진입 |
| 3 | `hist_gradient_quantile_v1` | -0.0124632129 | 0.0011887430 | -0.0112744699 | `relative_trend_score` | 미진입 |
| 4 | `rawnet_lgbm_v2` | -0.0125837819 | 0.0019730740 | -0.0106107079 | `ep_ratio`, `bp_ratio`, `flow_intensity_20d`, `disparity_120d`, `vpt_20d`, `ret_21_60d` | 미진입 |
| 5 | `extra_trees_v1` | -0.0132277176 | 0.0019416342 | -0.0112860835 | 20개 source group | 미진입 |
| 6 | `tail_lambdarank_v2` | -0.0132415453 | 0.0013347960 | -0.0119067494 | `bp_ratio` | 미진입 |

핵심 수치:

- 최선 후보는 ElasticNet이지만 lower bound가 **-0.0106266646**이다.
- 최선 후보도 0 기준보다 **0.0106266646** 낮다.
- 6개 모두 `screen_lower_bound <= 0`이며, 6개 전부 `screen-non-positive-lower-bound`로 기록됐다.
- screen 단계에서 양수 lower bound 후보가 0개이므로 one-SE 비교 및 full OOF 승격 대상도 0개다.

## 5. 최종 결과 JSON 판정

```json
{
  "status": "RESEARCH_ONLY",
  "study_complete": true,
  "next_action": "no-qualified-survivor",
  "selected_family": null,
  "survivors": [],
  "rejection_reason_counts": {
    "screen-non-positive-lower-bound": 6
  },
  "oof_fit_count": 0,
  "replay_count": 0,
  "elapsed_seconds": 334.591986916,
  "deadline_seconds": 900.0
}
```

이는 실행 실패나 budget 초과가 아니다. 현재 snapshot의 비용 반영 경제 증거가 양수인 family가 없어서, 추가 full OOF/replay를 수행하지 않고 fail-closed로 종료한 결과다. 따라서 운영 모델 채택, CAGR 확정, 앙상블 구성은 모두 보류한다.

## 6. 이전 900초 초과 원인과 최신 실행에서의 변화

이전 동일 계열 실행은 총 **961.132초**, ElasticNet full OOF **453.539초**, replay **3.466초**로 900초를 **61.132초** 초과했다. 당시에는 음수 screen 후보가 full OOF로 승격됐고, OOF 내부 deadline 경계가 부족했다.

이번 실행에서는 `screen_lower_bound > 0` admission gate가 먼저 적용돼:

1. 음수 경제 증거 6개를 full OOF로 승격하지 않음
2. `oof_fit_count=0`, `replay_count=0`
3. screen/cache 334.592초에서 종료
4. 외부 wall-clock 338.76초로 900초 이내 완료

따라서 최신 결과에서 확인되는 결론은 **“ElasticNet이 1위”가 아니라 “ElasticNet을 포함한 모든 후보가 경제적 채택 하한을 충족하지 못함”**이다.

## 7. 원본 산출물 보존 정책

실행 직후 JSON·stderr·snapshot 분석 스크립트로 수치를 검증했다. `sync` 절차에 따라 임시 `scratch/` 산출물은 삭제했으며, 필요한 수치와 실행 조건은 이 문서에 고정 보존했다.
