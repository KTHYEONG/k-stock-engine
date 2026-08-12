# 최신 Stock Alpha v2 81-Trial 실행 결과

## 실행 식별자

| 항목 | 값 |
|---|---|
| 실행일 | 2026-08-12 |
| Snapshot | `research_provisional_20160104_20260812_cost_master_v3_mh2` |
| Artifact | `lambdarank_v2_20260812_redesign_81trial_postsync` |
| 모드 | `research` |
| 정책 | `economic-selection-v2-proxy-one-finalist` |
| 명령 | `uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260812_redesign_81trial_postsync --snapshot-id research_provisional_20160104_20260812_cost_master_v3_mh2 --mode research --optuna-trials 81 --max-rss-mib 8000` |

이번 실행은 외부 timeout 없이 exit status `0`으로 종료되었고, 3개 route(5/10/15 sessions)에 각각 27 trials를 배정했다.

## 최종 판정

| 항목 | 결과 |
|---|---:|
| `promoted` | `false` |
| `no_trade` | `true` |
| `selection_status` | `no_economically_eligible_candidate` |
| `n_folds_evaluated` | 0 |
| 경제성 통과 후보 | 0 |
| terminal trials | 81 |
| screened / pruned | 72 / 9 |
| shortlisted | 18 |
| 최고 screen Rank-IC | **0.10341699** |

모든 finalist가 `no_filled_orders` 및 `non_positive_bootstrap_lower_bound` gate에서 탈락했다. 따라서 모델 artifact는 생성됐지만 champion 승격이나 paper/live 거래 전환은 수행되지 않았다.

## 실행 시간 및 메모리

### 외부 프로세스 측정

| 항목 | 값 |
|---|---:|
| Wall time | **16:07.07** |
| User time | 1,093.59 s |
| System time | 78.30 s |
| Maximum RSS | **7,613.02 MiB** |
| RSS hard limit | 8,000 MiB |
| Exit status | 0 |

### Workflow telemetry

| 항목 | 값 |
|---|---:|
| Baseline RSS | 1,559.38 MiB |
| Workflow peak RSS | **6,919.41 MiB** |
| Screen | 87.50 s |
| Full refit | **786.26 s** |
| Economic replay | 52.69 s |
| Cache bytes | 1,795,296,912 bytes |
| Proxy session stride | 6 |
| Route budget | 27 |
| Promotion width | 6 |
| Finalist width | 1 |

외부 RSS는 workflow telemetry보다 693.61 MiB 높았지만 8,000 MiB 제한 아래였고 OOM 또는 capacity failure는 발생하지 않았다. 다만 목표인 600초/7,000MiB는 각각 967.07초, 613.02MiB 초과했다.

## 경제 replay 성과

| Route | Finalist trial | Median Rank-IC | Attempted orders | Filled orders | Bootstrap lower bound | Strategy IR | 판정 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 sessions | 5 | 0.08661119 | 4,419 | 0 | -0.00040279 | 0.0 | 탈락 |
| 10 sessions | 26 | 0.10283674 | 2,302 | 0 | -0.00040279 | 0.0 | 탈락 |
| 15 sessions | 19 | 0.10164764 | 1,101 | 0 | -0.00040279 | 0.0 | 탈락 |
| **합계** | — | — | **7,822** | **0** | — | — | **NO_TRADE** |

예측 순위 상관은 양호했지만 실제 체결이 한 건도 없어 strategy IR은 0.0이었다. 음수 bootstrap 하한이 확인되어 경제성 gate를 완화하지 않고 fail-closed 처리한 결과다.

## 병목 및 성과 해석

- 탐색(screen)은 87.50초로 전체 시간의 약 9%다.
- full refit이 786.26초로 가장 큰 병목이며, 전체의 약 82%를 차지한다.
- economic replay는 52.69초로 약 5%다.
- 81 terminal trials와 18 shortlist evidence는 정상적으로 기록됐다.
- 메모리 측면에서는 OOM을 피했지만 외부 peak 7,613MiB로 7,000MiB 운영 목표에는 미달했다.
- 최적화의 다음 우선순위는 full refit fold 수/round budget 및 후보 materialization 동시 메모리 축소이며, 경제성 gate는 현재 결과상 완화할 근거가 없다.

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260812_redesign_81trial_postsync/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260812_redesign_81trial_postsync/manifest.json)
