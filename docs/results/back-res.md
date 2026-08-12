# Stock Alpha v2 최신 81-Trial 실행 결과

## 실행 개요

| 항목 | 값 |
|---|---|
| 실행일 | 2026-08-12 |
| Snapshot | `research_provisional_20160104_20260812_cost_master_v3_mh2` |
| Artifact | `lambdarank_v2_20260812_redesign_81trial` |
| 실행 모드 | `research` |
| 실행 명령 | `uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260812_redesign_81trial --snapshot-id research_provisional_20160104_20260812_cost_master_v3_mh2 --mode research --optuna-trials 81 --max-rss-mib 8000` |
| Route | 5 / 10 / 15 sessions |
| Selection policy | `economic-selection-v1` |

실행은 외부 timeout 없이 정상 종료되었고, terminal artifact가 생성되었다.

## 최종 판정

| 항목 | 값 |
|---|---:|
| `promoted` | `false` |
| `no_trade` | `true` |
| `selection_status` | `no_economically_eligible_candidate` |
| `economically_eligible_trials` | `0` |
| `capacity_failure_reason` | `null` |
| `n_terminal_trials` | `81` |

모든 route에서 finalist가 경제성 gate를 통과하지 못해 승격 가능한 champion이
없었다. 이는 메모리 실패가 아니라 `no_filled_orders`와
`non_positive_bootstrap_lower_bound`에 의한 의도된 `NO_TRADE` 결과다.

## 소요시간 및 메모리

### 외부 프로세스 측정 (`/usr/bin/time -v`)

| 항목 | 값 |
|---|---:|
| Wall time | **19:54.38** |
| User time | 1,323.49 s |
| System time | 176.20 s |
| Maximum RSS | **7,672.23 MiB** |
| RSS limit | 8,000 MiB |
| Exit status | `0` |

### Workflow telemetry

| 항목 | 값 |
|---|---:|
| Baseline RSS | 1,777.50 MiB |
| Workflow peak RSS | **7,146.48 MiB** |
| Screen | 373.21 s |
| Full refit | 753.47 s |
| Economic replay | 282.45 s |
| Prepared replay decisions | 1,520 |
| Replay peak RSS | 5,481.50 MiB |
| Cache bytes | 1,795,296,912 bytes |

외부 RSS와 내부 RSS 모두 8,000 MiB 제한 안에 들어왔으며 capacity failure는
발생하지 않았다.

## 탐색·최적화 결과

| 항목 | 값 |
|---|---:|
| Route budget | 27 trials × 3 routes |
| Terminal trials | 81 |
| Screened / pruned | 66 / 15 |
| Promotion width | 6 per route |
| Finalist width | 2 per route |
| Promoted trials | 6 per route |
| All-positive finalists | 2 per route |
| Economic replays | 6 total |
| Legacy fixed shortlist reference | 8 per route |

정상 경로에서 full-refit은 route당 10개 fold 단위로 제한되었고, replay는 route당
최대 2개 finalist만 수행되었다.

| Route | Screen | Full refit | Replay | Peak RSS | Best screen Rank-IC |
|---:|---:|---:|---:|---:|---:|
| 5 | 147.18 s | 348.76 s | 165.50 s | 5,556.56 MiB | 0.08499058 |
| 10 | 116.94 s | 238.39 s | 71.21 s | 7,146.48 MiB | **0.09533526** |
| 15 | 109.08 s | 166.32 s | 45.74 s | 7,146.48 MiB | 0.08754609 |

## 경제 replay 성과

최종 finalist 6개 모두 주문 체결이 없었고 경제성 gate에서 탈락했다.

| Route | Trial | Median Rank-IC | Attempted orders | Filled orders | Bootstrap lower bound | Strategy IR | 판정 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 10 | 0.08875145 | 4,519 | 0 | -0.00038778 | 0.0 | 탈락 |
| 5 | 25 | 0.09093118 | 4,518 | 0 | -0.00038778 | 0.0 | 탈락 |
| 10 | 16 | 0.10159450 | 2,322 | 0 | -0.00044101 | 0.0 | 탈락 |
| 10 | 21 | 0.10537896 | 2,253 | 0 | -0.00044101 | 0.0 | 탈락 |
| 15 | 12 | 0.10713674 | 1,624 | 0 | -0.00043084 | 0.0 | 탈락 |
| 15 | 26 | 0.10447098 | 1,596 | 0 | -0.00043084 | 0.0 | 탈락 |

합계 attempted orders는 16,832건, filled orders는 0건이다. 따라서 최종 artifact의
투자 성과 지표는 생성되지 않았으며, paper/live 승격도 수행되지 않았다.

## 병목 관찰

- 전체 wall time은 19분 54초로 81-trial 실행이 terminal 상태에 도달했다.
- Screen 373초보다 full refit 753초와 replay 282초가 더 큰 비용이었다.
- 최고 RSS는 h10/h15 replay·refit 구간의 7,146.48 MiB telemetry,
  프로세스 전체 peak는 7,672.23 MiB였다.
- `decision_preparation`은 route별 bootstrap batch와 replay guard 안에서 수행되었고,
  capacity guard 중단 없이 fail-closed 경제성 판정을 완료했다.

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260812_redesign_81trial/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260812_redesign_81trial/manifest.json)
