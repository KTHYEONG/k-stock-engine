# Stock Alpha v2 최신 ML 실행 결과

## 실행 개요

| 항목 | 값 |
|---|---|
| 실행일 | 2026-08-12 |
| Snapshot | `research_provisional_20160104_20260227_cost_master_v2_r1` |
| Artifact | `lambdarank_v2_20260812_replay_batched_1trial` |
| 실행 모드 | `research` |
| 실행 명령 | `uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260812_replay_batched_1trial --snapshot-id research_provisional_20160104_20260227_cost_master_v2_r1 --mode research --optuna-trials 1 --max-rss-mib 8000` |
| Route | 5 sessions (`residual_o2o_5d`) |
| Rebalance | 5 sessions |

이번 실행은 bootstrap workspace를 결정론적 batch 방식으로 제한한 후 `INNER_SELECTION_BASE_ONLY` 경제 replay까지 정상 완료했다. 이전의 `decision_preparation` 용량 가드 중단은 재현되지 않았다.

## 최종 판정

- `promoted=false`
- `no_trade=true`
- `promotion_reasons=["no-champion-trial"]`
- `selection_status=no_economically_eligible_candidate`
- `capacity_failure_reason=null`

NO_TRADE의 원인은 메모리 부족이 아니라 경제성 gate이다. 유일한 shortlist 후보가 `non_positive_bootstrap_lower_bound` 조건을 통과하지 못해 champion으로 선정되지 않았다. 따라서 paper/live 승격은 수행되지 않는다.

## 실행·자원 telemetry

| 항목 | 값 |
|---|---:|
| Optuna terminal trials | 1 |
| Screened / pruned trials | 1 / 0 |
| Shortlisted trials | 1 |
| Economically eligible trials | 0 |
| Screen time | 3.4047 s |
| Full refit time | 360.1087 s |
| Economic replay time | 198.5954 s |
| 주요 단계 합계 | 562.1088 s |
| Baseline RSS | 1,853.5 MiB |
| Workflow peak RSS | **6,253.6 MiB** |
| RSS limit | 8,000 MiB |
| Replay baseline RSS | 5,007.0 MiB |
| Replay peak RSS | **5,622.2 MiB** |
| Replay prepared decisions | 419 |
| Replay mode | `INNER_SELECTION_BASE_ONLY` |

### 메모리 최적화 효과

| Replay stage | 값 |
|---|---:|
| Replay market index | 347,519,119 bytes |
| Candidate score join | 411,590,033 bytes |
| Replay ADTV | 86,472,588 bytes |
| Decision preparation | **3,683,225,197 bytes** |
| Bootstrap workspace | 3,661,800,000 bytes |
| Bootstrap batch size | 200 draws |

기존 실행의 `decision_preparation` 예상치 14,301,256,442 bytes와 비교하면 이번 admission 규모는 약 **74.2% 감소**한 3.68GB이다. 8GB RSS 예산 안에서 실제 calibration/replay가 실행되었고, capacity guard는 fail-closed 상태를 유지했다.

## Screen 및 refit 결과

| 항목 | 값 |
|---|---:|
| Best screen Rank-IC | 0.06937652 |
| Fold Rank-IC | 0.06937652 / 0.08959503 / 0.08954546 |
| Median Rank-IC | 0.08954546 |
| Full-refit fold 0 / 1 / 2 | 7.2302 s / 80.0665 s / 74.0915 s |
| Fold 0 / 1 / 2 allocation estimate | 93.72 / 386.37 / 801.55 MiB |
| Replay finite | true |

## 경제 replay 및 후보 evidence

| 항목 | 값 |
|---|---:|
| Attempted orders | 2,104 |
| Filled orders | 2,104 |
| Planned cycles | 196 |
| Cash cycles | 200 |
| Strategy IR | 0.52473263 |
| Max drawdown | 0.16862692 |
| Turnover | 3.84779217 |
| Cost drag | 0.0039456708 |
| Bootstrap lower bound | **-0.00013084** |
| Average expected net alpha | -0.0004840764 |
| Candidate eligible | false |
| Candidate failure | `non_positive_bootstrap_lower_bound` |

Non-trade cycle 사유는 `no-feasible-allocation` 200회, `constraint:insufficient covariance data` 23회였다. 주문 자체는 2,104건 시도되어 모두 체결되었지만, bootstrap lower bound가 0보다 작아 경제성 gate에서 탈락했다.

## Calibration 상태

| 항목 | 값 |
|---|---:|
| Calibration history sessions | 2,085 |
| Bucket count | 10 |
| Minimum calibration sessions | 126 |
| Bootstrap draws | 200 |
| Bootstrap alpha | 0.05 |
| Block length | 5 |
| Participation limit | 0.01 |
| Round-trip cost | 0.0036 |
| Exit cost rate | 0.00295 |
| Eligible buckets | 6 / 10 |

양의 calibration evidence가 생성된 bucket은 4~9번이다.

| Bucket | Sample size | Expected active alpha | Alpha lower bound |
|---:|---:|---:|---:|
| 0 | 76,378 | null | null |
| 1 | 75,385 | null | null |
| 2 | 75,470 | null | null |
| 3 | 75,343 | null | null |
| 4 | 75,075 | 0.0007504979 | 0.0003183864 |
| 5 | 75,841 | 0.0025007707 | 0.0020641410 |
| 6 | 75,440 | 0.0028084588 | 0.0024094793 |
| 7 | 75,345 | 0.0032551659 | 0.0027702112 |
| 8 | 75,447 | 0.0040968520 | 0.0037170241 |
| 9 | 76,609 | 0.0052837966 | 0.0048736853 |

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260812_replay_batched_1trial/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260812_replay_batched_1trial/manifest.json)
