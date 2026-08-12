# Stock Alpha v2 최신 ML 실행 결과

- 실행일: `2026-08-12`
- Snapshot: `research_provisional_20160104_20260227_cost_master_v2_r1`
- Artifact: `lambdarank_v2_20260812_net_alpha_remediation_run`
- 실행 모드: `research`
- 실행 명령: `uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260812_net_alpha_remediation_run --snapshot-id research_provisional_20160104_20260227_cost_master_v2_r1 --mode research --optuna-trials 80 --max-rss-mib 8000`
- 최종 결과: **NO_TRADE / 미승격**

## 탐색·자원

| 항목 | 수치 |
|---|---:|
| Optuna terminal trials | 80 |
| Screened trials | 71 |
| Pruned trials | 9 |
| Shortlisted trials | 8 |
| Economically eligible trials | 0 |
| Prepared cache | 106,295,948 bytes |
| Peak RSS | 4,532.6 MiB / 8,000 MiB |
| Screen time | 217.07 s |
| Full refit time | 282.60 s |
| Economic replay time | 109.64 s |
| Selection status | `no_economically_eligible_candidate` |

## Shortlist 수치

| 항목 | 범위 |
|---|---:|
| Median Rank-IC | 0.07462526 ~ 0.08017009 |
| Average expected net alpha | -0.00169929 ~ -0.00097477 |
| Bootstrap excess lower bound | -0.00094304 ~ -0.00075492 |
| Attempted orders | 0 ~ 190 |
| Filled orders | 0 ~ 190 |
| Planned cycles | 0 ~ 15 |
| Strategy IR | 0.0000 ~ 1.5308 |
| Turnover | 0.0000 ~ 0.9369 |
| Cash cycles | 59 ~ 74 |

최고 screen Rank-IC 후보는 trial 58이다.

| 항목 | trial 58 |
|---|---:|
| Fold Rank-IC | 0.07573805 / 0.08384901 |
| Median Rank-IC | 0.07979353 |
| Average expected net alpha | -0.00122868 |
| Bootstrap excess lower bound | -0.00094304 |
| Attempted / filled orders | 0 / 0 |
| Cash cycles | 74 |

실제 체결이 가장 많았던 shortlist 후보도 attempted/filled `190 / 190`이었지만,
average expected net alpha `-0.00106769`, bootstrap 하한 `-0.00075492`로 경제성
게이트를 통과하지 못했다.

## Rank-IC 해석

Rank-IC `0.07979`는 순위 예측력만 보면 양수이며, 두 fold가 모두 양수(`0.07574`,
`0.08385`)이므로 신호 방향성이 무작위보다 낫다는 근거는 있다. 다만 Rank-IC는
순위의 정합도이지 비용 후 수익률·체결 가능성·하방 안정성을 측정하지 않는다.

이번 실행에서는 상위 bucket의 expected active alpha가 양수인 경우도 있었지만,
round-trip cost `0.0036`을 차감한 expected net alpha가 음수였고 bootstrap 하한도
음수였다. 따라서 `0.07979`는 “학습이 전혀 안 됨”은 아니지만, 투자 가능한 수준으로
학습·검증되었다는 의미도 아니다. 현재 증거 기준으로는 모델 순위 신호는 존재하나
포트폴리오 경제성으로 전환되지 않았으며, `NO_TRADE` 판정이 올바르다.

## 최종 판정

- `promoted=false`
- `no_trade=true`
- `promotion_reasons=["no-champion-trial"]`
- Outer OOS·forward holdout 성과: 미생성
- Paper/live 사용: 금지

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260812_net_alpha_remediation_run/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260812_net_alpha_remediation_run/manifest.json)
- [Model](../../data/artifacts/stocks/lambdarank_v2_20260812_net_alpha_remediation_run/model.joblib)
