# 2026-08-14 최신 net-alpha ML / 백테스트 실행 결과

현재 CLI 기본 경로(`stock_net_alpha_v1`)로 materialize와 학습을 새로 실행한 결과만 기록한다.
기존 LambdaRank·고정 5/10/15 결과는 이 문서에서 제거했다.

## 실행 요약

| 항목 | 값 |
|---|---|
| Snapshot | `research_provisional_20160104_20260814_net_alpha_v1_run8` |
| Artifact | `net_alpha_20260814_mainline_run12` |
| 후보 horizon | `3, 5, 8, 10, 15, 20 sessions` |
| feature rows | 932,193 |
| label rows | 4,823,274 (horizon별 독립 universe) |
| 학습 명령 | `uv run python -m src.stocks.cli.train ... --model-threads 1 --seed 7` |
| 종료 상태 | 정상 종료 (`exit 0`) |

## 소요시간 / RAM

`/usr/bin/time -v`가 측정한 학습 프로세스 기준이다.

| 항목 | 값 |
|---|---:|
| Wall time | **45.83초** |
| User / system CPU | 82.53초 / 31.11초 |
| Peak RSS | **6,123.3 MiB** (6,270,252 KiB) |
| RSS limit | 8,000 MiB |

## ML 및 백테스팅 성과

모델 성과 수치는 생성되지 않았다. 무결성 검사를 통과한 뒤 각 horizon의 정책 replay
evidence가 최소 block 수를 충족하지 못해, 경제성 검증 가능한 후보가 없어 보수적으로
`no_trade` artifact를 발행했다.

| 지표 | 결과 |
|---|---|
| `model_type` | `no_trade` |
| `promoted` | `false` |
| `no_trade` | `true` |
| horizon selection | 미선정 (`no-horizon-evidence`) |
| Fold Rank-IC / bootstrap CI | 산출 전 중단 |
| 체결 수·수익률·IR·MDD | 산출 전 중단 (백테스트 champion 없음) |

따라서 이번 결과는 “성능이 0”이라는 의미가 아니라, 비용·유동성·point-in-time
가용성·정책 replay evidence를 충족한 모델이 없어 거래를 차단한 결과다. ML/백테스트
성과를 주장할 수 있는 champion은 현재 없다.

## 산출물

- [Metrics](../../data/artifacts/stocks/net_alpha_20260814_mainline_run12/metrics.json)
- [Manifest](../../data/artifacts/stocks/net_alpha_20260814_mainline_run12/manifest.json)
- [Materialization log](../../scratch/net_alpha_materialize_run8.log)
- [Training resource log](../../scratch/net_alpha_train_run12.time)
