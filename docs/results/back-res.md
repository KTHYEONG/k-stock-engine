# 2026-08-14 최신 ML / 백테스트 실행 결과

현재 수정 코드로 실행한 최신 결과만 기록한다. 기존 completion-validation 결과는
삭제하고, partial checkpoint를 `--resume`으로 이어서 81-trial 실행을 완료했다.

## 실행

| 항목 | 값 |
|---|---|
| Artifact | `lambdarank_v2_20260814_redesign_validation` |
| Snapshot | `research_provisional_20160104_20260812_cost_master_v3_mh2` |
| Command | `uv run python -m src.stocks.cli.train --artifact-id lambdarank_v2_20260814_redesign_validation --snapshot-id research_provisional_20160104_20260812_cost_master_v3_mh2 --mode research --optuna-trials 81 --max-rss-mib 8000 --resume` |
| 실행 완료 | `2026-08-14 16:22:45 KST` |
| Optuna terminal trials | **81** |
| Screened / pruned | **28 / 53** |
| Confirmation attempts / confirmed | **6 / 6** |
| Exact replay | **3** |
| Resume wall time | 약 **7분 18초** |

## 자원 사용량

| 항목 | 값 |
|---|---:|
| Screen | **28.834 s** |
| Full refit | **168.558 s** |
| Economic replay | **9.370 s** |
| Baseline RSS | **1,776.312 MiB** |
| Peak RSS | **6,062.648 MiB** |
| RSS limit | `8,000 MiB` |

## Route별 exact 백테스트

| Route | Fold Rank-IC | Bootstrap lower bound | Strategy IR | MDD | 체결 |
|---:|---|---:|---:|---:|---:|
| 5 sessions | 0.083128 / 0.109571 / 0.088390 | **-0.00197981** | 0.372769 | 23.297% | 1,376 / 1,376 |
| 10 sessions | 0.095408 / 0.103027 / 0.092571 | **-0.00492472** | 0.198844 | 26.754% | 809 / 809 |
| 15 sessions | 0.096843 / 0.140104 / 0.104314 | **-0.01011171** | 1.076857 | 2.635% | 74 / 74 |

모든 exact 후보에서 주문은 **2,259 / 2,259건 체결**됐다. 그러나 세 후보 모두
bootstrap lower bound가 음수이고 DSR 승격 기준을 충족하지 못했다.

## 최종 판정

| 항목 | 값 |
|---|---|
| `promoted` / `no_trade` | `false / true` |
| `selection_status` | `no_economically_eligible_candidate` |
| Economically eligible trials | **0** |
| Promotion reason | `no-champion-trial` |
| 최종 champion | 없음 |

Rank-IC는 모든 route에서 양수였지만, 실제 비용·체결·bootstrap·DSR을 함께 적용한
경제성 검증은 실패했다. 따라서 최신 코드도 배포 가능한 champion을 생성하지 않고
보수적으로 `no_trade`로 종료했다.

## 산출물

- [Metrics](../../data/artifacts/stocks/lambdarank_v2_20260814_redesign_validation/metrics.json)
- [Manifest](../../data/artifacts/stocks/lambdarank_v2_20260814_redesign_validation/manifest.json)
- 실행 로그는 sync 정리 정책에 따라 삭제했으며, metrics/manifest를 영구 산출물로 보존한다.
