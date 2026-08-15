# 2026-08-15 최신 net-alpha ML 실행 결과

최신 수정 코드의 실행 결과만 기록한다. 이전 실행 기록은 제거했다.

## 실행 요약

| 항목 | 값 |
|---|---|
| Snapshot | `research_provisional_20160104_20260814_net_alpha_v1_run8` |
| 전체 후보 horizon 실행 | 3, 5, 8, 10, 15, 20 sessions — 20분 이상 소요되어 artifact 생성 전 중단 |
| 완료 실행 | 3-session 단일 horizon |
| Artifact | `net_alpha_20260815_validation_run2_h3` |
| seed / model threads | `7` / `1` |
| 종료 상태 | 완료 (`exit 0`) |

## 완료 실행 결과

| 지표 | 결과 |
|---|---|
| `model_type` | `no_trade` |
| `promoted` | `false` |
| `no_trade` | `true` |
| promotion reason | `no-horizon-evidence` |
| horizon | `3 sessions` |
| fold score std | `0.110006, 0.240101, 0.235004` |
| fold rank IC | `0.055391, -0.003490, 0.038208` |
| selected alpha fraction | `0.30` for all three folds |
| selected alpha | `0.042003, 0.038433, 0.032480` |
| OOF economic replay | 최소 유효 주문 block 기준 미충족 |

예측 점수는 더 이상 상수로 붕괴하지 않았고 세 fold 모두 유효한 점수 변동과
진단을 생성했다. 그러나 비용·유동성·causal calibration을 적용한 정책 replay가
최소 경제 evidence를 충족하지 못했으므로 보수적으로 거래를 차단했다. 이는
모델 성과가 0이라는 의미가 아니라, 검증 가능한 champion이 없다는 의미다.

## 산출물

- [Metrics](../../data/artifacts/stocks/net_alpha_20260815_validation_run2_h3/metrics.json)
- [Manifest](../../data/artifacts/stocks/net_alpha_20260815_validation_run2_h3/manifest.json)
