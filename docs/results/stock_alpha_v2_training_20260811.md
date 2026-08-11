# Stock Alpha v2 학습 결과

- 실행일: 2026-08-11
- 목적: 국내주식 연구용 LambdaRank + StableRankComposite 학습 및 승격 게이트 평가
- 범위: 연구용 provisional 데이터만 사용, 실전 매매 연동 제외
- 최종 판정: **NO_TRADE / 미승격**

## 실행 대상

### Snapshot

- Snapshot ID: `research_provisional_20160104_20260227_cost_master_v2_r1`
- Certification: `provisional`
- 연구 범위: `2016-01-04 .. 2026-02-27`
- Feature dataset: `krx_features_stock_alpha_v2_provisional_20160104_20260310_r1`
- Label dataset: `krx_labels_residual_o2o_5d_provisional_20160104_20260227_r1`
- Feature rows: `932,193`
- Label rows: `849,482`
- Feature coverage threshold: `75%`
- Snapshot catalog validation: PASS

### Label contract

- Definition: `residual_o2o_5d`
- Horizon: 5 sessions
- Entry: decision session 다음 세션 시가
- Exit: entry 후 5 sessions 시가
- Relevance: 세션별 초과수익 percentile을 `0..4`로 변환
- Label availability: exit session open 이후

## 학습 설정

- Artifact ID: `lambdarank_v2_20260811_cost_master_r2`
- Model: LightGBM LambdaRank 50% + StableRankComposite 50%
- Feature count: 34
- Objective: `lambdarank`
- Evaluation: `NDCG@10`, `NDCG@20`
- Label gain: `(0, 1, 3, 7, 15)`
- Seed: `42`
- Max estimators: `5,000`
- Early stopping: `200`
- Optuna trials: `80`
- Best trial: `79`
- Best development Rank-IC: `0.07694677`
- Evaluated folds: `1`

## 성과 및 승격 게이트

| 항목 | 결과 | 판정 |
|---|---:|---|
| 최종 median Rank-IC | 0.08836402 | 참고 지표 |
| 양의 fold IC 비율 | 1.0000 | 통과 |
| 초과수익 bootstrap 하한 | -0.00017468 | 실패 |
| 전략 IR | 0.000000 | 실패 |
| 벤치마크 IR | -0.302096 | 참고 |
| CAGR | 0.000000 | 유효 성과 없음 |
| 연환산 변동성 | 0.000000 | 유효 성과 없음 |
| MDD | 0.000000 | 유효 성과 없음 |
| Turnover | 0.000000 | 유효 성과 없음 |
| Cost drag | 0.000000 | 유효 성과 없음 |
| Stress cost gate | true | 통과 |
| Deflated Sharpe probability | 0.000000 | 실패 |
| 252-session forward holdout | 준비 안 됨 | 실패 |

최종 artifact metrics:

```json
{
  "promoted": false,
  "no_trade": true,
  "n_folds_evaluated": 1,
  "median_rank_ic": 0.0883640193512458,
  "promotion_reasons": [
    "gate1_positive_rank_ic_fraction=1.0000",
    "gate2_excess_lower_bound=-0.00017468",
    "gate3_strategy_ir=0.000000",
    "gate5_deflated_sharpe_probability=0.000000",
    "gate8_forward_holdout_ready=False"
  ]
}
```

## 해석

피처와 라벨 계약은 정상적으로 연결됐고, 모델의 개발 순위 예측력은
양수였습니다. 그러나 현재 결과는 단일 fold 평가이며, 이벤트 ledger의
전략 exposure·turnover가 0으로 기록됐습니다. 따라서 경제적 수익성과
체결 가능성을 입증한 결과가 아니며, 모델은 의도대로 `NO_TRADE`로
발행됐습니다.

이 결과를 수익률 성과로 해석하거나 실전 매매에 사용하는 것은 금지합니다.

## 다음 검증 작업

1. 학습 workflow를 다중 walk-forward fold로 실행해 fold별 IC 안정성을 확인합니다.
2. 이벤트 backtester에서 주문 생성·체결·포지션 ledger가 0 exposure가 되는 원인을 확인합니다.
3. 2026-03-10 이후 252개 세션의 신규 데이터를 확보해 고정 forward holdout을 구성합니다.
4. 기본 비용·스트레스 비용·bootstrap 하한·DSR을 다시 계산합니다.
5. 모든 게이트 통과 전까지 artifact는 `NO_TRADE`로 유지합니다.

## 산출물

- Metrics: `data/artifacts/stocks/lambdarank_v2_20260811_cost_master_r2/metrics.json`
- Manifest: `data/artifacts/stocks/lambdarank_v2_20260811_cost_master_r2/manifest.json`
- Model: `data/artifacts/stocks/lambdarank_v2_20260811_cost_master_r2/model.joblib`
- Snapshot manifest: `data/catalog/stocks/snapshots/research_provisional_20160104_20260227_cost_master_v2_r1/snapshot_manifest.json`

## 코드 검증

- `uv run pytest`: PASS
- `uv run ruff check .`: PASS
- `uv run mypy .`: PASS
- 폴드 최소 학습구간 회귀 테스트: PASS
