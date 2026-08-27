# Compound Alpha 비교 결과

실행일: 2026-08-27  
실행 모드: `--research-only-compound-alpha-study`  
후보 수: 24개 (`B00` 기준선 + `C01`~`C23`)

## 무엇을 비교했나

- `B00`: 기존 ElasticNet + calibration 기준선
- `C01`~`C03`: ElasticNet feature/weight/sector 변형
- `C04`~`C09`: LightGBM 평균·q20·downside·seed bagging 변형
- `C10`~`C12`: LambdaRank tail 및 RawNet 변형
- `C13`~`C16`: 시장 국면·sector·유동성 조건부 변형
- `C17`~`C20`: horizon/rank/mean/q20 앙상블 변형
- `C21`~`C23`: 비용 인식 및 sparse transition 변형

판정 기준은 기준선 대비 base/stress lower CAGR 개선폭, matched excess CAGR, MDD, 체결/관측 coverage다. 두 환경 모두 기준선 대비 최소 `+10%p` 개선되어야 승격 가능하다.

## 출력된 비교값

| 후보 | Base lower CAGR | Stress lower CAGR | 기준선 대비 (base/stress) |
|---|---:|---:|---:|
| B00 | 4.00% | 3.50% | 기준선 |
| C01 | 2.82% | 8.63% | -1.18%p / +5.13%p |
| C02 | 11.73% | 10.67% | +7.73%p / +7.17%p |
| C03 | 6.68% | 2.75% | +2.68%p / -0.75%p |
| C04 | 8.33% | 12.31% | +4.33%p / +8.81%p |
| C05 | 5.98% | 6.11% | +1.98%p / +2.61%p |
| C06 | 5.40% | 9.42% | +1.40%p / +5.92%p |
| C07 | 10.71% | 8.07% | +6.71%p / +4.57%p |
| C08 | 3.83% | 15.61% | -0.17%p / +12.11%p |
| C09 | 10.67% | 5.83% | +6.67%p / +2.33%p |
| C10 | **16.60%** | **14.41%** | **+12.60%p / +10.91%p** |
| C11 | 12.40% | 14.23% | +8.40%p / +10.73%p |
| C12 | 12.49% | 5.63% | +8.49%p / +2.13%p |
| C13 | 9.71% | 4.04% | +5.71%p / +0.54%p |
| C14 | 10.81% | 1.99% | +6.81%p / -1.51%p |
| C15 | 8.79% | 10.61% | +4.79%p / +7.11%p |
| C16 | 15.83% | 9.88% | +11.83%p / +6.38%p |
| C17 | 8.06% | 7.16% | +4.06%p / +3.66%p |
| C18 | 12.59% | 11.48% | +8.59%p / +7.98%p |
| C19 | 8.58% | 12.18% | +4.58%p / +8.68%p |
| C20 | 16.12% | 4.37% | +12.12%p / +0.87%p |
| C21 | 9.77% | 5.42% | +5.77%p / +1.92%p |
| C22 | 17.36% | 11.39% | +13.36%p / +7.89%p |
| C23 | 4.35% | 13.01% | +0.35%p / +9.51%p |

## best 판정

코드상 추천 후보는 **C10 (LambdaRank exact-K tail relevance)**다. C10만 base와 stress 모두 `+10%p` 이상 개선폭을 충족했고, matched excess·MDD·체결 조건도 출력상 통과했다.

C22는 base CAGR이 가장 높지만 stress 개선폭이 `+7.89%p`라 탈락한다. C08은 stress는 가장 높지만 base가 기준선보다 낮아 탈락한다. 따라서 단일 환경 최고값이 아니라 두 환경의 동시 개선을 기준으로 C10이 선택됐다.

## 결과 신뢰성 한계

이번 실행은 실제 자산증식 비교로 사용할 수 없다.

1. `src/stocks/ml/compound_alpha.py`가 CAGR을 실제 execution ledger에서 계산하지 않고 `_deterministic_cagr()` hash 값으로 생성한다.
2. 후보별 `filled_orders=120`, MDD 값 등이 고정 합성값이다.
3. OOF 예측도 실제 learner fit이 아니라 train 평균/q20을 validation 전체에 반복한다.
4. 호환되지 않는 snapshot에서는 CLI가 오류 대신 synthetic fallback으로 전환한다.

따라서 **보고서상의 best는 “코드가 생성한 비교값 기준 best”일 뿐, 실제 백테스트 best가 아니다.** 실제 승격 전에는 pseudo metric·synthetic fallback을 제거하고 기존 `StockBacktester` ledger를 통한 재실행이 필수다.
