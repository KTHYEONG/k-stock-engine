최신 `main`을 다시 기준으로 보면, 이전 분석 때보다 **코드 정합성과 데이터 관리 구조는 분명 좋아졌습니다.** 다만 “복리자산증식 가능한 ML을 발견하는 시스템”이라는 기준으로 보면 아직 핵심 병목이 남아 있습니다.

가장 중요한 결론부터 말하면:

> **지금 문제는 모델 종류가 부족해서가 아니라, `stock signal → 경제적 calibration → 투자 여부 → portfolio → compound growth` 사이에서 신호를 지나치게 잃고 있고, 동시에 screen/feature-selection 통계가 아직 제대로 정립되지 않았다는 것입니다.**

그리고 현재 저장된 결과만으로는 **“예측 가능한 신호 자체가 없다”라고 결론 내릴 수 없습니다.** 오히려 최신 일반 mainline 실행에서는 ElasticNet OOF Rank IC가 fold별 약 `0.030 ~ 0.037`, 평균 약 `0.0333`으로 양수인데도 최종적으로 `no-horizon-evidence → NO_TRADE`가 나왔습니다. 즉 **raw ranking signal과 실제 투자 의사결정 사이의 conversion layer가 현재 가장 의심스러운 구간**입니다.

---

# 1. 현재 상태에 대한 평가

| 영역                           | 현재 평가        | 판단                 |
| ---------------------------- | ------------ | ------------------ |
| PIT/Data integrity           | 좋음           | 최근 개편 유효           |
| Feature/Label 분리             | 매우 좋아짐       | 유지                 |
| Route ↔ training target 정렬   | 개선됨          | 이전 핵심 문제 해결        |
| 모델 family 구조                 | 개선됨          | `FamilySpec` 방향 좋음 |
| Family별 feature 최적화          | 부족           | 전 모델이 사실상 동일 view  |
| Screen sampling              | **문제 있음**    | P0                 |
| Nested feature selection     | **사실상 미완성**  | P0                 |
| Bootstrap/statistical gate   | **문제 있음**    | P0                 |
| Calibration → trade          | **지나치게 보수적** | P0                 |
| Portfolio compound selection | 부분 개선        | champion 정렬은 좋아짐   |
| Risk/coverage promotion gate | 부족           | 보완 필요              |
| Research ↔ production ML 통합  | 부족           | 이중 파이프라인           |
| 모델 자체 tuning                 | 매우 얕음        | 이후 개선              |
| 복리 목표와 전체 architecture       | 방향은 맞지만 미완성  | 핵심 개선 대상           |

즉 지금은 **데이터 파이프라인을 다시 뜯을 시점이 아니라 ML→경제성 변환부를 뜯을 시점**입니다.

---

# 2. 최근 데이터 구조 개편은 대체로 잘 됨

현재 `DirectMarketDataLoader`는:

* base
* features
* labels

를 명시적 dataset ID로 읽고,

```text
MlMarketData.frame
    = base + feature
    = (instrument_id, session) 당 정확히 1 row

labels_by_horizon[H]
    = horizon별 별도 narrow table
```

로 분리합니다.

이건 이전보다 상당히 좋은 구조입니다.

특히 과거처럼 H10/H20 label을 feature frame에 동시에 붙여서:

```text
feature × horizon
```

형태로 row가 불필요하게 증식하거나 universe가 암묵적으로 교집합화되는 문제가 줄었습니다.

또 fundamental도 `disclosure_date`가 없으면 `bp_ratio`, `ep_ratio`를 learner 입력에서 제외하도록 바뀌었습니다.

Feature registry 자체도 `bp_ratio`, `ep_ratio`의 availability를 `disclosure_date`에 묶고 있습니다.

이 방향은 유지하는 게 맞습니다.

---

# 3. 다만 Data Bundle contract를 한 단계 더 강화할 필요가 있음

현재 direct loader는 의도적으로 snapshot/catalog resolution을 우회하고 **정확한 dataset ID + manifest**를 사용합니다. 반대로 catalog 쪽은 append-only + content hash + active dataset policy라는 강한 구조를 갖습니다.

둘 다 개별적으로 합리적이지만 ML 실험 단위에서는 하나 더 묶는 게 좋습니다.

현재 최신 실행 ledger를 보면 H10 label dataset을 사용했는데도 상위 data metadata에:

```text
label_definition = none
label_horizon_sessions = 1
```

이 기록되어 있습니다. 실제 입력은 `...mh10_20` label dataset이고 H10 join을 사용합니다.

이는 계산 자체의 오류라고 단정할 수는 없지만 **experiment provenance 표현은 부정확합니다.**

권장 구조는 다음입니다.

```text
ResearchDataBundle
├── base
│   ├── dataset_id
│   └── content_hash
├── features
│   ├── dataset_id
│   ├── schema_hash
│   └── content_hash
├── labels
│   ├── H5  -> id/hash/definition/reference_notional
│   ├── H10 -> id/hash/definition/reference_notional
│   └── H20 -> ...
├── costs
│   ├── base_hash
│   └── stress_hash
├── universe_policy_hash
├── execution_policy_hash
└── research_window
```

그리고 이 전체를 하나의:

```text
data_bundle_fingerprint
```

로 묶는 것이 좋습니다.

최근 데이터 구조 변경의 다음 단계는 이것이지 또다시 physical directory를 바꾸는 것이 아닙니다.

---

# 4. 이전의 가장 큰 문제였던 target mismatch는 수정됨

이 부분은 중요한 개선입니다.

현재:

```python
route_training_target()
```

은 unhedged이면:

$$
y = gross\_return-reference\_cost
$$

hedged이면:

$$
y = risk\_residual-reference\_cost
$$

을 사용합니다.

따라서 과거처럼:

```text
학습: residual alpha
평가: unhedged absolute return
```

의 명백한 불일치는 사라졌습니다.

이는 올바른 수정입니다.

그런데 **그 결과 새로운 문제가 드러났습니다.**

---

# 5. P0: unhedged absolute target을 학습하면서 learner에는 ALPHA feature만 넣음

현재 feature contract의 원칙은 명시적으로:

> ALPHA만 learner matrix에 들어가며 RISK/LIQUIDITY/CONTROL은 residualization, covariance, sizing 등에만 사용

입니다.

이 구조는 과거 residual alpha를 예측할 때는 논리가 있었습니다.

그런데 지금 모델의 target은 unhedged 기준:

```text
gross return - cost
```

입니다.

그러면 target에는:

```text
시장 방향
sector 효과
beta
volatility regime
liquidity regime
idiosyncratic alpha
```

가 모두 포함됩니다.

그런데 모델에는 risk/regime context 상당수가 안 들어갑니다.

### 예를 들어

$$
R_{i,t,H}
=
M_{t,H}
+
S_{sector,t,H}
+
\alpha_{i,t,H}
+\epsilon
$$

라고 생각하면,

현재 모델은 주로:

```text
alpha-related cross-sectional X
```

만 받고,

target에는:

```text
M + S + alpha
```

전체가 들어갑니다.

이것은 상당한 noise를 학습 target에 추가합니다.

특히 시장 공통수익 \(M_t\)는 한 세션의 모든 종목에 거의 공통으로 붙기 때문에 **Top-K ranking에는 도움이 없는데 regression loss는 크게 흔듭니다.**

---

# 6. 그래서 단순 `gross-cost` regression이 최종 해답은 아님

이전 route mismatch를 고친 방향은 맞지만, 한 단계 더 가야 합니다.

제가 권하는 구조는 **stock selection과 market deployment를 분해하는 것**입니다.

### Stock Alpha Head

종목 선택용:

$$
y^{active}_{i,t,H}
=
R^{net}_{i,t,H}
-
\operatorname{median}_j(R^{net}_{j,t,H})
$$

또는 sector-neutral variant.

즉:

```text
"같은 날 다른 종목보다 얼마나 좋은가?"
```

를 학습합니다.

### Market / Deployment Head

별도로:

```text
다음 H일 시장 기대수익
시장 breadth
volatility regime
trend
cross-sectional dispersion
liquidity regime
```

를 이용해:

```text
이번에 90% 투자할 것인가
50% 투자할 것인가
현금으로 있을 것인가
```

를 판단합니다.

최종 expected return은:

$$
\hat \mu_i
=
\hat \mu_{market}
+
\hat \alpha_i
$$

가 됩니다.

이 방식이 현재의 **unhedged long-only + cash 가능 전략**과 훨씬 잘 맞습니다.

---

# 7. 현재 최신 실행에서 이 구조가 필요한 정황이 이미 보임

최근 일반 mainline 실행은:

* 730 sessions
* H10
* ElasticNet
* fold Rank IC 약 `0.02995`, `0.03667`
* 평균 약 `0.0333`

이었습니다.

금융 cross-section에서 Rank IC 0.03이 곧바로 돈이 된다는 뜻은 아닙니다.

하지만 최소한:

> **예측 score가 완전한 random noise라고 말할 수 없는 상황**

입니다.

그런데 최종:

```text
evidence_horizons = []
NO_TRADE
reason = no-horizon-evidence
```

입니다.

따라서 지금 가장 먼저 분석해야 할 것은:

```text
"ML이 못 맞혔나?"
```

가 아니라

```text
"맞힌 신호가 어느 gate에서 죽었나?"
```

입니다.

---

# 8. 현재 0 fill의 가장 유력한 구조적 원인: calibration gate

현재 `CausalAlphaCalibrator`는 단순 calibration이 아닙니다.

각 score percentile bucket에 대해:

1. 과거 OOS 결과만 사용
2. bucket expected alpha 계산
3. moving-block bootstrap
4. **lower bound > 0인 bucket만 사용**
5. 여기에 round-trip execution cost까지 차감
6. lower bound가 없으면 null
7. null이면 매수 불가

입니다.

즉:

```text
Raw ML score
    ↓
historical bucket
    ↓
expected alpha
    ↓
confidence lower bound > 0 ?
       │
       ├─ No → NULL → 현금
       │
       └─ Yes
            ↓
        cost 차감
            ↓
        portfolio
```

입니다.

최신 `ml-cmp`에서 LGBM/Quantile은 OOF까지 갔는데:

```text
legacy_overlay_5bps       → 0 fills
lower_bound_only          → 0 fills
lower_bound_half_kelly    → 0 fills
```

이었습니다.

특히 `lower_bound_only`도 0이라는 것은 **5bp band가 문제인 게 아닙니다.**

사실상:

```text
net_alpha_lower_bound > 0
```

인 investable observation 자체가 없었다는 뜻에 가깝습니다.

---

# 9. 그런데 model-selection에서 calibration에 너무 강한 α를 전달함

현재 family study는:

```python
candidate_count
=
family
× lookback
× H/C/K cell
× policy profiles
```

를 이용해:

$$
\alpha_{window}
=
\frac{0.05}{candidate\_count}
$$

를 만듭니다.

현재 예처럼 18개이면:

$$
0.05/18=0.002777...
$$

입니다.

그리고 이 `alpha_window`를 `win_request.bootstrap_alpha`에 넣습니다.

그 `win_request`가 다시 OOF calibration에 전달됩니다.

즉 **각 calibration bucket 하나하나가 99.72% 수준의 lower-bound positivity를 요구할 수 있습니다.**

그 다음 또 최종 portfolio replay에서도 같은 adjusted alpha로 growth lower bound를 검사합니다.

이건 과도하게 보수적입니다.

---

# 10. Multiplicity correction을 적용할 층이 잘못됨

제가 가장 강하게 수정 권고하는 부분 중 하나입니다.

Multiplicity correction의 목적은:

> 여러 candidate 중 좋은 놈을 골랐기 때문에 생기는 selection bias 통제

입니다.

그렇다면 correction은 주로:

```text
최종 family/H/K/C/profile candidate selection
```

단에서 적용해야 합니다.

그런데 현재는 calibration bucket에서도 강한 correction을 적용하고 다시 final replay에서도 적용합니다.

결과적으로:

```text
ML signal
→ bucket confidence hard test
→ cost
→ allocation
→ compound growth hard test
```

에서 **동일한 불확실성을 여러 번 공격적으로 잘라냅니다.**

### 권장

Calibration은 estimation으로 취급:

```text
alpha = 0.05 또는 0.10
```

정도의 합리적인 uncertainty estimate.

그리고 sizing에:

```text
μ
LB
SE
q20
```

를 연속적으로 전달합니다.

최종 모델 선택에서만:

```text
Holm / Romano-Wolf / family-wise corrected
stress compound growth evidence
```

를 사용하십시오.

---

# 11. 특히 `LB <= 0 → 무조건 NO_TRADE`는 정보 손실이 큼

예를 들어 모델이:

```text
expected return = +3.0%
SE = 1.8%
95% LB = -0.2%
```

라면 현재 논리는:

```text
LB < 0
→ trade 금지
```

입니다.

하지만 경제적 최적화에서는 오히려:

```text
confidence가 낮으니 position을 작게
```

가 자연스럽습니다.

예:

$$
w_i
\propto
\frac{\hat\mu_i}
{\hat\sigma_i^2+\lambda\,uncertainty_i^2}
$$

형태가 더 적합합니다.

즉 confidence는:

```text
binary gate
```

보다는

```text
continuous sizing penalty
```

로 쓰는 것이 복리 증식 목적과 더 잘 맞습니다.

물론:

```text
expected net alpha <= 0
```

라면 trade하지 않는 것이 맞습니다.

하지만:

```text
99.72% LB > 0
```

까지 요구하는 것은 전혀 다른 수준입니다.

---

# 12. P0: 최신 screen sampling 구조에는 실제 통계적 문제가 있음

현재 기본 screen budget은 대략:

```text
train rows      3000
validation rows 1000
```

입니다.

그리고 새로운 sampler는:

```python
names_per_session
=
K * screen_cross_section_multiplier
```

를 사용합니다.

현재 multiplier 기본이 4라면 K=12에서:

```text
48 names/session
```

이 필요합니다.

validation max 1000 rows이므로 사용할 수 있는 session 수는 최대:

$$
\lfloor 1000/48 \rfloor =20
$$

정도입니다.

그 다음 경제성 평가에서 다시:

```text
C = 10
```

cadence를 적용합니다.

이론적으로는 **fold당 실제 economic screen decision이 고작 약 2개 수준까지 줄어들 수 있습니다.**

이 구조는 수정해야 합니다.

---

# 13. Sampling 순서가 반대임

현재 흐름은 사실상:

```text
validation sessions
    ↓
20개 session sampling
    ↓
C=10 cadence
    ↓
약 2개 decision session
```

입니다.

원하는 것은:

```text
전체 validation calendar
    ↓
C=10 decision sessions 확정
    ↓
각 decision session에서 충분한 종목 sampling
```

이어야 합니다.

예를 들어 validation 126 sessions이면:

```text
C10
→ 약 12~13 decision dates
```

를 먼저 확정하고,

각 date에서:

```text
K12 × headroom 4
= 48 names
```

를 뽑아야 합니다.

즉 budget을:

```text
rows
```

가 아니라

```text
decision_sessions × names_per_session
```

으로 정의해야 합니다.

이건 현재 코드 기준 P0 버그에 가깝습니다.

---

# 14. P0: 새로 추가된 inner feature selection은 현재 direct 경로에서는 사실상 작동하지 않음

코드에 `_select_inner_feature_groups()`가 추가됐습니다.

그런데 이를 호출하는 조건이:

```python
if request is not None
and TARGET_COLUMN in cache.train_features.columns:
```

입니다.

반면 새 direct data contract는 명확히:

```text
MlMarketData.frame
= base/feature only
labels는 labels_by_horizon로 별도
```

입니다.

따라서 canonical direct path의:

```text
cache.train_features
```

에는 target이 없어야 정상입니다.

즉 이 new inner selection branch는 **정상 direct data 구조에서는 진입하지 않는 것이 자연스럽습니다.**

---

# 15. 게다가 inner selector 자체도 아직 진짜 feature selection이 아님

해당 함수는 feature group을 평가할 때 label predictive performance가 아니라:

```python
score = np.nanstd(feature_values)
```

즉 **feature 자체의 dispersion**으로 ranking합니다.

그리고 상위 절반을 고릅니다.

이는:

```text
변동성이 큰 feature
≈ 좋은 feature
```

라는 가정인데 통계적으로 근거가 없습니다.

예를 들어 완전한 random noise:

```text
X_random ~ N(0, 100)
```

은 dispersion이 매우 크지만 prediction signal은 0입니다.

따라서 이 구현은 삭제하거나 실제 nested selection으로 바꿔야 합니다.

---

# 16. 실제 feature prefix 선택은 아직 outer validation을 사용함

현재 economic screen에서는 결국 각 feature prefix를 fit한 뒤:

```text
outer validation
```

에서 `tail_excess_lower_bound`를 계산하고 가장 좋은 prefix를 선택합니다.

즉:

```text
Outer Validation
   ├── feature set 선택
   └── 그 feature set 성능 측정
```

을 동시에 하고 있습니다.

selection bias가 남아 있습니다.

올바른 구조는:

```text
Outer Train
│
├── Inner Fold 1
├── Inner Fold 2
└── Inner Fold 3
      ↓
 feature / hp 선택

Outer Validation
      ↓
 딱 한 번 평가
```

입니다.

현재 `_inner_folds_from_train()` 자체는 존재하지만 main economic prefix selection과 연결돼 있지 않습니다.

---

# 17. P0: moving-block bootstrap 수정도 현재 실제 결과에 반영되지 않음

코드를 보면 새로운:

```python
segmented_moving_block_lower_bound()
```

가 생겼습니다.

방향은 맞습니다.

그런데 현재 prefix evaluation에서:

```python
fold_values = np.full(
    len(fold_sessions),
    see.tail_excess_lower_bound
)
```

을 만듭니다.

즉 실제:

```text
session 1 return
session 2 return
session 3 return
...
```

을 block bootstrap하는 것이 아니라,

이미 계산된 하나의 scalar LB를:

```text
[-0.001, -0.001, -0.001, ...]
```

처럼 복제한 뒤 bootstrap합니다.

당연히 결과는 원 scalar와 거의 같습니다.

그리고 더 중요한 것은 계산한 `segmented_lb`가 실제 `prefix_evidences`에 반영되지 않고 원래 `see`가 저장된다는 점입니다.

즉 **현재 screen의 실제 통계는 여전히 기존 IID session bootstrap에 의존합니다.**

---

# 18. 게다가 fold lower bound를 여전히 평균냄

현재 aggregate도:

```python
agg_lb = mean(fold_lower_bounds)
```

입니다.

하지만:

$$
\frac{LB_1+LB_2+LB_3}{3}
$$

는 전체 OOF sample의 confidence lower bound가 아닙니다.

권장:

```text
Fold 0 per-session utility
Fold 1 per-session utility
Fold 2 per-session utility
          ↓
OOF segment id와 함께 concatenate
          ↓
segment-preserving moving block bootstrap
          ↓
study-level LB 1개
```

입니다.

---

# 19. `minimum_tail_draws=20`도 현재 실질적으로 보장되지 않음

새 settings에는:

```text
minimum_tail_draws = 20
```

이 있습니다.

좋은 아이디어입니다.

그런데 실제 main result에서는 여전히:

```text
adjusted alpha = 0.002777...
bootstrap = 360
```

입니다.

따라서 극단 tail에 기대되는 sample 수는:

$$
360\times0.002777 \approx 1
$$

입니다.

20개의 tail draw를 보장하려면 최소:

$$
20 / 0.002777 \approx 7200
$$

회가 필요합니다.

새 moving-block helper 내부에는 이를 늘리는 로직이 있지만 위에서 설명했듯 현재 economic screen evidence에 제대로 연결되어 있지 않습니다.

---

# 20. 좋은 변화: screen LB가 음수라도 상위 모델은 OOF로 보내도록 수정됨

이전에는:

```text
screen LB <= 0
→ 무조건 탈락
```

이었습니다.

현재 economic path는 finite candidate를 정렬한 뒤 **상위 `max_full_replay_families`를 OOF로 보냅니다.**

기본값은 2입니다.

이건 좋은 변경입니다.

그래서 최신 documented run에서도:

* HistGradient Quantile
* LGBM

이 screen LB가 미세하게 음수였음에도 full OOF/replay까지 갔습니다.

이 방향은 유지해야 합니다.

---

# 21. 좋은 변화: 최종 champion 선택도 수정됨

이전에는 survivor 첫 번째를 사실상 선택하는 구조가 있었습니다.

현재는 admissible family-profile에 대해:

1. stress lower bound 높은 순
2. base lower bound 높은 순
3. worst MDD 낮은 순
4. turnover 낮은 순
5. complexity 낮은 순

으로 champion을 선택합니다.

이는 **복리자산증식 목표에 훨씬 가까운 선택 기준**입니다.

이 부분은 유지해야 합니다.

---

# 22. 하지만 promotion gate에 MDD/coverage hard constraint가 부족함

현재 replay admission 코드를 보면 사실상:

```text
filled_orders > 0
base LB > 0
stress LB > 0
```

이면 `ReplayCandidateEvidence`를 만들고 admitted pool에 넣습니다.

`base_mdd`, `stress_mdd`는 계산하지만 주로 champion tie-break에 사용됩니다.

즉 현재 코드상 해당 구간에서는 명시적인:

```text
stress MDD <= request.compounding.max_drawdown
invested_interval_fraction >= threshold
filled_cycle_count >= threshold
coverage >= threshold
```

hard gate가 보이지 않습니다.

복리 목표라면 반드시 추가하는 것이 좋습니다.

특히:

```text
1~2번 투자해서 우연히 양수
```

인 모델이:

```text
꾸준히 자본을 배치해 복리 성장
```

하는 모델과 같은 admission 조건을 가지면 안 됩니다.

---

# 23. 가장 중요한 구조적 변경: 6개 모델을 “서로 경쟁하는 하나의 역할”로 보지 말 것

현재 family는:

* ElasticNet
* Huber
* ExtraTrees
* Quantile HGB
* LGBM
* LambdaRank

입니다.

현재 `FamilySpec`으로 fitting semantics를 모은 것은 좋은 방향입니다.

하지만 저는 이제 이 여섯 개를 모두:

```text
누가 하나의 최종 winner인가?
```

로 보는 구조 자체를 조금 바꾸는 것을 권합니다.

왜냐하면 모델 역할이 다르기 때문입니다.

---

# 24. 권장 ML 구조: 3개의 예측 Head + 1개의 Portfolio layer

## A. Alpha / Ranking Head

목적:

```text
어떤 종목을 살 것인가?
```

적합 모델:

* Elastic
* Huber
* LGBM mean
* LambdaRank

target:

```text
cross-sectional active net return
```

---

## B. Downside / Confidence Head

목적:

```text
이 종목의 downside가 얼마나 큰가?
```

적합:

```text
HistGradient Quantile q10/q20
```

Quantile model을 mean predictor와 같은 기준으로 경쟁시키지 말고:

$$
(\hat \mu,\hat q_{20})
$$

두 출력을 같이 사용하는 것이 좋습니다.

---

## C. Market / Exposure Head

목적:

```text
지금 전체 자본을 얼마나 투자할 것인가?
```

입력:

* market trend
* index momentum
* breadth
* volatility regime
* cross-sectional dispersion
* liquidity
* sector breadth

출력:

```text
expected market net return
또는
expected portfolio log-growth
```

---

## D. Portfolio optimizer

최종적으로:

$$
\hat\mu_i,\quad \hat q_{20,i},\quad
\Sigma,\quad Cost_i
$$

를 받아 position을 정합니다.

---

# 25. 이것이 복리자산증식 목표와 훨씬 직접적으로 연결됨

최종 목표는 사실:

$$
\max_w
E\left[
\log\left(
1+w^\top R-C(\Delta w)
\right)
\right]
$$

입니다.

개별 종목 ML에게 이 함수를 직접 학습시키기는 어렵습니다.

그래서:

```text
Prediction
 ↓
Distribution / uncertainty
 ↓
Portfolio weights
 ↓
Execution
 ↓
log wealth
```

로 나누는 것이 맞습니다.

즉 **CAGR을 직접 label로 만들자는 것은 아닙니다.**

현재 프로젝트의 execution-equivalent replay 철학은 그대로 유지해야 합니다.

---

# 26. 그런데 현재 FamilySpec은 아직 “알고리즘만 다양”함

현재 모든 family의:

```python
feature_view="winsor_rank_robust_v1"
```

가 동일합니다.

즉 모델 다양성은 대부분:

```text
estimator diversity
```

뿐입니다.

실제로는:

```text
data representation diversity
target diversity
error diversity
```

가 필요합니다.

---

# 27. Family별 feature view를 실제로 분리해야 함

제가 추천하는 형태는 다음과 같습니다.

| Family     | 권장 view                                             |
| ---------- | --------------------------------------------------- |
| Elastic    | rank + sector_rank + missing + sparse interaction   |
| Huber      | compact rank/robust                                 |
| LGBM       | raw/winsor + rank + sector_rank + missing + CONTEXT |
| ExtraTrees | winsor/raw + rank + CONTEXT                         |
| Quantile   | winsor/raw + downside/risk/context                  |
| LambdaRank | rank + sector_rank + graded/exact-K relevance       |

현재 `rawnet_lgbm_v2`도 실제로는 `winsor_rank_robust_v1`입니다.

이름과 inductive bias가 일치하지 않습니다.

---

# 28. 지금 feature role에는 `CONTEXT`가 필요함

현재는:

```text
ALPHA
RISK
LIQUIDITY
CONTROL
```

입니다.

여기에:

```text
CONTEXT
```

를 추가하는 것을 권합니다.

예:

```text
vol_regime
market trend
market breadth
sector regime
cross-sectional dispersion
market liquidity state
```

이런 변수는:

```text
이 자체가 종목 alpha인가?
```

보다는

```text
현재 alpha signal이 잘 작동할 환경인가?
```

에 가깝습니다.

따라서:

```text
ALPHA × CONTEXT
```

interaction이나 tree split에는 허용하되,

단순 standalone alpha attribution과는 구분하는 것이 좋습니다.

---

# 29. 현재 interaction 구현은 이전보다 정리됨

이전의 불가능했던:

```text
flow_intensity × vol_regime
```

은 제거됐고 현재 research schema에서는:

```text
flow_consensus × relative_trend_score
```

라는 ALPHA×ALPHA interaction만 남아 있습니다.

그리고 `FamilySpec`에서는 linear family만 interaction을 허용합니다.

이 부분은 이전보다 정합적입니다.

다만 저는 interaction을 무작정 늘리기보다 `CONTEXT`를 도입한 다음:

```text
flow × regime
momentum × vol regime
value × liquidity regime
```

정도의 경제적 interaction만 사전등록하는 편을 권합니다.

---

# 30. 모델 parameter 자체도 지금은 너무 약함

현재 `FamilySpec`을 보면 대략:

### ElasticNet

```text
screen max_iter = 20
full max_iter   = 50
```

### Huber

```text
20 / 50
```

### ExtraTrees

```text
30 / 50 trees
```

### LGBM

```text
20 / 50 boosting rounds
```

수준입니다.

이건 공정한 “알고리즘 비교”라고 보기 어렵습니다.

실제로 최신 documented run에서도 Huber convergence warning이 반복됐습니다.

---

# 31. 특히 Huber/Elastic의 iteration은 compute budget으로 줄이면 안 됨

선형 최적화는:

```text
20 iterations에서 끊어진 model
```

과

```text
수렴한 model
```

이 다른 알고리즘처럼 행동합니다.

Screen 비용을 줄이려면:

```text
row 수
feature 수
candidate prefix 수
```

를 줄여야지,

optimizer를 미수렴 상태로 멈추는 것은 좋지 않습니다.

권장:

```text
Elastic max_iter >= 1000
Huber max_iter >= 500
```

정도로 충분한 상한을 주고 실제 convergence를 확인하십시오.

---

# 32. Tree 계열도 지금은 baseline 수준

ExtraTrees 30~50 trees로:

```text
feature importance
+
model ranking
```

까지 판단하는 것은 불안정합니다.

권장 최소:

```text
200~500 trees
```

그리고 더 중요한 hyperparameter는:

```text
min_samples_leaf
max_features
max_depth
```

입니다.

금융 noise 문제에서는 leaf regularization이 매우 중요합니다.

---

# 33. LGBM도 현재 inductive bias를 거의 활용하지 못함

현재 실제 params는 사실상:

```text
objective=regression
metric=l2
seed
deterministic
threads
```

정도입니다.

필요한 것은 거대한 Optuna가 아니라 작은 사전등록 grid입니다.

예:

```text
num_leaves       15 / 31
min_data_in_leaf 100 / 300 / 600
feature_fraction .6 / .8 / 1.0
lambda_l2        0 / 1 / 10
```

정도면 충분합니다.

그리고 **inner OOF에서 선택**해야 합니다.

---

# 34. Quantile model은 winner 후보가 아니라 companion model에 가까움

현재 q=0.2 HGB는 다른 mean predictor와 동일한 방식으로 screen ranking을 비교합니다.

저라면:

```text
Mean model:
μ_i

Quantile model:
q20_i
```

를 같이 사용합니다.

예를 들어:

$$
score_i
=
\hat\mu_i
-
\lambda(\hat\mu_i-\hat q_{20,i})
$$

또는:

```text
μ로 종목 선택
q20으로 sizing 감소
```

를 하는 편이 훨씬 자연스럽습니다.

---

# 35. LambdaRank는 K 정합성은 현재 제대로 고쳐짐

이 부분도 개선됐습니다.

현재 LambdaRank는:

```text
training_top_k 필요
exact K relevance
lambdarank_truncation_level = K
ndcg_eval_at = K
```

가 연결됩니다.

과거의 K12 hard-code 문제는 현재 코드상 해결됐습니다.

다만 아직 binary relevance:

```text
Top K = 1
나머지 = 0
```

이라 signal을 많이 버립니다.

---

# 36. LambdaRank에는 graded relevance challenger를 추가할 가치가 큼

예:

```text
Top 1%   = 4
Top 5%   = 3
Top 10%  = 2
Top 25%  = 1
나머지   = 0
```

이렇게 하면:

* ordering 정보 증가
* K 민감도 감소
* K12/K16/K20 여러 execution K에 재활용 가능

합니다.

현재 exact-K 모델도 baseline으로 유지하고 둘을 비교하면 됩니다.

---

# 37. 최신 documented LambdaRank의 `-1e12`는 현재 코드 성능으로 해석하면 안 됨

`ml-cmp`에서 LambdaRank는 hard reject였습니다.

하지만 이후 관련 model-selection/sampling 코드가 바뀌었습니다.

그리고 문서에는 **hard rejection의 구체적 reason code가 남아 있지 않습니다.**

따라서:

```text
LambdaRank는 성능이 나쁘다
```

라고 결론 내릴 수 없습니다.

현재 result ledger에는 반드시:

```text
hard_reject_stage
hard_reject_reason
affected_fold
cross_section_min
query_count
K
```

정도를 bounded diagnostic으로 남기는 것이 좋습니다.

`-1e12` 하나는 debugging 정보로 너무 부족합니다.

---

# 38. H / C / K 구조는 지금도 한 단계 더 정리해야 함

현재 fast research study는 실제로 하나의:

```text
H
C
K
lookback
```

만 허용합니다.

run 하나를 빠르게 제한하는 것은 괜찮습니다.

문제는 여러 run을 사람이 따로 실행하면:

```text
H5
H10
H20

K12
K16
K20

lookback 504
756
1260
```

전체 탐색의 multiplicity가 개별 run 밖으로 빠져나갑니다.

---

# 39. 이제 Data Catalog에 이어 Experiment Catalog가 필요함

새 데이터 관리 구조를 활용해서 다음을 추가하는 것이 좋습니다.

```text
StudyManifest
├── study_id
├── data_bundle_fingerprint
├── families
├── horizons
├── training_lookbacks
├── train_K
├── execution_C
├── execution_K
├── profiles
├── bootstrap policy
├── feature view versions
└── total hypothesis family size
```

즉 먼저:

```text
"이번 research family가 무엇을 시도할 것인가"
```

를 등록합니다.

그 다음 개별 10분짜리 run을 여러 개 수행해도 **같은 study family**에 evidence를 누적합니다.

그러면 global multiplicity correction을 제대로 할 수 있습니다.

이것이 최근 data/catalog 구조 개편과 아주 잘 맞습니다.

---

# 40. H/C/K는 여전히 다음처럼 분리해야 함

### Regression / Quantile

```text
Model = H-dependent
```

이므로:

```text
H5 model
H10 model
H20 model
```

은 재학습.

하지만 C/K는:

```text
같은 OOF score를 replay
```

할 수 있습니다.

```text
H10 model
 ├─ C5/K12
 ├─ C5/K20
 ├─ C10/K12
 └─ C10/K20
```

---

### LambdaRank exact-K

K가 label에 들어가므로:

```text
H10/K12 ranker
H10/K20 ranker
```

는 별도 모델입니다.

현재 `FamilySpec.k_dependency`가 이를 표현하고 있는 것은 좋습니다.

---

# 41. 무엇보다 현재 screen을 제대로 고친 뒤 H/K 실험을 해야 함

현재 screen sampling/statistical 문제가 있는 상태에서:

```text
H5가 낫다
K20이 낫다
LGBM이 낫다
```

를 실험하면 또 잘못된 결론이 나올 수 있습니다.

순서를 바꾸면 안 됩니다.

---

# 42. 복리증식을 위한 추천 최종 Architecture

제가 현재 코드에서 목표로 할 구조는 다음입니다.

```text
[Immutable Data Bundle]
        │
        ▼
[PIT Feature Views]
        │
        ├──────────────┐
        ▼              ▼
[Stock Alpha]     [Market Regime]
 active return      market return
 rank / mean         deployment
        │              │
        ├──────┬───────┘
        │      │
        ▼      ▼
    μ_active  μ_market
        │      │
        └──┬───┘
           ▼
       μ_absolute
           │
     ┌─────┴─────┐
     ▼           ▼
[Downside q20] [Cost Model]
     │           │
     └─────┬─────┘
           ▼
   [Portfolio Optimizer]
   μ / q20 / covariance
   cost / turnover
   confidence
           │
           ▼
    [Execution Replay]
    base + stress
           │
           ▼
  per-period log growth
           │
           ▼
 [Study-level Selection]
 stress lower log-growth
 MDD / coverage constraints
           │
           ▼
        Champion
           │
           ▼
    Locked Holdout
```

이것이 제가 보는 현재 프로젝트의 가장 적합한 종착점입니다.

---

# 43. 현재 6-family 구조도 버릴 필요 없음

오히려 이렇게 재배치하십시오.

| 현재 Family     | 새 역할                                |
| ------------- | ----------------------------------- |
| ElasticNet    | stable active-alpha baseline        |
| Huber         | robust active-alpha baseline        |
| LGBM          | nonlinear active-alpha / absolute-μ |
| ExtraTrees    | nonlinear diversity challenger      |
| LambdaRank    | direct stock-selection ranker       |
| Hist Quantile | downside/uncertainty head           |

즉 Quantile은 winner competition에서 일부 분리합니다.

그리고 나중에:

```text
Elastic alpha + Quantile downside
LGBM alpha + Quantile downside
LambdaRank + LGBM magnitude
```

같은 구조를 exact replay로 비교할 수 있습니다.

---

# 44. “복리”를 위해 별도의 exposure model이 특히 중요함

주식 선택이 맞더라도 시장이 급락하면 long-only portfolio는 손실을 봅니다.

현재 구조는 이 문제를 주로:

```text
lower-bound positivity
→ 투자 or cash
```

로 해결하려고 합니다.

그래서 너무 많은 0-trade가 나옵니다.

더 자연스러운 구조는:

```text
Stock model:
상대적으로 좋은 종목

Exposure model:
지금 어느 정도 총자본을 넣을지
```

를 분리하는 것입니다.

예:

```text
regime strong:
gross 0.90

neutral:
gross 0.50

weak:
gross 0.20

very weak:
0.00
```

연속적으로 sizing하게 합니다.

이것이 compound wealth에서는 binary NO_TRADE보다 훨씬 유연합니다.

---

# 45. 현재 portfolio objective도 점진적으로 expected-log-growth 쪽으로 통일하는 게 좋음

현재 최종 replay의 실제 equity/log-growth를 truth source로 사용하는 것은 맞습니다.

그리고 champion이 stress LB 중심으로 선택되는 것도 맞습니다.

따라서 upstream에서도 가능하면:

```text
MSE
Rank IC
Top-K mean
```

은 진단용으로 두고,

최종 선택은 일관되게:

$$
LB_\alpha\left[
\frac{1}{T}
\sum_t
\log\left(\frac{W_{t+1}}{W_t}\right)
\right]
$$

으로 모으는 것이 좋습니다.

---

# 46. Raw Rank IC는 반드시 계속 기록해야 함

Rank IC는 최종 promotion metric으로 쓰면 안 되지만 **진단에는 필수**입니다.

현재 최신 direct run에서 바로 도움이 됐습니다.

```text
Rank IC ≈ +0.033
yet
NO_TRADE
```

를 통해 “raw learner가 완전히 죽었다”와 “경제성 conversion이 죽었다”를 구분할 수 있기 때문입니다.

각 family에 다음 waterfall을 결과에 남기십시오.

```text
Raw OOF
  Rank IC
  top-decile spread
  Top-K hit/excess
        ↓
Calibration
  positive mean buckets
  positive 95% LB buckets
  positive net-LB buckets
        ↓
Portfolio eligibility
  eligible names/session
  nonzero target sessions
        ↓
Orders
  generated orders
        ↓
Fills
  fills
        ↓
Economics
  gross growth
  cost drag
  net growth
  stress growth
```

현재 `0 fills`만 보면 어디서 신호가 죽었는지 알 수 없습니다.

---

# 47. 특히 다음 실험에서는 calibration ablation이 매우 중요함

같은 OOF score로 learner는 절대 다시 fit하지 말고 다음 3개만 replay하십시오.

```text
A. raw rank → Top-K equal/risk weight
B. calibrated expected mean → sizing
C. calibrated hard lower-bound gate → 현재 방식
```

그리고 모두 **동일한 execution engine + 동일한 cost**로 비교합니다.

결과가:

```text
A profitable
B profitable
C 0 fills
```

이면 범인은 ML이 아니라 confidence gate입니다.

반대로:

```text
A도 음수
```

이면 실제 alpha가 부족한 것입니다.

이 ablation을 하지 않고 feature/HPO를 계속 만지는 것은 비효율적입니다.

---

# 48. 지금 바로 추가 feature를 대량 생성하는 것은 추천하지 않음

현재는 feature count 부족보다 architecture/statistical 문제의 증거가 강합니다.

우선:

```text
market/context 약 5~10개
```

만 추가하면 충분합니다.

예를 들면:

* KOSPI/KOSDAQ index 5/20/60/120d return
* market breadth
* cross-sectional median return
* cross-sectional dispersion
* market realized vol
* advance/decline
* sector breadth
* aggregate foreign flow
* market turnover/liquidity regime

입니다.

그리고 반드시 PIT availability contract를 붙여야 합니다.

---

# 49. 실행 순서 — 현재 코드 기준으로 이것부터 수정하는 게 좋음

우선순위를 단순화하면 다음입니다.

1. **Screen sampler 수정**
   `calendar → C decision dates → names/session sampling` 순서로 변경. 최소 30~50개의 실제 decision observation 확보.

2. **진짜 nested feature selection 구현**
   새 dispersion selector 삭제. Outer train 내부 purged folds에서 family-specific economic metric으로 feature/hyperparameter 선택.

3. **Screen bootstrap 수정**
   per-session utility를 실제 moving-block bootstrap하고 fold LB 평균을 폐기. `minimum_tail_draws`가 실제 bootstrap에 적용되도록 연결.

4. **Calibration multiplicity 분리**
   calibration 자체는 일반 uncertainty estimation으로 하고 global correction은 최종 model/policy selection에 적용.

5. **Calibration waterfall telemetry 추가**
   raw signal이 어느 단계에서 0-trade가 되는지 측정.

6. **Active alpha + market exposure 분리**
   stock-selection target을 cross-sectional active net return으로 바꾸고 market/regime head 추가.

7. **Family-specific feature views 구현**
   `feature_view` 필드가 실제 matrix를 다르게 만들도록 변경.

8. **모델을 제대로 수렴시킨 뒤 비교**
   Elastic/Huber iteration 정상화, ExtraTrees/LGBM 최소 regularization grid.

9. **Quantile을 uncertainty head로 재배치**, LambdaRank graded relevance 추가.

10. **Promotion gate 보완**
    positive base/stress LB뿐 아니라 MDD, invested coverage, filled cycles를 hard constraint화.

11. **StudyManifest 추가**
    H/lookback/K/C 여러 개의 run을 하나의 hypothesis family로 관리.

12. **Research/Production ML path 통합**
    최종적으로 `FamilySpec` 하나가 screen → OOF → calibration → final fit → artifact serialize까지 동일하게 사용되도록 변경.

---

# 50. 가장 중요한 architecture 판단

현재 프로젝트를 다시 뜯어:

> “CAGR을 직접 예측하는 neural network 하나를 만들자”

로 갈 필요는 없습니다.

그건 오히려 나빠질 가능성이 큽니다.

현재의:

```text
ML signal
→ portfolio
→ exact execution
→ compound wealth
```

철학은 맞습니다.

바꿔야 하는 것은:

```text
현재
ML → 매우 엄격한 confidence gate → 대부분 NULL → cash
```

를

```text
추천
Stock alpha distribution
+
Market deployment signal
+
Downside uncertainty
+
Cost
        ↓
Continuous portfolio sizing
        ↓
Exact execution
        ↓
Stress compound-growth selection
```

으로 만드는 것입니다.

---

## 현재 코드에 대한 최종 판정

**데이터 관리 개편:** 긍정적입니다. 더 이상 핵심 병목이 아닙니다.

**Route-aligned target / FamilySpec / K semantics / champion selection:** 이전보다 확실히 좋아졌습니다.

**그러나 현재 ML 성능을 신뢰성 있게 비교할 수 있는가:** 아직 아닙니다. 특히 screen sampling, nested selection, bootstrap 연결에 구현 문제가 남아 있습니다.

**0 fill이 곧 alpha 부재를 의미하는가:** 아닙니다. 현재 calibration은 positive lower bound를 hard requirement로 사용하고, model-selection에서는 multiplicity-adjusted α까지 calibration에 전달합니다. 최신 일반 mainline에서 raw Rank IC가 양수였는데도 NO_TRADE였다는 점도 이 계층을 별도로 진단해야 함을 보여줍니다.

**복리자산증식 가능한 ML을 도출하려는 현재 큰 방향:** 유지해야 합니다. 다만 “6개의 모델 중 하나를 골라 끝”이 아니라, **stock-selection alpha + uncertainty + market exposure를 분리하고 이 조합을 exact replay의 stress lower log-growth로 선택하는 구조**가 현재 코드베이스에는 더 적합합니다.

현 시점에서는 **새 모델 추가보다 위 1~6번을 먼저 고치는 것이 기대효과가 훨씬 큽니다.** 그 뒤에 H5/H10/H20, K12/16/20/24, lookback 504/756/1260, family별 작은 HPO를 돌려야 결과를 해석할 수 있습니다.
