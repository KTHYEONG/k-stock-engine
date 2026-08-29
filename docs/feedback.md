## 결론

현재 프로젝트의 큰 방향은 **잘못되지 않았습니다.**

즉,

> 서로 다른 inductive bias를 가진 모델들을 별도로 학습 → OOF 경제성 검증 → 실제 실행 리플레이 → 최종적으로 복리자산증식에 가장 유리한 모델/정책 선택

이라는 구조는 합리적입니다. 오히려 단일 모델에 처음부터 CAGR을 직접 최적화시키는 것보다 안전합니다. 복리수익은 종목별 독립 label이 아니라 **종목 간 상관관계, 포지션 크기, turnover, 슬리피지, 현금 비중, MDD, 경로 의존성**까지 포함하는 포트폴리오 수준 함수이기 때문입니다. 현재 프로젝트가 실제 next-open fill, partial fill, capacity, T+2, base/stress cost를 동일 backtester에서 replay하도록 설계한 것은 이 관점에서 맞는 방향입니다.

문제는 그 사이의 연결입니다.

**현재는 “다양한 모델을 준비했다”는 것은 맞지만, 서로 다른 모델을 너무 동일한 데이터 표현·target·screen 조건으로 평가하고 있고, 최종 compound objective에 도달하기 전에 상당히 강한 surrogate gate가 개입합니다.**

특히 제가 보기에는 성능 개선 우선순위가 다음과 같습니다.

| 우선순위 | 문제                                                                  |   중요도 |
| ---- | ------------------------------------------------------------------- | ----: |
| P0   | **학습 target과 unhedged 평가 objective 불일치**                            | 매우 높음 |
| P0   | **screen에서 탈락하여 실제 compound replay까지 아무 모델도 못 감**                   | 매우 높음 |
| P0   | **LambdaRank의 K semantics가 완전히 일관되지 않음**                            | 매우 높음 |
| P0   | **feature/prefix 선택과 outer validation이 완전히 분리되지 않음**                | 매우 높음 |
| P0   | **최종 survivor 선택이 compound objective의 argmax가 아님**                  | 매우 높음 |
| P1   | 모든 family가 사실상 동일한 `winsor/rank/sector-rank/robust` feature view 사용 |    높음 |
| P1   | ALPHA/RISK 역할 설계 때문에 의도한 interaction 일부가 실제 생성되지 않음                 |    높음 |
| P1   | bootstrap 360회로 α=0.00278 tail 추정                                   |    높음 |
| P1   | 모델-selection 경로와 canonical ML 경로의 구현 중복/semantic drift              |    높음 |
| P1   | fixed reference-cost와 실제 compounding capital 사이의 차이                 |  중~높음 |
| P2   | family별 lookback/hyperparameter 부족                                  |    중간 |
| P2   | feature set의 horizon specialization 부족                              |    중간 |

그리고 중요한 점 하나가 있습니다.

**`ml-cmp.md`의 결과와 현재 `main` 코드를 동일한 것으로 보면 안 됩니다.** 최신 문서 직전 commit에서 median imputation, interaction, exact-K LambdaRank 등의 수정이 실제 코드에 추가되었습니다. 즉 현재 코드는 `ml-cmp`가 제안한 “다음 실험”의 일부를 이미 포함하고 있고, 그 코드로 재실행한 성능은 아직 이 문서로 확인되지 않습니다.

---

# 1. `ml-cmp.md` 결과를 먼저 정확히 해석해야 함

현재 H10 / C10 / K12 / lookback 1260 조건에서:

| Model                 | Tail-excess LB |
| --------------------- | -------------: |
| ElasticNet            |  **-0.004583** |
| Huber                 |      -0.005801 |
| LightGBM              |      -0.006986 |
| ExtraTrees            |      -0.009791 |
| LambdaRank            |      -0.010453 |
| HistGradient Quantile |      -0.011162 |

그리고 전부 screen 단계에서 탈락해:

* full OOF = **0**
* replay = **0**
* 최종 상태 = `no-qualified-survivor`

입니다.

따라서 이 결과가 실제로 말하는 것은:

> **"어떤 ML도 compound wealth를 만들지 못했다"가 아닙니다.**

정확히는:

> **"현재 H10/C10/K12 screen surrogate에서 어느 family도 corrected tail-excess lower bound > 0을 만들지 못했기 때문에 실제 포트폴리오 replay까지 가지 못했다."**

입니다.

둘은 상당히 다른 결론입니다.

현재 프로젝트의 최종 목표인 compound wealth 성능은 이 run에서는 **한 번도 측정되지 않았습니다.**

---

# 2. Oracle +0.134를 너무 강하게 해석하면 안 됨

`ml-cmp.md`에는 모든 fold에서 oracle tail LB가 양수이고 aggregate 약 +0.134라는 이유로:

> opportunity set은 존재하고 모델이 ranking을 복원하지 못했다

고 해석합니다.

여기에는 중요한 논리적 문제가 있습니다.

Oracle은 매 세션 **이미 실현된 미래 수익률을 보고** 2,000여 종목 중 가장 좋은 12개를 고릅니다.

이런 oracle은 실제 return이 순수 noise여도 상당히 높은 값을 만들 수 있습니다.

예를 들어:

```text
N개 종목의 미래 수익률
           ↓
실현값을 전부 관찰
           ↓
그중 최고 12개 선택
```

이면 cross-sectional dispersion만 있어도 Top-12 평균은 universe 평균보다 크게 높아집니다.

따라서:

**Oracle > 0**
→ realised cross-sectional dispersion이 존재함.

하지만

**Oracle > 0**
↛ 예측 가능한 signal이 존재함.

입니다.

### Oracle은 이렇게 써야 합니다

Oracle은:

* 데이터 파이프라인 sanity check
* maximum attainable upper bound
* cross-sectional dispersion 확인

용도로 남기되,

**predictability 증거로 사용하면 안 됩니다.**

예측 가능성 확인은 별도로:

* session 내 label permutation null model
* feature time-shift model
* simple momentum/value/flow baseline
* OOF rank IC
* Top-K precision/hit rate
* model vs shuffled-model paired tail utility

를 봐야 합니다.

이것은 현재 분석의 중요한 수정점입니다.

---

# 3. 가장 큰 문제: 학습하는 target과 평가하는 target이 다름

현재 label 정의는:

```text
gross_return
- point_in_time_risk_projection
- reference_cost
        ↓
session robust z-score
        ↓
net_alpha_target
```

입니다.

그런데 이번 `ml-cmp`의 route는:

```text
unhedged_absolute
```

이고 screen에서는:

```text
gross_return - reference_cost
```

를 사용합니다.

즉 모델에게는:

```text
"시장/위험 성분을 제거한 residual alpha를 예측해"
```

라고 가르쳐 놓고,

평가할 때는:

```text
"실제 long-only absolute return이 높은 종목을 골라"
```

라고 요구하고 있습니다.

### 이 차이는 작지 않습니다

예를 들어:

| 종목 | Gross | Risk projection | Cost | residual target 방향 |
| -- | ----: | --------------: | ---: | -----------------: |
| A  |   +8% |             +7% | 0.5% |              +0.5% |
| B  |   +5% |             +1% | 0.5% |              +3.5% |

Residual model은 B를 선호합니다.

하지만 unhedged absolute wealth라면 A가 더 좋습니다.

현재 구조에서는 B를 맞게 고른 모델이 screen에서 **오답 취급될 수도 있습니다.**

### 따라서 먼저 route별 target을 분리해야 합니다

#### Unhedged route

```text
y_mean = gross_return - expected_cost
```

또는 cross-sectional learner라면:

```text
y_rank = rank_session(gross_return - expected_cost)
```

#### Hedged route

```text
y_mean = risk_residual - expected_cost
```

#### Risk projection

unhedged에서는 target에서 빼버리는 대신:

```text
portfolio risk constraint
context feature
position sizing
market exposure control
```

로 보내는 것이 더 논리적입니다.

현재처럼 residual target을 계속 사용하려면 전략 자체를 **hedged residual strategy**로 정의해야 합니다.

---

# 4. 그렇다고 종목 모델이 CAGR을 직접 학습해야 하는 것은 아님

여기에서 반대로:

> 그러면 모델 label을 CAGR이나 log wealth로 바꿔야 하나?

라고 갈 필요는 없습니다.

그것도 권하지 않습니다.

복리 wealth는:

$$
W_{t+1}=W_t(1+r_{p,t})
$$

이고 \(r_p\)는 한 종목의 return이 아니라:

* 모든 종목의 기대수익
* covariance
* weight
* transaction cost
* current positions
* turnover
* fills
* cash
* capacity

의 함수입니다.

따라서 권장 구조는:

```text
ML
 ↓
Expected return distribution / ranking
 ↓
Calibration
 ↓
Portfolio optimizer
 ↓
Execution
 ↓
Actual portfolio return
 ↓
log-growth / CAGR
```

입니다.

즉 현재 프로젝트가 지향하는 **"ML 예측과 compound portfolio optimization의 분리"는 유지해야 합니다.**

잘못된 것은 목표가 아니라 **중간 objective alignment**입니다.

---

# 5. H / C / K를 전부 같은 성격의 hyperparameter로 보면 안 됨

이 부분이 질문의 핵심입니다.

현재 구조에서 H/C/K를 명확하게 분리하는 것이 좋습니다.

## H = forecast target

Horizon은 label 자체를 바꿉니다.

```text
H=5
H=10
H=20
```

은 서로 다른 예측 문제입니다.

따라서 **모든 family가 H별로 다시 학습되는 것이 맞습니다.**

현재 label dataset도 horizon을 서로 inner join하지 않고 독립 partition으로 유지하는데, 이 설계는 좋습니다.

---

## C = execution policy

C는 몇 session마다 portfolio를 다시 판단하는지를 뜻합니다.

원칙적으로 regression model을 C마다 다시 학습할 필요는 없습니다.

```text
Model(H=10)
   ├─ C=5 replay
   └─ C=10 replay
```

가 가능합니다.

하지만 현재 exact replay 구현은 **H가 실제 holding period를 강제하지 않고 decision window만 제한**한다고 명시합니다.

그래서:

```text
H20 / C10
```

이면 모델은 20-session outcome을 배우면서 포트폴리오는 10-session마다 교체될 수 있습니다.

이것은 반드시 잘못은 아니지만 target/execution mismatch가 다시 발생할 수 있습니다.

### 우선은

```text
H5/C5
H10/C10
H20/C20
```

로 signal 자체를 비교한 다음,

survivor에 대해:

```text
H10/C5
H20/C10
```

같은 faster-review policy를 별도 execution experiment로 보는 편이 더 깔끔합니다.

---

# 6. K는 모델마다 의미가 다름

이 부분은 **통일하면 안 되는 핵심 변수**입니다.

### Elastic / Huber / LGBM / ExtraTrees / Quantile

이 모델들에게 K는 기본적으로 **portfolio parameter**입니다.

모델은:

```text
score_i
```

를 출력하고,

그 이후:

```text
K=12
K=16
K=20
```

을 replay해 볼 수 있습니다.

즉:

```text
한 번 학습
 → 여러 K replay
```

가 맞습니다.

### LambdaRank

LambdaRank는 다릅니다.

현재 relevance 자체가:

```text
session별 실제 Top-K = 1
나머지 = 0
```

으로 만들어집니다.

따라서:

```text
LambdaRank(H10,K12)
```

과

```text
LambdaRank(H10,K20)
```

은 **다른 학습 문제**입니다.

따라서 현재처럼 K를 모든 모델에 동일한 execution parameter처럼 다루면 안 됩니다.

제가 추천하는 candidate 구조는:

```text
ModelCandidate
    family
    horizon
    lookback
    feature_view
    target_spec
    train_top_k?   # LambdaRank만 존재
```

그리고 별도로:

```text
ExecutionCell
    rebalance_cadence
    execution_top_k
    policy_profile
    cost_scenario
```

로 분리하는 것입니다.

---

# 7. 현재 LambdaRank에는 추가적인 K 관련 구현 문제가 있음

현재 `family_training_profile()`은:

```text
lambdarank_truncation_level = top_k
```

라고 정의합니다.

그런데 실제 `lgb.train()` params에는 해당 값을 넘기지 않는 경로가 있습니다.

즉 코드에서는 K와 truncation의 일치를 검사하지만 실제 learner에게는 라이브러리 default가 적용될 수 있습니다.

더 심각하게 full OOF `_fit_one_fold()`에서는:

```python
family_training_profile(
    family,
    top_k=12,
    screen=False,
)
```

처럼 K=12가 hard-coded 되어 있습니다.

따라서 앞으로:

```text
K16
K20
K24
```

를 실험하더라도 screen과 full OOF semantics가 달라질 수 있습니다.

### LambdaRank는 다음을 하나로 묶어야 합니다

```text
relevance K
=
lambdarank truncation K
=
evaluation NDCG@K
=
candidate K
```

또는 아예 exact-K binary relevance를 버리고:

```text
Top 1%  : relevance 4
Top 5%  : relevance 3
Top 10% : relevance 2
Top 25% : relevance 1
else    : 0
```

같은 graded relevance를 사용하면 하나의 ranker를 K12/K16/K20에 더 쉽게 재활용할 수 있습니다.

둘 중 어느 것이 더 좋은지는 실험 대상입니다.

---

# 8. 현재 feature engineering은 family 특성을 충분히 구분하지 못함

현재 `family_training_profile()`을 보면 사실상 모든 모델이:

```text
winsor_rank_robust
```

transform을 공유합니다.

Research schema에서는 source 하나당 대체로:

```text
winsor
rank
sector_rank
missing
robust
```

가 생성됩니다.

이 때문에 screen design이 87 columns까지 늘어났습니다.

이것은 **공통 input representation으로는 편하지만 모델별 최적 representation은 아닙니다.**

### ElasticNet

적합:

```text
rank
sector_rank
missing
robust standardized level
소수의 사전등록 interaction
```

여기서는 scale normalization이 중요합니다.

ElasticNet이 현재 1위였던 것도 이런 representation과 상대적으로 궁합이 좋을 가능성이 있습니다.

다만 이것은 현재 결과만으로 입증할 수는 없습니다.

---

### Huber

Elastic과 비슷하지만 feature 중복을 더 줄이는 것이 좋습니다.

예:

```text
winsor
robust
rank
sector_rank
```

네 개를 전부 동시에 넣으면 상당히 높은 collinearity가 생깁니다.

Huber는 outlier robustness와 **feature redundancy robustness가 같은 것이 아닙니다.**

따라서 Huber는:

```text
rank + sector_rank
```

중심의 compact view와

```text
robust level
```

중심 view를 나눠 비교할 가치가 있습니다.

---

### LightGBM

여기는 현재 구조가 특히 아쉽습니다.

Tree는 스스로 nonlinear threshold와 interaction을 찾을 수 있기 때문에:

```text
raw/winsor level
rank
sector_rank
missing
```

정도면 충분합니다.

`robust standardization`은 tree에 꼭 필요하지 않습니다.

오히려 현재처럼 동일 변수의 표현을 여러 개 제공하면 feature importance가 여러 correlated copy로 나뉘고 feature selection 안정성도 떨어질 수 있습니다.

그리고 현재 이름은 `rawnet_lgbm_v2`인데 실제 profile은 raw-only가 아니라 동일한 `winsor_rank_robust` matrix입니다.

---

### ExtraTrees

마찬가지로:

```text
raw/winsor
rank
sector rank
context
```

구조가 적합합니다.

현재 30/50 trees 수준은 90만 row 규모 noisy financial problem에서 최종 성능 설정이라기보다는 **screen용 저비용 baseline**에 가깝습니다.

성능 개선 단계에서는 최소한:

```text
n_estimators
max_features
min_samples_leaf
max_depth
```

에 대한 아주 작은 pre-registered grid가 필요합니다.

대규모 Optuna를 하자는 의미는 아닙니다.

---

### Quantile model

현재:

```text
quantile = 0.2
```

로 `net_alpha_target`의 조건부 q20을 학습합니다.

이 모델의 성격은 다른 모델과 명백히 다릅니다.

이는 mean return predictor가 아니라:

> **"상대적으로 보수적인 downside-adjusted score"**

에 가깝습니다.

그런데 현재 screen에서는 다른 모델과 똑같이:

```text
prediction 상위 K
→ 실제 평균 return
```

으로 평가됩니다.

완전히 틀린 평가는 아니지만 quantile model의 장점을 충분히 보지 못합니다.

향후에는:

```text
μ predictor
q20 predictor
```

를 별개로 두고,

예를 들어:

$$
score = \hat\mu-\lambda(\hat\mu-\hat q_{20})
$$

또는 q20을 no-trade / sizing constraint로 사용하는 편이 compound objective와 더 잘 맞습니다.

---

# 9. ALPHA / RISK / LIQUIDITY 역할 분리는 좋은데 너무 강함

현재 feature contract는:

> learner에는 ALPHA source만 들어가며 RISK / LIQUIDITY / CONTROL은 residualization, sizing, tradeability 등에만 사용

하도록 설계되어 있습니다.

개념적으로는 좋은 설계입니다.

문제는 **conditional alpha**입니다.

예를 들어:

```text
flow signal의 효과
    ×
volatility regime
```

는 단순 risk signal이 아니라:

```text
"이 alpha가 언제 작동하는가"
```

라는 조건부 정보입니다.

---

# 10. 실제로 `flow_intensity × vol_regime` interaction은 현재 생성되지 않음

이건 구체적인 구현 문제입니다.

현재 interaction 정의:

```python
flow_intensity_20d × vol_regime
flow_consensus × relative_trend_score
```

가 있습니다.

그런데 interaction은:

```text
representative source에 두 feature가 모두 존재할 때
```

만 생성됩니다.

문제는 `representative`가 ALPHA source에서만 만들어지고:

```text
vol_regime = RISK
```

로 등록되어 있다는 점입니다.

따라서:

```text
flow_intensity × vol_regime
```

는 현재 로직상 사실상 생성될 수 없습니다.

반면:

```text
flow_consensus × relative_trend_score
```

는 둘 다 ALPHA라 생성될 수 있습니다.

더구나 코드 주석은 이를 **linear-only interaction**이라고 하지만 `source_groups`는 모든 family가 공유하므로 생성된 interaction은 tree family에도 전달됩니다.

즉 현재는:

```text
의도:
linear model만 두 interaction 사용

실제:
interaction 1 → 생성 안 됨
interaction 2 → 모든 family가 볼 수 있음
```

입니다.

이 부분은 바로 수정해야 합니다.

### 역할 구조를 다음처럼 바꾸는 것을 권합니다

```text
ALPHA
CONTEXT
RISK
LIQUIDITY
CONTROL
```

그리고:

```text
ALPHA -> 직접 predictor 허용
CONTEXT -> 직접 alpha 사용 금지 또는 제한
ALPHA × CONTEXT -> 허용
RISK -> portfolio layer
LIQUIDITY -> cost/capacity layer
```

처럼 명확하게 해두는 편이 낫습니다.

---

# 11. Fundamental feature는 PIT certification을 다시 확인해야 함

이번 실험에서:

* `bp_ratio`
* `ep_ratio`

가 거의 모든 상위 모델 feature group에 등장합니다.

그런데 현재 v2 feature contract에서는 이 둘에 대해 명시적으로:

```text
disclosure_date
```

를 사용하도록 별도 강화되어 있습니다.

반면 canonical v1 contract는 더 일반적인 `next_session_open` availability rule을 갖습니다.

따라서 이번 snapshot의 `bp_ratio/ep_ratio`가 실제 공시 시점 기준 PIT인지 **dataset manifest까지 확인해야 합니다.**

이것이 확인되지 않는다면:

```text
bp_ratio
ep_ratio
```

는 성능 향상을 논하기 전에 제거하는 것이 맞습니다.

특히 현재 attribution에서 너무 자주 상위에 나오므로 이 검증은 우선순위가 높습니다.

---

# 12. Missing 처리는 현재 main에서는 상당 부분 개선됨

`ml-cmp`에는:

* `ret_21_60d` 14.584%
* `vol_regime` 14.584%
* `volatility_60d` 14.584%
* `bp_ratio`, `ep_ratio` 6.446%

정도의 missing이 있습니다.

현재 main에서는 fold-local training median을 freeze하고:

```text
median imputation
+
original missing indicator
```

를 만든 뒤 derived feature를 생성하도록 수정되었습니다.

이 방향은 맞습니다.

따라서 지금은 missing 자체보다:

> 모든 family가 동일한 imputed representation을 가져야 하는가?

가 더 중요한 질문입니다.

Tree에는 native missing semantics를 활용할 수 있는 모델이라면 별도 path를 제공할 가치가 있고, linear에는 현재 방식이 적합합니다.

---

# 13. Feature selection에 nested validation이 필요함

현재 main screen은 outer fold의:

```text
train
validation
```

을 만든 다음,

여러 feature prefix를 train에서 fit하고 **같은 validation에서 경제성 tail LB를 비교하여 최적 prefix를 선택**합니다.

그리고 선택된 prefix의 같은 validation LB를 screen evidence로 사용합니다.

즉:

```text
validation
   ├─ feature count 선택
   └─ 성능 측정
```

에 동시에 쓰입니다.

이는 selection bias를 발생시킵니다.

현재 전부 음수라 false positive가 발생하지 않았지만 성능이 개선되어 0 근처를 넘기 시작하면 문제가 됩니다.

### 수정

```text
Outer fold
│
├── Outer train
│     ├── Inner train
│     └── Inner validation
│          ↓
│        feature / hyperparameter 선택
│
└── Outer validation
      ↓
    단 한 번 평가
```

로 만들어야 합니다.

코드에 `_inner_folds_from_train()`도 존재하지만 main economic screen selection과 완전히 통합되어 있지는 않습니다.

---

# 14. 현재 bootstrap confidence는 숫자상 너무 거칠음

이번 run:

$$
\alpha=0.0027777778
$$

bootstrap:

$$
R=360
$$

입니다.

그러면 lower tail에 기대되는 bootstrap sample은:

$$
360 \times 0.002777 \approx 1
$$

개입니다.

즉 0.2778 percentile을 **사실상 최악의 bootstrap draw 1개로 추정하는 수준**입니다.

엄격한 confidence interval처럼 보이지만 실제로는 tail quantile estimator 자체의 Monte-Carlo noise가 큽니다.

예를 들어 tail에 최소 20개 draw를 확보하려면:

$$
R \ge \frac{20}{0.002777}
\approx 7200
$$

입니다.

50개라면 약 18,000회입니다.

session-level scalar bootstrap은 learner fit에 비하면 매우 저렴하므로 이 부분은 늘리는 것이 좋습니다.

---

# 15. IID session bootstrap도 H/C에 따라 수정 필요

screen helper는 선택된 session들을 다시 IID bootstrap합니다.

H10/C10은 겹침이 비교적 작아 아직 낫습니다.

하지만 앞으로 문서에서 제안한:

```text
H20/C10
```

을 하면 두 인접 rebalance의 forward outcome이 overlap합니다.

그런데 IID bootstrap을 하면 독립 관측으로 취급됩니다.

따라서:

```text
moving block bootstrap
stationary bootstrap
또는 OOF segment cluster bootstrap
```

이 필요합니다.

그리고 현재 fold별 lower bound를 각각 계산한 뒤 그 lower bound들을 단순 평균하는데, **quantile의 평균은 전체 confidence lower bound가 아닙니다.**

최종적으로는 outer validation session/block evidence를 합친 후 study-level block bootstrap을 수행하는 것이 더 적절합니다.

---

# 16. 90만 row라고 해서 독립 표본이 90만 개는 아님

이번 dataset:

* rows: 918,443
* sessions: 2,479
* instruments: 2,297

입니다.

전략 수익 관점에서 중요한 독립 단위는 row보다 **session/time block**입니다.

같은 날짜의 수백 종목 return은:

* 시장
* sector
* macro
* liquidity

를 공유합니다.

canonical `models.py`에는 이 문제 때문에 **각 session의 총 sample weight가 동일하도록 하는 session-balanced weighting**이 구현되어 있습니다.

그런데 model-selection의 family별 `_fit_family`와 `_fit_one_fold`에서는 이 weighting이 일관되게 사용되지 않습니다.

이것 역시 구현 경로가 두 군데 존재하면서 semantics가 갈라진 사례입니다.

최근에 상장 종목 수가 많으면 최근 session이 row 수만큼 더 큰 weight를 받을 수 있습니다.

따라서 모든 regression family에서 최소한:

```text
session total weight = constant
```

를 적용하는 것을 권합니다.

---

# 17. 현재 screen의 Top-K objective도 실제 portfolio와 다름

screen은 매 session:

```text
prediction 상위 K
→ K개 실제 net return 단순 평균
```

을 계산합니다.

하지만 실제 portfolio constructor는:

* score
* inverse volatility
* single-name cap
* sector cap
* participation
* gross exposure
* turnover
* alpha lower bound

등을 사용합니다.

따라서:

```text
screen Top-K equal-weight utility
```

와

```text
실제 portfolio compound utility
```

는 상당히 다릅니다.

screen이 cheap proxy인 것은 괜찮습니다.

문제는 현재 이 proxy가:

> **LB > 0이 아니면 실제 replay를 아예 하지 않는다**

는 hard gate라는 것입니다.

이번 run이 바로 그렇게 끝났습니다.

### 권장

screen은:

```text
obviously bad model 제거
```

정도로 사용하고,

예를 들어 상위 2~3 family는 **LB가 약간 음수여도 exact replay**까지 보내는 구조가 더 낫습니다.

특히 현재 Elastic:

```text
tail LB -0.00458
SE      +0.00588
```

수준은 압도적으로 실패했다기보다 불확실성 구간 내에 가까운 상태입니다.

여기서 실제 portfolio weighting/cash option까지 보지 않고 종료하는 것은 compute optimization 관점에서는 이해되지만 **alpha discovery 성능 관점에서는 너무 공격적인 pruning**입니다.

---

# 18. “final compound wealth”와 현재 champion selection도 완전히 일치하지 않음

full OOF/replay에서 survivor들이 생긴 뒤 현재 코드는:

```python
selected_family = str(survivors[0].family)
```

로 사실상 첫 survivor를 선택합니다.

즉 둘 이상의 family가 통과했을 때:

```text
stress lower log-growth가 가장 높은 모델
```

을 반드시 고르는 것이 아닙니다.

이는 사용자가 말한:

> 최종 복리자산증식 극대화

와 직접 충돌합니다.

### 최종 selection은 명시적으로

먼저 hard gates:

```text
base LB > 0
stress LB > 0
MDD <= limit
coverage >= limit
capacity pass
```

후,

$$
\arg\max_m
LB_\alpha
\left[
\sum_t \log(1+r_{p,t}^{stress,m})
\right]
$$

으로 해야 합니다.

동률이면:

1. stress LB
2. base LB
3. lower MDD
4. lower turnover
5. simpler model

순으로 tie-break하면 됩니다.

---

# 19. 지금 여섯 모델의 “다양성”은 알고리즘 다양성이지 정보 다양성은 아님

현재:

```text
Elastic
Huber
LGBM
ExtraTrees
Hist Quantile
LambdaRank
```

는 알고리즘은 다릅니다.

하지만 대부분:

```text
동일 source
동일 transformed matrix
거의 동일 residual target
동일 horizon
동일 screen Top-K
```

을 봅니다.

따라서 prediction correlation도 상당할 가능성이 있습니다. 실제 수치는 확인이 필요합니다.

모델 다양성은 ideally:

```text
algorithm diversity
+
target diversity
+
feature-view diversity
+
error diversity
```

여야 합니다.

추천 역할은:

| Family     | 역할                                     |
| ---------- | -------------------------------------- |
| Elastic    | 안정적 sparse linear mean/rank alpha      |
| Huber      | outlier-robust linear alpha            |
| LGBM       | nonlinear mean net-return              |
| ExtraTrees | bagged nonlinear / interaction         |
| Quantile   | downside / confidence estimator        |
| LambdaRank | direct cross-sectional top-tail ranker |

처럼 명시하는 것입니다.

현재처럼 마지막 세 모델까지 단순히 `score → Top-K`로만 비교하면 이 역할 차이가 상당 부분 사라집니다.

---

# 20. 모델별 권장 feature architecture

| Family     | 추천 feature view                                     | K 학습 의존성 |
| ---------- | --------------------------------------------------- | -------- |
| Elastic    | rank + sector_rank + missing + 제한 interaction       | 없음       |
| Huber      | compact rank/robust representation                  | 없음       |
| LGBM       | winsor/raw + rank + sector rank + context + missing | 없음       |
| ExtraTrees | winsor/raw + ranks + context                        | 없음       |
| Quantile   | raw/winsor + rank + downside context                | 없음       |
| LambdaRank | rank + sector rank + selected raw/context           | **있음**   |

특히 tree family에 대해서는:

```text
ALPHA
+
RISK context
+
LIQUIDITY context
```

를 허용하되,

risk/liquidity를 standalone alpha로 해석하지 않도록 attribution에서 역할을 구분하는 방식이 좋습니다.

---

# 21. Horizon에 따라 feature도 약간 달라져야 함

현재 feature set에는:

* ret 2~5
* ret 6~20
* ret 21~60
* 120d trend
* 20d flow
* valuation
* volatility

등 여러 scale이 이미 있습니다.

이 자체는 좋습니다.

다만 앞으로는 feature를 단순 고정 dictionary가 아니라:

```text
H
H/2
2H
4H
```

근처의 multi-scale representation으로 체계화하는 것이 좋습니다.

예를 들어 이는 **검증할 가설**이지 성능 보장은 아니지만:

```text
H5
 → short flow / reversal / intraday 비중 높은 view

H10
 → short-medium flow + momentum

H20
 → medium trend/value + regime interaction
```

같은 inductive prior를 family-specific feature view에 반영할 수 있습니다.

중요한 것은 H별로 완전히 다른 수십 개 feature를 임의로 만드는 것이 아니라 **동일 경제 개념의 time scale을 H에 맞게 확장하는 것**입니다.

---

# 22. Lookback 1260도 전 모델 공통 최적이라고 볼 근거는 없음

현재 run은 1,260 sessions입니다.

5년 정도의 rolling training window입니다.

이 값을 모든 family에 동일하게 적용할 이유는 없습니다.

테스트 grid 정도로:

| Family            | 우선 시험할 lookback        |
| ----------------- | ---------------------- |
| Elastic / Huber   | 756 / 1260 / expanding |
| LGBM / ExtraTrees | 504 / 756 / 1260       |
| Quantile          | 756 / 1260             |
| LambdaRank        | 504 / 756 / 1260       |

를 권합니다.

이 숫자들이 최적이라는 뜻은 아닙니다.

목적은:

* linear → 더 긴 history에서 안정화될 가능성
* nonlinear → regime drift 때문에 짧은 window가 유리할 가능성

이라는 가설을 검증하는 것입니다.

현재 fast model-selection study는 horizon과 lookback을 한 번에 하나씩만 허용하고 있습니다.

그러므로 별도 run을 한다면 **여러 run 전체를 하나의 experiment family로 묶은 multiplicity ledger**가 필요합니다.

각 run마다 α=0.05를 새로 시작하면 안 됩니다.

---

# 23. 현재 hyperparameter는 “모델 비교용 baseline” 수준에 가까움

예를 들어 current screen:

```text
LGBM: 20 rounds
full: 50 rounds

ExtraTrees: 30 / 50 trees

Quantile HGB: 30 / 100 iterations

Elastic l1_ratio = 0.5
Huber epsilon = 1.35
```

등 상당히 고정되어 있습니다.

성능을 위해 거대한 hyperparameter search를 돌릴 필요는 없습니다.

오히려 작은 사전등록 grid가 낫습니다.

예:

```text
Elastic
  alpha_fraction: 0.02 / 0.05 / 0.10
  l1_ratio:       0.2 / 0.5 / 0.8

LGBM
  leaves:         15 / 31
  min_data_leaf:  large / larger
  feature_frac:   0.7 / 1.0
  l1/l2:          2~3 presets

ExtraTrees
  leaf size:      20 / 50 / 100
  max_features:   sqrt / 0.5

Quantile
  q:              .10 / .20 / .30
```

를 inner validation에서 선택하는 정도가 적절합니다.

금융 데이터에서는 trial 수를 늘리는 것이 alpha를 늘리는 것보다 **selection overfit을 늘릴 위험**도 큽니다.

---

# 24. LambdaRank는 binary exact-K보다 한 단계 더 개선할 수 있음

현재 exact K relevance는:

```text
top K = 1
else = 0
```

입니다.

이렇게 하면 실제:

```text
+20%
+5%
+1%
```

인 세 종목도 top-K 안에서는 모두 똑같이 1입니다.

최종 wealth를 위해서는 순위의 magnitude도 중요합니다.

따라서 비교 실험을 권합니다.

### A. Exact-K binary

현재 방식.

장점:

* 실제 Top-K 목적과 직접 일치.

단점:

* 정보 손실이 큼.
* K마다 refit.

### B. Graded relevance

예:

```text
top 1%  -> 4
top 5%  -> 3
top 10% -> 2
top 25% -> 1
```

장점:

* 더 많은 ordering information.
* 여러 K에 재사용 가능.

### C. Utility-weighted rank objective

route-aligned net utility의 quantile/utility gain을 relevance로 변환.

개인적으로는 **B를 다음 challenger로 추가**할 가치가 가장 높습니다.

---

# 25. cost도 최종 compound 목표에서는 분리하는 편이 장기적으로 더 좋음

현재 label에는 `reference_cost`가 들어가고, screen도 reference cost를 사용합니다. 평균 cost가 이번 run에서 약 0.00436입니다.

하지만 compound wealth가 증가하면 position notional도 변합니다.

그러면:

```text
10M account의 cost ranking
```

과

```text
50M account의 cost ranking
```

이 동일하지 않을 수 있습니다.

최종적으로는:

```text
ML:
gross expected return distribution 예측

Cost model:
notional / ADTV / liquidity에 따른 expected execution cost

Portfolio:
expected gross - actual candidate cost
```

구조가 더 확장성이 높습니다.

즉 fixed reference cost는 **research normalization**에는 쓸 수 있지만 final alpha definition 자체에 너무 강하게 bake-in하지 않는 편이 좋습니다.

실제 replay가 stateful cost를 이미 처리한다는 점에서도 이 구조가 자연스럽습니다.

---

# 26. `compound_alpha.py`는 최종 evidence source로 사용하면 안 됨

현재 이 모듈에는 명시적으로:

```text
retired-pseudo-study
```

가 남아 있고, 과거 deterministic CAGR scaffolding도 존재합니다.

따라서 최종 compound evidence는 무조건:

```text
execution_equivalent replay
```

에서 가져와야 합니다.

현재 `execution_replay.py`가 정확히 이 역할을 하도록 설계되어 있으므로, **이 경로를 유일한 compound truth source로 만드는 것이 맞습니다.**

---

# 27. 최종적으로 추천하는 아키텍처

현재 코드를 완전히 갈아엎을 필요는 없습니다.

핵심 경계를 다시 정의하면 됩니다.

```text
┌──────────────────────────────┐
│ 1. PIT DATA                  │
│ price / flow / fundamental   │
│ disclosure / tradability     │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 2. FEATURE REGISTRY          │
│ ALPHA                        │
│ CONTEXT                      │
│ RISK                         │
│ LIQUIDITY                    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 3. ROUTE-SPECIFIC LABEL      │
│ unhedged → gross-cost        │
│ hedged   → residual-cost     │
└──────────────┬───────────────┘
               ↓
┌────────────────────────────────────────┐
│ 4. FAMILY-SPECIFIC MODEL SPEC          │
│                                        │
│ Elastic → linear view                  │
│ Huber   → robust linear view           │
│ LGBM    → nonlinear/context view       │
│ Extra   → bagged nonlinear view        │
│ Quantile→ downside target              │
│ Lambda  → ranking target + train K     │
└──────────────┬─────────────────────────┘
               ↓
┌──────────────────────────────┐
│ 5. OUTER PURGED WF           │
│                              │
│ Inner:                       │
│ feature / hp / K_train       │
│                              │
│ Outer:                       │
│ untouched OOF prediction     │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 6. OOF CALIBRATION           │
│ score → decimal μ            │
│ score → downside / q20       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 7. CHEAP ECONOMIC SCREEN     │
│ diagnostic / weak pruning    │
│ not hard final gate          │
└──────────────┬───────────────┘
               ↓
┌───────────────────────────────────────┐
│ 8. EXECUTION FRONTIER                 │
│                                       │
│ regression: (H model) × C × K         │
│ Lambda:     (H,K model) × C           │
│                                       │
│ policy profile                        │
│ base/stress cost                      │
└──────────────┬────────────────────────┘
               ↓
┌──────────────────────────────┐
│ 9. EXACT EXECUTION REPLAY    │
│ fills / T+2 / turnover       │
│ capacity / cash / cost       │
└──────────────┬───────────────┘
               ↓
┌───────────────────────────────────────┐
│ 10. CHAMPION SELECTION                │
│                                       │
│ hard constraints                     │
│   MDD / coverage / capacity           │
│                                       │
│ maximize                              │
│   stress lower-bound log growth       │
└──────────────┬────────────────────────┘
               ↓
┌──────────────────────────────┐
│ 11. OPTIONAL ENSEMBLE        │
│ only if paired incremental   │
│ lower bound > 0              │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 12. LOCKED HOLDOUT           │
│ final certification only     │
└──────────────────────────────┘
```

---

# 28. 코드 구조도 `if family == ...`에서 FamilySpec으로 바꾸는 것을 권함

현재 `model_selection.py` 안에 family별 로직이 여러 곳에 반복됩니다.

그 결과 이미:

* main screen LambdaRank exact-K
* 다른 attribution LambdaRank global median
* K=12 hard-code
* `family_training_profile`과 실제 params 중복
* canonical `models.py`의 session weighting과 research model-selection의 weighting 차이

같은 semantic drift가 생겼습니다.

다음 abstraction을 만드는 것이 좋습니다.

```python
FamilySpec:
    family_id
    target_kind
    feature_view
    estimator_factory
    k_dependency
    fit()
    predict()
    attribution()
    calibration_kind
```

그러면:

```text
screen
inner selection
full OOF
final fit
```

모두 같은 `FamilySpec`을 호출합니다.

이 변경은 단순 코드 정리가 아니라 **실험 validity를 보장하는 아키텍처 개선**입니다.

---

# 29. 당장 실험 순서는 이렇게 하는 것이 가장 효율적임

지금 바로 feature를 수십 개 추가하거나 Optuna를 돌리는 것은 권하지 않습니다.

### Stage 1 — correctness / alignment

먼저 이것만 수정:

1. **unhedged target을 gross-cost로 route align**
2. LambdaRank의 relevance도 같은 route utility 사용
3. `lambdarank_truncation_level=K` 실제 LightGBM params에 전달
4. full OOF K=12 hard-code 제거
5. `flow_intensity × vol_regime` interaction role 문제 수정
6. linear-only interaction 실제 enforce
7. feature-prefix selection inner/outer 분리
8. final survivor를 stress compound objective로 정렬
9. bootstrap을 block-based + 충분한 resample로 변경
10. family logic을 최소한 하나의 parameter/spec source로 통합

이 단계가 가장 중요합니다.

---

### Stage 2 — baseline rerun

다시:

```text
H10 / C10 / K12
lookback 1260
6 families
```

을 그대로 실행합니다.

이때 현재 `ml-cmp`와 비교해야 합니다.

그래야 architecture correction 하나로:

```text
tail capture
feature prefix
model ordering
```

이 얼마나 변했는지 알 수 있습니다.

---

### Stage 3 — horizon

그 후:

```text
H5/C5
H10/C10
H20/C20
```

을 비교합니다.

여기까지는 C=H로 통제합니다.

---

### Stage 4 — execution cadence

각 H survivor에 대해서만:

```text
H10 → C5/C10
H20 → C10/C20
```

을 exact replay합니다.

---

### Stage 5 — K

Regression family는 refit 없이:

```text
K12
K16
K20
K24
```

replay.

LambdaRank는:

```text
(H,K12)
(H,K16)
(H,K20)
```

별도 fit하거나 graded relevance challenger를 비교합니다.

현재 execution frontier 자체는 H/C/K를 별도로 표현할 수 있도록 이미 설계되어 있습니다.

---

### Stage 6 — family-specific feature views

여기서 비로소:

```text
linear-view
tree-view
ranking-view
downside-view
```

를 비교합니다.

---

### Stage 7 — small hyperparameter grids

마지막에 family별 작은 grid를 적용합니다.

---

# 30. 최종 판단

질문의 두 선택지 중에서는 명백히 **두 번째에 가깝습니다.**

> “다른 성격의 알고리즘을 두고 최종 복리자산증식을 기준으로 좋은 모델을 찾는 방향 자체가 잘못되었나?”

**아닙니다. 유지하는 것이 좋습니다.**

다만 현재는 단순히 “top-k/horizon 값이 조금 잘못됐다” 수준보다 더 근본적인 문제가 있습니다.

핵심은 다음 세 층의 정합성입니다.

```text
① 모델이 무엇을 학습하는가
        ↕
② screen이 무엇을 보상하는가
        ↕
③ 실제 portfolio가 무엇으로 돈을 버는가
```

현재:

```text
① residual standardized alpha
② unhedged Top-K arithmetic excess
③ constrained/cost-aware compound equity
```

로 세 목적이 다릅니다.

여기에 동일 preprocessing, LambdaRank K 문제, nested selection 문제, bootstrap 문제까지 겹쳐 있습니다.

따라서 지금 가장 큰 개선 여지는 **새 알고리즘 추가가 아니라 이 세 층을 정렬하는 것**입니다.

특히 제가 우선순위를 3개만 고른다면:

**1. `route-specific target` 도입**
unhedged면 모델도 `gross-cost` ordering을 배우게 할 것.

**2. `(model candidate)`와 `(execution cell)` 분리**
H는 모델, C는 execution, K는 regression에는 execution / LambdaRank에는 training+execution으로 명확하게 분리할 것.

**3. family별 feature/fit semantics를 하나의 `FamilySpec`으로 통합하고 nested OOF 이후 exact replay의 stress lower log-growth로 최종 champion을 선택할 것.**

이 세 가지가 정리된 뒤에야 H5/H10/H20, K12/16/20/24, lookback, feature engineering, LightGBM tuning을 하는 것이 의미가 있습니다.

현재 `ml-cmp`에서 ElasticNet이 1위라고 해서 linear model 중심으로 방향을 바꾸거나, LambdaRank가 꼴찌라고 해서 ranking model을 버릴 근거도 아직 없습니다. **현재 비교 환경 자체가 family별 목적을 공정하게 평가하도록 충분히 정렬되어 있지 않습니다.**
