가능하다. 이번에는 기존 `k-stock-engine`의 전략/ML/백테스트 구조를 사실상 폐기한다고 보고 설계하는 편이 낫다.

핵심 결론은 다음과 같다.

> **새 엔진의 v1은 ML을 사용하지 않는다.**
>
> 국내주식에서 비교적 근거가 있는 **Quality/Profitability + Value + Earnings Momentum + Foreign Flow** 네 종류의 독립적인 신호만 사용하고, Low-Volatility는 종목선정 알파보다는 포트폴리오 리스크 관리에 사용한다.
>
> 가격 Momentum은 한국 시장에서 장기적으로 일관되지 않은 연구결과가 있으므로 핵심 알파에서 제외하고 Challenger로 검증한다. 1983~2023 한국 데이터를 분석한 최근 연구에서는 개별주 momentum보다 reversal이 우세했다고 보고된 반면, 다른 한국 factor 연구에서는 momentum premium도 관찰되어 결과가 혼재한다. ([ScienceDirect][1])
>
> 최종 판단 기준은 **IC나 예측 정확도가 아니라, 동일한 실제 체결 엔진에서 발생한 OOS 순자산 성장률**이다.

아래 구조를 최종 목표 아키텍처로 권한다.

---

# 1. 시스템 목표부터 다시 정의

목표를

> "연 30% CAGR을 만드는 전략"

으로 정의하면 안 된다.

그 순간부터 연구자가 30%가 나올 때까지 파라미터를 건드리는 구조가 된다.

목표는 다음이어야 한다.

$$
\boxed{
\text{Maximize robust out-of-sample geometric growth}
}
$$

단 다음 제약을 만족해야 한다.

$$
MDD \le D_{\max}
$$

$$
Turnover \le T_{\max}
$$

$$
Capacity > Capital
$$

$$
StressReturn > 0
$$

그리고 수익률은 반드시

$$
r_t^{net}
=
r_t^{gross}
-
commission
-
tax
-
spread
-
slippage
-
impact
$$

여야 한다.

### Primary performance

$$
g
=
\frac{252}{N}
\sum_t \log(1+r_t^{net})
$$

즉 **Net annualized log-growth**를 가장 중요한 성과지표로 둔다.

CAGR과 거의 같은 의미지만 통계적으로 다루기 편하다.

---

# 2. 전체 아키텍처

복잡도를 크게 줄인다.

```text
                    ┌──────────────────────┐
                    │      KRX / DART      │
                    │      KIS / 기타      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Point-in-Time DB   │
                    │  immutable datasets  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Feature Engine     │
                    │ Q / V / E / Flow     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Strategy Engine    │
                    │ universe + ranking   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Portfolio Constructor│
                    │ risk + constraints   │
                    └──────────┬───────────┘
                               │
                               ▼
                       TargetPortfolio
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          Backtest Broker                 KIS Broker
                 │                           │
                 ▼                           ▼
               Fills                       Fills
                 └─────────────┬─────────────┘
                               ▼
                         Shared Ledger
                               │
                               ▼
                    Performance / Risk
```

가장 중요한 것은:

```python
TargetPortfolio = strategy.decide(
    market_snapshot,
    portfolio_state,
)
```

이 함수가 **백테스트와 실전에서 완전히 동일**해야 한다는 것이다.

백테스트 전용 전략 코드와 실전 전략 코드를 따로 만들지 않는다.

---

# 3. 코드 구조도 단순화

추천한다.

```text
src/
├── domain/
│   ├── instrument.py
│   ├── portfolio.py
│   ├── order.py
│   ├── fill.py
│   └── clock.py
│
├── data/
│   ├── schemas.py
│   ├── pit_store.py
│   ├── universe.py
│   └── corporate_actions.py
│
├── features/
│   ├── profitability.py
│   ├── value.py
│   ├── earnings.py
│   ├── foreign_flow.py
│   └── risk.py
│
├── strategy/
│   ├── scoring.py
│   ├── selection.py
│   └── portfolio.py
│
├── engine/
│   ├── decision.py
│   ├── backtest.py
│   ├── fill_model.py
│   └── ledger.py
│
├── adapters/
│   ├── krx.py
│   ├── dart.py
│   ├── kis.py
│   └── parquet.py
│
├── validation/
│   ├── walk_forward.py
│   ├── bootstrap.py
│   ├── metrics.py
│   └── robustness.py
│
└── live/
    ├── runner.py
    ├── reconciliation.py
    └── safety.py
```

이 정도면 충분하다.

### 없애는 것

초기에는 다음이 없다.

```text
models/
ml/
optimizer zoo
hedge sleeve
regime classifier
RL
deep learning
prediction server
factor discovery framework
10종류 portfolio optimizer
```

필요성이 증명됐을 때 추가한다.

---

# 4. 설계 원칙

전체 시스템에 딱 다섯 가지 원칙만 강제한다.

### ① Point-in-Time

미래정보 사용 불가능.

### ② Single Decision Engine

백테스트와 실전 전략 동일.

### ③ Net PnL First

IC가 아닌 거래 후 순수익이 최종 truth.

### ④ Minimal degrees of freedom

튜닝할 파라미터 수를 최소화.

### ⑤ Fail closed

데이터가 이상하면 적당히 보정해서 거래하지 않고 **NO_TRADE**.

---

# 5. 데이터부터 새로 구축

여기가 ML보다 훨씬 중요하다.

저장 구조는 간단히:

```text
data/
├── bronze/
├── silver/
├── gold/
└── artifacts/
```

## Bronze

원본 그대로 보존.

```text
KRX raw
DART raw
KIS raw
```

절대 overwrite하지 않는다.

---

# 6. Silver

정규화한 데이터.

핵심 테이블은 6개만 있으면 된다.

### `security_master`

```text
instrument_id
ticker
company_id
market
sector
listing_date
delisting_date
share_class
status
```

### `daily_market`

```text
session
instrument_id

open
high
low
close

volume
trading_value
market_cap
shares_outstanding
```

### `investor_flow`

```text
session
instrument_id

foreign_buy_value
foreign_sell_value
foreign_net_value

institution_net_value
retail_net_value
```

### `financial_facts`

```text
company_id
fiscal_period
filing_id
published_at
available_at

sales
gross_profit
operating_profit
net_income

assets
equity
cash
debt

operating_cashflow
capex
```

### `corporate_actions`

```text
instrument_id
effective_date
type

split
reverse_split
dividend
rights_issue
merger
delisting
```

### `disclosures`

```text
company_id
filing_id
filing_type
published_at
available_at
```

OpenDART는 상장사의 주요 재무계정 및 XBRL 원본 재무제표를 공식적으로 제공하므로 재무 데이터 기반은 DART 중심으로 만드는 것이 적절하다. ([Open DART][2])

---

# 7. 모든 데이터에 시간 4개를 생각해야 한다

가능한 데이터에는:

```text
event_time
published_at
available_at
ingested_at
```

개념을 둔다.

가장 중요한 것이:

```text
available_at
```

이다.

예를 들어 2025년 4분기 실적이 2026년 3월 15일 공시됐다면:

```text
fiscal_period = 2025Q4

available_at = 2026-03-15 이후
```

이다.

2026년 2월의 백테스트에서는 절대 존재하면 안 된다.

---

# 8. DART는 더 보수적으로 처리

역사적인 정확한 intraday 이용가능시간을 확실히 보장할 수 없다면 추정하지 않는다.

가장 안전한 정책:

> **DART 공시는 공시일 다음 거래일부터 사용.**

예:

```text
DART filing
2024-05-14

signal_available
2024-05-16 open
```

15일이 정상 거래일이라면 실제로는 하루 늦게 사용하는 셈이다.

그래도 좋다.

백테스트 수익률을 과장하는 것보다 훨씬 낫다.

---

# 9. Survivorship bias 제거

현재 상장 종목 목록으로 2015년을 백테스트하면 안 된다.

각 날짜마다:

```python
universe = securities_alive_at(t)
```

이어야 한다.

따라서:

```text
상장폐지주
합병된 회사
과거 관리종목
거래정지 종목
```

까지 역사적으로 존재해야 한다.

이게 확보되지 않으면 그 기간 백테스트는 **인증 불가능** 상태로 처리한다.

---

# 10. 가격은 두 종류를 갖는다

### Execution Price

실제 가격:

```text
raw open/high/low/close
```

### Research Price

Corporate action adjusted:

```text
adjusted_close
total_return_index
```

Factor 계산은 adjusted data.

주문 체결은 raw data.

둘을 섞으면 안 된다.

---

# 11. 첫 Universe

지나치게 넓히지 않는다.

### 포함

```text
KOSPI
KOSDAQ

보통주
비금융 기업
```

### v1 제외

```text
ETF
ETN
REIT
SPAC
우선주

금융업
관리종목
정리매매
거래정지

신규상장 < 252 trading days
```

금융회사는 재무구조 자체가 제조업/서비스업과 너무 다르므로 v1에서는 제외하는 편이 깔끔하다.

나중에 금융업 전용 fundamental model을 만들면 된다.

---

# 12. 유동성 필터

추천:

$$
MedianTradingValue_{60d}
$$

사용.

예를 들어 초기 연구 범위:

```text
60d median daily trading value
>= 20억원
```

정도로 시작한다.

단 `20억원`을 최적화해서는 안 된다.

Stress test만 한다.

```text
10억
20억
50억
```

에서 전략 특성이 크게 변하지 않는지만 확인한다.

---

# 13. 왜 이 4개 Alpha인가

새 Champion은 다음 네 개만 사용한다.

```text
1. Profitability / Quality
2. Value
3. Earnings Momentum
4. Foreign Flow
```

이론적으로 서로 상당히 다른 정보를 사용한다.

---

# 14. Alpha 1 — Profitability / Quality

가장 중심적인 factor로 두는 것을 권한다.

Novy-Marx는 gross profitability가 value와 비슷한 수준의 횡단면 수익률 설명력을 가질 수 있음을 보였고, 최근 retrospective 연구에서도 profitability가 여러 quality 전략을 설명하는 핵심 요소로 평가된다. ([NBER][3])

한국 시장 연구에서도 profitability factor의 abnormal return이 보고됐다. ([ScienceDirect][1])

사용할 feature는 세 개만 둔다.

$$
GP =
\frac{GrossProfit}{Assets}
$$

$$
ROE =
\frac{NetIncome}{Equity}
$$

$$
CFOA =
\frac{OperatingCashFlow}{Assets}
$$

각 값을 cross-sectional rank로 바꾼다.

```python
Q = mean(
    rank(gross_profitability),
    rank(roe),
    rank(cfo_to_assets),
)
```

---

# 15. Alpha 2 — Value

Fama-French 연구 등에서 value와 profitability는 대표적인 return characteristics이고, 한국 시장 factor 연구에서도 value premium이 관찰된다. ([ScienceDirect][4])

복잡하게 만들지 않는다.

```text
Book / Price
Earnings / Price
```

두 개면 일단 충분하다.

$$
V
=
\frac{
rank(B/P)+rank(E/P)
}{2}
$$

단 음수 earnings의 경우 E/P를 강제로 작은 값으로 만들지 않는다.

```text
E/P unavailable → 해당 feature neutral
```

로 처리한다.

---

# 16. Alpha 3 — Earnings Momentum

한국 시장에서는 이것을 상당히 중요하게 본다.

한국 시장 연구에서 earnings/revenue surprise가 price momentum의 일부를 설명한다는 결과가 있고, Post-Earnings Announcement Drift도 한국 시장에서 반복적으로 관찰됐다. ([Wiley Online Library][5])

Analyst consensus 데이터가 없더라도 구현 가능하다.

예를 들어:

$$
E_1 =
\frac{OI_q-OI_{q-4}}{Assets_{q-4}}
$$

$$
E_2 =
\frac{Sales_q-Sales_{q-4}}{Sales_{q-4}}
$$

$$
E_3 =
OperatingMargin_q
-
OperatingMargin_{q-4}
$$

그 후:

```python
E = mean(
    rank(operating_income_change),
    rank(sales_growth),
    rank(margin_change),
)
```

공시 이후 신호가 영원히 유지되지 않게 한다.

예:

```text
0~20 sessions  : 100%
21~40 sessions : 67%
41~60 sessions : 33%
>60 sessions   : 0
```

다만 이것도 v1에서는 더 단순하게:

```text
최근 60거래일 이내 공시
```

만 사용할 수도 있다.

---

# 17. Alpha 4 — Foreign Flow

한국 시장 특화 알파로 넣는다.

한국 시장의 오래된 연구에서 외국인이 매수한 종목이 매도한 종목보다 이후 수익과 영업성과가 더 높았고, 최근 연구에서도 외국인 trading/holdings가 향후 수익률과 관련된다는 결과가 계속 나오고 있다. ([KCI][6])

절대 순매수 금액을 쓰면 안 된다.

삼성전자 같은 종목이 항상 위로 올라온다.

따라서:

$$
F_5
=
\frac{
\sum_{i=0}^{4}ForeignNetBuy_{t-i}
}{
ADTV_{20}
}
$$

$$
F_{20}
=
\frac{
\sum_{i=0}^{19}ForeignNetBuy_{t-i}
}{
ADTV_{20}
}
$$

그리고:

$$
F
=
0.5 rank(F_5)
+
0.5 rank(F_{20})
$$

처음에는 50:50 고정한다.

튜닝하지 않는다.

---

# 18. 가격 Momentum을 뺀 이유

미국 등에서는 momentum evidence가 매우 강하고 고전적으로도 잘 알려져 있다. ([JSTOR][7])

하지만 한국은 그대로 적용하기 곤란하다.

한국 시장에서는 momentum premium을 발견하는 연구도 있는 반면, 1983~2023 장기 자료에서는 개별종목의 **reversal effect가 전체적으로 우세**했다는 2025년 연구가 있다. ([ScienceDirect][1])

그러므로:

```text
Price momentum = Challenger
```

이다.

Champion에 선험적으로 넣지 않는다.

---

# 19. Low Volatility도 Alpha에서 분리

한국 시장에서도 고위험주가 저위험주보다 낮은 수익을 보이는 low-volatility anomaly가 보고되었고, 특히 경기 하강기에서 강하다는 연구가 있다. ([한국학술지센터][8])

하지만 이걸 다섯 번째 factor로 넣으면:

```text
alpha
risk
portfolio sizing
```

역할이 뒤섞인다.

더 깔끔하게:

> **Low volatility는 position sizing에 사용한다.**

---

# 20. Feature 전처리

매우 중요하다.

모든 factor는 같은 처리 pipeline을 거친다.

```text
Raw
 ↓
Point-in-time filter
 ↓
Winsorization
 ↓
Sector normalization
 ↓
Cross-sectional percentile rank
 ↓
[-1, +1]
```

예:

$$
score(x_i)
=
2\times percentile(x_i)-1
$$

---

# 21. Sector Neutralization

단순 전체시장 rank만 하면:

```text
Value → 특정 산업
Profitability → 특정 산업
```

편향이 심해질 수 있다.

따라서 비교는 기본적으로:

```text
same sector
```

내에서 수행한다.

예:

```text
반도체 회사 ↔ 반도체 회사
화학 ↔ 화학
자동차 ↔ 자동차
```

그 뒤 sector별 rank를 합친다.

---

# 22. 최종 Alpha Score

가장 단순하게 간다.

$$
\boxed{
Alpha
=
0.25Q
+
0.25V
+
0.25E
+
0.25F
}
$$

Equal Weight다.

이유는 간단하다.

정확한 30/20/35/15 같은 비율을 찾기 시작하는 순간 overfit 자유도가 늘어난다.

v1에서 절대 최적화하지 않는다.

---

# 23. 결측값

복잡한 ML imputation을 하지 않는다.

### 최소 조건

```text
Q available
V available
E available
F available
```

4개가 모두 있어야 후보로 인정한다.

대신 universe coverage가 지나치게 줄어들면:

```text
3/4 factor requirement
missing factor = 0
```

을 Challenger로 비교한다.

처음부터 쓰지는 않는다.

---

# 24. Rebalance

추천:

$$
\boxed{5\ trading\ sessions}
$$

즉 대략 주 1회.

이유는:

* 재무정보는 매우 느림
* earnings도 quarterly
* foreign flow는 수일 단위
* 너무 빠른 turnover는 세금/슬리피지 증가
* 일봉 기반 알파와 자연스럽게 맞음

이다.

---

# 25. 종목 선택

v1에서는:

$$
\boxed{N=20}
$$

고정한다.

20은 최적의 숫자라고 주장하는 것이 아니다.

**연구 자유도를 제한하기 위한 설계값**이다.

1천만~수천만원 규모라면 충분히 실행 가능하고, 5종목보다 idiosyncratic risk가 훨씬 작다.

---

# 26. Hysteresis가 매우 중요

매주 Top 20을 새로 사면 turnover가 커진다.

다음 규칙을 사용한다.

### 신규진입

```text
rank <= 20
```

### 기존보유

```text
rank <= 40이면 유지 가능
```

구현:

```python
survivors = current_positions with rank <= 40

portfolio = survivors

while len(portfolio) < 20:
    add highest-ranked new stock
```

이렇게 하면 미세한 ranking 변화 때문에 사고파는 일이 크게 감소한다.

---

# 27. Weighting

Markowitz부터 하지 않는다.

v1:

$$
w_i^{raw}
=
\frac{1}{\sigma_{i,60}}
$$

즉 inverse volatility.

그 뒤:

```text
Single stock ≤ 7.5%
Sector ≤ 25%
Total exposure ≤ E_t
```

제약을 건다.

---

# 28. 왜 Markowitz를 안 쓰는가

$$
w=\Sigma^{-1}\mu
$$

형태의 mean-variance optimization은 매우 매력적으로 보이지만:

```text
expected return 추정오차
covariance 추정오차
parameter instability
corner solution
turnover
```

문제가 크다.

현재 목적에서는 unnecessary complexity다.

나중에 단순 inverse-vol보다 실제 OOS 개선이 확인될 때만 추가한다.

---

# 29. Risk Overlay는 하나만

Trend regime, breadth regime, HMM, recession classifier 등을 전부 넣지 않는다.

딱 하나:

> **Volatility scaling**

만 사용한다.

Moreira와 Muir는 시장·value·momentum·profitability 등 여러 factor에서 변동성이 높을 때 risk exposure를 줄이는 volatility-managed portfolio가 Sharpe를 개선할 수 있음을 보고했다. ([NBER][9])

---

# 30. Exposure

시장 realized volatility:

$$
\sigma_{mkt,t}
$$

를 계산하고:

$$
E_t
=
\min
\left(
1,
\frac{\sigma_{target}}
{\sigma_{mkt,t}}
\right)
$$

로 둔다.

예:

```text
risk target = 15% annualized
```

시장 volatility:

```text
10% → exposure 100%
15% → exposure 100%
20% → exposure 75%
30% → exposure 50%
```

Long-only이므로 leverage는 하지 않는다.

---

# 31. 이 구조의 장점

Bull/Bear 판단을 하지 않는다.

단순히:

> 위험할수록 돈을 덜 건다.

라는 구조다.

이론적으로도 훨씬 깨끗하다.

---

# 32. Drawdown 기반 매매 규칙은 v1에서 제외

예:

```text
MDD 10%면 50% 현금
MDD 20%면 전량 현금
```

같은 규칙은 직관적으로 좋아 보이지만 반등 직전에 노출을 끊는 문제가 있다.

따라서 v1에서는 쓰지 않는다.

Drawdown은:

```text
strategy stop / human review
```

용도로만 둔다.

---

# 33. 실행 시점

가장 중요한 look-ahead 방지 규칙이다.

```text
T 장 종료
 ↓
KRX 데이터 확정
 ↓
DART available data 반영
 ↓
18:00~20:00 signal 계산
 ↓
TargetPortfolio freeze
 ↓
T+1 execution
```

백테스트에서도 동일하다.

---

# 34. T 종가 체결은 금지

잘못된 백테스트:

```text
T close로 factor 계산
→
T close로 매수
```

미래정보 이용이다.

반드시:

```text
T data
→
T+1 fill
```

이어야 한다.

---

# 35. Backtest Engine

Event-driven 하나만 만든다.

```text
for session in calendar:

    1. settle cash

    2. apply corporate actions

    3. execute pending orders

    4. mark positions

    5. update ledger

    6. if decision_session:
           snapshot = data.asof(close)
           target = strategy.decide(snapshot)
           pending_orders = target - current
```

---

# 36. 동일한 Strategy를 Live에도 쓴다

```python
class Strategy:

    def decide(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> TargetPortfolio:
        ...
```

백테스트:

```python
target = strategy.decide(
    historical_snapshot,
    simulated_portfolio,
)
```

실전:

```python
target = strategy.decide(
    live_snapshot,
    real_portfolio,
)
```

똑같다.

---

# 37. Broker만 교체

```python
class Broker(Protocol):

    def submit(...): ...
    def fills(...): ...
```

구현체:

```text
BacktestBroker
PaperBroker
KISBroker
```

이다.

---

# 38. Ledger를 시스템의 최종 Truth로

전략 수익률을 별도 계산하지 않는다.

모든 성과는 Ledger에서 나온다.

```text
cash
unsettled_cash
positions
market_value

commission
tax
slippage

dividend
corporate_action

equity
```

를 기록한다.

$$
NAV_t
=
Cash_t
+
UnsettledCash_t
+
MarketValue_t
-
Costs_t
$$

이 NAV만 성과평가에 사용한다.

---

# 39. Fill model

이 부분에서 가짜 정밀도를 만들면 안 된다.

초기 모델:

$$
P_{fill}
=
P_{next\ open}
+
direction \times cost
$$

으로 하되 cost를 두 부분으로 분리한다.

$$
ExecutionCost
=
Spread
+
Impact
$$

Impact는 거래 규모가 커질수록 비선형적으로 증가하게 만든다.

실무/연구에서는 거래량 대비 주문규모와 변동성에 따른 비선형 market impact가 중요하므로, 단순히 모든 종목에 동일한 5bp를 사용하는 것보다 participation 기반 모델이 낫다. 거래비용이 factor 전략의 실현수익을 크게 바꿀 수 있다는 한국 시장 연구도 있다. ([MDPI][10])

---

# 40. Capacity

$$
Participation
=
\frac{OrderNotional}
{ADTV_{20}}
$$

그리고:

```text
base max participation = 0.25%
hard max              = 0.50%
```

정도로 제한한다.

개인 규모에서는 대부분 제약에 걸리지 않겠지만 백테스트가 큰 자본에서도 비현실적인 종목을 선택하지 못하게 한다.

---

# 41. Cost scenario는 최소 3개

```text
Ideal
Base
Stress
```

### Ideal

```text
실제 세금/수수료
market impact 없음
```

연구 진단용.

### Base

```text
실제 세금/수수료
estimated spread
estimated slippage
```

### Stress

```text
Base execution cost × 2
```

최종 PASS 판정은 Base와 Stress로 한다.

Ideal은 절대 promotion 판단에 사용하지 않는다.

---

# 42. Historical cost도 Point-in-Time

현재 2026년 세율을 과거 2016년에 적용하면 안 된다.

```text
effective_from
effective_to
market
commission
transaction_tax
other_tax
```

형태로 versioning한다.

---

# 43. Backtest 기간

가능하다면 최소:

$$
10+\ years
$$

을 목표로 한다.

중요한 것은 단순 년수가 아니라 서로 다른 regime다.

포함되어야 한다.

```text
급락
급반등
저금리
금리상승
대형주 장세
중소형주 장세
KOSDAQ 강세
KOSDAQ 약세
```

---

# 44. Walk-Forward

한 번 Train/Test split 하지 않는다.

예:

```text
2012~2016 → research
2017      → OOS

2013~2017 → research
2018      → OOS

2014~2018 → research
2019      → OOS
...
```

5년 → 다음 1년.

---

# 45. 하지만 v1에는 사실 학습이 거의 없다

이게 장점이다.

```text
factor definitions     fixed
weights                25/25/25/25
N                      20
rebalance              5 sessions
hysteresis             20/40
```

모두 사전에 고정한다.

따라서 Walk-Forward에서 model fitting을 거의 할 필요가 없다.

그만큼 과최적화 위험이 작다.

---

# 46. 모든 OOS를 이어 붙인다

```text
OOS 2017
+
OOS 2018
+
OOS 2019
+
...
```

하나의:

$$
Equity_{OOF}
$$

를 생성한다.

성과는 이것 하나에서 측정한다.

---

# 47. Historical holdout에 대한 중요한 원칙

이미 여러 차례 전략 개발에 사용한 과거 데이터는 사실상 완전한 holdout이 아니다.

White의 data-snooping 연구가 지적하듯 같은 데이터에서 반복적으로 모델/규칙을 찾아보면 우연히 좋은 결과를 선택할 위험이 커진다. ([DOI][11])

따라서:

> 진짜 최종 OOS는 향후 쌓이는 데이터다.

과거 데이터에서는 Walk-Forward robustness를 최대한 확인한다.

---

# 48. Bootstrap

Daily return IID bootstrap은 쓰지 않는다.

주식시장 수익률에는:

```text
volatility clustering
serial dependence
regime
```

가 있다.

따라서:

```text
Moving Block Bootstrap
```

또는

```text
Stationary Bootstrap
```

사용.

권장:

```text
block length = 20~60 trading days
resamples = 5,000+
```

---

# 49. 최종 평가 Metric

크게 5그룹만 관리하면 된다.

| 구분        | 지표                                               |
| --------- | ------------------------------------------------ |
| Growth    | CAGR, annualized log growth                      |
| Risk      | Volatility, MDD, Calmar, Sortino                 |
| Relative  | Excess CAGR, Information Ratio                   |
| Stability | yearly returns, rolling 1Y returns, fold returns |
| Execution | turnover, cost drag, fill ratio, capacity        |

Sharpe 하나로 결정하지 않는다.

---

# 50. Benchmark

최소 두 개.

### Benchmark 1

```text
Eligible Universe Cap Weighted
```

### Benchmark 2

```text
Eligible Universe Equal Weighted
```

왜 둘 다 필요한가?

Small/value factor를 쓰는 전략은 단순히 대형주 지수를 이겼다는 것만으로 alpha라고 할 수 없다.

Equal-weight universe도 이겨야 의미가 커진다.

---

# 51. PASS / FAIL Gate

이게 가장 중요하다.

전략 연구에 사람의 재량을 최대한 제거한다.

---

## Gate A — Data integrity

하나라도 있으면 즉시 FAIL.

```text
lookahead rows > 0
duplicate rows > 0
unknown corporate actions > 0
survivorship violation > 0
future filings used > 0
ledger mismatch > tolerance
```

---

# 52. Gate B — 순수 OOS 성과

초기 권장 기준:

```text
OOS Net CAGR          > benchmark + 3%p
OOS Sharpe            >= 0.8
OOS MDD               <= 25%
OOS Calmar            >= 0.5
```

절대 진리인 cutoff는 아니다.

실전 승격 기준을 미리 고정한다는 의미다.

---

# 53. Gate C — 연도별 안정성

```text
positive absolute return years
>= 70%

benchmark outperform years
>= 60%
```

그리고:

```text
한 해가 전체 compound alpha의
50% 이상을 설명
```

한다면 경고 또는 FAIL.

---

# 54. Gate D — Cost Stress

Stress에서:

```text
Net CAGR > 0
```

은 최소조건.

가능하다면:

```text
Stress CAGR > benchmark CAGR
```

까지 요구한다.

Base에서는 훌륭한데 cost를 2배로 하면 사라지는 전략은 실전 투입하지 않는다.

---

# 55. Gate E — Parameter Stability

예를 들어 공식 설정이:

```text
N = 20
rebalance = 5
```

라면 인접값:

```text
N:
15
20
25

rebalance:
4
5
10
```

도 확인한다.

PASS:

> 주변 설정에서도 기준 전략 성과의 약 70% 이상 유지.

정확히 20/5에서만 폭발적으로 좋다면 overfit 가능성이 높다.

---

# 56. Gate F — Factor ablation

반드시 한다.

전체:

```text
Q + V + E + F
```

에서 하나씩 제거.

```text
V + E + F
Q + E + F
Q + V + F
Q + V + E
```

를 테스트한다.

좋은 시스템이면:

> 하나를 제거했다고 전체 전략이 붕괴하지 않는다.

만약 Foreign Flow 하나를 빼면 수익률이 완전히 사라진다면:

```text
multi-factor strategy
```

가 아니라 사실상:

```text
foreign-flow strategy
```

다.

그 사실을 알고 운용해야 한다.

---

# 57. Factor 자체도 검증

포트폴리오 전에 각 factor마다:

```text
Spearman Rank IC
Top decile - Bottom decile
Quintile monotonicity
Turnover
Decay
```

를 본다.

하지만 IC가 PASS/FAIL의 최종 기준은 아니다.

Net portfolio return이 최종 기준이다.

---

# 58. Data Snooping 관리

모든 실험을 registry에 남긴다.

```text
experiment_id
timestamp
git_commit
dataset_id

hypothesis
parameters
result

accepted/rejected
```

절대:

```text
실패한 실험 삭제
```

하지 않는다.

100개 전략을 돌려놓고 마지막 최고 하나만 보면 결과가 크게 왜곡된다.

White's Reality Check 같은 multiple-testing 교정도 최종 후보군 평가에 적용할 수 있다. ([DOI][11])

---

# 59. 최종 v1 Specification

내가 실제 구현을 시작한다면 **이 설정 하나**를 baseline으로 고정한다.

```text
================================================
K-STOCK COMPOUNDING STRATEGY v1
================================================

ASSET
  KOSPI + KOSDAQ common stocks

EXCLUDE
  Financials
  REIT
  ETF / ETN
  SPAC
  Preferred
  Suspended / managed
  IPO age < 252 sessions

LIQUIDITY
  60d median trading value >= 20억원


ALPHA
  25% Profitability
  25% Value
  25% Earnings Momentum
  25% Foreign Flow


PROFITABILITY
  Gross Profit / Assets
  ROE
  CFO / Assets


VALUE
  Book / Price
  Earnings / Price


EARNINGS
  YoY operating profit change / assets
  YoY sales growth
  YoY operating margin change


FLOW
  Foreign net 5d / ADTV20
  Foreign net 20d / ADTV20


NORMALIZATION
  sector-relative percentile rank
  winsorized


REBALANCE
  every 5 trading sessions


SELECTION
  portfolio size = 20

  new entry:
      rank <= 20

  existing:
      keep while rank <= 40


WEIGHT
  inverse 60d volatility

  single stock <= 7.5%
  sector <= 25%


MARKET EXPOSURE
  volatility scaling

  target market vol = 15%
  exposure <= 100%

  no leverage
  no short


CAPACITY
  target order <= 0.25% ADTV20


SIGNAL
  T close


EXECUTION
  T+1


COST
  historical tax
  commission
  spread/slippage
  participation impact


BENCHMARK
  eligible universe cap-weight
  eligible universe equal-weight
================================================
```

이게 **Champion v1**이다.

---

# 60. 이 단계에서는 최적화하지 않을 것

특히 금지할 것은:

```text
factor weight optimization
N exhaustive search
rebalance exhaustive search
stop-loss optimization
take-profit
RSI
MACD
MA crossover
market timing grid
LightGBM
XGBoost
neural networks
PCA
HMM
Kelly per stock
```

이다.

왜냐하면 지금 필요한 것은:

> **잘 설명되는 baseline이 실제로 한국 시장에서 돈을 버는지 확인하는 것**

이기 때문이다.

---

# 61. Champion이 성공하면 Challenger 시작

그 다음에만 개선한다.

순서도 중요하다.

### Challenger A

Price reversal.

최근 장기 한국 자료에서 reversal evidence가 있기 때문이다. ([Tandfonline][12])

```text
Q + V + E + F
+
short-term reversal
```

---

### Challenger B

Price Momentum.

```text
6M residual momentum
12-1 residual momentum
```

을 따로 검증.

한국 evidence가 혼재하므로 반드시 OOS 승격 조건을 통과해야 한다.

---

### Challenger C

Short interest.

최근 한국 연구에서는 외국인 holdings/trading과 short interest를 함께 본 신호가 향후 횡단면 수익률을 예측한다는 결과도 있다. ([ScienceDirect][13])

---

### Challenger D

ML.

맨 마지막이다.

---

# 62. ML을 넣는다면 이것만

첫 모델:

```text
ElasticNet
```

그다음:

```text
LightGBM
```

까지만.

Input:

```text
Q components
V components
E components
F components
risk components
```

Target:

$$
R_{stock,T+1:T+21}
-
R_{sector,T+1:T+21}
-
cost
$$

즉:

```text
future 20d
sector-relative
net return
```

이다.

---

# 63. ML 승격 조건

질문을:

> ML 수익률이 좋은가?

로 하지 않는다.

다음이어야 한다.

$$
Performance(ML)
>
Performance(SimpleChampion)
$$

**동일한**

```text
data
universe
fold
cost
portfolio
execution
```

에서.

그리고 단순 Champion보다:

```text
OOS excess growth 증가
Stress 증가
MDD 악화 없음
turnover 증가 감안 후에도 증가
```

해야 한다.

아니면 ML을 버린다.

---

# 64. 실전 승격

백테스트 PASS 후에도 바로 실전 100% 투입하면 안 된다.

```text
Backtest PASS
     ↓
Shadow trading
     ↓
Paper
     ↓
10% capital
     ↓
25%
     ↓
50%
     ↓
100%
```

순서로 한다.

---

# 65. 실전에서 가장 중요한 지표

실전 시작 후 CAGR을 몇 달 만에 평가하는 것은 거의 의미 없다.

대신:

```text
signal parity
target parity

fill rate
slippage

cost
turnover

expected vs realized execution
ledger reconciliation
```

을 본다.

예:

```text
Backtest 예상 slippage = 11 bp
실제 평균              = 27 bp
```

라면 전략을 계속 늘릴 게 아니라 fill model부터 수정해야 한다.

---

# 66. 아키텍처에서 가장 핵심적인 부분

새 시스템은 다음 세 층만 엄격하게 분리하면 된다.

```text
┌──────────────────────────────┐
│          RESEARCH            │
│                              │
│ Is there an alpha?           │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│          STRATEGY            │
│                              │
│ What portfolio do I want?    │
└───────────────┬──────────────┘
                ↓
┌──────────────────────────────┐
│          EXECUTION           │
│                              │
│ What portfolio can I get?    │
└──────────────────────────────┘
```

이 세 질문이 코드에서도 섞이면 안 된다.

---

# 67. 이전 방식과 결정적으로 다른 점

이전 접근에서 흔히 생기는 구조는:

```text
feature 추가
↓
ML 개선
↓
IC 개선
↓
새 optimizer
↓
hedge
↓
복잡한 risk logic
↓
결국 NO_TRADE
```

이다.

새 방식:

```text
PIT DATA
↓
4개의 단순한 경제적 알파
↓
고정 score
↓
20종목
↓
inverse-vol
↓
vol scaling
↓
실제 execution
↓
NET OOS PnL
```

이다.

**설명할 수 없는 부분이 거의 없다.**

---

# 68. 내가 최종적으로 권하는 개발 순서

| Phase   | 구현                              | 목표          |
| ------- | ------------------------------- | ----------- |
| **P0**  | Domain + Ledger + clock         | 시스템 기초      |
| **P1**  | PIT KRX/DART dataset            | 백테스트 신뢰성    |
| **P2**  | Universe                        | 생존편향 제거     |
| **P3**  | Q/V/E/F feature                 | Alpha       |
| **P4**  | fixed scoring                   | 단순 Champion |
| **P5**  | 20-stock portfolio + hysteresis | turnover 관리 |
| **P6**  | inverse-vol + vol scaling       | risk        |
| **P7**  | unified event backtester        | 실제 Net PnL  |
| **P8**  | Walk-forward + bootstrap        | robustness  |
| **P9**  | stress/ablation/sensitivity     | PASS/FAIL   |
| **P10** | Paper/KIS adapter               | live        |
| **P11** | Challenger factors              | 개선          |
| **P12** | ML challenger                   | 필요할 경우만     |

---

# 69. 가장 중요한 세 가지 원칙

이 프로젝트를 앞으로 개발하면서 아래 세 가지는 바꾸지 않는 것이 좋다.

### 1.

**수익률 목표값을 정해놓고 전략을 맞추지 않는다.**

30% CAGR이 안 나왔다고 좋은 18% CAGR 전략을 버리면 안 된다.

---

### 2.

**복잡한 모델은 단순 Champion을 이긴 경우에만 존재할 수 있다.**

새 feature 하나도 예외가 아니다.

---

### 3.

**백테스트는 전략의 근거가 아니라 전략을 반증하기 위한 도구로 사용한다.**

질문을:

> 어떻게 하면 CAGR이 올라갈까?

보다

> 이 결과가 가짜일 수 있는 이유가 무엇인가?

로 두는 것이 맞다.

---

## 최종 판단

현재 목표에 가장 적합한 구조는 **“ML 기반 주가예측 시스템”이 아니다.**

내가 설계한다면 K-Stock Engine을 다음과 같이 정의한다.

> **Point-in-Time 한국시장 데이터를 바탕으로 Profitability, Value, Earnings underreaction, Foreign investor information 네 가지 독립적인 알파를 횡단면 ranking으로 결합하고, inverse-volatility allocation과 volatility scaling으로 위험을 제어하며, 동일한 event-driven engine에서 백테스트와 실전 체결을 수행하는 long-only compounding engine.**

한국 시장에서 profitability/value/low-risk factor의 존재를 뒷받침하는 연구가 있고, earnings drift와 외국인 trading의 예측력에 대한 별도의 한국시장 근거도 존재한다. 반대로 전통적 개별주 momentum은 한국에서 연구결과가 혼재하므로 처음부터 핵심 factor로 넣지 않는 것이 더 합리적이다. ([ScienceDirect][1])

그리고 **이 Champion v1조차 위의 Walk-Forward/비용 Stress/parameter stability/ablation gate를 통과하지 못한다면, 다음 행동은 ML 추가가 아니다.** Q/V/E/F 중 어떤 경제적 가설이 실제 한국 시장에서 실패하는지 분석해 전략 자체를 수정해야 한다.

이 구조라면 기존 프로젝트를 대규모로 갈아엎더라도 결과적으로 코드와 연구 과정은 지금보다 오히려 상당히 단순해진다. 동시에 백테스트 결과를 실전 성과로 연결하기 위한 논리적 연결고리는 훨씬 강해진다.

[1]: https://www.sciencedirect.com/science/article/pii/S0927538X21000652?utm_source=chatgpt.com "Enhanced factor investing in the Korean stock market - ScienceDirect"
[2]: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019019&utm_source=chatgpt.com "전자공시 OPENDART 시스템 | 개발가이드 | 상세"
[3]: https://www.nber.org/papers/w15940?utm_source=chatgpt.com "The Other Side of Value: Good Growth and the Gross Profitability Premium | NBER"
[4]: https://www.sciencedirect.com/science/article/pii/S0304405X14002323/pdf?utm_source=chatgpt.com "A five-factor asset pricing model - ScienceDirect"
[5]: https://onlinelibrary.wiley.com/doi/abs/10.1111/ajfs.12342?utm_source=chatgpt.com "Momentum and Earnings Information in the Korean Stock Market - Ha - 2021 - Asia-Pacific Journal of Financial Studies - Wiley Online Library"
[6]: https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART001546954&utm_source=chatgpt.com "Trading Behavior, Performance, and Stock Preference of Foreigners, Local Institutions, and Individual Investors: Evidence from the Korean Stock Market"
[7]: https://www.jstor.org/stable/i340162?utm_source=chatgpt.com "Vol. 48, No. 1, Mar., 1993 of The Journal of Finance | JSTOR"
[8]: https://journal.kci.go.kr/capm/archive/articleView?artiId=ART002163379&utm_source=chatgpt.com "Low Volatility Anomaly and Stock Returns (저위험 이례현상과 투자성과에 관한 연구)"
[9]: https://www.nber.org/papers/w22208?utm_source=chatgpt.com "Volatility Managed Portfolios | NBER"
[10]: https://www.mdpi.com/2071-1050/11/17/4797?utm_source=chatgpt.com "Is Factor Investing Sustainable after Price Impact Costs? The Capacity of Factor Investing in Korea"
[11]: https://doi.org/10.1111%2F1468-0262.00152?utm_source=chatgpt.com "A Reality Check for Data Snooping - White - 2000 - Econometrica - Wiley Online Library"
[12]: https://www.tandfonline.com/doi/abs/10.1080/10293523.2024.2448054?utm_source=chatgpt.com "Momentum and reversal effects in the Korean stock market: Investment Analysts Journal: Vol 54, No 4"
[13]: https://www.sciencedirect.com/science/article/abs/pii/S0927538X26000855?utm_source=chatgpt.com "Net arbitrage trading by foreign investors and short sellers and stock returns: Evidence from the Korean stock market - ScienceDirect"
