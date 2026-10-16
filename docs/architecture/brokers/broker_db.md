# DB Securities (DB금융투자) REST & WebSocket OpenAPI Specification

> **AI Agent Technical Reference:**  
> Machine-readable specification of DB Securities (DB금융투자 / DB증권) REST & WebSocket OpenAPI for quantitative factor research, high-capacity intraday bar collection, market microstructure flow analysis, MOC auction imbalance tracking, Nextrade ATS execution, and systematic order management.

---

## 1. System & Authentication Overview

### 1.1 Endpoints & Environments
DB Securities operates over standard HTTPS/WSS protocols on Linux 64-bit environments without ActiveX, COM, or 32-bit Windows dependencies.

* **REST Base URL (Production & Paper Trading):** `https://openapi.dbsec.co.kr:8443`
  * *Architecture Invariant:* Both production (`mode: "production"`) and demo/paper trading (`mode: "demo"`) share the identical REST base URL. The gateway demultiplexes routing internally based on the AppKey attribute (`prd_*` vs `vtl_*`).
* **OAuth Token URL:** `https://openapi.dbsec.co.kr:8443/oauth2/token`
* **OAuth Revoke URL:** `https://openapi.dbsec.co.kr:8443/oauth2/revoke`
* **WebSocket Live (Production - General):** `wss://openapi.dbsec.co.kr:7070/websocket`
* **WebSocket Live (Paper Trading - General):** `wss://openapi.dbsec.co.kr:17070/websocket`
* **WebSocket Live (Overseas Futures/Options):** `wss://openapi.dbsec.co.kr:7071/websocket` (Production only)

### 1.2 OAuth2 Token Issuance (`POST /oauth2/token`)
* **Headers:** `Content-Type: application/x-www-form-urlencoded`
* **Payload:**
  ```python
  payload = {
      "grant_type": "client_credentials",
      "appkey": "<DB_APP_KEY>",
      "appsecretkey": "<DB_APP_SECRET>",
      "scope": "oob",
  }
  ```
* **Response:**
  ```json
  {
      "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOi...",
      "token_type": "Bearer",
      "expires_in": 86400,
      "scope": "oob"
  }
  ```
* **Token Rate Limit & Lifecycle Invariants:**
  * **Strict 1 TPM Issuance Throttle:** The gateway limits token issuance to **1 request per minute (1 TPM)** per AppKey. Exceeding this triggers error HTTP 403 `{"error_code": "IGW00201", "error_description": "호출 거래건수를 초과하였습니다."}`.
  * **24-Hour Validity (86,400s):** Reissuing before expiration returns the identical cached token from DB server.
  * **Mandatory Host-Shared Caching:** All processes must coordinate token access via a host-level shared token cache (e.g. `SharedTokenStore` or local JSON cache). Stale or redundant in-flight token requests will cause severe pipeline cascading lockout due to the 1 TPM rate limit.

### 1.3 Common Request & Response Headers
* **Request Headers:**
  ```http
  content-type: application/json; charset=utf-8
  authorization: Bearer <access_token>
  cont_yn: <Y|N>
  cont_key: <CONTINUATION_KEY>
  ```
  *(Note: Unlike KIS or LS, DB Securities does NOT put `tr_cd` in the HTTP header; the TR identity is resolved from the REST URI path, and pagination cursors pass through `cont_yn` / `cont_key` headers).*
* **Response Headers:**
  * `cont_yn`: `"Y"` if more data pages remain; `"N"` when the final page is reached.
  * `cont_key`: Pagination cursor string passed verbatim into the subsequent request header `cont_key`.

### 1.4 Rate Limit, Concurrency & Single-Flight Invariants
* **App-Level Global Throttle:** **20 TPS** (20 requests/sec per AppKey across all endpoints).
* **Endpoint-Level Dedicated Rate Limits:**
  * `Trading (Orders)`: **10 TPS** (`/trading/kr-stock/order`, `/order-nxt`, etc.)
  * `Charts`: **4 TPS** (`/quote/kr-chart/*`)
  * `Current Price & Orderbook`: **3 ~ 5 TPS** (`/quote/kr-stock/inquiry/price`, `/orderbook`)
  * `Rankings & Ticker Master`: **3 TPS** (`/rank-list`, `/stock-ticker`)
  * `Account & Balances`: **2 TPS** (`/trading/kr-stock/inquiry/balance`, `/transaction-history`)
  * `Token Issuance`: **1 TPM** (`/oauth2/token`)
* **Throttling Detection:** HTTP 403 / 500 with `rsp_cd: "IGW00201"` (*"호출 거래건수를 초과하였습니다"*).
* **WebSocket Connection Invariants:**
  * Maximum 2 sessions per account.
  * Maximum 50 symbols subscribed per session.
  * Connection handshake limit: 6 TPM (max 6 connections per minute).
  * First message must be sent within 10 seconds of connection or the server disconnects.
  * 100ms packet pacing delay recommended between subscription bursts.

### 1.5 Venue Syntax & Nextrade (NXT) Support
DB Securities natively supports Nextrade (NXT) ATS alternative trading venue side-by-side with KRX across quotes, master dump, charts, and live order routing:

| Market Division Code (`InputCondMrktDivCode`) | Target Venue | Symbol Prefix Requirement | Example | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **`J`** | KRX Regular Exchange | Pure 6-digit code | `"005930"` | Includes Equities, ETFs |
| **`NJ`** | Nextrade (NXT) ATS | Prefix `"N-"` | `"N-005930"` | NXT dedicated orderbook/trades |
| **`UJ`** | Unified (KRX + NXT) | Prefix `"U-"` | `"U-005930"` | Consolidated volume & prices |
| **`E`** / **`EN`** | ETF / ETN | Pure 6-digit code | `"069500"` | Exchange Traded Funds/Notes |
| **`U`** | Industry / Index | 4-digit index code | `"1001"` | 1001: KOSPI, 2001: KOSDAQ |

---

## 2. Market Quotation TRs (`/api/v1/quote/kr-stock`)

### 2.1 `PRICE` — 주식 현재가 시세 조회 (Current Price Quote)
* **Path:** `POST /api/v1/quote/kr-stock/inquiry/price`
* **TPS:** 5
* **Request Body:**
  ```json
  {
    "In": {
      "InputCondMrktDivCode": "J",
      "InputIscd1": "005930"
    }
  }
  ```
* **Output (`Out` object):**
  * `Prpr`: 현재가 (KRW)
  * `PrdyVrss`: 전일대비
  * `PrdyCtrt`: 전일대비율 (%)
  * `AcmlVol`: 누적 거래량
  * `AcmlTrPbmn`: 누적 거래대금 (KRW)
  * `Oprc` / `Hprc` / `Lprc`: 당일 시가 / 고가 / 저가
  * `Mxpr` / `Llam`: 당일 상한가 / 하한가
  * `Askp1` / `Bidp1`: 최우선 매도 / 매수 호가
  * `Per` / `Pbr`: PER / PBR

### 2.2 `HOGA` — 주식 호가 조회 (Orderbook Snapshot)
* **Path:** `POST /api/v1/quote/kr-stock/inquiry/orderbook`
* **TPS:** 3
* **Request Body:** `{"In": {"InputCondMrktDivCode": "J", "InputIscd1": "005930"}}`
* **Output (`Out` object):**
  * 5-Level ask/bid price ladder: `Askp1` ~ `Askp5`, `Bidp1` ~ `Bidp5`.
  * 5-Level ask/bid quantity: `AskpRsqn1` ~ `AskpRsqn5`, `BidpRsqn1` ~ `BidpRsqn5`.
  * 5-Level residual delta: `AskpRsqnIcdc1` ~ `AskpRsqnIcdc5`, `BidpRsqnIcdc1` ~ `BidpRsqnIcdc5`.

### 2.3 `MULTIPRICE` — 국내주식 멀티 현재가 일괄 조회 (Multi-Stock Batch Price)
* **Path:** `POST /api/v1/quote/kr-stock/inquiry/multiprice`
* **TPS:** 2
* **Capacity:** **Up to 50 stocks per call**
* **Request Body:**
  ```json
  {
    "In": {
      "dataCnt": 3,
      "InputCondMrktDivCode1": "J",
      "InputIscd1": "005930",
      "InputCondMrktDivCode2": "J",
      "InputIscd2": "000660",
      "InputCondMrktDivCode3": "NJ",
      "InputIscd3": "N-005930"
    }
  }
  ```
* **Output (`Out` array):**
  * List of quote records containing `Iscd`, `KorIsnm`, `Prpr`, `PrdyVrss`, `PrdyCtrt`, `Oprc`, `Hprc`, `Lprc`, `Mxpr`, `Llam`.
* **Architecture Note:** Supports mixed multi-asset requests across KRX stocks, NXT stocks (`NJ`), index futures (`F`), and options (`O`) in a single payload.

### 2.4 `DAYTRADE` — 일별 체결 조회 & MOC 예상체결 캡처 (Daily Trades & Auction Match)
* **Path:** `POST /api/v1/quote/kr-stock/inquiry/daily-price`
* **TPS:** 3
* **Parameters (`In`):**
  * `InputCondMrktDivCode`: `"J"` (KRX) or `"NJ"` (NXT)
  * `InputIscd1`: Symbol code (e.g. `"005930"`)
  * `InputHourClsCode`: Time window selection code:
    * `"4"`: Regular continuous session (장중)
    * **`"8"`: Closing auction match price / Indicative auction (`예상` — MOC capture)**
    * `"1"`: After-hours single price (시간외단일가체결)
    * `"2"`: Pre-market + Regular + Post-market (장전+장중+장후)
    * `"5"`: All sessions including expected match (장전+장중+장후+장종료 예상포함)
* **Output (`Out` array):**
  * `Hour`: Timestamp (`HHMMSS`)
  * `Prpr`: Executed price or indicative auction match price
  * `PrdyVrssSign` / `PrdyVrss`: Sign and change vs previous day
  * `ShnuCnqn` / `SelnCnqn`: Buy / Sell execution volumes
* **Domain Significance:** **Crucial empirical research data source for KRX 15:20 ~ 15:30 MOC indicative match auction progression.**

### 2.5 `CONCLUSION` — 시간대별 체결 조회 (Intraday Time-Slice Executions)
* **Path:** `POST /api/v1/quote/kr-stock/inquiry/hour-price`
* **TPS:** 3
* **Output (`Out` array):** `Hour`, `Prpr`, `PrdyVrssSign`, `PrdyCtrt`, `CntgVol`.

### 2.6 `DAYSTOCKTJJ` — 일별 종목별 투자자 동향 (Investor Breakdown)
* **Path:** `POST /api/v1/quote/kr-stock/inquiry/daily-investor`
* **TPS:** 2
* **Parameters (`In`):** `InputCondMrktDivCode`, `InputIscd1`, `InputDate1` (`YYYYMMDD`), `InputDate2` (`YYYYMMDD`).
* **Output (`Out` array):**
  * `Date`: `YYYYMMDD`
  * `OrgnShnuVol` / `OrgnSelnVol`: Institutional gross buy / sell volume
  * `OrgnShnuTrPbmn` / `OrgnSelnTrPbmn`: Institutional buy / sell turnover (KRW)
  * `FrgnRegShnuVol` / `FrgnRegSelnVol`: Foreign gross buy / sell volume
  * `FrgnRegShnuTrPbmn` / `FrgnRegSelnTrPbmn`: Foreign buy / sell turnover (KRW)
  * `PrsnShnuVol` / `PrsnSelnVol`: Retail gross buy / sell volume
  * `PrsnShnuTrPbmn` / `PrsnSelnTrPbmn`: Retail buy / sell turnover (KRW)

---

## 3. Intraday & Historical Chart TRs (`/api/v1/quote/kr-chart`)

### 3.1 `CHARTMIN` — 주식 분차트 조회 (High-Capacity Minute Bar Chart)
* **Path:** `POST /api/v1/quote/kr-chart/min`
* **TPS:** 4
* **Capacity:** **Up to 2,000 bars per call** (Default: 400 bars)
* **Request Body:**
  ```json
  {
    "In": {
      "dataCnt": "2000",
      "InputCondMrktDivCode": "J",
      "InputIscd1": "005930",
      "InputDate1": "20261015",
      "InputDivXtick": "60",
      "InputOrgAdjPrc": "1"
    }
  }
  ```
* **Parameters:**
  * `dataCnt`: Requested bar count from `"1"` to `"2000"`.
  * `InputDivXtick`: Bar resolution in seconds (`60 * N`):
    * `"60"`: 1-minute bar
    * `"300"`: 5-minute bar
    * `"600"`: 10-minute bar
    * `"3600"`: 60-minute bar
  * `InputOrgAdjPrc`: `"0"` (Raw unadjusted), `"1"` (Corporate action-adjusted).
  * `InputDate1`: Anchor date (`YYYYMMDD`).
* **Pagination (`cont_yn`, `cont_key`):**
  * When `cont_yn == "Y"`, pass response header `cont_key` into subsequent request header `cont_key` to seamlessly walk backward in time across thousands of historical bars.
* **Output (`Out` array):**
  * `Date`: `YYYYMMDD`
  * `Hour`: `HHMMSS`
  * `Oprc`: Open price
  * `Hprc`: High price
  * `Lprc`: Low price
  * `Prpr`: Close price
  * `CntgVol`: Bar volume
* **High-Leverage Advantage:** A single request captures **2,000 bars** (~5 full regular sessions of 390 bars each). Highest single-call minute bar capacity among all 5 brokers (surpassing Kiwoom 900, LS 500, Toss 200, KIS 30/120).

### 3.2 `CHARTTICK` — 주식 틱차트 조회 (Tick Bar Chart)
* **Path:** `POST /api/v1/quote/kr-chart/tick`
* **TPS:** 4
* **Capacity:** Up to 2,000 ticks per call (`dataCnt: "2000"`).
* **Parameters:** `InputDivXtick` (e.g. `"100"` for 100-tick bars), `InputDate1`, `InputOrgAdjPrc`.

### 3.3 `CHARTDAY` / `WEEK` / `MONTH` — 주기별 차트 (Daily / Weekly / Monthly Chart)
* **Path:** `POST /api/v1/quote/kr-chart/day`
* **TPS:** 4
* **Parameters:** `InputDate1` (Start `YYYYMMDD`), `InputDate2` (End `YYYYMMDD`), `InputOrgAdjPrc` (`"0"` or `"1"`).
* **Venue Distinction:** When querying with `InputCondMrktDivCode: "UJ"`, `Prpr` represents the Nextrade (NXT) closing price if NXT session extends past KRX close.

---

## 4. Universe Discovery & Screening TRs

### 4.1 `JCODES` — 전 종목 마스터 조회 (1-Call Full Universe Dump)
* **Path:** `POST /api/v1/quote/kr-stock/inquiry/stock-ticker`
* **TPS:** 3
* **Request Body:**
  ```json
  {
    "In": {
      "InputCondMrktDivCode": "J"
    }
  }
  ```
  *(Options: `"J"` for KRX stocks, `"NJ"` for NXT stocks, `"UJ"` for Unified, `"E"` for ETF, `"EN"` for ETN).*
* **Output (`Out` array — up to 3,940+ rows in 1 single call without paging!):**
  * `Iscd`: 6-digit short ticker (단축종목코드, e.g. `"005930"`)
  * `StndIscd`: 12-digit standard ISIN code (표준종목코드, e.g. `"KR7005930003"`)
  * `KorIsnm`: Korean asset name (한글종목명, e.g. `"삼성전자"`)
  * `MrktClsCode`: Market division code (`1`: KOSDAQ, `4`: KOSPI)
  * `MrktClsName`: Exchange name (`"거래소"`, `"코스닥"`, `"코넥스"`, `"ETF"`, `"ETN"`)
* **Architecture Role:** **Top-tier vendor for complete daily universe master sync alongside Toss Securities.** Retrieves active exchange tickers and ISIN mappings in a single sub-second call.

### 4.2 `RANKLIST` — 주식 조건 상승/하락 순위 (Price Fluctuation Ranking)
* **Path:** `POST /api/v1/quote/kr-stock/inquiry/rank-list`
* **TPS:** 3
* **Capacity:** **400 stocks per call**
* **Request Body:**
  ```json
  {
    "In": {
      "InputDateClsCode": "0",
      "InputRankSortClsCode1": "12",
      "InputMrktClsCode": "A",
      "InputBstpIscd": ""
    }
  }
  ```
* **Parameters:**
  * `InputDateClsCode`: `"0"` (Today), `"1"` (Yesterday), `"2"` (Weekly), `"5"` (Monthly)
  * `InputRankSortClsCode1`: `"12"` (Top Gainers), `"11"` (Top Losers)
  * `InputMrktClsCode`: `"A"` (All markets), `"K"` (KOSPI), `"Q"` (KOSDAQ)
* **Output (`Out` array — 400 stocks):**
  * `DataRank`: Ranking position (1 ~ 400)
  * `Iscd`: Symbol code
  * `KorIsnm`: Symbol name
  * `Prpr`: Current price
  * `PrdyVrss`: Change vs previous day
  * `PrdyCtrt`: Fluctuation percentage (%)

---

## 5. Order Execution & Account Trading TRs (`/api/v1/trading/kr-stock`)

### 5.1 `CSPAT00600` — 주식종합주문 (KRX Equities Cash & Margin Order)
* **Path:** `POST /api/v1/trading/kr-stock/order`
* **TPS:** 10
* **Request Body:**
  ```json
  {
    "In": {
      "IsuNo": "005930",
      "OrdQty": 10,
      "OrdPrc": 0,
      "BnsTpCode": "2",
      "OrdprcPtnCode": "03",
      "MgntrnCode": "000",
      "LoanDt": "00000000",
      "OrdCndiTpCode": "0",
      "TrchNo": 1
    }
  }
  ```
* **Parameters:**
  * `IsuNo`: 6-digit symbol or `"A"` + 6-digit symbol (ETN: `"Q"`, ELW: `"J"`).
  * `OrdQty`: Integer order quantity.
  * `OrdPrc`: Order price (must be `0` for market/auction orders).
  * `BnsTpCode`: `"1"` (Sell / 매도), `"2"` (Buy / 매수).
  * `OrdprcPtnCode` (Order Type):
    * `"00"`: Limit (지정가)
    * `"03"`: Market (시장가)
    * `"05"`: Conditional Limit (조건부지정가)
    * `"06"`: Best Limit (최유리지정가)
    * `"07"`: First Priority Limit (최우선지정가)
    * `"14"`: Midpoint Order (중간가호가)
    * `"61"`: Pre-Market Off-Hours (장개시전시간외)
    * `"81"`: Post-Market Closing Price (시간외종가)
    * `"82"`: Post-Market Single Price (시간외단일가)
  * `MgntrnCode`: `"000"` (Cash / 일반현금주문).
  * `LoanDt`: `"00000000"` (Cash).
  * `OrdCndiTpCode`: `"0"` (None), `"1"` (IOC), `"2"` (FOK).
  * `TrchNo`: `1` (KRX regular execution trench).
* **Output (`Out` object):**
  * `OrdNo`: Broker-assigned order ID (주문번호)
  * `OrdTime`: Execution timestamp (`HHMMSSSSS`)
  * `ShtnIsuNo`: Short ticker
  * `SpotOrdQty`: Executed order quantity
  * `MnyOrdAmt`: Cash value
  * `OrdSplitYn`, `ConnOrdNo`, `NxtOrdNo`: SOR (Smart Order Routing) split identifiers

### 5.2 `CSPAT00610` — 주식종합주문 NXT 전용 (Nextrade Dedicated Order)
* **Path:** `POST /api/v1/trading/kr-stock/order-nxt`
* **TPS:** 10
* **Paper Trading:** ❌ Production Only
* **Request Body:** Same schema as `CSPAT00600`, but routes directly into Nextrade matching engine without `TrchNo`.
* **Prerequisite:** Account must have registered Best Execution Duty (최선집행의무) agreement with DB Securities.

### 5.3 `CSPAT00700` & `00800` — 주식 정정 / 취소 주문 (Amend & Cancel)
* **Modify Path:** `POST /api/v1/trading/kr-stock/order-modify` (`CSPAT00700`)
* **Cancel Path:** `POST /api/v1/trading/kr-stock/order-cancel` (`CSPAT00800`)
* **NXT Modify / Cancel Paths:** `/order-modify-nxt` (`CSPAT00710`), `/order-cancel-nxt` (`CSPAT00810`)
* **TPS:** 3

### 5.4 `CSPAQ03420` — 주식 잔고 조회 (Account Holdings & Equity)
* **Path:** `POST /api/v1/trading/kr-stock/inquiry/balance`
* **TPS:** 2
* **Account Resolution:** Implicit server-side mapping from AppKey (No account number needed in request payload).
* **Request Body:** `{"In": {"QryTpCode0": "0"}}` (`"0"`: All, `"1"`: Exclude unlisted, `"2"`: Exclude unlisted/KONEX/K-OTC).
* **Output:**
  * `Out` (Summary): `TotBuyAmt`, `TotEvalAmt`, `TotEvalPnlAmt`, `TotErnrat`, `Dps2` (D+2 cash deposit), `DpsastAmt` (Total asset value).
  * `Out1` (Positions array): `IsuNo`, `IsuNm`, `BalQty0` (T+2 executed balance), `AbleQty` (Available sell quantity), `ExecPrc` (Average entry price), `NowPrc` (Current price), `EvalPnlAmt`, `Ernrat`.

### 5.5 `CSPAQ04800` — 체결 / 미체결 내역 조회 (Execution & Unfilled Orders)
* **Path:** `POST /api/v1/trading/kr-stock/inquiry/transaction-history`
* **TPS:** 2
* **Request Body:**
  ```json
  {
    "In": {
      "ExecYn": "0",
      "BnsTpCode": "0",
      "IsuTpCode": "0",
      "QryTp": "0",
      "TrdMktCode": "0",
      "SorTpYn": "2"
    }
  }
  ```
  *(Options: `ExecYn`: `"0"` All, `"1"` Filled, `"2"` Unfilled/Cancelable; `TrdMktCode`: `"0"` All, `"1"` KRX, `"2"` NXT).*
* **Output (`Out1` array):** `OrdNo`, `OrgOrdNo`, `TrdMktCode`, `IsuNo`, `BnsTpCode`, `OrdQty`, `OrdPrc`, `AllExecQty`, `MrcAbleQty` (Amend/cancelable quantity), `AvrExecPrc`, `OrdTrxPtnCode` (`0`: Normal, `6`: Amended, `8`: Canceled).

---

## 6. Real-Time Streaming WebSocket (WSS)

### 6.1 WebSocket Session Protocol
* **Production Port:** `wss://openapi.dbsec.co.kr:7070/websocket`
* **Demo Port:** `wss://openapi.dbsec.co.kr:17070/websocket`
* **Subscription Handshake:**
  ```json
  {
    "header": {
      "token": "<access_token>",
      "tr_type": "1"
    },
    "body": {
      "tr_cd": "S01",
      "tr_key": "005930"
    }
  }
  ```
* **Protocol Control (`header.tr_type`):**
  * `"1"`: Subscribe real-time market data feed (`S00`, `S01`, `U00` 등).
  * `"2"`: Unsubscribe real-time market data feed.
  * `"3"`: Register account-level notification stream (`IS0`, `IS1`). Does not require unsubscription message; connection termination cleans up the registration. `tr_key` must be `<DB_APP_KEY>`.

### 6.2 `S01` — 실시간 주식 호가 (Real-time L10 Orderbook & Indicative Auction)
* **TR Code:** `S01`
* **Payload Fields:**
  * `BsopHour`: Orderbook timestamp (`HHMMSS`)
  * `ShrnIscd`: 6-digit stock code
  * `HourClscode`: Time division code
  * **Level-10 Ladder:**
    * `askp1` ~ `askp10`, `bidp1` ~ `bidp10`
    * `AskpRsqn1` ~ `AskpRsqn10`, `BidpRsqn1` ~ `BidpRsqn10`
    * `AskpRsqnicdc1` ~ `10`, `BidpRsqnicdc1` ~ `10` (Level depth delta)
  * `TotalAskprsqn` / `TotalBidprsqn`: Total aggregated ask/bid quantity
  * `TotalAskprsqnicdc` / `TotalBidprsqnicdc`: Aggregated ask/bid quantity delta
  * `OvtmTotalaskprsqn` / `OvtmTotalbidprsqn`: Off-hours total ask/bid depth
  * **Indicative MOC Auction Metrics:**
    * **`AntcCnpr`:** Indicative auction match price (예상체결가)
    * **`AntcCnqn`:** Indicative auction match quantity (예상체결량)
    * **`CnccAsprclscode`:** Auction division code (동시호가구분코드)
    * **`CompClscode`:** Competitive auction direction code (경쟁대량 방향구분)
* **Domain Significance:** Provides real-time streaming tick-by-tick orderbook updates with high-resolution closing auction indicative match price and quantity (`AntcCnpr` / `AntcCnqn`).

### 6.3 `S00` — 실시간 주식 체결가 (Real-time Trade Ticks)
* **TR Code:** `S00`
* **Payload Fields:**
  * `BsopDate`: Trade date (`YYYYMMDD`)
  * `ShrnIscd`: Stock code
  * `StckCntghour`: Execution time (`HHMMSS`)
  * `StckPrpr`: Executed trade price
  * `CntgVol`: Trade tick size (volume)
  * `AcmlVol`: Cumulative session volume
  * `AcmlTrpbmn`: Cumulative turnover (KRW)
  * `CntgClscode`: Execution side (1: Buy aggressor, 5: Sell aggressor)
  * `AntcNmixclscode`: Indicative match flag (`0`: Indicative auction tick, `1`: Continuous regular trade tick)
  * `NewMkopclscode`: Market operation division code

### 6.4 `IS0` & `IS1` — 실시간 주문 접수 / 체결 통보 (Account Notification Streams)
* **Registration:** `header.tr_type: "3"`, `body.tr_key: "<DB_APP_KEY>"`
* `IS0`: Live order acceptance event (접수 확인).
* `IS1`: Live order fill / execution event (체결 내역 통보).

---

## 7. Error Code Catalog & Operational Guardrails

### 7.1 Platform Gateway Error Codes (`rsp_cd`)

| HTTP Code | Error Code (`rsp_cd`) | Message / Meaning | Root Cause & Operational Remediation |
| :---: | :---: | :--- | :--- |
| **403** | `IGW00201` | **호출 거래건수를 초과하였습니다.** | Global 20 TPS or **Token 1 TPM** rate limit exceeded. Must back off using exponential backoff or reuse cached token. |
| **403** | `IGW00103` | 유효하지 않은 AppKey입니다. | Invalid `DB_APP_KEY`. Verify `~/.quant.env`. |
| **403** | `IGW00105` | 유효하지 않은 AppSecret입니다. | Invalid `DB_APP_SECRET`. |
| **403** | `IGW00121` | 유효하지 않은 token 입니다. | Stale token. Refresh via CAS token cache. |
| **403** | `IGW00123` | 기간이 만료된 token 입니다. | Token expired. Issue new token and update shared cache. |
| **403** | `IGW40342` | API 사용 권한이 없습니다. | Open API service agreement required on DB Securities portal. |
| **400** | `IGW40010` | 연속구분 (cont)은 필수입니다. | Missing `cont_yn` request header. |
| **400** | `IGW40011` | 단축코드 길이/타입 확인 필요. | Symbol length mismatch (e.g. not 6 digits). |
| **500** | `IGW40017` | datacnt 값과 요청 개수 불일치. | `MULTIPRICE` `dataCnt` parameter does not match actual count of symbols. |

### 7.2 Service Core Message Codes (`rsp_cd`)

| Code | Message | Description |
| :---: | :--- | :--- |
| `00000` | 정상 처리 되었습니다. / 조회가 완료되었습니다. | Successful execution. |
| `1001` | 호가유형 등을 잘못 입력하셨습니다. | Invalid order parameter or price tick size. |
| `1584` | 매도잔고가 부족합니다. | Insufficient balance to cover short sell. |
| `2257` | 원주문가와 정정주문가가 동일합니다. | Amend price identical to original order price. |
| `2611` | 장시작 전 또는 장마감 되었습니다. | Session closed for requested order type. |
| `2714` | 주문수량이 매매가능수량을 초과했습니다. | Order quantity exceeds available buying/selling power. |
| `2752` | 증거금부족으로 주문이 불가합니다. | Margin shortage. |
| `3563` | 정규매매장이 종료되었습니다. 시간외 호가구분을 입력하십시오. | Regular session closed. Switch to off-hours order code. |
| `3735` | 해당종목 NXT거래 불가합니다. | Target asset is not eligible for Nextrade ATS trading. |

---

## 8. Integration & Architectural Role in K-Stock Engine

DB Securities occupies a high-value niche within the multi-broker execution and quantitative research stack:

1. **Ultra-High Minute Bar Density:**  
   Delivers **2,000 bars per call** via `CHARTMIN` (`/quote/kr-chart/min`). A single call ingests up to ~5 full days of 1-minute regular session bars, minimizing network roundtrips and API quota consumption during historical backfill and closing bar collection (`DbsecClosingBarJob`).
2. **Native Dual-Venue (KRX + NXT) Execution:**  
   Provides dedicated endpoints for Nextrade (`/order-nxt`, `CSPAT00610`), unified combined quotes (`UJ`), and symbol master lookup (`NJ`), allowing seamless routing between primary and alternative venues.
3. **MOC Indicative Auction Telemetry:**  
   Captures granular closing auction imbalance and match evolution through:
   * REST: `DAYTRADE` (`/quote/kr-stock/inquiry/daily-price`) with `InputHourClsCode="8"` for historical second-by-second indicative match series.
   * WebSocket: `S01` with `AntcCnpr` (indicative price) and `AntcCnqn` (indicative quantity) streaming.
4. **1-Call Universe Master Synchronizer:**  
   Retrieves all active listed symbols (3,940+ assets) alongside official ISIN codes (`StndIscd`) in a single call via `JCODES` (`/quote/kr-stock/inquiry/stock-ticker`), powering `DbsecTickerMasterJob`.
5. **Fallback Daily Chart Harvesting:**  
   Captures unadjusted/adjusted daily OHLCV series via `CHARTDAY` (`/quote/kr-chart/day`), powering `DbsecDailyChartJob`.
