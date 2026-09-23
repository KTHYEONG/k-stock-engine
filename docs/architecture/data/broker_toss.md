# Toss Securities REST & WebSocket OpenAPI Specification

> **AI Agent Technical Reference:**  
> Machine-readable specification of Toss Securities (토스증권) REST & WebSocket OpenAPI for quantitative research, high-throughput multi-stock quotation, intraday candle extraction, market microstructure flow analysis, ranking, account/asset management, and automated order execution with server-side conditional triggers.

---

## 1. System & Authentication Overview

### 1.1 Endpoints & Environments
* **REST Base URL:** `https://openapi.tossinvest.com`
* **WebSocket Live URL:** `wss://openapi-ws.tossinvest.com/ws/v1`
* **OAuth Token URL:** `https://openapi.tossinvest.com/oauth2/token`
* **Platform Invariant:** Native 64-bit Linux HTTPS/WSS service. Operates via standard OAuth2 Client Credentials and REST/JSON envelopes, with zero Windows COM/ActiveX or binary DLL dependencies.

### 1.2 OAuth2 Token Issuance (`POST /oauth2/token`)
* **Endpoint:** `POST /oauth2/token`
* **Headers:** `Content-Type: application/x-www-form-urlencoded`
* **Request Payload (Form-Encoded):**
  ```python
  payload = {
      "grant_type": "client_credentials",
      "client_id": "<TOSS_APP_KEY>",
      "client_secret": "<TOSS_APP_SECRET>",
  }
  ```
* **Response Payload (Standard OAuth2 JSON):**
  ```json
  {
    "access_token": "eyJraWQiOiIyM...",
    "token_type": "Bearer",
    "expires_in": 86400
  }
  ```
* **Token Lifecycle & Invariants:**
  * Validity is 86,400 seconds (24 hours).
  * Standard JWT RS256 token containing client claims.
  * Token fetch should be guarded by an `asyncio.Lock()` to prevent redundant issuance across concurrent tasks.
  * Unlike domestic legacy brokers, token responses use standard OAuth2 schema without custom broker envelopes.

### 1.3 Common Request & Response Headers
* **Standard Headers (All Endpoints):**
  ```http
  Authorization: Bearer <access_token>
  Content-Type: application/json
  Accept-Encoding: gzip
  ```
* **Account-Scoped Endpoints (`Account`, `Asset`, `Order`, `Conditional Order`):**
  ```http
  X-Tossinvest-Account: <accountSeq>
  ```
  *(Omission triggers HTTP 400 with `code: "account-header-required"`).*
* **Tracing & Rate Limit Response Headers:**
  ```http
  x-request-id: <unique_request_identifier>
  X-RateLimit-Limit: <max_allowed_tps_for_group>
  X-RateLimit-Remaining: <remaining_calls_in_current_second>
  X-RateLimit-Reset: <seconds_until_limit_window_resets>
  Retry-After: <seconds_to_wait_on_429>
  ```

### 1.4 Rate Limit Matrix by Group
Toss Securities enforces strict token-bucket rate limits segmented by API group (`client_id × Group`). Exceeding limits returns `HTTP 429 Too Many Requests`.

| Rate Limits Group | General Limit | Peak Window Limit (09:00~09:10 KST) | Target Endpoints |
| :--- | :--- | :--- | :--- |
| `AUTH` | **5 req/s** | - | `POST /oauth2/token` |
| `ACCOUNT` | **1 req/s** | - | `GET /api/v1/accounts` |
| `ASSET` | **5 req/s** | - | `GET /api/v1/holdings` |
| `STOCK` | **5 req/s** | - | `GET /api/v1/stocks`, `GET /api/v1/stocks/{symbol}/warnings` |
| `STOCK_ALL` | **1 req/s** | - | `GET /api/v1/stocks/all` |
| `STOCK_TRADING_TREND`| **10 req/s** | - | Investor flows, Program trades, Short selling, Credit, Lending |
| `MARKET_INFO` | **3 req/s** | - | `GET /api/v1/exchange-rate`, `GET /api/v1/market-calendar/*` |
| `MARKET_DATA` | **15 req/s** | - | `GET /api/v1/prices`, `/orderbook`, `/trades`, `/price-limits` |
| `MARKET_DATA_CHART`| **20 req/s** | - | `GET /api/v1/candles` |
| `RANKING` | **5 req/s** | - | `GET /api/v1/rankings` |
| `MARKET_INDICATOR` | **10 req/s** | - | Index / Bond prices, indicators |
| `MARKET_INDICATOR_CHART` | **5 req/s** | - | Index candles |
| `ORDER` | **10 req/s** | 10 req/s | `POST /api/v1/orders`, `modify`, `cancel` |
| `ORDER_HISTORY` | **5 req/s** | - | `GET /api/v1/orders`, `GET /api/v1/orders/{orderId}` |
| `ORDER_INFO` | **6 req/s** | **3 req/s** | `GET /api/v1/buying-power`, `/sellable-quantity`, `/commissions` |
| `CONDITIONAL_ORDER` | **5 req/s** | - | `POST /api/v1/conditional-orders`, `modify`, `cancel` |
| `CONDITIONAL_ORDER_HISTORY` | **10 req/s** | - | `GET /api/v1/conditional-orders`, detail |

### 1.5 Protocol Invariants & Envelope Models
1. **Success Envelope (`HTTP 200`):**
   ```json
   {
     "result": { ... }
   }
   ```
2. **Error Envelope (`HTTP 4xx / 5xx`):**
   ```json
   {
     "error": {
       "requestId": "bFXoGdVYF10WCNOJ",
       "code": "invalid-request",
       "message": "상세 오류 원인 설명",
       "data": { ... }
     }
   }
   ```
3. **Decimal Precision:** All numeric amounts, prices, and quantities are serialized as string decimals (e.g. `"262000"`, `"0.2998"`), eliminating floating-point rounding errors.
4. **Timestamps:** ISO 8601 with KST timezone offset (`YYYY-MM-DDTHH:MM:SS.sss+09:00`).
5. **IP Whitelisting Requirement:** API access is locked to IP addresses registered in the Toss Securities WTS settings (`설정 > Open API > 허용 IP 관리`). Calls from unregistered IPs return `HTTP 403 Forbidden`.

---

## 2. Market Quotation & Orderbook TRs (`/api/v1/...`)

### 2.1 `GET /api/v1/prices` — 주식 현재가 일괄 조회 (Multi-Stock Price Quote)
* **Path:** `GET /api/v1/prices`
* **Rate Limits Group:** `MARKET_DATA` (15 req/s)
* **Query Parameters:**
  * `symbols` (string, required): Comma-separated tickers (up to **200 symbols** in a single call, e.g. `005930,000660,035420`).
* **Response Payload (`result` list):**
  ```json
  {
    "result": [
      {
        "symbol": "005930",
        "timestamp": "2026-09-11T19:59:59.000+09:00",
        "lastPrice": "262000",
        "currency": "KRW"
      }
    ]
  }
  ```
* **Quant Leverage:** At 200 symbols per request and 15 req/s capacity, Toss can refresh prices for **3,000 symbols/sec**, outperforming legacy brokers by an order of magnitude.

### 2.2 `GET /api/v1/orderbook` — 주식 현재가 호가 (Orderbook Snapshot)
* **Path:** `GET /api/v1/orderbook`
* **Rate Limits Group:** `MARKET_DATA` (15 req/s)
* **Query Parameters:**
  * `symbol` (string, required): 6-digit stock code (e.g. `005930`).
* **Response Payload (`result`):**
  ```json
  {
    "result": {
      "timestamp": "2026-09-11T20:00:00.000+09:00",
      "currency": "KRW",
      "asks": [
        {"price": "262000", "volume": "89385"},
        {"price": "262500", "volume": "78596"}
      ],
      "bids": [
        {"price": "261500", "volume": "34502"},
        {"price": "261000", "volume": "41200"}
      ]
    }
  }
  ```
* **Ladder Ordering:** `asks` are sorted low-to-high; `bids` are sorted high-to-low (up to 10 levels each).

### 2.3 `GET /api/v1/trades` — 최근 체결 내역 (Recent Trade Ticks)
* **Path:** `GET /api/v1/trades`
* **Rate Limits Group:** `MARKET_DATA` (15 req/s)
* **Query Parameters:**
  * `symbol` (string, required): 6-digit stock code.
  * `count` (integer, optional): Maximum ticks to return (max 50, default 50).
* **Response Payload (`result` list):**
  ```json
  {
    "result": [
      {
        "price": "262000",
        "volume": "1",
        "timestamp": "2026-09-11T19:59:59.000+09:00",
        "currency": "KRW"
      }
    ]
  }
  ```

### 2.4 `GET /api/v1/price-limits` — 상한가 / 하한가 / 기준가 (Price Limits)
* **Path:** `GET /api/v1/price-limits`
* **Rate Limits Group:** `MARKET_DATA` (15 req/s)
* **Query Parameters:** `symbol`
* **Response Payload (`result`):**
  ```json
  {
    "result": {
      "symbol": "005930",
      "basePrice": "269000",
      "upperLimitPrice": "349500",
      "lowerLimitPrice": "188500",
      "currency": "KRW"
    }
  }
  ```

---

## 3. Candles & Historical Data TRs (`/api/v1/candles`)

### 3.1 `GET /api/v1/candles` — 캔들 차트 조회 (1m / 1d Candle Bars)
* **Path:** `GET /api/v1/candles`
* **Rate Limits Group:** `MARKET_DATA_CHART` (**20 req/s**)
* **Query Parameters:**
  * `symbol` (string, required): 6-digit KR code or US ticker.
  * `interval` (string, required): Bar resolution.
    * `"1m"`: 1-minute bars
    * `"1d"`: Daily bars
  * `count` (integer, optional): Number of bars (max 200, default 100).
  * `before` (string, optional): Inclusive upper timestamp boundary in ISO 8601 format (e.g. `2026-09-11T15:30:00.000+09:00`) for backward cursor pagination.
  * `adjusted` (boolean, optional): Whether to apply corporate-action adjusted prices (`true` / `false`, default `true`).
* **Response Payload (`result`):**
  ```json
  {
    "result": {
      "candles": [
        {
          "timestamp": "2026-09-11T15:30:00.000+09:00",
          "openPrice": "261500",
          "highPrice": "262000",
          "lowPrice": "261500",
          "closePrice": "262000",
          "volume": "48391",
          "currency": "KRW"
        }
      ]
    }
  }
  ```
* **Intraday Session Paging:**
  * A full 390-minute KRX regular session requires only 2 consecutive requests (`count=200` each, using the oldest candle timestamp as `before`).
  * At 20 req/s, extracting full regular-session minute bars for 100 stocks completes in ~10 seconds.

---

## 4. Market Microstructure & Flow TRs (수급 / 프로그램 / 공매도 / 신용 / 대차)

All flow TRs belong to the `STOCK_TRADING_TREND` Rate Limits Group (**10 req/s**) and support backwards pagination via `count` (up to 100) and `until` (`YYYY-MM-DD`).

### 4.1 `GET /api/v1/stocks/{symbol}/investor-trading` — 투자자별 매매동향
* **Path:** `GET /api/v1/stocks/{symbol}/investor-trading`
* **Query Parameters:** `count` (max 100), `until` (`YYYY-MM-DD`, optional)
* **Response Payload (`result`):**
  ```json
  {
    "result": {
      "nextUntil": "2026-09-09",
      "records": [
        {
          "date": "2026-09-11",
          "updatedAt": "2026-09-11T20:21:29.000+09:00",
          "individual": {
            "buyVolume": "12545529",
            "sellVolume": "7767379",
            "netBuyVolume": "4778150"
          },
          "foreigner": {
            "buyVolume": "2590859",
            "sellVolume": "6560314",
            "netBuyVolume": "-3969455"
          },
          "institution": {
            "buyVolume": "4593167",
            "sellVolume": "7518467",
            "netBuyVolume": "-2925300",
            "breakdown": {
              "financialInvestment": {"buyVolume": "868903", "sellVolume": "2975867", "netBuyVolume": "-2106964"},
              "insurance": {"buyVolume": "210450", "sellVolume": "345100", "netBuyVolume": "-134650"},
              "investmentTrust": {"buyVolume": "1120000", "sellVolume": "1450000", "netBuyVolume": "-330000"},
              "privateEquityFund": {"buyVolume": "450000", "sellVolume": "310000", "netBuyVolume": "140000"},
              "bank": {"buyVolume": "12000", "sellVolume": "15000", "netBuyVolume": "-3000"},
              "otherFinancial": {"buyVolume": "50000", "sellVolume": "40000", "netBuyVolume": "10000"},
              "pensionFund": {"buyVolume": "1881814", "sellVolume": "2382500", "netBuyVolume": "-500686"}
            }
          },
          "otherCorporation": {
            "buyVolume": "250000",
            "sellVolume": "150000",
            "netBuyVolume": "100000"
          }
        }
      ]
    }
  }
  ```

### 4.2 `GET /api/v1/stocks/{symbol}/program-trades` — 프로그램 매매동향
* **Path:** `GET /api/v1/stocks/{symbol}/program-trades`
* **Response Payload (`result`):**
  ```json
  {
    "result": {
      "nextUntil": "2026-09-09",
      "records": [
        {
          "date": "2026-09-11",
          "arbitrage": {
            "buyVolume": "262654",
            "sellVolume": "291868",
            "netBuyVolume": "-29214"
          },
          "nonArbitrage": {
            "buyVolume": "2076706",
            "sellVolume": "5168865",
            "netBuyVolume": "-3092159"
          }
        }
      ]
    }
  }
  ```

### 4.3 `GET /api/v1/stocks/{symbol}/short-selling` — 공매도 동향
* **Path:** `GET /api/v1/stocks/{symbol}/short-selling`
* **Response Payload (`result`):**
  ```json
  {
    "result": {
      "nextUntil": "2026-09-09",
      "records": [
        {
          "date": "2026-09-11",
          "updatedAt": "2026-09-11T18:14:00.000+09:00",
          "shortSellingVolume": "1670974",
          "shortSellingAmount": "431937981500",
          "shortSellingVolumeRate": "0.11988",
          "shortSellingAmountRate": "0.1198"
        }
      ]
    }
  }
  ```

### 4.4 `GET /api/v1/stocks/{symbol}/credit-trades` & `/securities-lending` — 신용 및 대차거래
* **Credit Trades:** Daily margin loan and stock loan trading volume, outstanding balance, and ratio.
* **Securities Lending:** Daily executed, repaid, and balance quantity for lending transactions.

---

## 5. Universe Master, Screening & Ranking TRs

### 5.1 `GET /api/v1/stocks/all` — 마켓별 전체 종목 마스터 (Universe Master)
* **Path:** `GET /api/v1/stocks/all`
* **Rate Limits Group:** `STOCK_ALL` (1 req/s)
* **Query Parameters:**
  * `market` (string, required): `KOSPI`, `KOSDAQ`, `NYSE`, `NASDAQ`, `AMEX`, `KR_ETC`, `US_ETC`.
  * `status` (string, optional): `ACTIVE` (default), `SCHEDULED`, `DELISTED`.
  * `securityType` (string, optional): `STOCK`, `ETF`, `ETN`, `REIT`, `INFRASTRUCTURE_FUND`, `DEPOSITARY_RECEIPT`, `STOCK_WARRANTS`.
  * `commonShare` (boolean, optional): `true` (common shares only), `false` (preferred shares only), omitted for all.
* **Response Payload (`result` list):**
  ```json
  {
    "result": [
      {
        "symbol": "005930",
        "name": "삼성전자",
        "securityType": "STOCK",
        "isCommonShare": true,
        "isinCode": "KR7005930003"
      }
    ]
  }
  ```
* **Quant Advantage:** Full exchange universe in a single REST request without fragile pagination or rate limit exhaustion.

### 5.2 `GET /api/v1/stocks/{symbol}/warnings` — 매수 유의사항 및 투자경고
* **Path:** `GET /api/v1/stocks/{symbol}/warnings`
* **Rate Limits Group:** `STOCK` (5 req/s)
* **Output (`result` list of warning types):**
  * Warning Flags: `SHORT_OVERHEATED`, `INVESTMENT_CAUTION`, `INVESTMENT_WARNING`, `INVESTMENT_RISK`, `DELISTING_PROCEDURE`, `TRADING_HALT`, `VOLATILITY_INTERRUPTION`, etc.
  * Empty array `[]` indicates a clean, unencumbered trading candidate.

### 5.3 `GET /api/v1/rankings` — 주식 랭킹 조회 (Market Rankings)
* **Path:** `GET /api/v1/rankings`
* **Rate Limits Group:** `RANKING` (5 req/s)
* **Query Parameters:**
  * `type` (string, required):
    * `MARKET_TRADING_AMOUNT`: 시장 거래대금 상위
    * `MARKET_TRADING_VOLUME`: 시장 거래량 상위
    * `TOP_GAINERS`: 급상승 (등락률 상위) — *`duration != "realtime"` required, use `"1d"`*
    * `TOP_LOSERS`: 급하락 (등락률 하위) — *`duration != "realtime"` required, use `"1d"`*
    * `TOSS_SECURITIES_TRADING_AMOUNT`: 토스증권 거래대금 상위
    * `TOSS_SECURITIES_TRADING_VOLUME`: 토스증권 거래량 상위
  * `marketCountry` (string, required): `KR` or `US`.
  * `duration` (string, required): `realtime`, `1d`, `1w`, `1mo`, `3mo`, `6mo`, `1y`.
  * `excludeInvestmentCaution` (boolean, optional): Exclude cautioned stocks (default `false`).
  * `count` (integer, optional): Maximum items (max 100, default 100).
* **Response Payload (`result`):**
  ```json
  {
    "result": {
      "rankedAt": "2026-09-11T19:59:46.466+09:00",
      "rankings": [
        {
          "rank": 1,
          "symbol": "000660",
          "currency": "KRW",
          "price": {
            "lastPrice": "1832000",
            "basePrice": "1853000",
            "changeRate": "-0.0113"
          },
          "tradingVolume": "79878",
          "tradingAmount": "145941219000"
        }
      ]
    }
  }
  ```

### 5.4 `GET /api/v1/market-calendar/KR` & `/US` — 시장 운영 캘린더
* **Path:** `GET /api/v1/market-calendar/KR`
* **Rate Limits Group:** `MARKET_INFO` (3 req/s)
* **Output:** Precise schedules for KRX regular session, Nextrade (NXT) premarket (`08:00~09:00`), aftermarket (`15:30~20:00`), and single-price auction windows (`08:50~09:00`, `15:20~15:30`, `15:30~15:40`).

---

## 6. Account & Asset Management TRs (`/api/v1/...`)

### 6.1 `GET /api/v1/accounts` — 계좌 목록 조회
* **Path:** `GET /api/v1/accounts`
* **Rate Limits Group:** `ACCOUNT` (1 req/s)
* **Headers:** `Authorization: Bearer <access_token>`
* **Response Payload (`result` list):**
  ```json
  {
    "result": [
      {
        "accountNo": "14101527945",
        "accountSeq": 1,
        "accountType": "BROKERAGE"
      }
    ]
  }
  ```
* **Architecture Invariant:** The returned `accountSeq` integer must be supplied in all subsequent account-scoped requests via `X-Tossinvest-Account: <accountSeq>`.

### 6.2 `GET /api/v1/holdings` — 보유 주식 및 평가 현황 (Holdings & Valuation)
* **Path:** `GET /api/v1/holdings`
* **Rate Limits Group:** `ASSET` (5 req/s)
* **Headers:** `Authorization`, `X-Tossinvest-Account: <accountSeq>`
* **Query Parameters:** `symbol` (optional)
* **Response Payload (`result`):**
  ```json
  {
    "result": {
      "totalPurchaseAmount": {"krw": "5000000", "usd": null},
      "marketValue": {
        "amount": {"krw": "5300000", "usd": null},
        "amountAfterCost": {"krw": "5285000", "usd": null}
      },
      "profitLoss": {
        "amount": {"krw": "300000", "usd": null},
        "amountAfterCost": {"krw": "285000", "usd": null},
        "rate": "0.0600",
        "rateAfterCost": "0.0570"
      },
      "dailyProfitLoss": {"amount": {"krw": "120000", "usd": null}, "rate": "0.0231"},
      "items": [
        {
          "symbol": "005930",
          "name": "삼성전자",
          "quantity": "20",
          "sellableQuantity": "20",
          "purchasePrice": "250000",
          "marketPrice": "262000",
          "marketValue": "5240000",
          "profitLoss": "240000",
          "currency": "KRW"
        }
      ]
    }
  }
  ```

### 6.3 `GET /api/v1/buying-power` — 매수 가능 금액 (Cash Buying Power)
* **Path:** `GET /api/v1/buying-power`
* **Rate Limits Group:** `ORDER_INFO` (6 req/s; 3 req/s during 09:00~09:10)
* **Headers:** `Authorization`, `X-Tossinvest-Account: <accountSeq>`
* **Query Parameters:** `currency` (`KRW` or `USD`)
* **Output:** `{"result": {"currency": "KRW", "cashBuyingPower": "15000000"}}` (Strict cash basis, non-marginable).

---

## 7. Order & Conditional Order Management TRs (`/api/v1/...`)

### 7.1 `POST /api/v1/orders` — 주문 생성 (Create Order)
* **Path:** `POST /api/v1/orders`
* **Rate Limits Group:** `ORDER` (10 req/s)
* **Headers:** `Authorization: Bearer <token>`, `X-Tossinvest-Account: <accountSeq>`
* **Request Body (Quantity-Based):**
  ```json
  {
    "clientOrderId": "order-uuid-001",
    "symbol": "005930",
    "side": "BUY",
    "orderType": "LIMIT",
    "quantity": "10",
    "price": "262000",
    "confirmHighValueOrder": false
  }
  ```
* **Order Parameters & Invariants:**
  * `side`: `"BUY"` (매수) or `"SELL"` (매도).
  * `orderType`: `"LIMIT"` (지정가) or `"MARKET"` (시장가).
  * `price`: Required for `LIMIT`. Must be strictly omitted for `MARKET` (passing price to `MARKET` returns `400 invalid-request`).
  * `clientOrderId` (string, optional, max 36 chars): Client-assigned idempotency key. Repeating with same key within 10 minutes returns cached execution result without duplicate execution.
  * `confirmHighValueOrder` (boolean, default `false`): **Crucial safety invariant:** If gross order value >= 100,000,000 KRW, setting this flag to `true` is **mandatory**. Failure returns `400 confirm-high-value-required`.
  * Price tick validation: Must conform to KRX tick sizes (e.g. 100 KRW tick for 50,000~200,000 KRW). Invalid ticks return `400 invalid-request` with valid tick rules in `data`.

### 7.2 `POST /api/v1/orders/{orderId}/modify` & `/cancel` — 정정 및 취소
* **Modify (`POST /api/v1/orders/{orderId}/modify`):**
  ```json
  {
    "price": "262500",
    "quantity": "5"
  }
  ```
* **Cancel (`POST /api/v1/orders/{orderId}/cancel`):**
  ```json
  {}
  ```

### 7.3 `GET /api/v1/orders` — 주문 목록 및 내역 조회
* **Path:** `GET /api/v1/orders`
* **Rate Limits Group:** `ORDER_HISTORY` (5 req/s)
* **Headers:** `Authorization`, `X-Tossinvest-Account: <accountSeq>`
* **Query Parameters:**
  * `status` (string, required): Lifecycle group filter.
    * `"OPEN"`: Active orders (`PENDING`, `PARTIAL_FILLED`, `PENDING_CANCEL`, `PENDING_REPLACE`). Returns all without pagination.
    * `"CLOSED"`: Completed orders (`FILLED`, `CANCELED`, `REJECTED`, `REPLACED`).
  * `symbol`: Ticker filter (optional).
  * `from` / `to`: Date bounds (inclusive, KST).
  * `cursor` / `limit`: Pagination parameters for `CLOSED`.

### 7.4 `POST /api/v1/conditional-orders` — 서버 자동 조건주문 (SINGLE / OCO / OTO)
Toss Securities natively supports automated server-side conditional orders, executing without local bot keepalive:
* **Path:** `POST /api/v1/conditional-orders`
* **Rate Limits Group:** `CONDITIONAL_ORDER` (5 req/s)
* **Headers:** `Authorization`, `X-Tossinvest-Account: <accountSeq>`
* **Request Body (OCO Example: Stop-Loss & Take-Profit):**
  ```json
  {
    "symbol": "005930",
    "type": "OCO",
    "quantity": "10",
    "orderType": "LIMIT",
    "expireDate": "2026-09-30",
    "first": {
      "triggerPrice": "275000",
      "orderSide": "SELL",
      "orderPrice": "275000"
    },
    "second": {
      "triggerPrice": "255000",
      "orderSide": "SELL",
      "orderPrice": "254500"
    },
    "confirmHighValueOrder": false
  }
  ```
* **Types:**
  * `SINGLE`: Single price watch condition.
  * `OCO` (One-Cancels-the-Other): Watches two price triggers concurrently; triggering one immediately cancels the other.
  * `OTO` (One-Triggers-the-Other): Entry order execution automatically activates child profit/stop condition.

---

## 8. Realtime Streaming WebSocket (`/ws/v1`)

### 8.1 Handshake & Keepalive
* **Server Endpoint:** `wss://openapi-ws.tossinvest.com/ws/v1`
* **Authentication:** Handshake HTTP Upgrade request must supply `Authorization: Bearer <access_token>`.
* **Client Keepalive (Ping/Pong):**
  * Client sends raw text frame: `"PING"` (capital letters, 4 bytes, pure text, not JSON).
  * Recommended interval: Every 60 seconds.
  * Server responds with raw text frame: `"PONG"`.

### 8.2 Declarative Subscription Model
Subscribing is **declarative and state-replacing**: Each client message defines the entire active subscription set.
```json
[
  {"type": "trade:kr", "codes": ["005930", "000660"]},
  {"type": "orderbook:kr", "codes": ["005930"]},
  {"type": "personal:order", "codes": ["1"]}
]
```
* **Subscription Types:**
  * `trade:kr` / `trade:us`: Realtime trade tick stream.
  * `orderbook:kr` / `orderbook:us`: Realtime 10-level orderbook stream.
  * `personal:order`: Realtime fill and execution events for `codes: ["<accountSeq>"]`.

### 8.3 In-Band Frame Formats
1. **Subscription Confirmation (`subscriptionsAck`):**
   ```json
   {
     "subscribed": ["trade:kr:005930", "orderbook:kr:005930"],
     "rejected": []
   }
   ```
2. **Trade Stream Event:**
   ```json
   {
     "topic": "trade:kr:005930",
     "data": {
       "symbol": "005930",
       "timestamp": "2026-09-11T14:30:00.123+09:00",
       "price": "262000",
       "volume": "100",
       "accumulatedVolume": "15420300"
     }
   }
   ```
3. **Order Event Stream:**
   ```json
   {
     "topic": "personal:order:1",
     "data": {
       "orderId": "20260911000123",
       "symbol": "005930",
       "side": "BUY",
       "status": "FILLED",
       "executedPrice": "262000",
       "executedQuantity": "10",
       "executedAt": "2026-09-11T14:30:01.456+09:00"
     }
   }
   ```

### 8.4 WebSocket Rate Limits & Invariants
* **Concurrent Connections:** Maximum **2 connections per account**. Initiating a 3rd connection automatically closes the oldest existing connection.
* **Subscription Topic Cap:** Maximum **100 topics per connection** (`codes` combined across all channels).
* **Declaration Rate Limit:** Maximum **5 declaration requests per second** (exceeding triggers `error.code: "rate-limit-exceeded"`).

---

## 9. Comparative Architecture & Operational Positioning

### 9.1 Competitive Strengths
1. **High-Throughput Multi-Stock Price Query:** Fetching 200 stocks per call at 15 req/s enables scanning the whole KOSPI 200 index in a single request, or the entire 2,500 KRX universe in under 1 second.
2. **Instant Universe Master Dump:** `GET /api/v1/stocks/all?market=KOSPI` dumps all active stocks with ISIN codes in 1 call, avoiding legacy TR chunking.
3. **Modern JSON REST & Typed Schemas:** Completely avoids legacy quirks (such as signed number strings `+69700`, 1e6 unit multipliers, or obscure Korean TR codes).
4. **Native Idempotency & High-Value Order Shield:** Built-in `clientOrderId` (10-minute duplicate deduplication) and compulsory `confirmHighValueOrder` flag for >= 100M KRW orders.
5. **Server-Side OCO/OTO Conditional Orders:** Enables server-retained risk control without running a local order-monitoring process.

### 9.2 Critical Engineering Invariants
* **IP Whitelisting:** Calling from a cloud VM or runner without registering the outbound IP in Toss WTS immediately throws `403 Forbidden`.
* **Paging for Minute Bars:** A single candle request caps at 200 bars (unlike LS's 500 bars or Kiwoom's 900 bars). A full 390-minute trading day requires 2 calls using `before` timestamp cursor.
* **Account Header Requirement:** `X-Tossinvest-Account: <accountSeq>` is mandatory for private endpoints. Always call `GET /api/v1/accounts` first to retrieve `accountSeq`.
