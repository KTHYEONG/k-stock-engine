<!-- DEPRECATED_TOKENS: ka10024 -->
# Kiwoom Securities REST OpenAPI (Open API NEXT) Specification

> **AI Agent Technical Reference:**  
> Machine-readable specification of Kiwoom Securities (키움증권) REST OpenAPI for automated market screening, high-frequency tick capture, Nextrade ATS bar extraction, and quantitative flow analysis.

---

## 1. System & Authentication Overview

### 1.1 Endpoints & Environments
* **REST Base URL:** `https://api.kiwoom.com`
* **OAuth Token URL:** `https://api.kiwoom.com/oauth2/token`
* **OAuth Revoke URL (`/oauth2/revoke`):** `https://api.kiwoom.com/oauth2/revoke`
* **WebSocket URL (`/api/dostk/websocket`):** `wss://api.kiwoom.com:10000/api/dostk/websocket`
* **Linux 64-Bit Native:** Operates entirely over standard HTTPS/WSS requests without ActiveX, COM, or 32-bit Windows dependencies.

### 1.2 OAuth2 Token Lifecycle (`POST /oauth2/token`)
* **Endpoint:** `POST /oauth2/token`
* **Headers:** `Content-Type: application/json;charset=UTF-8`
* **Body:**
  ```json
  {
    "grant_type": "client_credentials",
    "appkey": "<KIWOOM_APP_KEY>",
    "secretkey": "<KIWOOM_SECRET_KEY>"
  }
  ```
* **Response:**
  ```json
  {
    "token": "eyJhbGciOi...",
    "token_type": "Bearer",
    "expires_in": 86400,
    "return_code": 0,
    "return_msg": "정상처리되었습니다."
  }
  ```
* **Concurrency Lock & CAS Refresh:** Token issuance uses single-flight concurrency locking ensuring concurrent tasks share an in-flight token fetch. Token refresh on code 8005 performs compare-and-swap (CAS) on the active token so peers do not overwrite valid tokens.

### 1.3 Token Revocation (`POST /oauth2/revoke` — `au10002`)
* **Endpoint:** `POST /oauth2/revoke`
* **Headers:** `Content-Type: application/json;charset=UTF-8`
* **Body:**
  ```json
  {
    "appkey": "<KIWOOM_APP_KEY>",
    "secretkey": "<KIWOOM_SECRET_KEY>",
    "token": "<ACTIVE_TOKEN>"
  }
  ```
* **Response:** `{"return_code": 0, "return_msg": "정상처리되었습니다."}`
* **Lifecycle:** Revoking an active token invalidates the 24h phase immediately and allows clean re-issuance without server-side stale session lockup.

### 1.4 Common Request Headers
```http
Content-Type: application/json;charset=UTF-8
authorization: Bearer <access_token>
api-id: <TR_API_ID>
cont-yn: <Y|N>
next-key: <CONTINUATION_KEY>
```

### 1.5 Rate Limit, Concurrency, and Token Bucket Invariants
* **TR-Level (api-id) Independent Buckets:**
  * Strict limit of **5.0 req/sec per `api-id`**.
  * Server Throttle Error: HTTP 429 with `{"return_code": 5, "return_msg": "허용된 API 요청 개수를 초과하였습니다. 유량=5, API ID=<api_id>"}`.
* **Architecture Invariant (Inter-TR Parallelism):**
  * Rate limiting is tracked per distinct `api-id`. Inquiries to different TRs (e.g. `ka10027` ranking alongside `ka10080` minute chart) run in parallel without cross-blocking.
  * On HTTP 429, wait exponential backoff (e.g. 1.2 s) and retry up to 3 attempts.

### 1.6 WebSocket Real-Time Streaming (`/api/dostk/websocket`)
* **Endpoint:** `wss://api.kiwoom.com:10000/api/dostk/websocket`
* **Authentication:** Handshake frames with issued Bearer token (`LOGIN` frame).
* **Stream Type Mapping to Canonical Streams:**
  * `0B`: KRX Real-Time Trade Execution Tick (canonical `H0STCNT0`)
  * `0D`: KRX Real-Time Level-10 Orderbook Depth (canonical `H0STASP0`)
* **Protocol:** JSON frames with `tr_type`, `tr_key`, `tr_data`. Client sends PING keep-alive frames every 30s.

---

## 2. Ranking & Screening TRs (`/api/dostk/rkinfo`)

### 2.1 `ka10027` — 전일대비 등락률 상위 요청 (Fluctuation Ranking)
* **Path:** `POST /api/dostk/rkinfo`
* **Headers:** `api-id: "ka10027"`, `cont-yn`, `next-key`
* **Request Body (Empirically Verified Required Schema):**
  ```json
  {
    "mrkt_tp": "000",
    "stex_tp": "3",
    "sort_tp": "1",
    "trde_qty_cnd": "0000",
    "stk_cnd": "0",
    "crd_cnd": "0",
    "updown_incls": "0",
    "pric_cnd": "0",
    "trde_prica_cnd": "0"
  }
  ```
* **Parameters:**
  * `mrkt_tp`: `"000"` (전체), `"001"` (코스피), `"101"` (코스닥)
  * `stex_tp`: `"1"` (KRX), `"2"` (NXT), `"3"` (통합/전체)
  * `sort_tp`: `"1"` (상승률 상위), `"2"` (하락률 상위)
  * `trde_qty_cnd`: `"0000"` (전체거래량)
  * `stk_cnd`: `"0"` (전체종목)
  * `crd_cnd`: `"0"` (전체신용)
  * `updown_incls`: `"0"` (상하한포함)
  * `pric_cnd`: `"0"` (전체가격)
  * `trde_prica_cnd`: `"0"` (전체거래대금)
* **Pagination & Output (`pred_pre_flu_rt_upper` array — 200 rows/page):**
  * Header `cont-yn == "Y"` and `next-key` permit paging up to `max_pages=5` (1,000 stocks).
  * Fields: `stk_cd` (`004490_AL`), `stk_nm`, `cur_prc` (`+69700`), `pred_pre` (`+5400`), `flu_rt` (`+8.40`), `now_trde_qty`, `cntr_str` (체결강도).
* **Architecture Role:** **Primary vendor for universe candidate scan.**

### 2.2 `ka10023` — 거래량 상위 순위 (Top Traded Volume)
* **Path:** `POST /api/dostk/rkinfo`
* **Headers:** `api-id: "ka10023"`
* **Body:** `{"mrkt_tp": "000", "stex_tp": "3", "trde_qty_tp": "0", "prc_tp": "0"}`
* **Output:** 당일 거래량 상위 100~200 종목 리스트 (`stk_cd`, `cur_prc`, `trde_qty`, `flu_rt`).

### 2.3 `ka10032` — 거래대금 상위 순위 (Top Trade Amount — Live Fix)
* **Path:** `POST /api/dostk/rkinfo`
* **Headers:** `api-id: "ka10032"`, `cont-yn`, `next-key`
* **Request Body (accepted live, 2026-11-03 20:16 KST):** `{"mrkt_tp": "000", "mang_stk_incls": "0", "stex_tp": "1"}` (`stex_tp` `"1"` = KRX).
* **Output (`trde_prica_upper` array — 100 rows/page):** `cont-yn: Y` with an 8-character `next-key` pages onward (3 pages / 300 rows walked).
  * Fields: `now_rank`, `pred_rank`, `stk_cd` (6-char, alphanumeric codes such as `0147B0` occur; no `_AL`/`_NX` suffix under `stex_tp=1`), `stk_nm`, `cur_prc` (signed), `pred_pre_sig`, `pred_pre`, `flu_rt` (signed), `sel_bid`, `buy_bid`, `now_trde_qty` (may hold the 32-bit overflow value 4294967295), `pred_trde_qty`, `trde_prica` (백만원).
* **Note:** 구 거래대금 순위 TR 코드는 `[1504] 해당 URI에서는 지원하는 API ID가 아닙니다` 에러를 반환하므로 거래대금 순위 조회는 반드시 `ka10032`를 사용한다.

### 2.4 `ka10028` — 시가총액 상위 순위 (Market Cap Ranking)
* **Path:** `POST /api/dostk/rkinfo`
* **Headers:** `api-id: "ka10028"`
* **Body:** `{"mrkt_tp": "000"}`
* **Output:** 시가총액 상위 종목 리스트 (`stk_cd`, `cur_prc`, `mkt_cap` 억원 단위).

### 2.5 `ka10034` — 체결강도 상위 순위 (High Trade Strength)
* **Path:** `POST /api/dostk/rkinfo`
* **Headers:** `api-id: "ka10034"`
* **Output:** 당일 체결강도(매수체결량/매도체결량) 상위 순위 종목.

---

## 3. Intraday & Historical Chart TRs (`/api/dostk/chart`)

### 3.1 `ka10080` — 주식 분봉 차트 (Minute Bar Chart)
* **Path:** `POST /api/dostk/chart`
* **Headers:** `api-id: "ka10080"`
* **Request Body:**
  ```json
  {
    "stk_cd": "005930_NX",
    "base_dt": "YYYYMMDD",
    "tic_scope": "1",
    "upd_stkpc_tp": "0"
  }
  ```
* **Parameters:**
  * `stk_cd`: 6-digit ticker, or with `_NX` suffix for Nextrade ATS.
  * `base_dt`: Target date (`YYYYMMDD`).
  * `tic_scope`: Bar interval (`"1"` for 1-minute bars).
  * `upd_stkpc_tp`: `"0"` (수정미반영), `"1"` (수정주가 반영).
* **Output (`stk_min_pole_chart_qry` — up to 900 bars/call):**
  * `cntr_tm`: `YYYYMMDDHHMMSS`
  * `cur_prc`: Close price (signed string)
  * `open_pric`, `high_pric`, `low_pric`
  * `trde_qty`: Bar traded volume
  * `acc_trde_qty`: Cumulative volume
* **Nextrade Slicing:**
  * **Premarket:** Sliced between `080000` and `085000`.
  * **Aftermarket:** Sliced between `154000` and `200000`.
  * A single call yields the full session. **Exclusive primary vendor for NXT intraday bars.**

### 3.2 `ka10079` — 주식 틱 차트 (Tick Chart)
* **Path:** `POST /api/dostk/chart`
* **Headers:** `api-id: "ka10079"`, `cont-yn`, `next-key`
* **Request Body:**
  ```json
  {
    "stk_cd": "005930",
    "base_dt": "YYYYMMDD",
    "tic_scope": "1",
    "upd_stkpc_tp": "1"
  }
  ```
* **Output (`stk_tic_chart_qry` — 900 ticks/call):**
  * Returns **900 ticks per page** (1.8x LS, 30x KIS).
  * Pagination via header `cont-yn == "Y"` and `next-key`.
  * Page budget cap: default 30 pages (~27,000 ticks). Budget exhaustion flags `truncated=True`.
  * Fields: `cntr_tm`, `cur_prc`, `trde_qty`, `open_pric`, `high_pric`, `low_pric`.
* **Architecture Role:** **Primary vendor for regular-session trade ticks.**

### 3.3 `ka10081` & `ka10082` & `ka10083` — 주식 일봉 / 주봉 / 월봉 차트
* **일봉 차트 (`ka10081`):** `stk_cd`, `base_dt`, `upd_stkpc_tp: "1"`. Output: `stk_dd_pole_chart_qry` (일자별 시가/고가/저가/종가/거래량/거래대금).
* **주봉 차트 (`ka10082`)** & **월봉 차트 (`ka10083`)**.

---

## 4. Stock Information & Quotation TRs (`/api/dostk/stkinfo`)

### 4.1 `ka10001` — 주식 기본정보 조회 (Basic Stock Info)
* **Path:** `POST /api/dostk/stkinfo`
* **Headers:** `api-id: "ka10001"`
* **Body:** `{"stk_cd": "005930"}`
* **Output:** `stk_nm` (종목명), `lstg_stk_qty` (상장주수), `par_prc` (액면가), `cptl` (자본금), `setl_mm` (결산월), `crd_lmt_rt` (신용보증금율).

### 4.2 `ka10003` — 주식 현재가 시세 (Current Price Quote)
* **Path:** `POST /api/dostk/stkinfo`
* **Headers:** `api-id: "ka10003"`
* **Body:** `{"stk_cd": "005930"}`
* **Output:** `cur_prc` (현재가), `pred_pre` (전일대비), `flu_rt` (등락률), `now_trde_qty` (거래량), `trde_amt` (거래대금), `high_pric`, `low_pric`, `open_pric`, `mkt_cap` (시가총액).

### 4.3 `ka10004` — 주식 호가 및 체결잔량 (Orderbook & Quotes)
* **Path:** `POST /api/dostk/stkinfo`
* **Headers:** `api-id: "ka10004"`
* **Body:** `{"stk_cd": "005930"}`
* **Output:** 10-level ask/bid prices (`sel_fprc_1~10`, `buy_fprc_1~10`) and remaining sizes (`sel_req_1~10`, `buy_req_1~10`), `tot_sel_req`, `tot_buy_req`.

### 4.4 `ka10005` & `ka10006` — 시간대별 / 일자별 체결내역
* **시간대별 체결 (`ka10005`):** 당일 체결시각, 체결가, 체결수량.
* **일자별 체결 (`ka10006`):** 일자별 종가, 대비, 등락률, 거래량 히스토리.

---

## 5. Flow & Microstructure TRs (`/api/dostk/stkinfo`)

### 5.1 `ka10014` — 종목별 투자자 매매동향 (Investor Trend)
* **Path:** `POST /api/dostk/stkinfo`
* **Headers:** `api-id: "ka10014"`
* **Body:** `{"stk_cd": "005930", "base_dt": "YYYYMMDD"}`
* **Output:** 일자별 개인, 외국인, 기관계, 금융투자, 보험, 투신, 기타금융, 은행, 연기금, 사모펀드, 기타법인의 순매수 수량 및 거래대금.

### 5.2 `ka10015` — 기관 / 외인 연속 순매수 (Consecutive Net Buy)
* **Path:** `POST /api/dostk/stkinfo`
* **Headers:** `api-id: "ka10015"`
* **Output:** N일 연속 순매수 중인 종목 리스트 (기관/외인 수급 지속성 분석용).

### 5.3 `ka10016` — 프로그램 매매동향 (Program Trading Trend)
* **Path:** `POST /api/dostk/stkinfo`
* **Headers:** `api-id: "ka10016"`
* **Body:** `{"stk_cd": "005930"}`
* **Output:** 당일 차익, 비차익, 전체 매도/매수 수량 및 순매수금액.

### 5.4 `ka10059` — 종목별 투자자/기관 일별 순매수 금액 (Investor/Institution Daily)
* **Path:** `POST /api/dostk/stkinfo`
* **Headers:** `api-id: "ka10059"`
* **Body:** `dt`, `stk_cd`, `amt_qty_tp`=1 (금액), `trde_tp`=0 (순매수), `unit_tp`=1000
* **Output:** 일자별 행 (`dt`, `orgn`, `frgnr_invsr`, `natfor`); KIS 외국인 순매수 = `frgnr_invsr + natfor`.
