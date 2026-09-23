# LS Securities (구 이베스트) OpenAPI Specification

> **AI Agent Technical Reference:**  
> Machine-readable specification of LS Securities (LS증권 / 구 이베스트투자증권) REST OpenAPI for quantitative factor research, high-capacity intraday bar collection, market microstructure flow analysis, and universe screening.

---

## 1. System & Authentication Overview

### 1.1 Endpoints & Environments
* **REST URL (Chart):** `https://openapi.ls-sec.co.kr:8080/stock/chart`
* **REST URL (Market Data):** `https://openapi.ls-sec.co.kr:8080/stock/market-data`
* **REST URL (Investor / Flow):** `https://openapi.ls-sec.co.kr:8080/stock/investor`
* **REST URL (Program Trade):** `https://openapi.ls-sec.co.kr:8080/stock/program`
* **REST URL (Ranking):** `https://openapi.ls-sec.co.kr:8080/stock/ranking`
* **REST URL (Item Search / Master):** `https://openapi.ls-sec.co.kr:8080/stock/item-search`
* **OAuth Token URL:** `https://openapi.ls-sec.co.kr:8080/oauth2/token`
* **WebSocket Live:** `wss://openapi.ls-sec.co.kr:9443`

### 1.2 OAuth2 Token Issuance (`POST /oauth2/token`)
* **Headers:** `content-type: application/x-www-form-urlencoded`
* **Payload:**
  ```python
  payload = {
      "grant_type": "client_credentials",
      "appkey": "<LS_APP_KEY>",
      "appsecretkey": "<LS_APP_SECRET>",
      "scope": "oob",
  }
  ```
* **Response:**
  ```json
  {
      "access_token": "a1b2c3d4e5f6...",
      "token_type": "Bearer",
      "expires_in": 86400,
      "scope": "oob"
  }
  ```
* **Lifecycle:** 86400s validity. Cached on `LsApiClient.token`.

### 1.3 Common Request Headers
```http
content-type: application/json; charset=utf-8
authorization: Bearer <access_token>
tr_cd: <TR_CD>
tr_cont: <Y|N>
tr_cont_key: <CONTINUATION_KEY>
```

### 1.4 Rate Limit & Single-Flight Invariants
* **Strict Server Limit:** Exceeding ~1 req/s triggers error `rsp_cd: "IGW00201"` (*"요청 제한 건수를 초과하였습니다"*).
* **Architecture Invariant (Serialization Lock):**
  * Guarded by `asyncio.Lock()` with `_min_interval = 1.05`s (~0.95 req/s).
  * In-flight concurrency is strictly **1**.
  * On `IGW00201`, sleep 1.2s and retry (up to 3 times).
* **URL Path Segregation:**
  * Chart TRs (`t8410`, `t8411`, `t8412`, `t8413`) must target `/stock/chart`.
  * Market data TRs (`t1101`, `t1102`, `t8407`, `t1301`) must target `/stock/market-data`.
  * Mismatched paths return `IGW00215` (*"유효하지 않은 TR CD 입니다"*).

---

## 2. Intraday & Historical Chart TRs (`/stock/chart`)

### 2.1 `t8412` — 주식 분봉 차트 조회 (Minute Chart)
* **Path:** `POST /stock/chart`
* **Request Headers:** `tr_cd: "t8412"`, `tr_cont: "N"`, `tr_cont_key: ""`
* **Request Body (`t8412InBlock`):**
  ```json
  {
    "t8412InBlock": {
      "shcode": "005930",
      "ncnt": 1,
      "qrycnt": 500,
      "nday": "0",
      "sdate": "YYYYMMDD",
      "stime": "090000",
      "edate": "YYYYMMDD",
      "etime": "153000",
      "cts_date": "",
      "cts_time": "",
      "comp_yn": "N"
    },
    "tr_cd": "t8412"
  }
  ```
* **Output (`t8412OutBlock1` array — up to 500 bars):**
  * `date` (`YYYYMMDD`), `time` (`HHMMSS`)
  * `open`, `high`, `low`, `close` (KRW 원)
  * `jdiff_vol`: Bar volume (봉별 체결수량)
  * `value`: Bar trade amount in **백만원 (1,000,000 KRW)**. Multiply by `1e6` to convert to KRW!
* **High-Leverage Advantage:** With `qrycnt=500`, a single call captures the entire 390-minute KRX regular session without pagination. **Primary choice for regular-session minute bars.**

### 2.2 `t8411` — 주식 틱 차트 조회 (Tick Chart)
* **Path:** `POST /stock/chart`
* **Request Body (`t8411InBlock`):** `shcode`, `ncnt: 1`, `qrycnt: 500`, `nday: "0"`, `sdate`, `stime: "090000"`, `edate`, `etime: "153000"`, `cts_date`, `cts_time`, `comp_yn: "N"`.
* **Continuation & Pagination:**
  * Header `tr_cont="Y"` and `tr_cont_key` + Body `cts_date` and `cts_time`.
  * Page budget capped (default 30~100 pages). Exhaustion returns `truncated=True`.
* **Output (`t8411OutBlock1`):** `date`, `time`, `open`, `high`, `low`, `close`, `jdiff_vol`, `value` (in **백만원**).

### 2.3 `t8410` & `t8413` — 주식 일봉 / 주기별 차트 (Daily Chart)
* **일봉 전용 (`t8413`):**
  * InBlock: `shcode`, `gubun: "2"` (일봉), `qrycnt: 500`, `sdate`, `edate`, `cts_date`, `comp_yn: "N"`.
  * OutBlock1: `date`, `open`, `high`, `low`, `close`, `jdiff_vol` (거래량), `value` (거래대금 백만원).
* **통합 주기별 차트 (`t8410`):**
  * InBlock: `shcode`, `gubun: "0"` (일), `"1"` (주), `"2"` (월), `"3"` (년), `qrycnt: 500`.

---

## 3. Market Quotation TRs (`/stock/market-data`)

### 3.1 `t1102` — 주식 현재가 시세 조회 (Current Price Quote)
* **Path:** `POST /stock/market-data`
* **Request Body:** `{"t1102InBlock": {"shcode": "005930"}, "tr_cd": "t1102"}`
* **Output (`t1102OutBlock`):**
  * `price`: 현재가
  * `change`: 전일대비
  * `diff`: 등락률 (%)
  * `volume`: 누적거래량
  * `value`: 누적거래대금 (백만원)
  * `high` / `low` / `open`: 당일 고/저/시가
  * `shcode`: 단축종목코드
  * `hname`: 종목명
  * `bidprice` / `offerprice`: 최우선 매수/매도 호가
  * `bidrem` / `offerrem`: 최우선 매수/매도 잔량

### 3.2 `t1101` — 주식 현재가 호가 조회 (Orderbook Snapshot)
* **Path:** `POST /stock/market-data`
* **Request Body:** `{"t1101InBlock": {"shcode": "005930"}, "tr_cd": "t1101"}`
* **Output (`t1101OutBlock`):** 10-level ask/bid ladder (`offerho1~10`, `bidho1~10`, `offerrem1~10`, `bidrem1~10`, `totalofferrem`, `totalbidrem`).

### 3.3 `t8407` — 주식 멀티 현재가 일괄 조회 (Multi-Stock Price)
* **Path:** `POST /stock/market-data`
* **Request Body:**
  ```json
  {
    "t8407InBlock": {
      "nrec": 10,
      "shcode": "005930000660035420..."
    },
    "tr_cd": "t8407"
  }
  ```
* **Advantage:** Retrieves quotes for up to 50 stock codes in a single API call (codes concatenated in 6-digit chunks).

### 3.4 `t1301` & `t1302` & `t1305` — 체결 및 기간별 주가 내역
* **시간대별 체결 (`t1301`):** `shcode`, `cvolume` (최저거래량). Output: `chetime` (체결시각), `price`, `volume`, `cgubun` (+ 체결매수, - 체결매도).
* **일별 체결 (`t1302`):** 일자별 주가, 전일대비, 거래량 히스토리.
* **기간별 주가 (`t1305`):** `shcode`, `sdate`, `edate`. 기간 내 일별 OHLCV 및 거래대금.

---

## 4. Market Microstructure & Flow TRs (수급 / 프로그램 / 공매도 / 신용)

### 4.1 `t1404` & `t1405` — 종목별 투자자 매매동향 (Stock Investor Flow)
* **Path:** `POST /stock/investor`
* **일별 동향 (`t1404`):**
  * Request Body: `{"t1404InBlock": {"shcode": "005930", "sdate": "YYYYMMDD", "edate": "YYYYMMDD"}}`
  * Output (`t1404OutBlock1` list): `date`, `price`, `person` (개인순매수), `foreigner` (외국인순매수), `organ` (기관순매수), `trust` (투신), `bank` (은행), `insu` (보험), `fund` (연기금), `etc` (기타법인).
* **시간대별 동향 (`t1405`):** 당일 장중 시간대별 외국인/기관 잠정 순매수 추이.

### 4.2 `t1471` — 시장 전체 투자자별 매매종합 (Market-Wide Investor Flow)
* **Path:** `POST /stock/investor`
* **Params:** `gubun1: "0"` (수량), `"1"` (금액), `market: "1"` (KOSPI), `"2"` (KOSDAQ).
* **Output:** 투자주체별 당일 순매수 총액 및 비중.

### 4.3 `t1441` & `t1452` — 프로그램 매매 추이 (Program Trading)
* **시간대별 프로그램 (`t1441`):** `POST /stock/program`. 시간대별 차익, 비차익, 전체 매수/매도 수량 및 금액.
* **일별 프로그램 (`t1452`):** `POST /stock/program`. 일별 프로그램 누적 순매수 추이.

### 4.4 `t1481` & `t1482` & `t1486` — 공매도 / 신용 / 대차거래 추이
* **공매도 추이 (`t1481`):** 일자별 공매도 체결수량, 공매도 거래대금, 공매도 비중(%).
* **신용잔고 추이 (`t1482`):** 신용융자 잔고수량, 대주 잔고수량, 공여율, 잔고율(%).
* **대차거래 추이 (`t1486`):** 일별 대차 체결수량, 대차 상환수량, 대차 잔고수량 및 잔고금액.

---

## 5. Ranking & Master Screening TRs (순위 / 종목검색)

### 5.1 `t1489` — 주식 등락률 순위 (Fluctuation Ranking)
* **Path:** `POST /stock/ranking`
* **Request Body:**
  ```json
  {
    "t1489InBlock": {
      "gubun": "0",
      "market": "0",
      "jongchk": "0",
      "idx": 0,
      "yesprice": 0
    },
    "tr_cd": "t1489"
  }
  ```
* **Parameters:** `gubun: "0"` (상승률 상위), `"1"` (하락률 상위). `market: "0"` (전체), `"1"` (코스피), `"2"` (코스닥).
* **Output (`t1489OutBlock1`):** `shcode`, `hname`, `price`, `diff` (등락률), `volume`, `value`.

### 5.2 `t1490` — 주식 거래량 순위 (Volume Ranking)
* **Path:** `POST /stock/ranking`
* **Request Body:** `{"t1490InBlock": {"gubun": "0", "market": "0", "jongchk": "0", "idx": 0}, "tr_cd": "t1490"}`
* **Output:** 당일 거래량 상위 종목 리스트.

### 5.3 `t1491` — 주식 거래대금 순위 (Trade Amount Ranking)
* **Path:** `POST /stock/ranking`
* **Request Body:** `{"t1491InBlock": {"market": "0", "jongchk": "0", "idx": 0}, "tr_cd": "t1491"}`
* **Output:** 당일 누적 거래대금 상위 종목 리스트.

### 5.4 `t8430` — 주식 종목 마스터 조회 (Stock Universe Master)
* **Path:** `POST /stock/item-search`
* **Request Body:** `{"t8430InBlock": {"gubun": "0"}, "tr_cd": "t8430"}`
* **Parameters:** `gubun: "0"` (전체), `"1"` (코스피), `"2"` (코스닥).
* **Output (`t8430OutBlock` list):** `shcode` (6자리 단축코드), `expcode` (12자리 표준코드), `hname` (한글명), `etfgubun` (ETF구분), `uplmtprice` (상한가), `dnlmtprice` (하한가), `recprice` (기준가), `parvalue` (액면가).

### 5.5 `t1809` — 서버 조건검색 실시간 신호 (Server Condition Search)
* **Path:** `POST /stock/item-search`
* **Parameters:** HTS에서 작성하여 서버에 저장한 조건식 번호.
* **Output:** 조건식 포착 종목 코드 및 시세.
