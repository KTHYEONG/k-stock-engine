<!-- DEPRECATED_TOKENS: fhpst01720000, /uapi/domestic-stock/v1/ranking/trade-amount, tttc0802u, vttc0802u, tttc0801u, vttc0801u -->
# Korea Investment & Securities (KIS) OpenAPI Specification

> **AI Agent Technical Reference:**  
> Machine-readable specification of Korea Investment & Securities (한국투자증권) OpenAPI for automated algorithmic trading, market data extraction, quantitative factor engineering, and order/account management.

---

## 1. System & Authentication Overview

### 1.1 Endpoints
* **Live Base URL:** `https://openapi.koreainvestment.com:9443`
* **Paper (Virtual) Base URL:** `https://openapivts.koreainvestment.com:29443`
* **WebSocket Live:** `ws://ops.koreainvestment.com:21000`, Paper: `ws://ops.koreainvestment.com:31000`

### 1.2 OAuth2 Token Issuance (`POST /oauth2/tokenP`)
* **Endpoint:** `POST /oauth2/tokenP`
* **Headers:** `content-type: application/json`
* **Body:**
  ```json
  {
    "grant_type": "client_credentials",
    "appkey": "<KIS_APP_KEY>",
    "appsecret": "<KIS_APP_SECRET>"
  }
  ```
* **Response:**
  ```json
  {
    "access_token": "eyJhbGciOi...",
    "token_type": "Bearer",
    "expires_in": 86400,
    "access_token_token_expired": "2026-09-10 20:50:00"
  }
  ```
* **Lifecycle:** 24h validity. Auto-refreshed when under 10m remains, or upon `EGW00121`/`EGW00123` errors.

### 1.3 Common Request Headers
```http
content-type: application/json
authorization: Bearer <access_token>
appKey: <KIS_APP_KEY>
appSecret: <KIS_APP_SECRET>
tr_id: <TR_CODE>
custtype: P
```
*(Inquire: `custtype: P`. Order TRs use exact live vs paper `tr_id` prefixes).*

### 1.4 Rate Limit & Concurrency Invariants
* **Rate Limiter:** Every KIS REST request passes a host-paced rate limiter at `KIS_REST_RATE_PER_S` (18.0 req/s) per AppKey. Admission is coordinated across host processes sharing token/limiter state under file locking.
* **Concurrency:** Synchronous or bounded by call-site semaphores; requests are serialized through the rate limiter.
* **Throttling Handling:** HTTP 429 or `msg_cd` `EGW00201` ("초당 거래건수 초과") triggers backoff retry up to safe retry limits.

### 1.5 Market Division Codes (`FID_COND_MRKT_DIV_CODE`)
* `J`: KRX (KOSPI/KOSDAQ)
* `NX`: Nextrade ATS
* `UN`: Unified (Blended)
* `U`: Index / Sector

### 1.6 WebSocket Connection & Approval Key (`POST /oauth2/Approval`)
* **Approval Key Endpoint:** `POST /oauth2/Approval`
* **Body:** `{"grant_type": "client_credentials", "appkey": "<KIS_APP_KEY>", "secretkey": "<KIS_APP_SECRET>"}`
* **Response:** `{"approval_key": "<APPROVAL_KEY>"}`
* **WebSocket Streams:**
  * `H0STCNT0`: KRX Real-Time Trade Execution Tick (실시간 체결가)
  * `H0STASP0`: KRX Real-Time Level-10 Orderbook (실시간 10단계 호가)
  * `H0NXCNT0`: Nextrade ATS Real-Time Trade Execution Tick (NXT 실시간 체결가)
  * `H0NXASP0`: Nextrade ATS Real-Time Level-10 Orderbook (NXT 실시간 호가)

### 1.7 Request Hashkey Generation (`POST /uapi/hashkey`)
* **Endpoint:** `POST /uapi/hashkey`
* **Headers:** `content-type: application/json`, `appKey: <KIS_APP_KEY>`, `appSecret: <KIS_APP_SECRET>`
* **Body:** The exact JSON payload string of the target POST request (e.g. order-cash).
* **Response:** `{"HASH": "<HASHKEY>"}`
* **Header Usage:** Attached as `hashkey: <HASHKEY>` to order placement requests for payload integrity verification.

---

## 2. Market Data & Quotation TRs (시세 / 체결 / 호가)

### 2.1 `FHKST01010100` — 주식 현재가 시세 (Inquire Price)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-price`
* **Params:** `fid_cond_mrkt_div_code` (`J`/`NX`), `fid_input_iscd` (6-digit code)
* **Output (`output`):**
  * `stck_prpr`: 현재가 (KRW)
  * `stck_oprc` / `stck_hgpr` / `stck_lwpr`: 시가 / 고가 / 저가
  * `stck_sdpr`: 전일 종가
  * `acml_vol`: 누적 거래량
  * `acml_tr_pbmn`: 누적 거래대금 (원 단위)
  * `prdy_ctrt`: 전일 대비율 (%)
  * `prdy_vrss`: 전일 대비 금액
  * `lstn_stcn`: 상장주수
  * `hts_avls`: HTS 시가총액 (억원 단위)
  * `rprs_mrkt_kor_name`: 대표 시장 한글명 (`KOSPI`, `KOSDAQ`)

### 2.2 `FHKST01010200` — 주식 현재가 호가 / 예상체결 (Orderbook & Auction)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`/`NX`), `FID_INPUT_ISCD`
* **Output Split:**
  * `output1` (10단계 호가 사다리): `aspr_acpt_hour`, `askp1`~`askp10`, `bidp1`~`bidp10`, `askp_rsqn1`~`askp_rsqn10`, `bidp_rsqn1`~`bidp_rsqn10`, `total_askp_rsqn`, `total_bidp_rsqn`.
  * `output2` (동시호가 예상체결): `antc_cnpr` (예상체결가), `antc_cntg_vrss`, `antc_cntg_prdy_ctrt`, `antc_vol` (예상체결량), `antc_mkop_cls_code`, `vi_cls_code` (VI발동여부).

### 2.3 `FHKST01010300` — 체결강도 조회 (Inquire CCNL)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-ccnl`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`), `FID_INPUT_ISCD`
* **Output (`output` list):** `stck_cntg_hour`, `stck_prpr`, `cntg_vol`, `tday_rltv` (체결강도 %).

### 2.4 `FHPST01060000` — 당일 시간대별 체결 (Trade Ticks)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-time-itemconclusion`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`), `FID_INPUT_ISCD`, `FID_INPUT_HOUR_1` (`""` or `HHMMSS`), `FID_PW_DATA_INCU_YN` (`"Y"`)
* **Output (`output2` list — 30 ticks/call):**
  * `stck_cntg_hour`: 체결시각 (`HHMMSS`)
  * `stck_prpr`: 체결가
  * `cntg_vol` / `cnqn`: 체결수량
  * `acml_vol`: 누적체결수량 (고유 단조증가 키로 dedup에 사용)
  * `tday_rltv`: 체결강도
* **Pagination:** Reverse cursor on `stck_cntg_hour`. Stall on duplicate ticks broken via 1s decrement. Forward-only (same-day).

### 2.5 `CTPF1002R` — 상품기본정보 / 주식 정보 조회 (Search Stock Info)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/search-stock-info`
* **Params:** `PRDT_TYPE_CD` (`"300"`: 주식), `PDNO` (6-digit symbol code)
* **Output (`output`):**
  * `pdno`: 종목코드
  * `prdt_type_cd`: 상품유형코드
  * `prdt_name`: 종목 한글명
  * `prdt_eng_name`: 종목 영문명
  * `std_pdno`: 표준종목코드 (ISIN)
  * `mkt_id_cd`: 시장구분코드 (`STK`: 코스피, `KSQ`: 코스닥)
  * `scts_mkt_lstg_dt`: 상장일자 (`YYYYMMDD`)

### 2.6 `FHPUP02100000` — 업종 / 지수 현재가 시세 (Inquire Index Price)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-index-price`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`"U"`), `FID_INPUT_ISCD` (`"0001"`: 코스피, `"1001"`: 코스닥, `"1028"`: 코스피200)
* **Output (`output`):**
  * `bstp_nmix_prpr`: 업종/지수 현재가
  * `bstp_nmix_prdy_ctrt`: 전일대비율 (%)
  * `bstp_nmix_prdy_vrss`: 전일대비
  * `acml_vol`: 누적거래량
  * `acml_tr_pbmn`: 누적거래대금 (백만원)

### 2.7 `FHKST01011800` — 당일 뉴스 제목 조회 (News Title)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/news-title`
* **Params:** `FID_NEWS_OFR_ENTP_CODE` (`""`), `FID_COND_MRKT_DIV_CODE` (`"J"`), `FID_INPUT_ISCD` (종목코드 또는 빈값)
* **Output (`output` list):**
  * `news_ofer_entp_code`: 뉴스제공업체코드
  * `dorg`: 배포일시 (`YYYYMMDDHHMMSS`)
  * `titl`: 뉴스 제목
  * `news_lpad_yn`: 중요뉴스 여부

---

## 3. Chart & Time Series TRs (차트 / 시계열)

### 3.1 `FHKST03010200` — 당일 분봉 조회 (Intraday Minute Chart)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`/`NX`), `FID_INPUT_ISCD`, `FID_INPUT_HOUR_1` (`HHMMSS`), `FID_PW_DATA_INCU_YN` (`"Y"`)
* **Output (`output2` — 30 bars/call):** `stck_cntg_hour`, `stck_oprc`, `stck_hgpr`, `stck_lwpr`, `stck_prpr`, `cntg_vol`, `acml_tr_pbmn` (누적원).

### 3.2 `FHKST03010230` — 일별 분봉 조회 (Historical Minute Chart Backfill)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`/`NX`), `FID_INPUT_ISCD`, `FID_INPUT_DATE_1` (`YYYYMMDD`), `FID_INPUT_HOUR_1` (`153000`), `FID_PW_DATA_INCU_YN` (`"Y"`), `FID_FAKE_TICK_INCU_YN` (`""` — Mandatory empty string, omitting triggers `OPSQ2001`).
* **Capacity & Retention:** 120 bars/call. Retains **~1 rolling year** on KIS servers.

### 3.3 `FHKST03010100` — 국내주식 기간별 시세 (Daily/Weekly/Monthly OHLCV)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice`
* **Params:** `fid_cond_mrkt_div_code` (`J`), `fid_input_iscd`, `fid_input_date_1` (start), `fid_input_date_2` (end), `fid_period_div_code` (`D`), `fid_org_adj_prc` (`"1"` 수정주가 반영).
* **Output (`output2` — 100 days/call):** `stck_bsop_date`, `stck_oprc`, `stck_hgpr`, `stck_lwpr`, `stck_clpr`, `acml_vol`, `acml_tr_pbmn`.

### 3.4 `FHKUP03500100` — 업종 / 지수 기간별 시세 (Market Index Chart)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice`
* **Params:** `fid_cond_mrkt_div_code` (`"U"`), `fid_input_iscd` (`0001`: 코스피, `1001`: 코스닥, `1028`: 코스피200), `fid_input_date_1`, `fid_input_date_2`, `fid_period_div_code` (`D`), `fid_org_adj_prc` (`"0"`).
* **Output:** `output1` (당일 지수 요약: `bstp_nmix_prpr`, `bstp_nmix_prdy_ctrt`), `output2` (일자별 지수 시계열).

### 3.5 `FHKUP03500200` — 업종 / 지수 분봉 시세 (Index Intraday Minute Chart)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-time-indexchartprice`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`"U"`), `FID_INPUT_ISCD` (`"0001"`: 코스피, `"1001"`: 코스닥), `FID_INPUT_HOUR_1` (`HHMMSS`), `FID_PW_DATA_INCU_YN` (`"Y"`)
* **Output (`output2` list):** `bsop_hour`, `bstp_nmix_prpr`, `bstp_nmix_oprc`, `bstp_nmix_hgpr`, `bstp_nmix_lwpr`, `acml_vol`, `acml_tr_pbmn`.

---

## 4. Market Microstructure & Flow TRs (수급 / 수량 / 공매도 / 신용)

### 4.1 `HHPTJ04160200` — 외인 / 기관 추정가집계 (Investor Trend Estimate)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/investor-trend-estimate`
* **Params:** `MKSC_SHRN_ISCD`
* **Output (`output2` list, index 0 is latest):** `bsop_hour`, `frgn_fake_ntby_qty` (외국인 순매수 추정주수), `orgn_fake_ntby_qty` (기관 순매수 추정주수), `frgn_fake_shnu_qty`, `orgn_fake_shnu_qty`.

### 4.2 `FHKST01010900` — 종목별 투자자 매매동향 (Daily Investor Trend)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/inquire-investor`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`), `FID_INPUT_ISCD`
* **Output (`output` list):** `stck_bsop_date`, `prsn_ntby_qty` (개인순매수), `frgn_ntby_qty` (외국인순매수), `orgn_ntby_qty` (기관계순매수), `scrt_ntby_qty` (금융투자), `insn_ntby_qty` (보험), `mrlf_ntby_qty` (투신), `bank_ntby_qty` (은행), `etc_finc_ntby_qty` (기타금융), `pnsn_ntby_qty` (연기금), `sfund_ntby_qty` (사모펀드), `etc_corp_ntby_qty` (기타법인).

### 4.3 `FHPPG04650101` & `FHPPG04650201` — 프로그램 매매 추이 (Program Trading)
* **실시간 당일추이:** `FHPPG04650101` (`/uapi/domestic-stock/v1/quotations/program-trade-by-stock`)
  * Output: 시간대별 차익/비차익 매수/매도 수량 및 순매수금액.
* **일별 추이:** `FHPPG04650201` (`/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily`)
  * Params: `FID_COND_MRKT_DIV_CODE`, `FID_INPUT_ISCD`, `FID_INPUT_DATE_1` (커서일자)
  * Output: 일별 차익/비차익/전체 순매수 거래량 및 거래대금.
* **종목별 투자자 일별 (`FHPTJ04160001`):** `GET /uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily`
  * Params: `FID_COND_MRKT_DIV_CODE`, `FID_INPUT_ISCD`, `FID_INPUT_DATE_1` 커서, `FID_ORG_ADJ_PRC`="", `FID_ETC_CLS_CODE`=""
  * Output (`output2`): `stck_bsop_date`, `frgn_ntby_tr_pbmn`, `orgn_ntby_tr_pbmn`.
* 두 일별 수급 TR은 모두 `FID_INPUT_DATE_1` 커서로만 페이징한다 (`tr_cont` 없음).

### 4.4 `FHPST04830000` — 국내주식 공매도 일별추이 (Daily Short Selling)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/daily-short-sale`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`), `FID_INPUT_ISCD`, `FID_INPUT_DATE_1`, `FID_INPUT_DATE_2`
* **Output (`output2` list — 100 rows/call):** `stck_bsop_date`, `ssts_cntg_qty` (공매도체결수량), `ssts_tr_pbmn` (공매도거래대금), `acml_vol` (총거래량), `ssts_vol_rlim` (공매도비중 %). 5년+ 백필 가능.

### 4.5 `FHPST04760000` — 국내주식 신용잔고 일별추이 (Credit Balance)
* **Path:** `GET /uapi/domestic-stock/v1/quotations/daily-credit-balance`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`), `FID_INPUT_ISCD`, `FID_COND_SCR_DIV_CODE` (`"20476"`), `FID_INPUT_DATE_1`
* **Output (`output` list):** `deal_date`, `whol_loan_new_stcn`, `whol_loan_rdmp_stcn`, `whol_loan_rmnd_stcn` (신용잔고수량), `whol_loan_rmnd_amt` (신용잔고금액), `whol_loan_rmnd_rate` (잔고율), `whol_stln_rmnd_stcn` (대주잔고).

---

## 5. Ranking & Screening TRs (순위 / 조건검색)

### 5.1 `FHPST01700000` — 등락률 순위 (Fluctuation Ranking)
> [!IMPORTANT]
> 실측 정정(2026-09-17): 이 TR은 14개 필드가 전부 필수다.
* **Path:** `GET /uapi/domestic-stock/v1/ranking/fluctuation`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`), `FID_COND_SCR_DIV_CODE` (`20170`), `FID_INPUT_ISCD` (`0000`), `FID_RANK_SORT_CLS_CODE` (`0`: 상승률순), `FID_INPUT_CNT_1` (`200`), `FID_PRC_CLS_CODE` (`0`: 전체), `FID_INPUT_PRICE_1` (`""`), `FID_INPUT_PRICE_2` (`""`), `FID_VOL_CNT` (`""`), `FID_TRGT_CLS_CODE` (`0`), `FID_TRGT_EXLS_CLS_CODE` (`0`), `FID_DIV_CLS_CODE` (`0`), `FID_RSFL_RATE1` (min%), `FID_RSFL_RATE2` (max%).
* **Output (`output`):** `stck_shrn_iscd`, `hts_kor_isnm`, `stck_prpr`, `prdy_ctrt`, `acml_vol`.

### 5.2 `FHPST01710000` — 거래량/거래대금 순위 (Volume/Trade-Amount Ranking, `FID_BLNG_CLS_CODE`로 정렬 기준 선택)
> [!IMPORTANT]
> 실측 정정: 이 화면은 하나의 TR로 여러 정렬 기준을 제공하는 템플릿이다.
> 거래대금순위는 아래처럼 `FID_BLNG_CLS_CODE="3"`(거래금액순)으로 조회한다 (`FID_BLNG_CLS_CODE="0"`은 평균거래량순).
* **Path:** `GET /uapi/domestic-stock/v1/quotations/volume-rank`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`), `FID_COND_SCR_DIV_CODE` (`20171`), `FID_INPUT_ISCD` (`0000`), `FID_DIV_CLS_CODE` (`0`), `FID_BLNG_CLS_CODE` (`0`: 평균거래량, `3`: 거래금액순), `FID_TRGT_CLS_CODE` (`0000000000`), `FID_TRGT_EXLS_CLS_CODE` (`0000001100`: ETF·ETN 제외), `FID_INPUT_PRICE_1` (`""`), `FID_INPUT_PRICE_2` (`""`), `FID_VOL_CNT` (`""`), `FID_INPUT_DATE_1` (`""`).
* **Output (`output`):** `mksc_shrn_iscd` (종목코드), `hts_kor_isnm`, `stck_prpr`, `prdy_ctrt`, `acml_vol`, `acml_tr_pbmn` (누적거래대금).

### 5.4 `FHKST01010600` — 시가총액 순위 (Market Cap Ranking)
* **Path:** `GET /uapi/domestic-stock/v1/ranking/market-cap`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`), `FID_COND_SCR_DIV_CODE` (`20160`), `FID_DIV_CLS_CODE` (`0`), `FID_INPUT_ISCD` (`0000`), `FID_TRGT_CLS_CODE` (`0000000000`), `FID_TRGT_EXLS_CLS_CODE` (`0000000000`), `FID_INPUT_PRICE_1` (`""`), `FID_INPUT_PRICE_2` (`""`), `FID_VOL_CNT` (`""`).
* **Output (`output`):** `stck_shrn_iscd`, `hts_kor_isnm`, `stck_prpr`, `stck_avls` (시가총액 억원).

### 5.5 `FHPST01740000` — 신고가 / 신저가 순위 (New High/Low Ranking)
* **Path:** `GET /uapi/domestic-stock/v1/ranking/new-high-low`
* **Params:** `FID_COND_MRKT_DIV_CODE` (`J`), `FID_COND_SCR_DIV_CODE` (`20174`), `FID_INPUT_ISCD` (`0000`), `FID_DIV_CLS_CODE` (`0`: 신고가, `1`: 신저가).
* **Output (`output`):** `stck_shrn_iscd`, `hts_kor_isnm`, `stck_prpr`, `prdy_ctrt`, `high_date`, `new_prc`.

### 5.6 `HHKST03900300` & `HHKST03900400` — HTS 조건검색 (Condition Search)
* **Title 목록:** `GET /uapi/domestic-stock/v1/quotations/psearch-title` (`HHKST03900300`, `user_id=<HTS_ID>`).
* **결과 조회:** `GET /uapi/domestic-stock/v1/quotations/psearch-result` (`HHKST03900400`, `user_id=<HTS_ID>`, `seq=<SEQ>`).

---

## 6. Order & Account Management TRs (주문 / 계좌 — KIS 전용)

> [!IMPORTANT]
> 주문 및 계좌 관련 TR은 실전계좌(`TTTC...`)와 모의투자(`VTTC...`)에서 서로 다른 `tr_id`를 사용합니다.

### 6.1 `TTTC0012U` / `VTTC0012U` — 주식 현금 매수 주문 (Buy Order)
> [!NOTE]
> 공식 저장소(`koreainvestment/open-trading-api`) 기준 현행 TR 규격은 `TTTC0012U`/`TTTC0011U`/`TTTC0013U`/`TTTC0081R`을 사용한다.
* **Path:** `POST /uapi/domestic-stock/v1/trading/order-cash`
* **Headers:** `tr_id: "TTTC0012U"` (실전) / `"VTTC0012U"` (모의투자), `custtype: "P"`
* **Body:**
  ```json
  {
    "CANO": "<계좌번호 앞8자리>",
    "ACNT_PRDT_CD": "<계좌상품코드 뒤2자리 (보통 01)>",
    "PDNO": "005930",
    "ORD_DVSN": "00",
    "ORD_QTY": "10",
    "ORD_UNPR": "70000",
    "EXCG_ID_DVSN_CD": "KRX",
    "SLL_TYPE": "",
    "CNDT_PRIC": ""
  }
  ```
* **Order Type (`ORD_DVSN`):**
  * `00`: 지정가 (Limit)
  * `01`: 시장가 (Market)
  * `02`: 조건부지정가
  * `03`: 최유리지정가
  * `04`: 최우선지정가
  * `05`: 장전 시간외종가 (08:30~08:40)
  * `06`: 장후 시간외종가 (15:40~16:00)
  * `07`: KRX 2026-09-14 세션 개편으로 대상 세션(16:00~18:00 단일가) 소멸 — 사용 금지.
* **Response Body (`output`):**
  * `KRX_FWDG_ORD_ORGNO`: 한국거래소 전송 주문조직번호
  * `ODNO`: 주문번호 (Order Number)
  * `ORD_TMD`: 주문시각 (`HHMMSS`)

### 6.2 `TTTC0011U` / `VTTC0011U` — 주식 현금 매도 주문 (Sell Order)
* **Path:** `POST /uapi/domestic-stock/v1/trading/order-cash`
* **Headers:** `tr_id: "TTTC0011U"` (실전) / `"VTTC0011U"` (모의투자), `custtype: "P"`
* **Body:** `CANO`, `ACNT_PRDT_CD`, `PDNO`, `ORD_DVSN`, `ORD_QTY`, `ORD_UNPR`, `EXCG_ID_DVSN_CD`, `SLL_TYPE`(`"01"` 일반매도), `CNDT_PRIC`
* **Response Body (`output`):** `ODNO` (주문번호), `ORD_TMD`.

### 6.3 `TTTC0013U` / `VTTC0013U` — 주식 정정 / 취소 주문 (Modify / Cancel)
* **Path:** `POST /uapi/domestic-stock/v1/trading/order-rvsecncl`
* **Headers:** `tr_id: "TTTC0013U"` (실전) / `"VTTC0013U"` (모의투자)
* **Body:**
  ```json
  {
    "CANO": "<계좌번호 8자리>",
    "ACNT_PRDT_CD": "01",
    "KRX_FWDG_ORD_ORGNO": "<원주문조직번호>",
    "ORGN_ODNO": "<원주문번호>",
    "ORD_DVSN": "00",
    "RVSE_CNCL_DVSN_CD": "01",
    "ORD_QTY": "10",
    "ORD_UNPR": "71000",
    "QTY_ALL_ORD_YN": "N"
  }
  ```
* **Parameters:**
  * `RVSE_CNCL_DVSN_CD`: `"01"` (정정), `"02"` (취소)
  * `QTY_ALL_ORD_YN`: `"Y"` (잔량전부), `"N"` (일부수량)

### 6.4 `TTTC8434R` / `VTTC8434R` — 주식 잔고 조회 (Account Positions)
* **Path:** `GET /uapi/domestic-stock/v1/trading/inquire-balance`
* **Headers:** `tr_id: "TTTC8434R"` (실전) / `"VTTC8434R"` (모의투자)
* **Query Params:**
  * `CANO`: 계좌번호 앞8자리
  * `ACNT_PRDT_CD`: `"01"`
  * `AFHR_FLPR_YN`: `"N"`
  * `OFL_YN`: `""`
  * `INQR_DVSN`: `"02"` (종목별)
  * `UNPR_DVSN`: `"01"`
  * `FUND_STTL_ICLD_YN`: `"N"`
  * `FNCG_AMT_AUTO_RDPT_YN`: `"N"`
  * `PRCS_DVSN`: `"00"` (전일매매포함)
  * `CTX_AREA_FK100`: `""` (연속조회키)
  * `CTX_AREA_NK100`: `""`
* **Response Body Split:**
  * `output1` (보유종목 리스트):
    * `pdno`: 종목코드
    * `prdt_name`: 종목명
    * `hldg_qty`: 보유수량
    * `ord_psbl_qty`: 매도가능수량
    * `pchs_avg_pric`: 매입평균가격
    * `pchs_amt`: 매입금액
    * `now_pric`: 현재가
    * `evlu_amt`: 평가금액
    * `evlu_pfls_amt`: 평가손익금액
    * `evlu_pfls_rt`: 평가손익율 (%)
  * `output2` (계좌 총평가 요약):
    * `tot_evlu_amt`: 총평가금액
    * `nass_amt`: 순자산금액
    * `pchs_amt_smtl_amt`: 매입금액합계
    * `evlu_amt_smtl_amt`: 평가금액합계
    * `evlu_pfls_smtl_amt`: 평가손익합계

### 6.5 `TTTC0081R` / `VTTC0081R` — 주식 체결 / 미체결 내역 조회 (Fills & Open Orders, 3개월 이내)
* **Path:** `GET /uapi/domestic-stock/v1/trading/inquire-daily-ccld`
* **Headers:** `tr_id: "TTTC0081R"` (실전) / `"VTTC0081R"` (모의투자) — 3개월 이전 조회는 `CTSC9215R`/`VTSC9215R`
* **Query Params:**
  * `CANO`, `ACNT_PRDT_CD`
  * `INQR_STRT_DT`: 조회시작일 (`YYYYMMDD`)
  * `INQR_END_DT`: 조회종료일 (`YYYYMMDD`)
  * `SLL_BUY_DVSN_CD`: `"00"` (전체), `"01"` (매도), `"02"` (매수)
  * `INQR_DVSN`: `"00"` (역순)
  * `PDNO`: 종목코드 (`""` 전체)
  * `CCLD_DVSN`: `"00"` (전체), `"01"` (체결), `"02"` (미체결)
  * `ORD_GNO_BRNO`: `""`
  * `ODNO`: `""`
  * `CTX_AREA_FK100`: `""`
  * `CTX_AREA_NK100`: `""`
* **Output (`output1`):** `odno` (주문번호), `orgn_odno` (원주문번호), `ord_tmd` (주문시각), `sll_buy_dvsn_cd_name` (매매구분), `pdno`, `ord_qty` (주문수량), `ord_unpr` (주문단가), `tot_ccld_qty` (총체결수량), `tot_ccld_amt` (총체결금액), `rmn_qty` (미체결잔량).

### 6.6 `TTTC8908R` / `VTTC8908R` — 예수금 / 주문가능현금 조회 (Buying Power)
* **Path:** `GET /uapi/domestic-stock/v1/trading/inquire-psbl-order`
* **Headers:** `tr_id: "TTTC8908R"` (실전) / `"VTTC8908R"` (모의투자)
* **Query Params:** `CANO`, `ACNT_PRDT_CD`, `PDNO`, `ORD_UNPR`, `ORD_DVSN`, `CMA_EVLU_AMT_ICLD_YN`, `OVRS_ICLD_YN`.
* **Output (`output`):**
  * `ord_psbl_cash`: 주문가능현금
  * `nrcv_buy_amt`: 미수없는매수금액
  * `max_buy_amt`: 최대매수금액
  * `dnca_tot_amt`: 예수금총금액 (D+2 예수금)

> [!NOTE]
> 주문/계좌 TR 전송 시 실측된 주문 거부 코드: `EGW02006` (주문 거부, `REJECTED`), `EGW00121`/`EGW00123` (토큰 무효, 재발급 후 1회 재시도),
> `EGW00201` (초당 거래건수 초과 — 전송은 HTTP 500 본문의 `msg_cd`로 도착하므로 상태코드가 아닌 본문으로 판정한다).
