# LS Securities (LS) Integration Specification

> **AI Agent & Operational Technical Reference:**  
> Machine-readable architectural specification of LS Securities (구 이베스트투자증권) OpenAPI integration for K-Stock Engine. Covers token authentication, the 1.05-second process mutex lock, investor flow harvesting, and daily OHLCV bar processing.

---

## 1. System & Authentication Overview

### 1.1 Base Endpoints
* **Live Base URL:** `https://openapi.ls-sec.co.kr:8080`
* **Paper Base URL:** `https://openapi.ls-sec.co.kr:29443`

### 1.2 OAuth2 Token Lifecycle (`POST /oauth2/token`)
* **Endpoint:** `POST /oauth2/token`
* **Headers:** `content-type: application/x-www-form-urlencoded`
* **Body:**
  ```text
  grant_type=client_credentials&appkey=<LS_APP_KEY>&appsecretkey=<LS_APP_SECRET>&scope=oob
  ```
* **Response Contract:**
  ```json
  {
    "access_token": "...",
    "token_type": "Bearer",
    "expires_in": 86400,
    "scope": "oob"
  }
  ```
* **Lifecycle Rules:**
  - 토큰 유효기간 24시간(`86400s`).
  - 프로세스 내 메모리 캐시 및 갱신 락을 운용하며, 만료 10분 전 자동 갱신.

---

## 2. Pacing & Concurrency Guard: Strict 1.05-Second Mutex Lock

LS증권 게이트웨이는 매우 엄격한 요청 빈도 제한을 강제합니다. 초당 1회를 초과하거나 동시 다중 요청(Concurrency > 1)이 유입되면 즉시 `rsp_cd: IGW00201` (초당 거래건수 초과) 오류를 반환하고 세션을 일시 차단합니다.

* **In-Process Mutex**: 모든 LS OpenAPI 요청은 `min_interval = 1.05s` 상호 배제 락을 거쳐 전송됩니다.
* **Effective Throughput**: 최대 ~0.95 req/s.
* **Throttling Detection**:
  - 응답 본문 `rsp_cd == "IGW00201"` 또는 HTTP 429 수신 시 지수 백오프(`Exponential Backoff`, 기본 1.5s -> 3.0s -> 6.0s) 후 최대 3회 재시도.

---

## 3. Supported Transaction Records (TR Code Reference)

| TR Code | 서비스명 | 호출 용도 | 처리 규격 및 특이사항 |
| :--- | :--- | :--- | :--- |
| **`t1405`** | 투자자별 매매종합 | 일별 기관/외인/개인 순매수 수급 | K-Stock Engine 투자자 수급의 **제1 주력 원천** (13개 투자 주체별 순매수 집계) |
| **`t8412`** | 주식 차트(일봉) | 일별 OHLCV 시세 | 전 종목 일별 수정주가 및 거래량 보조 검증용 |
| **`t8407`** | 다종목 시세 조회 | 복수 종목 시세 일괄 조회 | 1회 호출당 최대 50종목 묶음 조회 지원 |
| **`t1101`** | 주식 호가 | 정규장 10단계 호가 | 호가 잔량 및 스프레드 보조 확인 |

---

## 4. Investor Flow Ingestion Protocol (`ls_investor_flow`)

K-Stock Engine의 야간 수집 파이프라인(`kse-collect-eve.service`, 18:30 KST)에서 호출됩니다:

```bash
uv run python -m src.data.cli collect-ls-investor-flow
```

* **수집 대상**: 당일 거래된 전 상장 종목의 투자자별 매매 동향.
* **저장 위치**: `data/bronze/ls/investor_flow/dt=YYYY-MM-DD.jsonl`
* **Silver 통합 (`investor_flow_union`)**:
  - LS증권 수급 데이터가 누락되거나 통신 실패 시, KIS 수급 데이터(`collect-kis-investor-flow`)가 대체/보완 원천으로 작동하여 단일 Silver 데이터셋(`investor_flow_union`)으로 정규화.
