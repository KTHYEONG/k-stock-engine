# 리팩토링·수집 효율화 로드맵

기준일: 2026-09-24. 목적은 AI 작업 효율, 확장성, 유지보수성이다. 이 문서는 조사 결과와 단계 계획을 담고, 각 단계의 실행 계약은 `/spec`에서 따로 만든다.

## 1. 기준선 (측정값)

| 항목 | 값 |
| --- | --- |
| `src` | 134개 파일, 38.8k줄 (`src/data`만 56개 파일, 25.1k줄) |
| `tests` | 35.8k줄, 전체 스위트 37초, 기존 실패 5건 (LS share-unit fixture) |
| `legacy/` | 127k줄. `.gitignore`에 있지만 436개 파일이 추적 중이고, 어디서도 import하지 않음 |
| 거대 함수 | `cli.main` 1,072줄, `_dispatch_backtest` 491줄, `_parse_args` 326줄. 100줄 초과 함수 61개 |
| CLI | `if args.command ==` 분기 48개. 이 중 26개는 이미 삭제한 `data/*/stocks` 경로를 기본값으로 가짐 |
| 의존성 | 선언 23개 중 약 15개는 import 0건 (pandas, scipy, lightgbm, pykrx, gspread, aiohttp, numba 등) |
| 커버리지 예외 | `pragma: no cover` 170곳. 상당수가 fail-closed 도메인 가드에 붙어 있음 |
| Bronze | 21GB, 약 17만 페이지 디렉터리 (`investor_flow` 74k, `financial_facts` 67k) |
| 카탈로그 | publish마다 7.9만 항목을 담은 37MB 전체 스냅샷 기록 (리비전 18개, 610MB). `investor_flow`, `corporate_actions`, `industry`, `calendar`는 색인하지 않음 |

## 2. P0 정확성 결함 (구조 변경보다 먼저)

1. **재무 fact PIT 누수 ①: 가용시각 대체**
   - 의도한 정책은 "공시일 다음 KRX 세션 09:00"이다.
   - 실제로는 캘린더가 비면(`incremental_normalization.py:126-130`) 또는 다음 세션이 판단시각 이후이면(`normalization.py:176-182`) 공시일 00:00 KST로 조용히 대체한다.
   - 디스크의 641,094행 전부 `available_at == published_at`이다.
2. **재무 fact PIT 누수 ②: 정정값이 원공시일로 기록**
   - `fnlttSinglAcntAll`은 최신 정정본을 반환한다. standardized 페이지 표본의 2.7%(78/2,852)에서 레코드 `rcept_no`가 identity 공시일보다 늦다. 예: 2024Q3 원공시 2024-11-13 → 2026-07-01 정정값.
   - Silver는 이 정정값을 원공시일에 가용한 것으로 기록한다.
   - 수정: `available_at` = max(identity 공시일, 레코드 `rcept_no` 일자)의 다음 세션 09:00. 캘린더가 없으면 fail-closed.
   - Bronze에 `rcept_no`가 보존돼 있으므로 재수집 없이 Silver만 재빌드하면 된다.
3. **DART 쿼터 우회**
   - `document.xml`과 `corpCode.xml`이 쿼터 원장과 호출 간격 제어를 거치지 않는다 (`dart/client.py:344-379`).
   - 재시도가 중첩돼(클라이언트 3회 × xbrl 3회) 논리 요청 1건이 HTTP 최대 9회가 된다.
   - 공유 키 보호 조건을 깨므로, DART 재개 전에 반드시 수정한다.
4. **LS flow Silver 재빌드 불가**
   - KIS 페이지가 같은 `bronze/investor_flow/`를 쓰는데 `_parse_page`에 provider 필터가 없다 (`investor_flow_silver.py:149-207`).
5. **저장 위생**
   - DART 기업코드 Bronze에 `receipt.json`이 없다.
   - 리포트 2,293개가 스코프 밖(`data/bronze/artifacts/collections`)에 `{content_hash}.json`로 쌓이고, 빈 배치끼리 덮어쓴다.
   - 테스트가 상대 기본경로로 실제 `data/artifacts/`에 기록한다.
   - 카탈로그 `publish`에 락이 없다.
   - `collect_dart_disclosures`가 `TypeError`가 나면 필터 없는 전 시장 조회로 조용히 대체한다.
6. **Lineage 끊김**
   - LS flow(`investor_flow_32f16…`)는 삭제된 universe를 가리킨다.
   - `kis_supplement`와 `reference_benchmarks`는 삭제된 `market_panel_6c2a…`를 가리킨다.
   - `ordinary_universe`와 `financial_quality`는 이들을 재현할 CLI가 없다.
7. **구 백테스트 엔진 결함** (고치지 않고 폐기 대상)
   - 세율 0.23% 상수를 쓰고 틱 테이블이 하나뿐이다.
   - 가격제한폭을 무시한다.
   - T+1 체결용 ADTV에 체결일 거래량이 포함된다 (look-ahead).
   - 미체결 잔량이 유실된다.
   - 회계 항등식 검증이 동어반복이라 항상 통과한다.
   - **이 엔진의 결과는 무효로 간주한다.**

## 3. 구조 문제의 근본 원인

- **세대 중첩**
  - Silver/Gold가 두 세대다. 구 `EvidenceKind` 세대(약 7k줄)는 삭제된 루트만 읽고, `provider_version="fixture"` 때문에 자기 Gold로 연결되지도 못한다.
  - 현재 데이터를 만드는 것은 신 `<name>_<hash16>` 세대다.
  - 수집도 BronzeStore+체크포인트 경로와 scoped(카탈로그) 경로로 나뉘어 있다.
- **공통 추상화 부재**
  - "재개 가능한 수집 루프"가 7벌 있고, 완료 판정 방식이 6가지다.
  - 원자적 publish 8벌, 데이터셋 ID 계산 8벌, "universe 정확히 1개" 해석 5벌, `_fiscal_key` 7벌, 정수·날짜 파서 각 3벌이 있다.
  - HTTP·재시도·호출 간격·토큰 처리를 클라이언트 5개가 각자 구현한다.
- **CLI 비대화**
  - 레지스트리 없이 if-chain이다. 모든 명령이 strategy/engine을 import한다.
  - 백테스트 오케스트레이션(약 650줄)이 `src/data/cli.py` 안에 있다.
- **계층 역전**
  - 순환이 4쌍 있다: data↔strategy, data↔features, features↔strategy, engine↔validation.
  - 경계 테스트는 4개 패키지만 보고, 함수 내부의 lazy import를 놓친다.
- **이동을 막는 테스트**
  - `inspect.getsource`로 소스 문자열을 검사하는 테스트와 `cli` 속성을 monkeypatch하는 테스트가 파일을 옮기는 순간 깨진다.

## 4. 목표 구조

```
src/core/            time(O(1) 세션 인덱스), pit, instruments, fiscal
src/market/          krx_rules(틱·세율·가격제한, 날짜별), costs
src/integrations/    transport.py(공통 전송) + krx/ dart/ kis/ ls/ — 엔드포인트 래퍼만, Bronze 쓰기 금지
src/data/catalog/    receipt catalog v2 (append-only, 전 kind 색인)
src/data/ingest/     job.py(CollectionJob·run_job), report.py, jobs/<provider_dataset>.py
src/data/materialize/ contract·identity·manifest·publish·resolve·registry·lineage
src/data/silver/     ordinary_universe, daily_market, investor_flow/{ls,kis_supplement,union}, industry, financial_facts, financial_quality, corporate_actions
src/data/gold/       market_panel, reference_benchmarks
src/data/cli/        registry.py + 도메인별 명령 모듈 (lazy import)
src/backtest/        신규 엔진 (별도 로드맵 P0 스펙)
```

- **의존 방향**: core → market → integrations → data → backtest ← strategies. CLI만 여러 계층을 조립한다.
- **파일 예산**: 모듈 약 500줄 이하, 함수 약 80줄 이하.
  - ruff에 `C901`, `PLR0915`를 신규 코드 기준으로 켠다. 현재 mccabe 설정은 있으나 select에 빠져 있다.

## 5. 단계 계획 (각 단계는 독립 검증 가능)

| 단계 | 내용 | 검증 게이트 |
| --- | --- | --- |
| R0 기준선 | 미커밋 작업을 커밋하고 `archive/pre-refactor` 태그. 기존 실패 5건은 R2 삭제 대상인지 먼저 판정 | 전체 스위트 결과 고정 |
| R1 P0 정확성 | §2의 1~5 수정: fact 가용시각(fail-closed, rcept_no 반영), DART 쿼터 경유와 재시도 단일화, LS provider 필터, 리포트·경로·락 위생 | 경계 테스트(정정 공시, 빈 캘린더, 쿼터 계수), fact Silver 재빌드 후 `available_at > published_at` 100% |
| R2 삭제 | 죽은 CLI 26개, 구 Silver 세대(기업행위 resolver와 fact 경로는 보존), 마이그레이션 모듈, 구 백테스트 체인, 깨진 tools 4개, 미사용 의존성, 작업 트리의 `legacy/`. 보존 대상: `ledger` 기업행위 상태기계, `market_rules`, `CostSchedule`, fill 로직, 기업행위·lifecycle resolver | 스위트 통과, 삭제 심볼 grep 0건. 예상 −1.4~1.6만 줄(src의 35~40%) |
| R3 이동 준비 | AST 기반 import 경계 테스트(모든 패키지, lazy import 포함, 허용목록 축소), `getsource` 테스트를 행동 테스트로 교체, 공통 헬퍼 추출(fiscal, parsing, universe 해석) | 경계 테스트, 헬퍼 단위 테스트 |
| R4 전송 계층 | `ProviderTransport`: 토큰, RateLimiter, RetryPolicy(429/5xx, DART 800/900, KIS EGW00201, LS IGW00201만 재시도, 업무 오류는 재시도 안 함), 모든 호출이 쿼터 원장 경유. 어댑터는 raw row만 반환 | 가짜 HTTP로 재시도, 스로틀, 원장 계수 검증 |
| R5 카탈로그 v2 | append-only(SQLite 또는 delta 세그먼트) + 락 + 전 kind 백필(flow 74k, 기업행위 13k, 업종 2.5k, 캘린더). publish 비용 O(전체) → O(delta) | Bronze 디렉터리 수와 카탈로그 수 일치, 동시 publish 테스트 |
| R6 CollectionJob | `units / is_done(카탈로그) / fetch / to_payloads / request_cost` + `run_job`(서브배치 persist, 실패 격리, 헤드룸, 하트비트, `state_root` 아래 1회 리포트). KRX → DART facts·disclosures → KIS·LS flow → 업종 → 기업행위 순으로 이식하고 `tools/*_extension.py` 대체 | 가짜 어댑터로 재개, 격리, 헤드룸, 중단 복구 |
| R7 Materializer 계약 | `DatasetSpec` 레지스트리, `resolve_inputs`(명시 ID, 정확 prefix), ID는 파라미터·입력 ID·소스 digest로(출력 바이트 금지), 공통 publish, `verify-lineage`, 범용 `build <dataset>` | 8개 이식 시 ID·manifest 바이트 불변(union ID 규칙과 universe 멱등화만 예외로 명시) |
| R8 CLI 패키지 | `src/data/cli/` 레지스트리와 도메인 모듈, 단일 오류 계약, lazy import | parser 스냅샷(명령·플래그 불변) |
| R9 성능 | daily_market 벡터 파싱(588만 행 Python 루프 제거), flow 단일 스캔 파티션, KIS 페이지 카탈로그 선별, market_panel 버킷 1패스, 세션 조회 bisect/join_asof | 골든 ID 불변, 소요시간 비교 |
| R10 `pragma` 정리 | 도메인 가드의 `no cover`를 제거하고 시나리오 테스트 추가, 도달 불가 분기는 삭제 | diff-coverage |

신규 백테스트 엔진(정수 KRW 원장, 입금 흐름, KRX 규칙 단일 소스, 가격제한, 미체결 규칙)은 R2~R8 이후 별도 로드맵으로 진행한다.

## 6. API별 수집 전략 재검토

| Provider | 현재 | 결론 | 효과 | 주의 |
| --- | --- | --- | --- | --- |
| KRX Open API | 세션당 4회(1회 = 시장·일 전 종목) | 유지(이미 최적) | 2026 포워드 약 712회, 일일 4회 | 투자자별 데이터는 제공하지 않음 |
| DART 재무 `fnlttSinglAcntAll` | 키당 약 1.28회(CFS 77%·OFS 18%·문서 4.5%) | 유지 | 2016~2018 잔여 23,343키 ≈ 3만 회 ≈ 2일 | R1 쿼터 우회 수정이 선행 조건 |
| DART 일괄 TXT | 미사용 | **채택 안 함** | — | `rcept_no`·접수일이 없어 PIT 스탬프 불가, 정정 반영 재생성 스냅샷, 공식 프로그램 경로 없음. 교차검증 용도만 |
| DART `fnlttMultiAcnt` | 미사용 | 채택 안 함 | CFS 사전판정으로 약 4천 회(0.25일) 절감 가능 | CF 항목이 없어 필수 fact 불충분 |
| DART 공시목록 `list.json` | 기업별 2014~오늘. 종료일이 매일 바뀌어 캐시 무효 → 매 실행 전 기업 재조회(누적 97.5k회) | 전 시장 날짜창(≤3개월, 100건/페이지) + 증분 커서, 로컬에서 corp 필터 | 재수집 약 93%, 일일 증분 99% 이상 절감 | 상폐사 누락 방지를 위해 `corp_cls` 필터 금지, `last_reprt_at=N`으로 정정 이력 보존 |
| KIS `FHPTJ04160001` | 30행/회, 2015-11-23까지. 종목 내 호출 간격 제어 없음, 업무 오류도 재시도, 전 구간 요청(목표 대비 1.3배) | 공유 리미터(≤15rps, 주문 여유분 확보) + EGW00201 백오프 + 결측 구간만 요청 | 2026 포워드 약 1.65만 회/18분, 일일 2,750회/3분 | 20rps는 계좌 단위로 주문 트래픽과 공유 |
| LS `t1702` | ≤700세션/회, TPS 1 | 2019~2025 주 소스로 유지. 포워드 창도 LS로 연속성 확보 | 포워드 2,750회/48분 | 2019년 이전 이력 미검증(2016~2018은 KIS로 채움) |
| Kiwoom | 호출 간격 제어·재시도·상태 확인 없음, 스로틀 시 조용히 잘림 | 활성 경로에서 제거 | — | 현재 데이터 계보에 사용하지 않음 |
| KIS 업종 `CTPF1002R` | 종목당 1회(일회성) | 유지, 세션 재사용 | 영향 작음 | — |

- **수급 단위**: 현재 union은 순매수 **주식 수**로 일치한다(KIS supplement는 수량 필드 사용). 다만 KIS 어댑터 `_map_rows`는 **금액**을 내보내므로, R4에서 어댑터 출력을 raw로 통일한다.
- **문서 불일치**: `api_master.md`와 `broker_*.md`에 적힌 KIS 18rps 리미터와 Kiwoom 리미터는 코드에 없다. R4 이후 문서를 실제 구현에 맞춘다.

## 7. 결정 필요 사항

1. **삭제 방식**: 권장은 태그 후 git 삭제. `legacy/` 이동은 grep·인덱싱 노이즈를 남긴다.
2. **작업 트리의 `legacy/`(127k줄)**: 제거 권장. git 이력에 보존된다.
3. **DART 재개 시점**: 권장은 R1 직후, 고정 커밋의 git worktree에서 매일 실행. 쿼터는 일 단위라 기다린 날짜만큼 손실이고, 수집 코드가 R6에서 옮겨져도 영향이 없다.
4. **구 백테스트 엔진**: R2에서 삭제 권장. 결함이 있어 신 엔진 parity 기준으로 쓸 가치가 낮다.
