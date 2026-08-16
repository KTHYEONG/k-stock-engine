# Replay 결측 exit 포지션 처리안

## 확인 결과

최신 replay artifact `net_alpha_20260816_tradability_cost_run2`의
`policy_frontier`에서 실제 선택된 `entry=FILLED` 및 `exit=NO_BAR` 포지션을
분리했다. 고유 포지션은 8개이며, `KRX:057880`은 두 정책 프로필에 중복되어
기록상 9건이다. 모든 건의 상태는 `MISSING_EXIT_PRICE`, 해석은
`CONFIRMED_NO_BAR`이다. 즉 단순한 로컬 캐시 누락이 아니라 예정 exit일의
공식 KRX 가격 bar가 존재하지 않는다.

| 심볼 | 결정일 | 진입일 | 예정 exit일 | 확인된 상황 |
|---|---|---|---|---|
| KRX:025320 | 2019-09-02 | 2019-09-03 | 2019-09-06 | 상장적격성 심사 관련 거래정지/상장 이벤트 |
| KRX:049180 | 2019-11-11 | 2019-11-12 | 2019-11-15 | 상장적격성 심사 관련 거래정지 |
| KRX:019170 | 2020-04-01 | 2020-04-02 | 2020-04-07 | KRX bar 없음, 연결된 DART 원인 증거 없음 |
| KRX:008060 | 2020-04-22 | 2020-04-23 | 2020-04-28 | KRX bar 없음, 현재 연결된 공시 증거 없음 |
| KRX:057880 | 2022-03-18 | 2022-03-21 | 2022-03-24 | 풍문·보도 관련 거래정지 |
| KRX:024810 | 2023-05-04 | 2023-05-08 | 2023-05-11 | 풍문·보도 관련 거래정지 |
| KRX:001340 | 2023-07-12 | 2023-07-13 | 2023-07-18 | 거래정지 및 상장적격성 심사 |
| KRX:012030 | 2023-08-10 | 2023-08-11 | 2023-08-17 | 우회상장 심사 관련 거래정지 |

근거 artifact: `data/artifacts/stocks/net_alpha_20260816_tradability_cost_run2/metrics.json`.
연결된 공시 evidence: `data/evidence/stocks/tradability_events_20260816_v2.json`.

연결된 6개 종목의 공시일은 모두 결정일과 진입일 이후, 예정 exit일 이전이다.
따라서 공시가 나중에 발생했다는 사실을 이용해 과거 진입을 소급 제거하면
미래정보 누수가 발생한다. 현재 `selected-exit-unresolved:21/34`에는 이 결측
exit뿐 아니라 `MISSING_DECISION_INPUT`도 포함된다. 최신 projection의 차단
레코드는 총 55개이며, 결측 exit는 9개, 나머지 46개는 별도의 의사결정 입력
결측이다.

## 최종 처리 원칙

1. **결정 시점 사전 차단**
   - KRX/KIND의 공식 공시와 거래상태를 `publication_timestamp` 단위로 수집한다.
   - 결정 시점에 이미 공개된 거래정지·상장폐지·상장적격성 위험만 매매 후보에서 제외한다.
   - DART 접수일만 사용할 때는 장중/장후를 알 수 없으므로, 보수적으로 다음
     거래 세션부터 효력이 발생하도록 처리한다.

2. **진입 후 공시 발생**
   - 예정 exit일을 그대로 사용하지 않는다.
   - 공시를 알 수 있었던 시점 이후 최초의 실제 거래 가능 시가를
     `EVENT_DRIVEN_EXIT`로 기록하고, 기존 예정 exit와 구분한다.
   - 거래가 재개되지 않으면 임의의 가격이나 마지막 종가를 exit 가격으로
     대체하지 않는다.

3. **거래 재개·결제 처리**
   - 공식 현금청산액, 합병대가, 주식교부 비율이 있으면 공식 조건으로
     `SETTLED_CASH`를 계산한다.
   - 예를 들어 `KRX:019170`은 2020-04-08 다음 거래일 시가가 확인되므로,
     사전에 정의된 deferred-exit 정책에 따라 검증 후 처리할 수 있다.
   - `KRX:008060`처럼 기업행사 조건이 확인되지 않은 경우에는 KIND/거래소
     원문과 조정계수를 확인하기 전까지 가격을 추정하지 않는다.

4. **재현·학습 게이트**
   - `REALIZED`, 검증된 `EVENT_DRIVEN_EXIT`, 검증된 `SETTLED_CASH`만 수익률
     계산에 포함한다.
   - 실제 청산가격을 재현할 수 없는 포지션은 `UNEXECUTABLE_EXIT`로 보존하고,
     학습 label과 자산운용 후보 승격에서 제외한다.
   - 미래의 `MISSING_EXIT_PRICE` 상태를 이용해 과거 주문을 제거하지 않는다.
     해당 포지션은 당시 정보집합으로 주문을 생성한 뒤, event-driven 또는
     unexecutable 결과로 replay한다.
   - `MISSING_DECISION_INPUT`은 exit 결측과 별도 coverage 오류로 집계하고,
     입력 backfill 또는 후보 제외 규칙을 별도로 적용한다.

## 구현 순서

1. KRX/KIND 원문에서 공시 시각, 거래정지 시작·해제, 상장폐지·결제 조건을
   backfill하고 각 record의 source hash와 관측 시각을 저장한다.
2. `first_actionable_session`을 산출해 결정 시점과 진입 시점의 causality를
   검증한다. 같은 날짜의 공시는 시각이 없으면 다음 세션 적용으로 고정한다.
3. replay에 `EVENT_DRIVEN_EXIT`와 `SETTLED_CASH` 경로를 추가하고, 실제
   execution bar가 없을 때 `UNEXECUTABLE_EXIT`로 fail-closed한다.
4. policy frontier에 `horizon`, `profile_id`, `resolution_kind`,
   `actionable_session`을 함께 기록해 차단 원인을 후보별로 분리한다.
5. snapshot을 재생성한 뒤, 학습·백테스트 readiness가 통과하더라도 후보가
   `UNEXECUTABLE_EXIT`를 포함하면 승격하지 않는지 검증한다.

이 원칙을 적용하기 전에는 결측 exit 포지션을 임의 가격으로 backfill하거나,
공시 이후 사실을 이용해 진입을 제거해서는 안 된다. 두 방식 모두 실현 가능한
자산운용 성과와 정확한 historical replay를 왜곡한다.
