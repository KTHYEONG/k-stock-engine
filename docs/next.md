# 다음 작업

현재 Bronze → Silver 정규화와 기본 무결성 검증은 완료되었다. 다음 단계는 Silver를 백테스트가 직접 소비할 수 있는 Gold 입력으로 연결하는 것이다.

## 우선순위

1. **Historical universe 생성**
   - calendar·security_master·daily_market으로 검증일별 `U_t`를 산출한다.
   - `valid_from`, `valid_to`, `available_at` 기준을 적용해 PIT 누수를 차단한다.
   - 상장·상장폐지, 가격·거래대금·최소 이력 조건과 제외 사유를 함께 저장한다.

2. **2016년 검증 구간 적합성 검사**
   - 시작 전 최소 60거래일 warmup을 확인한다.
   - 종목별 daily bar 연속성, 결측, 중복, 비정상 OHLC를 검증한다.
   - 검증 결과를 재현 가능한 artifact와 manifest로 저장한다.

3. **DART fact 사용 가능성 판정**
   - 종목별 4개 연속 분기 fact와 필수 fact 세트를 점검한다.
   - 조건을 충족하지 못하는 종목은 조용히 보정하지 않고 사유별 제외한다.
   - action/status는 현재 `no_action` sentinel 상태이므로 실제 corporate action 데이터가 없으면 영향 종목·기간을 제외한다.

4. **Feature matrix 생성**
   - Gold universe를 기준으로 수급·재무·가격 feature를 PIT-safe lag로 결합한다.
   - 결측 처리, 최소 관측치, feature provenance를 기록한다.

5. **End-to-end 백테스트 실행**
   - 전략 입력, 체결, 수수료, 슬리피지, 리밸런싱을 연결한다.
   - 2016년 단일 종목 smoke test 후 전체 universe를 실행한다.
   - 성과, 거래 내역, 제외 사유, dataset/report hash를 run manifest에 남긴다.

6. **운영 정리 및 재현성 확인**
   - Gold·backtest 산출물의 immutable ID와 입력 hash를 고정한다.
   - 미사용 Bronze derived aggregate와 임시 artifact를 검증 후 정리한다.
   - 동일 입력 재실행 시 staging 재사용과 동일 report hash를 확인한다.

## 완료 기준

- 2016년 검증일별 `U_t`가 생성되고 표본 수·제외 사유가 설명 가능하다.
- 60거래일 warmup과 4개 연속 분기 fact 조건이 자동 검증된다.
- action/status 불충족 종목이 가격조정 없이 백테스트에 유입되지 않는다.
- 단일 종목 smoke test와 전체 백테스트가 CLI 한 번으로 재현된다.
