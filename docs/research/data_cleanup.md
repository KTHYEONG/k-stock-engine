# `data/` 정리 계획

기준일: 2026-09-21. **삭제 실행 전 계획**이다. 현재 `data/`는 약 21GiB이며, `data/bronze/stocks` 원본이 약 17.3GiB를 차지한다. 원본은 `kr_equity_v1` Scope Bronze로 검증 리베이스한 뒤 기존 `bronze/stocks`를 제거한다. `archive`, `artifacts`, 기존 Silver·Gold도 새 Scope release와 coverage 검증 뒤 함께 제거한다. `du`의 디스크 사용량은 파일 크기 합계보다 클 수 있다.

## 분류

| 구간 | 현재 규모 | 처분 계획 | 선행 확인 |
| --- | ---: | --- | --- |
| `data/bronze/stocks/*`의 공급처 원본 | 약 17.3GiB, 29.8만 파일 | **유지**. DART, KRX, 수급의 원본 바이트·receipt·해시를 새 기준 데이터 재구축에 사용 | 원본과 파생 집계 응답을 구분하고 해시·누락 검사 |
| `data/bronze/stocks` | 약 17.3GiB | Scope Bronze 리베이스 뒤 **제거** | receipt catalog와 필수 원천 coverage가 Scope hash에 일치 |
| `data/archive`, `data/artifacts` | 약 0.90GiB | **전체 제거** | 리베이스 report·새 state·quota가 `data/state/<scope>`에 존재 |
| `data/silver/stocks`, 기타 기존 Silver | 약 1.3GiB | **제거** | 새 Scope Silver release가 source dataset ID와 coverage hash를 보유 |
| `data/gold/stocks`, 기타 기존 Gold | 약 0.12GiB | **제거** | 새 Scope Gold release가 universe·feature·coverage hash를 보유 |

기존 artifact JSON의 참조는 보존 근거가 아니다. 삭제 전에는 경로 이름으로 추정하지 않고 Scope receipt catalog, coverage report, Silver·Gold release metadata를 검증한다.

## 실행 순서와 차단 조건

1. 모든 `data/` 파일에 대해 경로·크기·SHA256·원본/파생/체크포인트·참조 실행 ID 목록을 **읽기 전용**으로 작성한다. Bronze receipt의 `content_hash`와 실제 바이트 해시 불일치가 있으면 정리를 중단한다.
2. 수집 프로세스가 없고 DART·수급 재개 상태가 저장됐는지 확인한다. 미완료 수집의 quota/checkpoint/plan을 지우지 않는다.
3. 새 기준 원본 해시 집합과 정규화 데이터셋이 확정된 뒤 `archive`와 날짜가 붙은 옛 Silver/Gold/연구 artifact를 묶음별로 제거한다. 첫 묶음은 기존 보수적 검사에서 회수 가능하다고 나온 루트와 고아 staging이다.
4. 묶음마다 삭제 예정 목록과 회수 바이트를 검토하고, 실제 제거 후 남은 manifest·원본 해시·새 데이터셋 참조가 깨지지 않았는지 확인한다. 문제가 있으면 해당 묶음의 보관본으로 복원한다.
5. Bronze의 `aggregated:`/`manifest:` 같은 **재생성 가능한 파생 응답**은 원본 receipt가 남아 있고 도달성 검사를 통과한 경우에만 정리한다. 현재 `gc_superseded_bronze_aggregates()`는 오래된 집계본만 정리하는 안전한 출발점이다. 29.8만 원본 파일은 경로 계약을 변경하기 전 임의로 묶거나 지우지 않는다.

## 이번 재구축에 맞춘 실행 순서

1. `rebase-2019 --dry-run`으로 보존·거절 receipt를 확정한다.
2. 비 dry-run 리베이스로 Scope Bronze catalog와 필수 원천 coverage를 만든다.
3. 보통주 유니버스·가격 Silver와 DART 재무 Silver를 만들고 Scope-bound Gold metadata에 coverage hash를 기록한다.
4. `remove-legacy-data` 계획 모드의 검증 record를 확인한다.
5. `remove-legacy-data --apply`로 `archive`, `artifacts`, `bronze/stocks`, `silver/stocks`, `gold/stocks`를 제거한다.

목표는 Scope-bound 증거·파생 release·실행 기록만 남기고, 기존 실행 잔재를 완전히 제거하는 것이다.
