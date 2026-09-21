# `data/` 정리 계획

기준일: 2026-09-21. 정리는 완료됐다. 활성 Scope는 `kr_swing_2019_v1`이며, 원본은 약 11GiB, 보통주 유니버스 Silver는 약 59MiB다.

## 분류

| 구간 | 현재 규모 | 처분 계획 | 선행 확인 |
| --- | ---: | --- | --- |
| `data/bronze/stocks`, `data/bronze/artifacts` | 제거됨 | Scope Bronze로 대체 | rebase report와 catalog 검증 완료 |
| `data/archive`, `data/artifacts` | 제거됨 | Scope state로 대체 | DART quota 원장을 Scope state로 이관 |
| 기존 Silver·Gold 디렉터리 | 제거됨 | Scope Silver·Gold만 허용 | Scope ID 이외의 하위 디렉터리 제거 |

기존 artifact JSON의 참조는 보존 근거가 아니다. 삭제 전에는 경로 이름으로 추정하지 않고 Scope receipt catalog, coverage report, Silver·Gold release metadata를 검증한다.

## 남은 운영 규칙

1. 모든 신규 수집 원본은 `data/bronze/kr_swing_2019_v1`과 receipt catalog에만 기록한다.
2. quota·계획·감사 결과는 `data/state/kr_swing_2019_v1`에 기록한다.
3. Scope ID 밖의 Bronze·Silver·Gold 하위 디렉터리는 새 release를 검증한 후 제거한다.

## 이번 재구축에 맞춘 실행 순서

1. `rebase-2019 --dry-run`으로 보존·거절 receipt를 확정한다.
2. 비 dry-run 리베이스로 Scope Bronze catalog와 필수 원천 coverage를 만든다.
3. 보통주 유니버스·가격 Silver와 DART 재무 Silver를 만들고 Scope-bound Gold metadata에 coverage hash를 기록한다.
4. `remove-legacy-data` 계획 모드의 검증 record를 확인한다.
5. `remove-legacy-data --apply`로 `archive`, `artifacts`, `bronze/stocks`, `silver/stocks`, `gold/stocks`를 제거한다.

목표는 Scope-bound 증거·파생 release·실행 기록만 남기고, 기존 실행 잔재를 완전히 제거하는 것이다.
