# `data/` 정리 계획

기준일: 2026-09-20. **삭제 실행 전 계획**이다. 현재 약 21GiB 중 `data/bronze/stocks`의 원본 약 17.3GiB·29.8만 파일은 새 데이터셋 재구축의 원천이다. `data/archive` 약 0.60GiB, Silver 약 1.2GiB, Gold 약 0.12GiB, Artifacts 약 0.30GiB의 과거 실행·파생물이 주요 정리 대상이다. `du`의 디스크 사용량은 파일 크기 합계보다 클 수 있다.

## 분류

| 구간 | 현재 규모 | 처분 계획 | 선행 확인 |
| --- | ---: | --- | --- |
| `data/bronze/stocks/*`의 공급처 원본 | 약 17.3GiB, 29.8만 파일 | **유지**. DART, KRX, 수급의 원본 바이트·receipt·해시를 새 기준 데이터 재구축에 사용 | 원본과 파생 집계 응답을 구분하고 해시·누락 검사 |
| `data/bronze/artifacts` | 약 13MiB | 수집 원장 연결 확인 전 유지 | 원본 receipt 참조 및 활성 실행 확인 |
| `data/artifacts/quota`, `collection-plans`, `collection-checkpoints`, `checkpoints`, `dart_backfill` | 합계 약 73MiB | DART·수급 백필 완료까지 유지 | 재시도·쿼터·완료 증명의 마지막 사용시각 |
| `data/archive/*` | 약 0.60GiB | Bronze 보존과 새 정규화 대조가 끝나면 **전체 제거 후보** | 보관본에만 있는 원본·미이관 파일의 SHA256 목록이 0건 |
| `data/silver/stocks_prepared_*`, `stocks_refresh_*`, `stocks_provenance_*` | 약 0.64GiB | 새 시점별 기준 데이터셋이 생성되면 **전체 제거 후보** | 새 데이터 품질 보고서와 옛 실행의 필요한 감사 증거 분리 보존 |
| `data/gold/stocks_prepared_*`, `stocks_research_*` | 약 0.06GiB | 기존 전략 폐기와 함께 **전체 제거 후보** | 옛 Gold를 참조하는 백테스트·보고서만 남기지 않음 |
| `data/artifacts/streaming_staging`, `normalization_rebuild`, 옛 `gold_*`·`backtest_*` | 약 0.2GiB 이상 | 완료·중단된 실행의 **제거 후보** | 실행 중인 프로세스 부재, 필요한 결정·해시 요약만 별도 보존 |
| `data/silver/stocks`, `data/gold/stocks` | 약 0.58GiB | 새 자료로 대체 후 옛 dataset ID별 선택 제거 | 활성 manifest·새 백테스트가 참조하는 ID의 도달성 검사 |

현재 `plan_storage_root_retention()`의 보수적 검사에서는 Silver 루트 6개, Gold 루트 2개와 Gold의 고아 `.staging-*` 1개가 바로 회수 가능한 후보로 나온다. 나머지 옛 루트 상당수는 **옛 artifact JSON에 이름이 등장한다는 이유**로만 보존 판정된다. 옛 연구 실행을 폐기할 때는 해당 artifact도 함께 정리한 뒤 재검사한다. 단순히 루트 이름만 보고 삭제하지 않는다.

1차 회수 후보의 정확한 루트 이름은 Silver의 `stocks_prepared_20260910_v2`~`v6`, `stocks_refresh_20260911`, Gold의 `stocks_research_2017`, `stocks_research_2017_v2`와 `data/gold/stocks/.staging-d8a2c2fbc08af8d2cd0cf5c92e43d075-6ca26589`이다. 마지막 staging은 완료된 데이터셋 manifest가 없는 임시 디렉터리인지 확인한 뒤 제거한다.

## 실행 순서와 차단 조건

1. 모든 `data/` 파일에 대해 경로·크기·SHA256·원본/파생/체크포인트·참조 실행 ID 목록을 **읽기 전용**으로 작성한다. Bronze receipt의 `content_hash`와 실제 바이트 해시 불일치가 있으면 정리를 중단한다.
2. 수집 프로세스가 없고 DART·수급 재개 상태가 저장됐는지 확인한다. 미완료 수집의 quota/checkpoint/plan을 지우지 않는다.
3. 새 기준 원본 해시 집합과 정규화 데이터셋이 확정된 뒤 `archive`와 날짜가 붙은 옛 Silver/Gold/연구 artifact를 묶음별로 제거한다. 첫 묶음은 기존 보수적 검사에서 회수 가능하다고 나온 루트와 고아 staging이다.
4. 묶음마다 삭제 예정 목록과 회수 바이트를 검토하고, 실제 제거 후 남은 manifest·원본 해시·새 데이터셋 참조가 깨지지 않았는지 확인한다. 문제가 있으면 해당 묶음의 보관본으로 복원한다.
5. Bronze의 `aggregated:`/`manifest:` 같은 **재생성 가능한 파생 응답**은 원본 receipt가 남아 있고 도달성 검사를 통과한 경우에만 정리한다. 현재 `gc_superseded_bronze_aggregates()`는 오래된 집계본만 정리하는 안전한 출발점이다. 29.8만 원본 파일은 경로 계약을 변경하기 전 임의로 묶거나 지우지 않는다.

목표는 먼저 약 **1~2GiB의 옛 파생·아카이브·임시 파일**을 회수하고, 그다음 원본 파일 수가 많은 문제를 별도 보존 형식으로 다루는 것이다. 압축 묶음으로 전환할 경우 개별 원본 해시 조회와 receipt 재생, 중단 후 재개가 동등하게 작동해야 한다.
