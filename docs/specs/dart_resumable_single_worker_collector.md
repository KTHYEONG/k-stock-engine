# OpenDART 단일 worker 재개 수집기

## 목표

`list.json` 전체 기간 수집을 동시에 여러 달에 요청하지 않는다. 하나의 worker가 월을 순서대로, 페이지를 순서대로 수집하고, 각 성공 페이지를 즉시 검증·저장한다. 중단 후에는 검증된 페이지부터 이어가며, 불완전 결과를 `complete` evidence나 기업행동 검증 입력으로 사용하지 않는다.

이번 실제 수집에서 8 worker는 서버의 `RemoteDisconnected`로 실패했다. OpenDART 공시검색 API는 `list.json` GET endpoint이며, 요청 제한 응답 `020`은 통상 20,000건 이상 요청에서 발생할 수 있다고 안내한다. [OpenDART 공시검색 개발가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019001), [응답 코드 안내](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DE003&apiId=AE00036)를 artifact `sources`에 보관한다.

## 범위와 CLI

`OpenDartEvidenceCollector`에 아래 두 public method를 추가한다.

```python
collect_disclosure_partitions(output_dir, start, end, *, page_count=100, retry_policy=None) -> None
merge_disclosure_partitions(input_dir, start, end, output_path) -> None
```

CLI에는 아래 명령을 추가한다.

```text
dart-disclosures-resume --start YYYY-MM-DD --end YYYY-MM-DD --output-dir PATH
  [--page-count 100] [--max-attempts 5]
  [--initial-backoff-seconds 1] [--max-backoff-seconds 30]
  [--min-request-interval-seconds 0.2]

dart-disclosures-merge --start YYYY-MM-DD --end YYYY-MM-DD --input-dir PATH --output PATH
```

worker/concurrency option은 제공하지 않는다. 다른 process가 같은 output directory에 접근하면 lock 획득 실패로 즉시 종료한다.

## 저장 구조와 원자성

```text
data/evidence/stocks/dart_parts/
  manifest.json
  collector.lock                 # flock; 실행 종료 시 자동 해제
  pages/2018-04/00001.json       # 진행 중인 month의 원자적 page checkpoint
  months/2018-04.json            # 모든 page 검증 후 원자적 완성본
```

manifest schema는 `dart-disclosures-manifest-1`이다. 최상위에는 `requested_start`, `requested_end`, `partition: month`, `page_count`, retry policy, endpoint가 있다. 각 `months[YYYY-MM]` entry는 범위, status (`pending|in_progress|complete|incomplete|blocked`), `next_page`, 발견된 `total_page`, `record_count`, 완성 artifact path/SHA-256, 마지막 오류를 가진다.

각 page checkpoint에는 request parameter hash, page number, page size, raw records, record count, response total page, SHA-256를 기록한다. 성공 page는 sibling temp file을 통해 atomic replace한 뒤 manifest `next_page`를 갱신한다. 전체 month의 모든 page가 존재·해시 일치·`total_page` 일치할 때만 records를 `(rcept_dt, rcept_no)`로 정렬하고 exact duplicate receipt number를 거부한 뒤 `months/YYYY-MM.json`을 atomic write한다. 완료 뒤 page checkpoints는 삭제해 중복 저장을 줄인다.

재실행은 complete month artifact와 manifest digest가 일치할 때만 건너뛴다. checkpoint가 손상되면 해당 page부터 다시 요청한다. final merge는 모든 월이 complete인 경우에만 실행하며, output이 이미 있을 때 byte hash가 다르면 덮어쓰지 않는다.

## 오류 분류 및 백오프

`DartRetryPolicy` 기본값은 `max_attempts=5`, `initial_backoff_seconds=1`, `max_backoff_seconds=30`, `min_request_interval_seconds=0.2`다. 시도 횟수에는 최초 요청이 포함된다. 재시도 전 delay는 `min(max, initial * 2**retry_index)`이며 jitter를 넣지 않아 재현 가능하게 한다.

| 결과 | 처리 |
| --- | --- |
| `requests.RequestException`, HTTP 408/429/5xx, status `800`/`900` | 정책 한도까지 backoff 후 재시도 |
| status `013` | 유효한 빈 page/month로 처리 |
| status `020` | quota `blocked`; 즉시 중단, 재시도하지 않음 |
| HTTP 4xx(429 제외), status `010/011/012/021/100/101/901`, malformed JSON/list/receipt | terminal `incomplete`; 즉시 중단 |
| 재시도 소진 | `incomplete`; 마지막 오류·attempt count 저장 후 `EvidenceCollectionError` |

월의 terminal failure 뒤에는 이전 complete month를 보존하고 즉시 예외를 올린다. 자동으로 다음 월을 계속 처리해 공백을 감추지 않는다.

## 구현 순서와 검증

1. retry policy, injected `sleep`/`monotonic`, DART response classifier를 추가한다. 기존 one-shot `collect_disclosures`의 공개 동작은 유지한다.
2. KRX calendar의 partition-manifest 패턴을 재사용해 DART 월/page checkpoint, validation, `flock`를 구현한다.
3. CLI resume/merge command와 parser tests를 추가한다.
4. retry classification, interrupted page resume, status-013 empty month, digest tamper, range mismatch, concurrent lock, merge no-overwrite tests를 작성한다.
5. 실제 실행은 `dart_parts_single_worker/` 새 directory에서 시작한다. 기존 병렬 `dart_parts/`의 33개 partial 파일은 provenance용으로 유지하고 새 manifest의 complete month로 자동 승격하지 않는다.

성공 기준은 (a) 네트워크 중단 뒤 다음 실행이 마지막 검증 페이지에서 계속되고, (b) quota/영구 오류가 `complete` artifact를 만들지 않으며, (c) 전체 merge와 catalog 등록 전 모든 월 hash가 검증되는 것이다.
