# k-stock-engine ML/Backtesting Performance + Refactoring Architecture Review

대상 저장소:

* Repository: `KTHYEONG/k-stock-engine`
* Target branch: `feat-stock-architecture`
* 분석 기준 GitHub HEAD: `9260ffd6978797622b9cb7a0f467cfbaa61cd6a4`

이번 작업은 **구현 작업이 아니다.**
반드시 저장소에 제공된 **`spec` skill을 사용하여 충분한 재분석과 검증을 수행한 뒤 구현 계획/spec만 작성하라.**

소스 코드를 성급하게 수정하지 말고, 지금 제시된 가설도 정답으로 간주하지 마라. 실제 코드, call graph, 기존 결과 ledger, profiling 결과를 근거로 각 제안을 반박/수정/채택하라.

## 1. 최상위 목표

ML training 및 stock backtesting의:

1. wall-clock 실행시간을 최대한 크게 감소
2. peak RSS를 크게 감소
3. `max_rss_mib`가 설정된 경우 OOM이 발생하지 않는 구조 확보
4. 기존 quant / execution / temporal correctness 완전 유지
5. 지나치게 큰 파일을 responsibility 기준으로 분해
6. 인간 유지보수성과 AI agent의 context/token 효율 개선

사용자 우선순위는 다음과 같다.

`실행시간 감소 > 추가적인 RAM 감소`

단,

`OOM 또는 memory budget 위반은 절대 허용하지 않는다.`

즉 RAM을 무제한 소비하여 빠르게 만드는 방법은 허용하지 않는다.

## 2. 작업 제한

이번 단계에서는 구현하지 마라.

* production source 수정 금지
* 무단 algorithm 변경 금지
* 임의의 model/grid/horizon/profile 축소 금지
* statistical/economic gate를 성능 때문에 약화하지 말 것
* temporal integrity, purging, embargo, causal calibration을 약화하지 말 것
* base/stress cost semantics를 바꾸지 말 것
* execution/T+2/partial-fill/capacity semantics를 바꾸지 말 것
* deterministic seed/ordering을 바꾸지 말 것
* benchmark 결과 없이 "빨라질 것이다"라고 단정하지 말 것

`spec` skill이 요구하는 계획/spec 문서 작성만 허용한다.

spec 자체가 별도의 spec artifact를 생성해야 한다면 그것은 허용하지만 production 구현은 하지 않는다.

## 3. 저장소 규칙부터 읽어라

가장 먼저 다음을 확인하라.

* `AGENTS.md`
* `.agents/rules/performance.md`
* `.agents/rules/python.md`
* `.agents/rules/quant.md`
* 관련 architecture/spec 문서
* `docs/code_map.json`
* 기존 ML result ledger

특히 저장소가 요구하는:

* Spec → Implement → Check
* benchmark-driven optimization
* deterministic execution
* temporal integrity
* memory-aware worker planning

을 위반하지 마라.

현재 checkout이 실제로 `feat-stock-architecture`인지 확인하고 HEAD SHA도 기록하라.

분석 도중 target branch가 변경되었다면 위 SHA에 고정하지 말고 최신 checkout 상태를 기준으로 분석하되, 어떤 SHA를 분석했는지 spec에 명시하라.

## 4. 기존 실제 성능 데이터부터 조사하라

최소 다음을 조사하라.

* `docs/results/ml_runs/latest.json`
* `docs/results/ml_runs/recent.jsonl`
* `docs/results/back-res.md`

특히 최근 full-size run들의:

* row count
* horizon 수
* fold 수
* profile/candidate 수
* wall time
* per-phase elapsed
* RSS
* peak RSS
* OOF path evaluation count
* replay candidate count
* prepared cache telemetry

를 표로 정리하라.

기존 기록 중 대표적으로 약 92만-row / H=10,20 run에서 `horizon_discovery`가 약 129초, peak RSS가 약 5GiB 수준이었던 기록이 있으므로 실제 저장된 최신 기록과 다시 대조하라.

단, 과거 run과 현재 HEAD의 구현이 동일하다고 가정하지 마라.

## 5. Benchmark/measurement 자체의 신뢰성을 먼저 감사하라

현재 telemetry가 실제 prepare와 execute 비용을 정확히 분리하는지 확인하라.

특히 다음을 점검하라.

* `prepared_cache_bytes`
* `prepared_segment_build_count`
* `replay_prepare_elapsed_ms`
* `replay_execute_elapsed_ms`
* phase별 `peak_rss_mib`
* process lifetime high-water mark와 phase-local peak의 구분

현재 코드에서 prepare/execute 시간이 동일한 total elapsed로 기록되거나 cache bytes가 0으로 남는 경로가 있는지 확인하라.

성능 spec에는 반드시 **신뢰 가능한 baseline 측정 방법**을 먼저 포함하라.

가능하다면 다음을 구분하라.

* full process max RSS
* phase-local RSS peak
* Python heap
* NumPy/Polars/LightGBM/native allocation

필요하다면 `/usr/bin/time -v`, `psutil`, cProfile, py-spy, memray 등 적절한 도구를 검토하되 불필요한 dependency 추가를 확정하지 말고 비교하라.

## 6. Fail-fast architecture를 점검하라

최근 H=3 / cadence=(5,10,20) 계열 run이 execution frontier feasibility 때문에 fitting 전에 실패하면서도 상당한 data load 시간/RSS를 소비한 기록이 있다.

다음이 데이터 I/O 전에 검증 가능한지 확인하라.

* candidate horizon
* rebalance cadence
* top-k
* max exposure
* max single weight
* execution frontier feasibility
* request 자체의 정적 invariant

가능하다면:

`parse config → build/validate request → feasibility gate → data I/O`

순서가 되도록 하는 방안을 spec에 제시하라.

정상 run 성능과 invalid-run fail-fast 효과를 구분해 평가하라.

## 7. DirectMarketDataLoader를 매우 세밀하게 감사하라

집중 파일:

* `src/stocks/data/direct.py`
* `src/stocks/cli/train.py`
* `src/stocks/ml/data.py`
* 관련 parquet storage 코드

다음 가설을 검증하라.

### 가설 A — eager parquet read

현재 `pl.read_parquet(...)`로 전체 dataset을 읽은 뒤:

* column select
* session date filter

가 적용되는지 확인하라.

가능하다면:

* `pl.scan_parquet`
* projection pushdown
* predicate pushdown
* Hive partition pruning

을 이용해 실제 physical read를 줄일 수 있는지 분석하라.

반드시 Polars query plan 또는 실제 I/O/profile evidence로 확인하도록 계획하라.

### 가설 B — multi-horizon feature duplication

현재 direct composition에서:

`base + features → decision_frame → join long labels → composed`

과정을 통해 동일 `(instrument_id, session)` feature row가 horizon 수만큼 반복되는지 정확히 측정하라.

그 후 `compose_net_alpha_training_data()`가 이를 다시 unique/dedup하는지 확인하라.

가능하다면 architecture를:

* one-row-per-key feature/market frame
* independent narrow labels_by_horizon
* independent status/evidence sidecars

로 유지하여 feature와 label을 처음부터 물리적으로 합치지 않는 방안을 검토하라.

`MlMarketData.labels_by_horizon`이 이미 존재하는데 CLI가 이를 충분히 활용하지 않는지도 확인하라.

2/3/6 horizon일 때 현재 구조와 개선 구조의 예상 row/byte complexity를 식으로 제시하라.

### 가설 C — validation full scans

다음을 점검하라.

* `_validate_monotonic_sessions`
* `_validate_numeric_finiteness`
* duplicate validation

instrument마다 DataFrame filter를 반복하는 O(instruments × rows) 성격의 경로가 있다면 Polars window/aggregate 기반 O(rows) validation으로 바꾸는 계획을 제시하라.

feature column별 full scan도 단일 aggregate projection으로 통합 가능한지 검토하라.

## 8. ML fitting hot path를 profile하라

집중 파일:

* `src/stocks/ml/training.py`
* `src/stocks/ml/models.py`
* `src/stocks/ml/features.py`
* `src/stocks/ml/horizons.py`
* `src/stocks/research/folds.py`
* calibration 관련 코드

최소 다음 call chain을 추적하라.

`train_net_alpha_model`
→ horizon discovery
→ `_fit_oof`
→ nested ElasticNet alpha selection
→ final fold fit
→ causal calibration
→ execution replay
→ bootstrap selection

각 phase별:

* 호출 횟수
* DataFrame rows/columns
* NumPy matrix shape/dtype
* full-frame allocation 횟수
* join 횟수
* sort 횟수
* Polars→NumPy conversion 횟수

를 산출하거나 계측 계획을 작성하라.

## 9. PreparedTrainingMatrix architecture를 검증하라

다음 설계를 정답으로 가정하지 말고 실제 코드에 맞는지 평가하라.

```text
PreparedTrainingMatrix
├── X: float32 [N, F]
├── instrument_code: int32
├── session_code: int32
├── session timestamps
├── fold integer indices
└── labels/economic outcomes by horizon
```

목표는 feature transformation 이후 learner feature matrix를 단 한 번 materialize하고:

* outer folds
* nested folds
* ElasticNet path
* final fit
* LightGBM challenger

가 이를 공유하게 만드는 것이다.

검토해야 할 항목:

* 현재 `_float32_matrix()` 호출 횟수
* 동일 feature rows가 몇 번 NumPy로 변환되는지
* boolean indexing에 의한 full matrix copies
* standardization intermediate allocation
* sample-weight recomputation
* labels와 feature alignment 보장 방법
* fold-local statistics 누수 방지

Prepared matrix를 사용하더라도 scaler/statistics는 반드시 **fold-local train rows만** 사용해야 한다.

global preprocessing을 도입해 leakage를 만들면 안 된다.

## 10. session-balanced weights를 분석하라

현재 Polars group-by → Python dict → `to_list()` → Python comprehension 구조가 반복되는지 확인하라.

`session_code`와 `np.bincount` 또는 완전 vectorized Polars expression으로 동일한 weight semantics를 재현 가능한지 검토하라.

반드시 numerical parity test를 계획하라.

## 11. ElasticNet path / alpha selection의 반복 allocation을 조사하라

특히:

* `_select_elastic_alpha`
* `fit_weighted_elastic_path`
* `ElasticPathSolution.predict`
* Rank-IC 계산

을 조사하라.

alpha fraction마다:

* DataFrame 생성
* Series attach
* realized outcome join
* Rank-IC용 frame 생성

이 반복된다면 aligned array/index를 이용해 동일 결과를 계산하는 방안을 검토하라.

단, regularization grid, nested-fold methodology, Rank-IC definition 자체를 성능 목적으로 변경하지 마라.

추가로 `valid` mask와 sample weights indexing이 일부 invalid feature/target row가 존재할 때 shape-consistent한지 correctness audit도 수행하라.

## 12. execution replay architecture를 가장 중요하게 조사하라

집중 파일:

* `src/stocks/ml/execution_replay.py`
* replay 호출부 in `training.py`
* `src/stocks/backtesting/engine.py`

다음 불일치를 확인하라.

`stream_execution_replay_batch()`가 문서상 "one prepared segment at a time"이라고 되어 있지만 실제로 `prepare_execution_replay_batch()`가 모든 segment의:

* `segment_ordered`
* `PreparedReplayMarket`
* `scored_market`
* session index
* score overlay

를 먼저 만들어 `segment_data`에 보관하는지 확인하라.

확인된다면 **segment-major true streaming** 설계를 검토하라.

목표 구조:

```text
candidate accumulators 생성

for segment:
    prepare segment once

    execute every compatible candidate
    immediately aggregate compact evidence
    discard raw ledger/candidate state

    release prepared segment

finalize candidate evidence
```

비교해야 할 것:

* current candidate-major architecture
* segment-major architecture
* segment preparation count
* peak live bytes
* candidate replay count
* cache locality
* wall time

dense-shadow candidate도 동일 segment 내에서 처리할 수 있는지 검토하라.

## 13. max_rss_mib guard를 감사하라

현재 resource planning이 prepared batch allocation **이후** 실행되는 경로가 있는지 확인하라.

있다면 이는 OOM 예방 관점에서 잘못된 순서다.

설계는 가능한 한:

```text
metadata / shape estimation
→ resource budget
→ worker count
→ allocation
```

이어야 한다.

추가로 현재 `PreparedReplayMarket.cache_bytes`가 다음까지 포함하는지 확인하라.

* Polars frames
* all NumPy arrays
* Python dictionaries
* tuple keys
* datetime/string objects
* `_PreparedRow`
* scored market
* score overlay
* worker-local state

불완전하다면 resource estimator를 재설계하라.

container/cgroup 환경도 고려하고, host physical RAM만 보고 판단하는 구조가 안전한지 검토하라.

## 14. StockBacktester의 두 execution representation을 감사하라

집중 파일:

`src/stocks/backtesting/engine.py`

다음을 확인하라.

* `StockBacktester.run()`이 prepared branch 판단 전에 `partition_by()`를 수행하는지
* prepared run에서도 불필요한 `by_session` allocation이 발생하는지
* non-prepared path의 `rows_frame.to_dicts()`
* per-session `rows.to_dicts()`
* `(str, datetime) → _PreparedRow` Python mapping
* decision마다 `panel.filter(available_time <= decision_time)` 호출

각각 실제 호출 횟수와 비용을 profile하라.

장기적으로 standard backtest와 ML execution replay가 모두 하나의 **prepared columnar engine**을 사용할 수 있는지 검토하라.

기존 DataFrame engine과 prepared engine 두 개를 영구 유지하는 것과 하나의 canonical execution core로 통합하는 것을 trade-off 분석하라.

## 15. Columnar market index를 검토하라

현재 Python object-heavy key:

`(instrument_id: str, session: datetime)`

를 대량 dict entry로 저장하는 비용을 측정하라.

대안으로:

* instrument categorical/int32 code
* session int32 code
* session row ranges
* NumPy arrays
* integer row index
* sorted encoded keys + searchsorted
* per-session sparse index

를 비교하라.

full dense `[sessions × instruments]` matrix는 무조건 도입하지 말고 memory complexity부터 계산하라.

실제 positions/top-k 접근 패턴에 적합한 sparse/indexed representation을 선택하라.

## 16. as-of filtering을 조사하라

decision마다 대형 panel에:

`available_time <= decision_time`

filter가 반복된다면:

* monotonic cutoff
* precomputed stop index
* sorted available-time ranges
* zero-copy/view slice

로 대체 가능한지 검토하라.

단, downstream planner가 전체 historical frame이 필요한지 current decision cross-section만 필요한지를 먼저 확인하라.

semantic change를 최적화로 위장하지 마라.

## 17. Bootstrap implementation을 통합 가능한지 검토하라

다음을 비교하라.

* `src/stocks/ml/horizons.py::_cohort_bootstrap`
* `src/stocks/research/calibration_schedule.py`
* economic alpha bootstrap helper들

`horizons.py`에서 `n_bootstrap × N` index/sample matrix를 materialize하는 반면 calibration schedule에는 prefix-sum 기반 bounded-workspace kernel이 존재하는지 확인하라.

통계적으로 정확히 동일한 moving-block sampling semantics를 유지하면서 common bootstrap primitive를 재사용할 수 있는지 검토하라.

다음을 비교 측정할 계획을 작성하라.

* peak temporary bytes
* bootstrap wall time
* exact seeded draws
* p-value
* quantile/lower bound
* gate/admission parity

## 18. Parallelism은 마지막에 분석하라

다음을 함께 고려하라.

* Polars threads
* NumPy/BLAS threads
* sklearn
* LightGBM `num_threads`
* ThreadPoolExecutor replay workers

nested parallelism/oversubscription 가능성을 측정하라.

1 / 2 / 4 workers 또는 실제 CPU에 적합한 grid를 benchmark하고:

`time × peak RSS`

Pareto frontier를 작성하라.

Python-heavy replay가 GIL 때문에 thread scaling을 얻지 못한다면 worker 수를 늘리는 방안을 채택하지 마라.

ProcessPool은 DataFrame/array 복제로 RAM을 크게 늘릴 수 있으므로 특별한 근거 없이 사용하지 마라.

## 19. float32/Numba는 후순위로 취급하라

먼저 중복 allocation/scan/conversion 제거를 끝낸 뒤 다음을 별도 benchmark 후보로 둬라.

* standardization float32 유지
* target/economic arrays precision
* selected numerical kernels Numba
* contiguous arrays
* LightGBM data layout

float32로 인해:

* selected alpha
* Rank-IC
* bootstrap gate
* selected horizon/profile
* trades
* promotion result

가 바뀌지 않는다는 강한 regression test 없이는 채택하지 마라.

## 20. Quant correctness edge cases를 별도로 감사하라

성능 리팩터링 중 특히 다음을 검증하라.

* segment-local ADTV rolling calculation이 segment 시작 이전 lookback을 충분히 포함하는지
* volatility lookback warm-up
* OOF segment boundaries
* label availability
* purging/embargo
* initial calibration seed
* base/stress scenario independence
* T+2 settlement
* next-open execution
* partial fills/capacity
* transaction cost provenance
* score/market alignment
* duplicate decision keys

현재 잘못된 동작이 발견되더라도 성능 리팩터링에 조용히 섞지 말고 별도 correctness finding으로 명시하라.

## 21. Large-file / AI-context refactoring audit

반드시 repository의 Python 파일에 대해 실제 `wc -l`, file size, 주요 symbols를 조사하라.

특히 현재 대략 다음이 큰 것으로 보인다.

* `src/stocks/ml/training.py` ≈ 3,300 lines
* `src/stocks/backtesting/engine.py` ≈ 1,800 lines
* `src/stocks/ml/result_ledger.py` > 1,200 lines
* `src/stocks/ml/execution_replay.py` > 1,100 lines
* `src/stocks/ml/data.py`
* `src/stocks/ml/replay.py`
* `src/stocks/ml/contracts.py`
* `src/stocks/ml/horizons.py`
* `src/stocks/ml/models.py`
* large workflow/CLI modules

단순 line count 순으로 자르지 마라.

각 파일에 대해:

* responsibilities
* inbound imports
* outbound imports
* public symbols
* mutable state
* performance hotness
* test coverage
* change coupling

을 분석하라.

다음 조건 중 복수에 해당하는 파일을 우선 분리하라.

* 700~800+ lines
* 서로 다른 architectural layer 혼재
* algorithm + I/O + orchestration + serialization 혼재
* 독립 테스트 가능한 responsibility가 다수
* 한 수정에 파일 대부분의 context가 필요
* AI agent가 부분 context만으로 안전하게 수정하기 어려움

목표 파일 크기는 일반적으로 200~500 lines 정도를 선호하되 hard cap으로 사용하지 마라.

응집된 수치 알고리즘이라면 더 긴 파일도 허용한다.

micro-module 남발 역시 금지한다.

## 22. training.py decomposition을 검증하라

다음은 후보일 뿐이며 실제 dependency graph에 따라 수정하라.

```text
src/stocks/ml/training/
    __init__.py
    orchestrator.py
    preparation.py
    elastic_selection.py
    oof.py
    discovery.py
    evaluation.py
    publishing.py
    telemetry.py
```

기존:

`from src.stocks.ml.training import ...`

public import contract를 `__init__.py` re-export로 유지 가능한지 확인하라.

특히 dependency direction이:

`orchestrator → stage modules → numerical primitives/contracts`

방향이 되도록 하고 stage module끼리 circular dependency가 생기지 않도록 계획하라.

## 23. backtesting decomposition을 검증하라

이미 존재하는:

* `contracts.py`
* `execution.py`
* `market.py`
* `metrics.py`

를 활용하라.

추가 후보:

* state
* runner

를 검토하고 최종적으로:

`engine.py`가 compatibility facade/orchestrator에 가까워질 수 있는지 분석하라.

새 abstraction을 기존 것과 중복 생성하지 마라.

## 24. execution_replay decomposition

현재 module을 다음 responsibility로 분리할 가치가 있는지 분석하라.

* contracts
* resource planning
* preparation
* execution
* aggregation

package로 전환할 경우 기존 import path를 `__init__.py`로 보존할 수 있는지 확인하라.

`ml/replay.py`와 이름/책임이 겹치므로 먼저 두 모듈의 실제 call graph를 작성하라.

사용되지 않는 legacy 코드가 있다면 곧바로 삭제하라고 계획하지 말고 consumers/test/compatibility를 확인하라.

## 25. result_ledger.py는 hot path와 별도로 리팩터링하라

`result_ledger.py`의 큰 파일 문제는 performance hot path와 구분하라.

가능한 responsibility:

* schema/contracts
* run projection
* observability projection
* persistence/rotation
* Markdown/rendering
* recovery

를 확인하라.

성능 최적화 PR과 단순 ledger decomposition을 반드시 한 번에 묶어야 하는지도 판단하라.

변경 위험이 불필요하게 커지면 별도 phase/PR로 분리하라.

## 26. 성능 개선안을 Impact / Risk / Evidence로 평가하라

모든 후보를 다음 형식으로 ranking하라.

| Candidate | Current evidence | Runtime impact | RAM impact | Correctness risk | Complexity | Priority |
| --------- | ---------------- | -------------: | ---------: | ---------------: | ---------: | -------- |

최소 다음 후보를 평가하라.

* early config fail-fast
* lazy Parquet scan
* feature/label physical separation
* vectorized direct validation
* PreparedTrainingMatrix
* precomputed session coding/weights
* array-based alpha evaluation
* true segment streaming
* resource plan before allocation
* accurate memory estimator
* prepared backtester unification
* integer market indexing
* prefix-sum bootstrap reuse
* thread policy
* float32 expansion
* Numba

## 27. Benchmark matrix를 설계하라

최소 3단계 benchmark를 계획하라.

### Small deterministic benchmark

CI에서도 돌릴 수 있는 작은 fixture.

측정:

* exact outputs
* wall time
* allocation/RSS sanity

### Medium benchmark

대표적인 수십만 rows subset.

용도:

* 빠른 iteration
* profiler
* candidate optimization comparison

### Full benchmark

가능하면 기존 result ledger의 동일 데이터 IDs/기간/config를 그대로 사용.

로컬 dataset이 존재하지 않는다면 없는 데이터를 만들어내지 말고 reproducibility gap으로 기록하라.

full benchmark에서는 최소:

* total wall time
* data-load wall time
* horizon-discovery wall time
* replay prepare time
* replay execute time
* max RSS
* phase-local max RSS
* path evaluation count
* selected candidate outputs

을 기록하라.

## 28. Performance acceptance criteria

결과를 미리 보장하지 말되 다음 우선순위로 판단하라.

Hard gates:

1. no correctness regression
2. no temporal leakage
3. no deterministic-output regression beyond explicitly approved numerical tolerance
4. no OOM
5. configured memory budget respected with safety headroom

Optimization objective:

1. wall time 최소화
2. 그 다음 peak RSS 최소화

1차 engineering target으로는 현재 대표 full benchmark 대비:

* wall time 30%+ 감소
* peak RSS 25%+ 감소

를 목표로 검토하되 arbitrary target을 맞추기 위해 algorithm을 훼손하지 마라.

구조 변경이 성공적이라면 2× 또는 그 이상의 runtime 개선 가능성도 탐색하되 **측정 전에는 약속하지 마라.**

4GiB memory-budget 환경을 목표로 한다면 단순히 4096MiB 아래가 아니라 allocator/native spike용 안전 headroom을 포함하는 방안을 설계하라.

## 29. 반드시 작성할 최종 spec 내용

spec의 최종 산출물에는 반드시 다음이 있어야 한다.

### A. Current architecture map

ML과 backtesting의 실제 call/data flow.

### B. Measured baseline

현재 HEAD에서 확인 가능한 시간/RSS/row/allocation 정보.

### C. Bottleneck ranking

CPU / memory / I/O / Python-object / native-allocation으로 구분.

### D. Root-cause analysis

각 병목이 왜 발생하는지 code path 단위로 설명.

### E. Target architecture

변경 후 data flow와 module dependency 구조.

### F. Refactoring map

old file/function → target module/function mapping.

### G. Optimization phases

서로 독립적으로 benchmark/revert 가능한 작은 phase로 나눌 것.

권장 예:

* Phase 0: measurement + fail-fast
* Phase 1: data I/O/composition
* Phase 2: prepared ML representation
* Phase 3: OOF/Elastic allocation elimination
* Phase 4: execution replay streaming
* Phase 5: canonical backtest core
* Phase 6: bootstrap/kernel unification
* Phase 7: parallelism/precision experiments
* Phase 8: remaining structural refactor

단, 실제 분석에 따라 재조정하라.

### H. Test/parity strategy

기존 exact outputs와 무엇을 비교할지 명시.

### I. Benchmark strategy

각 phase 전/후 동일 command/data로 비교.

### J. Memory model

주요 구조별 theoretical bytes + measured RSS.

### K. Risks and rejected alternatives

왜 ProcessPool, blind Numba, blanket float32, arbitrary caching 등을 선택하거나 버렸는지 기록.

### L. Expected result

확정 수치가 아니라 evidence-based expected range와 uncertainty를 기록.

## 30. 중요한 작업 방식

처음부터 큰 리팩터링 계획을 확정하지 마라.

다음 순서로 사고하라.

```text
inspect
→ dependency/call graph
→ measure
→ identify dominant cost
→ formulate alternatives
→ estimate complexity
→ verify correctness constraints
→ rank alternatives
→ produce spec
```

현재 코드에서 이미 최적화된 부분을 다시 구현하지 마라.

예를 들어 calibration 쪽에 이미 incremental/prefix-sum optimization이 존재하면 재사용 가능성을 먼저 검토하라.

현재 구현이 이 프롬프트의 가설보다 더 나은 경우에는 프롬프트를 따르지 말고 실제 코드와 benchmark evidence를 우선하라.

## 31. 이번 단계의 종료 조건

**구현을 시작하지 마라.**

마지막에는 다음을 명확하게 제시하고 종료하라.

1. 현재 가장 큰 runtime 병목 Top 5
2. 현재 가장 큰 RAM 병목 Top 5
3. 즉시 적용할 가치가 높은 Low-risk / High-impact 변경
4. 별도 실험이 필요한 High-risk optimization
5. 권장 target architecture
6. 구체적인 module/file decomposition
7. 단계별 구현 순서
8. 각 단계 benchmark 및 regression gate
9. 예상되는 성능 개선 범위와 불확실성
10. 최종적으로 생성한 `spec`의 위치/이름

다시 강조한다.

**이번 요청은 implementation 요청이 아니다.**
`spec` skill을 이용해 현재 가설을 재검증하고, 실제 구현 담당 AI가 이후 안전하게 수행할 수 있을 정도로 구체적이고 검증 가능한 계획을 만드는 것이 목표다.
