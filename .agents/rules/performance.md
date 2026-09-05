---
trigger:
  - on_label: ["performance"]
  - on_file_path_regex: "src/.*(engine|data|features|strategy|validation|execution).*"
  - on_file_path_glob: ["src/engine/**/*.py", "src/data/**/*.py", "src/features/**/*.py", "src/strategy/**/*.py"]
priority: 10
---

# Performance & Optimization Directives (Measurement-Driven)

This document defines performance optimization guidelines focused on empirical benchmarks rather than static hardware mandates.

---

## 0. Completion Integrity (Non-Negotiable)
- **No Self-Imposed Timeouts:** Never attach an arbitrary `timeout`, `max_iterations`/`n_epochs` cap, or a shrunk sample/date-range to a backtest or ML training run just to finish faster or save tokens/turns. Use the caller-specified stopping criterion (full date range, convergence, epoch budget stated in the spec's `requirements`); if none is stated, ask rather than silently truncating.
- **Long-Running Is the Correct Outcome, Not a Bug:** A multi-hour backtest or training run completing in full is success, not something to optimize away. Only act on measured bottlenecks (section 1) — never by cutting scope or duration.
- **Hang Detection ≠ Time Limit:** A watchdog/timeout is only for detecting a genuine hang (no progress, deadlock). Set it as a generous multiple (5-10x) of the measured or estimated full-run duration, and treat a trip as an infrastructure bug to report — never as "partial results are good enough, ship it."

---

## 1. Measurement & Bottleneck Philosophy
- **Correctness First:** Prioritize code correctness and algorithmic soundness; optimize measured bottlenecks only.
- **Benchmark Driven:** Establish a benchmark before and after optimization to prove gains.
- **Hardware & System Resource Scaling:** Determine worker counts, process pools, and batch sizes dynamically based on system resource availability (`psutil`), workload footprint, and measured scaling.
- **Data I/O & Storage Format:** Utilize Parquet with PyArrow backend (`pyarrow`) for optimized columnar data reading and memory-mapped dataset access.

---

## 2. Memory & Precision Optimization
- **Precision Validation:** Use `float32` for large arrays or intermediate feature matrices only after numerical-error validation. Retain `float64` for sensitive matrix inversions or compounding returns.
- **Memory Footprint Management:** Prefer in-place operations or view slices for large arrays (`numpy`, `pandas`). Avoid unnecessary deep copying.
- **Targeted Memory Releases:** Invoke explicit garbage collection (`gc.collect()`) or GPU cache clearing (e.g., `torch.cuda.empty_cache()` if PyTorch is used) only when profiling indicates retained-memory pressure.

---

## 3. High-Performance Execution & Parallelism
- **Vectorization vs Loops:** Prefer vectorization (NumPy / Pandas, or Polars if introduced) for large hot-path computations. Allow standard Python loops for control-flow or lightweight tasks where vectorization overhead exceeds benefits.
- **Escalate Past Vectorization — Don't Stop at "Python Is Slow":** When a *measured* bottleneck resists vectorization (typically a sequential, state-dependent loop -- e.g. an equity curve or ledger where each step needs the previous step's result), that is not the ceiling. Actively evaluate the next tier before accepting a pure-Python loop as final:
  - JIT/compiled acceleration (e.g. Numba, JAX)
  - Native extensions (e.g. Rust via PyO3, Cython)
  - Out-of-core columnar query engines for large Parquet-backed aggregation/joins that don't need to be a DataFrame in memory (e.g. DuckDB, Polars)

  These are categories, not a fixed list -- pick whatever fits the actual bottleneck shape, including tools not named here. Still gated by **Benchmark Driven** above: adopt only with a measured before/after, and treat a new dependency the same as any other (`uv add`, check `pyproject.toml` first per `python.md`). For a hot path identified at design time, record the candidate approach in the spec's `design_rationale` so the evaluation happens once, deliberately, rather than being silently skipped during `implement`.
- **JIT Strategy (If Numba/JAX used):** Pass memory-contiguous arrays to JIT functions (`np.ascontiguousarray`). Use JIT caching when functions are repeatedly compiled across runs.
- **Measured Parallelization:** Restrict process pool or thread execution to heavy tasks where task computation time significantly outweighs inter-process/inter-thread communication overhead.

---

## 4. Performance Regression & Stability
- **Variance Tolerance:** Evaluate performance regressions using stable, repeatable benchmarks with realistic variance tolerances.