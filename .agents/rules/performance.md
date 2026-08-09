---
trigger: glob
priority: 10
---

# Performance & Optimization Directives (Measurement-Driven)

This document defines performance optimization guidelines focused on empirical benchmarks rather than static hardware mandates.

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
- **JIT Strategy (If Numba used):** Pass memory-contiguous arrays to JIT functions (`np.ascontiguousarray`). Use JIT caching when functions are repeatedly compiled across runs.
- **Measured Parallelization:** Restrict process pool or thread execution to heavy tasks where task computation time significantly outweighs inter-process/inter-thread communication overhead.

---

## 4. Performance Regression & Stability
- **Variance Tolerance:** Evaluate performance regressions using stable, repeatable benchmarks with realistic variance tolerances.