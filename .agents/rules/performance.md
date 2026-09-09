---
trigger:
  - on_label: ["performance"]
  - on_file_path_regex: "src/.*(engine|data|features|strategy|validation|execution|portfolio).*"
  - on_file_path_glob: ["src/engine/**/*.py", "src/data/**/*.py", "src/features/**/*.py", "src/strategy/**/*.py", "src/execution/**/*.py"]
priority: 10
---

# Performance & Optimization Directives (Measurement-Driven)

> **Never reduce workload scope to hide performance issues. Preserve correctness first. Profile and benchmark to find actual bottlenecks. Apply targeted optimizations suited to the bottleneck shape and verify gains before and after.**

## 1. Workload Semantics & Completion Integrity
- **Preserve Workload Semantics:** Never silently reduce the requested date range, dataset, iterations, epochs, or simulation scope merely for convenience or speed.
- **Distinguish Heavy Runs from Hangs:** Long runtime alone is not a bug; distinguish legitimate computation from hangs, deadlocks, or pathological scaling using observable progress, heartbeats, and profiling. Do not rely on arbitrary hardcoded timeouts.
- **Inference Over Interruption:** Derive reasonable execution parameters from existing configurations, schemas, and conventions; clarify with the user only when an ambiguity fundamentally alters requirements or performance guarantees.

## 2. Measurement-Driven Optimization
- **Correctness First:** Maintain algorithmic correctness, readability, and numerical stability; optimize measured bottlenecks only.
- **Empirical Benchmarking:** Establish repeatable before-and-after benchmarks with realistic variance tolerances to justify optimizations.
- **Measured Resource Scaling:** Dynamically size worker pools, thread concurrency, and batch sizes based on measured CPU, memory, and I/O scaling rather than static assumptions.

## 3. Storage, Memory & I/O Semantics
- **Efficient I/O:** Leverage column pruning, predicate pushdown, and stream/chunked reads to avoid loading unnecessary data into memory.
- **Pragmatic Memory Management:** Consider views, in-place modifications, or chunking only when memory is a measured bottleneck, ensuring copy/view semantics and code clarity are not compromised.
- **Precision Validation:** Downcast types (e.g., to `float32` or compact integers) only after verifying that numerical drift does not impact financial/statistical correctness.

## 4. Execution Escalation & Parallelism
- **Vectorization vs. Loops:** Prefer vectorized operations where array semantics naturally apply. Retain straightforward loops for control flow or lightweight logic where vectorization adds needless complexity.
- **Escalate When Justified:** If a measured hot path cannot be efficiently handled with standard vectorization (e.g., sequential order matching, ledger replay), evaluate JIT compilation, native code, out-of-core engines, or parallelism based on the bottleneck shape. Adopt added complexity only when benchmarks demonstrate meaningful gains.
- **Parallelism Overhead:** Restrict multi-processing or multi-threading to workloads where computational gains significantly outweigh process spawning and IPC/serialization overhead.