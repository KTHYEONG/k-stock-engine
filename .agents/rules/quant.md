---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(etf|filters|execution|data|validation|training|evaluation).*"
  - on_file_path_glob: ["src/**/etf/**/*.py", "src/**/filters/**/*.py", "src/**/execution/**/*.py", "src/**/data/**/*.py", "src/**/validation/**/*.py"]
priority: 10
---

# Quant & Financial Engineering Principles

This document provides quantitative and financial directives for building robust, evidence-based trading systems.

## 0. Priority Hierarchy
1. **Logic Robustness Over Metrics:** Prioritize sound financial reasoning and statistical validity over metric overfitting.
2. **Data Integrity & Realism:** Enforce strict temporal availability and cost accounting.
3. **Numerical Stability:** Handle division by zero and floating-point edge cases safely.

## 1. Safe Division & Numerical Stability
- **Safe Vectorized Division:** Use `np.divide` with explicit `out` initialization and `where` masks to avoid uninitialized memory or zero-division warnings.
  ```python
  result = np.zeros_like(numerator, dtype=float)
  np.divide(numerator, denominator, out=result, where=denominator != 0)
  ```
- **Log-space Operations:** Use `np.log1p()` and `np.expm1()` for compounding returns or small rates to avoid numerical underflow.

## 2. Information Availability & Pipeline Timestamps
- **Explicit Timestamp Semantics:** Define `observation_time` (event occurrence), `decision_time` (signal calculation), and `execution_time` (order fill).
- **Information Availability:** Enforce that all data used at `decision_time` was available in reality. Do not prescribe `.shift(1)` blindly unless required by pipeline semantics.
- **Data Alignment (`merge_asof`):** Align datasets using release timestamps and verified monotonic time ordering to prevent look-ahead bias.

## 3. Microstructure & Financial Realism
- **Trading Costs:** Model transaction fees, securities transaction tax (국내주식 거래세), bid-ask spread, tick size constraints, and execution slippage.
- **Settlement & Cash Drag:** Account for T+2 settlement cycles and cash margin/capital allocation when evaluating short-term holding and closing-price (종가매매) strategies.

## 4. Machine Learning & Labeling
- **Objective-Driven Metrics:** Select classification/regression labels and evaluation metrics (IC, Sharpe, Accuracy, R²) directly from the economic decision objective.
- **Purging & Embargoing:** Apply purged/embargoed validation when using overlapping label windows to prevent leakage between train and test sets.
- **Scaler Isolation:** Fit scalers strictly on training folds; transform validation and test folds independently.
