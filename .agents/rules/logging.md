---
trigger:
  - on_file_path_regex: "src/.*\\.py"
  - on_file_path_regex: "tests/.*\\.py"
priority: 9
---

# Unified Logging & Tagging Directives (AI-Optimized)

This document defines the strict logging rules and tag-based format requirements to optimize debugging efficiency, ensure token economy, and enable systematic grep-based log parsing.

---

## 1. Core Principles: AI-Reading Optimization

- **Structured Formats**: Output log messages in key-value structures; avoid verbose or conversational sentences (e.g., prefer structured logs over "Successfully started...").
- **Strictly Flat & Parsable**: Optimize every `DEBUG`/`TRACE` level log for direct programmatic extraction or regex parsing.
- **Categorized Isolation (Preferred over Unified Log)**:
  - Route high-frequency data (e.g., raw signal outputs, optimization trials) to dedicated, isolated files (e.g., `logs/optuna.jsonl`, `logs/memory.log`) to prevent clogging the main system log.
  - This allows targeting precise file paths, saving context window space and token usage.

---

## 1.5 Log Directory Hygiene & Path Scoping

- **Strict Path Scoping**: All operational logs MUST reside within the project's `logs/` directory. Writing to `/tmp` is strictly prohibited.
- **Log Isolation Strategy**:
  - **Persistent System Logs**: Main operational/service logs MUST be written to `logs/` root (e.g., `logs/sys.log`, `logs/algo.log`).
  - **Transient / Diagnostic Logs**: All temporary scripts, one-off verification outputs, scratch runs, and test logs MUST be written to `logs/scratch/` (e.g., `logs/scratch/scratch_verify.log`).
- **Transient Cleanup Directive**: Logs under `logs/scratch/` are considered ephemeral. The AI must purge old diagnostic logs in `logs/scratch/` or run `python tools/devops/clean_logs.py` when concluding diagnostic tasks.

---

### 2.1 INFO (Terminal-Clean Output)
- **Purpose**: Minimal progress reporting for humans.
- **Constraints**: 
  - Keep logs under 1 line per major phase transition.
  - Omit massive collections, lists, or matrix arrays from the log body.

### 2.2 DEBUG & TRACE (AI-Data Harvesting Output)
- **Purpose**: Targeted data capture for AI diagnostics and automated audits.
- **Format**: Use `[TAG] key1=value1 key2=value2 ...` to capture structured data cleanly without conversational descriptions.

---

## 3. Minimal Tag Schema (Max 4 Standard Tags)

To prevent tag proliferation, the AI MUST strictly categorize all debug/trace logs into one of these 4 tags:

| Standard Tag | Target Category | Required Payload Keys | Example |
| :--- | :--- | :--- | :--- |
| `[SYS]` | Memory, OS, runtime environment, execution speed | `stage`, `rss`, `delta`, `elapsed_ms` | `[SYS] stage=l2_sim_cache rss=397MB delta=+297MB` |
| `[DATA]` | Data integrity, inputs, dataset boundaries, shapes | `symbol`, `nan_pct`, `shape`, `status` | `[DATA] symbol=005930 nan_pct=0.0 status=PASS` |
| `[ALGO]` | Signals, weights, alpha allocation, active parameters | `symbol`, `sleeve`, `raw_mu`, `weight` | `[ALGO] symbol=005930 sleeve=closing_alpha raw_mu=0.9` |
| `[EVAL]` | Performance, metrics, optimization trials, backtest results | `trial`, `cagr`, `sharpe`, `mdd`, `er` | `[EVAL] trial=14 cagr=0.15 sharpe=1.2 mdd=0.08` |

---

## 4. Token & Parsability Optimizations

- **Formatter-Driven Prefixing**: Rely on the logger formatter for timestamps and filenames; omit them from the message body.
- **Float Formatting**: Limit float numbers to a maximum of 3 decimal places (e.g., use `%.3f` or `:.3f`) to save tokens.
- **Conditional Array Truncation**: When logging symbol lists or arrays, truncate after 5 items and suffix with `_truncated={count}`.
  - **Preferred**: `[ALGO] symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT'] truncated=45`
  - **Avoid**: `[ALGO] symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', ... 50 more symbols]`

