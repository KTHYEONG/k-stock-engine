"""Benchmark the vectorized outcome-coverage preflight against a large panel.

The ML realized-outcome integrity spec requires that the outcome preflight
(``HorizonOutcomeCoverage.build``) be Polars-join/group-by vectorized only and
not raise the measured full-run peak RSS (9,133 MiB) or wall time (370,582 ms)
by more than 5% absent a separately approved data rebuild. This harness runs
the preflight over a synthetic panel approximating the production decision
universe (~920k base-plus-feature rows, ~1.7k OOF sessions) and reports the
elapsed time and per-call peak RSS so a regression is visible before a
full-snapshot rebuild.

Usage:
    uv run python tools/agent_skills/benchmark_outcome_preflight.py
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import polars as pl

from legacy.stocks.ml.data import HorizonOutcomeCoverage
from legacy.stocks.ml.result_ledger import peak_rss_mib

OOF_SCORE_ROWS = 630_000
OOF_SESSIONS = 1_706
_STATUS_VOCABULARY = ("REALIZED", "PARTIAL_TAIL", "MISSING_EXIT_PRICE")


def _synthetic_score_keys(rows: int, sessions: int) -> pl.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    session_times = [
        start + timedelta(days=i) for i in range(sessions)
    ]
    names = rows // sessions
    session_column: list[datetime] = []
    instrument_column: list[str] = []
    for session in session_times:
        session_column.extend([session] * names)
        instrument_column.extend([f"KRX:{t + 1:06d}" for t in range(names)])
    return pl.DataFrame(
        {
            "instrument_id": instrument_column,
            "session": session_column,
        }
    )


def _synthetic_status(keys: pl.DataFrame) -> pl.DataFrame:
    states = pl.Series(
        [ _STATUS_VOCABULARY[i % len(_STATUS_VOCABULARY)] for i in range(keys.height) ]
    )
    return keys.with_columns(states.alias("outcome_status"))


def main() -> int:
    keys = _synthetic_score_keys(OOF_SCORE_ROWS, OOF_SESSIONS)
    status = _synthetic_status(keys)
    baseline_rss = peak_rss_mib()
    started = time.monotonic()
    coverage = HorizonOutcomeCoverage.build(3, keys, status)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    peak = peak_rss_mib()
    print(
        f"[BENCH] outcome_coverage decision_rows={coverage.decision_rows} "
        f"realised={coverage.realized_rows} elapsed_ms={elapsed_ms} "
        f"peak_rss_mib={peak} baseline_rss_mib={baseline_rss}"
    )
    if coverage.status_counts.unresolved == 0:
        raise SystemExit("benchmark status panel degenerated to zero unresolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
