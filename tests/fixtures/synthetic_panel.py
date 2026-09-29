"""Shared synthetic panel and research-cube helpers for strategy/simulator tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.research.cube import ResearchCube

_PANEL_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "market": pl.String,
    "open": pl.Int64,
    "high": pl.Int64,
    "low": pl.Int64,
    "close": pl.Int64,
    "base_price": pl.Int64,
    "volume": pl.Int64,
    "tick_size": pl.Int64,
    "upper_limit": pl.Int64,
    "lower_limit": pl.Int64,
    "sell_tax_rate": pl.Float64,
    "adtv20": pl.Float64,
    "ret_vol60": pl.Float64,
    "share_factor": pl.Float64,
    "eligible": pl.Boolean,
    "open_at_upper": pl.Boolean,
    "open_at_lower": pl.Boolean,
    "entry_blocked": pl.Boolean,
}

__all__ = ["synthetic_cube", "synthetic_panel_row", "synthetic_sessions", "write_synthetic_panel"]


def synthetic_sessions(n: int, start: date = date(2020, 1, 6)) -> list[date]:
    """Consecutive calendar sessions for synthetic tests."""
    return [start + timedelta(days=i) for i in range(n)]


def synthetic_panel_row(session: date, instrument_id: str, **overrides: Any) -> dict[str, Any]:
    """One Gold-panel row with liquid defaults."""
    row: dict[str, Any] = {
        "session": session,
        "instrument_id": instrument_id,
        "market": "KOSPI",
        "open": 10000,
        "high": 10100,
        "low": 9900,
        "close": 10000,
        "base_price": 10000,
        "volume": 100000,
        "tick_size": 1,
        "upper_limit": 13000,
        "lower_limit": 7000,
        "sell_tax_rate": 0.0,
        "adtv20": 1_000_000_000_000.0,
        "ret_vol60": 0.02,
        "share_factor": 1.0,
        "eligible": True,
        "open_at_upper": False,
        "open_at_lower": False,
        "entry_blocked": False,
    }
    row.update(overrides)
    return row


def write_synthetic_panel(root: Path, name: str, rows: list[dict[str, Any]]) -> Path:
    """Write a Gold panel directory with a verified manifest."""
    dataset = Path(root) / name
    by_year: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        session = row["session"]
        assert isinstance(session, date)
        by_year.setdefault(session.year, []).append(row)
    partitions: list[dict[str, Any]] = []
    for year in sorted(by_year):
        rel = f"year={year}/part.parquet"
        out_path = dataset / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(by_year[year], schema=_PANEL_SCHEMA)
        frame.write_parquet(out_path)
        partitions.append(
            {
                "year": year,
                "path": rel,
                "row_count": frame.height,
                "parquet_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            }
        )
    (dataset / "manifest.json").write_text(
        json.dumps({"dataset_id": name, "partitions": partitions}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset


def synthetic_cube(
    sessions: list[date],
    instrument_ids: list[str],
    *,
    close: float = 10000.0,
    ret_on: float = 0.0,
    ret_id: float = 0.0,
    adtv20: float = 1_000_000_000_000.0,
    vol60: float = 0.02,
    tick: float = 1.0,
) -> ResearchCube:
    """Minimal research cube with flat prices and controllable returns."""
    n_s, n_n = len(sessions), len(instrument_ids)
    shape = (n_s, n_n)
    ones_b = np.ones(shape, dtype=bool)
    zeros_b = np.zeros(shape, dtype=bool)
    arrays: dict[str, NDArray[Any]] = {
        "present": ones_b,
        "eligible": ones_b,
        "entry_blocked": zeros_b,
        "open_at_upper": zeros_b,
        "open_at_lower": zeros_b,
        "traded": ones_b,
        "volume": np.full(shape, 100000.0),
        "open": np.full(shape, close),
        "close": np.full(shape, close),
        "base_price": np.full(shape, close),
        "market_cap": np.full(shape, 1e12),
        "adtv20": np.full(shape, adtv20),
        "ret_vol60": np.full(shape, vol60),
        "tick_at_open": np.full(shape, tick),
        "sell_tax_rate": np.zeros(shape),
        "r_on": np.full(shape, ret_on),
        "r_id": np.full(shape, ret_id),
        "div_ret": np.zeros(shape),
        "ret_cc": np.zeros(shape),
    }
    return ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=list(sessions),
        instrument_ids=list(instrument_ids),
        arrays=arrays,
        exit_at=np.full(n_n, -1, dtype=np.int64),
        exit_halted=np.zeros(n_n, dtype=np.bool_),
    )
