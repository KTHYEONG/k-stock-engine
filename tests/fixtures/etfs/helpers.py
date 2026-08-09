"""Deterministic ETF/index fixture builder for parity regression."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl


def make_etf_fixture(
    n_days: int = 60,
    seed: int = 7,
    index_level: float = 2500.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build an index frame + bull/bear ETF frames with OHLCV columns.

    Mirrors the KRX raw naming used by the legacy ETF backtester
    (``OPNPRC_IDX`` etc. for the index, plain ``open/high/low/close`` for ETFs).
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2)

    def session_dates() -> list[datetime]:
        days: list[datetime] = []
        d = start
        while len(days) < n_days:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        return days

    dates = session_dates()

    index_rows: list[dict] = []
    close = index_level
    for d in dates:
        open_px = close * (1 + rng.normal(0, 0.002))
        close = open_px * (1 + rng.normal(0, 0.006))
        high = max(open_px, close) * (1 + abs(rng.normal(0, 0.002)))
        low = min(open_px, close) * (1 - abs(rng.normal(0, 0.002)))
        index_rows.append(
            {
                "ticker": "KOSPI",
                "date": d,
                "OPNPRC_IDX": f"{open_px:.2f}",
                "HGPRC_IDX": f"{high:.2f}",
                "LWPRC_IDX": f"{low:.2f}",
                "CLSPRC_IDX": f"{close:.2f}",
            }
        )

    etf_rows: list[dict] = []
    for ticker, drift in (("069500", 0.0004), ("114800", -0.0002)):
        px = 1000.0 if ticker == "069500" else 5000.0
        for d in dates:
            open_px = px * (1 + rng.normal(0, 0.002))
            px = open_px * (1 + rng.normal(drift, 0.005))
            high = max(open_px, px) * (1 + abs(rng.normal(0, 0.002)))
            low = min(open_px, px) * (1 - abs(rng.normal(0, 0.002)))
            etf_rows.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "open": f"{open_px:.2f}",
                    "high": f"{high:.2f}",
                    "low": f"{low:.2f}",
                    "close": f"{px:.2f}",
                    "volume": float(rng.integers(1000, 50000)),
                }
            )

    return pl.DataFrame(index_rows), pl.DataFrame(etf_rows)


def preprocess_index(index_df: pl.DataFrame) -> pl.DataFrame:
    """Mirror legacy ETFBacktester index preprocessing (OHLC aliases)."""
    return index_df.select(
        [
            pl.col("date"),
            pl.col("OPNPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("open"),
            pl.col("HGPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("high"),
            pl.col("LWPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("low"),
            pl.col("CLSPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("close"),
        ]
    ).filter(pl.col("close").is_not_null())
