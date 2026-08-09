"""Deterministic stock fixture builders shared by unit tests."""
from __future__ import annotations

from datetime import datetime, timedelta, UTC

import polars as pl

from src.core.instruments import AssetKind
from src.core.datasets import DatasetManifest, make_manifest


def stock_instrument_df(
    n_sessions: int = 40,
    n_tickers: int = 5,
    horizon: int = 5,
    start: datetime | None = None,
) -> pl.DataFrame:
    """Deterministic point-in-time daily bars for stock research tests."""
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for t in range(n_tickers):
        for s in range(n_sessions):
            obs = start + timedelta(days=s)
            close = 100.0 + float((t * 7 + s) % 20)
            rows.append(
                {
                    "session_index": s,
                    "session": obs,
                    "instrument_id": f"KRX:0{t + 1:05d}",
                    "observation_time": obs.replace(hour=15, minute=30, tzinfo=UTC),
                    "available_time": obs.replace(hour=15, minute=31, tzinfo=UTC),
                    "open": close - 1.0,
                    "high": close + 1.0,
                    "low": close - 1.5,
                    "close": close,
                    "volume": 1_000_000.0 + float(t) * 100_000.0,
                    "trading_value": close * (1_000_000.0 + float(t) * 100_000.0),
                    "market_cap": close * 10_000_000.0,
                    "feature_momentum_5d": float((t + s) % 7) / 7.0,
                    "is_universe": True,
                }
            )
    return pl.DataFrame(rows)


def stock_manifest(
    columns: list[str] | None = None,
    asset_kind: AssetKind = AssetKind.STOCK,
    feature_set: str = "stock_alpha_v1",
    horizon: int = 5,
    decision_time: datetime | None = None,
) -> DatasetManifest:
    cols = columns or [
        "session_index",
        "session",
        "instrument_id",
        "feature_momentum_5d",
    ]
    return make_manifest(
        asset_kind=asset_kind,
        columns=cols,
        feature_set=feature_set,
        label_definition="fwd_ret_5d",
        label_horizon_sessions=horizon,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 1, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=len(cols) * 10,
        generated_time=decision_time,
    )
