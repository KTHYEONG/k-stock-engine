"""Deterministic stock fixture builders shared by unit tests."""
from __future__ import annotations

from datetime import datetime, timedelta, UTC

import polars as pl

from src.core.instruments import AssetKind
from src.stocks.ml.dataset import DatasetManifest, make_manifest


def stock_instrument_df(
    n_sessions: int = 40,
    n_tickers: int = 5,
    horizon: int = 5,
    start: datetime | None = None,
) -> pl.DataFrame:
    """Deterministic daily bars + a momentum feature + forward label."""
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for t in range(n_tickers):
        for s in range(n_sessions):
            obs = start + timedelta(days=s)
            rows.append(
                {
                    "session_index": s,
                    "date": obs,
                    "instrument_id": f"KRX:0{t + 1:05d}",
                    "close": 100.0 + float((t * 7 + s) % 20),
                    "feature_momentum_5d": float((t + s) % 7) / 7.0,
                    "label_fwd_ret": float((t + s) % 3 - 1) / 10.0,
                    "is_universe": True,
                }
            )
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.col("label_fwd_ret").shift(-horizon).over("instrument_id").alias("label_eligible"),
    )


def stock_manifest(
    columns: list[str] | None = None,
    asset_kind: AssetKind = AssetKind.STOCK,
    feature_set: str = "stock_alpha_v1",
    horizon: int = 5,
    decision_time: datetime | None = None,
) -> DatasetManifest:
    cols = columns or [
        "session_index",
        "date",
        "instrument_id",
        "feature_momentum_5d",
        "label_fwd_ret",
    ]
    return make_manifest(
        asset_kind=asset_kind,
        columns=cols,
        feature_set=feature_set,
        label_definition="fwd_ret_5d",
        label_horizon_sessions=horizon,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 1, tzinfo=UTC),
        row_count=len(cols) * 10,
        generated_time=decision_time,
    )
