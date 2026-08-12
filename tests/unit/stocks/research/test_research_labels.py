"""Next-open to forward-close label semantics tests."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from src.stocks.research.labels import LabelDefinition


def test_label_is_log_next_open_to_forward_close() -> None:
    frame = pl.DataFrame(
        {
            "instrument_id": ["KRX:1"] * 6,
            "session": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)],
            "open": [99.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 120.0],
        }
    )
    out = LabelDefinition("fwd_ret_5d", "open", "close", 5).apply(frame)
    assert out["fwd_ret_5d"].to_list()[0] == pytest.approx(
        math.log(frame["close"].to_list()[5] / frame["open"].to_list()[1])
    )


def test_label_sorts_unsorted_panel_deterministically() -> None:
    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)]
    rows = [
        {
            "instrument_id": instrument,
            "session": sessions[i],
            "open": 99.0 + i,
            "close": 100.0 + i,
        }
        for instrument in ("KRX:A", "KRX:B")
        for i in range(6)
    ]
    shuffled = pl.DataFrame(rows).sample(fraction=1.0, seed=3, shuffle=True)
    out = LabelDefinition("fwd_ret_5d", "open", "close", 2).apply(shuffled)
    assert out.sort(["instrument_id", "session"]).equals(
        LabelDefinition("fwd_ret_5d", "open", "close", 2).apply(pl.DataFrame(rows))
    )


def test_label_requires_columns() -> None:
    frame = pl.DataFrame(
        {"instrument_id": ["KRX:1"], "session": [datetime(2024, 1, 1, tzinfo=UTC)]}
    )
    with pytest.raises(ValueError, match="open"):
        LabelDefinition("bad", "open", "close", 5).apply(frame)
