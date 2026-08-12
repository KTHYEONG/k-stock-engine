"""Calendar-aware labels: KRX-session timing and terminal availability."""
from __future__ import annotations

import math
from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.stocks.data.labels import LABEL_AVAILABLE_COLUMN, build_label_dataset
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.research.labels import LabelDefinition

SESSIONS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 8), date(2024, 1, 9)]
CALENDAR = KRXSessionCalendar(
    version="fixture-calendar",
    sessions=tuple(SESSIONS),
    generated_time=datetime(2026, 1, 1, tzinfo=UTC),
)
DEFINITION = LabelDefinition(
    name="fwd_ret_2d",
    entry_field="open",
    exit_field="close",
    horizon_sessions=2,
)


def base_panel(close: list[float], open_price: list[float] | None = None) -> pl.DataFrame:
    opens = open_price or [100.0 + i for i in range(len(close))]
    return pl.DataFrame(
        {
            "instrument_id": ["KRX:1"] * len(SESSIONS),
            "session": [datetime.combine(s, datetime.min.time(), tzinfo=UTC) for s in SESSIONS],
            "open": opens,
            "close": close,
        }
    )


class TestCalendarAwareLabels:
    def test_horizon_counts_sessions_not_calendar_days(self) -> None:
        # Sessions have a calendar gap (2024-01-05..2024-01-07 not traded).
        # Horizon=2 counts KRX sessions, not calendar days: the decision session
        # 01-02 exits at 01-04 (2 sessions later), not at 01-04+2 calendar days.
        frame = base_panel([105.0, 110.0, 115.0, 120.0, 125.0])
        out = build_label_dataset(frame, CALENDAR, DEFINITION)

        # Decision sessions whose 2-session horizon is complete: 01-02, 01-03, 01-04.
        assert out["session"].to_list() == SESSIONS[:3]
        assert out.columns == ["instrument_id", "session", "fwd_ret_2d", LABEL_AVAILABLE_COLUMN]
        # entry = next-session open; exit = close of T+horizon.
        expected = [
            math.log(115.0) - math.log(101.0),
            math.log(120.0) - math.log(102.0),
            math.log(125.0) - math.log(103.0),
        ]
        for row, exp in zip(out["fwd_ret_2d"].to_list(), expected, strict=True):
            assert row is not None
            assert abs(row - exp) < 1e-9

    def test_incomplete_future_horizon_is_absent(self) -> None:
        frame = base_panel([105.0, 110.0, 115.0, 120.0, 125.0])
        out = build_label_dataset(frame, CALENDAR, DEFINITION)
        # Terminal decision sessions 01-08 and 01-09 are absent.
        assert "2024-01-08" not in {s.isoformat() for s in out["session"].to_list()}
        assert "2024-01-09" not in {s.isoformat() for s in out["session"].to_list()}

    def test_every_label_has_terminal_label_available_time(self) -> None:
        frame = base_panel([105.0, 110.0, 115.0, 120.0, 125.0])
        out = build_label_dataset(frame, CALENDAR, DEFINITION)
        assert out[LABEL_AVAILABLE_COLUMN].null_count() == 0
        # Availability is at-or-after the terminal horizon session (06:31 UTC).
        terminal = datetime(2024, 1, 4, 6, 31, tzinfo=UTC)
        assert out[LABEL_AVAILABLE_COLUMN][0] >= terminal

    def test_missing_exit_price_drops_the_decision_row(self) -> None:
        # KRX:1 lacks the 2024-01-08 session (exit/entry for decisions 01-03/01-04).
        panel = base_panel([105.0, 110.0, 115.0, 120.0, 125.0])
        missing = panel.filter(
            (pl.col("session") != datetime(2024, 1, 8, tzinfo=UTC))
            | (pl.col("instrument_id") != "KRX:1")
        )
        extra = pl.DataFrame(
            {
                "instrument_id": ["KRX:2"] * 2,
                "session": [
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 3, tzinfo=UTC),
                ],
                "open": [90.0, 91.0],
                "close": [95.0, 96.0],
            }
        )
        out = build_label_dataset(pl.concat([missing, extra]), CALENDAR, DEFINITION)
        krx1 = out.filter(pl.col("instrument_id") == "KRX:1")
        # Only decision 01-02 has a complete horizon (exit 01-04 present).
        assert krx1["session"].to_list() == [SESSIONS[0]]
        assert out["label_available_time"].null_count() == 0

    def test_non_calendar_session_is_rejected(self) -> None:
        frame = base_panel([105.0, 110.0, 115.0, 120.0, 125.0])
        bad = pl.concat(
            [
                frame,
                pl.DataFrame(
                    {
                        "instrument_id": ["KRX:1"],
                        "session": [datetime(2024, 1, 6, tzinfo=UTC)],  # not a KRX session
                        "open": [50.0],
                        "close": [55.0],
                    }
                ),
            ]
        )
        with pytest.raises(ValueError, match="non-calendar sessions"):
            build_label_dataset(bad, CALENDAR, DEFINITION)

    def test_missing_price_columns_are_rejected(self) -> None:
        frame = base_panel([105.0, 110.0, 115.0, 120.0, 125.0]).drop("open")
        with pytest.raises(ValueError, match="price columns"):
            build_label_dataset(frame, CALENDAR, DEFINITION)
