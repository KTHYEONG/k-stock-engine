"""Calendar-aware labels: KRX-session timing and terminal availability."""
from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src.stocks.data.labels import (
    LABEL_AVAILABLE_COLUMN,
    build_label_dataset,
    build_multi_horizon_residual_label_dataset,
    label_available_time,
)
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

def _weekday_calendar(n_sessions: int = 70) -> KRXSessionCalendar:
    sessions: list[date] = []
    cursor = date(2024, 1, 2)
    while len(sessions) < n_sessions:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    return KRXSessionCalendar(
        version="fixture-weekday-calendar",
        sessions=tuple(sessions),
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _wide_base_panel(calendar: KRXSessionCalendar, n_tickers: int = 24) -> pl.DataFrame:
    rows: list[dict] = []
    for t in range(n_tickers):
        price = 100.0
        for session in calendar.sessions:
            price = max(10.0, price * 1.001)
            rows.append(
                {
                    "instrument_id": f"KRX:{t + 1:06d}",
                    "session": datetime.combine(session, datetime.min.time(), tzinfo=UTC),
                    "open": price,
                }
            )
    return pl.DataFrame(rows)


class TestMultiHorizonResidualLabels:
    def test_emits_key_aligned_ordered_multi_horizon_schema(self) -> None:
        calendar = _weekday_calendar()
        base = _wide_base_panel(calendar)
        out = build_multi_horizon_residual_label_dataset(base, calendar)
        assert out.columns == [
            "instrument_id",
            "session",
            "residual_o2o_5d",
            "relevance_5d",
            "label_available_time_5d",
            "residual_o2o_10d",
            "relevance_10d",
            "label_available_time_10d",
            "residual_o2o_15d",
            "relevance_15d",
            "label_available_time_15d",
        ]
        assert not out.is_empty()
        assert out["instrument_id"].n_unique() == 24
        assert not out.select(
            pl.any_horizontal(
                pl.col(c).is_null() for c in out.columns if c != "instrument_id"
            )
        ).to_series().any()

    def test_all_horizons_share_one_universe(self) -> None:
        calendar = _weekday_calendar()
        base = _wide_base_panel(calendar)
        out = build_multi_horizon_residual_label_dataset(base, calendar)
        keys = out.select("instrument_id", "session")
        for column in ("residual_o2o_5d", "residual_o2o_10d", "residual_o2o_15d"):
            assert out[column].is_not_null().all()
            assert out[column].is_finite().all()

    def test_each_horizon_has_independent_terminal_availability(self) -> None:
        calendar = _weekday_calendar()
        base = _wide_base_panel(calendar)
        out = build_multi_horizon_residual_label_dataset(base, calendar)
        ordered = out.sort(["instrument_id", "session"])
        expected_gaps = {
            "label_available_time_5d": 6,
            "label_available_time_10d": 11,
            "label_available_time_15d": 16,
        }
        for column, gap in expected_gaps.items():
            decision = datetime(2024, 1, 3, tzinfo=UTC)
            exit_session = calendar.sessions[1 + gap]
            expected = label_available_time(exit_session)
            row = ordered.filter(
                pl.col("instrument_id") == "KRX:000001"
            ).filter(pl.col("session") == decision)
            actual = row[column][0]
            assert actual == expected
            assert actual > decision

    def test_availability_monotonic_across_horizons(self) -> None:
        calendar = _weekday_calendar()
        base = _wide_base_panel(calendar)
        out = build_multi_horizon_residual_label_dataset(base, calendar)
        earlier = out["label_available_time_5d"]
        later = out["label_available_time_10d"]
        latest = out["label_available_time_15d"]
        assert (later > earlier).all()
        assert (latest > later).all()

    def test_rejects_invalid_horizon_sets(self) -> None:
        calendar = _weekday_calendar()
        base = _wide_base_panel(calendar)
        with pytest.raises(ValueError, match="non-empty"):
            build_multi_horizon_residual_label_dataset(base, calendar, horizons=())
        with pytest.raises(ValueError, match="ascending and unique"):
            build_multi_horizon_residual_label_dataset(base, calendar, horizons=(5, 5))
        with pytest.raises(ValueError, match="ascending and unique"):
            build_multi_horizon_residual_label_dataset(base, calendar, horizons=(10, 5))
        with pytest.raises(ValueError, match="unsupported"):
            build_multi_horizon_residual_label_dataset(base, calendar, horizons=(7,))

    def test_publisher_requires_ordered_schema_and_control_availability(self, tmp_path) -> None:
        from src.stocks.data.labels import (
            publish_multi_horizon_residual_label_dataset,
        )
        from src.storage.parquet_datasets import ParquetDatasetStore

        calendar = _weekday_calendar()
        base = _wide_base_panel(calendar)
        labels = build_multi_horizon_residual_label_dataset(base, calendar)
        result = publish_multi_horizon_residual_label_dataset(
            labels,
            destination_root=tmp_path / "labels",
            dataset_id="multi_5_10_15d_v1",
            base_panel_hash="base-hash",
            calendar_hash="cal-hash",
        )
        assert result.row_count == labels.height
        manifest = result.manifest
        assert manifest.label_definition == "residual_o2o_multi_5_10_15d"
        assert manifest.label_horizon_sessions == 5
        store = ParquetDatasetStore(tmp_path / "labels")
        assert store.content_columns(result.dataset_id) == labels.columns

    def test_multi_horizon_uses_terminal_availability_not_decision_session(self) -> None:
        calendar = _weekday_calendar()
        base = _wide_base_panel(calendar)
        out = build_multi_horizon_residual_label_dataset(base, calendar)
        decision = datetime(2024, 1, 2, tzinfo=UTC)
        row = out.filter(
            (pl.col("session") == decision)
            & (pl.col("instrument_id") == "KRX:000001")
        )
        assert row.height == 1
        for column in (
            "label_available_time_5d",
            "label_available_time_10d",
            "label_available_time_15d",
        ):
            available = row[column][0]
            assert available > decision
        assert row["label_available_time_5d"][0] < row["label_available_time_10d"][0]
        assert row["label_available_time_10d"][0] < row["label_available_time_15d"][0]
