"""Point-in-time view: session slices, as-of cuts, and perturbation tests."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.backtest.market import MarketArrays, load_market_arrays
from src.backtest.view import AsOfTable, PITView
from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from tests.unit.backtest.test_market import DAY0, DAY1, DAY2, _mrow, _write_panel

def _five_session_panel(root: Path, name: str, close_after: int | None = None) -> MarketArrays:
    sessions = [date(2020, 1, 6 + offset) for offset in range(5)]
    rows = []
    for idx, session in enumerate(sessions):
        close = 100 + idx
        if close_after is not None and idx > close_after:
            close = 10_000 + idx
        rows.append(_mrow(session, "KRX:A", close=close))
        rows.append(_mrow(session, "KRX:B", close=200 + idx))
    panel = _write_panel(root / "gold", name, rows)
    return load_market_arrays(panel_dir=panel, cache_root=root / "cache")


def _decision(session: date, hour: int = 18, minute: int = 0) -> datetime:
    return datetime.combine(session, time(hour, minute), tzinfo=KRX_TZ)


def _asof_frame(values: list[tuple[datetime, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"available_at": [when for when, _ in values], "v": [val for _, val in values]},
        schema={"available_at": pl.Datetime("us", "Asia/Seoul"), "v": pl.Int64},
    )


def test_field_slice_ends_at_t(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    view = PITView(arrays=arrays, t=2, decision_time=_decision(DAY2), asof_tables={})
    assert view.t == 2
    close = view.field("close")
    assert close.shape == (3, 2)
    assert close.tolist() == [[100, 200], [101, 201], [102, 202]]
    with pytest.raises(ValueError, match="read-only"):
        close[0, 0] = -1


def test_asof_cut_excludes_later_rows(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    table = AsOfTable(
        _asof_frame([
            (datetime(2020, 1, 6, 9, 0, tzinfo=KRX_TZ), 1),
            (datetime(2020, 1, 7, 9, 0, tzinfo=KRX_TZ), 2),
            (datetime(2020, 1, 8, 9, 0, tzinfo=KRX_TZ), 3),
        ])
    )
    view = PITView(
        arrays=arrays, t=1, decision_time=_decision(DAY1), asof_tables={"facts": table}
    )
    assert view.table("facts")["v"].to_list() == [1, 2]


def test_future_perturbation_invariance(tmp_path: Path) -> None:
    base = _five_session_panel(tmp_path / "base", "market_panel_base")
    perturbed = _five_session_panel(tmp_path / "pert", "market_panel_pert", close_after=2)
    assert not np.array_equal(
        base.int_fields["close"][3:], perturbed.int_fields["close"][3:]
    )
    for t in range(3):
        session = date(2020, 1, 6 + t)
        first = PITView(arrays=base, t=t, decision_time=_decision(session), asof_tables={})
        second = PITView(arrays=perturbed, t=t, decision_time=_decision(session), asof_tables={})
        assert np.array_equal(first.field("close"), second.field("close"))
        assert np.array_equal(first.field("market"), second.field("market"))


def test_naive_decision_time_rejected(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    with pytest.raises(PITDataError):
        PITView(
            arrays=arrays,
            t=1,
            decision_time=datetime(2020, 1, 7, 18, 0),
            asof_tables={},
        )


def test_early_decision_time_rejected(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    with pytest.raises(PITDataError):
        PITView(arrays=arrays, t=1, decision_time=_decision(DAY1, 17, 59), asof_tables={})


def test_session_index_out_of_range_rejected(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    with pytest.raises(PITDataError):
        PITView(arrays=arrays, t=5, decision_time=_decision(date(2020, 1, 10)), asof_tables={})
    with pytest.raises(PITDataError):
        PITView(arrays=arrays, t=True, decision_time=_decision(DAY0), asof_tables={})


def test_unknown_field_raises_key_error(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    view = PITView(arrays=arrays, t=0, decision_time=_decision(DAY0), asof_tables={})
    with pytest.raises(KeyError):
        view.field("no_such_field")
    with pytest.raises(KeyError):
        view.table("no_such_table")


def test_float_and_bool_fields_are_read_only_slices(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    view = PITView(arrays=arrays, t=4, decision_time=_decision(date(2020, 1, 10)), asof_tables={})
    assert view.field("share_factor").shape == (5, 2)
    assert view.field("eligible").shape == (5, 2)
    assert view.field("market").shape == (5, 2)


def test_asof_table_requires_tz_aware_column(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(PITDataError):
        AsOfTable(pl.DataFrame({"v": [1, 2]}))
    naive = pl.DataFrame(
        {"available_at": [datetime(2020, 1, 6, 9, 0), datetime(2020, 1, 7, 9, 0)]},
        schema={"available_at": pl.Datetime("us")},
    )
    with pytest.raises(PITDataError):
        AsOfTable(naive)


def test_asof_upto_rejects_naive_decision_time(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    assert arrays.sessions[0] == DAY0
    table = AsOfTable(_asof_frame([(datetime(2020, 1, 6, 9, 0, tzinfo=KRX_TZ), 1)]))
    with pytest.raises(PITDataError):
        table.upto(datetime(2020, 1, 6, 18, 0))


def test_view_hides_market_arrays(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    view = PITView(arrays=arrays, t=0, decision_time=_decision(DAY0), asof_tables={})
    assert not hasattr(view, "arrays")
    assert not hasattr(view, "market_arrays")
    assert isinstance(view.t, int)


_SNAPSHOT_AT = datetime(2026, 9, 24, 18, 0, tzinfo=KRX_TZ)


def _industry_table(*, delisted: date | None = None) -> AsOfTable:
    return AsOfTable(
        pl.DataFrame(
            {
                "instrument_id": ["KRX:005930"],
                "industry_name": ["전기·전자"],
                "available_at": [_SNAPSHOT_AT],
                "known_since": [_SNAPSHOT_AT],
                "attribute_basis": ["static_snapshot"],
                "delisted_on": [delisted],
            },
            schema={
                "instrument_id": pl.String,
                "industry_name": pl.String,
                "available_at": pl.Datetime("us", "Asia/Seoul"),
                "known_since": pl.Datetime("us", "Asia/Seoul"),
                "attribute_basis": pl.String,
                "delisted_on": pl.Date,
            },
        )
    )


def _industry_view(tmp_path: Path, table: AsOfTable) -> PITView:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    view = PITView(
        arrays=arrays, t=0, decision_time=_decision(DAY0), asof_tables={"industry": table}
    )
    assert view.allow_static_industry is False
    return view


def test_industry_strict_pit_rejects_future_snapshot(tmp_path: Path) -> None:
    view = _industry_view(tmp_path, _industry_table())
    with pytest.raises(PITDataError):
        view.industry_as_of(datetime(2020, 1, 2, 18, 0, tzinfo=KRX_TZ), allow_static=False)


def test_industry_opt_in_returns_rows_and_records_assumption(tmp_path: Path) -> None:
    view = _industry_view(tmp_path, _industry_table())
    rows = view.industry_as_of(datetime(2020, 1, 2, 18, 0, tzinfo=KRX_TZ), allow_static=True)
    assert rows.height == 1
    assert rows["instrument_id"].to_list() == ["KRX:005930"]
    assert view.assumptions == ("static_industry_snapshot:2026-09-24",)


def test_industry_opt_in_keeps_delisted_names(tmp_path: Path) -> None:
    view = _industry_view(tmp_path, _industry_table(delisted=date(2024, 5, 1)))
    rows = view.industry_as_of(datetime(2020, 1, 2, 18, 0, tzinfo=KRX_TZ), allow_static=True)
    assert rows.height == 1
    assert rows["delisted_on"].to_list() == [date(2024, 5, 1)]


def test_industry_after_snapshot_records_no_assumption(tmp_path: Path) -> None:
    view = _industry_view(tmp_path, _industry_table())
    rows = view.industry_as_of(datetime(2026, 9, 25, 18, 0, tzinfo=KRX_TZ), allow_static=False)
    assert rows.height == 1
    assert view.assumptions == ()


def test_industry_opt_in_rejected_for_non_static_table(tmp_path: Path) -> None:
    arrays = _five_session_panel(tmp_path, "market_panel_test")
    table = AsOfTable(
        pl.DataFrame(
            {
                "instrument_id": ["KRX:005930"],
                "available_at": [_SNAPSHOT_AT],
            },
            schema={
                "instrument_id": pl.String,
                "available_at": pl.Datetime("us", "Asia/Seoul"),
            },
        )
    )
    view = PITView(
        arrays=arrays, t=0, decision_time=_decision(DAY0), asof_tables={"industry": table}
    )
    with pytest.raises(PITDataError):
        view.industry_as_of(datetime(2020, 1, 2, 18, 0, tzinfo=KRX_TZ), allow_static=True)
