"""Engine-event indexing: splits, exits, and dividend entitlement tests."""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import pytest

from src.backtest.events import ExitEvent, ExitKind, build_engine_events
from src.backtest.market import MarketArrays, load_market_arrays
from src.core.pit import PITDataError
from tests.unit.backtest.test_market import DAY0, DAY1, DAY2, _mrow, _write_panel

DAY3 = date(2020, 1, 9)

_EXITS_SCHEMA: dict[str, Any] = {
    "instrument_id": pl.String,
    "last_session": pl.Date,
    "last_close": pl.Int64,
    "last_volume": pl.Int64,
    "exit_kind": pl.String,
}


def _write_exits(panel: Path, rows: list[dict[str, Any]]) -> None:
    out_path = panel / "instrument_exits.parquet"
    frame = pl.DataFrame(rows, schema=_EXITS_SCHEMA) if rows else pl.DataFrame([], schema=_EXITS_SCHEMA)
    frame.write_parquet(out_path)


def _exit_row(
    instrument_id: str,
    last_session: date,
    *,
    last_close: int = 100,
    last_volume: int = 1000,
    exit_kind: str = "traded_exit",
) -> dict[str, Any]:
    return {
        "instrument_id": instrument_id,
        "last_session": last_session,
        "last_close": last_close,
        "last_volume": last_volume,
        "exit_kind": exit_kind,
    }


def _panel(tmp_path: Path, rows: list[dict[str, Any]], exits: list[dict[str, Any]]) -> MarketArrays:
    panel = _write_panel(tmp_path / "gold", "market_panel_test", rows)
    _write_exits(panel, exits)
    return load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")


def _div_row(
    instrument_id: str,
    ex_session: Any,
    pay_session: Any,
    *,
    dps_krw: Any = 100,
) -> dict[str, Any]:
    return {
        "instrument_id": instrument_id,
        "ex_session": ex_session,
        "pay_session": pay_session,
        "dps_krw": dps_krw,
    }


def _div_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "instrument_id": pl.String,
            "ex_session": pl.Date,
            "pay_session": pl.Date,
            "dps_krw": pl.Int64,
        },
    )


def test_share_factor_indexed_on_ex_session(tmp_path: Path) -> None:
    arrays = _panel(
        tmp_path,
        [
            _mrow(DAY0, "KRX:A"),
            _mrow(DAY1, "KRX:A"),
            _mrow(DAY2, "KRX:A", share_factor=50.0, base_price=100),
        ],
        [],
    )
    events = build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)
    assert events.share_factor_by_session[2] == ((0, 50.0, 100),)


def test_exit_keyed_after_last_session(tmp_path: Path) -> None:
    arrays = _panel(
        tmp_path,
        [
            _mrow(DAY0, "KRX:A"),
            _mrow(DAY1, "KRX:A"),
            _mrow(DAY2, "KRX:A"),
            _mrow(DAY0, "KRX:B"),
            _mrow(DAY1, "KRX:B"),
            _mrow(DAY2, "KRX:B"),
        ],
        [
            _exit_row("KRX:A", DAY1, last_close=110, exit_kind="halted_exit"),
            _exit_row("KRX:B", DAY2, last_close=220),
        ],
    )
    events = build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)
    assert events.exits_by_session[2] == (
        ExitEvent(instrument_idx=0, last_session_idx=1, last_close=110, kind=ExitKind.HALTED),
    )
    assert 3 not in events.exits_by_session


def test_dividends_none_flags_price_return_only(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    events = build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)
    assert events.dividends_integrated is False
    assert events.dividends_by_ex_session == {}


def test_dividend_paying_before_ex_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A"), _mrow(DAY2, "KRX:A")], [])
    dividends = _div_frame([_div_row("KRX:A", DAY1, DAY0)])
    with pytest.raises(PITDataError):
        build_engine_events(
            arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=dividends
        )


def test_non_session_pay_date_rolls_forward(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY2, "KRX:A")], [])
    dividends = pl.DataFrame(
        [
            {
                "instrument_id": "KRX:A",
                "ex_session": DAY0,
                "pay_session": DAY1,
                "dps_krw": 50,
            }
        ],
        schema={
            "instrument_id": pl.String,
            "ex_session": pl.Date,
            "pay_session": pl.Date,
            "dps_krw": pl.Int64,
        },
    )
    events = build_engine_events(
        arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=dividends
    )
    assert events.dividends_integrated is True
    assert events.dividends_by_ex_session[0][0].pay_session_idx == 1
    assert events.dividends_by_ex_session[0][0].dps_krw == 50


def test_missing_exits_table_yields_no_events(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A")])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    assert events.exits_by_session == {}


def test_unreadable_exits_table_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    (tmp_path / "gold" / "market_panel_test" / "instrument_exits.parquet").write_bytes(b"garbage")
    with pytest.raises(PITDataError):
        build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)


def test_exits_table_missing_columns_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    pl.DataFrame({"instrument_id": ["KRX:A"]}).write_parquet(
        tmp_path / "gold" / "market_panel_test" / "instrument_exits.parquet"
    )
    with pytest.raises(PITDataError):
        build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)


def test_exit_unknown_instrument_rejected(tmp_path: Path) -> None:
    arrays = _panel(
        tmp_path,
        [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")],
        [_exit_row("KRX:ZZZ", DAY0)],
    )
    with pytest.raises(PITDataError):
        build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)


def test_exit_unknown_session_rejected(tmp_path: Path) -> None:
    arrays = _panel(
        tmp_path,
        [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")],
        [_exit_row("KRX:A", DAY3)],
    )
    with pytest.raises(PITDataError):
        build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)


def test_exit_invalid_kind_rejected(tmp_path: Path) -> None:
    arrays = _panel(
        tmp_path,
        [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")],
        [_exit_row("KRX:A", DAY0, exit_kind="bogus")],
    )
    with pytest.raises(PITDataError):
        build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)


def test_exit_invalid_close_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    frame = pl.DataFrame(
        [{"instrument_id": "KRX:A", "last_session": DAY0, "last_close": None, "exit_kind": "traded_exit"}],
        schema={"instrument_id": pl.String, "last_session": pl.Date, "last_close": pl.Int64, "exit_kind": pl.String},
    )
    frame.write_parquet(tmp_path / "gold" / "market_panel_test" / "instrument_exits.parquet")
    with pytest.raises(PITDataError):
        build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)


def test_invalid_share_factor_rejected(tmp_path: Path) -> None:
    arrays = _panel(
        tmp_path,
        [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A", share_factor=float("inf"))],
        [],
    )
    with pytest.raises(PITDataError):
        build_engine_events(arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=None)


def test_dividend_table_missing_columns_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    with pytest.raises(PITDataError):
        build_engine_events(
            arrays=arrays,
            panel_dir=tmp_path / "gold" / "market_panel_test",
            dividends=pl.DataFrame({"instrument_id": ["KRX:A"]}),
        )


def test_dividend_unknown_instrument_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    with pytest.raises(PITDataError):
        build_engine_events(
            arrays=arrays,
            panel_dir=tmp_path / "gold" / "market_panel_test",
            dividends=_div_frame([_div_row("KRX:ZZZ", DAY0, DAY1)]),
        )


def test_dividend_unknown_ex_session_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    with pytest.raises(PITDataError):
        build_engine_events(
            arrays=arrays,
            panel_dir=tmp_path / "gold" / "market_panel_test",
            dividends=_div_frame([_div_row("KRX:A", DAY3, DAY3)]),
        )


def test_dividend_pay_after_panel_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    with pytest.raises(PITDataError):
        build_engine_events(
            arrays=arrays,
            panel_dir=tmp_path / "gold" / "market_panel_test",
            dividends=_div_frame([_div_row("KRX:A", DAY1, date(2020, 2, 1))]),
        )


def test_dividend_non_positive_dps_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    with pytest.raises(PITDataError):
        build_engine_events(
            arrays=arrays,
            panel_dir=tmp_path / "gold" / "market_panel_test",
            dividends=_div_frame([_div_row("KRX:A", DAY0, DAY1, dps_krw=0)]),
        )


def test_dividend_non_date_session_rejected(tmp_path: Path) -> None:
    arrays = _panel(tmp_path, [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")], [])
    dividends = pl.DataFrame(
        [{"instrument_id": "KRX:A", "ex_session": "2020-01-06", "pay_session": DAY1, "dps_krw": 10}],
        schema={
            "instrument_id": pl.String,
            "ex_session": pl.String,
            "pay_session": pl.Date,
            "dps_krw": pl.Int64,
        },
    )
    with pytest.raises(PITDataError):
        build_engine_events(
            arrays=arrays, panel_dir=tmp_path / "gold" / "market_panel_test", dividends=dividends
        )
