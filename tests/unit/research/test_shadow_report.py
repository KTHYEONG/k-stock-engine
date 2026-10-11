"""Shadow slippage table: adverse signs, missing captures and open mismatches."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.core.time import KRX_TZ

DAY = date(2026, 10, 5)
OPEN = datetime(2026, 10, 5, 9, 0, tzinfo=KRX_TZ)


def _rules() -> object:
    from src.core.market_rules import load_krx_market_rules

    return load_krx_market_rules(Path("config/market/krx_market_rules.toml"))


def _fills() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session": [DAY, DAY],
            "instrument_id": ["KRX:005930", "KRX:000660"],
            "side": ["buy", "sell"],
            "quantity": [10, 20],
            "price": [10010, 9990],
        },
        schema={"session": pl.Date, "instrument_id": pl.String, "side": pl.String, "quantity": pl.Int64, "price": pl.Int64},
    )


def _market() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session": [DAY, DAY],
            "instrument_id": ["KRX:005930", "KRX:000660"],
            "open": [10000, 10000],
            "market": ["KOSPI", "KOSPI"],
            "adtv20": [1_000_000_000.0, 1_000_000_000.0],
        },
        schema={"session": pl.Date, "instrument_id": pl.String, "open": pl.Int64, "market": pl.String, "adtv20": pl.Float64},
    )


def _auction(*, captured: int = 10000, expected: int | None = 9950) -> pl.DataFrame:
    rows = [
        {
            "session": DAY, "symbol": "005930", "kind": "expected",
            "scheduled_at": OPEN - timedelta(minutes=1), "received_at": OPEN - timedelta(minutes=1),
            "output1": json.dumps({"antc_cnpr": str(expected)}), "output2": "[]",
        },
        {
            "session": DAY, "symbol": "000660", "kind": "expected",
            "scheduled_at": OPEN - timedelta(minutes=1), "received_at": OPEN - timedelta(minutes=1),
            "output1": json.dumps({"antc_cnpr": str(expected)}), "output2": "[]",
        },
        {
            "session": DAY, "symbol": "005930", "kind": "open",
            "scheduled_at": OPEN, "received_at": OPEN,
            "output1": json.dumps({"stck_oprc": str(captured)}), "output2": "{}",
        },
        {
            "session": DAY, "symbol": "000660", "kind": "open",
            "scheduled_at": OPEN, "received_at": OPEN,
            "output1": json.dumps({"stck_oprc": str(captured)}), "output2": "{}",
        },
    ]
    if expected is None:
        rows = rows[2:]
    return pl.DataFrame(
        rows,
        schema={"session": pl.Date, "symbol": pl.String, "kind": pl.String,
                "scheduled_at": pl.Datetime("us", "Asia/Seoul"), "received_at": pl.Datetime("us", "Asia/Seoul"),
                "output1": pl.String, "output2": pl.String},
    )


def test_adverse_sign_convention() -> None:
    from src.research.shadow_report import slippage_table

    table = slippage_table(_fills(), _auction(), _market(), _rules())
    by_inst = {row["instrument_id"]: row for row in table.to_dicts()}
    assert by_inst["KRX:005930"]["model_minus_open_ticks"] == 1.0
    assert by_inst["KRX:000660"]["model_minus_open_ticks"] == 1.0


def test_missing_capture_keeps_fill() -> None:
    from src.research.shadow_report import slippage_table

    empty = pl.DataFrame(
        [], schema={"session": pl.Date, "symbol": pl.String, "kind": pl.String,
                    "scheduled_at": pl.Datetime("us", "Asia/Seoul"), "received_at": pl.Datetime("us", "Asia/Seoul"),
                    "output1": pl.String, "output2": pl.String},
    )
    table = slippage_table(_fills(), empty, _market(), _rules())
    assert table.height == 2
    assert table["captured_open"].to_list() == [None, None]
    assert table["model_minus_open_ticks"].to_list() == [1.0, 1.0]


def test_mismatch_flagged() -> None:
    from src.research.shadow_report import slippage_table

    table = slippage_table(_fills(), _auction(captured=10200), _market(), _rules())
    assert table["open_mismatch"].to_list() == [True, True]
    calm = slippage_table(_fills(), _auction(captured=10000), _market(), _rules())
    assert calm["open_mismatch"].to_list() == [False, False]


def test_frames_validated() -> None:
    import pytest

    from src.research.shadow_report import slippage_table

    with pytest.raises(ValueError, match="fills frame"):
        slippage_table(pl.DataFrame({"session": [DAY]}), _auction(), _market(), _rules())
    with pytest.raises(ValueError, match="market frame"):
        slippage_table(_fills(), _auction(), pl.DataFrame({"session": [DAY]}), _rules())


def test_malformed_rows_skipped() -> None:
    from src.research.shadow_report import slippage_table

    auction = pl.DataFrame(
        [
            {"session": DAY, "symbol": "005930", "kind": "open",
             "scheduled_at": OPEN, "received_at": OPEN, "output1": "not json", "output2": "{}"},
            {"session": DAY, "symbol": "005930", "kind": "open",
             "scheduled_at": OPEN, "received_at": OPEN, "output1": json.dumps([1, 2]), "output2": "{}"},
            {"session": DAY, "symbol": "005930", "kind": "open",
             "scheduled_at": OPEN, "received_at": OPEN, "output1": json.dumps({}), "output2": "{}"},
        ],
        schema={"session": pl.Date, "symbol": pl.String, "kind": pl.String,
                "scheduled_at": pl.Datetime("us", "Asia/Seoul"), "received_at": pl.Datetime("us", "Asia/Seoul"),
                "output1": pl.String, "output2": pl.String},
    )
    market = pl.DataFrame(
        {"session": [DAY], "instrument_id": ["KRX:005930"], "open": [0], "market": ["KOSPI"], "adtv20": [1.0]},
        schema={"session": pl.Date, "instrument_id": pl.String, "open": pl.Int64, "market": pl.String, "adtv20": pl.Float64},
    )
    table = slippage_table(_fills().head(1), auction, market, _rules())
    assert table["captured_open"].to_list() == [None]
    assert table["official_open"].to_list() == [None]
    assert table["model_minus_open_ticks"].to_list() == [None]


def test_unparseable_cells_skipped() -> None:
    from src.research.shadow_report import slippage_table

    market = pl.DataFrame(
        {"session": [DAY, DAY], "instrument_id": ["KRX:005930", "KRX:000660"], "open": ["10000", None], "adtv20": ["lots", None]},
    )
    auction = pl.DataFrame(
        {"session": [DAY, DAY], "symbol": ["005930", "000660"], "kind": ["expected", "expected"],
         "scheduled_at": ["x", "soon"],
         "output1": [json.dumps({"other": 1}), json.dumps({"antc_cnpr": "9999"})],
         "output2": ["[]", "[]"]},
    )
    table = slippage_table(_fills(), auction, market, _rules())
    by_inst = {row["instrument_id"]: row for row in table.to_dicts()}
    assert by_inst["KRX:005930"]["official_open"] == 10000
    assert by_inst["KRX:005930"]["notional_over_adtv20"] is None
    assert by_inst["KRX:000660"]["official_open"] is None
