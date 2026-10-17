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
            "output1": json.dumps({"askp1": "9940", "bidp1": "9950"}),
            "output2": json.dumps({"antc_cnpr": str(expected), "stck_oprc": "0"}),
        },
        {
            "session": DAY, "symbol": "000660", "kind": "expected",
            "scheduled_at": OPEN - timedelta(minutes=1), "received_at": OPEN - timedelta(minutes=1),
            "output1": json.dumps({"askp1": "9940", "bidp1": "9950"}),
            "output2": json.dumps({"antc_cnpr": str(expected), "stck_oprc": "0"}),
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


def _expected_auction(*, output1: str, output2: str | None, offset_min: int = 1) -> pl.DataFrame:
    rows = [
        {
            "session": DAY, "symbol": "005930", "kind": "expected",
            "scheduled_at": OPEN - timedelta(minutes=offset_min), "received_at": OPEN - timedelta(minutes=offset_min),
            "output1": output1, "output2": output2,
        },
        {
            "session": DAY, "symbol": "000660", "kind": "expected",
            "scheduled_at": OPEN - timedelta(minutes=offset_min), "received_at": OPEN - timedelta(minutes=offset_min),
            "output1": output1, "output2": output2,
        },
        {
            "session": DAY, "symbol": "005930", "kind": "open",
            "scheduled_at": OPEN, "received_at": OPEN,
            "output1": json.dumps({"stck_oprc": "10000"}), "output2": "{}",
        },
        {
            "session": DAY, "symbol": "000660", "kind": "open",
            "scheduled_at": OPEN, "received_at": OPEN,
            "output1": json.dumps({"stck_oprc": "10000"}), "output2": "{}",
        },
    ]
    return pl.DataFrame(
        rows,
        schema={"session": pl.Date, "symbol": pl.String, "kind": pl.String,
                "scheduled_at": pl.Datetime("us", "Asia/Seoul"), "received_at": pl.Datetime("us", "Asia/Seoul"),
                "output1": pl.String, "output2": pl.String},
    )


def test_expected_price_read_from_output2_object() -> None:
    from src.research.shadow_report import slippage_table

    table = slippage_table(_fills(), _auction(), _market(), _rules())
    by_inst = {row["instrument_id"]: row for row in table.to_dicts()}
    assert by_inst["KRX:005930"]["expected_minus_open_ticks_m1"] == -5.0
    assert by_inst["KRX:000660"]["expected_minus_open_ticks_m1"] == 5.0


def test_expected_price_from_list_shaped_output2() -> None:
    from src.research.shadow_report import slippage_table

    output2 = json.dumps([{"antc_cnpr": "9950"}])
    table = slippage_table(_fills(), _expected_auction(output1=json.dumps({"askp1": "1"}), output2=output2), _market(), _rules())
    by_inst = {row["instrument_id"]: row for row in table.to_dicts()}
    assert by_inst["KRX:005930"]["expected_minus_open_ticks_m1"] == -5.0
    assert by_inst["KRX:000660"]["expected_minus_open_ticks_m1"] == 5.0


def test_zero_antc_cnpr_is_missing() -> None:
    from src.research.shadow_report import slippage_table

    output2 = json.dumps({"antc_cnpr": "0", "stck_oprc": "10000"})
    table = slippage_table(_fills(), _expected_auction(output1=json.dumps({"askp1": "1"}), output2=output2), _market(), _rules())
    assert table["expected_minus_open_ticks_m1"].to_list() == [None, None]
    for bad_value in ("abc", "", "  ", None):
        bad_block = json.dumps({"antc_cnpr": bad_value, "stck_oprc": "10000"})
        bad_table = slippage_table(
            _fills(), _expected_auction(output1=json.dumps({"askp1": "1"}), output2=bad_block), _market(), _rules()
        )
        assert bad_table["expected_minus_open_ticks_m1"].to_list() == [None, None]
    missing = json.dumps({"stck_oprc": "10000"})
    missing_table = slippage_table(
        _fills(), _expected_auction(output1=json.dumps({"askp1": "1"}), output2=missing), _market(), _rules()
    )
    assert missing_table["expected_minus_open_ticks_m1"].to_list() == [None, None]


def test_output1_antc_cnpr_ignored() -> None:
    from src.research.shadow_report import slippage_table

    table = slippage_table(
        _fills(),
        _expected_auction(output1=json.dumps({"antc_cnpr": "9950"}), output2="[]"),
        _market(), _rules(),
    )
    assert table["expected_minus_open_ticks_m1"].to_list() == [None, None]


def test_malformed_output2_skipped() -> None:
    from src.research.shadow_report import slippage_table

    for bad in ("not json", "[]", "[1]", "5", None):
        table = slippage_table(_fills(), _expected_auction(output1=json.dumps({"askp1": "1"}), output2=bad), _market(), _rules())
        assert table["expected_minus_open_ticks_m1"].to_list() == [None, None]
        assert table["captured_open"].to_list() == [10000, 10000]


def test_invalid_output1_does_not_drop_expected() -> None:
    from src.research.shadow_report import slippage_table

    output2 = json.dumps({"antc_cnpr": "9950", "stck_oprc": "0"})
    table = slippage_table(_fills(), _expected_auction(output1="not json", output2=output2), _market(), _rules())
    by_inst = {row["instrument_id"]: row for row in table.to_dicts()}
    assert by_inst["KRX:005930"]["expected_minus_open_ticks_m1"] == -5.0
    assert by_inst["KRX:000660"]["expected_minus_open_ticks_m1"] == 5.0


def test_frame_without_output2_column() -> None:
    from src.research.shadow_report import slippage_table

    auction = _auction().drop("output2")
    assert "output2" not in auction.columns
    table = slippage_table(_fills(), auction, _market(), _rules())
    assert table["captured_open"].to_list() == [10000, 10000]
    for offset in (-20, -10, -5, -2, -1):
        assert table[f"expected_minus_open_ticks_m{abs(offset)}"].to_list() == [None, None]


def test_all_offsets_populated() -> None:
    from src.core.time import OPEN_SNAPSHOT_OFFSETS_MIN

    from src.research.shadow_report import slippage_table

    rows: list[dict[str, object]] = []
    for symbol in ("005930", "000660"):
        for offset in OPEN_SNAPSHOT_OFFSETS_MIN:
            antc = 10000 + int(offset) * 10
            rows.append(
                {
                    "session": DAY, "symbol": symbol, "kind": "expected",
                    "scheduled_at": OPEN + timedelta(minutes=int(offset)),
                    "received_at": OPEN + timedelta(minutes=int(offset)),
                    "output1": json.dumps({"askp1": "1"}),
                    "output2": json.dumps({"antc_cnpr": str(antc), "stck_oprc": "0"}),
                }
            )
    rows.extend(
        {
            "session": DAY, "symbol": symbol, "kind": "open",
            "scheduled_at": OPEN, "received_at": OPEN,
            "output1": json.dumps({"stck_oprc": "10000"}), "output2": "{}",
        }
        for symbol in ("005930", "000660")
    )
    auction = pl.DataFrame(
        rows,
        schema={"session": pl.Date, "symbol": pl.String, "kind": pl.String,
                "scheduled_at": pl.Datetime("us", "Asia/Seoul"), "received_at": pl.Datetime("us", "Asia/Seoul"),
                "output1": pl.String, "output2": pl.String},
    )
    table = slippage_table(_fills(), auction, _market(), _rules())
    by_inst = {row["instrument_id"]: row for row in table.to_dicts()}
    for offset in OPEN_SNAPSHOT_OFFSETS_MIN:
        col = f"expected_minus_open_ticks_m{abs(int(offset))}"
        assert by_inst["KRX:005930"][col] == float(int(offset))
        assert by_inst["KRX:000660"][col] == float(-int(offset))
