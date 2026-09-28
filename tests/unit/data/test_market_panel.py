"""Gold market-panel materialization tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.core.market_rules import load_krx_market_rules
from src.core.time import KRX_TZ
from src.data.market_panel import MarketPanelPolicy, materialize_market_panel
from src.core.pit import PITDataError

RULES = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))

DAY0 = date(2020, 1, 6)
DAY1 = date(2020, 1, 7)
DAY2 = date(2020, 1, 8)


def _sessions(start: date, count: int) -> tuple[date, ...]:
    return tuple(start + timedelta(days=index) for index in range(count))


def _drow(
    session: date,
    ticker: str,
    *,
    market: str = "KOSPI",
    open_price: int = 10500,
    high: int = 11200,
    low: int = 10300,
    close: int = 11000,
    change: int = 1000,
    volume: int = 1000,
    trading_value: int = 11000000,
    market_cap: int = 660000000000,
    listed_shares: int = 60000000,
    price_state: str = "tradable",
) -> dict[str, object]:
    return {
        "session": session,
        "instrument_id": f"KRX:{ticker}",
        "ticker": ticker,
        "market": market,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "change": change,
        "base_price": close - change,
        "volume": volume,
        "trading_value": trading_value,
        "market_cap": market_cap,
        "listed_shares": listed_shares,
        "price_state": price_state,
        "invalid_reason": None,
        "available_at": datetime(session.year, session.month, session.day, 18, 0, tzinfo=KRX_TZ),
        "source_hash": hashlib.sha256(session.isoformat().encode()).hexdigest(),
        "policy_version": "krx-daily-market-v1",
    }


def _urow(instrument_id: str, *, eligible: bool = True, reason: str = "eligible") -> dict[str, object]:
    return {"instrument_id": instrument_id, "ticker": instrument_id.removeprefix("KRX:"), "eligible": eligible, "exclusion_reason": reason}


def _write_dataset(
    root: Path,
    name: str,
    sessions_rows: dict[date, list[dict[str, object]]],
    *,
    universe: bool = False,
) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    partitions = {}
    for session in sorted(sessions_rows):
        rows = sorted(sessions_rows[session], key=lambda row: str(row["instrument_id"]))
        if rows:
            frame = pl.DataFrame(rows)
        elif universe:
            frame = pl.DataFrame(
                [],
                schema={"instrument_id": pl.String, "ticker": pl.String, "eligible": pl.Boolean, "exclusion_reason": pl.String},
            )
        else:
            frame = pl.DataFrame(
                [],
                schema={
                    "session": pl.Date, "instrument_id": pl.String, "ticker": pl.String, "market": pl.String,
                    "open": pl.Int64, "high": pl.Int64, "low": pl.Int64, "close": pl.Int64,
                    "change": pl.Int64, "base_price": pl.Int64, "volume": pl.Int64,
                    "trading_value": pl.Int64, "market_cap": pl.Int64, "listed_shares": pl.Int64,
                    "price_state": pl.String, "invalid_reason": pl.String,
                    "available_at": pl.Datetime("us", "Asia/Seoul"), "source_hash": pl.String,
                    "policy_version": pl.String,
                },
            )
        partitions[f"session={session.isoformat()}/part.parquet"] = frame
    return publish_dataset(
        layer_root=root,
        identity=DatasetIdentity(
            kind="ordinary_universe" if universe else "daily_market",
            layer=DatasetLayer.SILVER,
            policy_version="test-v1",
            inputs={},
            params={},
        ),
        partitions=partitions,
    ).path


def _inputs(
    tmp_path: Path,
    daily: dict[date, list[dict[str, object]]],
    universe: dict[date, list[dict[str, object]]],
) -> tuple[Path, Path, Path]:
    silver = tmp_path / "silver"
    daily_path = _write_dataset(silver, "daily_market_test", daily)
    universe_path = _write_dataset(silver, "ordinary_universe_test", universe, universe=True)
    return daily_path, universe_path, tmp_path / "gold"


def _full_universe(daily: dict[date, list[dict[str, object]]]) -> dict[date, list[dict[str, object]]]:
    return {
        session: [_urow(str(row["instrument_id"])) for row in rows]
        for session, rows in daily.items()
    }


def _panel_frame(result_path: Path, year: int) -> pl.DataFrame:
    return pl.read_parquet(result_path / f"year={year}" / "part.parquet")


def test_materialize_computes_price_return_from_krx_base(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930", close=10000, change=0)],
        DAY1: [_drow(DAY1, "005930", close=11000, change=1000)],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020)
    assert frame.filter(pl.col("session") == DAY1)["ret_price"].to_list() == [0.1]
    assert frame.filter(pl.col("session") == DAY0)["ret_price"].to_list() == [None]


def test_materialize_computes_split_share_factor(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930", close=20000, change=0)],
        DAY1: [_drow(DAY1, "005930", close=10500, change=500)],
        DAY2: [_drow(DAY2, "005930", close=10500, change=0)],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020)
    assert frame.filter(pl.col("session") == DAY1)["share_factor"].to_list() == [2.0]
    assert frame.filter(pl.col("session") == DAY2)["share_factor"].to_list() == [1.0]
    assert result.corporate_action_rows == 1


def test_materialize_nulls_first_observation_fields(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930"), _drow(DAY0, "000660", price_state="invalid")],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020)
    assert frame["ret_price"].to_list() == [None, None]
    assert frame["share_factor"].to_list() == [None, None]
    assert frame["upper_limit"].to_list() == [None, None]
    assert frame["lower_limit"].to_list() == [None, None]
    assert frame["limits_applicable"].to_list() == [False, False]
    assert result.limit_inapplicable_rows == 2


def test_materialize_detects_gap_before(tmp_path: Path) -> None:
    sessions = (DAY0, DAY1, DAY2)
    daily = {
        DAY0: [_drow(DAY0, "005930", close=10000, change=0), _drow(DAY0, "000660", close=5000, change=0)],
        DAY1: [_drow(DAY1, "005930", close=10100, change=100)],
        DAY2: [
            _drow(DAY2, "005930", close=10200, change=100),
            _drow(DAY2, "000660", close=5100, change=100),
        ],
    }
    assert set(sessions) == set(daily)
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020)
    gap_row = frame.filter((pl.col("session") == DAY2) & (pl.col("ticker") == "000660"))
    assert gap_row["gap_before"].to_list() == [True]
    assert gap_row["share_factor"].to_list() == [None]
    assert gap_row["ret_price"].to_list() == [100 / 5000]
    assert result.gap_rows == 1


def test_materialize_flags_open_at_upper_limit(tmp_path: Path) -> None:
    daily = {
        DAY0: [
            _drow(DAY0, "005930", close=10000, change=0),
            _drow(DAY0, "000660", close=8000, change=0),
        ],
        DAY1: [
            _drow(DAY1, "005930", open_price=13000, high=13000, low=12900, close=13000, change=3000),
            _drow(DAY1, "000660", open_price=7000, high=7500, low=7000, close=7500, change=-2500),
        ],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020).sort("ticker")
    day1 = frame.filter(pl.col("session") == DAY1)
    lower_row = day1.filter(pl.col("ticker") == "000660")
    assert lower_row["lower_limit"].to_list() == [7000]
    assert lower_row["open_at_lower"].to_list() == [True]
    upper_row = day1.filter(pl.col("ticker") == "005930")
    assert upper_row["upper_limit"].to_list() == [13000]
    assert upper_row["open_at_upper"].to_list() == [True]
    assert upper_row["close_at_upper"].to_list() == [True]
    assert result.open_at_upper_rows == 1
    assert result.open_at_lower_rows == 1


def test_materialize_detects_no_limit_session(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930", close=10000, change=0)],
        DAY1: [_drow(DAY1, "005930", open_price=10500, high=11200, low=2000, close=11000, change=1000)],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020)
    row = frame.filter(pl.col("session") == DAY1)
    assert row["limits_applicable"].to_list() == [False]
    assert row["open_at_upper"].to_list() == [False]
    assert row["open_at_lower"].to_list() == [False]
    assert row["close_at_upper"].to_list() == [False]
    assert row["close_at_lower"].to_list() == [False]


def test_materialize_never_locks_zero_volume_rows(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930", close=10000, change=0)],
        DAY1: [_drow(DAY1, "005930", open_price=13000, high=13000, low=13000, close=13000,
                     change=3000, volume=0, trading_value=0, price_state="zero_volume")],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020)
    row = frame.filter(pl.col("session") == DAY1)
    assert row["close_at_upper"].to_list() == [False]
    assert row["open_at_upper"].to_list() == [False]
    assert row["price_state"].to_list() == ["zero_volume"]


def test_materialize_ends_adtv_window_at_own_session(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 2, 3), 21)
    daily = {
        session: [_drow(session, "005930", close=100 + index, change=0, trading_value=index + 1)]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020)
    assert frame.filter(pl.col("session") == sessions[19])["adtv20"].to_list() == [10.5]
    assert frame.filter(pl.col("session") == sessions[18])["adtv20"].to_list() == [None]
    daily2 = dict(daily)
    daily2[sessions[20]] = [_drow(sessions[20], "005930", close=120, change=0, trading_value=9999)]
    daily_path2, universe_path2, gold_root2 = _inputs(tmp_path / "rerun", daily2, _full_universe(daily2))
    rerun = materialize_market_panel(
        daily_market_path=daily_path2, universe_path=universe_path2, rules=RULES, gold_root=gold_root2
    )
    rerun_frame = _panel_frame(rerun.dataset_path, 2020)
    assert rerun_frame.filter(pl.col("session") == sessions[19])["adtv20"].to_list() == [10.5]


def test_materialize_is_invariant_to_future_perturbation(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 4, 6), 6)
    daily = {
        session: [
            _drow(session, "005930", close=10000 + index * 10, change=10 * (index > 0), trading_value=1000 + index),
            _drow(session, "000660", close=5000, change=0, trading_value=500),
        ]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    perturbed = {
        session: [
            _drow(session, ticker, close=row["close"] * 2 + 7 if index > 2 else row["close"],
                  change=row["change"], trading_value=row["trading_value"] + 11 if index > 2 else row["trading_value"],
                  volume=row["volume"] + 5 if index > 2 else row["volume"])
            for ticker, row in (("005930", rows[0]), ("000660", rows[1]))
        ]
        for index, (session, rows) in enumerate(daily.items())
    }
    daily_path2, universe_path2, gold_root2 = _inputs(tmp_path / "perturbed", perturbed, _full_universe(perturbed))
    rerun = materialize_market_panel(
        daily_market_path=daily_path2, universe_path=universe_path2, rules=RULES, gold_root=gold_root2
    )
    cutoff = sessions[2]
    left = _panel_frame(result.dataset_path, 2020).filter(pl.col("session") <= cutoff).sort(["session", "instrument_id"])
    right = _panel_frame(rerun.dataset_path, 2020).filter(pl.col("session") <= cutoff).sort(["session", "instrument_id"])
    assert_frame_equal(left, right)


def test_materialize_reports_daily_volatility_units(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 3, 2), 61)
    daily = {sessions[0]: [_drow(sessions[0], "005930", close=10000, change=0)]}
    for index, session in enumerate(sessions[1:]):
        sign = 1 if index % 2 == 0 else -1
        daily[session] = [_drow(session, "005930", close=10000 + sign * 100, change=sign * 100)]
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020)
    assert frame.filter(pl.col("session") == sessions[-1])["ret_vol60"].to_list() == pytest.approx(
        [0.01 * (60 / 59) ** 0.5], rel=1e-9
    )
    assert frame.filter(pl.col("session") == sessions[-2])["ret_vol60"].to_list() == [None]


def test_materialize_resolves_tick_and_tax_by_date(tmp_path: Path) -> None:
    day_before = date(2023, 1, 24)
    day_after = date(2023, 1, 25)
    daily = {
        day_before: [_drow(day_before, "005930", open_price=1500, high=1500, low=1500, close=1500, change=0)],
        day_after: [_drow(day_after, "005930", open_price=1500, high=1500, low=1500, close=1500, change=0)],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2023).sort("session")
    assert frame["tick_size"].to_list() == [5, 1]
    assert frame["sell_tax_rate"].to_list() == [0.002, 0.002]


def test_materialize_separates_exit_table(tmp_path: Path) -> None:
    sessions = (DAY0, DAY1, DAY2)
    daily = {
        DAY0: [
            _drow(DAY0, "005930", close=10000, change=0),
            _drow(DAY0, "000660", close=5000, change=0),
            _drow(DAY0, "000020", close=3000, change=0),
        ],
        DAY1: [
            _drow(DAY1, "005930", close=10100, change=100),
            _drow(DAY1, "000660", close=5000, change=0, volume=0, trading_value=0),
        ],
        DAY2: [_drow(DAY2, "005930", close=10200, change=100)],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    assert result.exits_traded == 1
    assert result.exits_halted == 1
    exits = pl.read_parquet(result.dataset_path / "instrument_exits.parquet").sort("instrument_id")
    assert exits.filter(pl.col("instrument_id") == "KRX:000660")["exit_kind"].to_list() == ["halted_exit"]
    assert exits.filter(pl.col("instrument_id") == "KRX:000020")["exit_kind"].to_list() == ["traded_exit"]
    assert "KRX:005930" not in exits["instrument_id"].to_list()
    panel = _panel_frame(result.dataset_path, 2020)
    assert "exit_kind" not in panel.columns
    assert "last_session" not in panel.columns
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["exits"]["decision_safe"] is False
    assert manifest["dividends"] == "not_integrated"
    exit_partition = next(item for item in manifest["partitions"] if item["path"] == "instrument_exits.parquet")
    assert len(exit_partition["sha256"]) == 64


def test_materialize_marks_rows_absent_from_universe(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    universe = {DAY0: [_urow("KRX:000660")]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, universe)
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020)
    assert frame["eligible"].to_list() == [False]
    assert frame["exclusion_reason"].to_list() == ["absent_from_universe"]


def test_materialize_rejects_session_set_mismatch(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    universe = {DAY0: [_urow("KRX:005930")], DAY1: [_urow("KRX:005930")]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, universe)
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
        )


def test_materialize_carries_december_tail_into_january(tmp_path: Path) -> None:
    december = _sessions(date(2025, 12, 12), 20)
    january = _sessions(date(2026, 1, 1), 3)
    daily = {
        session: [_drow(session, "005930", close=100 + index, change=0, trading_value=index + 1)]
        for index, session in enumerate([*december, *january])
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    assert result.years == (2025, 2026)
    january_frame = _panel_frame(result.dataset_path, 2026)
    assert january_frame.filter(pl.col("session") == january[0])["adtv20"].to_list() == [11.5]
    assert set(january_frame["session"].to_list()) == set(january)
    december_frame = _panel_frame(result.dataset_path, 2025)
    assert set(december_frame["session"].to_list()) == set(december)


def test_materialize_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930", close=10000, change=0)],
        DAY1: [_drow(DAY1, "005930", close=10100, change=100)],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    first = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    before = (first.dataset_path / "manifest.json").read_bytes()
    second = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    assert first.dataset_id == second.dataset_id
    assert second.dataset_path == first.dataset_path
    assert (first.dataset_path / "manifest.json").read_bytes() == before
    assert first.dataset_id.startswith("market_panel_")


def test_materialize_rejects_differing_existing_dataset(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    (result.dataset_path / "manifest.json").write_text('{"dataset_id": "tampered"}', encoding="utf-8")
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
        )


def test_materialize_rejects_unreadable_existing_dataset(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    (result.dataset_path / "manifest.json").unlink()
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
        )


def test_materialize_rejects_session_outside_rule_coverage(tmp_path: Path) -> None:
    old = date(2015, 1, 5)
    daily = {old: [_drow(old, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
        )


def test_materialize_rejects_duplicate_keys(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0), _drow(DAY0, "005930", close=10100, change=100)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
        )


def test_materialize_rejects_tampered_partition(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    part_path = daily_path / f"session={DAY0.isoformat()}" / "part.parquet"
    part_path.write_bytes(part_path.read_bytes() + b" ")
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
        )


def test_materialize_rejects_invalid_policy_windows(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root,
            policy=MarketPanelPolicy(adtv_short_sessions=0, adtv_long_sessions=60, return_vol_sessions=60),
        )


def test_materialize_rejects_broken_input_manifests(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=silver / "missing-daily", universe_path=silver / "missing-universe",
            rules=RULES, gold_root=tmp_path / "gold",
        )
    for name in ("daily_market_bad", "ordinary_universe_bad"):
        bad = silver / name
        bad.mkdir(parents=True, exist_ok=True)
        (bad / "manifest.json").write_text('{"dataset_id": "other", "partitions": []}', encoding="utf-8")
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=silver / "daily_market_bad", universe_path=silver / "ordinary_universe_bad",
            rules=RULES, gold_root=tmp_path / "gold",
        )


@pytest.mark.parametrize(
    "partitions",
    [
        ["nope"],
        [{"session": DAY0.isoformat()}],
        [{"session": "xx", "path": "p", "parquet_sha256": "s"}],
        [
            {"session": DAY1.isoformat(), "path": "p", "parquet_sha256": "s"},
            {"session": DAY0.isoformat(), "path": "p", "parquet_sha256": "s"},
        ],
        [
            {"session": DAY0.isoformat(), "path": "p", "parquet_sha256": "s"},
            {"session": DAY0.isoformat(), "path": "q", "parquet_sha256": "s"},
        ],
    ],
)
def test_materialize_rejects_malformed_partitions(tmp_path: Path, partitions: list[object]) -> None:
    silver = tmp_path / "silver"
    for name in ("daily_market_bad", "ordinary_universe_bad"):
        bad = silver / name
        bad.mkdir(parents=True, exist_ok=True)
        (bad / "manifest.json").write_text(
            json.dumps({"dataset_id": name, "partitions": partitions}), encoding="utf-8"
        )
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=silver / "daily_market_bad", universe_path=silver / "ordinary_universe_bad",
            rules=RULES, gold_root=tmp_path / "gold",
        )


def test_materialize_rejects_unreadable_partition(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    (daily_path / f"session={DAY0.isoformat()}" / "part.parquet").unlink()
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
        )


def test_materialize_uses_custom_policy_windows(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 5, 4), 5)
    daily = {
        session: [_drow(session, "005930", close=100 + index, change=0, trading_value=index + 1)]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root,
        policy=MarketPanelPolicy(adtv_short_sessions=3, adtv_long_sessions=5, return_vol_sessions=4),
    )
    frame = _panel_frame(result.dataset_path, 2020)
    assert frame.filter(pl.col("session") == sessions[4])["adtv20"].to_list() == [4.0]
    assert frame.filter(pl.col("session") == sessions[4])["adtv60"].to_list() == [3.0]
    assert frame.filter(pl.col("session") == sessions[2])["adtv20"].to_list() == [2.0]


def test_materialize_young_listing_across_year_boundary_stays_null(tmp_path: Path) -> None:
    sessions = _sessions(date(2025, 12, 27), 20)
    daily = {}
    prev_close = 10000
    for index, session in enumerate(sessions):
        close = prev_close + 10
        change = 10 if index else 0
        if index == 0:
            close, change = 10000, 0
        daily[session] = [_drow(session, "005930", close=close, change=change, trading_value=1000)]
        prev_close = close
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = pl.concat([_panel_frame(result.dataset_path, 2025), _panel_frame(result.dataset_path, 2026)]).sort("session")
    assert frame["adtv20"].to_list() == [None] * 19 + [frame["adtv20"].to_list()[-1]]
    assert frame["adtv20"].to_list()[-1] is not None
    assert frame["adtv60"].to_list() == [None] * 20
    assert frame["ret_vol60"].to_list() == [None] * 20


def test_materialize_windows_ignore_year_boundaries(tmp_path: Path) -> None:
    import statistics

    sessions = _sessions(date(2025, 11, 1), 70)
    daily = {}
    prev_close = 10000
    closes: list[int] = []
    values: list[float] = []
    for index, session in enumerate(sessions):
        if index == 0:
            close, change = 10000, 0
        else:
            close, change = prev_close + 37, 37
        daily[session] = [_drow(session, "005930", close=close, change=change, trading_value=1000 + index)]
        closes.append(close)
        values.append(float(1000 + index))
        prev_close = close
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = pl.concat([_panel_frame(result.dataset_path, 2025), _panel_frame(result.dataset_path, 2026)]).sort("session")
    jan_first = next(s for s in sessions if s.year == 2026)
    pos = sessions.index(jan_first)
    row = frame.filter(pl.col("session") == jan_first)
    assert row["adtv20"].to_list() == [sum(values[pos - 19:pos + 1]) / 20]
    assert row["adtv60"].to_list() == [sum(values[pos - 59:pos + 1]) / 60]
    rets = [37 / closes[i - 1] for i in range(pos - 59, pos + 1)]
    assert row["ret_vol60"].to_list() == pytest.approx([statistics.stdev(rets)], rel=1e-9)


def test_materialize_bucket_count_never_changes_output(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 6, 1), 8)
    daily = {
        session: [
            _drow(session, "005930", close=10000 + index * 10, change=10 * (index > 0)),
            _drow(session, "000660", close=5000 + index * 5, change=5 * (index > 0)),
        ]
        for index, session in enumerate(sessions)
    }
    results = []
    for buckets in (1, 3, 16):
        daily_path, universe_path, gold_root = _inputs(tmp_path / f"b{buckets}", daily, _full_universe(daily))
        results.append(
            materialize_market_panel(
                daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
                gold_root=gold_root, instrument_buckets=buckets,
            )
        )
    assert results[0].dataset_id == results[1].dataset_id == results[2].dataset_id
    for year in (2020,):
        digests = [
            hashlib.sha256((r.dataset_path / f"year={year}" / "part.parquet").read_bytes()).hexdigest()
            for r in results
        ]
        assert digests[0] == digests[1] == digests[2]
    exits = [pl.read_parquet(r.dataset_path / "instrument_exits.parquet").sort("instrument_id") for r in results]
    for other in exits[1:]:
        assert_frame_equal(exits[0], other)


def test_materialize_price_limits_equal_scalar_rules(tmp_path: Path) -> None:
    from src.core.market_rules import KrxMarket

    bases = [999, 1000, 1001, 4999, 5000, 5001, 9999, 10000, 1001, 19999, 20000, 20001,
             49999, 50000, 50001, 99999, 100000, 100001, 199999, 200000, 200001, 499999, 500000, 500001]
    pairs = [(date(2020, 1, 6), date(2020, 1, 7)), (date(2023, 2, 1), date(2023, 2, 2))]
    daily: dict[date, list[dict[str, object]]] = {s: [] for pair in pairs for s in pair}
    tickers: dict[str, tuple[str, date, int]] = {}
    seq = 100000
    for market in ("KOSPI", "KOSDAQ"):
        for day0, day1 in pairs:
            for base in bases:
                seq += 1
                ticker = str(seq)
                tickers[ticker] = (market, day1, base)
                daily[day0].append(_drow(day0, ticker, market=market, open_price=base, high=base,
                                         low=base, close=base, change=0))
                daily[day1].append(_drow(day1, ticker, market=market, open_price=base, high=base,
                                         low=base, close=base, change=0))
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = pl.concat([_panel_frame(result.dataset_path, 2020), _panel_frame(result.dataset_path, 2023)])
    for ticker, (market, session, base) in tickers.items():
        row = frame.filter((pl.col("session") == session) & (pl.col("ticker") == ticker))
        upper, lower = RULES.price_limits(session=session, market=KrxMarket(market), base_price=base)
        assert row["upper_limit"].to_list() == [upper]
        assert row["lower_limit"].to_list() == [lower]
        assert row["tick_size"].to_list() == [RULES.tick_size(session=session, market=KrxMarket(market), price=base)]


def test_materialize_volatility_skips_invalid_rows(tmp_path: Path) -> None:
    import statistics

    sessions = _sessions(date(2020, 7, 1), 63)
    daily = {}
    prev_close = 10000
    valid_rets: list[float] = []
    for index, session in enumerate(sessions):
        if index == 0:
            daily[session] = [_drow(session, "005930", close=10000, change=0)]
            prev_close = 10000
        elif index <= 60:
            close = prev_close + 37
            daily[session] = [_drow(session, "005930", close=close, change=37)]
            valid_rets.append(37 / prev_close)
            prev_close = close
        elif index == 61:
            daily[session] = [_drow(session, "005930", close=12000, change=100, price_state="invalid")]
        else:
            close = 12000 + 50
            daily[session] = [_drow(session, "005930", close=close, change=50)]
            prev_close = close
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    frame = _panel_frame(result.dataset_path, 2020).sort("session")
    vol60 = frame["ret_vol60"].to_list()
    assert vol60[60] == pytest.approx(statistics.stdev(valid_rets), rel=1e-9)
    assert vol60[61] == pytest.approx(vol60[60], rel=1e-12)
    expected_next = statistics.stdev([*valid_rets[1:], 50 / 12000])
    assert vol60[62] == pytest.approx(expected_next, rel=1e-9)


def test_materialize_policy_version_bump(tmp_path: Path) -> None:
    from src.data.market_panel import POLICY_VERSION

    assert POLICY_VERSION == "krx-market-panel-v4"
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy_version"] == "krx-market-panel-v4"
    v2_id = "market_panel_" + hashlib.sha256(
        "\n".join((
            "krx-market-panel-v3", "20", "60", "60", RULES.version,
            manifest["daily_market_dataset_id"],
            manifest["universe_dataset_id"],
        )).encode("utf-8")
    ).hexdigest()[:16]
    assert result.dataset_id != v2_id


def test_materialize_rejects_non_positive_bucket_count(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
            gold_root=gold_root, instrument_buckets=0,
        )


def test_materialize_removes_shards_before_publish(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930", close=10000, change=0)],
        DAY1: [_drow(DAY1, "005930", close=10100, change=100)],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    names = sorted(str(p.relative_to(result.dataset_path)) for p in result.dataset_path.rglob("*") if p.is_file())
    assert names == ["instrument_exits.parquet", "manifest.json", "year=2020/part.parquet"]


def test_materialize_rejects_vectorized_null_rule_lookup(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930", market="NYSE", close=10000, change=0)],
        DAY1: [_drow(DAY1, "005930", market="NYSE", close=10100, change=100)],
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
        )


def test_market_panel_input_manifest_boundaries(tmp_path: Path, monkeypatch) -> None:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.market_panel import _load_input_manifest
    import src.data.market_panel as panel_module

    empty = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=DatasetIdentity("daily_market", DatasetLayer.SILVER, "fixture-v1", {}, {}),
        partitions={},
    )
    with pytest.raises(PITDataError, match="invalid market-panel input manifest"):
        _load_input_manifest(empty.path, expected_kind="daily_market")

    valid = _write_dataset(tmp_path / "silver", "daily_market_boundary", {DAY0: [_drow(DAY0, "005930")]})
    monkeypatch.setattr(panel_module, "load_manifest", lambda _path: (_ for _ in ()).throw(PITDataError("bad")))
    assert _load_input_manifest(valid, expected_kind="daily_market")[0] == valid.name
    monkeypatch.undo()

    wrong_kind = _write_dataset(
        tmp_path / "silver", "ordinary_universe_boundary", {DAY0: [_urow("KRX:005930")]}, universe=True
    )
    with pytest.raises(PITDataError, match="kind mismatch"):
        _load_input_manifest(wrong_kind, expected_kind="daily_market")

    no_session = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=DatasetIdentity("daily_market", DatasetLayer.SILVER, "fixture-v1", {}, {"case": "no-session"}),
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )
    with pytest.raises(PITDataError, match="has no session"):
        _load_input_manifest(no_session.path, expected_kind="daily_market")

    bad_session = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=DatasetIdentity("daily_market", DatasetLayer.SILVER, "fixture-v1", {}, {"case": "bad-session"}),
        partitions={"session=bad/part.parquet": pl.DataFrame({"value": [1]})},
    )
    with pytest.raises(PITDataError, match="invalid session"):
        _load_input_manifest(bad_session.path, expected_kind="daily_market")

    unreadable = tmp_path / "unreadable-input"
    monkeypatch.setattr(panel_module, "dataset_partition_paths", lambda *_args, **_kwargs: (unreadable / "part.parquet",))
    monkeypatch.setattr(panel_module, "load_manifest", lambda _path: None)
    with pytest.raises(PITDataError, match="unreadable"):
        _load_input_manifest(unreadable, expected_kind="daily_market")
    monkeypatch.undo()

    multiple = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=DatasetIdentity("daily_market", DatasetLayer.SILVER, "fixture-v1", {}, {"case": "multiple-session"}),
        partitions={"session=2020-01-06/part.parquet": pl.DataFrame({"session": [DAY0, DAY1]})},
    )
    with pytest.raises(PITDataError, match="no unique session"):
        _load_input_manifest(multiple.path, expected_kind="daily_market")

    string_session = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=DatasetIdentity("daily_market", DatasetLayer.SILVER, "fixture-v1", {}, {"case": "string-session"}),
        partitions={"session=2020-01-06/part.parquet": pl.DataFrame({"session": ["2020-01-06"]})},
    )
    assert _load_input_manifest(string_session.path, expected_kind="daily_market")[1] == [DAY0]

    unordered = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=DatasetIdentity("daily_market", DatasetLayer.SILVER, "fixture-v1", {}, {"case": "unordered"}),
        partitions={
            "session=2020-01-06/part.parquet": pl.DataFrame({"session": [DAY0]}),
            "session=2020-01-07/part.parquet": pl.DataFrame({"session": [DAY1 - timedelta(days=1)]}),
        },
    )
    with pytest.raises(PITDataError, match="strictly ordered"):
        _load_input_manifest(unordered.path, expected_kind="daily_market")


def _publish_actions(
    root: Path, rows: list[dict[str, object]], *, schema_override: dict[str, object] | None = None,
) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    schema = {
        "instrument_id": pl.String, "ticker": pl.String, "kind": pl.String, "rcept_no": pl.String,
        "announced_on": pl.Date, "available_at": pl.Datetime("us", "Asia/Seoul"),
        "effective_start": pl.Date, "effective_end": pl.Date, "cancellation": pl.Boolean,
        "policy_version": pl.String,
    }
    if schema_override is not None:
        schema = dict(schema_override)
    frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame([], schema=schema)
    return publish_dataset(
        layer_root=root,
        identity=DatasetIdentity(
            kind="market_actions", layer=DatasetLayer.SILVER,
            policy_version="market-actions-v1", inputs={}, params={},
        ),
        partitions={"part-00000.parquet": frame},
    ).path


def _action_row(
    announced: date, available: date, ticker: str, kind: str, rcept_no: str, *, cancellation: bool = False,
) -> dict[str, object]:
    return {
        "instrument_id": f"KRX:{ticker}", "ticker": ticker, "kind": kind, "rcept_no": rcept_no,
        "announced_on": announced,
        "available_at": datetime(available.year, available.month, available.day, 9, 0, tzinfo=KRX_TZ),
        "effective_start": None, "effective_end": None, "cancellation": cancellation,
        "policy_version": "market-actions-v1",
    }


def test_entry_blocked_only_after_publication(tmp_path: Path) -> None:
    sessions = (DAY0, DAY1, DAY2)
    tickers = ("005930", "000660", "000020")
    daily = {
        session: [
            _drow(session, ticker, close=10000 + index * 100, change=100 * (index > 0))
            for ticker in tickers
        ]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    naive_schema = {
        "instrument_id": pl.String, "ticker": pl.String, "kind": pl.String, "rcept_no": pl.String,
        "announced_on": pl.Date, "available_at": pl.Datetime("us"),
        "effective_start": pl.Date, "effective_end": pl.Date, "cancellation": pl.Boolean,
        "policy_version": pl.String,
    }

    def _naive(announced: date, available: date, ticker: str, kind: str, rcept_no: str) -> dict[str, object]:
        row = _action_row(announced, available, ticker, kind, rcept_no)
        row["available_at"] = datetime(available.year, available.month, available.day, 9, 0)
        return row

    actions_path = _publish_actions(
        tmp_path / "silver",
        [
            _naive(DAY0, DAY1, "005930", "delisting_decided", "20200107000001"),
            _naive(DAY1, DAY2, "000660", "liquidation_trading", "20200108000002"),
            _naive(DAY0, DAY0, "000020", "administrative_designated", "20200106000003"),
            _naive(DAY1, DAY2, "000020", "administrative_released", "20200108000004"),
        ],
        schema_override=naive_schema,
    )
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, market_actions_path=actions_path,
    )
    frame = _panel_frame(result.dataset_path, 2020).sort(["instrument_id", "session"])
    blocked = frame.filter(pl.col("instrument_id") == "KRX:005930")
    assert blocked["entry_blocked"].to_list() == [False, True, True]
    assert blocked["entry_block_reason"].to_list() == ["", "delisting_decided", "delisting_decided"]
    assert blocked["eligible"].to_list() == [True, True, True]
    liquidated = frame.filter(pl.col("instrument_id") == "KRX:000660")
    assert liquidated["entry_blocked"].to_list() == [False, False, True]
    assert liquidated["entry_block_reason"].to_list() == ["", "", "liquidation_trading"]
    managed = frame.filter(pl.col("instrument_id") == "KRX:000020")
    assert managed["entry_blocked"].to_list() == [True, True, False]
    assert managed["entry_block_reason"].to_list() == ["administrative_designated"] * 2 + [""]
    assert result.entry_blocked_rows == 5
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["market_actions_dataset_id"] == actions_path.name


def test_entry_blocked_administrative_follows_policy(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930", close=10000, change=0)],
        DAY1: [_drow(DAY1, "005930", close=10100, change=100)],
    }
    actions_path = _publish_actions(
        tmp_path / "silver",
        [_action_row(DAY0, DAY0, "005930", "administrative_designated", "20200106000001")],
    )
    blocked_policy = MarketPanelPolicy(block_administrative=True, delisting_block_max_sessions=250)
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    blocked = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, policy=blocked_policy, market_actions_path=actions_path,
    )
    frame = _panel_frame(blocked.dataset_path, 2020).sort("session")
    assert frame["entry_blocked"].to_list() == [True, True]
    assert frame["entry_block_reason"].to_list() == ["administrative_designated"] * 2
    open_policy = MarketPanelPolicy(block_administrative=False)
    daily_path2, universe_path2, gold_root2 = _inputs(tmp_path / "open", daily, _full_universe(daily))
    opened = materialize_market_panel(
        daily_market_path=daily_path2, universe_path=universe_path2, rules=RULES,
        gold_root=gold_root2, policy=open_policy, market_actions_path=actions_path,
    )
    assert blocked.dataset_id != opened.dataset_id
    reopened = _panel_frame(opened.dataset_path, 2020).sort("session")
    assert reopened["entry_blocked"].to_list() == [False, False]


def test_future_action_never_leaks(tmp_path: Path) -> None:
    sessions = (DAY0, DAY1, DAY2)
    daily = {
        session: [_drow(session, "005930", close=10000 + index * 100, change=100 * (index > 0))]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    future = date(2020, 1, 9)
    actions_path = _publish_actions(
        tmp_path / "silver",
        [_action_row(DAY2, future, "005930", "delisting_decided", "20200109000001")],
    )
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, market_actions_path=actions_path,
    )
    frame = _panel_frame(result.dataset_path, 2020).sort("session")
    assert frame["entry_blocked"].to_list() == [False, False, False]
    perturbed_path = _publish_actions(
        tmp_path / "silver-perturbed",
        [_action_row(DAY2, future, "005930", "liquidation_trading", "20200109000001")],
    )
    daily_path2, universe_path2, gold_root2 = _inputs(tmp_path / "perturbed", daily, _full_universe(daily))
    rerun = materialize_market_panel(
        daily_market_path=daily_path2, universe_path=universe_path2, rules=RULES,
        gold_root=gold_root2, market_actions_path=perturbed_path,
    )
    rerun_frame = _panel_frame(rerun.dataset_path, 2020).sort("session")
    assert_frame_equal(
        frame.select(["session", "instrument_id", "entry_blocked", "entry_block_reason"]),
        rerun_frame.select(["session", "instrument_id", "entry_blocked", "entry_block_reason"]),
    )


def test_default_block_administrative_reads_config(tmp_path: Path) -> None:
    from src.data.market_panel import _default_block_administrative

    assert _default_block_administrative() is True
    enabled = tmp_path / "enabled.toml"
    enabled.write_text("block_administrative = true\n", encoding="utf-8")
    assert _default_block_administrative(enabled) is True
    disabled = tmp_path / "disabled.toml"
    disabled.write_text("block_administrative = false\n", encoding="utf-8")
    assert _default_block_administrative(disabled) is False
    missing_key = tmp_path / "missing-key.toml"
    missing_key.write_text("other = 1\n", encoding="utf-8")
    with pytest.raises(PITDataError, match="boolean block_administrative"):
        _default_block_administrative(missing_key)
    non_bool = tmp_path / "non-bool.toml"
    non_bool.write_text("block_administrative = 'yes'\n", encoding="utf-8")
    with pytest.raises(PITDataError, match="boolean block_administrative"):
        _default_block_administrative(non_bool)
    invalid = tmp_path / "invalid.toml"
    invalid.write_text("block_administrative = \n", encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid TOML"):
        _default_block_administrative(invalid)
    with pytest.raises(PITDataError, match="missing"):
        _default_block_administrative(tmp_path / "absent.toml")


def test_entry_block_frame_rejects_bad_actions(tmp_path: Path) -> None:
    from src.data.market_panel import _entry_block_frame

    sessions = [DAY0, DAY1]
    good = _action_row(DAY0, DAY0, "005930", "delisting_decided", "20200106000001")
    reduced_schema = {
        "instrument_id": pl.String, "ticker": pl.String, "kind": pl.String, "rcept_no": pl.String,
        "announced_on": pl.Date, "available_at": pl.Datetime("us", "Asia/Seoul"),
        "effective_start": pl.Date, "effective_end": pl.Date, "policy_version": pl.String,
    }
    missing_columns = _publish_actions(
        tmp_path / "silver-missing",
        [{key: value for key, value in good.items() if key != "cancellation"}],
        schema_override=reduced_schema,
    )
    with pytest.raises(PITDataError, match="missing columns"):
        _entry_block_frame(missing_columns, sessions, block_administrative=True, delisting_block_max_sessions=250)
    bad_available = dict(good)
    bad_available["available_at"] = "not-a-date"
    bad_frame = _publish_actions(
        tmp_path / "silver-bad", [bad_available],
        schema_override={
            "instrument_id": pl.String, "ticker": pl.String, "kind": pl.String,
            "rcept_no": pl.String, "announced_on": pl.Date, "available_at": pl.String,
            "effective_start": pl.Date, "effective_end": pl.Date,
            "cancellation": pl.Boolean, "policy_version": pl.String,
        },
    )
    with pytest.raises(PITDataError, match="invalid available_at"):
        _entry_block_frame(bad_frame, sessions, block_administrative=True, delisting_block_max_sessions=250)
    unknown_kind = dict(good)
    unknown_kind["kind"] = "mystery"
    unknown_frame = _publish_actions(tmp_path / "silver-unknown", [unknown_kind])
    with pytest.raises(PITDataError, match="unknown kind"):
        _entry_block_frame(unknown_frame, sessions, block_administrative=True, delisting_block_max_sessions=250)


def test_delisting_block_lapses_after_the_configured_cap(tmp_path: Path) -> None:
    from datetime import timedelta

    from src.data.market_panel import _entry_block_frame

    sessions = [DAY0 + timedelta(days=offset) for offset in range(6)]
    row = _action_row(sessions[1], sessions[1], "005930", "delisting_decided", "20200106000001")
    path = _publish_actions(tmp_path / "silver-cap", [row])

    frame = _entry_block_frame(path, sessions, block_administrative=True, delisting_block_max_sessions=2)

    # 공시 다음 세션부터 차단되고, 마지막 공시로부터 cap을 넘긴 세션부터 풀린다(과거 정보만 사용).
    assert frame["session"].to_list() == sessions[1:4]
    assert set(frame["entry_block_reason"].to_list()) == {"delisting_decided"}


def test_new_notice_restarts_the_delisting_block_window(tmp_path: Path) -> None:
    from datetime import timedelta

    from src.data.market_panel import _entry_block_frame

    sessions = [DAY0 + timedelta(days=offset) for offset in range(8)]
    rows = [
        _action_row(sessions[0], sessions[0], "005930", "delisting_decided", "20200106000001"),
        _action_row(sessions[3], sessions[3], "005930", "liquidation_trading", "20200109000001"),
    ]
    path = _publish_actions(tmp_path / "silver-restart", rows)

    frame = _entry_block_frame(path, sessions, block_administrative=True, delisting_block_max_sessions=2)

    assert frame["session"].to_list() == sessions[0:6]
    assert frame["entry_block_reason"].to_list()[3:] == ["delisting_decided"] * 3


def test_market_panel_policy_reads_the_delisting_cap_and_rejects_bad_values(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.market_panel import _default_delisting_block_max_sessions

    assert _default_delisting_block_max_sessions() == 250
    for body in ("delisting_block_max_sessions = 0\n", 'delisting_block_max_sessions = "x"\n', "block_administrative = true\n"):
        path = tmp_path / "policy.toml"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(PITDataError, match="delisting_block_max_sessions"):
            _default_delisting_block_max_sessions(path)


def _bounded_action_row(announced: date, available: date, ticker: str, kind: str, rcept_no: str, *, start: date | None, end: date | None) -> dict[str, object]:
    row = _action_row(announced, available, ticker, kind, rcept_no)
    row["effective_start"] = start
    row["effective_end"] = end
    return row


def test_bounded_delisting_blocks_through_stated_delisting_date(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 1, 6), 7)
    daily = {
        session: [_drow(session, "005930", close=10000 + index * 10, change=10 * (index > 0))]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    actions_path = _publish_actions(
        tmp_path / "silver",
        [_bounded_action_row(sessions[0], sessions[0], "005930", "delisting_decided", "20200106000001",
                             start=sessions[2], end=sessions[5])],
    )
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, market_actions_path=actions_path,
        policy=MarketPanelPolicy(delisting_block_max_sessions=1),
    )
    frame = _panel_frame(result.dataset_path, 2020).sort("session")
    assert frame["entry_blocked"].to_list() == [True] * 6 + [False]
    assert frame["entry_block_reason"].to_list() == (
        ["delisting_decided"] * 2 + ["liquidation_trading"] * 4 + [""]
    )


def test_bounded_block_never_starts_before_availability(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 1, 6), 3)
    daily = {
        session: [_drow(session, "005930", close=10000 + index * 10, change=10 * (index > 0))]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    actions_path = _publish_actions(
        tmp_path / "silver",
        [_bounded_action_row(sessions[1], sessions[1], "005930", "delisting_decided", "20200107000001",
                             start=date(2019, 12, 1), end=sessions[2])],
    )
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, market_actions_path=actions_path,
    )
    frame = _panel_frame(result.dataset_path, 2020).sort("session")
    assert frame["entry_blocked"].to_list() == [False, True, True]


def test_cancellation_lifts_bounded_block(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 1, 6), 4)
    daily = {
        session: [_drow(session, "005930", close=10000 + index * 10, change=10 * (index > 0))]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    bounded = _bounded_action_row(sessions[0], sessions[0], "005930", "delisting_decided", "20200106000001",
                                  start=None, end=sessions[3])
    cancel = _action_row(sessions[2], sessions[2], "005930", "delisting_decided", "20200108000002", cancellation=True)
    actions_path = _publish_actions(tmp_path / "silver", [bounded, cancel])
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, market_actions_path=actions_path,
    )
    frame = _panel_frame(result.dataset_path, 2020).sort("session")
    assert frame["entry_blocked"].to_list() == [True, True, False, False]


def test_unbounded_actions_keep_lapse_cap(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 1, 6), 4)
    daily = {
        session: [_drow(session, "005930", close=10000 + index * 10, change=10 * (index > 0))]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    actions_path = _publish_actions(
        tmp_path / "silver",
        [_action_row(sessions[0], sessions[0], "005930", "liquidation_trading", "20200106000001")],
    )
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, market_actions_path=actions_path,
        policy=MarketPanelPolicy(delisting_block_max_sessions=1),
    )
    frame = _panel_frame(result.dataset_path, 2020).sort("session")
    assert frame["entry_blocked"].to_list() == [True, True, False, False]


def _limitless_daily() -> dict[date, list[dict[str, object]]]:
    day0, day1 = DAY0, DAY1
    return {
        day0: [_drow(day0, "005930", close=1000, change=0)],
        day1: [_drow(day1, "005930", close=100, change=-900, high=1100, low=50)],
    }


def test_unexplained_limitless_move_fails_the_build(tmp_path: Path) -> None:
    daily = _limitless_daily()
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    actions_path = _publish_actions(tmp_path / "silver", [])
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
            gold_root=gold_root, market_actions_path=actions_path,
        )
    assert list((tmp_path / "gold").glob("market_panel_*")) == []


def test_explained_or_exempt_moves_pass_the_audit(tmp_path: Path) -> None:
    daily = _limitless_daily()
    blocked_actions = _publish_actions(
        tmp_path / "silver-blocked",
        [_action_row(DAY0, DAY1, "005930", "delisting_decided", "20200107000001")],
    )
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    blocked = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, market_actions_path=blocked_actions,
    )
    assert blocked.unexplained_limitless_moves == 0

    single = {DAY0: [_drow(DAY0, "005930", close=100, change=0, high=1100, low=50)]}
    single_path, single_universe, single_gold = _inputs(tmp_path / "single", single, _full_universe(single))
    single_actions = _publish_actions(tmp_path / "silver-single", [])
    first_row = materialize_market_panel(
        daily_market_path=single_path, universe_path=single_universe, rules=RULES,
        gold_root=single_gold, market_actions_path=single_actions,
    )
    assert first_row.unexplained_limitless_moves == 0

    quiet = {
        DAY0: [_drow(DAY0, "005930", close=1000, change=0)],
        DAY1: [_drow(DAY1, "005930", close=100, change=-900, high=1100, low=50, volume=0, trading_value=0)],
    }
    quiet_path, quiet_universe, quiet_gold = _inputs(tmp_path / "quiet", quiet, _full_universe(quiet))
    quiet_actions = _publish_actions(tmp_path / "silver-quiet", [])
    skipped = materialize_market_panel(
        daily_market_path=quiet_path, universe_path=quiet_universe, rules=RULES,
        gold_root=quiet_gold, market_actions_path=quiet_actions,
    )
    assert skipped.unexplained_limitless_moves == 0


def test_tolerance_admits_known_residue(tmp_path: Path) -> None:
    import json as _json

    daily = _limitless_daily()
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    actions_path = _publish_actions(tmp_path / "silver", [])
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, market_actions_path=actions_path,
        policy=MarketPanelPolicy(max_unexplained_limitless_moves=1),
    )
    assert result.unexplained_limitless_moves == 1
    manifest = _json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["details"]["limitless_move_audit"]["unexplained"] == 1
    assert len(manifest["details"]["limitless_move_audit"]["samples"]) == 1


def test_audit_never_alters_panel_content(tmp_path: Path) -> None:
    daily = {
        DAY0: [_drow(DAY0, "005930", close=10000, change=0)],
        DAY1: [_drow(DAY1, "005930", close=10100, change=100)],
    }
    actions_path = _publish_actions(tmp_path / "silver", [])
    strict_path, strict_universe, strict_gold = _inputs(tmp_path / "strict", daily, _full_universe(daily))
    strict = materialize_market_panel(
        daily_market_path=strict_path, universe_path=strict_universe, rules=RULES,
        gold_root=strict_gold, market_actions_path=actions_path,
        policy=MarketPanelPolicy(limitless_move_audit_threshold=0.3),
    )
    loose_path, loose_universe, loose_gold = _inputs(tmp_path / "loose", daily, _full_universe(daily))
    loose = materialize_market_panel(
        daily_market_path=loose_path, universe_path=loose_universe, rules=RULES,
        gold_root=loose_gold, market_actions_path=actions_path,
        policy=MarketPanelPolicy(limitless_move_audit_threshold=0.9),
    )
    assert strict.dataset_id == loose.dataset_id
    left = (strict.dataset_path / "year=2020" / "part.parquet").read_bytes()
    right = (loose.dataset_path / "year=2020" / "part.parquet").read_bytes()
    assert left == right


def test_audit_skipped_without_market_actions(tmp_path: Path) -> None:
    import json as _json

    daily = _limitless_daily()
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root,
    )
    manifest = _json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["details"]["limitless_move_audit"] == "not_audited"


def test_default_audit_policy_reads_config(tmp_path: Path) -> None:
    from src.data.market_panel import (
        _default_limitless_move_audit_threshold,
        _default_max_unexplained_limitless_moves,
    )

    assert _default_limitless_move_audit_threshold() == 0.30
    assert _default_max_unexplained_limitless_moves() == 0
    custom = tmp_path / "custom.toml"
    custom.write_text(
        "block_administrative = true\ndelisting_block_max_sessions = 250\n"
        "limitless_move_audit_threshold = 0.5\nmax_unexplained_limitless_moves = 2\n",
        encoding="utf-8",
    )
    assert _default_limitless_move_audit_threshold(custom) == 0.5
    assert _default_max_unexplained_limitless_moves(custom) == 2
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "block_administrative = true\ndelisting_block_max_sessions = 250\n"
        "limitless_move_audit_threshold = 1.5\nmax_unexplained_limitless_moves = 0\n",
        encoding="utf-8",
    )
    with pytest.raises(PITDataError):
        _default_limitless_move_audit_threshold(bad)
    with pytest.raises(PITDataError):
        _default_max_unexplained_limitless_moves(tmp_path / "absent.toml")
    bad_max = tmp_path / "bad-max.toml"
    bad_max.write_text(
        "block_administrative = true\ndelisting_block_max_sessions = 250\n"
        "limitless_move_audit_threshold = 0.3\nmax_unexplained_limitless_moves = -1\n",
        encoding="utf-8",
    )
    with pytest.raises(PITDataError):
        _default_max_unexplained_limitless_moves(bad_max)


def test_bounded_liquidation_blocks_through_stated_end(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 1, 6), 5)
    daily = {
        session: [_drow(session, "005930", close=10000 + index * 10, change=10 * (index > 0))]
        for index, session in enumerate(sessions)
    }
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    actions_path = _publish_actions(
        tmp_path / "silver",
        [_bounded_action_row(sessions[0], sessions[0], "005930", "liquidation_trading", "20200106000001",
                             start=sessions[1], end=sessions[3])],
    )
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
        gold_root=gold_root, market_actions_path=actions_path,
        policy=MarketPanelPolicy(delisting_block_max_sessions=1),
    )
    frame = _panel_frame(result.dataset_path, 2020).sort("session")
    assert frame["entry_blocked"].to_list() == [True, True, True, True, False]
    assert frame["entry_block_reason"].to_list() == (
        ["delisting_decided", "liquidation_trading", "liquidation_trading", "liquidation_trading", ""]
    )


def test_materialize_rejects_invalid_audit_policy(tmp_path: Path) -> None:
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
            gold_root=gold_root,
            policy=MarketPanelPolicy(limitless_move_audit_threshold=1.5),
        )
    with pytest.raises(PITDataError):
        materialize_market_panel(
            daily_market_path=daily_path, universe_path=universe_path, rules=RULES,
            gold_root=gold_root,
            policy=MarketPanelPolicy(max_unexplained_limitless_moves=-1),
        )
