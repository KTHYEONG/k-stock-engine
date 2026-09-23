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
from src.data.schemas import PITDataError

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
    dataset = root / name
    partitions: list[dict[str, object]] = []
    for session in sorted(sessions_rows):
        rows = sorted(sessions_rows[session], key=lambda row: str(row["instrument_id"]))
        rel = f"session={session.isoformat()}/part.parquet"
        out_path = dataset / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
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
        frame.write_parquet(out_path)
        partitions.append({
            "session": session.isoformat(),
            "path": rel,
            "source_hash": hashlib.sha256(session.isoformat().encode()).hexdigest(),
            "row_count": frame.height,
            "parquet_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        })
    manifest = {"dataset_id": name, "policy_version": "test-v1", "partitions": partitions}
    (dataset / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


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
    assert len(manifest["exits"]["parquet_sha256"]) == 64


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

    assert POLICY_VERSION == "krx-market-panel-v2"
    daily = {DAY0: [_drow(DAY0, "005930", close=10000, change=0)]}
    daily_path, universe_path, gold_root = _inputs(tmp_path, daily, _full_universe(daily))
    result = materialize_market_panel(
        daily_market_path=daily_path, universe_path=universe_path, rules=RULES, gold_root=gold_root
    )
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy_version"] == "krx-market-panel-v2"
    v1_id = "market_panel_" + hashlib.sha256(
        "\n".join((
            "krx-market-panel-v1", "20", "60", "60", RULES.version,
            manifest["daily_market_dataset_id"],
            manifest["universe_dataset_id"],
        )).encode("utf-8")
    ).hexdigest()[:16]
    assert result.dataset_id != v1_id


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
