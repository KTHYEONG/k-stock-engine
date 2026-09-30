"""Research cube point-in-time invariants."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from src.core.pit import PITDataError
from src.research import cube as cube_mod
from src.research.cube import (
    ResearchCube,
    assemble_dividends,
    assemble_flows,
    assemble_fundamentals,
    assemble_releases,
    research_cube_id,
)

KRX = ZoneInfo("Asia/Seoul")

INST = "KRX:000001"
INST2 = "KRX:000002"


def _sessions(n: int = 8, start: date = date(2020, 3, 30)) -> list[date]:
    from datetime import timedelta

    return [start + timedelta(days=i) for i in range(n)]


def _dt(day: date, hour: int = 9, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=KRX)


def _fact(
    inst: str,
    period: str,
    fact: str,
    value: float | None,
    available: datetime,
    consolidated: bool = True,
) -> dict[str, object]:
    return {
        "company_id": inst,
        "ticker": inst.split(":")[1] if ":" in inst else inst,
        "fiscal_period": period,
        "fact": fact,
        "value": value,
        "available_at": available,
        "consolidated": consolidated,
    }


def test_q4_three_month_derivation_and_ttm() -> None:
    """Q4 three-month equals annual minus Q1..Q3; TTM sums four quarters."""
    sessions = _sessions(8)
    avail_q1 = _dt(sessions[1])
    avail_q2 = _dt(sessions[2])
    avail_q3 = _dt(sessions[3])
    avail_q4 = _dt(sessions[4])
    rows = [
        _fact(INST, "2020Q1", "operating_profit", 10.0, avail_q1),
        _fact(INST, "2020Q2", "operating_profit", 20.0, avail_q2),
        _fact(INST, "2020Q3", "operating_profit", 30.0, avail_q3),
        _fact(INST, "2020Q4", "operating_profit", 100.0, avail_q4),
        _fact(INST, "2020Q1", "equity", 50.0, avail_q1),
        _fact(INST, "2020Q2", "equity", 50.0, avail_q2),
        _fact(INST, "2020Q3", "equity", 50.0, avail_q3),
        _fact(INST, "2020Q4", "equity", 50.0, avail_q4),
    ]
    out = assemble_fundamentals(pl.DataFrame(rows), sessions=sessions, instrument_ids=[INST])
    assert out["f_operating_profit_q"][4, 0] == pytest.approx(40.0)
    assert out["f_operating_profit_ttm"][4, 0] == pytest.approx(100.0)


def test_late_prior_year_and_interim_quarters_hidden_until_available() -> None:
    """A prior-year quarter or a Q4 input quarter filed late never shows before its own availability."""
    sessions = _sessions(10)
    rows = [
        # 전년동기(2019Q4)와 Q3가 현재 분기(2020Q4)보다 늦게 가용
        _fact(INST, "2019Q1", "operating_profit", 1.0, _dt(sessions[0])),
        _fact(INST, "2019Q2", "operating_profit", 1.0, _dt(sessions[0])),
        _fact(INST, "2019Q3", "operating_profit", 1.0, _dt(sessions[0])),
        _fact(INST, "2019Q4", "operating_profit", 8.0, _dt(sessions[6])),
        _fact(INST, "2020Q1", "operating_profit", 10.0, _dt(sessions[1])),
        _fact(INST, "2020Q2", "operating_profit", 20.0, _dt(sessions[1])),
        _fact(INST, "2020Q3", "operating_profit", 30.0, _dt(sessions[5])),
        _fact(INST, "2020Q4", "operating_profit", 100.0, _dt(sessions[2])),
    ]
    out = assemble_fundamentals(pl.DataFrame(rows), sessions=sessions, instrument_ids=[INST])
    q = out["f_operating_profit_q"][:, 0]
    q_ly = out["f_operating_profit_q_ly"][:, 0]
    ttm = out["f_operating_profit_ttm"][:, 0]
    assert out["f_qk"][2, 0] == 2020 * 4 + 3
    assert np.isnan(q[2:5]).all()
    assert q[5] == pytest.approx(40.0)
    assert np.isnan(ttm[2:5]).all()
    assert ttm[5] == pytest.approx(100.0)
    assert np.isnan(q_ly[2:6]).all()
    assert q_ly[6] == pytest.approx(5.0)


def test_missing_interim_quarter_voids_q4() -> None:
    """Q4 three-month is NaN without Q2 while stock values still show."""
    sessions = _sessions(8)
    rows = [
        _fact(INST, "2020Q1", "operating_profit", 10.0, _dt(sessions[1])),
        _fact(INST, "2020Q3", "operating_profit", 30.0, _dt(sessions[3])),
        _fact(INST, "2020Q4", "operating_profit", 100.0, _dt(sessions[4])),
        _fact(INST, "2020Q4", "equity", 77.0, _dt(sessions[4])),
    ]
    out = assemble_fundamentals(pl.DataFrame(rows), sessions=sessions, instrument_ids=[INST])
    assert bool(np.isnan(out["f_operating_profit_q"][4, 0]))
    assert out["f_equity"][4, 0] == pytest.approx(77.0)


def test_availability_placement_at_18h_cut() -> None:
    """09:00 records show same-row; 18:30 records wait for the next session."""
    sessions = [date(2020, 3, 30), date(2020, 3, 31), date(2020, 4, 1)]
    early = datetime(2020, 3, 31, 9, 0, tzinfo=KRX)
    late = datetime(2020, 3, 31, 18, 30, tzinfo=KRX)
    rows = [
        _fact(INST, "2019Q4", "equity", 10.0, early),
        _fact(INST2, "2019Q4", "equity", 20.0, late),
    ]
    out = assemble_fundamentals(pl.DataFrame(rows), sessions=sessions, instrument_ids=[INST, INST2])
    assert out["f_equity"][1, 0] == pytest.approx(10.0)
    assert bool(np.isnan(out["f_equity"][1, 1]))
    assert out["f_equity"][2, 1] == pytest.approx(20.0)


def test_latest_quarter_wins_over_late_older_filing() -> None:
    """A delayed older quarter never replaces a newer visible quarter."""
    sessions = _sessions(8)
    rows = [
        _fact(INST, "2020Q3", "equity", 30.0, _dt(sessions[2])),
        _fact(INST, "2020Q2", "equity", 20.0, _dt(sessions[5])),
    ]
    out = assemble_fundamentals(pl.DataFrame(rows), sessions=sessions, instrument_ids=[INST])
    assert out["f_qk"][6, 0] == pytest.approx(2020 * 4 + 3 - 1)


def _filing_q4_rows(sessions: list[date], op_q4_annual: float, avail_idx: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, quarter in enumerate(("2020Q1", "2020Q2", "2020Q3")):
        rows.append(_fact(INST, quarter, "operating_profit", 20.0, _dt(sessions[i])))
    rows.append(_fact(INST, "2020Q4", "operating_profit", op_q4_annual, _dt(sessions[avail_idx])))
    for i, quarter in enumerate(("2020Q1", "2020Q2", "2020Q3")):
        rows.append(_fact(INST, quarter, "sales", 100.0, _dt(sessions[i])))
    rows.append(_fact(INST, "2020Q4", "sales", 400.0, _dt(sessions[avail_idx])))
    rows.append(_fact(INST, "2020Q4", "equity", 500.0, _dt(sessions[avail_idx])))
    return rows


def test_release_precedes_filing_then_yields_on_tie() -> None:
    """Merged values track the release until the filing arrives, keeping first availability."""
    sessions = _sessions(10)
    facts = pl.DataFrame(_filing_q4_rows(sessions, 100.0, 6))
    fundamentals = assemble_fundamentals(facts, sessions=sessions, instrument_ids=[INST])
    releases = pl.DataFrame(
        [
            {
                "instrument_id": INST,
                "fiscal_period": "2020Q4",
                "metric": "operating_profit",
                "span": "quarter",
                "value_krw": 11.0,
                "prior_year_value_krw": 5.0,
                "available_at": _dt(sessions[4]),
                "basis": "consolidated",
                "release_kind": "preliminary",
            }
        ]
    )
    merged = assemble_releases(releases, fundamentals, facts, sessions=sessions, instrument_ids=[INST])
    assert merged["earn_operating_profit_q"][4, 0] == pytest.approx(11.0)
    assert merged["earn_operating_profit_q"][6, 0] == pytest.approx(40.0)
    assert merged["earn_avail_t"][6, 0] == pytest.approx(4.0)


def test_correction_visible_only_from_own_availability() -> None:
    """Rows before a correction keep showing the original release values."""
    sessions = _sessions(10)
    facts = pl.DataFrame(
        [
            _fact(INST, "2020Q1", "sales", 10.0, _dt(sessions[0])),
            _fact(INST, "2020Q1", "operating_profit", 5.0, _dt(sessions[0])),
        ]
    )
    fundamentals = assemble_fundamentals(facts, sessions=sessions, instrument_ids=[INST])
    releases = pl.DataFrame(
        [
            {
                "instrument_id": INST,
                "fiscal_period": "2020Q1",
                "metric": "sales",
                "span": "quarter",
                "value_krw": 100.0,
                "prior_year_value_krw": 80.0,
                "available_at": _dt(sessions[2]),
                "basis": "consolidated",
                "release_kind": "preliminary",
            },
            {
                "instrument_id": INST,
                "fiscal_period": "2020Q1",
                "metric": "sales",
                "span": "quarter",
                "value_krw": 120.0,
                "prior_year_value_krw": 80.0,
                "available_at": _dt(sessions[5]),
                "basis": "consolidated",
                "release_kind": "preliminary",
            },
        ]
    )
    merged = assemble_releases(releases, fundamentals, facts, sessions=sessions, instrument_ids=[INST])
    assert merged["e_sales_q"][3, 0] == pytest.approx(100.0)
    assert merged["e_sales_q"][5, 0] == pytest.approx(120.0)


def test_consolidated_preferred_over_separate() -> None:
    """Consolidated versions win over separate ones for the same quarter."""
    sessions = _sessions(8)
    facts = pl.DataFrame([_fact(INST, "2020Q1", "sales", 10.0, _dt(sessions[0]))])
    fundamentals = assemble_fundamentals(facts, sessions=sessions, instrument_ids=[INST])
    releases = pl.DataFrame(
        [
            {
                "instrument_id": INST,
                "fiscal_period": "2020Q1",
                "metric": "sales",
                "span": "quarter",
                "value_krw": 1.0,
                "prior_year_value_krw": 1.0,
                "available_at": _dt(sessions[2]),
                "basis": "separate",
                "release_kind": "preliminary",
            },
            {
                "instrument_id": INST,
                "fiscal_period": "2020Q1",
                "metric": "sales",
                "span": "quarter",
                "value_krw": 2.0,
                "prior_year_value_krw": 1.0,
                "available_at": _dt(sessions[3]),
                "basis": "consolidated",
                "release_kind": "preliminary",
            },
        ]
    )
    merged = assemble_releases(releases, fundamentals, facts, sessions=sessions, instrument_ids=[INST])
    assert merged["e_sales_q"][4, 0] == pytest.approx(2.0)


def test_profit_change_converts_annual_to_q4() -> None:
    """Annual profit-change values convert to Q4 quarters using visible facts."""
    sessions = _sessions(10)
    rows: list[dict[str, object]] = []
    for i, value in enumerate((20.0, 20.0, 20.0)):
        rows.append(_fact(INST, f"2020Q{i + 1}", "operating_profit", value, _dt(sessions[i])))
    for i, value in enumerate((15.0, 15.0, 15.0)):
        rows.append(_fact(INST, f"2019Q{i + 1}", "operating_profit", value, _dt(sessions[i])))
    facts = pl.DataFrame(rows)
    fundamentals = assemble_fundamentals(facts, sessions=sessions, instrument_ids=[INST])
    releases = pl.DataFrame(
        [
            {
                "instrument_id": INST,
                "fiscal_period": "2020Q4",
                "metric": "operating_profit",
                "span": "annual",
                "value_krw": 100.0,
                "prior_year_value_krw": 80.0,
                "available_at": _dt(sessions[4]),
                "basis": "consolidated",
                "release_kind": "profit_change",
            }
        ]
    )
    merged = assemble_releases(releases, fundamentals, facts, sessions=sessions, instrument_ids=[INST])
    assert merged["e_operating_profit_q"][4, 0] == pytest.approx(40.0)
    assert merged["e_operating_profit_q_ly"][4, 0] == pytest.approx(35.0)


def test_flow_placed_at_availability_session_in_krw() -> None:
    """Share flows convert with the flow-session close and land on the next session."""
    sessions = [date(2020, 3, 30), date(2020, 3, 31), date(2020, 4, 1)]
    close = np.array([[5000.0], [5000.0], [5000.0]])
    flows = pl.DataFrame(
        [
            {
                "instrument_id": INST,
                "session": sessions[0],
                "foreign_net_shares": 100,
                "institution_net_shares": 0,
                "individual_net_shares": 0,
                "available_at": datetime(2020, 3, 31, 8, 0, tzinfo=KRX),
            }
        ]
    )
    out = assemble_flows(flows, close, sessions=sessions, instrument_ids=[INST])
    assert bool(np.isnan(out["flow_for_krw"][0, 0]))
    assert out["flow_for_krw"][1, 0] == pytest.approx(500_000.0)


def test_return_identity() -> None:
    """Overnight, intraday and dividend pieces reconcile with close-to-close."""
    sessions = [date(2020, 3, 30), date(2020, 3, 31)]
    base = np.array([[100.0], [100.0]])
    close = np.array([[100.0], [99.0]])
    withholding = Decimal("0.154")
    dividends = pl.DataFrame(
        [
            {
                "instrument_id": INST,
                "ex_session": sessions[1],
                "dps_krw": 1,
                "available_at": _dt(sessions[0]),
            }
        ]
    )
    out = assemble_dividends(
        dividends, base, close, sessions=sessions, instrument_ids=[INST], withholding_rate=withholding
    )
    div_ret = out["div_ret"][1, 0]
    w = float(withholding)
    assert div_ret == pytest.approx(0.01 * (1 - w))
    r_on = 102 / 100 - 1 + div_ret
    r_id = 99 / 102 - 1
    ret_cc = 99 / 100 - 1
    assert r_on == pytest.approx(0.02 + 0.01 * (1 - w))
    assert (1 + r_on - div_ret) * (1 + r_id) == pytest.approx(1 + ret_cc, abs=1e-12)


def test_exit_indexing(tmp_path: Path) -> None:
    """Exits settle the session after the last trading day; panel-final rows never exit."""
    sessions = _sessions(8)
    frame = pl.DataFrame(
        [
            {"instrument_id": INST, "last_session": sessions[5], "last_close": 100, "last_volume": 10},
            {"instrument_id": INST2, "last_session": sessions[-1], "last_close": 100, "last_volume": 0},
        ]
    )
    frame.write_parquet(tmp_path / "instrument_exits.parquet")
    exit_at, halted = cube_mod._exit_arrays(tmp_path, sessions, [INST, INST2])
    assert list(exit_at) == [6, -1]
    assert list(halted) == [False, False]


def test_cache_integrity_rebuilds_on_flip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single flipped cache byte forces a rebuild instead of silent reuse."""
    from src.research.cube import load_research_cube

    sessions = _sessions(4)
    arrays = {"close": np.ones((4, 1)), "present": np.ones((4, 1), dtype=bool)}
    cube = ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=sessions,
        instrument_ids=[INST],
        arrays=arrays,
        exit_at=np.array([-1]),
        exit_halted=np.array([False]),
    )
    monkeypatch.setattr(cube_mod, "research_cube_id", lambda _inputs: cube.cube_id)
    monkeypatch.setattr(cube_mod, "build_research_cube", lambda _inputs: cube)
    inputs = object()
    first = load_research_cube(inputs, cache_root=tmp_path)  # type: ignore[arg-type]
    assert first.cube_id == cube.cube_id
    cache_file = tmp_path / f"{cube.cube_id}.npz"
    assert cache_file.is_file()
    calls = {"n": 0}
    rebuilt = ResearchCube.from_arrays(
        cube_id=cube.cube_id,
        sessions=sessions,
        instrument_ids=[INST],
        arrays={"close": np.full((4, 1), 2.0), "present": np.ones((4, 1), dtype=bool)},
        exit_at=np.array([-1]),
        exit_halted=np.array([False]),
    )

    def _rebuild(_inputs: object) -> ResearchCube:
        calls["n"] += 1
        return rebuilt

    monkeypatch.setattr(cube_mod, "build_research_cube", _rebuild)
    raw = cache_file.read_bytes()
    flipped = bytearray(raw)
    flipped[len(flipped) // 2] ^= 1
    cache_file.write_bytes(bytes(flipped))
    loaded = load_research_cube(inputs, cache_root=tmp_path)  # type: ignore[arg-type]
    assert calls["n"] == 1
    assert loaded.arrays["close"][0, 0] == pytest.approx(2.0)


def test_arrays_are_read_only() -> None:
    """Built cube arrays reject mutation."""
    sessions = _sessions(3)
    cube = ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=sessions,
        instrument_ids=[INST],
        arrays={"close": np.ones((3, 1))},
        exit_at=np.array([-1]),
        exit_halted=np.array([False]),
    )
    with pytest.raises(ValueError, match="read-only"):
        cube.arrays["close"][0, 0] = 5.0


def test_cube_id_is_stable_and_kind_checked(tmp_path: Path) -> None:
    """Cube ids hash inputs; unknown dataset kinds fail closed."""
    from src.core.market_rules import load_krx_market_rules

    rules = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))
    base = {
        "market_panel": Path("gold/scope/market_panel_0123456789abcdef"),
        "dividend_events": Path("silver/scope/dividend_events_0123456789abcdef"),
        "financial_facts": Path("silver/scope/financial_facts_0123456789abcdef"),
        "investor_flow": Path("silver/scope/investor_flow_0123456789abcdef"),
        "earnings_releases": Path("silver/scope/earnings_releases_0123456789abcdef"),
        "market_rules": rules,
        "dividend_withholding_rate": Decimal("0.154"),
    }
    first = research_cube_id(cube_mod.CubeInputs(**base))  # type: ignore[arg-type]
    second = research_cube_id(cube_mod.CubeInputs(**base))  # type: ignore[arg-type]
    assert first == second
    assert first.startswith("research_cube_")
    with pytest.raises(PITDataError):
        cube_mod._require_kind(tmp_path, "market_panel")


def test_helper_edge_branches() -> None:
    """Exercise parsing, mapping and validation fallbacks."""
    sessions = _sessions(4)
    with pytest.raises(PITDataError, match="wrong shape"):
        ResearchCube.from_arrays(
            cube_id="research_cube_0123456789abcdef",
            sessions=sessions,
            instrument_ids=[INST],
            arrays={"close": np.ones((4, 2))},
            exit_at=np.array([-1]),
            exit_halted=np.array([False]),
        )
    with pytest.raises(PITDataError, match="exit arrays"):
        ResearchCube.from_arrays(
            cube_id="research_cube_0123456789abcdef",
            sessions=sessions,
            instrument_ids=[INST],
            arrays={"close": np.ones((4, 1))},
            exit_at=np.array([-1, -1]),
            exit_halted=np.array([False]),
        )
    assert cube_mod._avail_index(None, sessions) is None
    assert cube_mod._avail_index(sessions[0], sessions) == 0
    assert cube_mod._avail_index("not-a-date", sessions) is None
    assert cube_mod._avail_index(_dt(sessions[-1], 19), sessions) is None
    assert cube_mod._parse_qk(123) is None
    assert cube_mod._parse_qk("nope") is None
    assert cube_mod._instrument_of({"ticker": "000001"}) == "KRX:000001"
    assert cube_mod._instrument_of({"company_id": "000002"}) == "KRX:000002"
    assert cube_mod._instrument_of({"instrument_id": "000003"}) == "KRX:000003"
    assert cube_mod._instrument_of({"ticker": "???"}) is None
    assert cube_mod._coverage((2, 1), np.ones((2, 1), dtype=bool)) == 1.0
    assert cube_mod._coverage((0, 0), np.zeros((0, 0))) == 0.0
    junk = pl.DataFrame(
        [
            {"company_id": "???", "fiscal_period": "2020Q1", "fact": "equity", "value": 1.0,
             "available_at": _dt(sessions[0]), "ticker": "???"},
            {"company_id": INST, "fiscal_period": "bogus", "fact": "equity", "value": 1.0,
             "available_at": _dt(sessions[0]), "ticker": INST.split(":")[1]},
            {"company_id": INST, "fiscal_period": "2020Q1", "fact": "unknown_fact", "value": 1.0,
             "available_at": _dt(sessions[0]), "ticker": INST.split(":")[1]},
            {"company_id": INST, "fiscal_period": "2020Q1", "fact": "equity", "value": None,
             "available_at": _dt(sessions[0]), "ticker": INST.split(":")[1]},
            {"company_id": INST, "fiscal_period": "2020Q1", "fact": "equity", "value": "bad",
             "available_at": _dt(sessions[0]), "ticker": INST.split(":")[1]},
            {"company_id": INST, "fiscal_period": "2020Q1", "fact": "equity", "value": float("inf"),
             "available_at": _dt(sessions[0]), "ticker": INST.split(":")[1]},
            {"company_id": INST, "fiscal_period": "2020Q1", "fact": "equity", "value": 1.0,
             "available_at": _dt(sessions[-1], 19), "ticker": INST.split(":")[1]},
        ]
    )
    out = assemble_fundamentals(junk, sessions=sessions, instrument_ids=[INST])
    assert bool(np.isnan(out["f_equity"][0, 0]))
    with pytest.raises(PITDataError, match="wrong shape"):
        assemble_dividends(
            pl.DataFrame([]), np.ones((2, 1)), np.ones((4, 1)),
            sessions=sessions, instrument_ids=[INST], withholding_rate=Decimal("0.1"),
        )
    with pytest.raises(PITDataError, match="unknown instrument"):
        cube_mod._check_references(
            pl.DataFrame([{"instrument_id": "KRX:999999"}]), {INST}, "probe",
        )
    with pytest.raises(PITDataError, match="unreadable dataset"):
        cube_mod._read_frame(Path("/nonexistent/dataset_xyz"))


def test_build_assembles_market_returns_and_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End-to-end build with mocked IO covers market math, exits and freezing."""
    from src.backtest.market import MarketArrays
    from src.core.market_rules import load_krx_market_rules

    sessions = _sessions(6)
    insts = [INST, INST2]
    n_s, n_n = len(sessions), len(insts)
    shape = (n_s, n_n)
    close_i = np.array([[100, 200], [101, 202], [102, 204], [103, 206], [104, 208], [105, 210]], dtype=np.int64)
    base_i = np.array([[100, 200], [100, 200], [101, 202], [102, 204], [103, 206], [104, 208]], dtype=np.int64)
    open_i = np.array([[100, 200], [100, 201], [101, 203], [102, 205], [103, 207], [104, 209]], dtype=np.int64)
    fake = MarketArrays(
        dataset_id="market_panel_0123456789abcdef",
        sessions=tuple(sessions),
        instrument_ids=tuple(insts),
        int_fields={
            "open": open_i, "high": open_i, "low": open_i, "close": close_i,
            "base_price": base_i, "volume": np.ones(shape, dtype=np.int64),
            "tick_size": np.zeros(shape, dtype=np.int64),
            "upper_limit": np.zeros(shape, dtype=np.int64),
            "lower_limit": np.zeros(shape, dtype=np.int64),
        },
        float_fields={
            "sell_tax_rate": np.zeros(shape), "adtv20": np.full(shape, 1e8),
            "ret_vol60": np.full(shape, 0.01), "share_factor": np.ones(shape),
        },
        bool_fields={
            "present": np.ones(shape, dtype=bool),
            "open_at_upper": np.zeros(shape, dtype=bool), "open_at_lower": np.zeros(shape, dtype=bool),
            "entry_blocked": np.zeros(shape, dtype=bool),
        },
        market=np.array([[1, 2]] * n_s, dtype=np.int8),
    )
    fake.market[0, 1] = 0
    monkeypatch.setattr(cube_mod, "load_market_arrays", lambda panel_dir, cache_root: fake)
    monkeypatch.setattr(
        cube_mod, "_panel_float_columns",
        lambda _dir, names, sessions, ids: {n: np.full((len(sessions), len(ids)), 5.0) for n in names},
    )
    monkeypatch.setattr(cube_mod, "_require_kind", lambda _path, _kind: None)
    avail = _dt(sessions[1])
    facts = pl.DataFrame(
        [
            _fact(INST, "2020Q1", "equity", 100.0, avail),
            _fact(INST, "2020Q1", "sales", 50.0, avail),
            _fact(INST2, "2020Q1", "equity", 200.0, avail),
            _fact(INST2, "2020Q1", "sales", 60.0, avail),
        ]
    )
    dividends = pl.DataFrame(
        [
            {"instrument_id": INST, "ex_session": sessions[2], "dps_krw": 2,
             "available_at": _dt(sessions[1])},
        ]
    )
    flows = pl.DataFrame(
        [
            {"instrument_id": INST, "session": sessions[0], "foreign_net_shares": 10,
             "institution_net_shares": 5, "individual_net_shares": -3,
             "available_at": _dt(sessions[1])},
            {"instrument_id": INST, "session": sessions[0], "foreign_net_shares": 99,
             "institution_net_shares": 99, "individual_net_shares": 99,
             "available_at": _dt(sessions[1])},
        ]
    )
    releases = pl.DataFrame(
        [
            {"instrument_id": INST, "fiscal_period": "2020Q1", "metric": "sales",
             "span": "quarter", "value_krw": 50.0, "prior_year_value_krw": 40.0,
             "available_at": _dt(sessions[2]), "basis": "consolidated",
             "release_kind": "preliminary"},
        ]
    )

    def _frame(path: Path, columns: object = None) -> pl.DataFrame:
        name = Path(path).name
        if "financial_facts" in name:
            return facts
        if "dividend" in name:
            return dividends
        if "investor_flow" in name:
            return flows
        return releases

    monkeypatch.setattr(cube_mod, "_read_frame", _frame)
    monkeypatch.setattr(
        cube_mod, "_exit_arrays",
        lambda _panel, _sess, _insts: (np.array([3, -1], dtype=np.int64), np.array([False, True])),
    )
    rules = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))
    inputs = cube_mod.CubeInputs(
        market_panel=tmp_path, dividend_events=tmp_path / "dividend",
        financial_facts=tmp_path / "financial_facts", investor_flow=tmp_path / "investor_flow",
        earnings_releases=tmp_path / "releases", market_rules=rules,
        dividend_withholding_rate=Decimal("0.1"),
    )
    cube = cube_mod.build_research_cube(inputs)
    assert cube.sessions == tuple(sessions)
    assert cube.arrays["r_on"].shape == shape
    assert cube.arrays["adj_px"][0, 0] == pytest.approx(1.0)
    assert list(cube.exit_at) == [3, -1]


def test_junk_releases_flows_and_dividends() -> None:
    """Malformed event rows are skipped without guessing."""
    sessions = _sessions(6)
    facts = pl.DataFrame(
        [
            _fact(INST, "2020Q1", "operating_profit", 10.0, _dt(sessions[0])),
            _fact(INST, "2020Q2", "operating_profit", 20.0, _dt(sessions[1])),
            _fact(INST, "2020Q3", "operating_profit", 30.0, _dt(sessions[1])),
            _fact(INST, "2020Q4", "equity", 99.0, _dt(sessions[2])),
        ]
    )
    fundamentals = assemble_fundamentals(facts, sessions=sessions, instrument_ids=[INST])
    assert bool(np.isnan(fundamentals["f_operating_profit_q"][2, 0]))
    releases = pl.DataFrame(
        [
            {"instrument_id": "KRX:999999", "fiscal_period": "2020Q1", "metric": "sales",
             "span": "quarter", "value_krw": 1.0, "prior_year_value_krw": 1.0,
             "available_at": _dt(sessions[1]), "basis": "consolidated", "release_kind": "preliminary"},
            {"instrument_id": INST, "fiscal_period": "bogus", "metric": "sales",
             "span": "quarter", "value_krw": 1.0, "prior_year_value_krw": 1.0,
             "available_at": _dt(sessions[1]), "basis": "consolidated", "release_kind": "preliminary"},
            {"instrument_id": INST, "fiscal_period": "2020Q1", "metric": "sales",
             "span": "quarter", "value_krw": 1.0, "prior_year_value_krw": 1.0,
             "available_at": _dt(sessions[-1], 19), "basis": "consolidated", "release_kind": "preliminary"},
            {"instrument_id": INST, "fiscal_period": "2020Q1", "metric": "sales",
             "span": "quarter", "value_krw": "bad", "prior_year_value_krw": "bad",
             "available_at": _dt(sessions[1]), "basis": "weird", "release_kind": "preliminary"},
            {"instrument_id": INST, "fiscal_period": "2020Q1", "metric": "unknown_metric",
             "span": "quarter", "value_krw": 1.0, "prior_year_value_krw": 1.0,
             "available_at": _dt(sessions[1]), "basis": "consolidated", "release_kind": "preliminary"},
            {"instrument_id": INST, "fiscal_period": "2020Q4", "metric": "operating_profit",
             "span": "annual", "value_krw": "bad", "prior_year_value_krw": "bad",
             "available_at": _dt(sessions[2]), "basis": "consolidated", "release_kind": "profit_change"},
            {"instrument_id": INST, "fiscal_period": "2020Q4", "metric": "operating_profit",
             "span": "annual", "value_krw": 100.0, "prior_year_value_krw": 80.0,
             "available_at": _dt(sessions[2]), "basis": "consolidated", "release_kind": "profit_change"},
        ]
    )
    merged = assemble_releases(releases, fundamentals, facts, sessions=sessions, instrument_ids=[INST])
    assert merged["e_qk"].shape == (len(sessions), 1)
    close = np.full((len(sessions), 1), 5000.0)
    flows = pl.DataFrame(
        [
            {"instrument_id": INST, "session": "not-a-date", "foreign_net_shares": 1,
             "institution_net_shares": 1, "individual_net_shares": 1, "available_at": _dt(sessions[1])},
            {"instrument_id": INST, "session": date(1999, 1, 4).isoformat(), "foreign_net_shares": 1,
             "institution_net_shares": 1, "individual_net_shares": 1, "available_at": _dt(sessions[1])},
            {"instrument_id": INST, "session": sessions[0].isoformat(), "foreign_net_shares": 1,
             "institution_net_shares": 1, "individual_net_shares": 1,
             "available_at": _dt(sessions[-1], 19)},
            {"instrument_id": INST, "session": sessions[1].isoformat(), "foreign_net_shares": "bad",
             "institution_net_shares": 1, "individual_net_shares": 1, "available_at": _dt(sessions[2])},
            {"instrument_id": INST, "session": sessions[2].isoformat(), "foreign_net_shares": 7,
             "institution_net_shares": 8, "individual_net_shares": 9, "available_at": _dt(sessions[3])},
        ]
    )
    flow_out = assemble_flows(flows, close, sessions=sessions, instrument_ids=[INST])
    assert flow_out["flow_for_krw"][3, 0] == pytest.approx(7 * 5000.0)
    dividends = pl.DataFrame(
        [
            {"instrument_id": INST, "ex_session": "bogus", "dps_krw": 1, "available_at": _dt(sessions[0])},
            {"instrument_id": INST, "ex_session": date(1999, 1, 4).isoformat(), "dps_krw": 1,
             "available_at": _dt(sessions[0])},
            {"instrument_id": INST, "ex_session": sessions[1].isoformat(), "dps_krw": None,
             "available_at": _dt(sessions[0])},
            {"instrument_id": INST, "ex_session": sessions[1].isoformat(), "dps_krw": "bad",
             "available_at": _dt(sessions[0])},
            {"instrument_id": INST, "ex_session": sessions[1].isoformat(), "dps_krw": -5,
             "available_at": _dt(sessions[0])},
            {"instrument_id": INST, "ex_session": sessions[1].isoformat(), "dps_krw": 3,
             "available_at": _dt(sessions[-1], 19)},
            {"instrument_id": INST, "ex_session": sessions[1].isoformat(), "dps_krw": 4,
             "available_at": _dt(sessions[0])},
        ]
    )
    base = np.full((len(sessions), 1), 100.0)
    div_out = assemble_dividends(
        dividends, base, close, sessions=sessions, instrument_ids=[INST],
        withholding_rate=Decimal("0"),
    )
    assert div_out["div_ret"][1, 0] == pytest.approx(0.04)


def test_require_kind_and_reference_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Dataset kind gates and bare-ticker references fail closed."""
    from types import SimpleNamespace

    monkeypatch.setattr(cube_mod, "load_manifest", lambda _p: SimpleNamespace(kind="market_panel"))
    cube_mod._require_kind(tmp_path, "market_panel")
    monkeypatch.setattr(cube_mod, "load_manifest", lambda _p: SimpleNamespace(kind="other"))
    with pytest.raises(PITDataError, match="kind mismatch"):
        cube_mod._require_kind(tmp_path, "market_panel")
    with pytest.raises(PITDataError, match="unknown instrument"):
        cube_mod._check_references(
            pl.DataFrame([{"instrument_id": "999999"}]), {INST}, "probe",
        )


def test_build_validation_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Duplicate fact keys and unknown instruments abort the build."""
    from src.backtest.market import MarketArrays
    from src.core.market_rules import load_krx_market_rules

    sessions = _sessions(4)
    insts = [INST]
    shape = (len(sessions), 1)
    zeros_i = np.zeros(shape, dtype=np.int64)
    ones_i = np.ones(shape, dtype=np.int64)
    fake = MarketArrays(
        dataset_id="market_panel_0123456789abcdef",
        sessions=tuple(sessions),
        instrument_ids=tuple(insts),
        int_fields={"open": ones_i, "high": ones_i, "low": ones_i, "close": ones_i,
                    "base_price": ones_i, "volume": ones_i, "tick_size": zeros_i,
                    "upper_limit": zeros_i, "lower_limit": zeros_i},
        float_fields={"sell_tax_rate": np.zeros(shape), "adtv20": np.zeros(shape),
                      "ret_vol60": np.zeros(shape), "share_factor": np.ones(shape)},
        bool_fields={"present": np.ones(shape, dtype=bool), "eligible": np.ones(shape, dtype=bool),
                     "open_at_upper": np.zeros(shape, dtype=bool),
                     "open_at_lower": np.zeros(shape, dtype=bool),
                     "entry_blocked": np.zeros(shape, dtype=bool)},
        market=np.zeros(shape, dtype=np.int8),
    )
    monkeypatch.setattr(cube_mod, "load_market_arrays", lambda panel_dir, cache_root: fake)
    monkeypatch.setattr(cube_mod, "_require_kind", lambda _path, _kind: None)
    rules = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))

    def _inputs() -> cube_mod.CubeInputs:
        return cube_mod.CubeInputs(
            market_panel=tmp_path, dividend_events=tmp_path / "d",
            financial_facts=tmp_path / "f", investor_flow=tmp_path / "i",
            earnings_releases=tmp_path / "e", market_rules=rules,
            dividend_withholding_rate=Decimal("0"),
        )

    dup_facts = pl.DataFrame(
        [
            {"company_id": INST, "fiscal_period": "2020Q1", "fact": "equity", "value": 1.0,
             "available_at": _dt(sessions[0]), "ticker": INST.split(":")[1]},
            {"company_id": INST, "fiscal_period": "2020Q1", "fact": "equity", "value": 2.0,
             "available_at": _dt(sessions[1]), "ticker": INST.split(":")[1]},
        ]
    )
    empty = pl.DataFrame([])
    monkeypatch.setattr(cube_mod, "_read_frame", lambda _p, columns=None: dup_facts)
    with pytest.raises(PITDataError, match="duplicated"):
        cube_mod.build_research_cube(_inputs())
    good_facts = pl.DataFrame(
        [{"company_id": INST, "fiscal_period": "2020Q1", "fact": "equity", "value": 1.0,
          "available_at": _dt(sessions[0]), "ticker": INST.split(":")[1]}]
    )
    bad_flows = pl.DataFrame([{"instrument_id": "KRX:999999", "session": sessions[0]}])

    def _frame(path: Path, columns: object = None) -> pl.DataFrame:
        name = Path(path).name
        if name == "f":
            return good_facts
        if name == "i":
            return bad_flows
        return empty

    monkeypatch.setattr(cube_mod, "_read_frame", _frame)
    with pytest.raises(PITDataError, match="unknown instrument"):
        cube_mod.build_research_cube(_inputs())


def test_cache_id_and_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale cube ids and edited checksums never load silently."""
    import numpy as np

    sessions = _sessions(4)
    arrays = {"close": np.ones((4, 1)), "present": np.ones((4, 1), dtype=bool)}
    cube = ResearchCube.from_arrays(
        cube_id="research_cube_0123456789abcdef",
        sessions=sessions,
        instrument_ids=[INST],
        arrays=arrays,
        exit_at=np.array([-1]),
        exit_halted=np.array([False]),
    )
    monkeypatch.setattr(cube_mod, "research_cube_id", lambda _inputs: cube.cube_id)
    monkeypatch.setattr(cube_mod, "build_research_cube", lambda _inputs: cube)
    inputs = object()
    cube_mod.load_research_cube(inputs, cache_root=tmp_path)  # type: ignore[arg-type]
    cache_file = tmp_path / f"{cube.cube_id}.npz"
    with np.load(str(cache_file), allow_pickle=True) as store:
        payload = {key: store[key] for key in store.files}
    payload["cube_id"] = np.asarray("research_cube_ffffffffffffffff")
    np.savez(str(cache_file), **payload)
    calls = {"n": 0}

    def _rebuild(_inputs: object) -> ResearchCube:
        calls["n"] += 1
        return cube

    monkeypatch.setattr(cube_mod, "build_research_cube", _rebuild)
    cube_mod.load_research_cube(inputs, cache_root=tmp_path)  # type: ignore[arg-type]
    assert calls["n"] == 1
    with np.load(str(cache_file), allow_pickle=True) as store:
        payload = {key: store[key] for key in store.files}
    payload["checksum"] = np.asarray("0" * 64)
    np.savez(str(cache_file), **payload)
    cube_mod.load_research_cube(inputs, cache_root=tmp_path)  # type: ignore[arg-type]
    assert calls["n"] == 2


def test_exits_edge_cases(tmp_path: Path) -> None:
    """String dates parse, bad rows skip, empty panels stay open."""
    sessions = _sessions(6)
    frame = pl.DataFrame(
        [
            {"instrument_id": INST, "last_session": "2020-04-05",
             "last_close": 10, "last_volume": 5},
            {"instrument_id": INST, "last_session": sessions[2].isoformat(),
             "last_close": 10, "last_volume": 5},
            {"instrument_id": "KRX:999999", "last_session": sessions[2].isoformat(),
             "last_close": 10, "last_volume": 5},
            {"instrument_id": INST2, "last_session": "bogus", "last_close": 10, "last_volume": 5},
            {"instrument_id": INST2, "last_session": sessions[3].isoformat(), "last_close": 10, "last_volume": "bad"},
        ]
    )
    frame.write_parquet(tmp_path / "instrument_exits.parquet")
    exit_at, halted = cube_mod._exit_arrays(tmp_path, sessions, [INST, INST2])
    assert list(exit_at) == [3, 4]
    assert list(halted) == [False, False]
    empty_dir = tmp_path / "empty_panel"
    empty_dir.mkdir()
    exit_at, _ = cube_mod._exit_arrays(empty_dir, sessions, [INST])
    assert list(exit_at) == [-1]


def test_trailing_history_and_idle_instrument() -> None:
    """Eight-quarter histories fill TTM-LY; instruments without facts stay NaN."""
    sessions = _sessions(12)
    rows: list[dict[str, object]] = []
    for i, quarter in enumerate(
        ("2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3")
    ):
        rows.append(_fact(INST, quarter, "operating_profit", 10.0 + i, _dt(sessions[i])))
    rows.append(_fact(INST, "2020Q4", "operating_profit", 200.0, _dt(sessions[7])))
    for i, quarter in enumerate(("2019Q1", "2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3")):
        rows.append(_fact(INST, quarter, "sales", 5.0, _dt(sessions[i])))
    rows.append(_fact(INST, "2020Q4", "sales", 100.0, _dt(sessions[7])))
    for i, quarter in enumerate(("2019Q1", "2019Q2", "2019Q3")):
        rows.append(_fact(INST, quarter, "capex", 2.0, _dt(sessions[i])))
    for i, quarter in enumerate(("2020Q1", "2020Q2", "2020Q3")):
        rows.append(_fact(INST, quarter, "capex", 3.0, _dt(sessions[i + 4])))
    rows.append(_fact(INST, "2020Q4", "capex", 30.0, _dt(sessions[7])))
    out = assemble_fundamentals(
        pl.DataFrame(rows), sessions=sessions, instrument_ids=[INST, INST2]
    )
    assert bool(np.isfinite(out["f_operating_profit_ttm_ly"][8, 0]))
    assert bool(np.isnan(out["f_sales_ttm_ly"][8, 0]))
    assert bool(np.isnan(out["f_capex_ttm_ly"][8, 0]))
    assert bool(np.isnan(out["f_equity"][:, 1]).all())


def test_release_nan_and_separate_only() -> None:
    """NaN release values stay undefined; separate-only quarters still display."""
    sessions = _sessions(8)
    facts = pl.DataFrame([_fact(INST, "2020Q1", "sales", 10.0, _dt(sessions[0]))])
    fundamentals = assemble_fundamentals(facts, sessions=sessions, instrument_ids=[INST])
    releases = pl.DataFrame(
        [
            {"instrument_id": INST, "fiscal_period": "2020Q1", "metric": "sales",
             "span": "quarter", "value_krw": float("nan"), "prior_year_value_krw": float("nan"),
             "available_at": _dt(sessions[1]), "basis": "separate", "release_kind": "preliminary"},
            {"instrument_id": INST, "fiscal_period": "2020Q2", "metric": "operating_profit",
             "span": "annual", "value_krw": 50.0, "prior_year_value_krw": 40.0,
             "available_at": _dt(sessions[2]), "basis": "consolidated", "release_kind": "profit_change"},
        ]
    )
    merged = assemble_releases(releases, fundamentals, facts, sessions=sessions, instrument_ids=[INST])
    assert bool(np.isnan(merged["e_sales_q"][1, 0]))
    assert merged["e_qk"][1, 0] == pytest.approx(2020 * 4 + 1 - 1)
    assert merged["e_qk"][2, 0] == pytest.approx(2020 * 4 + 2 - 1)
    assert bool(np.isnan(merged["e_operating_profit_q"][3, 0]))


def test_unknown_instruments_skipped_in_events() -> None:
    """Events for instruments outside the panel never enter the arrays."""
    sessions = _sessions(4)
    close = np.full((4, 1), 100.0)
    flows = pl.DataFrame(
        [
            {"instrument_id": "KRX:999999", "session": sessions[0].isoformat(),
             "foreign_net_shares": 5, "institution_net_shares": 5, "individual_net_shares": 5,
             "available_at": _dt(sessions[1])},
        ]
    )
    out = assemble_flows(flows, close, sessions=sessions, instrument_ids=[INST])
    assert bool(np.isnan(out["flow_for_krw"]).all())
    dividends = pl.DataFrame(
        [
            {"instrument_id": "KRX:999999", "ex_session": sessions[1].isoformat(),
             "dps_krw": 5, "available_at": _dt(sessions[0])},
        ]
    )
    div_out = assemble_dividends(
        dividends, close, close, sessions=sessions, instrument_ids=[INST],
        withholding_rate=Decimal("0"),
    )
    assert bool((div_out["div_ret"] == 0.0).all())


def test_build_facts_unknown_instrument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Facts pointing outside the panel abort the build."""
    from src.backtest.market import MarketArrays
    from src.core.market_rules import load_krx_market_rules

    sessions = _sessions(4)
    shape = (len(sessions), 1)
    ones_i = np.ones(shape, dtype=np.int64)
    zeros_i = np.zeros(shape, dtype=np.int64)
    fake = MarketArrays(
        dataset_id="market_panel_0123456789abcdef",
        sessions=tuple(sessions),
        instrument_ids=(INST,),
        int_fields={"open": ones_i, "high": ones_i, "low": ones_i, "close": ones_i,
                    "base_price": ones_i, "volume": ones_i, "tick_size": zeros_i,
                    "upper_limit": zeros_i, "lower_limit": zeros_i},
        float_fields={"sell_tax_rate": np.zeros(shape), "adtv20": np.zeros(shape),
                      "ret_vol60": np.zeros(shape), "share_factor": np.ones(shape)},
        bool_fields={"present": np.ones(shape, dtype=bool), "eligible": np.ones(shape, dtype=bool),
                     "open_at_upper": np.zeros(shape, dtype=bool),
                     "open_at_lower": np.zeros(shape, dtype=bool),
                     "entry_blocked": np.zeros(shape, dtype=bool)},
        market=np.zeros(shape, dtype=np.int8),
    )
    monkeypatch.setattr(cube_mod, "load_market_arrays", lambda panel_dir, cache_root: fake)
    monkeypatch.setattr(
        cube_mod, "_panel_float_columns",
        lambda _dir, names, sessions, ids: {n: np.full((len(sessions), len(ids)), 5.0) for n in names},
    )
    monkeypatch.setattr(cube_mod, "_require_kind", lambda _path, _kind: None)
    facts = pl.DataFrame(
        [{"company_id": "KRX:999999", "fiscal_period": "2020Q1", "fact": "equity", "value": 1.0,
          "available_at": _dt(sessions[0]), "ticker": "999999"}]
    )
    empty_flows = pl.DataFrame(
        schema={"instrument_id": pl.String, "session": pl.Date, "foreign_net_shares": pl.Int64,
                "institution_net_shares": pl.Int64, "individual_net_shares": pl.Int64,
                "available_at": pl.Datetime("us", "Asia/Seoul")}
    )

    def _frame(path: Path, columns: object = None) -> pl.DataFrame:
        if path.name == "f":
            return facts
        return empty_flows if path.name == "i" else pl.DataFrame([])

    monkeypatch.setattr(cube_mod, "_read_frame", _frame)
    rules = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))
    inputs = cube_mod.CubeInputs(
        market_panel=tmp_path, dividend_events=tmp_path / "d",
        financial_facts=tmp_path / "f", investor_flow=tmp_path / "i",
        earnings_releases=tmp_path / "e", market_rules=rules,
        dividend_withholding_rate=Decimal("0"),
    )
    cube = cube_mod.build_research_cube(inputs)
    assert bool(np.isnan(cube.arrays["f_equity"]).all())


def test_fact_instruments_outside_panel_are_reported() -> None:
    """Facts of instruments that never traded are identified, not treated as build errors."""
    facts = pl.DataFrame(
        [
            {"company_id": "000001", "ticker": "000001"},
            {"company_id": "000800", "ticker": "000800"},
        ]
    )

    assert cube_mod._fact_instruments_outside_panel(facts, {INST}) == {"KRX:000800"}
    assert cube_mod._fact_instruments_outside_panel(pl.DataFrame({"x": [1]}), {INST}) == set()


def test_cache_hit_returns_verified_cube(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An untouched cache loads without rebuilding."""
    sessions = _sessions(4)
    cube = ResearchCube.from_arrays(
        cube_id="research_cube_abcdef0123456789",
        sessions=sessions,
        instrument_ids=[INST],
        arrays={"close": np.full((4, 1), 3.0), "present": np.ones((4, 1), dtype=bool)},
        exit_at=np.array([-1]),
        exit_halted=np.array([False]),
    )
    monkeypatch.setattr(cube_mod, "research_cube_id", lambda _inputs: cube.cube_id)
    monkeypatch.setattr(cube_mod, "build_research_cube", lambda _inputs: cube)
    inputs = object()
    cube_mod.load_research_cube(inputs, cache_root=tmp_path)  # type: ignore[arg-type]

    def _forbid(_inputs: object) -> ResearchCube:
        raise AssertionError("rebuild must not run on a cache hit")

    monkeypatch.setattr(cube_mod, "build_research_cube", _forbid)
    loaded = cube_mod.load_research_cube(inputs, cache_root=tmp_path)  # type: ignore[arg-type]
    assert loaded.arrays["close"][0, 0] == pytest.approx(3.0)


def test_flow_skipped_without_reference_price() -> None:
    """Flows on sessions without a usable close never enter the arrays."""
    sessions = _sessions(4)
    close = np.full((4, 1), 5000.0)
    close[1, 0] = 0.0
    flows = pl.DataFrame(
        [
            {"instrument_id": INST, "session": sessions[1].isoformat(), "foreign_net_shares": 5,
             "institution_net_shares": 5, "individual_net_shares": 5,
             "available_at": _dt(sessions[2])},
        ]
    )
    out = assemble_flows(flows, close, sessions=sessions, instrument_ids=[INST])
    assert bool(np.isnan(out["flow_for_krw"]).all())


def test_flow_contract_violations_fail_closed() -> None:
    """Missing columns, naive availability and a misshaped close array raise instead of dropping flows."""
    sessions = _sessions(3)
    close = np.full((3, 1), 5000.0)
    row = {"instrument_id": INST, "session": sessions[0], "foreign_net_shares": 1,
           "institution_net_shares": 1, "individual_net_shares": 1, "available_at": _dt(sessions[1])}
    with pytest.raises(PITDataError, match="required columns"):
        assemble_flows(pl.DataFrame([{k: v for k, v in row.items() if k != "individual_net_shares"}]),
                       close, sessions=sessions, instrument_ids=[INST])
    naive = dict(row, available_at=datetime(2020, 3, 31, 8, 0))
    with pytest.raises(PITDataError, match="timezone-aware"):
        assemble_flows(pl.DataFrame([naive]), close, sessions=sessions, instrument_ids=[INST])
    with pytest.raises(PITDataError, match="shape"):
        assemble_flows(pl.DataFrame([row]), np.full((2, 1), 5000.0), sessions=sessions, instrument_ids=[INST])


def test_empty_flow_frame_yields_nan_arrays() -> None:
    """No flow rows means every flow cell is unknown (NaN), never zero."""
    sessions = _sessions(3)
    empty = pl.DataFrame(
        schema={"instrument_id": pl.String, "session": pl.Date, "foreign_net_shares": pl.Int64,
                "institution_net_shares": pl.Int64, "individual_net_shares": pl.Int64,
                "available_at": pl.Datetime("us", "Asia/Seoul")}
    )
    out = assemble_flows(empty, np.full((3, 1), 5000.0), sessions=sessions, instrument_ids=[INST])
    assert all(bool(np.isnan(arr).all()) for arr in out.values())


def test_panel_float_columns_read_real_parquet_partitions(tmp_path: Path) -> None:
    """Market cap, trading value and listed shares come from the panel parquet, not the dense loader."""
    import hashlib
    import json

    sessions = _sessions(2)
    ids = [INST, INST2]
    dataset = tmp_path / "market_panel_x"
    rows = [
        {"session": sessions[0], "instrument_id": INST, "trading_value": 10, "market_cap": 1000, "listed_shares": 7},
        {"session": sessions[1], "instrument_id": INST, "trading_value": 20, "market_cap": 2000, "listed_shares": 7},
        {"session": sessions[1], "instrument_id": INST2, "trading_value": 30, "market_cap": None, "listed_shares": 9},
    ]
    part = dataset / "year=2020" / "part.parquet"
    part.parent.mkdir(parents=True)
    frame = pl.DataFrame(rows)
    frame.write_parquet(part)
    (dataset / "manifest.json").write_text(
        json.dumps(
            {"dataset_id": dataset.name, "partitions": [{
                "year": 2020, "path": "year=2020/part.parquet", "row_count": frame.height,
                "parquet_sha256": hashlib.sha256(part.read_bytes()).hexdigest()}]},
        ),
        encoding="utf-8",
    )

    out = cube_mod._panel_float_columns(dataset, ("market_cap", "trading_value"), sessions, ids)

    assert out["market_cap"][0, 0] == 1000.0
    assert out["market_cap"][1, 0] == 2000.0
    assert bool(np.isnan(out["market_cap"][1, 1]))
    assert out["trading_value"][1, 1] == 30.0
    assert bool(np.isnan(out["trading_value"][0, 1]))
