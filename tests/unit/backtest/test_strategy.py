"""Strategy targets validation and EqualWeightLiquid baseline tests."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import pytest

from src.backtest.strategy import STRATEGIES, EqualWeightLiquid, PortfolioSnapshot, Targets
from src.backtest.view import PITView
from src.core.time import KRX_TZ
from tests.unit.backtest.test_market import _mrow, _write_panel

from src.backtest.market import load_market_arrays


_PANEL_SEQ = 0


def _view(tmp_path: Any, session: date, rows: list[dict[str, Any]], t: int) -> PITView:
    global _PANEL_SEQ
    _PANEL_SEQ += 1
    panel = _write_panel(tmp_path / "gold", f"market_panel_{_PANEL_SEQ}", rows)
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    return PITView(
        arrays=arrays,
        t=t,
        decision_time=datetime.combine(session, time(18, 0), tzinfo=KRX_TZ),
        asof_tables={},
    )


def _snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(session_idx=0, cash=0, nav=0, positions={})


def test_targets_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        Targets(weights={0: 0.7, 1: 0.5})
    with pytest.raises(ValueError):
        Targets(weights={0: -0.1})
    with pytest.raises(ValueError):
        Targets(weights={0: float("nan")})
    with pytest.raises(ValueError):
        Targets(weights={0: float("inf")})
    with pytest.raises(ValueError):
        Targets(weights={0: True})
    with pytest.raises(ValueError):
        Targets(weights={0: "0.5"})
    assert Targets(weights={0: 0.5, 1: 0.5}).weights == {0: 0.5, 1: 0.5}
    assert Targets(weights={}).weights == {}


def test_equal_weight_liquid_respects_liquidity_floor(tmp_path: Any) -> None:
    day = date(2020, 1, 6)
    view = _view(
        tmp_path,
        day,
        [
            _mrow(day, "KRX:A", adtv20=500_000_000.0),
            _mrow(day, "KRX:B", adtv20=2_000_000_000.0),
            _mrow(day, "KRX:C", adtv20=3_000_000_000.0),
        ],
        t=0,
    )
    strategy = EqualWeightLiquid(min_adtv20_krw=1_000_000_000, max_names=10)
    assert strategy.decide(view, _snapshot()).weights == {1: 0.5, 2: 0.5}


def test_equal_weight_liquid_selects_top_names_and_empty(tmp_path: Any) -> None:
    day = date(2020, 1, 6)
    rows = [
        _mrow(day, "KRX:A", adtv20=500_000_000.0),
        _mrow(day, "KRX:B", adtv20=2_000_000_000.0),
        _mrow(day, "KRX:C", adtv20=3_000_000_000.0),
    ]
    view = _view(tmp_path, day, rows, t=0)
    assert EqualWeightLiquid(min_adtv20_krw=0, max_names=1).decide(view, _snapshot()).weights == {
        2: 1.0
    }
    tie_rows = [
        _mrow(day, "KRX:A", adtv20=1_000_000_000.0),
        _mrow(day, "KRX:B", adtv20=1_000_000_000.0),
    ]
    tie_view = _view(tmp_path, day, tie_rows, t=0)
    assert EqualWeightLiquid(min_adtv20_krw=0, max_names=1).decide(
        tie_view, _snapshot()
    ).weights == {0: 1.0}
    assert EqualWeightLiquid(
        min_adtv20_krw=9_999_999_999_999, max_names=10
    ).decide(view, _snapshot()).weights == {}


def test_equal_weight_liquid_rebalances_monthly(tmp_path: Any) -> None:
    sessions = [date(2020, 1, 30), date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 4)]
    panel = _write_panel(
        tmp_path / "gold", "market_panel_test", [_mrow(day, "KRX:A") for day in sessions]
    )
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    strategy = EqualWeightLiquid(min_adtv20_krw=0, max_names=5)
    seen = []
    for t, day in enumerate(sessions):
        view = PITView(
            arrays=arrays,
            t=t,
            decision_time=datetime.combine(day, time(18, 0), tzinfo=KRX_TZ),
            asof_tables={},
        )
        seen.append(strategy.is_rebalance(view))
    assert seen == [True, False, True, False]


def test_equal_weight_liquid_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        EqualWeightLiquid(min_adtv20_krw=-1, max_names=5)
    with pytest.raises(ValueError):
        EqualWeightLiquid(min_adtv20_krw=0, max_names=0)
    with pytest.raises(ValueError):
        EqualWeightLiquid(min_adtv20_krw=True, max_names=5)


def test_strategies_registry_exposes_baseline() -> None:
    strategy = STRATEGIES["equal_weight_liquid"](min_adtv20_krw=0, max_names=5)
    assert strategy.name == "equal_weight_liquid"
    assert strategy.params() == {"min_adtv20_krw": 0, "max_names": 5}
    assert all(isinstance(v, (str, int, float, bool)) for v in strategy.params().values())
