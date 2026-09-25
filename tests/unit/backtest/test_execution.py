"""Execution pricing: auction bases, locks, capacity, and tick rounding."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from src.backtest.costs import CostConfig, Side
from src.backtest.execution import (
    ExecutionConfig,
    ExecutionScenario,
    Order,
    price_orders,
)
from src.backtest.market import MarketArrays, load_market_arrays
from src.core.market_rules import KrxMarketRules, load_krx_market_rules
from src.core.pit import PITDataError
from tests.unit.backtest.test_market import DAY0, DAY1, _mrow, _write_panel

RULES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "market" / "krx_market_rules.toml"


def _rules() -> KrxMarketRules:
    return load_krx_market_rules(RULES_PATH)


def _costs(impact_k: float = 1.0) -> CostConfig:
    return CostConfig(commission_rate=Decimal("0.00015"), impact_k=impact_k)


def _arrays(tmp_path: Path, name: str, rows: list[dict[str, Any]]) -> MarketArrays:
    panel = _write_panel(tmp_path / "gold", name, rows)
    return load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")


def _open_config() -> ExecutionConfig:
    return ExecutionConfig(
        scenario=ExecutionScenario.OPEN_AUCTION, max_participation=0.1, carry_unfilled=False
    )


def test_open_auction_fills_at_open_with_adverse_tick_rounding(tmp_path: Path) -> None:
    prev = _mrow(DAY0, "KRX:A", close=10_050, adtv20=100_500_000.0, ret_vol60=0.03)
    curr = _mrow(
        DAY1,
        "KRX:A",
        open=10_050,
        high=10_100,
        low=10_000,
        close=10_080,
        upper_limit=12_000,
        lower_limit=8_000,
    )
    arrays = _arrays(tmp_path, "market_panel_test", [prev, curr])
    rules = _rules()
    buy = Order(instrument_idx=0, side=Side.BUY, quantity=100, decision_session_idx=0)
    sell = Order(instrument_idx=0, side=Side.SELL, quantity=100, decision_session_idx=0)
    fills, rejects = price_orders(
        orders=(buy, sell), arrays=arrays, t=1, config=_open_config(), costs=_costs(), rules=rules
    )
    assert rejects == ()
    assert [(fill.quantity, fill.price) for fill in fills] == [(100, 10_100), (100, 10_000)]


def test_halted_session_rejects(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=1e9, ret_vol60=0.02),
            _mrow(DAY1, "KRX:A", open=10_000, volume=0),
        ],
    )
    order = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=0)
    fills, rejects = price_orders(
        orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(), rules=_rules()
    )
    assert fills == ()
    assert [reject.reason for reject in rejects] == ["halted"]


def test_missing_session_rejects(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=1e9, ret_vol60=0.02),
            _mrow(DAY0, "KRX:B", close=10_000, adtv20=1e9, ret_vol60=0.02),
            _mrow(DAY1, "KRX:A", open=10_000),
        ],
    )
    order = Order(instrument_idx=1, side=Side.BUY, quantity=10, decision_session_idx=0)
    fills, rejects = price_orders(
        orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(), rules=_rules()
    )
    assert fills == ()
    assert [reject.reason for reject in rejects] == ["missing_price"]


def test_open_at_upper_blocks_buys_only(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=1e9, ret_vol60=0.02),
            _mrow(
                DAY1,
                "KRX:A",
                open=11_000,
                high=11_000,
                low=10_500,
                close=10_900,
                upper_limit=11_000,
                lower_limit=9_000,
                open_at_upper=True,
            ),
        ],
    )
    buy = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=0)
    sell = Order(instrument_idx=0, side=Side.SELL, quantity=10, decision_session_idx=0)
    fills, rejects = price_orders(
        orders=(buy, sell), arrays=arrays, t=1, config=_open_config(), costs=_costs(impact_k=0.0),
        rules=_rules(),
    )
    assert [fill.order.side for fill in fills] == [Side.SELL]
    assert [reject.reason for reject in rejects] == ["limit_locked"]


def test_open_at_lower_blocks_sells(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=1e9, ret_vol60=0.02),
            _mrow(DAY1, "KRX:A", open=9_000, lower_limit=9_000, open_at_lower=True),
        ],
    )
    order = Order(instrument_idx=0, side=Side.SELL, quantity=10, decision_session_idx=0)
    _, rejects = price_orders(
        orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(impact_k=0.0),
        rules=_rules(),
    )
    assert [reject.reason for reject in rejects] == ["limit_locked"]


def test_capacity_uses_decision_day_liquidity(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=100_000_000.0, ret_vol60=0.02),
            _mrow(DAY1, "KRX:A", open=10_000, adtv20=1_000_000_000_000.0,
                  upper_limit=12_000, lower_limit=8_000),
        ],
    )
    config = ExecutionConfig(
        scenario=ExecutionScenario.OPEN_AUCTION, max_participation=0.01, carry_unfilled=False
    )
    order = Order(instrument_idx=0, side=Side.BUY, quantity=500, decision_session_idx=0)
    fills, rejects = price_orders(
        orders=(order,), arrays=arrays, t=1, config=config, costs=_costs(impact_k=0.0),
        rules=_rules(),
    )
    assert [(fill.quantity, fill.price) for fill in fills] == [(100, 10_000)]
    assert [reject.reason for reject in rejects] == ["capacity"]


def test_zero_participation_budget_rejects(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=None, ret_vol60=0.02),
            _mrow(DAY1, "KRX:A", open=10_000),
        ],
    )
    order = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=0)
    fills, rejects = price_orders(
        orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(), rules=_rules()
    )
    assert fills == ()
    assert [reject.reason for reject in rejects] == ["capacity"]


def test_stale_order_rejected(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [_mrow(DAY0, "KRX:A", close=10_000), _mrow(DAY1, "KRX:A", open=10_000)],
    )
    order = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=-1)
    with pytest.raises(PITDataError):
        price_orders(
            orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(),
            rules=_rules(),
        )


def test_invalid_side_rejected(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [_mrow(DAY0, "KRX:A", close=10_000), _mrow(DAY1, "KRX:A", open=10_000)],
    )
    order = Order(instrument_idx=0, side="buy", quantity=10, decision_session_idx=0)  # type: ignore[arg-type]
    with pytest.raises(PITDataError):
        price_orders(
            orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(),
            rules=_rules(),
        )


def test_buy_capped_at_upper_limit(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=12_900, adtv20=129_000_000.0, ret_vol60=0.1),
            _mrow(DAY1, "KRX:A", open=12_900, upper_limit=13_000, lower_limit=10_000),
        ],
    )
    order = Order(instrument_idx=0, side=Side.BUY, quantity=100, decision_session_idx=0)
    fills, _ = price_orders(
        orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(), rules=_rules()
    )
    assert [(fill.quantity, fill.price) for fill in fills] == [(100, 13_000)]


def test_sell_floored_at_lower_limit(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=7_100, adtv20=71_000_000.0, ret_vol60=0.2),
            _mrow(DAY1, "KRX:A", open=7_100, upper_limit=9_000, lower_limit=7_000),
        ],
    )
    order = Order(instrument_idx=0, side=Side.SELL, quantity=100, decision_session_idx=0)
    fills, _ = price_orders(
        orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(), rules=_rules()
    )
    assert [(fill.quantity, fill.price) for fill in fills] == [(100, 7_000)]


def test_close_auction_fills_at_close(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=9_813, adtv20=1e9, ret_vol60=0.02),
            _mrow(DAY1, "KRX:A", open=9_800, close=9_813, upper_limit=11_000, lower_limit=8_000),
        ],
    )
    config = ExecutionConfig(
        scenario=ExecutionScenario.CLOSE_AUCTION, max_participation=1.0, carry_unfilled=False
    )
    buy = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=0)
    sell = Order(instrument_idx=0, side=Side.SELL, quantity=10, decision_session_idx=0)
    fills, rejects = price_orders(
        orders=(buy, sell), arrays=arrays, t=1, config=config, costs=_costs(impact_k=0.0),
        rules=_rules(),
    )
    assert rejects == ()
    assert [(fill.quantity, fill.price) for fill in fills] == [(10, 9_820), (10, 9_810)]


def test_vwap_proxy_adds_half_tick_adverse(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_100, adtv20=1e9, ret_vol60=0.02),
            _mrow(
                DAY1, "KRX:A", high=10_200, low=9_900, close=10_100,
                upper_limit=12_000, lower_limit=8_000,
            ),
        ],
    )
    config = ExecutionConfig(
        scenario=ExecutionScenario.VWAP_PROXY, max_participation=1.0, carry_unfilled=False
    )
    buy = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=0)
    sell = Order(instrument_idx=0, side=Side.SELL, quantity=10, decision_session_idx=0)
    fills, rejects = price_orders(
        orders=(buy, sell), arrays=arrays, t=1, config=config, costs=_costs(impact_k=0.0),
        rules=_rules(),
    )
    assert rejects == ()
    assert [(fill.quantity, fill.price) for fill in fills] == [(10, 10_100), (10, 10_000)]


def test_close_locked_ranges_reject(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=1e9, ret_vol60=0.02),
            _mrow(
                DAY1, "KRX:A", open=10_000, high=10_500, low=11_000, close=10_800,
                upper_limit=11_000, lower_limit=9_000,
            ),
        ],
    )
    config = ExecutionConfig(
        scenario=ExecutionScenario.CLOSE_AUCTION, max_participation=1.0, carry_unfilled=False
    )
    buy = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=0)
    _, rejects = price_orders(
        orders=(buy,), arrays=arrays, t=1, config=config, costs=_costs(impact_k=0.0),
        rules=_rules(),
    )
    assert [reject.reason for reject in rejects] == ["limit_locked"]

    arrays = _arrays(
        tmp_path,
        "market_panel_test2",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=1e9, ret_vol60=0.02),
            _mrow(
                DAY1, "KRX:A", open=10_000, high=9_000, low=8_500, close=8_800,
                upper_limit=11_000, lower_limit=9_000,
            ),
        ],
    )
    sell = Order(instrument_idx=0, side=Side.SELL, quantity=10, decision_session_idx=0)
    _, rejects = price_orders(
        orders=(sell,), arrays=arrays, t=1, config=config, costs=_costs(impact_k=0.0),
        rules=_rules(),
    )
    assert [reject.reason for reject in rejects] == ["limit_locked"]


def test_zero_base_price_rejects(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=1e9, ret_vol60=0.02),
            _mrow(DAY1, "KRX:A", open=0),
        ],
    )
    order = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=0)
    fills, rejects = price_orders(
        orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(), rules=_rules()
    )
    assert fills == ()
    assert [reject.reason for reject in rejects] == ["missing_price"]


def test_unknown_market_code_rejected(tmp_path: Path) -> None:
    arrays = _arrays(
        tmp_path,
        "market_panel_test",
        [
            _mrow(DAY0, "KRX:A", close=10_000, adtv20=1e9, ret_vol60=0.02),
            _mrow(DAY1, "KRX:A", open=10_000, market="OTHER"),
        ],
    )
    order = Order(instrument_idx=0, side=Side.BUY, quantity=10, decision_session_idx=0)
    with pytest.raises(PITDataError):
        price_orders(
            orders=(order,), arrays=arrays, t=1, config=_open_config(), costs=_costs(),
            rules=_rules(),
        )
