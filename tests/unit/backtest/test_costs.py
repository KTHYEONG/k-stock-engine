"""Cost model: regime sell tax, truncation, and square-root impact."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.backtest.costs import CostConfig, Side, fill_cost, impact_fraction

_CONFIG = CostConfig(commission_rate=Decimal("0.00015"), impact_k=1.0)


def test_sell_tax_by_regime() -> None:
    early = fill_cost(
        side=Side.SELL, quantity=100, price=10_000, sell_tax_rate=Decimal("0.0030"), config=_CONFIG
    )
    assert early.sell_tax == 3_000
    assert early.commission == 150
    late = fill_cost(
        side=Side.SELL, quantity=100, price=10_000, sell_tax_rate=Decimal("0.0015"), config=_CONFIG
    )
    assert late.sell_tax == 1_500


def test_buy_pays_no_tax() -> None:
    cost = fill_cost(
        side=Side.BUY, quantity=100, price=10_000, sell_tax_rate=Decimal("0.0030"), config=_CONFIG
    )
    assert cost.sell_tax == 0
    assert cost.commission == 150


def test_truncation_to_won() -> None:
    cost = fill_cost(
        side=Side.BUY, quantity=1, price=33_333, sell_tax_rate=Decimal("0"), config=_CONFIG
    )
    assert cost.commission == 4


def test_impact_grows_with_sqrt_participation() -> None:
    config = CostConfig(commission_rate=Decimal("0.00015"), impact_k=2.0)
    base = impact_fraction(notional=1_000_000.0, adtv20=100_000_000.0, vol60=0.04, config=config)
    assert base == pytest.approx(0.008)
    quadrupled = impact_fraction(
        notional=4_000_000.0, adtv20=100_000_000.0, vol60=0.04, config=config
    )
    assert quadrupled == pytest.approx(2.0 * base)


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        fill_cost(
            side="buy",  # type: ignore[arg-type]
            quantity=1,
            price=10_000,
            sell_tax_rate=Decimal("0"),
            config=_CONFIG,
        )


def test_invalid_impact_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        impact_fraction(notional=1_000.0, adtv20=1_000_000.0, vol60=float("nan"), config=_CONFIG)
    with pytest.raises(ValueError):
        impact_fraction(notional=1_000.0, adtv20=0.0, vol60=0.01, config=_CONFIG)
    with pytest.raises(ValueError):
        impact_fraction(notional=-1.0, adtv20=1_000_000.0, vol60=0.01, config=_CONFIG)
