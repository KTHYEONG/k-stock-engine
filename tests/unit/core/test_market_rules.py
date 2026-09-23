"""KRX market-rules contract tests."""

from __future__ import annotations

import copy
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.core.market_rules import KrxMarket, load_krx_market_rules, parse_krx_market_rules
from src.core.pit import PITDataError

RULES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "market" / "krx_market_rules.toml"


@pytest.fixture
def rules():
    return load_krx_market_rules(RULES_PATH)


@pytest.fixture
def document():
    import tomllib

    with open(RULES_PATH, "rb") as handle:
        return copy.deepcopy(tomllib.load(handle))


def test_loads_repository_rules_file_with_seven_tax_regimes(rules):
    assert rules.version == "krx-market-rules-v1"
    assert len(rules.sell_tax_regimes) == 7
    dates = [r.effective_from for r in rules.sell_tax_regimes]
    assert dates == sorted(dates)


def test_sell_tax_rate_follows_effective_date_boundary(rules):
    assert rules.sell_tax_rate(session=date(2019, 5, 31), market=KrxMarket.KOSPI) == Decimal("0.0030")
    assert rules.sell_tax_rate(session=date(2019, 6, 3), market=KrxMarket.KOSPI) == Decimal("0.0025")


def test_sell_tax_rate_2025_and_2026_regimes(rules):
    assert rules.sell_tax_rate(session=date(2025, 12, 30), market=KrxMarket.KOSDAQ) == Decimal("0.0015")
    assert rules.sell_tax_rate(session=date(2026, 1, 2), market=KrxMarket.KOSDAQ) == Decimal("0.0020")


def test_pre_coverage_lookups_fail_closed(rules):
    with pytest.raises(PITDataError):
        rules.sell_tax_rate(session=date(2015, 12, 30), market=KrxMarket.KOSPI)
    with pytest.raises(PITDataError):
        rules.tick_size(session=date(2015, 12, 30), market=KrxMarket.KOSPI, price=1000)
    with pytest.raises(PITDataError):
        rules.price_limits(session=date(2015, 6, 12), market=KrxMarket.KOSPI, base_price=10000)


def test_tick_reform_switches_on_unified_schedule_date(rules):
    assert rules.tick_size(session=date(2023, 1, 24), market=KrxMarket.KOSPI, price=1500) == 5
    assert rules.tick_size(session=date(2023, 1, 25), market=KrxMarket.KOSPI, price=1500) == 1


def test_kosdaq_legacy_top_band_and_unified_band(rules):
    assert rules.tick_size(session=date(2022, 6, 2), market=KrxMarket.KOSDAQ, price=600000) == 100
    assert rules.tick_size(session=date(2023, 2, 1), market=KrxMarket.KOSDAQ, price=600000) == 1000


def test_band_boundary_is_inclusive_lower(rules):
    assert rules.tick_size(session=date(2020, 1, 2), market=KrxMarket.KOSPI, price=4999) == 5
    assert rules.tick_size(session=date(2020, 1, 2), market=KrxMarket.KOSPI, price=5000) == 10


def test_price_limits_around_base(rules):
    assert rules.price_limits(session=date(2020, 1, 2), market=KrxMarket.KOSPI, base_price=10000) == (13000, 7000)


def test_price_limits_snap_to_bound_tick(rules):
    upper, lower = rules.price_limits(session=date(2022, 1, 3), market=KrxMarket.KOSPI, base_price=4000)
    assert (upper, lower) == (5200, 2800)
    assert lower < 4000 < upper
    assert upper % 10 == 0
    assert lower % 5 == 0


def test_non_positive_price_rejected(rules):
    with pytest.raises(PITDataError):
        rules.tick_size(session=date(2020, 1, 2), market=KrxMarket.KOSPI, price=0)
    with pytest.raises(PITDataError):
        rules.price_limits(session=date(2020, 1, 2), market=KrxMarket.KOSPI, base_price=0)


def test_rates_keep_decimal_precision(rules):
    rate = rules.sell_tax_rate(session=date(2019, 6, 3), market=KrxMarket.KOSPI)
    assert isinstance(rate, Decimal)
    assert rate == Decimal("0.0025")
    assert str(rate) == "0.0025"


def test_regime_accessors_resolve_by_session(rules):
    assert rules.tick_regime_at(date(2023, 1, 25)).effective_from == date(2023, 1, 25)
    assert rules.sell_tax_regime_at(date(2026, 6, 1)).effective_from == date(2026, 1, 1)
    assert rules.price_limit_regime_at(date(2020, 1, 2)).effective_from == date(2015, 6, 15)


def test_alternate_document_forms_accepted(document):
    document["tick_regimes"][0]["KOSPI"] = [
        [0, 1],
        [1000, 5],
        [5000, 10],
        [10000, 50],
        [50000, 100],
        [100000, 500],
        [500000, 1000],
    ]
    document["tick_regimes"][0]["KOSDAQ"] = [
        {"lower_price_inclusive": 0, "tick": 1},
        {"lower_price_inclusive": 1000, "tick": 5},
        {"lower_price_inclusive": 5000, "tick": 10},
        {"lower_price_inclusive": 10000, "tick": 50},
        {"lower_price_inclusive": 50000, "tick": 100},
    ]
    document["tick_regimes"][0]["effective_from"] = datetime(2016, 1, 1, 12, 0)
    document["sell_tax_regimes"][0]["KOSPI"] = 0
    parsed = parse_krx_market_rules(document)
    assert parsed.tick_size(session=date(2020, 1, 2), market=KrxMarket.KOSPI, price=500) == 1
    assert parsed.sell_tax_regime_at(date(2016, 6, 1)).rates[KrxMarket.KOSPI] == Decimal(0)


def test_malformed_documents_rejected(document):
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0]["KOSPI"] = [{"lower": 100, "tick": 1}]
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)

    bad = copy.deepcopy(document)
    bad["tick_regimes"] = sorted(bad["tick_regimes"], key=lambda r: r["effective_from"], reverse=True)
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)

    bad = copy.deepcopy(document)
    del bad["tick_regimes"][0]["KOSDAQ"]
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)

    bad = copy.deepcopy(document)
    bad["sell_tax_regimes"][0]["KOSPI"] = "0.5"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)


def test_malformed_shapes_rejected(document):
    with pytest.raises(PITDataError):
        parse_krx_market_rules([1, 2, 3])  # type: ignore[arg-type]
    bad = copy.deepcopy(document)
    bad["extra"] = 1
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["version"] = "other-v1"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["sources"] = []
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"] = []
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0] = "nope"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0]["KOSPI"] = [{"lower": 0, "tick": 0}]
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0]["KOSPI"] = [{"lower": 0, "tick": 1}, {"lower": 0, "tick": 5}]
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0]["KOSPI"] = [{"lower": 0, "tick": 1, "extra": 2}]
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0]["KOSPI"] = ["nope"]
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0]["KOSPI"] = [{"lower": -1, "tick": 1}]
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0]["KOSPI"] = []
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0]["effective_from"] = "not-a-date"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["tick_regimes"][0]["effective_from"] = 123
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)


def test_malformed_tax_and_limit_shapes_rejected(document):
    bad = copy.deepcopy(document)
    bad["sell_tax_regimes"] = []
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["sell_tax_regimes"][0] = "nope"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["sell_tax_regimes"][0]["KOSPI"] = 0.0025
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["sell_tax_regimes"][0]["KOSPI"] = True
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["sell_tax_regimes"][0]["KOSPI"] = "abc"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["sell_tax_regimes"][0]["KOSPI"] = "-0.001"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["sell_tax_regimes"].append(copy.deepcopy(bad["sell_tax_regimes"][-1]))
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["sell_tax_regimes"][0]["EXTRA"] = "0.001"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)

    bad = copy.deepcopy(document)
    bad["price_limit_regimes"] = []
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["price_limit_regimes"][0] = "nope"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["price_limit_regimes"][0]["ratio"] = "1.5"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["price_limit_regimes"][0]["ratio"] = "0"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["price_limit_regimes"][0]["extra"] = "0.3"
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["price_limit_regimes"].append(copy.deepcopy(bad["price_limit_regimes"][-1]))
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
    bad = copy.deepcopy(document)
    bad["price_limit_regimes"][0]["ratio"] = 0.3
    with pytest.raises(PITDataError):
        parse_krx_market_rules(bad)
