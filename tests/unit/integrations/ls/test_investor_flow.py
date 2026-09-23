from datetime import date

import pytest

from src.core.pit import PITDataError
from src.integrations.ls.investor_flow import LsInvestorFlowCollector


def _balanced_row(**overrides):
    row = {
        "date": "20260306",
        "tjj0000": "-100",
        "tjj0001": "-50",
        "tjj0002": "-30",
        "tjj0003": "-20",
        "tjj0004": "-10",
        "tjj0005": "-10",
        "tjj0006": "-8",
        "tjj0007": "100",
        "tjj0008": "927",
        "tjj0009": "-800",
        "tjj0010": "-28",
        "tjj0011": "29",
        "tjj0016": "-828",
        "tjj0017": "129",
        "tjj0018": "-228",
    }
    row.update(overrides)
    return row


def test_ls_investor_flow_collector_maps_multi_year_sessions() -> None:
    class MockLsClient:
        def inquire_investor_trend(self, symbol: str, start_date: date, end_date: date, unit: str = "shares"):
            assert unit == "shares"
            return (_balanced_row(),)

    collector = LsInvestorFlowCollector(("005930",), client=MockLsClient())
    pages = list(collector.fetch_investor_flow(date(2026, 3, 1), date(2026, 3, 6)))
    assert len(pages) == 1
    records = pages[0]["records"]
    assert len(records) == 1
    row = records[0]
    assert row["session"] == "2026-03-06"
    assert row["ticker"] == "005930"
    assert row["_source_provider"] == "LS"
    assert row["individual_net_shares"] == 927
    assert row["foreign_net_shares"] == -828
    assert row["institution_net_shares"] == -228
    assert row["other_net_shares"] == 129


def test_map_rows_records_share_quantities_without_scaling() -> None:
    collector = LsInvestorFlowCollector(("005930",), client=object())
    (record,) = collector._map_rows("005930", (_balanced_row(),))
    assert record["individual_net_shares"] == 927
    assert record["foreign_net_shares"] == -828
    assert record["institution_net_shares"] == -228
    assert record["other_net_shares"] == 129
    assert record["unit"] == "shares"
    assert not any(key.endswith("_value") for key in record)


def test_map_rows_uses_foreign_total_tjj0016() -> None:
    collector = LsInvestorFlowCollector(("005930",), client=object())
    (record,) = collector._map_rows("005930", (_balanced_row(),))
    assert record["foreign_net_shares"] == -828


def test_map_rows_zero_sum_violation_fails_closed() -> None:
    collector = LsInvestorFlowCollector(("005930",), client=object())
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0008="928", tjj0000="-99", tjj0001="-51"),))


def test_map_rows_subgroup_identity_violation_fails_closed() -> None:
    collector = LsInvestorFlowCollector(("005930",), client=object())
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0000="-99"),))
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0009="-799"),))
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0007="101"),))


def test_map_rows_non_integral_quantity_rejected() -> None:
    collector = LsInvestorFlowCollector(("005930",), client=object())
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0008="1.5"),))
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0008=""),))


def test_map_rows_accepts_int_and_integral_float_quantities() -> None:
    collector = LsInvestorFlowCollector(("005930",), client=object())
    row = _balanced_row()
    int_row = {**row, "tjj0008": 927, "tjj0016": -828, "tjj0017": 129, "tjj0018": -228}
    (record,) = collector._map_rows("005930", (int_row,))
    assert record["individual_net_shares"] == 927
    float_row = {**row, "tjj0008": 927.0}
    (record,) = collector._map_rows("005930", (float_row,))
    assert record["individual_net_shares"] == 927


def test_map_rows_rejects_bool_float_and_non_numeric_quantities() -> None:
    collector = LsInvestorFlowCollector(("005930",), client=object())
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0008=True),))
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0008=927.5),))
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0008="abc"),))
    with pytest.raises(PITDataError):
        collector._map_rows("005930", (_balanced_row(tjj0008=None),))
