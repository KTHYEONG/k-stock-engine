from datetime import date

from src.integrations.ls.investor_flow import LsInvestorFlowCollector


def test_ls_investor_flow_collector_maps_multi_year_sessions() -> None:
    class MockLsClient:
        def inquire_investor_trend(self, symbol: str, start_date: date, end_date: date, unit: str = "amount"):
            return (
                {
                    "date": "20260306",
                    "tjj0008": "8764447",   # retail net (million KRW)
                    "tjj0009": "-6254129",  # foreign net (million KRW)
                    "tjj0018": "-3014565",  # institution net (million KRW)
                },
            )

    collector = LsInvestorFlowCollector(("005930",), client=MockLsClient())
    pages = list(collector.fetch_investor_flow(date(2026, 3, 1), date(2026, 3, 6)))
    assert len(pages) == 1
    records = pages[0]["records"]
    assert len(records) == 1
    row = records[0]
    assert row["session"] == "2026-03-06"
    assert row["ticker"] == "005930"
    assert row["retail_net_value"] == 8764447000000.0
    assert row["foreign_net_value"] == -6254129000000.0
    assert row["institution_net_value"] == -3014565000000.0
