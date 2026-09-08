from datetime import date

from src.integrations.kiwoom.investor_flow import KiwoomInvestorFlowCollector


def test_kiwoom_investor_flow_collector_maps_sessions() -> None:
    class MockKiwoomClient:
        def inquire_investor_trend(self, symbol: str, target_date: date, unit: str = "amount"):
            return (
                {
                    "dt": "20260306",
                    "ind_invsr": "8764447",
                    "frgnr_invsr": "-6254129",
                    "orgn": "-3014565",
                },
            )

    collector = KiwoomInvestorFlowCollector(("005930",), client=MockKiwoomClient())
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
