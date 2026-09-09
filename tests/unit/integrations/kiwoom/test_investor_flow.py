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


def test_kiwoom_investor_flow_persist_raw_page_writes_investor_flow_receipt(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.core.pit import EvidenceKind
    from src.integrations.kiwoom.investor_flow import KiwoomInvestorFlowCollector

    collector = KiwoomInvestorFlowCollector(("005930",), client=object())
    receipt = collector._persist_raw_page(
        "005930",
        date(2026, 3, 6),
        ({"dt": "20260306", "ind_invsr": "1", "frgnr_invsr": "2", "orgn": "3"},),
        bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 3, 6, tzinfo=UTC),
    )

    assert receipt.kind is EvidenceKind.INVESTOR_FLOW
    assert len(receipt.content_hash) == 64
    assert receipt.payload_path.exists()
