from datetime import date

from src.integrations.kis.investor_flow import KisInvestorFlowCollector


def test_kis_collection_can_limit_a_call_to_one_planned_symbol() -> None:
    class Client:
        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            return (
                {
                    "stck_bsop_date": anchor.strftime("%Y%m%d"),
                    "frgn_shnu_tr_pbmn": "1",
                    "frgn_seln_tr_pbmn": "0",
                    "frgn_ntby_tr_pbmn": "1",
                    "orgn_ntby_tr_pbmn": "0",
                    "prsn_ntby_tr_pbmn": "-1",
                },
            )

    pages = tuple(
        KisInvestorFlowCollector(("000001", "000002"), client=Client()).fetch_investor_flow(
            date(2024, 1, 2), date(2024, 1, 2), symbols=("000002",)
        )
    )

    assert pages[0]["symbol"] == "000002"


def _single_row_client(calls=None, *, session_values=None):
    class Client:
        def __init__(self) -> None:
            self.calls: list = [] if calls is None else calls

        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            self.calls.append((symbol, anchor))
            return (
                {
                    "stck_bsop_date": anchor.strftime("%Y%m%d"),
                    "frgn_shnu_tr_pbmn": "1",
                    "frgn_seln_tr_pbmn": "0",
                    "frgn_ntby_tr_pbmn": "1",
                    "orgn_ntby_tr_pbmn": "0",
                    "prsn_ntby_tr_pbmn": "-1",
                },
            )

    return Client()


def _counting_spy(monkeypatch):
    import src.data.bronze_aggregation as agg

    calls: list = []
    real = agg.discover_verified_bronze_receipts

    def spy(*, bronze_root, kinds=None):
        calls.append((bronze_root, kinds))
        return real(bronze_root=bronze_root, kinds=kinds)

    monkeypatch.setattr(agg, "discover_verified_bronze_receipts", spy)
    return calls


def test_kis_verified_anchor_lookup_scans_once_across_anchors(tmp_path, monkeypatch) -> None:
    from datetime import date
    from src.core.pit import EvidenceKind
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    spy_calls = _counting_spy(monkeypatch)
    collector = KisInvestorFlowCollector(("000001",), client=_single_row_client())
    pages = tuple(
        collector.fetch_investor_flow(date(2024, 1, 1), date(2024, 1, 3), bronze_root=tmp_path / "bronze")
    )

    assert len(pages) == 3
    assert len(spy_calls) <= 1
    assert spy_calls[0][1] == frozenset({EvidenceKind.INVESTOR_FLOW})


def test_kis_verified_anchor_index_reused_across_repeated_calls(tmp_path, monkeypatch) -> None:
    from datetime import date
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    spy_calls = _counting_spy(monkeypatch)
    collector = KisInvestorFlowCollector(("000001",), client=_single_row_client())
    tuple(collector.fetch_investor_flow(date(2024, 1, 1), date(2024, 1, 3), bronze_root=tmp_path / "bronze"))
    first_count = len(spy_calls)
    assert first_count == 1
    tuple(collector.fetch_investor_flow(date(2024, 1, 1), date(2024, 1, 3), bronze_root=tmp_path / "bronze"))

    assert len(spy_calls) == 1


def test_kis_verified_anchor_index_rebuilt_once_per_bronze_root(tmp_path, monkeypatch) -> None:
    from datetime import date
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    spy_calls = _counting_spy(monkeypatch)
    collector = KisInvestorFlowCollector(("000001",), client=_single_row_client())
    tuple(collector.fetch_investor_flow(date(2024, 1, 2), date(2024, 1, 2), bronze_root=tmp_path / "b1"))
    tuple(collector.fetch_investor_flow(date(2024, 1, 2), date(2024, 1, 2), bronze_root=tmp_path / "b2"))
    tuple(collector.fetch_investor_flow(date(2024, 1, 2), date(2024, 1, 2), bronze_root=tmp_path / "b1"))

    assert len(spy_calls) == 2


def test_kis_reused_page_content_matches_live_fetch_without_new_write(tmp_path) -> None:
    import json
    from datetime import date
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    bronze_root = tmp_path / "bronze"
    first = tuple(
        KisInvestorFlowCollector(("000001",), client=_single_row_client()).fetch_investor_flow(
            date(2024, 1, 2), date(2024, 1, 2), bronze_root=bronze_root
        )
    )
    before = sorted((bronze_root / "investor_flow").rglob("payload.json"))

    class ExplodingClient:
        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            raise AssertionError("live fetch must not be triggered for a verified anchor")

    second = tuple(
        KisInvestorFlowCollector(("000001",), client=ExplodingClient()).fetch_investor_flow(
            date(2024, 1, 2), date(2024, 1, 2), bronze_root=bronze_root
        )
    )
    after = sorted((bronze_root / "investor_flow").rglob("payload.json"))

    assert [row for page in second for row in page["records"]] == [
        row for page in first for row in page["records"]
    ]
    assert json.dumps(second[0]["records"], sort_keys=True) == json.dumps(first[0]["records"], sort_keys=True)
    assert after == before


def test_kis_empty_records_page_still_triggers_live_fetch(tmp_path) -> None:
    import json
    from datetime import UTC, date, datetime
    from src.data.bronze import BronzeStore
    from src.data.schemas import EvidenceKind
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    bronze_root = tmp_path / "bronze"
    store = BronzeStore(bronze_root)
    payload = {
        "provider": "KIS",
        "endpoint": "investor-trade-by-stock-daily",
        "symbol": "000001",
        "anchor": "2024-01-02",
        "records": [],
    }
    store.import_bytes(
        json.dumps(payload, sort_keys=True).encode("utf-8"),
        kind=EvidenceKind.INVESTOR_FLOW,
        retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
        source_label="KIS:investor-trade-by-stock-daily:000001:2024-01-02",
    )

    client = _single_row_client()
    pages = tuple(
        KisInvestorFlowCollector(("000001",), client=client).fetch_investor_flow(
            date(2024, 1, 2), date(2024, 1, 2), bronze_root=bronze_root
        )
    )

    assert client.calls != []
    assert pages[0]["records"] != []
