def test_collection_rejects_missing_investor_flow_without_partial_certification(tmp_path) -> None:
    from datetime import UTC, date, datetime
    from unittest.mock import Mock

    import pytest

    from src.data.collection import ChampionCollectionRequest, collect_champion_evidence
    from src.data.schemas import PITDataError

    krx = Mock()
    krx.fetch_daily_market.return_value = ({'records': [{'session': '2024-01-02'}]},)
    krx.fetch_investor_flow.return_value = ()
    krx.fetch_master_lineage.return_value = ({'records': [{'ticker': '000001'}]},)
    krx.fetch_corporate_actions.return_value = ({'records': [{'action_id': 'none'}]},)
    dart = Mock()
    dart.fetch_disclosures.return_value = ({'records': [{'filing_id': 'F1'}]},)
    dart.fetch_xbrl_facts.return_value = ({'records': [{'fact': 'sales'}]},)
    request = ChampionCollectionRequest(tmp_path / 'bronze', date(2024, 1, 2), date(2024, 1, 2), datetime(2024, 1, 3, tzinfo=UTC))

    with pytest.raises(PITDataError, match=r'investor.flow'):
        collect_champion_evidence(request, krx=krx, dart=dart)
    assert not (tmp_path / 'artifacts' / 'collections').exists()


def test_collection_artifact_keeps_each_page_receipt(tmp_path) -> None:
    from datetime import UTC, date, datetime

    from src.data.collection import ChampionCollectionRequest, collect_champion_evidence

    class KRX:
        def fetch_daily_market(self, start, end): return ({'records': [{'session': '2024-01-02'}]}, {'records': [{'session': '2024-01-03'}]})
        def fetch_investor_flow(self, start, end): return ({'records': [{'session': '2024-01-02'}]},)
        def fetch_master_lineage(self, start, end): return ({'records': [{'ticker': '000001'}]},)
        def fetch_corporate_actions(self, start, end): return ({'records': [{'action_id': 'A'}]},)

    class DART:
        def fetch_disclosures(self, start, end): return ({'records': [{'filing_id': 'F1'}]},)
        def fetch_xbrl_facts(self, filing_ids): return ({'records': [{'filing_id': 'F1', 'fact': 'sales'}]},)

    artifact = collect_champion_evidence(ChampionCollectionRequest(tmp_path / 'bronze', date(2024, 1, 2), date(2024, 1, 3), datetime(2024, 1, 4, tzinfo=UTC)), krx=KRX(), dart=DART())

    assert len(artifact.page_receipts['daily_market']) == 2
    assert artifact.report_path.exists()

def test_collect_dart_financial_facts_end_to_end_with_concurrent_collector(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.collection import collect_dart_financial_facts
    from src.integrations.dart.xbrl import DartXbrlCollector

    # Given: 6 distinct identities, each resolving on the first CFS attempt.
    def ok_response(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        return {
            "status": "000",
            "list": [
                {
                    "rcept_no": f"R{params['corp_code']}",
                    "bsns_year": "2020",
                    "corp_code": params["corp_code"],
                    "reprt_code": "11011",
                    "account_id": "ifrs-full_Revenue",
                    "account_nm": "매출액",
                    "fs_div": "CFS",
                    "thstrm_amount": "1000",
                }
            ],
        }

    collector = DartXbrlCollector(api_key="k", request_json=ok_response, max_workers=6)
    identities = tuple(
        {
            "corp_code": f"{i:08d}",
            "filing_id": f"F{i}",
            "rcept_no": f"R{i:08d}",
            "biz_year": "2020",
            "reprt_code": "11011",
            "fs_div": "CFS",
        }
        for i in range(6)
    )

    # When
    artifact = collect_dart_financial_facts(
        dart=collector,
        identities=identities,
        bronze_root=tmp_path,
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    # Then: every identity produced exactly one persisted page.
    assert len(artifact.page_receipts["financial_facts"]) == 6
