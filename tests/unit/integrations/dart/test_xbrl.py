from src.integrations.dart.xbrl import DartXbrlCollector


def test_dart_falls_back_to_separate_statements_after_empty_consolidated_response() -> None:
    calls: list[str] = []

    def request_json(_endpoint, params):
        calls.append(params["fs_div"])
        if params["fs_div"] == "CFS":
            return {"status": "014"}
        return {"status": "000", "list": [{"account_nm": "매출액"}]}

    pages = tuple(
        DartXbrlCollector(api_key="test-key", request_json=request_json).fetch_xbrl_facts(
            (
                {
                    "corp_code": "001",
                    "filing_id": "F1",
                    "biz_year": "2016",
                    "reprt_code": "11011",
                    "fs_div": "CFS",
                },
            )
        )
    )

    assert calls == ["CFS", "OFS"]
    assert pages[0]["fs_div"] == "OFS"


def test_dart_disclosure_batch_uses_monthly_global_pages_not_per_corp_calls() -> None:
    from datetime import date
    from src.integrations.dart.xbrl import DartXbrlCollector

    calls: list[object] = []
    class Client:
        def list_disclosures(self, start, end, *, corp_code=None):
            calls.append(corp_code)
            return [{'rcept_no': '1', 'rcept_dt': '20240102', 'corp_code': 'A'}, {'rcept_no': '2', 'rcept_dt': '20240102', 'corp_code': 'B'}, {'rcept_no': '3', 'rcept_dt': '20240102', 'corp_code': 'Z'}]
    pages = tuple(DartXbrlCollector(client=Client()).fetch_disclosures(date(2024, 1, 1), date(2024, 1, 31), corp_codes=('A', 'B')))
    assert calls == [None]
    assert {row['corp_code'] for row in pages[0]['records']} == {'A', 'B'}
