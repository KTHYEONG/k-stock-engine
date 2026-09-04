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
