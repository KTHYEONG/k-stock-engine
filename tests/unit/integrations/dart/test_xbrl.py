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


def test_dart_disclosure_batch_queries_per_company_corp_code_filter() -> None:
    from datetime import date

    from src.integrations.dart.xbrl import DartXbrlCollector

    calls: list[tuple[object, object, object]] = []

    class Client:
        def list_disclosures(self, start, end, *, corp_code=None):
            calls.append((start, end, corp_code))
            data = {
                "A": [{"rcept_no": "1", "rcept_dt": "20240102", "corp_code": "A"}],
                "B": [{"rcept_no": "2", "rcept_dt": "20240102", "corp_code": "B"}],
            }
            return data.get(corp_code, [])

    pages = tuple(
        DartXbrlCollector(client=Client()).fetch_disclosures(
            date(2024, 1, 1), date(2024, 1, 31), corp_codes=("A", "B")
        )
    )

    # Then: exactly one call per company, scoped to the full requested range (no month loop).
    assert calls == [
        (date(2024, 1, 1), date(2024, 1, 31), "A"),
        (date(2024, 1, 1), date(2024, 1, 31), "B"),
    ]
    assert len(pages) == 2
    assert pages[0]["records"] == [{"rcept_no": "1", "rcept_dt": "20240102", "corp_code": "A"}]
    assert pages[1]["records"] == [{"rcept_no": "2", "rcept_dt": "20240102", "corp_code": "B"}]
    assert pages[0]["corp_code"] == "A"
    assert pages[1]["corp_code"] == "B"


def test_dart_disclosure_batch_includes_empty_pages_without_raising() -> None:
    from datetime import date

    from src.integrations.dart.xbrl import DartXbrlCollector

    class Client:
        def list_disclosures(self, start, end, *, corp_code=None):
            if corp_code == "A":
                return [{"rcept_no": "1", "rcept_dt": "20240102", "corp_code": "A"}]
            return []

    pages = tuple(
        DartXbrlCollector(client=Client()).fetch_disclosures(
            date(2024, 1, 1), date(2024, 1, 31), corp_codes=("A", "B")
        )
    )

    assert len(pages) == 2
    assert pages[0]["records"] != []
    assert pages[1]["records"] == []
    assert pages[1]["corp_code"] == "B"

def test_dart_xbrl_collector_validates_max_workers() -> None:
    import pytest

    from src.integrations.dart.xbrl import DartXbrlCollector

    # When/Then: every malformed max_workers fails closed.
    with pytest.raises(ValueError, match="max_workers"):
        DartXbrlCollector(api_key="k", max_workers=0)
    with pytest.raises(ValueError, match="max_workers"):
        DartXbrlCollector(api_key="k", max_workers=-5)
    with pytest.raises(ValueError, match="max_workers"):
        DartXbrlCollector(api_key="k", max_workers=True)
    with pytest.raises(ValueError, match="max_workers"):
        DartXbrlCollector(api_key="k", max_workers=1.5)  # type: ignore[arg-type]

    # And: a valid value constructs and is retained.
    collector = DartXbrlCollector(api_key="k", max_workers=5)
    assert collector._max_workers == 5

    # And: the default is 20 when unspecified.
    default_collector = DartXbrlCollector(api_key="k")
    assert default_collector._max_workers == 20


def test_fetch_financial_fact_sources_preserves_input_order_under_concurrency() -> None:
    import time

    from src.integrations.dart.xbrl import DartXbrlCollector

    # Given: response latency deliberately inverted relative to input order.
    def variable_latency(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        idx = int(params["corp_code"])
        time.sleep(0.02 * (5 - idx))
        return {
            "status": "000",
            "list": [
                {
                    "rcept_no": "R",
                    "bsns_year": "2020",
                    "corp_code": params["corp_code"],
                    "reprt_code": "11011",
                    "account_id": "ifrs-full_Revenue",
                    "account_nm": "매출액",
                    "fs_div": "CFS",
                    "thstrm_amount": str(idx),
                }
            ],
        }

    identities = tuple(
        {"corp_code": str(i), "filing_id": f"F{i}", "biz_year": "2020", "reprt_code": "11011", "fs_div": "CFS"}
        for i in range(5)
    )
    collector = DartXbrlCollector(api_key="k", request_json=variable_latency, max_workers=5)

    # When
    pages = list(collector.fetch_financial_fact_sources(identities))

    # Then: output order matches input order, not completion order.
    values = [page["records"][0]["value"] for page in pages]
    assert values == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_fetch_financial_fact_sources_raises_on_first_failing_identity_in_order() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.dart.xbrl import DartXbrlCollector

    # Given: identity 2 (0-indexed) returns an unrecognized status.
    def failing_at_index_2(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        if params["corp_code"] == "2":
            return {"status": "999", "list": []}
        return {"status": "013", "list": []}

    def empty_bytes(_endpoint: str, _params: dict[str, str]) -> bytes:
        return b""

    identities = tuple(
        {"corp_code": str(i), "filing_id": f"F{i}", "biz_year": "2020", "reprt_code": "11011", "fs_div": "CFS"}
        for i in range(5)
    )
    collector = DartXbrlCollector(
        api_key="k", request_json=failing_at_index_2, request_bytes=empty_bytes, max_workers=5
    )

    # When/Then
    with pytest.raises(PITDataError, match="F2"):
        list(collector.fetch_financial_fact_sources(identities))


def test_fetch_financial_fact_sources_runs_identities_concurrently() -> None:
    import time

    from src.integrations.dart.xbrl import DartXbrlCollector

    # Given: a slow, empty-statement responder (falls through to an empty archive).
    def slow_empty(_endpoint: str, _params: dict[str, str]) -> dict[str, object]:
        time.sleep(0.05)
        return {"status": "013", "list": []}

    def empty_bytes(_endpoint: str, _params: dict[str, str]) -> bytes:
        return b""

    identities = tuple(
        {"corp_code": f"{i:08d}", "filing_id": f"F{i}", "biz_year": "2020", "reprt_code": "11011", "fs_div": "CFS"}
        for i in range(20)
    )
    collector = DartXbrlCollector(
        api_key="k", request_json=slow_empty, request_bytes=empty_bytes, max_workers=20
    )

    # When
    t0 = time.perf_counter()
    pages = list(collector.fetch_financial_fact_sources(identities))
    elapsed = time.perf_counter() - t0

    # Then: 20 x 0.05s would be 1.0s sequential; concurrent execution stays well under that.
    assert len(pages) == 20
    assert elapsed < 0.5


def test_fetch_financial_fact_sources_single_identity_skips_thread_pool(monkeypatch) -> None:
    import src.integrations.dart.xbrl as xbrl_module

    # Given: ThreadPoolExecutor construction is a hard failure for this test.
    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ThreadPoolExecutor must not be constructed for a single identity")

    monkeypatch.setattr(xbrl_module, "ThreadPoolExecutor", _forbidden)

    def ok_response(_endpoint: str, _params: dict[str, str]) -> dict[str, object]:
        return {
            "status": "000",
            "list": [
                {
                    "rcept_no": "R",
                    "bsns_year": "2020",
                    "corp_code": "00000001",
                    "reprt_code": "11011",
                    "account_id": "ifrs-full_Revenue",
                    "account_nm": "매출액",
                    "fs_div": "CFS",
                    "thstrm_amount": "100",
                }
            ],
        }

    collector = xbrl_module.DartXbrlCollector(api_key="k", request_json=ok_response, max_workers=20)
    identity = {"corp_code": "00000001", "filing_id": "F0", "biz_year": "2020", "reprt_code": "11011", "fs_div": "CFS"}

    # When/Then: no AssertionError means the thread pool was never touched.
    pages = list(collector.fetch_financial_fact_sources((identity,)))
    assert len(pages) == 1


def test_dart_xbrl_collector_forwards_quota_store_and_pacing_to_api_client(monkeypatch) -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    captured: dict[str, object] = {}

    class _FakeApiClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("src.integrations.dart.client.DartApiClient", _FakeApiClient)
    quota_store = object()
    now = object()

    collector = DartXbrlCollector(api_key="key", quota_store=quota_store, now=now, min_interval=2.0)

    assert captured["quota_store"] is quota_store
    assert captured["now"] is now
    assert captured["min_interval"] == 2.0
    assert collector._client is not None
