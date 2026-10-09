from src.integrations.dart.xbrl import DartXbrlCollector


def test_dart_falls_back_to_separate_statements_after_empty_consolidated_response() -> None:
    calls: list[str] = []

    def request_json(_endpoint, params):
        calls.append(params["fs_div"])
        if params["fs_div"] == "CFS":
            return {"status": "014"}
        return {"status": "000", "list": [{"account_nm": "매출액"}]}

    pages = tuple(
        DartXbrlCollector(api_key="test-key", min_interval=0.0, max_workers=1, request_json=request_json).fetch_xbrl_facts(
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
        DartXbrlCollector(api_key="k", min_interval=0.0, max_workers=1, client=Client()).fetch_disclosures(
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
        DartXbrlCollector(api_key="k", min_interval=0.0, max_workers=1, client=Client()).fetch_disclosures(
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
        DartXbrlCollector(min_interval=0.0, api_key="k", max_workers=0)
    with pytest.raises(ValueError, match="max_workers"):
        DartXbrlCollector(min_interval=0.0, api_key="k", max_workers=-5)
    with pytest.raises(ValueError, match="max_workers"):
        DartXbrlCollector(min_interval=0.0, api_key="k", max_workers=True)
    with pytest.raises(ValueError, match="max_workers"):
        DartXbrlCollector(api_key="k", min_interval=0.0, max_workers=1.5)  # type: ignore[arg-type]

    # And: a valid value constructs and is retained.
    collector = DartXbrlCollector(min_interval=0.0, api_key="k", max_workers=5)
    assert collector._max_workers == 5

    # And: pacing and workers have no defaults or env fallback; they are required.
    with pytest.raises(TypeError):
        DartXbrlCollector(api_key="k")  # type: ignore[call-arg]


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
    collector = DartXbrlCollector(min_interval=0.0, api_key="k", request_json=variable_latency, max_workers=5)

    # When
    pages = list(collector.fetch_financial_fact_sources(identities))

    # Then: output order matches input order, not completion order.
    values = [page["records"][0]["value"] for page in pages]
    assert values == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_fetch_financial_fact_sources_isolates_failing_identity_without_raising() -> None:
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
        api_key="k", request_json=failing_at_index_2, request_bytes=empty_bytes, max_workers=5, min_interval=0.0
    )

    # When: per-identity failures isolate instead of aborting the batch.
    pages = list(collector.fetch_financial_fact_sources(identities))

    # Then
    assert len(pages) == 5
    assert pages[2]["source_kind"] == "unavailable"
    assert any(str(entry).startswith("dart_error:") for entry in pages[2]["diagnostics"])


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
        api_key="k", request_json=slow_empty, request_bytes=empty_bytes, max_workers=20, min_interval=0.0
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

    collector = xbrl_module.DartXbrlCollector(min_interval=0.0, api_key="k", request_json=ok_response, max_workers=20)
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

    collector = DartXbrlCollector(
        api_key="key",
        quota_store=quota_store,
        now=now,
        min_interval=2.0,
        max_workers=2,
        daily_request_limit=16000,
    )

    assert captured["quota_store"] is quota_store
    assert captured["now"] is now
    assert captured["min_interval"] == 2.0
    assert captured["daily_request_limit"] == 16000
    assert collector._client is not None


def _fact_identity(corp_code: str, filing_id: str) -> dict[str, str]:
    return {
        "corp_code": corp_code,
        "filing_id": filing_id,
        "rcept_no": filing_id,
        "biz_year": "2020",
        "reprt_code": "11011",
        "fs_div": "CFS",
        "published_at": "",
        "ticker": "",
    }


def _fact_success_payload(corp_code: str) -> dict[str, object]:
    return {
        "status": "000",
        "list": [
            {
                "rcept_no": "R",
                "bsns_year": "2020",
                "corp_code": corp_code,
                "reprt_code": "11011",
                "account_id": "ifrs-full_Revenue",
                "account_nm": "매출액",
                "fs_div": "CFS",
                "thstrm_amount": "100",
            }
        ],
    }


def test_fetch_one_financial_fact_source_does_not_retry_retryable_status(monkeypatch) -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import ProviderRetryableError

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    calls: list[str] = []

    class _FlakyClient:
        def _request_validated(self, _endpoint: str, params: dict[str, str]) -> dict[str, object]:
            calls.append(params["fs_div"])
            raise ProviderRetryableError("DART status 900: transient")

    collector = xbrl_module.DartXbrlCollector(max_workers=1, min_interval=0.0, client=_FlakyClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then: a single call per fs_div; retries live in the client, never the collector.
    assert page["source_kind"] == "unavailable"
    assert calls == ["CFS"]


def test_fetch_one_financial_fact_source_isolates_exhausted_retryable_status(monkeypatch) -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import ProviderRetryableError

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    calls: list[str] = []

    class _AlwaysRetryableClient:
        def _request_validated(self, _endpoint: str, params: dict[str, str]) -> dict[str, object]:
            calls.append(params["fs_div"])
            raise ProviderRetryableError("DART status 900: transient")

    collector = xbrl_module.DartXbrlCollector(max_workers=1, min_interval=0.0, client=_AlwaysRetryableClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then: exhausted budget isolates as unavailable, never raises, never blocked.
    assert page["source_kind"] == "unavailable"
    assert page["source_kind"] != "blocked"
    assert any(str(entry).startswith("dart_error:") for entry in page["diagnostics"])
    assert calls == ["CFS"]


def test_fetch_one_financial_fact_source_isolates_quota_exhaustion_as_blocked() -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import ProviderQuotaExhaustedError

    archive_calls: list[str] = []

    class _QuotaBlockedClient:
        def _request_validated(self, _endpoint: str, _params: dict[str, str]) -> dict[str, object]:
            raise ProviderQuotaExhaustedError("DART status 020: quota exceeded")

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            archive_calls.append(rcept_no)
            raise AssertionError("document archive must not be fetched after quota exhaustion")

    collector = xbrl_module.DartXbrlCollector(max_workers=1, min_interval=0.0, client=_QuotaBlockedClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then: blocked record without exhausting the fallback chain.
    assert page["source_kind"] == "blocked"
    assert "dart_quota_exhausted" in page["diagnostics"]
    assert archive_calls == []


def test_fetch_one_financial_fact_source_isolates_unexpected_terminal_status() -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import ProviderTerminalError

    class _TerminalClient:
        def _request_validated(self, _endpoint: str, _params: dict[str, str]) -> dict[str, object]:
            raise ProviderTerminalError("DART status 101: unknown status")

    collector = xbrl_module.DartXbrlCollector(max_workers=1, min_interval=0.0, client=_TerminalClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then: original error text is preserved in diagnostics, nothing raised.
    assert page["source_kind"] == "unavailable"
    assert any("101" in str(entry) for entry in page["diagnostics"])


def test_fetch_financial_fact_sources_stops_early_on_first_blocked_identity() -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import ProviderQuotaExhaustedError

    requested: list[str] = []

    class _BatchClient:
        def _request_validated(self, _endpoint: str, params: dict[str, str]) -> dict[str, object]:
            requested.append(params["corp_code"])
            if params["corp_code"] == "00000003":
                raise ProviderQuotaExhaustedError("DART status 020: quota exceeded")
            return _fact_success_payload(params["corp_code"])

    collector = xbrl_module.DartXbrlCollector(min_interval=0.0, client=_BatchClient(), api_key="k", max_workers=1)
    identities = tuple(
        _fact_identity(f"{i:08d}", f"F{i}") for i in range(1, 5)
    )

    # When
    pages = list(collector.fetch_financial_fact_sources(identities))

    # Then: two successes plus the blocked record; the 4th identity never requested.
    assert len(pages) == 3
    assert [page["source_kind"] for page in pages] == ["opendart_standard", "opendart_standard", "blocked"]
    assert requested == ["00000001", "00000002", "00000003"]


def test_fetch_financial_fact_sources_completes_when_no_identity_blocked() -> None:
    import src.integrations.dart.xbrl as xbrl_module

    class _OkClient:
        def _request_validated(self, _endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return _fact_success_payload(params["corp_code"])

    collector = xbrl_module.DartXbrlCollector(min_interval=0.0, client=_OkClient(), api_key="k", max_workers=1)
    identities = tuple(_fact_identity(f"{i:08d}", f"F{i}") for i in range(1, 4))

    # When
    pages = list(collector.fetch_financial_fact_sources(identities))

    # Then
    assert len(pages) == 3
    assert all(page["source_kind"] == "opendart_standard" for page in pages)


def test_fetch_one_financial_fact_source_isolates_terminal_error_without_status_code() -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import ProviderTerminalError

    class _NoStatusClient:
        def _request_validated(self, _endpoint: str, _params: dict[str, str]) -> dict[str, object]:
            raise ProviderTerminalError("connection reset by peer")

    collector = xbrl_module.DartXbrlCollector(max_workers=1, min_interval=0.0, client=_NoStatusClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then: status falls back, original text preserved.
    assert page["source_kind"] == "unavailable"
    assert page["status"] == "013"
    assert any("connection reset" in str(entry) for entry in page["diagnostics"])


def test_fetch_one_financial_fact_source_isolates_generic_transport_error() -> None:
    import src.integrations.dart.xbrl as xbrl_module

    class _BoomClient:
        def _request_validated(self, _endpoint: str, _params: dict[str, str]) -> dict[str, object]:
            raise RuntimeError("boom")

    collector = xbrl_module.DartXbrlCollector(max_workers=1, min_interval=0.0, client=_BoomClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then
    assert page["source_kind"] == "unavailable"
    assert any(str(entry) == "dart_error:boom" for entry in page["diagnostics"])


def test_fetch_one_financial_fact_source_isolates_raw_retryable_status() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    def raw_retryable(_endpoint: str, _params: dict[str, str]) -> dict[str, object]:
        return {"status": "900", "list": []}

    def empty_bytes(_endpoint: str, _params: dict[str, str]) -> bytes:
        return b""

    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", request_json=raw_retryable, request_bytes=empty_bytes)

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then
    assert page["source_kind"] == "unavailable"
    assert page["status"] == "900"
    assert any(str(entry).startswith("dart_error:") for entry in page["diagnostics"])


def test_fetch_one_financial_fact_source_isolates_raw_quota_status_as_blocked() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    archive_calls: list[str] = []

    def raw_blocked(_endpoint: str, _params: dict[str, str]) -> dict[str, object]:
        return {"status": "020", "message": "quota exceeded"}

    def forbidden_bytes(_endpoint: str, _params: dict[str, str]) -> bytes:
        archive_calls.append(_endpoint)
        raise AssertionError("document archive must not be fetched after quota exhaustion")

    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", request_json=raw_blocked, request_bytes=forbidden_bytes)

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then
    assert page["source_kind"] == "blocked"
    assert page["status"] == "020"
    assert "dart_quota_exhausted" in page["diagnostics"]
    assert archive_calls == []


def test_fetch_one_financial_fact_source_isolates_malformed_response() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    def malformed(_endpoint: str, _params: dict[str, str]) -> dict[str, object]:
        return {}

    def empty_bytes(_endpoint: str, _params: dict[str, str]) -> bytes:
        return b""

    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", request_json=malformed, request_bytes=empty_bytes)

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then
    assert page["source_kind"] == "unavailable"
    assert any(str(entry).startswith("dart_error:") for entry in page["diagnostics"])


def test_filing_identities_from_bronze_maps_report_kind_to_fiscal_period(tmp_path) -> None:
    from datetime import date

    from src.data.dart_disclosures import DisclosureRecord, periodic_filing_identities

    records = [
        DisclosureRecord(corp_code="00126380", rcept_no="20160516000001", rcept_dt=date(2016, 5, 16), report_nm="분기보고서 (2016.03)"),
        DisclosureRecord(corp_code="00126380", rcept_no="20160816000002", rcept_dt=date(2016, 8, 16), report_nm="반기보고서 (2016.06)"),
        DisclosureRecord(corp_code="00126380", rcept_no="20161115000003", rcept_dt=date(2016, 11, 15), report_nm="분기보고서 (2016.09)"),
        DisclosureRecord(corp_code="00126380", rcept_no="20170331000004", rcept_dt=date(2017, 3, 31), report_nm="사업보고서 (2016.12)"),
        DisclosureRecord(corp_code="00126380", rcept_no="20170331000005", rcept_dt=date(2017, 3, 31), report_nm="감사보고서 (2016.12)"),
    ]

    identities = periodic_filing_identities(
        records, start=date(2016, 1, 1), end=date(2017, 12, 31),
        ticker_by_corp_code={"00126380": "005930"}, required_periods=None, corp_codes=None,
    )

    assert {(i["reprt_code"], i["fiscal_period"]) for i in identities} == {
        ("11013", "2016Q1"),
        ("11012", "2016Q2"),
        ("11014", "2016Q3"),
        ("11011", "2016Q4"),
    }


def test_dart_collector_requires_api_key_without_a_seam() -> None:
    import pytest

    from src.integrations.dart.xbrl import DartXbrlCollector

    with pytest.raises(ValueError, match="api_key is required"):
        DartXbrlCollector(api_key="", min_interval=0.0, max_workers=1)


def test_fetch_corp_code_records_falls_back_to_a_policy_paced_client(monkeypatch) -> None:
    from src.integrations.dart.client import DartCorpCodeRecord
    from src.integrations.dart.xbrl import DartXbrlCollector

    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def load_corp_code_records(self) -> tuple[DartCorpCodeRecord, ...]:
            return (DartCorpCodeRecord(ticker="005930", corp_code="00126380", corp_name="A"),)

    monkeypatch.setattr("src.integrations.dart.client.DartApiClient", _Client)
    collector = DartXbrlCollector(api_key="k", min_interval=0.5, max_workers=1)
    collector._client = None

    assert collector.fetch_corp_code_records()[0].ticker == "005930"
    assert captured["min_interval"] == 0.5


def test_transport_failure_page_detects_unavailable_markers() -> None:
    from src.integrations.dart.xbrl import is_transport_failure_page

    assert is_transport_failure_page({"source_kind": "unavailable", "diagnostics": ["transport failed: timeout"]})
    assert not is_transport_failure_page({"source_kind": "unavailable", "diagnostics": ["DART status 013"]})
    assert not is_transport_failure_page({"source_kind": "opendart_standard"})

"""OpenDART standard quarterly financial facts."""

from pathlib import Path
from typing import Any


def test_opendart_standard_facts_parsed_with_values_and_fiscal_period() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    mock_raw = {
        "status": "000",
        "message": "정상",
        "list": [
            {
                "rcept_no": "20150515001111",
                "bsns_year": "2015",
                "corp_code": "00126380",
                "stock_code": "005930",
                "reprt_code": "11013",
                "account_id": "ifrs-full_Revenue",
                "account_nm": "매출액",
                "fs_div": "CFS",
                "thstrm_amount": "47,117,896,000,000",
            },
            {
                "rcept_no": "20150515001111",
                "bsns_year": "2015",
                "corp_code": "00126380",
                "stock_code": "005930",
                "reprt_code": "11013",
                "account_id": "ifrs-full_OperatingProfit",
                "account_nm": "영업이익",
                "fs_div": "CFS",
                "thstrm_amount": "5,979,343,000,000",
            },
            {
                "rcept_no": "20150515001111",
                "bsns_year": "2015",
                "corp_code": "00126380",
                "stock_code": "005930",
                "reprt_code": "11013",
                "account_id": "ifrs-full_Assets",
                "account_nm": "자산총계",
                "fs_div": "CFS",
                "thstrm_amount": "233,401,659,000,000",
            },
        ],
    }
    collector = DartXbrlCollector(
        api_key="fixture-key",
        min_interval=0.0,
        max_workers=1,
        request_json=lambda _endpoint, _params: mock_raw,
    )
    identity = {
        "corp_code": "00126380",
        "filing_id": "20150515001111",
        "biz_year": "2015",
        "reprt_code": "11013",
        "fs_div": "CFS",
        "published_at": "2015-05-15",
    }
    pages = list(collector.fetch_financial_fact_sources((identity,)))
    assert len(pages) == 1
    page = pages[0]
    assert page["source_kind"] == "opendart_standard"
    assert page["status"] == "000"
    records = page["records"]
    assert len(records) == 3
    sales_rec = next(r for r in records if r["fact"] == "sales")
    assert sales_rec["value"] == 47117896000000.0
    assert sales_rec["fiscal_period"] == "2015Q1"
    assert sales_rec["unit"] == "KRW"
    assert sales_rec["company_id"] == "00126380"
    assert sales_rec["consolidated"] is True


def test_normalize_dart_financial_facts_accepts_opendart_standard_records() -> None:
    from datetime import UTC, datetime
    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    page = {
        "source_kind": "opendart_standard",
        "status": "000",
        "mapping_version": "dart-fact-map-v1",
        "raw_document_hash": None,
        "corp_code": "00126380",
        "filing_id": "20150515001111",
        "fiscal_period": "2015Q1",
        "published_at": "2015-05-15T09:00:00+09:00",
        "records": [
            {
                "company_id": "00126380",
                "fiscal_period": "2015Q1",
                "filing_id": "20150515001111",
                "fact": "sales",
                "value": 47117896000000.0,
                "unit": "KRW",
                "consolidated": True,
                "restatement_id": "r0",
                "source_kind": "opendart_standard",
                "mapping_version": "dart-fact-map-v1",
                "raw_document_hash": None,
            },
            {
                "company_id": "00126380",
                "fiscal_period": "2015Q1",
                "filing_id": "20150515001111",
                "fact": "operating_profit",
                "value": 5979343000000.0,
                "unit": "KRW",
                "consolidated": True,
                "restatement_id": "r0",
                "source_kind": "opendart_standard",
                "mapping_version": "dart-fact-map-v1",
                "raw_document_hash": None,
            },
        ],
    }
    decision_time = datetime(2016, 1, 1, 9, 0, tzinfo=UTC)
    calendar = SessionCalendar((datetime(2015, 5, 18, 9, 0, tzinfo=UTC),))
    df, _ = normalize_dart_financial_facts_with_quarantine(
        pages=[page],
        disclosure_rows=(),
        source_hash="a" * 64,
        calendar=calendar,
        decision_time=decision_time,
        ticker_by_corp_code={"00126380": "005930"},
        bridge_receipt_hash="b" * 64,
    )
    assert df.height == 2
    assert set(df["fact"].to_list()) == {"sales", "operating_profit"}
    assert df["company_id"].to_list() == ["005930", "005930"]
    assert df["ticker"].to_list() == ["005930", "005930"]
    assert df["dart_corp_code"].to_list() == ["00126380", "00126380"]
    assert df["fiscal_period"].to_list() == ["2015Q1", "2015Q1"]
    assert df["value"].to_list() == [47117896000000.0, 5979343000000.0]


def test_filing_identities_from_bronze_multi_receipt(tmp_path: Any) -> None:
    from datetime import date

    from src.data.dart_disclosures import DisclosureRecord, periodic_filing_identities

    records = [
        DisclosureRecord(corp_code="00126380", rcept_no="20150515001111", rcept_dt=date(2015, 5, 15), report_nm="분기보고서 (2015.03)"),
        DisclosureRecord(corp_code="00126380", rcept_no="20150817002222", rcept_dt=date(2015, 8, 17), report_nm="반기보고서 (2015.06)"),
    ]

    identities = periodic_filing_identities(
        records, start=date(2015, 1, 1), end=date(2015, 12, 31),
        ticker_by_corp_code=None, required_periods=None, corp_codes=None,
    )
    assert len(identities) == 2
    fids = {item["filing_id"] for item in identities}
    assert fids == {"20150515001111", "20150817002222"}
    reprt_codes = {item["reprt_code"] for item in identities}
    assert reprt_codes == {"11013", "11012"}


def test_filing_identities_attach_frozen_ticker_and_required_period_only(tmp_path) -> None:
    from datetime import date

    from src.data.dart_disclosures import DisclosureRecord, periodic_filing_identities

    records = [
        DisclosureRecord(corp_code="00126380", rcept_no="20150515000001", rcept_dt=date(2015, 5, 15), report_nm="분기보고서 (2015.03)"),
        DisclosureRecord(corp_code="00126380", rcept_no="20151115000002", rcept_dt=date(2015, 11, 15), report_nm="분기보고서 (2015.09)"),
    ]

    rows = periodic_filing_identities(
        records, start=date(2015, 1, 1), end=date(2015, 12, 31),
        ticker_by_corp_code={"00126380": "005930"}, required_periods=frozenset({"2015Q1"}), corp_codes=None,
    )

    assert len(rows) == 1
    assert rows[0]["ticker"] == "005930"
    assert rows[0]["reprt_code"] == "11013"

def test_fetch_one_financial_fact_source_matches_original_per_identity_behavior() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    # Given: the same fixture as the pre-existing CFS-success test.
    mock_raw = {
        "status": "000",
        "list": [
            {
                "rcept_no": "20150515001111",
                "bsns_year": "2015",
                "corp_code": "00126380",
                "reprt_code": "11013",
                "account_id": "ifrs-full_Revenue",
                "account_nm": "매출액",
                "fs_div": "CFS",
                "thstrm_amount": "47,117,896,000,000",
            }
        ],
    }
    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="fixture-key", request_json=lambda _e, _p: mock_raw)
    identity = {
        "corp_code": "00126380",
        "filing_id": "20150515001111",
        "rcept_no": "20150515001111",
        "biz_year": "2015",
        "reprt_code": "11013",
        "fs_div": "CFS",
        "published_at": "2015-05-15",
        "ticker": "",
    }

    # When
    page = collector._fetch_one_financial_fact_source(identity)

    # Then
    assert page["source_kind"] == "opendart_standard"
    assert page["status"] == "000"
    sales_rec = next(r for r in page["records"] if r["fact"] == "sales")
    assert sales_rec["value"] == 47117896000000.0
    assert sales_rec["consolidated"] is True


def test_fetch_one_financial_fact_source_client_transport_success_and_error_wrapping() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    identity = {
        "corp_code": "00000001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "2020",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # Given/When/Then: client transport succeeds (status 013, no request_json set).
    class FakeClientOK:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b""

    collector_ok = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", client=FakeClientOK())
    page = collector_ok._fetch_one_financial_fact_source(identity)
    assert page["source_kind"] == "unavailable"

    # Given/When/Then: client raises a DART-specific exception -> wrapped.
    class FakeClientDartError:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            from src.integrations.dart.client import DartApiError

            raise DartApiError("boom")

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b""

    collector_dart_err = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", client=FakeClientDartError())
    err_page = collector_dart_err._fetch_one_financial_fact_source(identity)
    assert err_page["source_kind"] == "unavailable"
    assert any("boom" in str(entry) for entry in err_page["diagnostics"])

    # Given/When/Then: client raises a generic exception -> also wrapped.
    class FakeClientGeneric:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            raise RuntimeError("network blip")

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b""

    collector_generic = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", client=FakeClientGeneric())
    generic_page = collector_generic._fetch_one_financial_fact_source(identity)
    assert generic_page["source_kind"] == "unavailable"
    assert any("network blip" in str(entry) for entry in generic_page["diagnostics"])


def test_fetch_one_financial_fact_source_raises_when_no_transport_configured() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.dart.xbrl import DartXbrlCollector

    identity = {
        "corp_code": "00000001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "2020",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # Given: a constructed collector with both request transports cleared post-construction.
    collector_no_request = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k")
    collector_no_request._client = None
    collector_no_request._request_json = None

    # When/Then: the request stage fails closed.
    with pytest.raises(PITDataError, match="not configured"):
        collector_no_request._fetch_one_financial_fact_source(identity)

    # Given: a collector that can request but cannot fetch the archive fallback.
    collector_no_archive = DartXbrlCollector(
        api_key="k", min_interval=0.0, max_workers=1, request_json=lambda _e, _p: {"status": "013", "list": []}
    )
    collector_no_archive._client = None
    collector_no_archive._request_bytes = None

    # When/Then: the archive stage fails closed too.
    with pytest.raises(PITDataError, match="not configured"):
        collector_no_archive._fetch_one_financial_fact_source(identity)

    # Given/When/Then: an empty (falsy) raw response is isolated as unavailable, naming the filing.
    collector_empty_raw = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", request_json=lambda _e, _p: {})
    empty_page = collector_empty_raw._fetch_one_financial_fact_source(identity)
    assert empty_page["source_kind"] == "unavailable"
    assert any("F1" in str(entry) for entry in empty_page["diagnostics"])


def test_fetch_one_financial_fact_source_records_every_row_diagnostic_kind() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    # Given: one row per malformed shape, plus a valid row with no fiscal_period source.
    rows: list[object] = [
        "not-a-dict",
        {"account_id": "unknown_xyz", "account_nm": "???", "bsns_year": "2020"},
        {"account_id": "ifrs-full_Revenue", "account_nm": "매출액", "bsns_year": "2020"},
        {"account_id": "ifrs-full_Revenue", "account_nm": "매출액", "thstrm_amount": "abc", "bsns_year": "2020"},
        {"account_id": "ifrs-full_Revenue", "account_nm": "매출액", "thstrm_amount": "inf", "bsns_year": "2020"},
        {"account_id": "ifrs-full_Revenue", "account_nm": "매출액", "thstrm_amount": "100"},
    ]
    collector = DartXbrlCollector(
        api_key="k", min_interval=0.0, max_workers=1, request_json=lambda _e, _p: {"status": "000", "list": rows}
    )
    # identity.biz_year is empty so the last row (no row-level bsns_year) cannot resolve a period.
    identity = {
        "corp_code": "00000001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # When
    page = collector._fetch_one_financial_fact_source(identity)

    # Then: every malformed row produced its own diagnostic and was skipped.
    diagnostics = page["diagnostics"]
    assert any(d.startswith("unknown_account") for d in diagnostics)
    assert any(d.startswith("missing_amount") for d in diagnostics)
    assert any(d.startswith("non_finite") for d in diagnostics)
    assert any(d.startswith("missing_fiscal_period") for d in diagnostics)
    assert page["records"] == []

    # Given/When/Then: a status-000 response with an empty facts list falls through
    # to the archive fallback rather than raising.
    empty_collector = DartXbrlCollector(
        api_key="k",
        min_interval=0.0,
        max_workers=1,
        request_json=lambda _e, _p: {"status": "000", "list": []},
        request_bytes=lambda _e, _p: b"",
    )
    ofs_identity = {**identity, "biz_year": "2020", "fs_div": "OFS"}
    empty_page = empty_collector._fetch_one_financial_fact_source(ofs_identity)
    assert empty_page["source_kind"] == "unavailable"


def test_fetch_one_financial_fact_source_archive_fetch_error_handling() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.dart.xbrl import DartXbrlCollector

    identity = {
        "corp_code": "00000001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "2020",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # Given/When/Then: the client-based archive fetch path is used and succeeds.
    class FakeClientArchiveOK:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b""

    collector_ok = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", client=FakeClientArchiveOK())
    page = collector_ok._fetch_one_financial_fact_source(identity)
    assert page["source_kind"] == "unavailable"

    # Given/When/Then: a PITDataError from the client archive fetch propagates as-is.
    class FakeClientArchiveRaisesPIT:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            raise PITDataError("archive boom")

    collector_pit = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", client=FakeClientArchiveRaisesPIT())
    with pytest.raises(PITDataError, match="archive boom"):
        collector_pit._fetch_one_financial_fact_source(identity)

    # Given/When/Then: a generic exception from the client archive fetch is wrapped.
    class FakeClientArchiveRaisesGeneric:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            raise RuntimeError("archive network blip")

    collector_generic = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", client=FakeClientArchiveRaisesGeneric())
    with pytest.raises(PITDataError, match="F1"):
        collector_generic._fetch_one_financial_fact_source(identity)


def test_fetch_one_financial_fact_source_rejects_invalid_and_parses_valid_legacy_archive() -> None:
    import io
    import zipfile

    from src.integrations.dart.xbrl import DartXbrlCollector

    def make_legacy_archive(files: dict[str, str]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in files.items():
                zf.writestr(name, content.encode("utf-8"))
        return buf.getvalue()

    identity = {
        "corp_code": "001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "2020",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # Given/When/Then: a non-empty, non-zip archive is rejected explicitly.
    collector_bad_zip = DartXbrlCollector(
        api_key="k",
        min_interval=0.0,
        max_workers=1,
        request_json=lambda _e, _p: {"status": "013", "list": []},
        request_bytes=lambda _e, _p: b"not-a-zip-archive",
    )
    bad_zip_page = collector_bad_zip._fetch_one_financial_fact_source(identity)
    assert bad_zip_page["diagnostics"] == ("invalid_document_archive",)

    # Given/When/Then: a valid zip without form sections verifies nothing.
    good_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<document>"
        "<account><account_nm>\ub9e4\ucd9c\uc561</account_nm><amount>100</amount><unit>KRW</unit></account>"
        "<account><account_nm>\uc790\uc0b0\ucd1d\uacc4</account_nm><amount>1000</amount><unit>KRW</unit></account>"
        "</document>"
    )
    good_archive = make_legacy_archive({"F1.xml": good_xml})
    collector_good = DartXbrlCollector(
        api_key="k",
        min_interval=0.0,
        max_workers=1,
        request_json=lambda _e, _p: {"status": "013", "list": []},
        request_bytes=lambda _e, _p: good_archive,
    )
    good_page = collector_good._fetch_one_financial_fact_source(identity)
    assert good_page["source_kind"] == "document_verified"
    assert good_page["status"] == "extraction_failed"
    assert good_page["records"] == []
    assert good_page["raw_archive"] == good_archive

    # Given/When/Then: a valid zip with an ambiguous (duplicate) statement fails extraction.
    ambiguous_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<document>"
        "<account><account_nm>\ub9e4\ucd9c\uc561</account_nm><amount>100</amount><unit>KRW</unit></account>"
        "<account><account_nm>\ub9e4\ucd9c\uc561</account_nm><amount>200</amount><unit>KRW</unit></account>"
        "</document>"
    )
    ambiguous_archive = make_legacy_archive({"F1.xml": ambiguous_xml})
    collector_ambiguous = DartXbrlCollector(
        api_key="k",
        min_interval=0.0,
        max_workers=1,
        request_json=lambda _e, _p: {"status": "013", "list": []},
        request_bytes=lambda _e, _p: ambiguous_archive,
    )
    ambiguous_page = collector_ambiguous._fetch_one_financial_fact_source(identity)
    assert ambiguous_page["source_kind"] == "document_verified"
    assert ambiguous_page["status"] == "extraction_failed"
    assert ambiguous_page["records"] == []


def _collector_identity() -> dict[str, str]:
    return {
        "corp_code": "00126380",
        "filing_id": "20150515001111",
        "rcept_no": "20150515001111",
        "biz_year": "2015",
        "reprt_code": "11013",
        "fs_div": "CFS",
        "published_at": "2015-05-15",
        "ticker": "",
    }


def test_fetch_one_fact_source_does_not_multiply_retries() -> None:
    from src.integrations.dart.client import ProviderRetryableError
    from src.integrations.dart.xbrl import DartXbrlCollector

    calls: list[object] = []

    def always_retryable(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        calls.append(params.get("fs_div"))
        raise ProviderRetryableError("DART status 900: transient")

    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", request_json=always_retryable)

    page = collector._fetch_one_financial_fact_source(_collector_identity())

    assert calls == ["CFS"]
    assert page["source_kind"] == "unavailable"


def test_fetch_one_fact_source_quota_block_stops_after_single_call() -> None:
    from src.integrations.dart.client import ProviderQuotaExhaustedError
    from src.integrations.dart.xbrl import DartXbrlCollector

    calls: list[object] = []

    def quota_blocked(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        calls.append(params.get("fs_div"))
        raise ProviderQuotaExhaustedError("DART status 020: quota exceeded")

    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", request_json=quota_blocked)

    page = collector._fetch_one_financial_fact_source(_collector_identity())

    assert calls == ["CFS"]
    assert page["source_kind"] == "blocked"



def _ok_response(params: dict[str, str]) -> dict[str, object]:
    return {
        "status": "000",
        "message": "정상",
        "list": [
            {
                "rcept_no": "20160515001111",
                "bsns_year": params["bsns_year"],
                "corp_code": params["corp_code"],
                "reprt_code": params["reprt_code"],
                "account_id": "ifrs-full_Revenue",
                "account_nm": "매출액",
                "fs_div": "CFS",
                "sj_div": "IS",
                "thstrm_amount": "1,000",
            }
        ],
    }


def _identities(count: int) -> tuple[dict[str, str], ...]:
    return tuple({**_collector_identity(), "corp_code": f"{index:08d}", "filing_id": f"2016051500{index:04d}"} for index in range(count))


def test_circuit_breaker_stops_after_consecutive_transport_failures_and_keeps_good_pages() -> None:
    from src.integrations.dart.client import ProviderRetryableError
    from src.integrations.dart.xbrl import DartXbrlCollector

    calls: list[str] = []

    def flaky(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        calls.append(params["corp_code"])
        if int(params["corp_code"]) < 2:
            return _ok_response(params)
        raise ProviderRetryableError("DART transport failed for fnlttSinglAcntAll.json: connection reset")

    collector = DartXbrlCollector(min_interval=0.0, api_key="k", request_json=flaky, max_workers=1)
    pages = list(collector.fetch_financial_fact_sources(_identities(20)))

    assert collector.aborted is True
    assert len(calls) < 20 * 2
    assert all(page["identity"]["corp_code"] in {"00000000", "00000001"} for page in pages)
    assert not any(page.get("source_kind") == "unavailable" and "transport" in str(page["diagnostics"]) for page in pages)


def test_circuit_breaker_ignores_isolated_failures() -> None:
    from src.integrations.dart.client import ProviderRetryableError
    from src.integrations.dart.xbrl import DartXbrlCollector

    def one_bad(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        if params["corp_code"] == "00000003":
            raise ProviderRetryableError("DART transport failed for fnlttSinglAcntAll.json: reset")
        return _ok_response(params)

    collector = DartXbrlCollector(min_interval=0.0, api_key="k", request_json=one_bad, max_workers=1)
    pages = list(collector.fetch_financial_fact_sources(_identities(8)))

    assert collector.aborted is False
    assert len(pages) == 8


def test_circuit_breaker_also_stops_parallel_fetch() -> None:
    from src.integrations.dart.client import ProviderRetryableError
    from src.integrations.dart.xbrl import DartXbrlCollector

    calls: list[str] = []

    def always_down(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        calls.append(params["corp_code"])
        raise ProviderRetryableError("DART transport failed for fnlttSinglAcntAll.json: reset")

    collector = DartXbrlCollector(min_interval=0.0, api_key="k", request_json=always_down, max_workers=4)
    pages = list(collector.fetch_financial_fact_sources(_identities(60)))

    assert collector.aborted is True
    assert pages == []
    assert len(calls) < 60


def test_health_check_uses_the_client_ping_and_requires_a_client() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.dart.xbrl import DartXbrlCollector

    pinged: list[bool] = []

    class _Client:
        def ping(self) -> None:
            pinged.append(True)

    DartXbrlCollector(api_key="k", min_interval=0.0, max_workers=1, client=_Client()).health_check()
    assert pinged == [True]
    with pytest.raises(PITDataError):
        DartXbrlCollector(api_key="k", min_interval=0.0, max_workers=1, request_json=lambda *_: {}).health_check()


def test_collector_delegates_listing_and_archive_fetch_to_the_client() -> None:
    from datetime import date

    import pytest

    from src.core.pit import PITDataError
    from src.integrations.dart.xbrl import DartXbrlCollector

    class _Client:
        def list_disclosures(self, start, end, *, disclosure_filter=None):
            code = getattr(disclosure_filter, "code", disclosure_filter)
            return [{"rcept_no": "1", "detail_type": code, "start": start.isoformat(), "end": end.isoformat()}]

        def fetch_document_archive(self, rcept_no):
            return bytearray(f"zip:{rcept_no}".encode())

    from src.config.providers import DisclosureFilter

    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", client=_Client())
    assert collector.list_disclosures(
        date(2020, 1, 1), date(2020, 1, 31),
        disclosure_filter=DisclosureFilter(code="I001", parameter="pblntf_detail_ty"),
    ) == [
        {"rcept_no": "1", "detail_type": "I001", "start": "2020-01-01", "end": "2020-01-31"}
    ]
    assert collector.fetch_document_archive("20200101000001") == b"zip:20200101000001"
    unconfigured = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", request_json=lambda *_: {})
    with pytest.raises(PITDataError):
        unconfigured.list_disclosures(date(2020, 1, 1), date(2020, 1, 31))
    with pytest.raises(PITDataError):
        unconfigured.fetch_document_archive("20200101000001")


def test_document_not_found_is_recorded_as_absence_not_retried_forever() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    class _Client:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return "<?xml version='1.0'?><result><status>014</status><message>파일이 존재하지 않습니다.</message></result>".encode()

    page = DartXbrlCollector(api_key="k", min_interval=0.0, max_workers=1, client=_Client())._fetch_one_financial_fact_source(_collector_identity())

    assert page["source_kind"] == "legacy_document"
    assert page["records"] == []
    assert page["diagnostics"] == ("document_not_found",)


def test_archive_transport_failure_returns_a_page_instead_of_losing_the_chunk() -> None:
    from src.integrations.dart.client import ProviderQuotaExhaustedError, ProviderRetryableError
    from src.integrations.dart.xbrl import DartXbrlCollector, _is_transport_failure

    def collector_raising(error: Exception) -> DartXbrlCollector:
        class _Client:
            def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
                return {"status": "013", "list": []}

            def fetch_document_archive(self, rcept_no: str) -> bytes:
                raise error

        return DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="k", client=_Client())

    reset = collector_raising(ProviderRetryableError("DART transport failed for document.xml: reset"))._fetch_one_financial_fact_source(_collector_identity())
    cooldown = collector_raising(ProviderQuotaExhaustedError("OpenDART document.xml blocked until later"))._fetch_one_financial_fact_source(_collector_identity())
    exhausted = collector_raising(ProviderQuotaExhaustedError("DART status 020"))._fetch_one_financial_fact_source(_collector_identity())

    assert _is_transport_failure(reset)
    assert cooldown["source_kind"] == "blocked"
    assert exhausted["source_kind"] == "blocked"


def test_dart_status_800_and_900_count_toward_the_circuit_breaker() -> None:
    from src.integrations.dart.xbrl import _is_transport_failure

    assert _is_transport_failure({"source_kind": "unavailable", "diagnostics": ("dart_error:DART status 900: overloaded",)})
    assert not _is_transport_failure({"source_kind": "unavailable", "diagnostics": ("invalid_document_archive",)})
    assert not _is_transport_failure({"source_kind": "opendart_standard", "diagnostics": ()})

def test_dart_xbrl_env_workers_and_filing_identity_filter(monkeypatch, tmp_path: Path) -> None:
    from datetime import date

    from src.data.dart_disclosures import DisclosureRecord, periodic_filing_identities
    from src.integrations.dart.xbrl import DartXbrlCollector

    monkeypatch.setenv("OPENDART_MAX_WORKERS", "3")
    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="key", request_json=lambda *_: {})
    assert collector._max_workers == 1
    records = [
        DisclosureRecord(corp_code="001", rcept_no="r1", rcept_dt=date(2026, 3, 6), report_nm="(2025.12) 사업보고서"),
        DisclosureRecord(corp_code="002", rcept_no="r2", rcept_dt=date(2026, 3, 6), report_nm="(2025.12) 사업보고서"),
    ]
    assert periodic_filing_identities(
        records,
        start=date(2026, 1, 1),
        end=date(2026, 12, 31),
        ticker_by_corp_code=None,
        required_periods=None,
        corp_codes=frozenset({"001"}),
    ) == ({
        "corp_code": "001",
        "filing_id": "r1",
        "rcept_no": "r1",
        "biz_year": "2025",
        "reprt_code": "11011",
        "fs_div": "CFS",
        "published_at": "2026-03-06",
        "correction_of": "",
        "fiscal_period": "2025Q4",
    },)

import pytest

from src.core.pit import PITDataError


def test_dart_rejects_missing_filing_identity() -> None:
    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key='test-key', request_json=lambda endpoint, params: {})
    with pytest.raises(PITDataError, match='filing identity'):
        tuple(collector.fetch_xbrl_facts(({'filing_id': 'F1'},)))


def _verified_zip_bytes() -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("doc.xml", "<html>fixture</html>")
    return buf.getvalue()


def test_verified_document_page_carries_basis_and_archive(monkeypatch) -> None:
    from datetime import date

    from src.integrations.dart.document_statements import REVISION as DOCUMENT_STATEMENTS_REVISION
    from src.integrations.dart.document_statements import (
        DocumentParseResult,
        PeriodBasis,
        StatementFact,
        VerifiedStatements,
    )
    from src.integrations.dart.xbrl import DartXbrlCollector

    statements = VerifiedStatements(
        consolidated=True,
        period_end=date(2020, 12, 31),
        report_kind="annual",
        unit_multipliers={"BS": 1},
        facts=(
            StatementFact(fact="assets", value=1000, basis=PeriodBasis.POINT_IN_TIME, label="자산총계"),
            StatementFact(fact="sales", value=2000, basis=PeriodBasis.ANNUAL, label="매출액"),
        ),
        checks=("bs_balance",),
    )
    monkeypatch.setattr(
        "src.integrations.dart.document_statements.parse_filing_document",
        lambda archive, *, reprt_code, biz_year: DocumentParseResult(statements=statements, diagnostics=()),
    )
    archive = _verified_zip_bytes()
    collector = DartXbrlCollector(
        api_key="k",
        min_interval=0.0,
        max_workers=1,
        request_json=lambda _e, _p: {"status": "013", "list": []},
        request_bytes=lambda _e, _p: archive,
    )
    page = collector._fetch_one_financial_fact_source(_collector_identity())

    assert page["source_kind"] == "document_verified"
    assert page["status"] == "000"
    assert page["fs_div"] == "CFS"
    assert page["parser_version"] == DOCUMENT_STATEMENTS_REVISION
    assert page["checks"] == ["bs_balance"]
    assert page["raw_archive"] == archive
    assert page["raw_document_hash"] is not None
    assert len(page["records"]) == 2
    assert all(record["period_basis"] for record in page["records"])
    assert all(record["parser_version"] == DOCUMENT_STATEMENTS_REVISION for record in page["records"])


def test_unverified_document_is_extraction_failed(monkeypatch) -> None:
    from src.integrations.dart.document_statements import DocumentParseResult
    from src.integrations.dart.xbrl import DartXbrlCollector

    monkeypatch.setattr(
        "src.integrations.dart.document_statements.parse_filing_document",
        lambda archive, *, reprt_code, biz_year: DocumentParseResult(statements=None, diagnostics=("identity_failed:bs_balance",)),
    )
    archive = _verified_zip_bytes()
    collector = DartXbrlCollector(
        api_key="k",
        min_interval=0.0,
        max_workers=1,
        request_json=lambda _e, _p: {"status": "013", "list": []},
        request_bytes=lambda _e, _p: archive,
    )
    page = collector._fetch_one_financial_fact_source(_collector_identity())

    assert page["source_kind"] == "document_verified"
    assert page["status"] == "extraction_failed"
    assert page["records"] == []
    assert page["raw_archive"] == archive
