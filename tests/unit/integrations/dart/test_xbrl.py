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
        api_key="k", request_json=failing_at_index_2, request_bytes=empty_bytes, max_workers=5
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

    collector = DartXbrlCollector(
        api_key="key",
        quota_store=quota_store,
        now=now,
        min_interval=2.0,
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


def test_fetch_one_financial_fact_source_retries_retryable_status_to_recovery(monkeypatch) -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import DartRetryableError

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    calls: list[str] = []

    class _FlakyClient:
        def _request_validated(self, _endpoint: str, params: dict[str, str]) -> dict[str, object]:
            calls.append(params["fs_div"])
            if len(calls) == 1:
                raise DartRetryableError("DART status 900: transient")
            return _fact_success_payload(params["corp_code"])

    collector = xbrl_module.DartXbrlCollector(client=_FlakyClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then: recovered within the retry budget on the same fs_div.
    assert page["source_kind"] == "opendart_standard"
    assert calls == ["CFS", "CFS"]


def test_fetch_one_financial_fact_source_isolates_exhausted_retryable_status(monkeypatch) -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import DartRetryableError

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    calls: list[str] = []

    class _AlwaysRetryableClient:
        def _request_validated(self, _endpoint: str, params: dict[str, str]) -> dict[str, object]:
            calls.append(params["fs_div"])
            raise DartRetryableError("DART status 900: transient")

    collector = xbrl_module.DartXbrlCollector(client=_AlwaysRetryableClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then: exhausted budget isolates as unavailable, never raises, never blocked.
    assert page["source_kind"] == "unavailable"
    assert page["source_kind"] != "blocked"
    assert any(str(entry).startswith("dart_error:") for entry in page["diagnostics"])
    assert calls == ["CFS", "CFS", "CFS"]


def test_fetch_one_financial_fact_source_isolates_quota_exhaustion_as_blocked() -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import DartQuotaExhaustedError

    archive_calls: list[str] = []

    class _QuotaBlockedClient:
        def _request_validated(self, _endpoint: str, _params: dict[str, str]) -> dict[str, object]:
            raise DartQuotaExhaustedError("DART status 020: quota exceeded")

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            archive_calls.append(rcept_no)
            raise AssertionError("document archive must not be fetched after quota exhaustion")

    collector = xbrl_module.DartXbrlCollector(client=_QuotaBlockedClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then: blocked record without exhausting the fallback chain.
    assert page["source_kind"] == "blocked"
    assert "dart_quota_exhausted" in page["diagnostics"]
    assert archive_calls == []


def test_fetch_one_financial_fact_source_isolates_unexpected_terminal_status() -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import DartTerminalError

    class _TerminalClient:
        def _request_validated(self, _endpoint: str, _params: dict[str, str]) -> dict[str, object]:
            raise DartTerminalError("DART status 101: unknown status")

    collector = xbrl_module.DartXbrlCollector(client=_TerminalClient(), api_key="k")

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then: original error text is preserved in diagnostics, nothing raised.
    assert page["source_kind"] == "unavailable"
    assert any("101" in str(entry) for entry in page["diagnostics"])


def test_fetch_financial_fact_sources_stops_early_on_first_blocked_identity() -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import DartQuotaExhaustedError

    requested: list[str] = []

    class _BatchClient:
        def _request_validated(self, _endpoint: str, params: dict[str, str]) -> dict[str, object]:
            requested.append(params["corp_code"])
            if params["corp_code"] == "00000003":
                raise DartQuotaExhaustedError("DART status 020: quota exceeded")
            return _fact_success_payload(params["corp_code"])

    collector = xbrl_module.DartXbrlCollector(client=_BatchClient(), api_key="k", max_workers=1)
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

    collector = xbrl_module.DartXbrlCollector(client=_OkClient(), api_key="k", max_workers=1)
    identities = tuple(_fact_identity(f"{i:08d}", f"F{i}") for i in range(1, 4))

    # When
    pages = list(collector.fetch_financial_fact_sources(identities))

    # Then
    assert len(pages) == 3
    assert all(page["source_kind"] == "opendart_standard" for page in pages)


def test_fetch_one_financial_fact_source_isolates_terminal_error_without_status_code() -> None:
    import src.integrations.dart.xbrl as xbrl_module
    from src.integrations.dart.client import DartTerminalError

    class _NoStatusClient:
        def _request_validated(self, _endpoint: str, _params: dict[str, str]) -> dict[str, object]:
            raise DartTerminalError("connection reset by peer")

    collector = xbrl_module.DartXbrlCollector(client=_NoStatusClient(), api_key="k")

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

    collector = xbrl_module.DartXbrlCollector(client=_BoomClient(), api_key="k")

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

    collector = DartXbrlCollector(api_key="k", request_json=raw_retryable, request_bytes=empty_bytes)

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

    collector = DartXbrlCollector(api_key="k", request_json=raw_blocked, request_bytes=forbidden_bytes)

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

    collector = DartXbrlCollector(api_key="k", request_json=malformed, request_bytes=empty_bytes)

    # When
    page = collector._fetch_one_financial_fact_source(_fact_identity("00000001", "F0"))

    # Then
    assert page["source_kind"] == "unavailable"
    assert any(str(entry).startswith("dart_error:") for entry in page["diagnostics"])
