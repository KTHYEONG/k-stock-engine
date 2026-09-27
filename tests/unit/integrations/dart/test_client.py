"""DART status classification and public-collector-API invariants."""


def test_status_classification() -> None:
    import pytest

    from src.integrations.dart.client import classify_dart_status
    from src.integrations.errors import (
        ProviderQuotaExhaustedError,
        ProviderRetryableError,
        ProviderTerminalError,
    )

    classify_dart_status("000", {"status": "000"}, "ep")
    classify_dart_status("013", {"status": "013"}, "ep")
    classify_dart_status("014", {"status": "014"}, "ep")
    with pytest.raises(ProviderQuotaExhaustedError):
        classify_dart_status("020", {"status": "020"}, "ep")
    with pytest.raises(ProviderRetryableError):
        classify_dart_status("800", {"status": "800"}, "ep")
    with pytest.raises(ProviderRetryableError):
        classify_dart_status("900", {"status": "900"}, "ep")
    with pytest.raises(ProviderTerminalError):
        classify_dart_status("010", {"status": "010"}, "ep")


def test_collector_uses_only_public_client_api() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    class _FakeClient:
        def request_validated(self, endpoint, params):
            assert endpoint == "fnlttSinglAcntAll.json"
            return {
                "status": "000",
                "list": [
                    {
                        "account_id": "ifrs-full_Revenue",
                        "account_nm": "매출액",
                        "thstrm_amount": "1000",
                        "bsns_year": "2023",
                        "reprt_code": "11011",
                        "corp_code": "00126380",
                        "rcept_no": "20240101000001",
                        "fs_div": "CFS",
                    }
                ],
            }

        def fetch_document_archive(self, rcept_no):
            raise AssertionError("archive must not be needed for a standard hit")

        def list_disclosures(self, start, end, **kwargs):
            return []

        def ping(self):
            pass

        def load_corp_code_records(self):
            return ()

    assert not hasattr(_FakeClient, "_request_validated")
    collector = DartXbrlCollector(api_key="key", min_interval=0.0, max_workers=1, client=_FakeClient())
    pages = tuple(
        collector.fetch_financial_fact_sources(
            (
                {
                    "corp_code": "00126380",
                    "filing_id": "20240101000001",
                    "rcept_no": "20240101000001",
                    "biz_year": "2023",
                    "reprt_code": "11011",
                    "fs_div": "CFS",
                    "published_at": "2024-01-01",
                    "ticker": "005930",
                },
            )
        )
    )

    assert len(pages) == 1
    assert pages[0]["source_kind"] == "opendart_standard"


def test_pace_delegates_to_transport() -> None:
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=0.0)
    calls: list[None] = []
    client._transport._pace = lambda: calls.append(None)  # type: ignore[method-assign]

    client._pace()

    assert calls == [None]


def test_generic_quota_exhaustion_becomes_dart_quota_error() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient, ProviderQuotaExhaustedError

    class _GenericGate:
        def acquire(self, *, endpoint: str) -> None:
            raise ProviderQuotaExhaustedError("full", provider="OpenDART", endpoint=endpoint)

        def record_rate_limit(self, *, endpoint: str, retry_after: float | None) -> None:
            pass

    client = DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=0.0)
    client._transport._quota = _GenericGate()

    with pytest.raises(ProviderQuotaExhaustedError, match="full"):
        client.request_validated("list.json", {"bgn_de": "20240101"})


def test_seam_unknown_status_is_terminal() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient, ProviderTerminalError

    client = DartApiClient(
        api_key="key",
        quota_provider="OpenDART",
        min_interval=0.0,
        raw_request_json=lambda _e, _p: {"status": "010", "message": "bad"},
    )

    with pytest.raises(ProviderTerminalError, match="010"):
        client.request_validated("list.json", {"bgn_de": "20240101"})


def test_list_disclosures_through_transport(tmp_path) -> None:
    from datetime import date
    from types import SimpleNamespace

    from src.integrations.dart.client import DartApiClient

    pages = {
        "1": {"status": "000", "total_page": "1", "list": [
            {"rcept_no": "20240101000001", "rcept_dt": "20240101", "corp_code": "001",
             "corp_name": "A", "report_nm": "분기보고서 (2024.03)", "rm": ""},
        ]},
    }

    def fake_get(_url, *, params=None, headers=None, timeout=None):
        return SimpleNamespace(status_code=200, headers={}, json=lambda: pages[params["page_no"]])

    client = DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=0.0)
    client._session = SimpleNamespace(get=fake_get)  # type: ignore[assignment]

    rows = client.list_disclosures(date(2024, 1, 1), date(2024, 1, 2))

    assert [row["rcept_no"] for row in rows] == ["20240101000001"]


def test_list_disclosures_rejects_contradictory_receipts() -> None:
    import pytest
    from datetime import date

    from src.integrations.dart.client import DartApiError, DartApiClient

    pages = {
        "1": {"status": "000", "total_page": "2", "list": [
            {"rcept_no": "20240101000001", "rcept_dt": "20240101", "corp_code": "001",
             "corp_name": "A", "report_nm": "X", "rm": ""},
        ]},
        "2": {"status": "000", "total_page": "2", "list": [
            {"rcept_no": "20240101000001", "rcept_dt": "20240101", "corp_code": "001",
             "corp_name": "B", "report_nm": "X", "rm": ""},
        ]},
    }
    client = DartApiClient(
        api_key="key", quota_provider="OpenDART", min_interval=0.0,
        request_json=lambda _e, p: {"status": "000", "total_page": "2", "list": pages[p["page_no"]]["list"]},
    )

    with pytest.raises(DartApiError, match="contradictory"):
        client.list_disclosures(date(2024, 1, 1), date(2024, 1, 2))


def test_fetch_multi_accounts_through_seam() -> None:
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(
        api_key="key", quota_provider="OpenDART", min_interval=0.0,
        raw_request_json=lambda _e, _p: {"status": "000", "list": [{"corp_code": "00126380"}]},
    )

    assert client.fetch_multi_accounts(("00126380",), biz_year="2023", reprt_code="11011") == [{"corp_code": "00126380"}]


def test_fetch_document_archive_seam_error_payload_rejected() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient, ProviderTerminalError

    client = DartApiClient(
        api_key="key", quota_provider="OpenDART", min_interval=0.0,
        request_bytes=lambda _e, _p: b'{"status": "013"}',
    )

    with pytest.raises(ProviderTerminalError, match="error payload"):
        client.fetch_document_archive("20240101000001")


def test_collector_fallback_to_legacy_private_seam() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    class _LegacyFakeClient:
        def _request_validated(self, endpoint, params):
            return {
                "status": "000",
                "list": [
                    {
                        "account_id": "ifrs-full_Revenue",
                        "account_nm": "매출액",
                        "thstrm_amount": "1000",
                        "bsns_year": "2023",
                        "reprt_code": "11011",
                        "corp_code": "00126380",
                        "rcept_no": "20240101000001",
                        "fs_div": "CFS",
                    }
                ],
            }

        def fetch_document_archive(self, rcept_no):
            raise AssertionError("archive must not be needed")

    assert not hasattr(_LegacyFakeClient, "request_validated")
    collector = DartXbrlCollector(api_key="key", min_interval=0.0, max_workers=1, client=_LegacyFakeClient())
    pages = tuple(
        collector.fetch_financial_fact_sources(
            (
                {
                    "corp_code": "00126380",
                    "filing_id": "20240101000001",
                    "rcept_no": "20240101000001",
                    "biz_year": "2023",
                    "reprt_code": "11011",
                    "fs_div": "CFS",
                    "published_at": "2024-01-01",
                    "ticker": "005930",
                },
            )
        )
    )

    assert pages[0]["source_kind"] == "opendart_standard"


def test_fetch_xbrl_facts_falls_back_to_legacy_private_seam() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    class _LegacyFactsClient:
        def _request_validated(self, endpoint, params):
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no):
            raise AssertionError("no archive in this path")

    collector = DartXbrlCollector(
        api_key="key",
        min_interval=0.0,
        max_workers=1,
        client=_LegacyFactsClient(),
    )
    import pytest

    from src.core.pit import PITDataError

    with pytest.raises(PITDataError, match="empty"):
        tuple(
            collector.fetch_xbrl_facts(
                ({"corp_code": "00126380", "filing_id": "20240101000001", "biz_year": "2023",
                  "reprt_code": "11011", "fs_div": "CFS"},)
            )
        )

import pytest


@pytest.fixture(autouse=True)
def _primary_dart_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic primary key so tests using ``key`` meter the historical ``OpenDART`` ledger regardless of the developer shell."""
    monkeypatch.setenv("OPENDART_API_KEY", "key")


def test_dart_client_rejects_non_success_api_status() -> None:
    from datetime import date

    import pytest

    from src.integrations.dart.client import DartApiClient, DartApiError

    client = DartApiClient(
        api_key='key', quota_provider='OpenDART',
        min_interval=0.0,
        request_json=lambda endpoint, params: {'status': '020', 'message': 'blocked'},
    )

    with pytest.raises(DartApiError, match='020'):
        client.list_disclosures(start=date(2026, 1, 1), end=date(2026, 1, 2))


def test_list_disclosures_collects_all_pages_deduplicates_and_orders() -> None:
    from datetime import date
    from src.integrations.dart.client import DartApiClient

    pages = {"1": {"status": "000", "total_page": "2", "list": [{"rcept_no": "20150515000002", "rcept_dt": "20150515", "corp_code": "001", "corp_name": "A", "report_nm": "분기보고서 (2015.03)", "rm": ""}]}, "2": {"status": "000", "total_page": "2", "list": [{"rcept_no": "20150515000001", "rcept_dt": "20150515", "corp_code": "001", "corp_name": "A", "report_nm": "분기보고서 (2015.03)", "rm": ""}]}}
    client = DartApiClient(api_key="key", quota_provider="OpenDART", request_json=lambda _endpoint, p: pages[p["page_no"]], min_interval=0.0)

    rows = client.list_disclosures(date(2015, 1, 1), date(2015, 12, 31))

    assert [row["rcept_no"] for row in rows] == ["20150515000001", "20150515000002"]


# test_dart_client_preserves_success_and_no_data_pages
def test_fetch_corporate_action_decisions_preserves_success_and_no_data_pages(monkeypatch) -> None:
    from datetime import date

    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key='test-key', min_interval=0.0)
    responses = iter([{'status': '000', 'list': [{'rcept_no': '20240102000001'}]}, {'status': '013', 'list': []}] * 3)
    monkeypatch.setattr(client, 'request_validated', lambda endpoint, params: next(responses))
    monkeypatch.setattr(client, '_request_validated', lambda endpoint, params: next(responses))

    pages = client.fetch_corporate_action_decisions(corp_codes=['00123456'], start=date(2024, 1, 1), end=date(2024, 1, 31))

    assert len(pages) == 5
    assert {page.status for page in pages} == {'000', '013'}
    assert pages[0].records == ({'rcept_no': '20240102000001'},)


def test_fetch_corporate_action_decisions_rejects_invalid_arguments() -> None:
    from datetime import date
    import pytest
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="test-key", min_interval=0.0)
    with pytest.raises(ValueError, match="start"):
        client.fetch_corporate_action_decisions(corp_codes=("1",), start=date(2024, 2, 1), end=date(2024, 1, 1))
    with pytest.raises(ValueError, match="corp_codes"):
        client.fetch_corporate_action_decisions(corp_codes=(), start=date(2024, 1, 1), end=date(2024, 1, 2))


def test_fetch_dividend_disclosures_queries_every_report_code(monkeypatch) -> None:
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="test-key", min_interval=0.0)
    seen: list[dict[str, str]] = []

    def fake_request_validated(endpoint, params):
        seen.append(dict(params))
        if params["reprt_code"] == "11011":
            return {"status": "000", "list": [{"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444"}]}
        return {"status": "013", "list": []}

    monkeypatch.setattr(client, "request_validated", fake_request_validated)
    monkeypatch.setattr(client, "_request_validated", fake_request_validated)

    pages = client.fetch_dividend_disclosures(corp_codes=["00126380"], bsns_years=["2022"])

    assert len(pages) == 4
    assert {page.reprt_code for page in pages} == {"11011", "11012", "11013", "11014"}
    assert all(page.corp_code == "00126380" and page.bsns_year == "2022" for page in pages)
    annual = next(page for page in pages if page.reprt_code == "11011")
    assert annual.status == "000"
    assert annual.records == ({"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444"},)
    assert {call["reprt_code"] for call in seen} == {"11011", "11012", "11013", "11014"}


def test_fetch_dividend_disclosures_rejects_invalid_arguments() -> None:
    import pytest
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="test-key", min_interval=0.0)
    with pytest.raises(ValueError, match="corp_codes"):
        client.fetch_dividend_disclosures(corp_codes=(), bsns_years=("2022",))
    with pytest.raises(ValueError, match="bsns_years"):
        client.fetch_dividend_disclosures(corp_codes=("00126380",), bsns_years=())
def test_dart_client_retries_transport_errors_then_succeeds(monkeypatch) -> None:
    from types import SimpleNamespace

    import requests

    from src.integrations.dart.client import DartApiClient

    attempts = {"n": 0}

    def flaky_get(*_args, **_kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise requests.exceptions.ConnectionError("Connection aborted.")
        return SimpleNamespace(status_code=200, json=lambda: {"status": "000", "list": []})

    client = DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=0.0)
    client._session = SimpleNamespace(get=flaky_get)
    sleeps: list[float] = []
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda seconds: sleeps.append(seconds))

    result = client._request("list.json", {"bgn_de": "20240101", "end_de": "20240101"})

    assert result == {"status": "000", "list": []}
    assert attempts["n"] == 3
    assert len(sleeps) == 2


def test_dart_client_circuit_breaks_after_exhausting_transport_retries(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    import pytest
    import requests

    from src.integrations.dart.client import DartApiClient, ProviderRetryableError
    from src.integrations.quota import ProviderQuotaExhaustedError, ProviderQuotaStateStore

    calls = {"n": 0}

    def always_fail(*_args, **_kwargs):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("Connection aborted.")

    client = DartApiClient(
        api_key="key", quota_provider="OpenDART",
        min_interval=0.0,
        quota_store=ProviderQuotaStateStore(tmp_path),
        now=lambda: datetime(2026, 9, 13, tzinfo=UTC),
    )
    client._session = SimpleNamespace(get=always_fail)
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    with pytest.raises(ProviderRetryableError, match="transport failed"):
        client._request("fnlttSinglAcntAll.json", {"corp_code": "00126380"})
    assert calls["n"] == 3

    with pytest.raises(ProviderQuotaExhaustedError):
        client._request("fnlttSinglAcntAll.json", {"corp_code": "00126380"})
    assert calls["n"] == 3


def test_dart_client_circuit_breaks_on_quota_exceeded_status(tmp_path) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    import pytest

    from src.integrations.dart.client import DartApiClient
    from src.integrations.quota import ProviderQuotaExhaustedError, ProviderQuotaStateStore

    calls: list[object] = []
    client = DartApiClient(
        api_key="key", quota_provider="OpenDART",
        min_interval=0.0,
        quota_store=ProviderQuotaStateStore(tmp_path),
        now=lambda: datetime(2026, 9, 13, tzinfo=UTC),
    )
    client._session = SimpleNamespace(
        get=lambda *_a, **_kw: calls.append(object())
        or SimpleNamespace(status_code=200, json=lambda: {"status": "020", "message": "quota exceeded"})
    )

    with pytest.raises(ProviderQuotaExhaustedError, match="020"):
        client._request_validated("list.json", {"bgn_de": "20240101", "end_de": "20240101"})
    with pytest.raises(ProviderQuotaExhaustedError):
        client._request_validated("list.json", {"bgn_de": "20240102", "end_de": "20240102"})

    assert len(calls) == 1


def test_dart_client_paces_consecutive_requests_by_min_interval(monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from src.integrations.dart.client import DartApiClient

    monotonic_values = iter([1000.0, 1000.0, 1000.3, 1000.3])
    monkeypatch.setattr("src.integrations.dart.client.time.monotonic", lambda: next(monotonic_values))
    sleeps: list[float] = []
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda seconds: sleeps.append(seconds))
    client = DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=1.0)
    responses = iter([SimpleNamespace(status_code=200, json=lambda: {"status": "000", "list": []})] * 2)
    client._session = SimpleNamespace(get=lambda *_a, **_kw: next(responses))

    client._request("list.json", {"bgn_de": "20240101", "end_de": "20240101"})
    client._request("list.json", {"bgn_de": "20240102", "end_de": "20240102"})

    assert sleeps == [pytest.approx(0.7)]


def test_dart_client_pacing_is_explicit_with_no_env_fallback(monkeypatch) -> None:
    import pytest
    from src.integrations.dart.client import DartApiClient

    monkeypatch.setenv("OPENDART_REQUEST_MIN_INTERVAL_SECONDS", "2.5")
    client = DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=0.0)
    assert client._min_interval == 0.0

    monkeypatch.delenv("OPENDART_REQUEST_MIN_INTERVAL_SECONDS", raising=False)
    client_default = DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=0.0)
    assert client_default._min_interval == 0.0

    client_explicit = DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=3.0)
    assert client_explicit._min_interval == 3.0
    with pytest.raises(ValueError, match="daily_request_limit"):
        DartApiClient(api_key="key", quota_provider="OpenDART", daily_request_limit=0, min_interval=0.0)


def test_dart_client_passes_disclosure_detail_type() -> None:
    from datetime import date

    from src.integrations.dart.client import DartApiClient

    seen: dict[str, str] = {}

    def request(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        seen.update(params)
        return {"status": "000", "list": []}

    DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=0.0, request_json=request).list_disclosures(
        date(2024, 1, 1), date(2024, 1, 1), detail_type="A001"
    )

    assert seen["pblntf_detail_ty"] == "A001"


def test_request_validated_raises_distinguishable_quota_exhausted_type() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient, ProviderQuotaExhaustedError
    from src.integrations.errors import ProviderError

    rate_limit_calls: list[dict[str, object]] = []

    class _QuotaStore:
        def record_rate_limit(self, **kwargs: object) -> None:
            rate_limit_calls.append(dict(kwargs))

    client = DartApiClient(
        api_key="key", quota_provider="OpenDART",
        min_interval=0.0,
        raw_request_json=lambda _endpoint, _params: {"status": "020", "message": "quota exceeded"},
        quota_store=_QuotaStore(),  # type: ignore[arg-type]
    )

    # When/Then: distinguishable type that remains a ProviderError, with quota recorded once.
    with pytest.raises(ProviderQuotaExhaustedError, match="020") as exc_info:
        client._request_validated("list.json", {"bgn_de": "20240101", "end_de": "20240101"})
    assert isinstance(exc_info.value, ProviderError)
    assert len(rate_limit_calls) == 1
    assert rate_limit_calls[0]["provider"] == "OpenDART"


def _ledgered_client(tmp_path, monkeypatch, *, responses, daily_request_limit=None):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from src.integrations.dart.client import DartApiClient
    from src.integrations.quota import ProviderQuotaStateStore

    now = datetime(2026, 9, 13, tzinfo=UTC)
    script = list(responses)
    gets: list[object] = []

    def fake_get(*_args, **_kwargs):
        gets.append(object())
        action = script[min(len(gets) - 1, len(script) - 1)]
        if isinstance(action, Exception):
            raise action
        return action

    class _CountingStore(ProviderQuotaStateStore):
        def __init__(self, root):
            super().__init__(root)
            self.acquire_calls = 0
            self.attempt_calls = 0

        def acquire(self, **kwargs):
            self.acquire_calls += 1
            return super().acquire(**kwargs)

        def acquire_and_record(self, **kwargs):
            self.acquire_calls += 1
            self.attempt_calls += 1
            return super().acquire_and_record(**kwargs)

        def record_attempt(self, **kwargs):
            self.attempt_calls += 1
            return super().record_attempt(**kwargs)

    store = _CountingStore(tmp_path)
    client = DartApiClient(
        api_key="key", quota_provider="OpenDART",
        min_interval=0.0,
        quota_store=store,
        now=lambda: now,
        daily_request_limit=daily_request_limit,
    )
    client._session = SimpleNamespace(get=fake_get)
    return client, store, gets, store, store, now


def _ok_response(status="000"):
    from types import SimpleNamespace

    return SimpleNamespace(status_code=200, json=lambda status=status: {"status": status, "list": []})


def test_fetch_document_archive_is_ledgered(tmp_path, monkeypatch) -> None:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("document.xml", "<doc/>")
    payload = buf.getvalue()

    from types import SimpleNamespace

    client, store, gets, _acquire, _attempts, now = _ledgered_client(
        tmp_path, monkeypatch, responses=[SimpleNamespace(status_code=200, content=payload)]
    )
    before = store.remaining_daily_attempts(provider="OpenDART", now=now, daily_limit=15_200)

    assert client.fetch_document_archive("20240101000001") == payload
    assert len(gets) == 1
    assert before - store.remaining_daily_attempts(provider="OpenDART", now=now, daily_limit=15_200) == 1


def test_load_corp_code_records_is_ledgered_and_paced(tmp_path, monkeypatch) -> None:
    import io
    import zipfile

    xml = (
        "<result><list><stock_code>005930</stock_code>"
        "<corp_code>00126380</corp_code><corp_name>삼성전자</corp_name></list></result>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("CORPCODE.xml", xml.encode("utf-8"))

    from types import SimpleNamespace

    client, _store, gets, _acquire, attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[SimpleNamespace(status_code=200, content=buf.getvalue())]
    )
    paces: list[None] = []
    monkeypatch.setattr(client._transport, "_pace", lambda: paces.append(None))

    records = client.load_corp_code_records()

    assert [(rec.ticker, rec.corp_code) for rec in records] == [("005930", "00126380")]
    assert len(gets) == 1
    assert attempts.attempt_calls == 1
    assert len(paces) == 1


def test_request_validated_status_900_shares_attempt_budget(tmp_path, monkeypatch) -> None:
    import pytest

    from src.integrations.dart.client import ProviderRetryableError

    client, _store, gets, acquire, attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[_ok_response("900")] * 3
    )
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    with pytest.raises(ProviderRetryableError, match="900"):
        client._request_validated("fnlttSinglAcntAll.json", {"corp_code": "00126380"})

    assert len(gets) == 3
    assert acquire.acquire_calls == 3
    assert attempts.attempt_calls == 3


def test_request_validated_status_900_then_success(tmp_path, monkeypatch) -> None:
    client, _store, gets, acquire, attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[_ok_response("900"), _ok_response("000")]
    )
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    payload = client._request_validated("fnlttSinglAcntAll.json", {"corp_code": "00126380"})

    assert payload["status"] == "000"
    assert len(gets) == 2
    assert acquire.acquire_calls == 2
    assert attempts.attempt_calls == 2


def test_request_mixed_causes_share_attempt_budget(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest
    import requests

    from src.integrations.dart.client import ProviderRetryableError

    def bad_json():
        raise ValueError("No JSON could be decoded")

    client, _store, gets, acquire, attempts, _now = _ledgered_client(
        tmp_path,
        monkeypatch,
        responses=[
            requests.exceptions.ConnectionError("Connection aborted."),
            SimpleNamespace(status_code=503, json=lambda: {}),
            SimpleNamespace(status_code=200, json=bad_json),
        ],
    )
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    with pytest.raises(ProviderRetryableError):
        client._request_validated("fnlttSinglAcntAll.json", {"corp_code": "00126380"})

    assert len(gets) == 3
    assert acquire.acquire_calls == 3
    assert attempts.attempt_calls == 3


def test_request_validated_quota_status_is_not_retried(tmp_path, monkeypatch) -> None:
    import pytest

    from src.integrations.dart.client import ProviderQuotaExhaustedError

    client, _store, gets, acquire, attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[_ok_response("020")]
    )

    with pytest.raises(ProviderQuotaExhaustedError, match="020"):
        client._request_validated("list.json", {"bgn_de": "20240101", "end_de": "20240101"})

    assert len(gets) == 1
    assert acquire.acquire_calls == 1
    assert attempts.attempt_calls == 1


def test_blocked_ledger_sends_nothing(tmp_path, monkeypatch) -> None:
    import pytest

    from src.integrations.quota import ProviderQuotaExhaustedError

    client, _store, gets, _acquire, _attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[_ok_response("000")], daily_request_limit=1
    )

    client._request("list.json", {"bgn_de": "20240101", "end_de": "20240101"})
    assert len(gets) == 1
    with pytest.raises(ProviderQuotaExhaustedError):
        client._request("list.json", {"bgn_de": "20240102", "end_de": "20240102"})
    assert len(gets) == 1


def test_request_json_seam_bypasses_http() -> None:
    from types import SimpleNamespace

    from src.integrations.dart.client import DartApiClient

    def no_http(*_args, **_kwargs):
        raise AssertionError("HTTP must not be used by the injected seam")

    client = DartApiClient(api_key="key", quota_provider="OpenDART", request_json=lambda _e, _p: {"status": "000", "list": []}, min_interval=0.0)
    client._session = SimpleNamespace(get=no_http)

    assert client._request("list.json", {"bgn_de": "20240101"}) == {"status": "000", "list": []}


def test_http_get_adds_api_key_and_records_single_attempt(tmp_path) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from src.integrations.dart.client import DartApiClient
    from src.integrations.quota import ProviderQuotaStateStore

    now = datetime(2026, 9, 13, tzinfo=UTC)
    seen: dict[str, object] = {}
    store = ProviderQuotaStateStore(tmp_path)
    client = DartApiClient(api_key="key", quota_provider="OpenDART", quota_store=store, now=lambda: now, min_interval=0.0)

    def fake_get(_url, params=None, **_kwargs):
        seen.update(dict(params or {}))
        return SimpleNamespace(status_code=200, content=b"{}")

    client._session = SimpleNamespace(get=fake_get)
    before = store.remaining_daily_attempts(provider="OpenDART", now=now, daily_limit=15_200)

    response = client._http_get("document.xml", {"rcept_no": "20240101000001"})

    assert response.status_code == 200
    assert seen["crtfc_key"] == "key"
    assert seen["rcept_no"] == "20240101000001"
    assert before - store.remaining_daily_attempts(provider="OpenDART", now=now, daily_limit=15_200) == 1


def test_request_terminal_status_is_not_retried(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from src.integrations.dart.client import ProviderTerminalError

    client, _store, gets, _acquire, attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[SimpleNamespace(status_code=404, json=lambda: {})]
    )

    with pytest.raises(ProviderTerminalError, match="404"):
        client._request("list.json", {"bgn_de": "20240101", "end_de": "20240101"})

    assert len(gets) == 1
    assert attempts.attempt_calls == 1


def test_request_non_object_json_payload_rejected(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from src.integrations.dart.client import ProviderTerminalError

    client, _store, gets, _acquire, _attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[SimpleNamespace(status_code=200, json=lambda: ["not", "an", "object"])]
    )

    with pytest.raises(ProviderTerminalError, match="must be an object"):
        client._request("list.json", {"bgn_de": "20240101", "end_de": "20240101"})

    assert len(gets) == 1


def test_request_seams_reject_non_object_payload() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient, ProviderTerminalError

    with pytest.raises(ProviderTerminalError, match="must be an object"):
        DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=0.0, raw_request_json=lambda _e, _p: ["nope"])._request(
            "list.json", {"bgn_de": "20240101"}
        )
    with pytest.raises(ProviderTerminalError, match="must be an object"):
        DartApiClient(api_key="key", quota_provider="OpenDART", min_interval=0.0, request_json=lambda _e, _p: ["nope"])._request(
            "list.json", {"bgn_de": "20240101"}
        )


def test_fetch_document_archive_retries_retryable_status(tmp_path, monkeypatch) -> None:
    import io
    import zipfile
    from types import SimpleNamespace

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("document.xml", "<doc/>")
    payload = buf.getvalue()

    client, _store, gets, _acquire, attempts, _now = _ledgered_client(
        tmp_path,
        monkeypatch,
        responses=[
            SimpleNamespace(status_code=503, json=lambda: {}),
            SimpleNamespace(status_code=200, content=payload),
        ],
    )
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    assert client.fetch_document_archive("20240101000001") == payload
    assert len(gets) == 2
    assert attempts.attempt_calls == 2


def test_fetch_document_archive_empty_body_fails_closed(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from src.integrations.dart.client import ProviderTerminalError

    client, _store, gets, _acquire, _attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[SimpleNamespace(status_code=200, content=b"")]
    )
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    with pytest.raises(ProviderTerminalError, match="empty"):
        client.fetch_document_archive("20240101000001")

    assert len(gets) == 3


def test_fetch_document_archive_error_payload_rejected(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from src.integrations.dart.client import ProviderTerminalError

    client, _store, gets, _acquire, _attempts, _now = _ledgered_client(
        tmp_path,
        monkeypatch,
        responses=[SimpleNamespace(status_code=200, content=b'{"status": "013"}')],
    )

    with pytest.raises(ProviderTerminalError, match="error payload"):
        client.fetch_document_archive("20240101000001")

    assert len(gets) == 1


def test_quota_provider_is_per_key_and_never_embeds_the_secret() -> None:
    from src.integrations.dart.client import dart_ledger_for_key, dart_quota_provider

    assert dart_quota_provider("primary-key", primary_api_key="primary-key") == "OpenDART"
    assert dart_quota_provider(None) == "OpenDART"
    secondary = dart_quota_provider("second-key", primary_api_key="primary-key")
    assert secondary.startswith("OpenDART#")
    assert len(secondary) == len("OpenDART#") + 8
    assert "second-key" not in secondary
    assert secondary == dart_quota_provider("second-key", primary_api_key="primary-key")
    assert secondary != dart_quota_provider("third-key", primary_api_key="primary-key")
    assert (
        dart_ledger_for_key(key_env="OPENDART_API_KEY_2", primary_key_env="OPENDART_API_KEY", api_key="second-key")
        == secondary
    )


def test_secondary_key_requests_are_metered_in_their_own_ledger(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.integrations.dart.client import DartApiClient, dart_quota_provider
    from src.integrations.quota import ProviderQuotaStateStore

    now = datetime(2026, 9, 24, 3, tzinfo=UTC)
    store = ProviderQuotaStateStore(tmp_path)

    class _Session:
        def get(self, *_args: object, **_kwargs: object) -> object:
            class _Response:
                status_code = 200

                @staticmethod
                def json() -> dict[str, object]:
                    return {"status": "000"}

            return _Response()

    client = DartApiClient(api_key="second-key", quota_store=store, now=lambda: now, min_interval=0.0)
    client._session = _Session()  # type: ignore[assignment]
    client._request("list.json", {})

    assert store.remaining_daily_attempts(provider="OpenDART", now=now, daily_limit=100) == 100
    assert store.remaining_daily_attempts(provider=dart_quota_provider("second-key"), now=now, daily_limit=100) == 99


def test_ping_is_ledgered_and_uses_company_endpoint(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    from src.integrations.dart.client import DartApiClient
    from src.integrations.quota import ProviderQuotaStateStore

    now = datetime(2026, 9, 24, 3, tzinfo=UTC)
    store = ProviderQuotaStateStore(tmp_path)
    seen: list[str] = []

    class _Session:
        def get(self, url: str, **_kwargs: object) -> object:
            seen.append(url)

            class _Response:
                status_code = 200

                @staticmethod
                def json() -> dict[str, object]:
                    return {"status": "000"}

            return _Response()

    client = DartApiClient(api_key="key", quota_provider="OpenDART", quota_store=store, now=lambda: now, min_interval=0.0)
    client._session = _Session()  # type: ignore[assignment]
    client.ping()

    assert seen == ["https://opendart.fss.or.kr/api/company.json"]
    assert store.remaining_daily_attempts(provider="OpenDART", now=now, daily_limit=10) == 9


def test_dart_client_requires_api_key_without_a_seam() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient

    with pytest.raises(ValueError, match="api_key is required"):
        DartApiClient(api_key="", min_interval=0.0)

def test_dart_client_transport_module_is_importable() -> None:
    from datetime import date

    from src.integrations.dart.client import DartApiClient, DartApiError

    calls: list[tuple[str, dict[str, str]]] = []

    def request(endpoint: str, params: dict[str, str]) -> dict[str, object]:
        calls.append((endpoint, params))
        return {
            "status": "000",
            "list": [{"rcept_no": "202601020001", "rcept_dt": "20260102"}],
        }

    client = DartApiClient(api_key="key", request_json=request, min_interval=0.0)
    records = client.list_disclosures(start=date(2026, 1, 2), end=date(2026, 1, 2))

    assert records == [
        {
            "rcept_no": "202601020001",
            "rcept_dt": "20260102",
            "corp_code": "",
            "corp_name": "",
            "report_nm": "",
            "rm": "",
        }
    ]
    assert calls[0][0] == "list.json"
    assert calls[0][1]["crtfc_key"] == "key"
    assert issubclass(DartApiError, RuntimeError)


def test_list_disclosures_rejects_reversed_window() -> None:
    from datetime import date

    import pytest

    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="key", min_interval=0.0, request_json=lambda e, p: {"status": "000", "list": []})
    with pytest.raises(ValueError, match="start must not be after end"):
        client.list_disclosures(start=date(2026, 2, 1), end=date(2026, 1, 1))


def test_fetch_dividend_disclosures_rejects_empty_corp_codes() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="key", min_interval=0.0, request_json=lambda e, p: {"status": "000", "list": []})
    with pytest.raises(ValueError, match="corp_codes must not be empty"):
        client.fetch_dividend_disclosures(corp_codes=["  "], bsns_years=["2024"])


def test_list_disclosures_rejects_non_object_seam_payload() -> None:
    import pytest
    from datetime import date

    from src.integrations.dart.client import DartApiClient, ProviderTerminalError

    client = DartApiClient(
        api_key="key", quota_provider="OpenDART", min_interval=0.0,
        request_json=lambda _e, _p: ["not", "an", "object"],
    )

    with pytest.raises(ProviderTerminalError, match="must be an object"):
        client.list_disclosures(date(2024, 1, 1), date(2024, 1, 2))


def test_fetch_document_archive_rejects_empty_seam_payload() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient, ProviderTerminalError

    client = DartApiClient(
        api_key="key", quota_provider="OpenDART", min_interval=0.0,
        request_bytes=lambda _e, _p: b"",
    )

    with pytest.raises(ProviderTerminalError, match="archive is empty"):
        client.fetch_document_archive("20240101000001")
