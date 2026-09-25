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
        api_key='key',
        request_json=lambda endpoint, params: {'status': '020', 'message': 'blocked'},
    )

    with pytest.raises(DartApiError, match='020'):
        client.list_disclosures(start=date(2026, 1, 1), end=date(2026, 1, 2))


def test_list_disclosures_collects_all_pages_deduplicates_and_orders() -> None:
    from datetime import date
    from src.integrations.dart.client import DartApiClient

    pages = {"1": {"status": "000", "total_page": "2", "list": [{"rcept_no": "20150515000002", "rcept_dt": "20150515", "corp_code": "001", "corp_name": "A", "report_nm": "분기보고서 (2015.03)", "rm": ""}]}, "2": {"status": "000", "total_page": "2", "list": [{"rcept_no": "20150515000001", "rcept_dt": "20150515", "corp_code": "001", "corp_name": "A", "report_nm": "분기보고서 (2015.03)", "rm": ""}]}}
    client = DartApiClient(api_key="key", request_json=lambda _endpoint, p: pages[p["page_no"]])

    rows = client.list_disclosures(date(2015, 1, 1), date(2015, 12, 31))

    assert [row["rcept_no"] for row in rows] == ["20150515000001", "20150515000002"]


# test_dart_client_preserves_success_and_no_data_pages
def test_fetch_corporate_action_decisions_preserves_success_and_no_data_pages(monkeypatch) -> None:
    from datetime import date

    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key='test-key')
    responses = iter([{'status': '000', 'list': [{'rcept_no': '20240102000001'}]}, {'status': '013', 'list': []}] * 3)
    monkeypatch.setattr(client, '_request_validated', lambda endpoint, params: next(responses))

    pages = client.fetch_corporate_action_decisions(corp_codes=['00123456'], start=date(2024, 1, 1), end=date(2024, 1, 31))

    assert len(pages) == 5
    assert {page.status for page in pages} == {'000', '013'}
    assert pages[0].records == ({'rcept_no': '20240102000001'},)


def test_fetch_corporate_action_decisions_rejects_invalid_arguments() -> None:
    from datetime import date
    import pytest
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="test-key")
    with pytest.raises(ValueError, match="start"):
        client.fetch_corporate_action_decisions(corp_codes=("1",), start=date(2024, 2, 1), end=date(2024, 1, 1))
    with pytest.raises(ValueError, match="corp_codes"):
        client.fetch_corporate_action_decisions(corp_codes=(), start=date(2024, 1, 1), end=date(2024, 1, 2))


def test_fetch_dividend_disclosures_queries_every_report_code(monkeypatch) -> None:
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="test-key")
    seen: list[dict[str, str]] = []

    def fake_request_validated(endpoint, params):
        seen.append(dict(params))
        if params["reprt_code"] == "11011":
            return {"status": "000", "list": [{"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444"}]}
        return {"status": "013", "list": []}

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

    client = DartApiClient(api_key="test-key")
    with pytest.raises(ValueError, match="corp_codes"):
        client.fetch_dividend_disclosures(corp_codes=(), bsns_years=("2022",))
    with pytest.raises(ValueError, match="bsns_years"):
        client.fetch_dividend_disclosures(corp_codes=("00126380",), bsns_years=())


from datetime import date, datetime

from src.core.time import KRX_TZ, SessionCalendar
from src.data.lifecycle import LifecycleCandidate, LifecycleResolutionKind
from src.integrations.dart.lifecycle import DartLifecycleCollector

class FakeDart:
    def load_corp_code_records(self):
        return ()

def test_dart_lifecycle_collector_persists_unresolved_mapping_gap():
    session = datetime(2016, 1, 21, 9, tzinfo=KRX_TZ)
    candidate = LifecycleCandidate('KRX:003945', '003945', datetime(2016, 1, 20, 9, tzinfo=KRX_TZ), session, ('master-hash',))
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar((candidate.last_tradable_session, session)), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.resolution_kind is LifecycleResolutionKind.UNRESOLVED
    assert evidence.evidence_reason == 'missing_corp_code'
    assert evidence.cash_settlement_per_share is None


def test_dart_lifecycle_collector_resolves_cleanup_cash_settlement():
    from datetime import date, datetime
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.lifecycle import LifecycleCandidate
    from src.integrations.dart.lifecycle import DartLifecycleCollector
    from src.data.lifecycle import LifecycleResolutionKind
    sessions = (datetime(2016, 5, 3, 9, tzinfo=KRX_TZ), datetime(2016, 5, 10, 9, tzinfo=KRX_TZ), datetime(2016, 5, 18, 9, tzinfo=KRX_TZ), datetime(2016, 5, 19, 9, tzinfo=KRX_TZ))
    candidate = LifecycleCandidate("KRX:008020", "008020", sessions[2], sessions[3], ("master-hash",))
    archive = "정리매매 기간은 2016.05.10.부터 2016.05.18.까지이며 2016.05.19.자로 상장폐지. 주당 10,200원.".encode()
    class FakeDart:
        def load_corp_code_records(self):
            from src.integrations.dart.client import DartCorpCodeRecord
            return (DartCorpCodeRecord(ticker="008020", corp_code="00101336", corp_name="경남에너지"),)
        def list_disclosures(self, start, end, *, corp_code=None):
            assert corp_code == "00101336"
            return [{"rcept_no": "20160502800560", "rcept_dt": "20160502", "corp_code": corp_code, "corp_name": "경남에너지", "report_nm": "기타경영사항(자율공시)(상장폐지결정 및 정리매매)", "rm": "유"}]
        def fetch_document_archive(self, rcept_no):
            assert rcept_no == "20160502800560"
            return archive
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar(sessions), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.resolution_kind is LifecycleResolutionKind.CASH_SETTLEMENT
    assert evidence.cash_settlement_per_share == 10200.0
    assert evidence.document_receipt_no == "20160502800560"


def test_dart_lifecycle_collector_unresolved_when_no_lifecycle_filings():
    from datetime import date, datetime
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.lifecycle import LifecycleCandidate, LifecycleResolutionKind
    from src.integrations.dart.lifecycle import DartLifecycleCollector
    session = datetime(2016, 4, 15, 9, tzinfo=KRX_TZ)
    candidate = LifecycleCandidate("KRX:051310", "051310", datetime(2016, 4, 14, 9, tzinfo=KRX_TZ), session, ("h",))
    class FakeDart:
        def load_corp_code_records(self):
            from src.integrations.dart.client import DartCorpCodeRecord
            return (DartCorpCodeRecord(ticker="051310", corp_code="00181934", corp_name="플랜텍"),)
        def list_disclosures(self, start, end, *, corp_code=None):
            return [{"rcept_no": "20160128800671", "rcept_dt": "20160128", "corp_code": corp_code, "corp_name": "플랜텍", "report_nm": "분기보고서", "rm": "유"}]
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar((datetime(2016, 4, 14, 9, tzinfo=KRX_TZ), session)), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.resolution_kind is LifecycleResolutionKind.UNRESOLVED
    assert evidence.evidence_reason == "missing_terms"


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

    client = DartApiClient(api_key="key")
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

    from src.integrations.dart.client import DartApiClient, DartRetryableError
    from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

    calls = {"n": 0}

    def always_fail(*_args, **_kwargs):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("Connection aborted.")

    client = DartApiClient(
        api_key="key",
        quota_store=ProviderQuotaStateStore(tmp_path),
        now=lambda: datetime(2026, 9, 13, tzinfo=UTC),
    )
    client._session = SimpleNamespace(get=always_fail)
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    with pytest.raises(DartRetryableError, match="transport failed"):
        client._request("fnlttSinglAcntAll.json", {"corp_code": "00126380"})
    assert calls["n"] == 3

    with pytest.raises(ProviderQuotaBlocked):
        client._request("fnlttSinglAcntAll.json", {"corp_code": "00126380"})
    assert calls["n"] == 3


def test_dart_client_circuit_breaks_on_quota_exceeded_status(tmp_path) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    import pytest

    from src.integrations.dart.client import DartApiClient, DartApiError
    from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

    calls: list[object] = []
    client = DartApiClient(
        api_key="key",
        quota_store=ProviderQuotaStateStore(tmp_path),
        now=lambda: datetime(2026, 9, 13, tzinfo=UTC),
    )
    client._session = SimpleNamespace(
        get=lambda *_a, **_kw: calls.append(object())
        or SimpleNamespace(status_code=200, json=lambda: {"status": "020", "message": "quota exceeded"})
    )

    with pytest.raises(DartApiError, match="020"):
        client._request_validated("list.json", {"bgn_de": "20240101", "end_de": "20240101"})
    with pytest.raises(ProviderQuotaBlocked):
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
    client = DartApiClient(api_key="key", min_interval=1.0)
    responses = iter([SimpleNamespace(status_code=200, json=lambda: {"status": "000", "list": []})] * 2)
    client._session = SimpleNamespace(get=lambda *_a, **_kw: next(responses))

    client._request("list.json", {"bgn_de": "20240101", "end_de": "20240101"})
    client._request("list.json", {"bgn_de": "20240102", "end_de": "20240102"})

    assert sleeps == [pytest.approx(0.7)]


def test_dart_client_min_interval_reads_env_var_and_defaults_to_disabled(monkeypatch) -> None:
    import pytest
    from src.integrations.dart.client import DartApiClient

    monkeypatch.setenv("OPENDART_REQUEST_MIN_INTERVAL_SECONDS", "2.5")
    client = DartApiClient(api_key="key")
    assert client._min_interval == 2.5

    monkeypatch.delenv("OPENDART_REQUEST_MIN_INTERVAL_SECONDS", raising=False)
    client_default = DartApiClient(api_key="key")
    assert client_default._min_interval == 0.0

    client_explicit = DartApiClient(api_key="key", min_interval=3.0)
    assert client_explicit._min_interval == 3.0
    with pytest.raises(ValueError, match="daily_request_limit"):
        DartApiClient(api_key="key", daily_request_limit=0)


def test_dart_client_passes_disclosure_detail_type() -> None:
    from src.integrations.dart.client import DartApiClient

    seen: dict[str, str] = {}

    def request(_endpoint: str, params: dict[str, str]) -> dict[str, object]:
        seen.update(params)
        return {"status": "000", "list": []}

    DartApiClient(api_key="key", request_json=request).list_disclosures(
        date(2024, 1, 1), date(2024, 1, 1), detail_type="A001"
    )

    assert seen["pblntf_detail_ty"] == "A001"


def test_request_validated_raises_distinguishable_quota_exhausted_type() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient, DartApiError, DartQuotaExhaustedError

    rate_limit_calls: list[dict[str, object]] = []

    class _QuotaStore:
        def record_rate_limit(self, **kwargs: object) -> None:
            rate_limit_calls.append(dict(kwargs))

    client = DartApiClient(
        api_key="key",
        raw_request_json=lambda _endpoint, _params: {"status": "020", "message": "quota exceeded"},
        quota_store=_QuotaStore(),  # type: ignore[arg-type]
    )

    # When/Then: distinguishable type that remains a DartApiError, with quota recorded once.
    with pytest.raises(DartQuotaExhaustedError, match="020") as exc_info:
        client._request_validated("list.json", {"bgn_de": "20240101", "end_de": "20240101"})
    assert isinstance(exc_info.value, DartApiError)
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

        def record_attempt(self, **kwargs):
            self.attempt_calls += 1
            return super().record_attempt(**kwargs)

    store = _CountingStore(tmp_path)
    client = DartApiClient(
        api_key="key",
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
    monkeypatch.setattr(client, "_pace", lambda: paces.append(None))

    records = client.load_corp_code_records()

    assert [(rec.ticker, rec.corp_code) for rec in records] == [("005930", "00126380")]
    assert len(gets) == 1
    assert attempts.attempt_calls == 1
    assert len(paces) == 1


def test_request_validated_status_900_shares_attempt_budget(tmp_path, monkeypatch) -> None:
    import pytest

    from src.integrations.dart.client import DartRetryableError

    client, _store, gets, acquire, attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[_ok_response("900")] * 3
    )
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    with pytest.raises(DartRetryableError, match="900"):
        client._request_validated("fnlttSinglAcntAll.json", {"corp_code": "00126380"})

    assert len(gets) == 3
    assert acquire.acquire_calls == 1
    assert attempts.attempt_calls == 3


def test_request_validated_status_900_then_success(tmp_path, monkeypatch) -> None:
    client, _store, gets, acquire, attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[_ok_response("900"), _ok_response("000")]
    )
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    payload = client._request_validated("fnlttSinglAcntAll.json", {"corp_code": "00126380"})

    assert payload["status"] == "000"
    assert len(gets) == 2
    assert acquire.acquire_calls == 1
    assert attempts.attempt_calls == 2


def test_request_mixed_causes_share_attempt_budget(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest
    import requests

    from src.integrations.dart.client import DartRetryableError

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

    with pytest.raises(DartRetryableError):
        client._request_validated("fnlttSinglAcntAll.json", {"corp_code": "00126380"})

    assert len(gets) == 3
    assert acquire.acquire_calls == 1
    assert attempts.attempt_calls == 3


def test_request_validated_quota_status_is_not_retried(tmp_path, monkeypatch) -> None:
    import pytest

    from src.integrations.dart.client import DartQuotaExhaustedError

    client, _store, gets, acquire, attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[_ok_response("020")]
    )

    with pytest.raises(DartQuotaExhaustedError, match="020"):
        client._request_validated("list.json", {"bgn_de": "20240101", "end_de": "20240101"})

    assert len(gets) == 1
    assert acquire.acquire_calls == 1
    assert attempts.attempt_calls == 1


def test_blocked_ledger_sends_nothing(tmp_path, monkeypatch) -> None:
    import pytest

    from src.integrations.quota import ProviderQuotaBlocked

    client, _store, gets, _acquire, _attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[_ok_response("000")], daily_request_limit=1
    )

    client._request("list.json", {"bgn_de": "20240101", "end_de": "20240101"})
    assert len(gets) == 1
    with pytest.raises(ProviderQuotaBlocked):
        client._request("list.json", {"bgn_de": "20240102", "end_de": "20240102"})
    assert len(gets) == 1


def test_request_json_seam_bypasses_http() -> None:
    from types import SimpleNamespace

    from src.integrations.dart.client import DartApiClient

    def no_http(*_args, **_kwargs):
        raise AssertionError("HTTP must not be used by the injected seam")

    client = DartApiClient(api_key="key", request_json=lambda _e, _p: {"status": "000", "list": []})
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
    client = DartApiClient(api_key="key", quota_store=store, now=lambda: now)

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

    from src.integrations.dart.client import DartTerminalError

    client, _store, gets, _acquire, attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[SimpleNamespace(status_code=404, json=lambda: {})]
    )

    with pytest.raises(DartTerminalError, match="404"):
        client._request("list.json", {"bgn_de": "20240101", "end_de": "20240101"})

    assert len(gets) == 1
    assert attempts.attempt_calls == 1


def test_request_non_object_json_payload_rejected(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from src.integrations.dart.client import DartTerminalError

    client, _store, gets, _acquire, _attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[SimpleNamespace(status_code=200, json=lambda: ["not", "an", "object"])]
    )

    with pytest.raises(DartTerminalError, match="must be an object"):
        client._request("list.json", {"bgn_de": "20240101", "end_de": "20240101"})

    assert len(gets) == 1


def test_request_seams_reject_non_object_payload() -> None:
    import pytest

    from src.integrations.dart.client import DartApiClient, DartTerminalError

    with pytest.raises(DartTerminalError, match="must be an object"):
        DartApiClient(api_key="key", raw_request_json=lambda _e, _p: ["nope"])._request(
            "list.json", {"bgn_de": "20240101"}
        )
    with pytest.raises(DartTerminalError, match="must be an object"):
        DartApiClient(api_key="key", request_json=lambda _e, _p: ["nope"])._request(
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

    from src.integrations.dart.client import DartTerminalError

    client, _store, gets, _acquire, _attempts, _now = _ledgered_client(
        tmp_path, monkeypatch, responses=[SimpleNamespace(status_code=200, content=b"")]
    )
    monkeypatch.setattr("src.integrations.dart.client.time.sleep", lambda _seconds: None)

    with pytest.raises(DartTerminalError, match="empty"):
        client.fetch_document_archive("20240101000001")

    assert len(gets) == 3


def test_fetch_document_archive_error_payload_rejected(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from src.integrations.dart.client import DartTerminalError

    client, _store, gets, _acquire, _attempts, _now = _ledgered_client(
        tmp_path,
        monkeypatch,
        responses=[SimpleNamespace(status_code=200, content=b'{"status": "013"}')],
    )

    with pytest.raises(DartTerminalError, match="error payload"):
        client.fetch_document_archive("20240101000001")

    assert len(gets) == 1


def test_quota_provider_is_per_key_and_never_embeds_the_secret(monkeypatch) -> None:
    from src.integrations.dart.client import dart_quota_provider

    monkeypatch.setenv("OPENDART_API_KEY", "primary-key")

    assert dart_quota_provider("primary-key") == "OpenDART"
    assert dart_quota_provider(None) == "OpenDART"
    secondary = dart_quota_provider("second-key")
    assert secondary.startswith("OpenDART#")
    assert len(secondary) == len("OpenDART#") + 8
    assert "second-key" not in secondary
    assert secondary == dart_quota_provider("second-key")
    assert secondary != dart_quota_provider("third-key")


def test_secondary_key_requests_are_metered_in_their_own_ledger(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    from src.integrations.dart.client import DartApiClient, dart_quota_provider
    from src.integrations.quota import ProviderQuotaStateStore

    monkeypatch.setenv("OPENDART_API_KEY", "primary-key")
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

    client = DartApiClient(api_key="second-key", quota_store=store, now=lambda: now)
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

    client = DartApiClient(api_key="key", quota_store=store, now=lambda: now)
    client._session = _Session()  # type: ignore[assignment]
    client.ping()

    assert seen == ["https://opendart.fss.or.kr/api/company.json"]
    assert store.remaining_daily_attempts(provider="OpenDART", now=now, daily_limit=10) == 9
