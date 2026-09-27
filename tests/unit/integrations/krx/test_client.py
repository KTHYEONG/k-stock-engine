"""KRX client invariants: validation names the session, quota comes from the transport."""
from __future__ import annotations

from datetime import date

import pytest

SESSION = date(2026, 1, 5)


class _FakeResponse:
    def __init__(self, payload=None, *, json_error=False):  # type: ignore[no-untyped-def]
        self._payload = payload
        self._json_error = json_error

    def json(self):  # type: ignore[no-untyped-def]
        if self._json_error:
            raise ValueError("not json")
        return self._payload


class _FakeTransport:
    """Scripted stand-in for HttpTransport.get."""

    def __init__(self, handler):  # type: ignore[no-untyped-def]
        self._handler = handler
        self.calls: list[tuple] = []

    def get(self, endpoint, params, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
        self.calls.append((endpoint, dict(params), dict(headers or {})))
        response = self._handler(endpoint, dict(params))
        if classify is not None:
            classify(response)
        return response


def _client(handler, **kwargs):  # type: ignore[no-untyped-def]
    from src.integrations.krx.client import KrxApiClient

    return KrxApiClient("test-key", transport=_FakeTransport(handler), **kwargs)


def _daily_row(**overrides):  # type: ignore[no-untyped-def]
    row = {"TDD_CLSPRC": "1000", "MKTCAP": "5000", "LIST_SHRS": "100"}
    row.update(overrides)
    return row


def test_daily_records_missing_close_raises_naming_session() -> None:
    from src.core.pit import PITDataError

    client = _client(lambda endpoint, params: _FakeResponse({"OutBlock_1": [_daily_row(TDD_CLSPRC="")]}))

    with pytest.raises(PITDataError, match=SESSION.isoformat()):
        client.fetch_daily_records(SESSION)


def test_daily_records_missing_capitalization_raises() -> None:
    from src.core.pit import PITDataError

    client = _client(lambda endpoint, params: _FakeResponse({"OutBlock_1": [_daily_row(MKTCAP=None)]}))

    with pytest.raises(PITDataError, match="MKTCAP"):
        client.fetch_daily_records(SESSION)


def test_daily_records_roundtrip_valid_rows() -> None:
    client = _client(lambda endpoint, params: _FakeResponse({"OutBlock_1": [_daily_row()]}))

    records = client.fetch_daily_records(SESSION)

    assert len(records) == 2
    assert all(row["TDD_CLSPRC"] == "1000" for row in records)


def test_quota_error_propagates_without_internal_counter() -> None:
    from src.integrations.errors import ProviderQuotaExhaustedError

    def _raise(endpoint, params):  # type: ignore[no-untyped-def]
        raise ProviderQuotaExhaustedError("KRX daily quota safety limit reached")

    client = _client(_raise)

    with pytest.raises(ProviderQuotaExhaustedError):
        client.fetch_daily_records(SESSION)
    assert not hasattr(client, "_request_count")


def test_empty_answer_returns_empty_without_validation() -> None:
    client = _client(lambda endpoint, params: _FakeResponse({"OutBlock_1": []}))

    assert client.fetch_daily_records(SESSION) == []
    assert client.fetch_master_records(SESSION) == []


def test_holiday_report_raises_for_review_naming_session() -> None:
    from src.integrations.krx.client import KrxHolidayError

    payload = {"OutBlock_1": [], "RESULT": {"MESSAGE": "휴장일로 거래가 없습니다"}}
    client = _client(lambda endpoint, params: _FakeResponse(payload))

    with pytest.raises(KrxHolidayError, match=SESSION.isoformat()):
        client.fetch_daily_records(SESSION)


def test_master_records_require_identity_naming_session() -> None:
    from src.core.pit import PITDataError

    client = _client(
        lambda endpoint, params: _FakeResponse({"OutBlock_1": [{"ISU_SRT_CD": "005930", "ISU_CD": ""}]})
    )

    with pytest.raises(PITDataError, match=SESSION.isoformat()):
        client.fetch_master_records(SESSION)


def test_master_records_roundtrip() -> None:
    client = _client(
        lambda endpoint, params: _FakeResponse(
            {"OutBlock_1": [{"ISU_SRT_CD": "005930", "ISU_CD": "KR7005930003"}]}
        )
    )

    records = client.fetch_master_records(SESSION, market="KOSPI")

    assert [row["ISU_SRT_CD"] for row in records] == ["005930"]


def test_single_market_hits_one_endpoint() -> None:
    client = _client(lambda endpoint, params: _FakeResponse({"OutBlock_1": [_daily_row()]}))

    client.fetch_daily_records(SESSION, market="KOSDAQ")

    assert [call[0] for call in client._transport.calls] == ["sto/ksq_bydd_trd"]


def test_transport_failures_classify() -> None:
    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError

    retryable = _client(lambda endpoint, params: _FakeResponse(json_error=True))
    with pytest.raises(ProviderRetryableError):
        retryable.fetch_daily_records(SESSION)

    non_object = _client(lambda endpoint, params: _FakeResponse([1, 2]))
    with pytest.raises(ProviderTerminalError):
        non_object.fetch_daily_records(SESSION)

    non_list = _client(lambda endpoint, params: _FakeResponse({"OutBlock_1": {}}))
    with pytest.raises(ProviderTerminalError):
        non_list.fetch_daily_records(SESSION)


def test_invalid_arguments_fail_closed() -> None:
    client = _client(lambda endpoint, params: _FakeResponse({"OutBlock_1": []}))

    with pytest.raises(ValueError, match="as_of"):
        client.fetch_daily_records("2026-01-05")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown KRX market"):
        client.fetch_daily_records(SESSION, market="NYSE")
    from src.integrations.krx.client import KrxApiClient

    with pytest.raises(ValueError, match="api_key"):
        KrxApiClient("   ", transport=_FakeTransport(lambda e, p: _FakeResponse({})))


def test_default_transport_construction_issues_no_request(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.krx.client import KrxApiClient
    from src.integrations.quota import ProviderQuotaStateStore

    metered = KrxApiClient(
        "key",
        quota_store=ProviderQuotaStateStore(tmp_path / "quota"),
        daily_limit=10,
        min_interval_seconds=0,
    )
    unmetered = KrxApiClient("key", min_interval_seconds=0)
    assert metered.health_check() is None
    assert unmetered.health_check() is None


def test_scoped_builder_reads_declared_key_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config
    from src.integrations.krx.client import build_scoped_krx_client
    from src.integrations.quota import ProviderQuotaStateStore

    provider = load_provider_policy(load_runtime_config())
    monkeypatch.setenv(provider.krx.api_key_env, "scoped-key")
    client = build_scoped_krx_client(
        policy=provider.krx, quota_store=ProviderQuotaStateStore(tmp_path / "quota")
    )
    assert client._api_key == "scoped-key"

    monkeypatch.setenv(provider.krx.api_key_env, "''")
    with pytest.raises(ValueError, match=provider.krx.api_key_env):
        build_scoped_krx_client(
            policy=provider.krx, quota_store=ProviderQuotaStateStore(tmp_path / "quota")
        )


def test_non_dict_rows_are_filtered() -> None:
    client = _client(
        lambda endpoint, params: _FakeResponse({"OutBlock_1": [_daily_row(), "junk", None]})
    )

    assert len(client.fetch_daily_records(SESSION, market="KOSPI")) == 1


def test_master_kosdaq_endpoint() -> None:
    client = _client(
        lambda endpoint, params: _FakeResponse(
            {"OutBlock_1": [{"ISU_SRT_CD": "000660", "ISU_CD": "KR7000660001"}]}
        )
    )

    client.fetch_master_records(SESSION, market="KOSDAQ")

    assert [call[0] for call in client._transport.calls] == ["sto/ksq_isu_base_info"]


def test_daily_records_missing_listed_shares_raises() -> None:
    from src.core.pit import PITDataError

    client = _client(lambda endpoint, params: _FakeResponse({"OutBlock_1": [_daily_row(LIST_SHRS="")]}))

    with pytest.raises(PITDataError, match="LIST_SHRS"):
        client.fetch_daily_records(SESSION)


def test_master_records_reject_non_date_session() -> None:
    client = _client(lambda endpoint, params: _FakeResponse({"OutBlock_1": []}))

    with pytest.raises(ValueError, match="as_of"):
        client.fetch_master_records("2026-01-05")  # type: ignore[arg-type]
