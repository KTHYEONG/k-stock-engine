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


def _hedge_handler(etf_rows, index_rows):  # type: ignore[no-untyped-def]
    def _handle(endpoint, params):  # type: ignore[no-untyped-def]
        if endpoint == "etp/etf_bydd_trd":
            return _FakeResponse({"OutBlock_1": [dict(row) for row in etf_rows]})
        if endpoint == "idx/kosdaq_dd_trd":
            return _FakeResponse({"OutBlock_1": [dict(row) for row in index_rows]})
        raise AssertionError(f"unexpected endpoint {endpoint}")

    return _handle


def _hedge_etf_row(**overrides):  # type: ignore[no-untyped-def]
    row = {"ISU_CD": "251340", "ISU_SRT_CD": "251340", "BAS_DD": "20260105", "TDD_CLSPRC": "5000"}
    row.update(overrides)
    return row


def _hedge_index_row(**overrides):  # type: ignore[no-untyped-def]
    row = {"IDX_CLSS": "KOSDAQ", "IDX_NM": "코스닥 150", "BAS_DD": "20260105", "CLSPRC_IDX": "1234.56"}
    row.update(overrides)
    return row


def test_hedge_records_return_tagged_rows() -> None:
    etf_rows = [_hedge_etf_row(), {"ISU_CD": "999999", "BAS_DD": "20260105", "TDD_CLSPRC": "1"}]
    index_rows = [_hedge_index_row(), {"IDX_CLSS": "KOSDAQ", "IDX_NM": "코스닥 소형", "BAS_DD": "20260105"}]
    client = _client(_hedge_handler(etf_rows, index_rows))

    records = client.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")

    assert len(records) == 2
    by_tag = {record["_endpoint"]: record for record in records}
    assert set(by_tag) == {"etf", "index"}
    assert by_tag["etf"]["ISU_CD"] == "251340"
    assert by_tag["index"]["IDX_NM"] == "코스닥 150"
    assert by_tag["etf"]["TDD_CLSPRC"] == "5000"
    assert by_tag["index"]["CLSPRC_IDX"] == "1234.56"


def test_hedge_records_tolerate_pre_listing() -> None:
    other_etf = {"ISU_CD": "999999", "ISU_SRT_CD": "999999", "BAS_DD": "20260105", "TDD_CLSPRC": "7000"}
    client = _client(_hedge_handler([other_etf], [_hedge_index_row()]))

    records = client.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")

    assert len(records) == 1
    assert records[0]["_endpoint"] == "index"


def test_hedge_records_reject_incomplete_page() -> None:
    from src.integrations.errors import ProviderTerminalError

    only_etf = _client(_hedge_handler([_hedge_etf_row()], []))
    with pytest.raises(ProviderTerminalError):
        only_etf.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")
    only_index = _client(_hedge_handler([], [_hedge_index_row()]))
    with pytest.raises(ProviderTerminalError):
        only_index.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")
    both_empty = _client(_hedge_handler([], []))
    assert both_empty.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150") == []


def test_hedge_records_reject_date_mismatch_and_duplicates() -> None:
    from src.integrations.errors import ProviderTerminalError

    bad_date = _client(_hedge_handler([_hedge_etf_row()], [_hedge_index_row(BAS_DD="20260106")]))
    with pytest.raises(ProviderTerminalError):
        bad_date.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")
    duplicated = _client(_hedge_handler([_hedge_etf_row()], [_hedge_index_row(), _hedge_index_row()]))
    with pytest.raises(ProviderTerminalError):
        duplicated.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")
    dup_etf = _client(_hedge_handler([_hedge_etf_row(), _hedge_etf_row()], [_hedge_index_row()]))
    with pytest.raises(ProviderTerminalError):
        dup_etf.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")
    non_positive = _client(_hedge_handler([_hedge_etf_row(TDD_CLSPRC="0")], [_hedge_index_row()]))
    with pytest.raises(ProviderTerminalError):
        non_positive.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")


def test_hedge_records_holiday_passthrough() -> None:
    from src.integrations.krx.client import KrxHolidayError

    payload = {"OutBlock_1": [], "RESULT": {"MESSAGE": "휴장일로 거래가 없습니다"}}
    client = _client(lambda endpoint, params: _FakeResponse(payload))

    with pytest.raises(KrxHolidayError):
        client.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")


def test_hedge_records_reject_invalid_arguments() -> None:
    client = _client(_hedge_handler([_hedge_etf_row()], [_hedge_index_row()]))

    with pytest.raises(ValueError, match="as_of"):
        client.fetch_hedge_records("2026-01-05", etf_tickers=("251340",), index_name="코스닥 150")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="index_name"):
        client.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="  ")
    with pytest.raises(ValueError, match="etf_tickers"):
        client.fetch_hedge_records(SESSION, etf_tickers=("  ",), index_name="코스닥 150")


def test_hedge_records_reject_malformed_bas_dd() -> None:
    from src.integrations.errors import ProviderTerminalError

    malformed = _client(_hedge_handler([_hedge_etf_row()], [_hedge_index_row(BAS_DD="2026-01-05")]))
    with pytest.raises(ProviderTerminalError):
        malformed.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")
    impossible = _client(_hedge_handler([_hedge_etf_row()], [_hedge_index_row(BAS_DD="20261301")]))
    with pytest.raises(ProviderTerminalError):
        impossible.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")


def test_hedge_records_reject_bad_prices() -> None:
    from src.integrations.errors import ProviderTerminalError

    bad_index = _client(_hedge_handler([_hedge_etf_row()], [_hedge_index_row(CLSPRC_IDX="abc")]))
    with pytest.raises(ProviderTerminalError):
        bad_index.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")
    zero_index = _client(_hedge_handler([_hedge_etf_row()], [_hedge_index_row(CLSPRC_IDX="0")]))
    with pytest.raises(ProviderTerminalError):
        zero_index.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")
    bad_etf = _client(_hedge_handler([_hedge_etf_row(TDD_CLSPRC="abc")], [_hedge_index_row()]))
    with pytest.raises(ProviderTerminalError):
        bad_etf.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")


def _kospi_handler(etf_rows, index_rows):  # type: ignore[no-untyped-def]
    def _handle(endpoint, params):  # type: ignore[no-untyped-def]
        if endpoint == "etp/etf_bydd_trd":
            return _FakeResponse({"OutBlock_1": [dict(row) for row in etf_rows]})
        if endpoint == "idx/kospi_dd_trd":
            return _FakeResponse({"OutBlock_1": [dict(row) for row in index_rows]})
        raise AssertionError(f"unexpected endpoint {endpoint}")

    return _handle


def _kospi_index_row(**overrides):  # type: ignore[no-untyped-def]
    row = {"IDX_CLSS": "KOSPI", "IDX_NM": "코스피 200", "BAS_DD": "20260105", "CLSPRC_IDX": "1109.05"}
    row.update(overrides)
    return row


def test_hedge_records_kospi_class_reads_kospi_page() -> None:
    client = _client(_kospi_handler([_hedge_etf_row(ISU_CD="114800")], [_kospi_index_row(IDX_CLSS=" KOSPI ")]))

    records = client.fetch_hedge_records(
        SESSION, etf_tickers=("114800",), index_name="코스피 200", index_class="KOSPI"
    )

    assert len(records) == 2
    by_tag = {record["_endpoint"]: record for record in records}
    assert by_tag["index"]["CLSPRC_IDX"] == "1109.05"
    assert [call[0] for call in client._transport.calls] == ["etp/etf_bydd_trd", "idx/kospi_dd_trd"]
    assert "idx/kosdaq_dd_trd" not in [call[0] for call in client._transport.calls]


def test_hedge_records_default_class_is_unchanged() -> None:
    etf_rows = [_hedge_etf_row()]
    index_rows = [_hedge_index_row()]
    client = _client(_hedge_handler(etf_rows, index_rows))

    records = client.fetch_hedge_records(SESSION, etf_tickers=("251340",), index_name="코스닥 150")

    assert [call[0] for call in client._transport.calls] == ["etp/etf_bydd_trd", "idx/kosdaq_dd_trd"]
    assert len(records) == 2


def test_hedge_records_unknown_class_rejected_before_request() -> None:
    client = _client(_hedge_handler([_hedge_etf_row()], [_hedge_index_row()]))

    with pytest.raises(ValueError, match="index_class"):
        client.fetch_hedge_records(
            SESSION, etf_tickers=("251340",), index_name="코스닥 150", index_class="KRX"
        )

    assert client._transport.calls == []


@pytest.mark.parametrize(
    "row", [_kospi_index_row(IDX_NM="코스피 100"), _kospi_index_row(IDX_CLSS="KOSDAQ"),
            _kospi_index_row(IDX_NM="코스피 200 ")],
)
def test_hedge_records_missing_kospi_row_fails_closed(row) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.errors import ProviderTerminalError

    client = _client(_kospi_handler([_hedge_etf_row(ISU_CD="114800")], [row]))

    with pytest.raises(ProviderTerminalError):
        client.fetch_hedge_records(
            SESSION, etf_tickers=("114800",), index_name="코스피 200", index_class="KOSPI"
        )
