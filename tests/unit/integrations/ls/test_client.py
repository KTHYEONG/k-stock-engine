

def test_ls_client_waits_for_monotonic_request_slot() -> None:
    from src.integrations.ls.client import LS_MIN_REQUEST_INTERVAL_SECONDS, LsClient, LsCredentials

    clock = iter((100.0, 100.0))
    sleeps: list[float] = []
    client = LsClient(LsCredentials('key', 'secret'), session=object(), monotonic=lambda: next(clock), sleeper=sleeps.append)
    client._wait_for_request_slot()
    client._wait_for_request_slot()
    assert sleeps == [LS_MIN_REQUEST_INTERVAL_SECONDS]


def test_ls_client_retries_throttle_response_and_rejects_malformed_payload() -> None:
    from datetime import date
    import pytest
    from src.data.schemas import PITDataError
    from src.integrations.ls.client import LsClient, LsCredentials

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = 0

        def post(self, *_args, **_kwargs):
            result = self.responses[self.calls]
            self.calls += 1
            return result

    session = Session([Response({'rsp_cd': 'IGW00201'}), Response({'rsp_cd': '00000', 't1702OutBlock1': [{'date': '20161229'}]})])
    client = LsClient(LsCredentials('key', 'secret'), session=session, monotonic=lambda: 0.0, sleeper=lambda _delay: None)
    client._token = 'token'
    assert client.inquire_investor_trend('000020', date(2016, 12, 29), date(2016, 12, 29)) == ({'date': '20161229'},)
    assert session.calls == 2
    bad = LsClient(LsCredentials('key', 'secret'), session=Session([Response({'rsp_cd': '00000'})]), monotonic=lambda: 0.0, sleeper=lambda _delay: None)
    bad._token = 'token'
    with pytest.raises(PITDataError, match='t1702'):
        bad.inquire_investor_trend('000020', date(2016, 12, 29), date(2016, 12, 29))


def test_ls_client_skips_sleep_when_interval_elapsed() -> None:
    from src.integrations.ls.client import LsClient, LsCredentials

    clock = iter((100.0, 200.0))
    sleeps: list[float] = []
    LsClient._last_slot = None
    client = LsClient(
        LsCredentials("key", "secret"),
        session=object(),
        monotonic=lambda: next(clock),
        sleeper=sleeps.append,
    )
    client._wait_for_request_slot()
    client._wait_for_request_slot()
    assert sleeps == []


def test_ls_client_retries_http_429_and_rejects_errors() -> None:
    from datetime import date

    import pytest
    import requests

    from src.data.schemas import PITDataError
    from src.integrations.ls.client import LsClient, LsCredentials

    class HttpResponse:
        def __init__(self, payload=None, status=None, bad_json=False, error=None):
            self._payload = payload
            self.status_code = status
            self._bad_json = bad_json
            self._error = error

        def raise_for_status(self):
            if self._error is not None:
                raise self._error

        def json(self):
            if self._bad_json:
                raise ValueError("bad json")
            return self._payload

    def http_error(status):
        exc = requests.HTTPError(f"http {status}")
        exc.response = type("Resp", (), {"status_code": status})()
        return exc

    ok_payload = {"rsp_cd": "00000", "t1702OutBlock1": [{"date": "20161229"}]}

    session = type(
        "Session",
        (),
        {
            "calls": 0,
            "responses": [
                HttpResponse(error=http_error(429), status=429),
                HttpResponse(payload=ok_payload),
            ],
            "post": lambda self, *a, **k: (
                setattr(self, "calls", self.calls + 1) or self.responses[self.calls - 1]
            ),
        },
    )()
    client = LsClient(
        LsCredentials("key", "secret"),
        session=session,
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    client._token = "token"
    assert client.inquire_investor_trend(
        "000020", date(2016, 12, 29), date(2016, 12, 29)
    ) == ({"date": "20161229"},)

    bad_http = type(
        "Session",
        (),
        {
            "post": lambda self, *a, **k: HttpResponse(
                error=http_error(500), status=500
            ),
        },
    )()
    failing = LsClient(
        LsCredentials("key", "secret"),
        session=bad_http,
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    failing._token = "token"
    with pytest.raises(PITDataError, match="HTTP error"):
        failing.inquire_investor_trend(
            "000020", date(2016, 12, 29), date(2016, 12, 29)
        )

    bad_json_session = type(
        "Session",
        (),
        {"post": lambda self, *a, **k: HttpResponse(bad_json=True)},
    )()
    bad_json_client = LsClient(
        LsCredentials("key", "secret"),
        session=bad_json_session,
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    bad_json_client._token = "token"
    with pytest.raises(PITDataError, match="malformed JSON"):
        bad_json_client.inquire_investor_trend(
            "000020", date(2016, 12, 29), date(2016, 12, 29)
        )

    non_mapping_session = type(
        "Session",
        (),
        {"post": lambda self, *a, **k: HttpResponse(payload=["not-mapping"])},
    )()
    non_mapping_client = LsClient(
        LsCredentials("key", "secret"),
        session=non_mapping_session,
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    non_mapping_client._token = "token"
    with pytest.raises(PITDataError, match="non-mapping"):
        non_mapping_client.inquire_investor_trend(
            "000020", date(2016, 12, 29), date(2016, 12, 29)
        )

    denied_session = type(
        "Session",
        (),
        {
            "post": lambda self, *a, **k: HttpResponse(
                payload={"rsp_cd": "IGW99999"}
            )
        },
    )()
    denied_client = LsClient(
        LsCredentials("key", "secret"),
        session=denied_session,
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    denied_client._token = "token"
    with pytest.raises(PITDataError, match="non-success"):
        denied_client.inquire_investor_trend(
            "000020", date(2016, 12, 29), date(2016, 12, 29)
        )


def test_ls_client_covers_fallback_status_and_exhausted_throttle() -> None:
    from datetime import date

    import pytest
    import requests

    from src.data.schemas import PITDataError
    from src.integrations.ls.client import LsClient, LsCredentials

    class GenericFailResponse:
        status_code = 429

        def raise_for_status(self):
            raise requests.HTTPError("boom")

        def json(self):
            raise AssertionError("must not parse")

    generic_session = type(
        "Session",
        (),
        {"post": lambda self, *a, **k: GenericFailResponse()},
    )()
    generic_client = LsClient(
        LsCredentials("key", "secret"),
        session=generic_session,
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    generic_client._token = "token"
    with pytest.raises(PITDataError, match="HTTP error"):
        generic_client.inquire_investor_trend(
            "000020", date(2016, 12, 29), date(2016, 12, 29)
        )

    class ThrottledResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"rsp_cd": "IGW00201"}

    throttled_session = type(
        "Session",
        (),
        {
            "calls": 0,
            "post": lambda self, *a, **k: (
                setattr(self, "calls", self.calls + 1) or ThrottledResponse()
            ),
        },
    )()
    throttled_client = LsClient(
        LsCredentials("key", "secret"),
        session=throttled_session,
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    throttled_client._token = "token"
    with pytest.raises(PITDataError, match="throttled"):
        throttled_client.inquire_investor_trend(
            "000020", date(2016, 12, 29), date(2016, 12, 29)
        )
    assert throttled_session.calls == 3


def test_ls_client_close_is_idempotent_without_request() -> None:
    from src.integrations.ls.client import LsClient, LsCredentials

    class RecordingSession:
        def __init__(self) -> None:
            self.close_calls = 0
            self.posts = 0

        def post(self, *_args: object, **_kwargs: object) -> object:
            self.posts += 1
            raise AssertionError("no request expected")

        def close(self) -> None:
            self.close_calls += 1

    session = RecordingSession()
    client = LsClient(LsCredentials("key", "secret"), session=session)  # type: ignore[arg-type]
    client.close()
    client.close()
    assert session.close_calls == 1
    assert session.posts == 0


def test_ls_client_context_closes_session_after_success() -> None:
    from datetime import date

    from src.integrations.ls.client import LsClient, LsCredentials

    class OkResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"rsp_cd": "00000", "t1702OutBlock1": [{"date": "20161229"}]}

    class RecordingSession:
        def __init__(self) -> None:
            self.closed = False

        def post(self, *_args: object, **_kwargs: object) -> OkResponse:
            return OkResponse()

        def close(self) -> None:
            self.closed = True

    session = RecordingSession()
    LsClient._last_slot = None
    with LsClient(
        LsCredentials("key", "secret"),
        session=session,  # type: ignore[arg-type]
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    ) as client:
        client._token = "token"
        assert client.inquire_investor_trend("000020", date(2016, 12, 29), date(2016, 12, 29)) == (
            {"date": "20161229"},
        )
    assert session.closed is True


def test_ls_client_context_closes_session_after_provider_failure() -> None:
    from datetime import date

    import pytest

    from src.data.schemas import PITDataError
    from src.integrations.ls.client import LsClient, LsCredentials

    class DeniedResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"rsp_cd": "IGW99999"}

    class RecordingSession:
        def __init__(self) -> None:
            self.closed = False

        def post(self, *_args: object, **_kwargs: object) -> DeniedResponse:
            return DeniedResponse()

        def close(self) -> None:
            self.closed = True

    session = RecordingSession()
    LsClient._last_slot = None
    client = LsClient(
        LsCredentials("key", "secret"),
        session=session,  # type: ignore[arg-type]
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    client._token = "token"

    def run_failing_request() -> None:
        with client:
            client.inquire_investor_trend("000020", date(2016, 12, 29), date(2016, 12, 29))

    with pytest.raises(PITDataError, match="non-success"):
        run_failing_request()
    assert session.closed is True


def test_ls_client_pacing_survives_consecutive_requests() -> None:
    from src.integrations.ls.client import LS_MIN_REQUEST_INTERVAL_SECONDS, LsClient, LsCredentials

    LsClient._last_slot = None
    sleeps: list[float] = []
    client = LsClient(
        LsCredentials("key", "secret"),
        session=object(),  # type: ignore[arg-type]
        monotonic=lambda: 100.0,
        sleeper=sleeps.append,
    )
    client._wait_for_request_slot()
    client._wait_for_request_slot()
    assert sleeps == [LS_MIN_REQUEST_INTERVAL_SECONDS]


def test_ls_client_lifecycle_preserves_retry_contract() -> None:
    from datetime import date

    import pytest
    import requests

    from src.data.schemas import PITDataError
    from src.integrations.ls.client import LsClient, LsCredentials

    def http_error(status: int) -> requests.HTTPError:
        exc = requests.HTTPError(f"http {status}")
        exc.response = type("Resp", (), {"status_code": status})()  # type: ignore[attr-defined]
        return exc

    class HttpResponse:
        def __init__(self, payload: object = None, error: Exception | None = None) -> None:
            self._payload = payload
            self._error = error

        def raise_for_status(self) -> None:
            if self._error is not None:
                raise self._error

        def json(self) -> object:
            return self._payload

    backoffs: list[float] = []

    class FlakySession:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, *_args: object, **_kwargs: object) -> HttpResponse:
            self.calls += 1
            if self.calls == 1:
                return HttpResponse(error=http_error(429))
            return HttpResponse(payload={"rsp_cd": "00000", "t1702OutBlock1": [{"date": "20161229"}]})

        def close(self) -> None:
            return None

    LsClient._last_slot = None
    session = FlakySession()
    client = LsClient(
        LsCredentials("key", "secret"),
        session=session,  # type: ignore[arg-type]
        monotonic=lambda: 0.0,
        sleeper=backoffs.append,
    )
    client._token = "token"
    assert client.inquire_investor_trend("000020", date(2016, 12, 29), date(2016, 12, 29)) == (
        {"date": "20161229"},
    )
    assert session.calls == 2

    class ThrottledResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"rsp_cd": "IGW00201"}

    class ThrottledSession:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, *_args: object, **_kwargs: object) -> ThrottledResponse:
            self.calls += 1
            return ThrottledResponse()

        def close(self) -> None:
            return None

    LsClient._last_slot = None
    throttled = ThrottledSession()
    throttled_client = LsClient(
        LsCredentials("key", "secret"),
        session=throttled,  # type: ignore[arg-type]
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    throttled_client._token = "token"
    with pytest.raises(PITDataError, match="throttled"):
        throttled_client.inquire_investor_trend("000020", date(2016, 12, 29), date(2016, 12, 29))
    assert throttled.calls == 3


def test_ls_investor_flow_collector_close_delegates_to_client() -> None:
    from src.integrations.ls.investor_flow import LsInvestorFlowCollector

    class FakeClient:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    inner = FakeClient()
    collector = LsInvestorFlowCollector(("005930",), client=inner)
    with collector:
        pass
    assert inner.close_calls == 1

    plain = LsInvestorFlowCollector(("005930",), client=object())
    plain.close()


def test_inquire_investor_trend_posts_share_quantity_mode() -> None:
    from datetime import date

    from src.integrations.ls.client import LsClient, LsCredentials

    captured: dict[str, object] = {}

    class OkResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"rsp_cd": "00000", "t1702OutBlock1": [{"date": "20260306"}]}

    class RecordingSession:
        def post(self, _url: object, *, headers: object = None, json: object = None, timeout: object = None) -> OkResponse:
            captured["body"] = json
            return OkResponse()

    LsClient._last_slot = None
    client = LsClient(
        LsCredentials("key", "secret"),
        session=RecordingSession(),  # type: ignore[arg-type]
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    client._token = "token"
    assert client.inquire_investor_trend("005930", date(2026, 3, 1), date(2026, 3, 6)) == ({"date": "20260306"},)
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["t1702InBlock"]["volvalgb"] == "1"


def test_inquire_investor_trend_rejects_amount_mode_before_io() -> None:
    from datetime import date

    import pytest

    from src.integrations.ls.client import LsClient, LsCredentials

    class NoIoSession:
        def post(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("no HTTP post expected")

    client = LsClient(
        LsCredentials("key", "secret"),
        session=NoIoSession(),  # type: ignore[arg-type]
        monotonic=lambda: 0.0,
        sleeper=lambda _delay: None,
    )
    client._token = "token"
    with pytest.raises(ValueError, match="shares"):
        client.inquire_investor_trend("005930", date(2026, 3, 1), date(2026, 3, 6), unit="amount")  # type: ignore[arg-type]
