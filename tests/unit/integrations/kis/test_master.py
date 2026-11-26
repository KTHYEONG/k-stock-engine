"""KIS stock-master client: container validation and raw bytes passthrough."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

import pytest

from src.integrations.errors import ProviderTerminalError
from src.integrations.kis.master import KisMasterClient

_MAX_BYTES = 10_000


@dataclass
class _Response:
    content: bytes
    headers: dict[str, str] = field(default_factory=dict)


class _FakeTransport:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.endpoints: list[str] = []

    def get(self, endpoint: str, params: dict[str, str], **_: object) -> _Response:
        self.endpoints.append(endpoint)
        return self._response


def _zip_bytes(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


def _client(response: _Response) -> tuple[KisMasterClient, _FakeTransport]:
    transport = _FakeTransport(response)
    client = KisMasterClient(
        transport=transport,  # type: ignore[arg-type]
        files={"kospi": "kospi_code.mst.zip", "kosdaq": "kosdaq_code.mst.zip"},
        max_bytes=_MAX_BYTES,
    )
    return client, transport


def test_fetch_returns_bytes_and_last_modified() -> None:
    body = _zip_bytes("kospi_code.mst", "005930 삼성전자 ST\n".encode("cp949"))
    client, transport = _client(_Response(body, {"Last-Modified": "Wed, 26 Nov 2026 01:55:08 GMT"}))

    master = client.fetch("kospi")

    assert transport.endpoints == ["kospi_code.mst.zip"]
    assert master.content == body
    assert master.last_modified == "Wed, 26 Nov 2026 01:55:08 GMT"
    assert master.file_name == "kospi_code.mst.zip"
    assert master.market == "kospi"


def test_fetch_last_modified_is_none_when_header_absent() -> None:
    body = _zip_bytes("kosdaq_code.mst", "x\n".encode("cp949"))
    client, _ = _client(_Response(body))

    assert client.fetch("kosdaq").last_modified is None


def test_fetch_rejects_wrong_member() -> None:
    client, _ = _client(_Response(_zip_bytes("other.mst", "x\n".encode("cp949"))))

    with pytest.raises(ProviderTerminalError):
        client.fetch("kospi")


def test_fetch_rejects_extra_member() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("kospi_code.mst", "x\n".encode("cp949"))
        archive.writestr("extra.txt", "y")
    client, _ = _client(_Response(buffer.getvalue()))

    with pytest.raises(ProviderTerminalError):
        client.fetch("kospi")


@pytest.mark.parametrize("body", [b"", b"x" * (_MAX_BYTES + 1)])
def test_fetch_rejects_empty_and_oversize_body(body: bytes) -> None:
    client, _ = _client(_Response(body))

    with pytest.raises(ProviderTerminalError):
        client.fetch("kospi")


def test_fetch_rejects_non_zip_body() -> None:
    client, _ = _client(_Response(b"<html>not a zip</html>"))

    with pytest.raises(ProviderTerminalError):
        client.fetch("kospi")


def test_fetch_rejects_member_that_is_not_cp949() -> None:
    client, _ = _client(_Response(_zip_bytes("kospi_code.mst", b"\xff\xfe\xfd")))

    with pytest.raises(ProviderTerminalError):
        client.fetch("kospi")


def test_fetch_rejects_member_without_rows() -> None:
    client, _ = _client(_Response(_zip_bytes("kospi_code.mst", b"\n  \n")))

    with pytest.raises(ProviderTerminalError):
        client.fetch("kospi")


def test_fetch_unknown_market() -> None:
    client, _ = _client(_Response(b""))

    with pytest.raises(ValueError, match="unknown KIS master market"):
        client.fetch("konex")


def test_fetch_rejects_oversize_member() -> None:
    client, _ = _client(_Response(_zip_bytes("kospi_code.mst", b"0" * (_MAX_BYTES + 1))))

    with pytest.raises(ProviderTerminalError):
        client.fetch("kospi")


def test_build_client_uses_policy_files_without_network(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config
    from src.integrations.kis.master import build_kis_master_client
    from src.integrations.quota import ProviderQuotaStateStore

    policy = load_provider_policy(load_runtime_config()).kis_master
    client = build_kis_master_client(policy=policy, quota_store=ProviderQuotaStateStore(tmp_path / "quota"))

    with pytest.raises(ValueError, match="unknown KIS master market"):
        client.fetch("konex")
