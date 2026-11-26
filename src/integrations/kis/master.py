"""KIS whole-market stock-master file client: public download, transport-only."""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import requests

from src.config.providers import KisMasterPolicy
from src.integrations.errors import ProviderTerminalError
from src.integrations.quota import LedgerQuotaGate, ProviderQuotaStateStore
from src.integrations.transport import HttpTransport, RetryPolicy

__all__ = [
    "KisMasterClient",
    "KisMasterFile",
    "build_kis_master_client",
]

_PROVIDER: Final = "KISMASTER"
_MAX_HTTP_ATTEMPTS: Final = 3
_MEMBER_ENCODING: Final = "cp949"


@dataclass(frozen=True, slots=True)
class KisMasterFile:
    """One raw master zip as published, with its server-side modification stamp."""

    market: str
    file_name: str
    content: bytes
    last_modified: str | None


class KisMasterClient:
    """Fetches one declared master zip and validates its container, not its rows.

    Row parsing and status flags belong to the Silver layer; this client only
    proves the bytes are the expected single-member cp949 file.
    """

    def __init__(self, *, transport: HttpTransport, files: Mapping[str, str], max_bytes: int) -> None:
        self._transport = transport
        self._files = dict(files)
        self._max_bytes = max_bytes

    def fetch(self, market: str) -> KisMasterFile:
        """Download one market's master zip.

        Raises:
            ValueError: ``market`` is not a declared master file.
            ProviderTerminalError: the body is empty, oversized, not a zip with
                exactly the expected member, undecodable, or has no rows.
        """
        if market not in self._files:
            raise ValueError(f"unknown KIS master market {market!r}")
        file_name = self._files[market]
        response = self._transport.get(file_name, {})
        content = response.content
        last_modified = response.headers.get("Last-Modified")
        _validate_container(content, file_name=file_name, max_bytes=self._max_bytes)
        return KisMasterFile(market=market, file_name=file_name, content=content, last_modified=last_modified)


def _terminal(message: str, file_name: str) -> ProviderTerminalError:
    return ProviderTerminalError(message, provider=_PROVIDER, endpoint=file_name)


def _validate_container(content: bytes, *, file_name: str, max_bytes: int) -> None:
    if not content:
        raise _terminal("KIS master body is empty", file_name)
    if len(content) > max_bytes:
        raise _terminal("KIS master body exceeds the byte limit", file_name)
    member = file_name.removesuffix(".zip")
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise _terminal("KIS master body is not a valid zip", file_name) from exc
    with archive:
        names = archive.namelist()
        if names != [member]:
            raise _terminal("KIS master zip must contain exactly the expected member", file_name)
        if archive.getinfo(member).file_size > max_bytes:
            raise _terminal("KIS master member exceeds the byte limit", file_name)
        raw = archive.read(member)
    try:
        text = raw.decode(_MEMBER_ENCODING)
    except UnicodeDecodeError as exc:
        raise _terminal("KIS master member is not cp949", file_name) from exc
    if not any(line.strip() for line in text.splitlines()):
        raise _terminal("KIS master member has no rows", file_name)


def build_kis_master_client(*, policy: KisMasterPolicy, quota_store: ProviderQuotaStateStore) -> KisMasterClient:
    """Build the public master-file client from its provider policy. No credentials."""
    transport = HttpTransport(
        provider=_PROVIDER,
        base_url=policy.base_url,
        min_interval_seconds=policy.min_interval_seconds,
        retry=RetryPolicy(max_attempts=_MAX_HTTP_ATTEMPTS),
        quota=LedgerQuotaGate(quota_store, provider=_PROVIDER, daily_limit=policy.daily_limit),
        timeout_seconds=policy.timeout_seconds,
        session=requests.Session(),
        sleep=lambda seconds: time.sleep(seconds),
        monotonic=lambda: time.monotonic(),
    )
    return KisMasterClient(transport=transport, files=policy.files, max_bytes=policy.max_bytes)
