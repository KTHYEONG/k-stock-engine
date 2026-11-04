"""Token-state failures must preserve shared state and prevent upstream issuance."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import traceback
from pathlib import Path
from typing import Any

import pytest

from src.integrations import host_broker_contract as contract


def _store(path: Path, *, asynchronous: bool, vendor: str = "ls") -> Any:
    store_type = contract.AsyncTokenStore if asynchronous else contract.TokenStore
    return store_type(
        path,
        vendor=vendor,
        lock_timeout_seconds=1,
        expiry_margin_seconds=600,
        clock=lambda: dt.datetime(2026, 11, 4, tzinfo=dt.UTC),
        issue_throttle_policy="raise",
    )


def _operate(store: Any, *, asynchronous: bool, operation: str, calls: list[str]) -> Any:
    def issue() -> contract.IssuedToken:
        calls.append("issued")
        return contract.IssuedToken("synthetic-token", None)

    async def aissue() -> contract.IssuedToken:
        return issue()

    method = getattr(store, operation)
    args = ("rejected",) if operation == "replace_rejected" else ()
    if asynchronous:
        return asyncio.run(method(*args, aissue))
    return method(*args, issue)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["get_or_issue", "replace_rejected"])
@pytest.mark.parametrize("contents", [b"", b"{sensitive-token", b"\xffsensitive-token", b"{}", b"[]", b"[" * 2000 + b"]" * 2000])
def test_invalid_record_never_issues(tmp_path: Path, asynchronous: bool, operation: str, contents: bytes) -> None:
    path = tmp_path / "token.json"
    path.write_bytes(contents)
    calls: list[str] = []
    with pytest.raises(contract.TokenStoreReadError) as error:
        _operate(_store(path, asynchronous=asynchronous), asynchronous=asynchronous, operation=operation, calls=calls)
    assert calls == []
    assert path.read_bytes() == contents
    assert "sensitive-token" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["get_or_issue", "replace_rejected"])
def test_schema_invalid_record_never_issues(tmp_path: Path, asynchronous: bool, operation: str) -> None:
    path = tmp_path / "token.json"
    contents = json.dumps(
        {
            "schema_version": 1,
            "access_token": "sensitive-token",
            "issued_at": "2026-11-04T00:00:00+00:00",
            "expires_at": None,
            "generation": False,
        }
    )
    path.write_text(contents, encoding="utf-8")
    calls: list[str] = []
    with pytest.raises(contract.TokenStoreReadError, match="schema is invalid"):
        _operate(_store(path, asynchronous=asynchronous), asynchronous=asynchronous, operation=operation, calls=calls)
    assert calls == []
    assert path.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["get_or_issue", "replace_rejected"])
@pytest.mark.parametrize("failure", ["directory", "permission"])
def test_unreadable_record_never_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, operation: str, failure: str
) -> None:
    path = tmp_path / "token.json"
    if failure == "directory":
        path.mkdir()
    else:
        original = Path.read_text

        def read_text(self: Path, *args: Any, **kwargs: Any) -> str:
            if self == path:
                raise PermissionError("sensitive-token")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)
    calls: list[str] = []
    with pytest.raises(contract.TokenStoreReadError, match="record is unreadable") as error:
        _operate(_store(path, asynchronous=asynchronous), asynchronous=asynchronous, operation=operation, calls=calls)
    assert calls == []
    assert "sensitive-token" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["get_or_issue", "replace_rejected"])
@pytest.mark.parametrize("contents", [b"", b"sensitive-token", b"nan", b"inf", b"-1", b"\xff", b"0" * 65])
def test_malformed_db_guard_never_issues(tmp_path: Path, asynchronous: bool, operation: str, contents: bytes) -> None:
    path = contract.token_record_path(tmp_path, "dbsec", "synthetic-key")
    guard = contract.issue_guard_path(tmp_path, "dbsec", "synthetic-key")
    guard.write_bytes(contents)
    calls: list[str] = []
    with pytest.raises(contract.TokenStoreReadError, match="guard is malformed") as error:
        _operate(
            _store(path, asynchronous=asynchronous, vendor="dbsec"),
            asynchronous=asynchronous,
            operation=operation,
            calls=calls,
        )
    assert calls == []
    assert not path.exists()
    assert guard.read_bytes() == contents
    assert "sensitive-token" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["get_or_issue", "replace_rejected"])
@pytest.mark.parametrize("failure", ["directory", "permission"])
def test_unreadable_db_guard_never_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, operation: str, failure: str
) -> None:
    path = contract.token_record_path(tmp_path, "dbsec", "synthetic-key")
    guard = contract.issue_guard_path(tmp_path, "dbsec", "synthetic-key")
    if failure == "directory":
        guard.mkdir()
    else:
        original = contract.os.open

        def open_file(name: Any, *args: Any, **kwargs: Any) -> int:
            if Path(name) == guard:
                raise PermissionError("sensitive-token")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(contract.os, "open", open_file)
    calls: list[str] = []
    with pytest.raises(contract.TokenStoreReadError, match="guard is unreadable") as error:
        _operate(
            _store(path, asynchronous=asynchronous, vendor="dbsec"),
            asynchronous=asynchronous,
            operation=operation,
            calls=calls,
        )
    assert calls == []
    assert not path.exists()
    assert "sensitive-token" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["get_or_issue", "replace_rejected"])
@pytest.mark.parametrize("guard_exists", [False, True])
def test_absent_record_and_absent_or_valid_old_guard_allow_issuance(
    tmp_path: Path, asynchronous: bool, operation: str, guard_exists: bool
) -> None:
    path = contract.token_record_path(tmp_path, "dbsec", "synthetic-key")
    guard = contract.issue_guard_path(tmp_path, "dbsec", "synthetic-key")
    if guard_exists:
        guard.write_text("0.0", encoding="ascii")
    calls: list[str] = []
    store = _store(path, asynchronous=asynchronous, vendor="dbsec")
    record = _operate(store, asynchronous=asynchronous, operation=operation, calls=calls)
    assert calls == ["issued"]
    assert record.access_token == "synthetic-token"
    assert store.read() == record
    assert float(guard.read_text(encoding="ascii")) == dt.datetime(2026, 11, 4, tzinfo=dt.UTC).timestamp()


def test_local_adapter_exposes_read_error() -> None:
    from src.integrations import host_broker

    assert host_broker.TokenStoreReadError is contract.TokenStoreReadError
    assert "TokenStoreReadError" in host_broker.__all__
