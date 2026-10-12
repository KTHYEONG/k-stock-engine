"""Backup failure persistence and secret provisioning boundaries."""

import fcntl
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.devops import lake_backup as backup_tool
from tools.devops import provision_env


def test_push_failure_records_status_and_carries_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    (root / "bronze").mkdir(parents=True)
    (root / "bronze" / "page.json").write_text("{}")
    config = tmp_path / "backup.toml"
    config.write_text('remote = "unused"\n[[source]]\nname="bronze"\nroot="data/bronze"\n')
    status = root / "state" / "status.json"
    status.parent.mkdir()
    status.write_text(json.dumps({"last_ok_at": "2026-10-07T00:00:00+00:00"}))
    monkeypatch.setattr(sys, "argv", ["lake_backup", "push", "--config", str(config), "--data-root", str(root),
                                     "--status", str(status), "--store-dir", str(tmp_path / "remote"), "--no-lock"])

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("sensitive-credential")

    monkeypatch.setattr(backup_tool, "push", fail)
    assert backup_tool.main() == 1
    result = json.loads(status.read_text())
    assert result["rc"] == 1
    assert result["last_ok_at"] == "2026-10-07T00:00:00+00:00"
    assert "sensitive-credential" not in status.read_text()


def test_missing_roots_skip_when_optional_and_never_abort_later_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    (root / "bronze").mkdir(parents=True)
    (root / "bronze" / "page.json").write_text("{}")
    (root / "state").mkdir()
    config = tmp_path / "backup.toml"
    config.write_text(
        'remote = "unused"\n'
        '[[source]]\nname="needed"\nroot="data/needed"\n'
        '[[source]]\nname="legacy"\nroot="data/legacy"\noptional=true\n'
        '[[source]]\nname="bronze"\nroot="data/bronze"\n'
    )
    status = root / "state" / "status.json"
    monkeypatch.setattr(sys, "argv", ["lake_backup", "push", "--config", str(config), "--data-root", str(root),
                                     "--status", str(status), "--store-dir", str(tmp_path / "remote"), "--no-lock"])
    assert backup_tool.main() == 1
    sources = json.loads(status.read_text())["sources"]
    assert sources["needed"] == {"error": "FileNotFoundError"}
    assert sources["legacy"] == {"skipped": "root_missing"}
    assert sources["bronze"]["stored"] == 1


def test_shared_lock_timeout_is_75(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "backup.toml"
    config.write_text('remote="unused"\n[[source]]\nname="bronze"\nroot="data/bronze"\n')
    lock_path = tmp_path / "shared.lock"
    status = tmp_path / "status.json"
    monkeypatch.setattr(sys, "argv", ["lake_backup", "push", "--config", str(config), "--status", str(status),
                                     "--lock-file", str(lock_path), "--lock-wait-s", "0"])
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert backup_tool.main() == 75
    assert json.loads(status.read_text())["rc"] == 75


def test_provision_allowlist_and_stdin_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "env"
    allowed = provision_env._allow_list()
    source.write_text("\n".join(f"export {key}='secret'" for key in allowed) + "\nUNRELATED=ignored\n")
    payload = provision_env._render_env(provision_env._read_secret_file(source), allowed)
    assert "UNRELATED" not in payload
    assert "KIS_POOL_SLOT=5\n" in payload

    def ssh(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "secret" not in " ".join(command)
        assert kwargs["input"] == payload
        assert "chmod 600" in command[-1]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(provision_env.subprocess, "run", ssh)
    provision_env._push_over_ssh(host="fixture", user="ubuntu", ssh_key=None, dest="fixture.env", payload=payload,
                                 key_names=allowed)
    with pytest.raises(SystemExit, match="single-line"):
        provision_env._render_env(dict.fromkeys(allowed, "secret\nINJECTED=1"), allowed)
    with pytest.raises(SystemExit, match="missing keys"):
        provision_env._render_env({}, allowed)
