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


def test_provision_optional_alert_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    optional = provision_env._optional_keys()
    assert optional == ["ALERT_GMAIL_USER", "ALERT_GMAIL_APP_PASSWORD", "ALERT_GMAIL_TO"]
    allowed = provision_env._allow_list()
    base = dict.fromkeys(allowed, "secret")
    assert "ALERT_GMAIL_USER" not in provision_env._render_env(dict(base), allowed)
    full = dict(base, ALERT_GMAIL_USER="u", ALERT_GMAIL_APP_PASSWORD="p", ALERT_GMAIL_TO="t")
    payload = provision_env._render_env(dict(full), allowed)
    assert "ALERT_GMAIL_USER=u\n" in payload
    assert payload.index("ECOS_API=") < payload.index("ALERT_GMAIL_USER=")
    assert payload.index("ALERT_GMAIL_TO=") < payload.index("KIS_POOL_SLOT=")
    with pytest.raises(SystemExit, match="single-line"):
        provision_env._render_env(dict(full, ALERT_GMAIL_TO="a\nb"), allowed)
    with pytest.raises(SystemExit, match="single-line"):
        provision_env._render_env(dict(full, ALERT_GMAIL_TO="\n"), allowed)
    assert "ALERT_GMAIL_TO=" not in provision_env._render_env(dict(full, ALERT_GMAIL_TO=" "), allowed)
    with pytest.raises(SystemExit, match="missing keys"):
        provision_env._render_env({"ALERT_GMAIL_USER": "u"}, allowed)
    config_dir = tmp_path / "config" / "data"
    config_dir.mkdir(parents=True)
    (config_dir / "provision_env.toml").write_text('secret_keys = ["REQUIRED"]\n')
    monkeypatch.setattr(provision_env, "REPO", tmp_path)
    assert provision_env._optional_keys() == []


def test_provision_rejects_multiline_secret_file(tmp_path: Path) -> None:
    source = tmp_path / "secrets.env"
    source.write_text('ALERT_GMAIL_APP_PASSWORD="first\nsecond"\n')
    with pytest.raises(SystemExit, match="single-line: ALERT_GMAIL_APP_PASSWORD"):
        provision_env._read_secret_file(source)


def test_provision_main_pushes_combined_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    allowed = provision_env._allow_list()
    optional = provision_env._optional_keys()
    values = dict.fromkeys(allowed + optional, "private-value")
    source = tmp_path / "secrets.env"
    source.write_text("\n".join(f"{key}={value}" for key, value in values.items()))
    monkeypatch.setattr(sys, "argv", ["provision_env", "--secrets-file", str(source), "--host", "fixture"])

    def push(**kwargs: object) -> None:
        assert kwargs["key_names"] == allowed + optional
        assert kwargs["payload"] == provision_env._render_env(values, allowed)
        assert kwargs["dest"] == "/home/ubuntu/quant-secrets/k-stock-engine.env"

    monkeypatch.setattr(provision_env, "_push_over_ssh", push)
    assert provision_env.main() == 0
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--dry-run"])
    assert provision_env.main() == 0
    output = capsys.readouterr().out
    assert "private-value" not in output
    assert ",".join(allowed + optional) in output
