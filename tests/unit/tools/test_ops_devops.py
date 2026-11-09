"""Backup failure persistence and secret provisioning boundaries."""

import fcntl
import json
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from src.config.providers import load_provider_policy
from src.config.runtime import load_runtime_config
from tools.devops import lake_backup as backup_tool
from tools.devops import provision_env

REPO = Path(__file__).resolve().parents[3]


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


def test_provision_preserves_db_credentials(tmp_path: Path) -> None:
    allowed = provision_env._allow_list()
    assert allowed[-3:] == ["ECOS_API", "DB_APP_KEY", "DB_APP_SECRET"]
    assert "DB_APP_KEY" not in provision_env._optional_keys()
    assert "DB_APP_SECRET" not in provision_env._optional_keys()
    source = tmp_path / "secrets.env"
    source.write_text("\n".join(f"{key}=value-{key}" for key in allowed) + "\nUNRELATED=ignored\n")
    payload = provision_env._render_env(provision_env._read_secret_file(source), allowed)
    assert payload.count("DB_APP_KEY=") == 1
    assert payload.count("DB_APP_SECRET=") == 1
    assert "DB_APP_KEY=value-DB_APP_KEY\n" in payload
    assert "DB_APP_SECRET=value-DB_APP_SECRET\n" in payload
    assert "UNRELATED" not in payload
    assert payload.index("ECOS_API=") < payload.index("DB_APP_KEY=")
    assert payload.index("DB_APP_SECRET=") < payload.index("KIS_POOL_SLOT=")
    assert payload.endswith("KIS_POOL_SLOT=5\nKIS_REQUIRE_SHARED_ADMISSION=true\n")


@pytest.mark.parametrize("key", ["DB_APP_KEY", "DB_APP_SECRET"])
@pytest.mark.parametrize("value", [None, "", " "])
def test_provision_aborts_without_db_credentials(tmp_path: Path, key: str, value: str | None) -> None:
    allowed = provision_env._allow_list()
    values = dict.fromkeys(allowed, "secret")
    if value is None:
        del values[key]
    else:
        values[key] = value
    source = tmp_path / "secrets.env"
    source.write_text("\n".join(f"{name}={secret}" for name, secret in values.items()))
    with pytest.raises(SystemExit, match=f"^missing keys in secret file: {key}$"):
        provision_env._render_env(provision_env._read_secret_file(source), allowed)


def test_provider_credentials_are_provisioned() -> None:
    provider = load_provider_policy(load_runtime_config(create_directories=False))
    assert provider.kis.pool_slot is not None
    names = [
        *provider.dart.keys,
        provider.krx.api_key_env,
        provider.ls.app_key_env,
        provider.ls.app_secret_env,
        provider.dbsec.app_key_env,
        provider.dbsec.app_secret_env,
    ]
    allowed = set(provision_env._allow_list())
    for name in names:
        assert name in allowed


def test_provision_dry_run_lists_db_keys_without_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    allowed = provision_env._allow_list()
    values = dict.fromkeys(allowed, "secret")
    values["DB_APP_KEY"] = "private-db"
    source = tmp_path / "secrets.env"
    source.write_text("\n".join(f"{key}={value}" for key, value in values.items()))

    def no_push(**kwargs: object) -> None:
        raise AssertionError("dry-run must not push")

    monkeypatch.setattr(provision_env, "_push_over_ssh", no_push)
    monkeypatch.setattr(
        sys, "argv", ["provision_env", "--secrets-file", str(source), "--host", "fixture", "--dry-run"]
    )
    assert provision_env.main() == 0
    output = capsys.readouterr().out
    assert "DB_APP_KEY,DB_APP_SECRET" in output
    assert "private-db" not in output


def _gc_argv(tmp_path: Path, *extra: str) -> list[str]:
    config = tmp_path / "backup.toml"
    config.write_text('remote="unused"\n[[source]]\nname="bronze"\nroot="data/bronze"\n')
    return ["lake_backup", "gc", "--config", str(config), "--data-root", str(tmp_path / "data"),
            "--store-dir", str(tmp_path / "remote"), *extra]


def test_gc_cli_prints_a_dry_run_plan_and_apply_requires_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "remote" / "bronze").mkdir(parents=True)
    (tmp_path / "remote" / "bronze" / "seg-00000000000000cc.tar.zst").write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", _gc_argv(tmp_path, "--no-lock"))
    assert backup_tool.main() == 0
    plan = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["bronze"]
    assert plan["orphans"] == 1
    assert plan["applied"] is False
    assert plan["states"] == 0
    assert (tmp_path / "remote" / "bronze" / "seg-00000000000000cc.tar.zst").exists()
    monkeypatch.setattr(sys, "argv", _gc_argv(tmp_path, "--apply", "--no-lock"))
    with pytest.raises(SystemExit, match="requires the shared Drive lock"):
        backup_tool.main()
    monkeypatch.setattr(sys, "argv", _gc_argv(tmp_path, "--apply", "--lock-file", str(tmp_path / "gdrive.lock")))
    assert backup_tool.main() == 0
    assert not (tmp_path / "remote" / "bronze" / "seg-00000000000000cc.tar.zst").exists()


def test_restore_at_parses_state_numbers_and_aware_timestamps() -> None:
    from datetime import UTC, datetime

    assert backup_tool._parse_at("3") == 3
    assert backup_tool._parse_at("2026-10-30T17:30:00+00:00") == datetime(2026, 10, 30, 17, 30, tzinfo=UTC)
    with pytest.raises(Exception, match="UTC offset"):
        backup_tool._parse_at("2026-10-30T17:30:00")


def _known_hosts_from(command: list[str]) -> Path:
    option = next(item for item in command if item.startswith("UserKnownHostsFile="))
    return Path(option.removeprefix("UserKnownHostsFile="))


def test_provision_pins_host_key_and_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def ssh(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(provision_env.subprocess, "run", ssh)
    provision_env._push_over_ssh(host="fixture", user="ubuntu", ssh_key=None, dest="fixture.env",
                                 payload="SECRET=value\n", key_names=["X"])
    command, kwargs = calls[0]
    assert "StrictHostKeyChecking=yes" in command
    assert "BatchMode=yes" in command
    assert _known_hosts_from(command) != Path("/dev/null")
    assert not any("StrictHostKeyChecking=no" in item or "/dev/null" in item for item in command)
    assert kwargs["input"] == "SECRET=value\n"
    assert "SECRET=value" not in " ".join(command)


@pytest.mark.parametrize("returncode", [0, 255])
def test_known_hosts_file_has_pinned_key_mode_0600_and_is_removed(
    monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    observed: dict[str, object] = {}

    def ssh(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        known_hosts = _known_hosts_from(command)
        observed["path"] = known_hosts
        observed["text"] = known_hosts.read_text(encoding="utf-8")
        observed["mode"] = stat.S_IMODE(known_hosts.stat().st_mode)
        return subprocess.CompletedProcess(command, returncode, "", "")

    monkeypatch.setattr(provision_env.subprocess, "run", ssh)
    try:
        provision_env._push_over_ssh(host="fixture", user="ubuntu", ssh_key=None, dest="fixture.env",
                                     payload="x\n", key_names=["X"])
    except SystemExit:
        assert returncode != 0
    assert observed["text"] == f"* {provision_env._host_key()}\n"
    assert observed["mode"] == 0o600
    assert not observed["path"].exists()
    assert not observed["path"].parent.exists()


@pytest.mark.parametrize(
    ("config_text", "value"),
    [
        ('secret_keys = ["REQUIRED"]\n', None),
        ('vps_host_key = "ssh-ed25519"\nsecret_keys = ["REQUIRED"]\n', "ssh-ed25519"),
        ('vps_host_key = "ssh-ed25519 AAAA extra"\nsecret_keys = ["REQUIRED"]\n', "extra"),
        ('vps_host_key = "AAAA-not-ssh AAAA"\nsecret_keys = ["REQUIRED"]\n', "AAAA-not-ssh"),
    ],
)
def test_missing_or_malformed_host_key_aborts_before_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_text: str, value: str | None
) -> None:
    config_dir = tmp_path / "config" / "data"
    config_dir.mkdir(parents=True)
    (config_dir / "provision_env.toml").write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(provision_env, "REPO", tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(provision_env.subprocess, "run", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(SystemExit) as excinfo:
        provision_env._push_over_ssh(host="fixture", user="ubuntu", ssh_key=None, dest="fixture.env",
                                     payload="x\n", key_names=["X"])
    assert not calls
    assert value is None or value not in str(excinfo.value)


def test_host_key_matches_deploy_workflow() -> None:
    config = tomllib.loads((REPO / "config" / "data" / "provision_env.toml").read_text(encoding="utf-8"))
    workflow = (REPO / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    match = re.search(r"^\s*VPS_HOST_KEY:\s*(.+?)\s*$", workflow, re.MULTILINE)
    assert match is not None
    assert config["vps_host_key"] == match.group(1)


def test_no_weak_ssh_options_in_tools() -> None:
    for path in (REPO / "tools").rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        data = path.read_bytes()
        assert b"StrictHostKeyChecking=no" not in data, path
        assert b"UserKnownHostsFile=/dev/null" not in data, path
