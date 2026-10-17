"""VPS deploy asset invariants: wrapper profiles, safe units, timer placement."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SYSTEMD = REPO / "deploy" / "systemd"
WRAPPER = REPO / "deploy" / "kse-run"
INSTALLER = REPO / "deploy" / "install_systemd.sh"
FIXTURE = Path(__file__).parent / "fixtures" / "legacy_docker_argv.json"
PINNED_LIB_SHA256 = "d3a0228685c6110b04cb6bb7f0dd20aafff24c2e1cf1adf2de7c7fb84660b191"
IMAGE = "ghcr.io/kthyeong/k-stock-engine:latest"
DRIVE_SLOTS = {(9, 15), (21, 0), (21, 30), (22, 15), (23, 30)}
SIBLING_WINDOWS = [((8, 50), (9, 40)), ((15, 10), (15, 43))]
UNIT_JOBS = {
    "kse-auction-open.service": "auction-open",
    "kse-backup.service": "backup",
    "kse-collect-dbsec.service": "collect-dbsec",
    "kse-collect-eve.service": "collect-eve",
    "kse-daily-audit.service": "daily-audit",
    "kse-morning.service": "morning",
    "kse-weekly-report.service": "weekly-report",
    "kse-alert@.service": "alert",
}


def _services() -> list[Path]:
    return sorted(path for path in SYSTEMD.glob("kse-*.service") if "@" not in path.name)


def _exec_start(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("ExecStart=")]
    assert len(lines) == 1
    return lines[0]


def _print_argv(job: str, extra: tuple[str, ...] = (), home: str = "/tmp/fakehome") -> list[str]:  # noqa: S108
    env = {"PATH": "/usr/bin:/bin", "HOME": home}
    result = subprocess.run(  # noqa: S603,S607
        ["bash", str(WRAPPER), "--print", job, *extra],  # noqa: S603,S607
        capture_output=True, text=False, env=env, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith(b"\0")
    return result.stdout.decode().split("\0")[:-1]


def _unit_effective(unit: str, home: str = "/tmp/fakehome") -> list[str]:  # noqa: S108
    start = _exec_start((SYSTEMD / unit).read_text(encoding="utf-8")).removeprefix("ExecStart=")
    if unit == "kse-alert@.service":
        shell = shlex.split(start)
        assert shell[:2] == ["/bin/sh", "-c"]
        start = shell[2].split(" | ", 1)[1]
    elif unit == "kse-backup.service":
        prefix = "/usr/bin/flock -w 7200 %t/quant-gdrive.lock "
        assert start.startswith(prefix)
        start = start.removeprefix(prefix)
    tokens = shlex.split(start)
    assert tokens[0] == "%h/.local/lib/kse/kse-run"
    assert tokens[1] == UNIT_JOBS[unit]
    return _print_argv(tokens[1], tuple(tokens[2:]), home=home)


def _take_value(flag: str, tokens: list[str], pos: int) -> tuple[str, int]:
    if "=" in flag:
        return flag.split("=", 1)[1], 1
    return tokens[pos + 1], 2


def _normalize(tokens: list[str]) -> dict:
    toks = list(tokens)
    for pos, token in enumerate(toks):
        if token == "docker" or token.endswith("/docker"):
            toks = toks[pos + 1 :]
            break
    if toks and toks[0] == "run":
        toks = toks[1:]
    scalars: dict[str, str] = {}
    rm = False
    stdin = False
    log_opts: list[str] = []
    env_files: list[str] = []
    envs: list[str] = []
    vols: list[str] = []
    image = ""
    tail: list[str] = []
    pos = 0
    while pos < len(toks):
        token = toks[pos]
        if token == "--rm":
            rm = True
            pos += 1
        elif token == "-i":
            stdin = True
            pos += 1
        elif token in ("--name", "--user"):
            value, step = _take_value(token, toks, pos)
            scalars[token[2:]] = value
            pos += step
        elif token.startswith("--memory-swap") or token.startswith("--memory") or token.startswith("--cpus"):
            name = token[2:].split("=", 1)[0]
            value, step = _take_value(token, toks, pos)
            scalars[name] = value
            pos += step
        elif token == "--log-opt":
            log_opts.append(toks[pos + 1])
            pos += 2
        elif token == "--env-file":
            env_files.append(toks[pos + 1])
            pos += 2
        elif token == "-e":
            envs.append(toks[pos + 1])
            pos += 2
        elif token in ("-v", "--volume"):
            vols.append(toks[pos + 1])
            pos += 2
        elif token.startswith("-"):
            raise AssertionError(f"unknown flag {token}")
        else:
            image = token
            tail = toks[pos + 1 :]
            break
    return {
        "scalars": scalars, "rm": rm, "stdin": stdin,
        "log_opts": sorted(log_opts), "env_files": env_files,
        "envs": sorted(envs), "vols": sorted(vols),
        "image": image, "tail": tail,
    }


def _legacy_normalized(unit: str, home: str, uidgid: str) -> dict:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))[unit]
    rendered = raw
    start = rendered.find("/usr/bin/docker")
    assert start >= 0, unit
    fragment = rendered[start:].strip()
    if fragment.endswith("%i"):
        fragment = fragment[: -len("%i")].strip()
    if fragment.endswith("'"):
        fragment = fragment[:-1]
    tokens = [token.replace("%h", home).replace("%U:%G", uidgid) for token in shlex.split(fragment)]
    return _normalize(tokens)


def _truncate_at_alert(tail: list[str]) -> list[str]:
    assert "alert" in tail
    return tail[: tail.index("alert") + 1]


def test_effective_command_equivalent_to_legacy() -> None:
    with tempfile.TemporaryDirectory(prefix="kse home ") as home:
        uidgid = subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603,S607
        uidgid += ":" + subprocess.run(["id", "-g"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603,S607
        assert len(json.loads(FIXTURE.read_text(encoding="utf-8"))) == 8
        for unit in UNIT_JOBS:
            effective = _normalize(_unit_effective(unit, home=home))
            legacy = _legacy_normalized(unit, home, uidgid)
            assert effective["scalars"] == legacy["scalars"], unit
            assert effective["rm"] == legacy["rm"], unit
            assert effective["stdin"] == legacy["stdin"], unit
            assert effective["log_opts"] == legacy["log_opts"], unit
            assert effective["env_files"] == legacy["env_files"], unit
            assert effective["envs"] == legacy["envs"], unit
            assert effective["vols"] == legacy["vols"], unit
            assert effective["image"] == legacy["image"] == IMAGE, unit
            tail_eff, tail_leg = effective["tail"], legacy["tail"]
            if UNIT_JOBS[unit] == "alert":
                tail_eff, tail_leg = _truncate_at_alert(tail_eff), _truncate_at_alert(tail_leg)
            assert tail_eff == tail_leg, unit


def test_alert_passthrough_args_appended() -> None:
    tokens = _print_argv("alert", ("--unit", "u.service", "--result", "exit-code", "--exit-status", "1"))
    norm = _normalize(tokens)
    assert norm["stdin"] is True
    tail = norm["tail"]
    assert tail[-6:] == ["--unit", "u.service", "--result", "exit-code", "--exit-status", "1"]
    assert tail[tail.index("alert") + 1:] == ["--unit", "u.service", "--result", "exit-code", "--exit-status", "1"]


def test_alert_preserves_empty_and_shell_metacharacter_args() -> None:
    extra = ("--unit", "unit with spaces.service", "", "$(false)", "*", "line1\nline2")
    tokens = _print_argv("alert", extra)
    assert tokens[tokens.index("alert") + 1:] == list(extra)


def test_wrapper_exec_preserves_argv_stdin_pid_and_exit_status(tmp_path: Path) -> None:
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'pid': os.getpid(), 'argv': sys.argv[1:], 'stdin': sys.stdin.read()}))\n"
        "sys.exit(23)\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    home = str(tmp_path / "home with spaces")
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin", "HOME": home}
    for job in UNIT_JOBS.values():
        extra = ("--unit", "u service", "", "$(false)") if job == "alert" else ()
        payload = "journal line\n" if job == "alert" else ""
        process = subprocess.Popen(  # noqa: S603
            [str(WRAPPER), job, *extra], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env,
        )
        stdout, stderr = process.communicate(payload, timeout=10)
        assert process.returncode == 23, stderr
        actual = json.loads(stdout)
        assert actual["pid"] == process.pid
        assert actual["argv"] == _print_argv(job, extra, home)[1:]
        assert actual["stdin"] == payload


def test_non_alert_jobs_reject_extra_args() -> None:
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp/fakehome"}  # noqa: S108
    result = subprocess.run(["bash", str(WRAPPER), "--print", "morning", "extra"],  # noqa: S603,S607
                            capture_output=True, env=env, check=False)
    assert result.returncode == 64
    assert result.stdout == b""
    assert b"status=USAGE" in result.stderr


def test_usage_errors_fail_before_external_commands(tmp_path: Path) -> None:
    env = {"PATH": str(tmp_path), "HOME": str(tmp_path)}
    invalid = [(), ("--print",), ("nope",), ("--print", "nope"), ("bad\njob",)]
    invalid.extend((job, "extra") for job in UNIT_JOBS.values() if job != "alert")
    for args in invalid:
        result = subprocess.run(  # noqa: S603
            ["/bin/bash", str(WRAPPER), *args], capture_output=True, text=True, env=env, check=False,
        )
        assert result.returncode == 64
        assert result.stdout == ""
        lines = result.stderr.splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("[SYS] kse-run status=USAGE job=")


def test_unknown_job_rejected() -> None:
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp/fakehome"}  # noqa: S108
    for argv in (["bash", str(WRAPPER), "--print", "nope"], ["bash", str(WRAPPER)]):  # noqa: S603,S607
        result = subprocess.run(argv, capture_output=True, env=env, check=False)  # noqa: S603,S607
        assert result.returncode == 64, argv
        assert b"status=USAGE" in result.stderr


def test_swap_equals_memory_for_every_job() -> None:
    for unit in UNIT_JOBS:
        norm = _normalize(_unit_effective(unit))
        assert norm["scalars"]["memory"]
        assert norm["scalars"]["memory"] == norm["scalars"]["memory-swap"], unit


def test_units_carry_no_docker_flags() -> None:
    units = sorted(SYSTEMD.glob("kse-*.service"))
    assert len(units) == 8
    for path in units:
        text = path.read_text(encoding="utf-8")
        assert "docker run" not in text, path.name
        assert "--memory" not in text, path.name
        assert "--env-file" not in text, path.name
        assert "ghcr.io/" not in text, path.name
        assert " -v " not in text, path.name
        assert "%h/.local/lib/kse/kse-run" in _exec_start(text), path.name


def test_stop_post_matches_profile_name() -> None:
    for unit in UNIT_JOBS:
        text = (SYSTEMD / unit).read_text(encoding="utf-8")
        stop = [line.strip() for line in text.splitlines() if line.strip().startswith("ExecStopPost=")]
        assert len(stop) == 1, unit
        name = stop[0].split("docker rm -f", 1)[1].strip()
        assert name == _normalize(_unit_effective(unit))["scalars"]["name"], unit


def test_mount_allowlist_on_effective_argv() -> None:
    with tempfile.TemporaryDirectory() as home:
        for unit in UNIT_JOBS:
            norm = _normalize(_unit_effective(unit, home=home))
            for mount in norm["vols"]:
                assert mount.startswith(home + "/k-stock-engine/data:") or mount.startswith(home + "/k-stock-engine/logs:") \
                    or mount.startswith(home + "/.cache/kis:") or mount.startswith(home + "/.cache/dbsec:") \
                    or mount.startswith(home + "/.config/rclone:"), f"{unit}: {mount}"
                if "/.cache/dbsec:" in mount:
                    assert unit == "kse-collect-dbsec.service", unit
                    assert mount.endswith(":ro"), unit
                if "/.config/rclone:" in mount:
                    assert unit == "kse-backup.service", unit
                    assert mount.endswith(":ro"), unit


def test_alert_credentials_reach_only_audit_and_alert_units() -> None:
    holders = sorted(unit for unit in UNIT_JOBS
                     if any("quant-secrets/kse-alert.env" in env for env in _normalize(_unit_effective(unit))["env_files"]))
    assert holders == ["kse-alert@.service", "kse-daily-audit.service"]
    installer = INSTALLER.read_text(encoding="utf-8")
    assert 'touch "${HOME}/quant-secrets/kse-alert.env"' in installer
    assert 'chmod 600 "${HOME}/quant-secrets/kse-alert.env"' in installer


def test_resource_overrides_preserved() -> None:
    eve = _normalize(_unit_effective("kse-collect-eve.service"))
    assert eve["scalars"]["memory"] == "3g"
    assert "KSE_KIND_TRANSPORT=relay" in eve["envs"]
    dbsec = _normalize(_unit_effective("kse-collect-dbsec.service"))
    assert dbsec["scalars"]["memory"] == "1g"
    assert dbsec["scalars"]["cpus"] == "0.5"
    morning = _normalize(_unit_effective("kse-morning.service"))
    assert morning["scalars"]["memory"] == "4g"
    assert not any(env.startswith("KSE_KIND_TRANSPORT=") for env in morning["envs"])


def test_backup_keeps_drive_flock() -> None:
    start = _exec_start((SYSTEMD / "kse-backup.service").read_text(encoding="utf-8"))
    assert start.startswith("ExecStart=/usr/bin/flock -w 7200 %t/quant-gdrive.lock ")
    assert _normalize(_unit_effective("kse-backup.service"))["tail"][-2:] == ["backup", "--no-lock"]


def test_wrapper_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True, check=False)  # noqa: S603,S607
    assert result.returncode == 0, result.stderr
    assert os.stat(WRAPPER).st_mode & stat.S_IXUSR
    git_result = subprocess.run(["git", "ls-files", "-s", "--", "deploy/kse-run"],  # noqa: S603,S607
                                capture_output=True, text=True, cwd=str(REPO), check=False)
    if git_result.stdout.strip():
        assert "100755" in git_result.stdout


def test_installer_enables_timers_by_glob() -> None:
    text = INSTALLER.read_text(encoding="utf-8")
    literals = [name for name in re.findall(r"kse-[a-z-]+\.timer", text) if not name.startswith("kse-*")]
    assert literals == []
    assert "kse-*.timer" in text
    assert '"${SRC_DIR}"/kse-*.timer' in text
    assert "mv -f" in text
    assert "kse-run" in text
    assert text.index("mv -f") < text.index("kse-*.service")


def test_installer_atomically_installs_wrapper_and_discovers_timers(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    shutil.copytree(SYSTEMD.parent, source)
    (source / "systemd" / "kse-future.timer").write_text("[Timer]\n", encoding="utf-8")
    home = tmp_path / "home with spaces"
    (home / "quant-secrets").mkdir(parents=True)
    target_dir = home / ".local" / "lib" / "kse"
    target_dir.mkdir(parents=True)
    target = target_dir / "kse-run"
    target.write_text("old wrapper", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    events = tmp_path / "events.jsonl"
    for name in ("systemctl", "loginctl"):
        command = bin_dir / name
        command.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "home = pathlib.Path(os.environ['HOME'])\n"
            "assert (home / '.local/lib/kse/kse-run').stat().st_mode & 0o111\n"
            "assert len(list((home / '.config/systemd/user').glob('kse-*.service'))) == 8\n"
            "with open(os.environ['EVENT_LOG'], 'a') as log:\n"
            "    log.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')\n",
            encoding="utf-8",
        )
        command.chmod(0o755)
    env = {"HOME": str(home), "USER": "tester", "PATH": f"{bin_dir}:/usr/bin:/bin", "EVENT_LOG": str(events)}
    with target.open(encoding="utf-8") as old:
        result = subprocess.run(  # noqa: S603
            ["/bin/bash", str(source / "install_systemd.sh")], capture_output=True, text=True, env=env, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert old.read() == "old wrapper"
        assert os.fstat(old.fileno()).st_ino != target.stat().st_ino
    assert target.read_bytes() == WRAPPER.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert stat.S_IMODE(target_dir.stat().st_mode) == 0o700
    assert (home / ".cache/kis/.host-admission").is_file()
    assert stat.S_IMODE((home / "quant-secrets/kse-alert.env").stat().st_mode) == 0o600
    expected_timers = sorted(path.name for path in (source / "systemd").glob("kse-*.timer"))
    assert [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()] == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", *expected_timers],
        ["loginctl", "enable-linger", "tester"],
    ]


def test_morning_timeout_is_within_late_seal_window() -> None:
    text = (SYSTEMD / "kse-morning.service").read_text(encoding="utf-8")
    assert "TimeoutStartSec=30min" in text
    norm = _normalize(_unit_effective("kse-morning.service"))
    assert norm["scalars"]["memory"] == "4g"
    assert norm["scalars"]["memory-swap"] == "4g"


def test_units_are_safe_by_construction() -> None:
    services = _services()
    assert len(services) == 7
    with tempfile.TemporaryDirectory() as home:
        prefixes = (home + "/k-stock-engine/data:", home + "/k-stock-engine/logs:",
                    home + "/.cache/kis:", home + "/.cache/dbsec:", home + "/.config/rclone:")
        for path in services:
            text = path.read_text(encoding="utf-8")
            assert "TimeoutStartSec=" in text
            assert "OnFailure=kse-alert@%n.service" in text
            assert "Type=oneshot" in text
            assert "--privileged" not in text
            assert "Environment=TZ=Asia/Seoul" in text
            norm = _normalize(_unit_effective(path.name, home=home))
            assert norm["image"] == IMAGE
            assert norm["tail"][:4] == ["uv", "run", "python", "-m"]
            assert norm["scalars"]["memory"] == norm["scalars"]["memory-swap"]
            assert "HOME=/tmp" in norm["envs"]
            assert "UV_CACHE_DIR=/tmp/uv-cache" in norm["envs"]
            assert norm["vols"]
            for mount in norm["vols"]:
                assert mount.startswith(prefixes), f"{path.name}: {mount}"
                if mount.startswith(home + "/.config/rclone:"):
                    assert path.name == "kse-backup.service"
                    assert ":ro" in mount
            if path.name == "kse-backup.service":
                assert _exec_start(text).startswith("ExecStart=/usr/bin/flock -w 7200 %t/quant-gdrive.lock ")
                assert norm["tail"][-2:] == ["backup", "--no-lock"]
            assert f"ExecStopPost=-/usr/bin/docker rm -f {norm['scalars']['name']}" in text


def test_alert_template_falls_back_to_shared() -> None:
    path = SYSTEMD / "kse-alert@.service"
    text = path.read_text(encoding="utf-8")
    assert "OnFailure=kca-alert@%n.service" in text
    assert "%i" in text
    assert "Type=oneshot" in text
    assert "TimeoutStartSec=5min" in text
    assert "Environment=TZ=Asia/Seoul" in text
    assert "WorkingDirectory=%h/k-stock-engine" in text
    start = _exec_start(text)
    assert start.startswith("ExecStart=/bin/sh -c ")
    assert 'journalctl --user -u "$$0" -n 40 -o cat --no-pager' in start
    assert "%h/.local/lib/kse/kse-run alert" in start
    assert '--unit "$$0"' in start
    assert '--result "$$(systemctl --user show -p Result --value "$$0")"' in start
    assert '--exit-status "$$(systemctl --user show -p ExecMainStatus --value "$$0")"' in start
    assert start.rstrip().endswith("%i")
    norm = _normalize(_unit_effective("kse-alert@.service"))
    assert norm["stdin"] is True
    assert norm["scalars"]["name"] == "kse-alert-notify"
    assert norm["scalars"]["memory"] == "256m"
    assert norm["scalars"]["memory"] == norm["scalars"]["memory-swap"]
    assert any("k-stock-engine.env" in env for env in norm["env_files"])
    assert any("kse-alert.env" in env for env in norm["env_files"])
    assert not any("kis-data.env" in env for env in norm["env_files"])
    assert norm["tail"][4:6] == ["src.ops.jobs", "alert"]
    assert "k-closing-alpha" not in text
    assert "kca-" not in start.replace("kca-alert@%n.service", "")
    assert "ExecStopPost=-/usr/bin/docker rm -f kse-alert-notify" in text


def _calendars(path: Path) -> list[tuple[int, int]]:
    marks: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("OnCalendar="):
            continue
        spec = line[len("OnCalendar="):]
        clock = spec.split()[-2]
        hour, minute, *_ = clock.split(":")
        marks.append((int(hour), int(minute)))
    return marks


def _in_window(mark: tuple[int, int], window: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    return window[0] <= mark < window[1]


def test_timers_avoid_sibling_windows() -> None:
    timers = sorted(SYSTEMD.glob("kse-*.timer"))
    assert len(timers) == 7
    for path in timers:
        for mark in _calendars(path):
            assert mark not in DRIVE_SLOTS, f"{path.name}: {mark}"
            if path.name == "kse-auction-open.timer":
                continue
            for window in SIBLING_WINDOWS:
                assert not _in_window(mark, window), f"{path.name}: {mark}"
    assert (SYSTEMD / "kse-backup.timer").read_text(encoding="utf-8").count("Persistent=true") == 1


def test_vendored_deploy_library_is_pinned() -> None:
    target = REPO / "deploy" / "vps-deploy-lib.sh"
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == PINNED_LIB_SHA256
    assert "VPS_DEPLOY_CONTRACT_VERSION=1" in target.read_text(encoding="utf-8")


def test_no_order_path_in_ops_jobs() -> None:
    for path in (REPO / "src" / "ops").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "kis_broker" not in text, str(path)
        assert "place_order" not in text, str(path)
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("kis_broker" not in alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert "kis_broker" not in node.module


def test_timer_calendars_parse_with_systemd() -> None:
    specs = [line.split("=", 1)[1]
             for path in SYSTEMD.glob("kse-*.timer")
             for line in path.read_text().splitlines() if line.startswith("OnCalendar=")]
    result = subprocess.run(["systemd-analyze", "calendar", *specs], capture_output=True, text=True, check=False)  # noqa: S603,S607,S607
    assert result.returncode == 0, result.stderr


def test_collect_eve_unit_selects_relay_transport() -> None:
    norm = _normalize(_unit_effective("kse-collect-eve.service"))
    assert "KSE_KIND_TRANSPORT=relay" in norm["envs"]
    assert norm["scalars"]["memory"] == "3g"


def test_morning_unit_has_no_kind_transport() -> None:
    norm = _normalize(_unit_effective("kse-morning.service"))
    assert not any(env.startswith("KSE_KIND_TRANSPORT=") for env in norm["envs"])


def _relay_workflow() -> Path:
    return REPO / ".github" / "workflows" / "kind-relay.yml"


def _relay_text() -> str:
    return _relay_workflow().read_text(encoding="utf-8")


def test_kind_relay_workflow_schedule_and_permissions() -> None:
    text = _relay_text()
    assert "10 8 * * *" in text
    assert "5 9 * * *" in text
    assert "workflow_dispatch" in text
    assert "src.ops.kind_relay" in text
    match = re.search(r"^permissions:\n((?:[ \t]+.*\n)+)", text, re.MULTILINE)
    assert match is not None
    entries = re.findall(r"^\s+([A-Za-z0-9_-]+):\s*(\S+)", match.group(1), re.MULTILINE)
    assert dict(entries) == {"contents": "read"}
    assert "group: kind-relay" in text


def test_kind_relay_workflow_never_uploads_artifacts() -> None:
    text = _relay_text()
    assert "upload-artifact" not in text
    assert "actions/cache" not in text


def test_kind_relay_workflow_uses_only_existing_secret_names() -> None:
    text = _relay_text()
    names = set(re.findall(r"secrets\.([A-Za-z0-9_]+)", text))
    assert names
    assert names <= {"TS_OAUTH_CLIENT_ID", "TS_OAUTH_SECRET", "SSH_PRIVATE_KEY", "HOST", "USERNAME"}


def test_kind_relay_delivery_is_atomic() -> None:
    text = _relay_text()
    assert ".incoming" in text
    assert re.search(r"mv -f \S*\.incoming \S*bundle\.json", text) is not None
    assert text.index('"mkdir -p ~/k-stock-engine/data/state/kr_swing_2019_v1/kind_relay/inbox"') < text.index("          scp ")


def test_kind_relay_delivery_path_matches_importer_inbox() -> None:
    from src.config.runtime import load_runtime_config
    from src.data.kind_relay_import import relay_inbox_path
    from src.data.research_scope import load_research_scope
    from src.data.workspace import build_workspace

    config = load_runtime_config(create_directories=False)
    scope = load_research_scope(config.default_scope)
    root = Path("/data")
    inbox = relay_inbox_path(build_workspace(data_root=root, scope=scope).state_root).parent
    assert f"~/k-stock-engine/data/{inbox.relative_to(root).as_posix()}" in _relay_text()


def test_dbsec_collection_waits_for_collect_eve_and_is_enabled() -> None:
    service = (SYSTEMD / "kse-collect-dbsec.service").read_text(encoding="utf-8")
    pres = [line.strip() for line in service.splitlines() if line.strip().startswith("ExecStartPre=")]
    assert len(pres) == 2
    assert "kse-collect-eve.service" in pres[1]
    installer = INSTALLER.read_text(encoding="utf-8")
    assert "kse-*.timer" in installer
    assert '.cache/dbsec"' in installer


def test_dbsec_unit_mounts_host_broker_dir() -> None:
    norm = _normalize(_unit_effective("kse-collect-dbsec.service"))
    assert any(mount.endswith(".cache/kis:/app/.cache/kis") for mount in norm["vols"])
    assert "KIS_TOKEN_CACHE_DIR=/app/.cache/kis" in norm["envs"]


def test_dbsec_legacy_mount_is_read_only() -> None:
    with tempfile.TemporaryDirectory() as home:
        holders: dict[str, list[str]] = {}
        for unit in ("kse-collect-dbsec.service", "kse-collect-eve.service", "kse-morning.service",
                     "kse-auction-open.service", "kse-backup.service", "kse-daily-audit.service",
                     "kse-weekly-report.service"):
            norm = _normalize(_unit_effective(unit, home=home))
            legacy = [mount for mount in norm["vols"] if "/.cache/dbsec:" in mount]
            if legacy:
                holders[unit] = [mount.replace(home, "%h") for mount in legacy]
        assert holders == {"kse-collect-dbsec.service": ["%h/.cache/dbsec:/app/.cache/dbsec:ro"]}
    assert "DBSEC_TOKEN_CACHE_DIR=/app/.cache/dbsec" in _normalize(_unit_effective("kse-collect-dbsec.service"))["envs"]


def test_dbsec_unit_checks_marker_first() -> None:
    text = (SYSTEMD / "kse-collect-dbsec.service").read_text(encoding="utf-8")
    pres = [line.strip() for line in text.splitlines() if line.strip().startswith("ExecStartPre=")]
    assert pres[0] == "ExecStartPre=/usr/bin/test -f %h/.cache/kis/.host-admission"
    assert "kse-collect-eve.service" in pres[1]


def test_dbsec_unit_resources_unchanged() -> None:
    norm = _normalize(_unit_effective("kse-collect-dbsec.service"))
    assert norm["scalars"]["memory"] == "1g"
    assert norm["scalars"]["memory-swap"] == "1g"
    assert norm["scalars"]["cpus"] == "0.5"
    assert [Path(env).name for env in norm["env_files"]] == ["k-stock-engine.env"]
    assert not any("kis-data.env" in env for env in norm["env_files"])
    assert not any(env.startswith("BROKER_ADMISSION_CLASS=") for env in norm["envs"])


def test_broker_dir_units_share_one_container_path() -> None:
    found = False
    for unit in ("kse-auction-open.service", "kse-backup.service", "kse-collect-dbsec.service",
                 "kse-collect-eve.service", "kse-daily-audit.service", "kse-morning.service",
                 "kse-weekly-report.service"):
        norm = _normalize(_unit_effective(unit))
        kis_mounts = [mount for mount in norm["vols"] if "/.cache/kis:" in mount]
        if not kis_mounts:
            continue
        found = True
        assert len(kis_mounts) == 1
        assert kis_mounts[0].endswith(".cache/kis:/app/.cache/kis")
        assert "KIS_TOKEN_CACHE_DIR=/app/.cache/kis" in norm["envs"]
    assert found
