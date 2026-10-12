"""VPS deploy asset invariants: safe units, timer placement, pinned lib, no order path."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SYSTEMD = REPO / "deploy" / "systemd"
PINNED_LIB_SHA256 = "d3a0228685c6110b04cb6bb7f0dd20aafff24c2e1cf1adf2de7c7fb84660b191"
IMAGE = "ghcr.io/kthyeong/k-stock-engine:latest"
DRIVE_SLOTS = {(9, 15), (21, 0), (21, 30), (22, 15), (23, 30)}
SIBLING_WINDOWS = [((8, 50), (9, 40)), ((15, 10), (15, 43))]
ALLOWED_MOUNT_PREFIXES = (
    "%h/k-stock-engine/data:",
    "%h/k-stock-engine/logs:",
    "%h/.cache/kis:",
    "%h/.config/rclone:",
)


def _services() -> list[Path]:
    return sorted(path for path in SYSTEMD.glob("kse-*.service") if "@" not in path.name)


def _exec_start(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("ExecStart=")]
    assert len(lines) == 1
    return lines[0]


def test_units_are_safe_by_construction() -> None:
    services = _services()
    assert len(services) == 6
    for path in services:
        text = path.read_text(encoding="utf-8")
        assert "TimeoutStartSec=" in text
        assert "OnFailure=kse-alert@%n.service" in text
        assert "Type=oneshot" in text
        start = _exec_start(text)
        assert "--privileged" not in start
        assert IMAGE in start
        assert "uv run python -m" in start
        memory = _flag(start, "--memory=")
        swap = _flag(start, "--memory-swap=")
        assert memory
        assert memory == swap
        assert "-e HOME=/tmp" in start
        assert "-e UV_CACHE_DIR=/tmp/uv-cache" in start
        assert "Environment=TZ=Asia/Seoul" in text
        mounts = [token for token in start.split() if token in ("-v", "--volume")]
        assert mounts
        for token in _mounts(start):
            assert token.startswith(ALLOWED_MOUNT_PREFIXES), f"{path.name}: {token}"
            if token.startswith("%h/.config/rclone:"):
                assert path.name == "kse-backup.service"
                assert ":ro" in token
        if path.name == "kse-backup.service":
            assert start.startswith("ExecStart=/usr/bin/flock -w 7200 %t/quant-gdrive.lock ")
            assert "--no-lock" in start
        assert f"ExecStopPost=-/usr/bin/docker rm -f {path.stem}" in text


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
    assert "--rm -i" in start
    assert "--name kse-alert-notify " in start
    assert "--memory=256m" in start
    assert _flag(start, "--memory=") == _flag(start, "--memory-swap=")
    assert "--env-file %h/quant-secrets/k-stock-engine.env" in start
    assert "kis-data.env" not in start
    assert "src.ops.jobs alert" in start
    assert '--unit "$$0"' in start
    assert '--result "$$(systemctl --user show -p Result --value "$$0")"' in start
    assert '--exit-status "$$(systemctl --user show -p ExecMainStatus --value "$$0")"' in start
    assert start.rstrip().endswith("%i")
    for token in _mounts(start):
        assert token.startswith(ALLOWED_MOUNT_PREFIXES), f"{path.name}: {token}"
    assert "k-closing-alpha" not in text
    assert "kca-" not in start.replace("kca-alert@%n.service", "")
    assert "ExecStopPost=-/usr/bin/docker rm -f kse-alert-notify" in text
    for service in _services():
        assert f"--name {service.stem} " not in start


def _flag(start: str, name: str) -> str:
    for token in start.split():
        if token.startswith(name):
            return token[len(name):]
    return ""


def _mounts(start: str) -> list[str]:
    parts = start.split()
    out: list[str] = []
    for pos, token in enumerate(parts):
        if token in ("-v", "--volume") and pos + 1 < len(parts):
            out.append(parts[pos + 1])
    return out


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
    assert len(timers) == 6
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
    result = subprocess.run(["systemd-analyze", "calendar", *specs], capture_output=True, text=True, check=False)  # noqa: S603,S607
    assert result.returncode == 0, result.stderr
