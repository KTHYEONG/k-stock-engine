"""Back up or restore raw data trees against the Drive lake.

Usage:
  uv run python tools/devops/lake_backup.py push [--source NAME ...]
  uv run python tools/devops/lake_backup.py verify [--source NAME ...]
  uv run python tools/devops/lake_backup.py restore --source NAME --dest DIR [--pattern GLOB ...]
  uv run python tools/devops/lake_backup.py push --name alt_pilot --root scratch/alt --exclude '*.npy'

Runs under the cross-project Drive lock so it never overlaps another project's rclone job.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import sys
import tempfile
import tomllib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config.secrets import read_secret  # noqa: E402
from src.data.lake_backup import BackupSource, LocalDirStore, RcloneStore, push, restore, verify  # noqa: E402

DEFAULT_CONFIG = REPO / "config" / "data" / "lake_backup.toml"
STATUS_PATH = REPO / "data" / "state" / "lake_backup_status.json"
LOCK_WAIT_NOTE = "waiting for the shared Drive lock"


def _sources(args: argparse.Namespace, config: dict) -> list[BackupSource]:
    if args.root:
        return [BackupSource(args.name, (REPO / args.root).resolve(), exclude=tuple(args.exclude or ()))]
    wanted = set(args.source or [])
    out = []
    for row in config["source"]:
        if wanted and row["name"] not in wanted:
            continue
        out.append(
            BackupSource(
                row["name"],
                (Path(args.data_root) / Path(row["root"]).relative_to("data")).resolve()
                if getattr(args, "data_root", None) is not None else (REPO / row["root"]).resolve(),
                exclude=tuple(row.get("exclude", ())),
                immutable=bool(row.get("immutable", False)),
                segment_bytes=int(row.get("segment_mib", 256)) * 1024 * 1024,
            )
        )
    missing = wanted - {s.name for s in out}
    if missing:
        raise SystemExit(f"unknown source(s): {sorted(missing)}")
    return out


def _store(args: argparse.Namespace, config: dict, name: str):
    if args.store_dir:
        return LocalDirStore(Path(args.store_dir) / name)
    rclone = shutil.which("rclone") or str(Path.home() / ".local/bin/rclone")
    return RcloneStore(f"{config['remote']}/{name}", rclone_bin=rclone)


LOCK_WAIT_S_DEFAULT = 7200
LOCK_TIMEOUT_RC = 75


def _default_lock_file() -> Path:
    return Path(read_secret("XDG_RUNTIME_DIR", default=tempfile.gettempdir())) / "quant-gdrive.lock"


def _previous_last_ok(status_path: Path) -> str | None:
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = payload.get("last_ok_at")
    return str(value) if value else None


def _write_status(status_path: Path, *, rc: int, sources: dict[str, object]) -> None:
    finished_at = datetime.now(UTC).isoformat()
    last_ok_at = finished_at if rc == 0 else _previous_last_ok(status_path)
    payload = {"finished_at": finished_at, "rc": rc, "last_ok_at": last_ok_at, "sources": sources}
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(status_path.parent),
        prefix=".lake_backup_status.", suffix=".tmp", delete=False,
    ) as handle:
        handle.write(json.dumps(payload, indent=1))
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, status_path)


def _acquire_shared_lock(lock: object, wait_s: int, log: logging.Logger) -> bool:
    import time as _time

    deadline = _time.monotonic() + max(0, wait_s)
    log.info("[SYS] lake_backup %s", LOCK_WAIT_NOTE)
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[arg-type]
            return True
        except OSError:
            if _time.monotonic() >= deadline:
                return False
            _time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["push", "verify", "restore"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", action="append")
    parser.add_argument("--name", default="adhoc")
    parser.add_argument("--root")
    parser.add_argument("--exclude", action="append")
    parser.add_argument("--store-dir")
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--pattern", action="append")
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument("--lock-wait-s", type=int, default=LOCK_WAIT_S_DEFAULT)
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("lake_backup")
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    sources = _sources(args, config)
    if args.command == "restore" and (args.dest is None or len(sources) != 1):
        raise SystemExit("restore needs exactly one --source (or --root) and --dest")
    status_path = Path(args.status) if args.status is not None else STATUS_PATH

    rc = 0
    results: dict[str, object] = {}
    lock_path = Path(args.lock_file) if args.lock_file is not None else _default_lock_file()
    if args.no_lock:
        rc = _run_sources(args, config, sources, results, log)
    else:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock:
            if not _acquire_shared_lock(lock, int(args.lock_wait_s), log):
                log.warning("[SYS] lake_backup lock timeout path=%s wait_s=%d", str(lock_path), int(args.lock_wait_s))
                if args.command == "push":
                    _write_status(status_path, rc=LOCK_TIMEOUT_RC, sources={})
                sys.stdout.write(json.dumps({"error": "lock timeout", "lock": str(lock_path)}, ensure_ascii=False) + "\n")
                return LOCK_TIMEOUT_RC
            rc = _run_sources(args, config, sources, results, log)
    if args.command == "push":
        _write_status(status_path, rc=rc, sources=results)
    sys.stdout.write(json.dumps(results, ensure_ascii=False) + "\n")
    return rc


def _run_sources(
    args: argparse.Namespace, config: dict, sources: list[BackupSource], results: dict[str, object],
    log: logging.Logger,
) -> int:
    rc = 0
    for source in sources:
        work = None
        try:
            if not source.root.is_dir():
                raise FileNotFoundError("backup source root is missing")
            store = _store(args, config, source.name)
            work_root = (args.data_root if args.data_root is not None else REPO / "data") / "state"
            work_root.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=f"lake_backup_{source.name}_", dir=work_root))
            if args.command == "push":
                report = push(source, store, work, now=datetime.now(UTC))
                results[source.name] = asdict(report)
            elif args.command == "verify":
                problems = verify(store, work, source.name)
                results[source.name] = {"problems": problems}
                rc = rc or (1 if problems else 0)
            else:
                report = restore(source, store, args.dest, work, patterns=tuple(args.pattern or ()))
                results[source.name] = asdict(report)
            log.info("[DATA] lake_backup command=%s source=%s result=%s", args.command, source.name, results[source.name])
        except Exception as exc:
            results[source.name] = {"error": type(exc).__name__}
            log.error("[DATA] lake_backup command=%s source=%s error=%s", args.command, source.name, type(exc).__name__)
            return 1
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
