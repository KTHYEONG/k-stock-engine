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
                (REPO / row["root"]).resolve(),
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("lake_backup")
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    sources = _sources(args, config)
    if args.command == "restore" and (args.dest is None or len(sources) != 1):
        raise SystemExit("restore needs exactly one --source (or --root) and --dest")

    lock_dir = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir())
    rc = 0
    results: dict[str, object] = {}
    with open(lock_dir / "quant-gdrive.lock", "w") as lock:
        log.info("[SYS] lake_backup %s", LOCK_WAIT_NOTE)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for source in sources:
            store = _store(args, config, source.name)
            work = Path(tempfile.mkdtemp(prefix=f"lake_backup_{source.name}_", dir=REPO / "data" / "state"))
            try:
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
            finally:
                shutil.rmtree(work, ignore_errors=True)
    if args.command == "push":
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(
            json.dumps({"finished_at": datetime.now(UTC).isoformat(), "rc": rc, "sources": results}, indent=1),
            encoding="utf-8",
        )
    print(json.dumps(results, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
