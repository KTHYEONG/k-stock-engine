"""Provision the workstation-managed VPS secret file over SSH.

Builds the ``k-stock-engine.env`` file (0600) from a local KEY=VALUE secret
file through an explicit allow-list, then pushes it over SSH stdin only.
Only key names are ever logged; values never leave the SSH channel.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CODE_OWNED = {"KIS_POOL_SLOT": "5", "KIS_REQUIRE_SHARED_ADMISSION": "true"}
KIS_DATA_MARKER = "kis-data"


def _allow_list() -> list[str]:
    config = tomllib.loads((REPO / "config" / "data" / "provision_env.toml").read_text(encoding="utf-8"))
    return list(config["secret_keys"])


def _read_secret_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _render_env(values: dict[str, str], allowed: list[str]) -> str:
    missing = [key for key in allowed if not values.get(key, "").strip()]
    if missing:
        raise SystemExit(f"missing keys in secret file: {', '.join(missing)}")
    if any("\n" in values[key] or "\r" in values[key] for key in allowed):
        raise SystemExit("secret values must be single-line")
    lines = [f"{key}={values[key]}" for key in allowed]
    lines.extend(f"{key}={value}" for key, value in sorted(CODE_OWNED.items()))
    return "\n".join(lines) + "\n"


def _push_over_ssh(*, host: str, user: str, ssh_key: str | None, dest: str, payload: str, key_names: list[str]) -> None:
    if KIS_DATA_MARKER in dest:
        raise SystemExit("refusing to write the shared pool file")
    quoted_dest = shlex.quote(dest)
    remote = (
        f"umask 077 && mkdir -p $(dirname {quoted_dest})"
        f" && cat > {quoted_dest}.tmp && chmod 600 {quoted_dest}.tmp"
        f" && mv {quoted_dest}.tmp {quoted_dest}"
    )
    command = ["ssh"]
    if ssh_key:
        command += ["-i", ssh_key]
    command += [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        f"{user}@{host}", remote,
    ]
    sys.stdout.write(f"[SYS] provision_env keys={','.join(key_names)}\n")
    completed = subprocess.run(command, input=payload, capture_output=True, text=True, check=False)  # noqa: S603
    if completed.returncode != 0:
        raise SystemExit(f"ssh push failed rc={completed.returncode}")
    sys.stdout.write(f"[SYS] provision_env pushed keys={','.join(key_names)}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secrets-file", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--dest", default=str(Path("/home") / "ubuntu" / "quant-secrets" / "k-stock-engine.env"))
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    allowed = _allow_list()
    values = _read_secret_file(Path(args.secrets_file))
    payload = _render_env(values, allowed)
    if args.dry_run:
        sys.stdout.write(f"[SYS] provision_env dry-run keys={','.join(allowed)}\n")
        return 0
    _push_over_ssh(host=args.host, user=args.user, ssh_key=args.ssh_key, dest=args.dest, payload=payload,
                   key_names=allowed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
