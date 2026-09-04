"""Run a command with dotenv values decrypted from a SOPS file in memory."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_ENTRY = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENTRY.match(line)
        if match is None:
            raise ValueError("decrypted dotenv contains an invalid assignment")
        value = match.group(2).strip()
        if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        values[match.group(1)] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command with SOPS dotenv values injected in memory")
    parser.add_argument("--env-file", type=Path, default=Path(".env.enc"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("command is required after --")
    result = subprocess.run(  # noqa: S603
        ["/usr/local/bin/sops", "--input-type", "dotenv", "--output-type", "dotenv", "-d", str(args.env_file)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write("encrypted environment could not be decrypted\n")
        return result.returncode or 1
    try:
        values = _parse_dotenv(result.stdout)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    environment = os.environ.copy()
    environment.update(values)
    return subprocess.run(command, env=environment, check=False).returncode  # noqa: S603


if __name__ == "__main__":
    raise SystemExit(main())
