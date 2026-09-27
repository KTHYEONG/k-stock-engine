"""Entry point of the remote DART worker (runs on the collection host, not locally).

Usage: python -m tools.vps_dart.worker --root ~/kse-collect --key-env KEY_ENV

The job file names the job and the scope; the worker plans from catalog
state through the shared budgeted runner like every local run.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from src.data.remote_dart_worker import run_named_job


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key-env", required=True)
    args = parser.parse_args()

    result = run_named_job(root=args.root, key_env=args.key_env)
    sys.stdout.write(json.dumps(dataclasses.asdict(result)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
