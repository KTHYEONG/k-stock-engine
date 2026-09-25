"""Entry point of the remote DART worker (runs on the collection host, not locally).

Usage: python tools/vps_dart/worker.py --root ~/kse-collect --key-env OPENDART_API_KEY_2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.collection import collect_dart_financial_facts  # noqa: E402
from src.data.remote_dart_worker import run_worker  # noqa: E402
from src.integrations.dart.xbrl import DartXbrlCollector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key-env", required=True)
    args = parser.parse_args()

    def build(api_key: str, store: object, policy: dict[str, object]) -> DartXbrlCollector:
        return DartXbrlCollector(
            api_key=api_key,
            quota_store=store,  # type: ignore[arg-type]
            max_workers=1,
            min_interval=float(policy["min_interval_seconds"]),  # type: ignore[arg-type]
            daily_request_limit=int(policy["daily_budget"]),  # type: ignore[call-overload]
        )

    result = run_worker(root=args.root, key_env=args.key_env, collect=collect_dart_financial_facts, build_collector=build)
    sys.stdout.write(json.dumps(dataclasses.asdict(result)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
