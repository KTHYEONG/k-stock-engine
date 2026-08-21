"""Read-only economic truth diagnostic for a frozen net-alpha replay.

The command deliberately reports a research status and never writes a model
artifact.  Full replay is owned by the training adapter; this entry point keeps
the acceptance thresholds and provenance visible for operators and CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate net-alpha economic recovery without publishing"
    )
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--cadence", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main(args: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(args)
    if parsed.horizon < 1 or parsed.cadence < 1 or parsed.top_k < 1:
        raise SystemExit("horizon, cadence, and top-k must be positive")
    sys.stdout.write(
        json.dumps(
            {
                "status": "RESEARCH_ONLY",
                "diagnostic": "net_alpha_economic_recovery",
                "snapshot_id": parsed.snapshot_id,
                "horizon_sessions": parsed.horizon,
                "rebalance_frequency_sessions": parsed.cadence,
                "top_k": parsed.top_k,
                "thresholds": {
                    "minimum_invested_fraction": 0.98,
                    "maximum_turnover_ratio": 0.60,
                    "minimum_absolute_cagr": 0.16,
                    "minimum_lower_cagr": 0.0,
                    "maximum_mdd": 0.12,
                },
                "artifact_published": False,
                "as_of": date.today().isoformat(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
