"""Read-only model artifact retention checks with explicit apply semantics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legacy.stocks.paths import STOCK_ARTIFACT_ROOT
from legacy.stocks.research.artifacts import ModelArtifactRegistry


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect model artifact retention")
    parser.add_argument("--artifact-root", type=Path, default=STOCK_ARTIFACT_ROOT)
    parser.add_argument("--keep", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="report candidates without deleting")
    parser.add_argument("--apply", action="store_true", help="delete listed candidates")
    parsed = parser.parse_args(args)

    registry = ModelArtifactRegistry(parsed.artifact_root)
    candidates = registry.retention_candidates(parsed.keep)
    if not candidates:
        sys.stdout.write("no artifact retention candidates\n")
        return 0
    for candidate in candidates:
        sys.stdout.write(f"{candidate.artifact_id}\t{candidate.reason}\n")
    if parsed.apply:
        deleted = registry.prune(candidates, apply=True)
        sys.stdout.write(f"deleted {deleted} artifact(s)\n")
    else:
        sys.stdout.write("dry-run: no artifacts changed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
