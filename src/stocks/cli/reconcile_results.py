"""Rebuild the bounded ML result ledger from published artifact files.

Recovery is best-effort: reconstructed records carry the evidence persisted in
each artifact's ``manifest.json``/``metrics.json`` but no request or runtime
context. Artifact files under ``data/artifacts`` remain the source of truth.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.paths import PROJECT_ROOT, STOCK_ARTIFACT_ROOT
from src.stocks.ml.result_ledger import MlResultLedger
from src.stocks.research.artifacts import ModelArtifactRegistry

logger = logging.getLogger("stocks.cli.reconcile_results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the bounded ML result ledger from artifact files"
    )
    parser.add_argument("--registry", type=Path, default=STOCK_ARTIFACT_ROOT)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "docs" / "results",
        help="directory owning the generated result ledger (default docs/results)",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(args)
    ledger = MlResultLedger(parsed.results_root)
    stats = ledger.rebuild_from_registry(ModelArtifactRegistry(parsed.registry))
    logger.info("reconciled result ledger: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
