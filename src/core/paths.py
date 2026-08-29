"""Repository-local data paths for modern bounded contexts.

Filesystem locations are deterministic paths below ``data/``.  The modern
pipelines intentionally do not load ``.env`` for dataset or artifact paths;
environment variables remain limited to legacy credential adapters.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

# Retired compatibility roots. Direct training uses the canonical/derived
# roots below and does not read these deleted layouts.
STOCK_DATASET_ROOT = DATA_ROOT / "curated" / "stocks"
STOCK_ARTIFACT_ROOT = DATA_ROOT / "artifacts" / "stocks"

# Retired compatibility source; retained only for explicit migration tooling.
STOCK_FEATURE_SOURCE_ROOT = DATA_ROOT / "processed" / "features"

# Cataloged lakehouse storage hierarchy.
STOCK_CATALOG_ROOT = DATA_ROOT / "catalog" / "stocks"
STOCK_CANONICAL_ROOT = DATA_ROOT / "canonical" / "stocks"
STOCK_DERIVED_ROOT = DATA_ROOT / "derived" / "stocks"
STOCK_BASE_PANEL_ROOT = STOCK_CANONICAL_ROOT / "base_panel"
STOCK_LABEL_ROOT = STOCK_CANONICAL_ROOT / "labels"
STOCK_FEATURE_PANEL_ROOT = STOCK_DERIVED_ROOT / "features"
# Optional historical replay surface; direct training does not require it.
STOCK_SNAPSHOT_ROOT = DATA_ROOT / "snapshots" / "stocks"
# Research ledgers are human-auditable records, not dataset inputs.
STOCK_RESULTS_DOC_ROOT = PROJECT_ROOT / "docs" / "results"
# Compatibility alias for callers that imported the old constant.
STOCK_RESULTS_ROOT = STOCK_RESULTS_DOC_ROOT
STOCK_EVIDENCE_ROOT = DATA_ROOT / "evidence" / "stocks"

# Diagnostic run logs.
RUN_DIAGNOSTIC_ROOT = PROJECT_ROOT / "logs" / "runs"
