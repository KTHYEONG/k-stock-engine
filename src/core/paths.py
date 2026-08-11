"""Repository-local data paths for modern bounded contexts.

Filesystem locations are deterministic paths below ``data/``.  The modern
pipelines intentionally do not load ``.env`` for dataset or artifact paths;
environment variables remain limited to legacy credential adapters.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

# Canonical, manifest-backed stock inputs and model outputs.
STOCK_DATASET_ROOT = DATA_ROOT / "curated" / "stocks"
STOCK_ARTIFACT_ROOT = DATA_ROOT / "artifacts" / "stocks"

# Existing preprocessed files are an input source for curation, not a
# replacement for the manifest-backed dataset store above.
STOCK_FEATURE_SOURCE_ROOT = DATA_ROOT / "processed" / "features"

# Cataloged lakehouse roots (see docs/specs/stock_research_data_foundation.md).
# The catalog is the append-only registry; canonical holds immutable base/label
# datasets; derived holds reusable feature panels; snapshots hold the only
# train/backtest input selectors; results hold backtest reports/equity curves.
STOCK_CATALOG_ROOT = DATA_ROOT / "catalog" / "stocks"
STOCK_CANONICAL_ROOT = DATA_ROOT / "canonical" / "stocks"
STOCK_DERIVED_ROOT = DATA_ROOT / "derived" / "stocks"
STOCK_BASE_PANEL_ROOT = STOCK_CANONICAL_ROOT / "base_panel"
STOCK_LABEL_ROOT = STOCK_CANONICAL_ROOT / "labels"
STOCK_FEATURE_PANEL_ROOT = STOCK_DERIVED_ROOT / "features"
STOCK_SNAPSHOT_ROOT = DATA_ROOT / "snapshots" / "stocks"
STOCK_RESULTS_ROOT = DATA_ROOT / "results" / "stocks"
STOCK_EVIDENCE_ROOT = DATA_ROOT / "evidence" / "stocks"
