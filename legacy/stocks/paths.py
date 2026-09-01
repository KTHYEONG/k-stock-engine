"""Archived stock-specific paths (legacy)."""
from __future__ import annotations

from pathlib import Path

from src.core.paths import DATA_ROOT, PROJECT_ROOT

STOCK_DATASET_ROOT = DATA_ROOT / "curated" / "stocks"
STOCK_ARTIFACT_ROOT = DATA_ROOT / "artifacts" / "stocks"
STOCK_FEATURE_SOURCE_ROOT = DATA_ROOT / "processed" / "features"
STOCK_CATALOG_ROOT = DATA_ROOT / "catalog" / "stocks"
STOCK_CANONICAL_ROOT = DATA_ROOT / "canonical" / "stocks"
STOCK_DERIVED_ROOT = DATA_ROOT / "derived" / "stocks"
STOCK_BASE_PANEL_ROOT = STOCK_CANONICAL_ROOT / "base_panel"
STOCK_LABEL_ROOT = STOCK_CANONICAL_ROOT / "labels"
STOCK_FEATURE_PANEL_ROOT = STOCK_DERIVED_ROOT / "features"
STOCK_SNAPSHOT_ROOT = DATA_ROOT / "snapshots" / "stocks"
STOCK_RESULTS_DOC_ROOT = PROJECT_ROOT / "docs" / "results"
STOCK_RESULTS_ROOT = STOCK_RESULTS_DOC_ROOT
STOCK_EVIDENCE_ROOT = DATA_ROOT / "evidence" / "stocks"
RUN_DIAGNOSTIC_ROOT = PROJECT_ROOT / "logs" / "runs"
