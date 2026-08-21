"""Runtime configuration for stock bounded-context operations.

Operational settings (paths, DEBUG mode, retention) use pydantic-settings.
Financial defaults are NOT owned here; they belong to
``CanonicalResearchProfile`` in ``research.py``.
"""
from __future__ import annotations

from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore[no-redef]

from src.core.paths import PROJECT_ROOT, RUN_DIAGNOSTIC_ROOT


class StockRuntimeSettings(BaseSettings):
    """Operational runtime settings for stock pipelines.

    All fields are keyword-only and have safe defaults.  Financial thresholds
    are deliberately absent; they live in ``CanonicalResearchProfile``.
    """

    model_config = {"env_prefix": "STOCK_", "env_file": ".env"}

    diagnostics_enabled: bool = False
    diagnostics_required: bool = False
    diagnostics_root: Path = RUN_DIAGNOSTIC_ROOT
    results_root: Path = PROJECT_ROOT / "data" / "results" / "stocks"
    evidence_root: Path = PROJECT_ROOT / "data" / "evidence" / "stocks"
    max_rss_mib: float | None = None
    debug: bool = False
