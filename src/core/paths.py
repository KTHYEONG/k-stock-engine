"""Repository-local data paths for modern bounded contexts.

Filesystem locations are deterministic paths below ``data/``.  The modern
pipelines intentionally do not load ``.env`` for dataset or artifact paths;
environment variables remain limited to legacy credential adapters.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
