"""Invariant guards for log cleanup roots."""

from __future__ import annotations

from pathlib import Path


def test_logs_dir_resolves_to_repo_root() -> None:
    """Logs dir is repo root invariant."""
    from tools.devops import clean_logs

    repo_root = Path(clean_logs.__file__).resolve().parents[2]
    assert repo_root / "logs" == clean_logs.LOGS_DIR
