"""CLI parser tests for evidence collection commands."""
from __future__ import annotations

import pytest

from src.stocks.cli import collect_evidence


def test_collect_evidence_rejects_missing_command() -> None:
    with pytest.raises(SystemExit):
        collect_evidence.main([])
