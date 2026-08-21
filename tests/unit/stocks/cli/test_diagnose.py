from __future__ import annotations

from src.stocks.cli.diagnose import main


def test_diagnose_cli_entrypoint_exists() -> None:
    assert callable(main)
