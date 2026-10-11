"""Run the execution CLI as ``python -m src.execution``."""

from __future__ import annotations

from src.execution.cli import main as _cli_main


def main() -> int:
    """Dispatch the execution package command line."""
    return _cli_main()

if __name__ == "__main__":
    raise SystemExit(main())
