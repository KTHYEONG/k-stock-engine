"""Execution package entry point."""

from __future__ import annotations

import pytest

def test_execution_module_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["src.execution", "auction-open", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("src.execution", run_name="__main__", alter_sys=True)
    assert exc.value.code == 0
    assert "--decisions-dir" in capsys.readouterr().out
