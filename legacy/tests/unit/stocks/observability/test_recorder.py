from __future__ import annotations

from legacy.stocks.observability.contracts import (
    DiagnosticCategory,
    DiagnosticEvent,
    DiagnosticStage,
    DiagnosticStatus,
    RunIdentity,
)
from legacy.stocks.observability.recorder import JsonlRunDiagnostics, NullRunDiagnostics


def test_null_recorder_is_safe() -> None:
    sink = NullRunDiagnostics()
    sink.emit(DiagnosticEvent("run", 0, DiagnosticCategory.SYS, "test", DiagnosticStage.INPUT, "x", DiagnosticStatus.INFO))
    sink.close("PASS")


def test_jsonl_recorder_writes_manifest(tmp_path) -> None:
    sink = JsonlRunDiagnostics(RunIdentity("run"), tmp_path, max_stream_bytes=1024)
    sink.emit(DiagnosticEvent("run", 0, DiagnosticCategory.SYS, "test", DiagnosticStage.INPUT, "x", DiagnosticStatus.INFO))
    sink.close("PASS")

    assert (tmp_path / "run" / "sys.jsonl").exists()
    assert (tmp_path / "run" / "manifest.json").exists()
