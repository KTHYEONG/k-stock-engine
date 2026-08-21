from __future__ import annotations

from src.stocks.observability.report import analyze_run


def test_missing_run_reports_required_checkpoints(tmp_path) -> None:
    report = analyze_run(tmp_path, "missing")

    assert report.status == "UNKNOWN"
    assert "input" in report.missing_checkpoints
