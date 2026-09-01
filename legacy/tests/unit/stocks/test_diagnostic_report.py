"""Tests for result ledger diagnostic report integration.

Scenarios:
- LEDGER_11: DiagnosticReport integrates with result ledger projections.
"""
from __future__ import annotations

from legacy.stocks.observability.report import DiagnosticReport


class TestDiagnosticReportIntegration:
    """DiagnosticReport integrates with result ledger."""

    def test_report_to_json_structure(self) -> None:
        report = DiagnosticReport(
            run_id="test_run",
            status="PASS",
            first_fail_sequence=None,
            first_fail_event=None,
            first_fail_component=None,
        )
        json_str = report.to_json()
        import json
        parsed = json.loads(json_str)
        assert parsed["run_id"] == "test_run"
        assert parsed["status"] == "PASS"
        assert parsed["first_fail_sequence"] is None

    def test_report_with_fail(self) -> None:
        report = DiagnosticReport(
            run_id="test_run",
            status="FAIL",
            first_fail_sequence=5,
            first_fail_event="candidate_select",
            first_fail_component="ml.training",
            missing_checkpoints=["settlement", "terminal"],
        )
        json_str = report.to_json()
        import json
        parsed = json.loads(json_str)
        assert parsed["status"] == "FAIL"
        assert parsed["first_fail_sequence"] == 5
        assert "settlement" in parsed["missing_checkpoints"]
