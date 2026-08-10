"""CLI parser tests for evidence collection commands."""
from __future__ import annotations

import pytest

from src.stocks.cli import collect_evidence
from src.stocks.data.evidence_collectors import KRXEvidenceCollector


def test_collect_evidence_rejects_missing_command() -> None:
    with pytest.raises(SystemExit):
        collect_evidence.main([])


def test_collect_evidence_requires_paths_for_new_commands() -> None:
    with pytest.raises(SystemExit):
        collect_evidence.main(
            ["krx-calendar-resume", "--start", "2024-01-01", "--end", "2024-01-31"]
        )
    with pytest.raises(SystemExit):
        collect_evidence.main(
            [
                "krx-calendar-merge",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
                "--input-dir",
                "parts",
            ]
        )


def test_collect_evidence_resume_and_merge_commands(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KRX_OPENAPI_KEY", "fixture-key")
    calls: list[str] = []

    def fake_collect(
        self: KRXEvidenceCollector, output_dir, start, end
    ) -> None:
        calls.append("collect")

    def fake_merge(self: KRXEvidenceCollector, input_dir, start, end, output_path) -> None:
        calls.append("merge")

    monkeypatch.setattr(KRXEvidenceCollector, "collect_calendar_partitions", fake_collect)
    monkeypatch.setattr(KRXEvidenceCollector, "merge_calendar_partitions", fake_merge)

    output_dir = tmp_path / "parts"
    final = tmp_path / "calendar.json"
    assert (
        collect_evidence.main(
            [
                "krx-calendar-resume",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert (
        collect_evidence.main(
            [
                "krx-calendar-merge",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
                "--input-dir",
                str(output_dir),
                "--output",
                str(final),
            ]
        )
        == 0
    )
    assert calls == ["collect", "merge"]
