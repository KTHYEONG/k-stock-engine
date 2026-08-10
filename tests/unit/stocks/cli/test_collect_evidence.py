"""CLI parser tests for evidence collection commands."""
from __future__ import annotations

import pytest

from src.stocks.cli import collect_evidence
from src.stocks.data.evidence_collectors import (
    KRXEvidenceCollector,
    OpenDartEvidenceCollector,
)


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


def test_collect_evidence_dart_resume_and_merge_commands(monkeypatch, tmp_path) -> None:
    # scenario: SCENARIO_DART_CLI_SINGLE_WORKER
    monkeypatch.setenv("OPENDART_API_KEY", "fixture-key")
    calls: list[object] = []

    def fake_collect(
        self: OpenDartEvidenceCollector,
        output_dir,
        start,
        end,
        *,
        page_count,
        retry_policy,
    ) -> None:
        calls.append(
            (
                "collect",
                page_count,
                retry_policy.max_attempts,
                retry_policy.initial_backoff_seconds,
                retry_policy.min_request_interval_seconds,
            )
        )

    def fake_merge(self: OpenDartEvidenceCollector, input_dir, start, end, output_path) -> None:
        calls.append("merge")

    monkeypatch.setattr(
        OpenDartEvidenceCollector, "collect_disclosure_partitions", fake_collect
    )
    monkeypatch.setattr(OpenDartEvidenceCollector, "merge_disclosure_partitions", fake_merge)

    output_dir = tmp_path / "parts"
    final = tmp_path / "disclosures.json"
    assert (
        collect_evidence.main(
            [
                "dart-disclosures-resume",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
                "--output-dir",
                str(output_dir),
                "--page-count",
                "50",
                "--max-attempts",
                "3",
                "--initial-backoff-seconds",
                "2",
                "--max-backoff-seconds",
                "10",
                "--min-request-interval-seconds",
                "0.5",
            ]
        )
        == 0
    )
    assert (
        collect_evidence.main(
            [
                "dart-disclosures-merge",
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
    assert calls == [("collect", 50, 3, 2.0, 0.5), "merge"]


def test_collect_evidence_dart_resume_exposes_no_workers_option() -> None:
    with pytest.raises(SystemExit):
        collect_evidence.main(
            [
                "dart-disclosures-resume",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
                "--output-dir",
                "parts",
                "--workers",
                "4",
            ]
        )
