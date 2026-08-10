"""CLI entry point for collecting KRX and OpenDART evidence artifacts."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.stocks.data.evidence_collectors import (
    DartRetryPolicy,
    KRXEvidenceCollector,
    OpenDartEvidenceCollector,
)


def main(args: list[str] | None = None) -> int:
    """Collect one evidence artifact without modifying curated datasets."""
    parser = argparse.ArgumentParser(description="Collect stock identity, calendar, and disclosure evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    master = subparsers.add_parser("krx-master")
    master.add_argument("--date", required=True, type=date.fromisoformat)
    master.add_argument("--output", required=True, type=Path)
    master.add_argument("--market", choices=("ALL", "KOSPI", "KOSDAQ"), default="ALL")

    calendar = subparsers.add_parser("krx-calendar")
    calendar.add_argument("--start", required=True, type=date.fromisoformat)
    calendar.add_argument("--end", required=True, type=date.fromisoformat)
    calendar.add_argument("--output", required=True, type=Path)

    calendar_resume = subparsers.add_parser("krx-calendar-resume")
    calendar_resume.add_argument("--start", required=True, type=date.fromisoformat)
    calendar_resume.add_argument("--end", required=True, type=date.fromisoformat)
    calendar_resume.add_argument("--output-dir", required=True, type=Path)

    calendar_merge = subparsers.add_parser("krx-calendar-merge")
    calendar_merge.add_argument("--start", required=True, type=date.fromisoformat)
    calendar_merge.add_argument("--end", required=True, type=date.fromisoformat)
    calendar_merge.add_argument("--input-dir", required=True, type=Path)
    calendar_merge.add_argument("--output", required=True, type=Path)

    dart_resume = subparsers.add_parser("dart-disclosures-resume")
    dart_resume.add_argument("--start", required=True, type=date.fromisoformat)
    dart_resume.add_argument("--end", required=True, type=date.fromisoformat)
    dart_resume.add_argument("--output-dir", required=True, type=Path)
    dart_resume.add_argument("--page-count", type=int, default=100)
    dart_resume.add_argument("--max-attempts", type=int, default=5)
    dart_resume.add_argument("--initial-backoff-seconds", type=float, default=1.0)
    dart_resume.add_argument("--max-backoff-seconds", type=float, default=30.0)
    dart_resume.add_argument("--min-request-interval-seconds", type=float, default=0.2)

    dart_merge = subparsers.add_parser("dart-disclosures-merge")
    dart_merge.add_argument("--start", required=True, type=date.fromisoformat)
    dart_merge.add_argument("--end", required=True, type=date.fromisoformat)
    dart_merge.add_argument("--input-dir", required=True, type=Path)
    dart_merge.add_argument("--output", required=True, type=Path)

    for name in ("dart-disclosures", "dart-actions"):
        command = subparsers.add_parser(name)
        command.add_argument("--start", required=True, type=date.fromisoformat)
        command.add_argument("--end", required=True, type=date.fromisoformat)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--corp-code")

    parsed = parser.parse_args(args)
    if parsed.command == "krx-master":
        collector = KRXEvidenceCollector()
        collector.write_master_snapshot(
            parsed.output, collector.collect_master_snapshot(parsed.date, parsed.market)
        )
    elif parsed.command == "krx-calendar":
        collector = KRXEvidenceCollector()
        collector.write_calendar(
            parsed.output, collector.collect_session_calendar(parsed.start, parsed.end)
        )
    elif parsed.command == "krx-calendar-resume":
        collector = KRXEvidenceCollector()
        collector.collect_calendar_partitions(parsed.output_dir, parsed.start, parsed.end)
    elif parsed.command == "krx-calendar-merge":
        collector = KRXEvidenceCollector()
        collector.merge_calendar_partitions(
            parsed.input_dir, parsed.start, parsed.end, parsed.output
        )
    elif parsed.command == "dart-disclosures-resume":
        dart_collector = OpenDartEvidenceCollector()
        dart_collector.collect_disclosure_partitions(
            parsed.output_dir,
            parsed.start,
            parsed.end,
            page_count=parsed.page_count,
            retry_policy=DartRetryPolicy(
                max_attempts=parsed.max_attempts,
                initial_backoff_seconds=parsed.initial_backoff_seconds,
                max_backoff_seconds=parsed.max_backoff_seconds,
                min_request_interval_seconds=parsed.min_request_interval_seconds,
            ),
        )
    elif parsed.command == "dart-disclosures-merge":
        dart_collector = OpenDartEvidenceCollector()
        dart_collector.merge_disclosure_partitions(
            parsed.input_dir, parsed.start, parsed.end, parsed.output
        )
    else:
        dart_collector = OpenDartEvidenceCollector()
        if parsed.command == "dart-disclosures":
            records = dart_collector.collect_disclosures(
                parsed.start, parsed.end, corp_code=parsed.corp_code
            )
            dart_collector.write_disclosure_artifact(parsed.output, records)
        else:
            candidates = dart_collector.collect_corporate_action_candidates(
                parsed.start, parsed.end, corp_code=parsed.corp_code
            )
            dart_collector.write_corporate_action_candidates(parsed.output, candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
