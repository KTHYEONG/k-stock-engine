"""CLI interface for bounded diagnostic queries and reports.

``main`` is the entry point for the ``diagnose`` CLI command that analyzes
a run's diagnostic bundles and produces a deterministic report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legacy.stocks.observability.report import analyze_run


def main(args: list[str] | None = None) -> int:
    """Entry point for the diagnostic CLI.

    Parameters
    ----------
    args:
        Command-line arguments.  When ``None``, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for analysis failure, 2 for argument error.
    """
    parser = argparse.ArgumentParser(
        description="Analyze diagnostic bundles for a training/backtest run."
    )
    parser.add_argument(
        "run_id",
        help="The run identifier to analyze.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Root directory for run logs (default: logs/runs).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json).",
    )

    parsed = parser.parse_args(args)

    if parsed.runs_root is None:
        from legacy.stocks.paths import RUN_DIAGNOSTIC_ROOT

        runs_root = RUN_DIAGNOSTIC_ROOT
    else:
        runs_root = parsed.runs_root

    if not runs_root.exists():
        sys.stderr.write(f"Error: runs root {runs_root} does not exist\n")
        return 1

    report = analyze_run(runs_root, parsed.run_id)

    if parsed.format == "json":
        sys.stdout.write(report.to_json() + "\n")
    else:
        lines = [f"Run: {report.run_id}", f"Status: {report.status}"]
        if report.first_fail_sequence is not None:
            lines.append(
                f"First FAIL: seq={report.first_fail_sequence} "
                f"event={report.first_fail_event} "
                f"component={report.first_fail_component}"
            )
        if report.stage_summaries:
            lines.append("\nStage summaries:")
            lines.extend(
                f"  {s.stage}: {s.elapsed_ms:.0f}ms ({s.event_count} events)"
                for s in report.stage_summaries
            )
        if report.missing_checkpoints:
            lines.append(f"\nMissing checkpoints: {', '.join(report.missing_checkpoints)}")
        sys.stdout.write("\n".join(lines) + "\n")

    return 0 if report.status != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
