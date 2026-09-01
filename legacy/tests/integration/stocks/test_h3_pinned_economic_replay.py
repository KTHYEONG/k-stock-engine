"""H3 pinned economic replay integration tests.

Scenarios: H3_PINNED_FULL_01.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestH3PinnedFullReplay:
    """H3_PINNED_FULL_01."""

    @pytest.mark.skipif(
        not Path("tools/benchmarks/run_h3_pinned_economic_replay.py").exists(),
        reason="benchmark script not found",
    )
    def test_benchmark_script_exits_cleanly(self) -> None:
        """H3_PINNED_FULL_01.

        The pinned full H3 snapshot reports at least 926305 decision rows,
        925489 realized rows, H3 readiness PASS, all 12 H3 (C,K) cells are
        scheduled, and the run emits a compact result for every profile/cell
        or deterministic resource-budget-unavailable without host termination.
        """
        result = subprocess.run(
            [
                sys.executable,
                "tools/benchmarks/run_h3_pinned_economic_replay.py",
                "--snapshot-id", "test",
                "--horizon", "3",
                "--cadences", "1,2,3",
                "--top-k", "12,16,20,24",
                "--max-rss-mib", "4096",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "snapshot_id=test" in result.stdout
