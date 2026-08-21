"""Read-only acceptance contract for the economic recovery diagnostic."""
from __future__ import annotations

import json

from tools.benchmarks.validate_net_alpha_economic_recovery import main


def test_economic_truth_04(capsys) -> None:
    """ECONOMIC_TRUTH_04: diagnostics never publish an artifact."""
    assert main(
        [
            "--snapshot-id",
            "research_provisional_20160104_20260310_net_alpha_v1_backfill_durable_v1",
            "--horizon",
            "10",
            "--cadence",
            "5",
            "--top-k",
            "20",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False
    assert payload["thresholds"]["minimum_invested_fraction"] == 0.98
