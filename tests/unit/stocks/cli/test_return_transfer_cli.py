"""CLI contract for the read-only return-transfer study."""

from __future__ import annotations

from types import SimpleNamespace

from src.stocks.cli.train import run_research_only_return_transfer_study
from src.stocks.ml.contracts import NetAlphaTrainingRequest


def test_RETURN_TRANSFER_08_READ_ONLY_CLI() -> None:
    """RETURN_TRANSFER_08_READ_ONLY_CLI: bounded no-publication envelope."""
    parsed = SimpleNamespace(snapshot_id=None, registry="tmp")
    payload = run_research_only_return_transfer_study(
        parsed, NetAlphaTrainingRequest(artifact_id="rt08")
    )
    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False
    assert {"DATA", "ALGO", "EVAL", "SYS"} <= set(payload)
    assert "scores" not in payload
    assert "labels" not in payload
