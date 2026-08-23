"""Read-only acceptance contract for the growth-route research command."""
from __future__ import annotations

import json
from types import SimpleNamespace

from src.stocks.cli import train

GROWTH_ROUTE_05_READ_ONLY_RESEARCH = "GROWTH_ROUTE_05_READ_ONLY_RESEARCH"


class _FakeSnapshot:
    costs = None


class _FakeRepository:
    def __init__(self, **kwargs):
        del kwargs

    def compose_labeled_training_snapshot(self, snapshot, **kwargs):
        del snapshot, kwargs
        return SimpleNamespace()


def _install_research_fakes(monkeypatch, evaluation) -> None:
    """Patch data resolution seams and the training-side route evaluation."""
    monkeypatch.setattr(
        train, "resolve_snapshot_for_mode", lambda *a, **k: _FakeSnapshot()
    )
    monkeypatch.setattr(train, "ResearchDataRepository", _FakeRepository)
    monkeypatch.setattr(
        train, "compose_net_alpha_training_data", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(
        "src.stocks.ml.training.evaluate_growth_route_research",
        lambda data, request: evaluation(request),
    )


def _rejected_evaluation(request):
    del request
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "certificate": {
            "passed": False,
            "reasons": [
                "non-positive-stress-lower-cagr",
                "matched-benchmark-missing",
            ],
            "base_lower_cagr": 0.12,
            "stress_lower_cagr": -0.0002,
            "matched_lower_excess_cagr": None,
        },
        "growth_route": {
            "version": "v1",
            "candidate_count": 24,
            "segment_count": 3,
            "selected_policy": None,
            "base_lower_cagr": 0.12,
            "stress_lower_cagr": -0.0002,
            "matched_lower_excess_cagr": None,
            "observed_intervals": 720,
            "invested_intervals": 96,
            "filled_orders": 0,
            "rejection_reason_counts": {
                "matched-benchmark-missing": 1,
                "non-positive-stress-lower-cagr": 1,
            },
        },
    }


def _certified_evaluation(request):
    del request
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "certificate": {
            "passed": True,
            "reasons": [],
            "base_lower_cagr": 0.0841,
            "stress_lower_cagr": 0.0198,
            "matched_lower_excess_cagr": 0.0312,
        },
        "growth_route": {
            "version": "v1",
            "candidate_count": 24,
            "segment_count": 3,
            "selected_policy": "10:5:20:lower_bound_only",
            "base_lower_cagr": 0.0841,
            "stress_lower_cagr": 0.0198,
            "matched_lower_excess_cagr": 0.0312,
            "observed_intervals": 720,
            "invested_intervals": 640,
            "filled_orders": 4599,
            "rejection_reason_counts": {},
        },
    }


def test_growth_route_05_read_only_research(monkeypatch, capsys) -> None:
    """GROWTH_ROUTE_05_READ_ONLY_RESEARCH.

    The read-only research command exits 0, emits JSON status RESEARCH_ONLY
    with artifact_published=false, and reports either certificate.passed=true
    with all three lower-CAGR predicates > 0 or certificate.passed=false with
    at least one explicit normalized rejection reason.
    """
    _install_research_fakes(monkeypatch, _rejected_evaluation)
    rc = train.main(
        [
            "--artifact-id",
            "gr05_research",
            "--snapshot-id",
            "research_snap_gr05",
            "--research-only-growth-route",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False
    rejected = payload["certificate"]
    assert rejected["passed"] is False
    assert rejected["reasons"]
    assert all(isinstance(reason, str) and reason for reason in rejected["reasons"])

    _install_research_fakes(monkeypatch, _certified_evaluation)
    assert (
        train.main(
            [
                "--artifact-id",
                "gr05_research",
                "--snapshot-id",
                "research_snap_gr05",
                "--research-only-growth-route",
            ]
        )
        == 0
    )
    certified = json.loads(capsys.readouterr().out)
    assert certified["status"] == "RESEARCH_ONLY"
    assert certified["artifact_published"] is False
    certificate = certified["certificate"]
    assert certificate["passed"] is True
    assert certificate["base_lower_cagr"] > 0.0
    assert certificate["stress_lower_cagr"] > 0.0
    assert certificate["matched_lower_excess_cagr"] > 0.0
