"""Read-only acceptance contract for the growth-route research command."""
from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from src.stocks.cli import train
from src.stocks.data.contracts import CoverageRange
from src.stocks.ml.result_ledger import MlResultLedger
from src.stocks.research.artifacts import ModelArtifactRegistry

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
        lambda data, request, *, registry: evaluation(request),
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


def test_growth_route_registry_01_is_caller_owned(monkeypatch, capsys) -> None:
    """GROWTH_ROUTE_REGISTRY_01: CLI owns the replay artifact registry."""
    received = []

    def evaluation(data, request, *, registry):
        del data, request
        received.append(registry)
        return _rejected_evaluation(None)

    monkeypatch.setattr(
        train, "resolve_snapshot_for_mode", lambda *a, **k: _FakeSnapshot()
    )
    monkeypatch.setattr(train, "ResearchDataRepository", _FakeRepository)
    monkeypatch.setattr(
        train, "compose_net_alpha_training_data", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(
        "src.stocks.ml.training.evaluate_growth_route_research", evaluation
    )

    assert train.main([
        "--artifact-id", "growth_registry", "--snapshot-id", "research_snap",
        "--research-only-growth-route",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False
    assert len(received) == 1
    assert isinstance(received[0], ModelArtifactRegistry)


TEMPORAL_WINDOW_05_READ_ONLY = "TEMPORAL_WINDOW_05_READ_ONLY"
TEMPORAL_WINDOW_08_SINGLE_WINDOW_REGRESSION = (
    "TEMPORAL_WINDOW_08_SINGLE_WINDOW_REGRESSION"
)


class _HashBoundSnapshot:
    def __init__(self) -> None:
        self.costs = SimpleNamespace(path="costs/fake.json", content_hash="hash123")
        self.execution_range = CoverageRange(
            start=date(2020, 1, 1), end=date(2026, 3, 10)
        )


class _CostEvidence:
    base_schedule_kind = "base"
    stress_schedule_kind = "stress"
    base_liquidity_model = SimpleNamespace(name="base_liq")
    stress_liquidity_model = SimpleNamespace(name="stress_liq")

    @staticmethod
    def base_schedule():
        return SimpleNamespace(kind="base")

    @staticmethod
    def stress_schedule():
        return SimpleNamespace(kind="stress")


def _install_temporal_readonly_seams(monkeypatch, evaluation_payload) -> dict:
    """Patch snapshot seams plus registry/ledger writers as tripwires."""
    monkeypatch.setattr(
        train,
        "resolve_snapshot_for_mode",
        lambda *a, **k: _HashBoundSnapshot(),
    )
    monkeypatch.setattr(train, "ResearchDataRepository", _FakeRepository)
    monkeypatch.setattr(
        train,
        "compose_net_alpha_training_data",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(train, "load_cost_evidence", lambda path, rng: _CostEvidence)

    def _forbidden_writer(*args, **kwargs):
        raise AssertionError("read-only study must never persist evidence")

    monkeypatch.setattr(ModelArtifactRegistry, "publish", _forbidden_writer)
    monkeypatch.setattr(ModelArtifactRegistry, "write_metrics", _forbidden_writer)
    monkeypatch.setattr(
        ModelArtifactRegistry, "write_forward_holdout", _forbidden_writer
    )
    monkeypatch.setattr(MlResultLedger, "record_completed", _forbidden_writer)
    monkeypatch.setattr(MlResultLedger, "record_failed", _forbidden_writer)

    def fake_study(data, request, settings, *, registry):
        del data, request, settings, registry
        return evaluation_payload

    monkeypatch.setattr(
        "src.stocks.ml.window_research.evaluate_temporal_window_study", fake_study
    )


def test_temporal_window_05_read_only_cli(monkeypatch, capsys) -> None:
    """TEMPORAL_WINDOW_05_READ_ONLY.

    The snapshot-backed temporal-window study emits exactly one bounded JSON
    object with status=RESEARCH_ONLY and artifact_published=false, carries no
    holdout outcome arrays or per-instrument values, and performs zero
    registry publishes and zero result-ledger writes.
    """
    payload_stub = {
        "study_complete": False,
        "next_action": "research-signal-objective",
        "common_fold_count": 3,
        "adjusted_bootstrap_alpha": 0.0125,
        "bootstrap_resamples": 2000,
        "recommended_lookback_sessions": None,
        "recommended_is_expanding": False,
        "rejection_reason_counts": {"invested-coverage-insufficient": 4},
        "candidates": [
            {
                "training_lookback_sessions": 504,
                "is_expanding": False,
                "status": "RESEARCH_ONLY",
                "certificate": {
                    "passed": False,
                    "reasons": ["invested-coverage-insufficient"],
                    "base_lower_cagr": None,
                },
                "growth_route": {
                    "candidate_count": 24,
                    "rejection_reason_counts": {
                        "invested-coverage-insufficient": 1
                    },
                },
            }
        ],
    }
    _install_temporal_readonly_seams(monkeypatch, payload_stub)

    rc = train.main(
        [
            "--artifact-id",
            "tw05_research",
            "--snapshot-id",
            "research_snap_tw05",
            "--mode",
            "research",
            "--research-only-temporal-window-study",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False

    forbidden = {
        "orders",
        "holdout_outcomes",
        "scores",
        "returns",
        "labels",
        "per_instrument_values",
        "instrument_values",
    }
    assert not forbidden.intersection(payload)
    for candidate in payload["candidates"]:
        assert not forbidden.intersection(candidate)


def test_temporal_window_08_single_window_regression(monkeypatch, capsys) -> None:
    """TEMPORAL_WINDOW_08_SINGLE_WINDOW_REGRESSION.

    The single-window growth-route caller keeps its prior fold boundary: it
    never passes min_oof_train_sessions, whose signature default stays None.
    """
    import inspect

    from src.stocks.ml.training import evaluate_growth_route_research

    parameter = inspect.signature(evaluate_growth_route_research).parameters[
        "min_oof_train_sessions"
    ]
    assert parameter.default is None

    received_kwargs: list[dict] = []

    def evaluation(data, request, **kwargs):
        del data, request
        received_kwargs.append(kwargs)
        return _rejected_evaluation(None)

    monkeypatch.setattr(
        train, "resolve_snapshot_for_mode", lambda *a, **k: _FakeSnapshot()
    )
    monkeypatch.setattr(train, "ResearchDataRepository", _FakeRepository)
    monkeypatch.setattr(
        train,
        "compose_net_alpha_training_data",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr("src.stocks.ml.training.evaluate_growth_route_research", evaluation)

    assert train.main([
        "--artifact-id", "tw08_regression", "--snapshot-id", "research_snap",
        "--research-only-growth-route",
    ]) == 0
    json.loads(capsys.readouterr().out)
    assert len(received_kwargs) == 1
    assert "min_oof_train_sessions" not in received_kwargs[0]
