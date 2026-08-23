"""Bounded ML result ledger: projection, retention, path safety, recovery."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.ml.data import compose_net_alpha_training_data
from src.stocks.ml.result_ledger import (
    MAX_RECORD_BYTES,
    SCHEMA_FILENAME,
    CostRunContext,
    MlResultLedger,
    MlRunContext,
    SCHEMA_VERSION,
    _digest_summary,
    _encode,
    summarize_numeric,
)
from src.stocks.research.artifacts import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    ModelArtifactRegistry,
)
from src.stocks.research.models import ModelManifest
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)


class _MutableClock:
    def __init__(self, start: datetime) -> None:
        self.value = start

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


def _context(
    artifact_id: str = "na_ledger",
    snapshot_id: str = "research_snap_1",
    started_at: datetime | None = None,
) -> MlRunContext:
    df = stock_net_alpha_composed_df(n_sessions=60, n_tickers=4, audit_clean=True)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, datetime(2024, 12, 31, tzinfo=UTC), (3, 5, 8, 10, 15, 20)
    )
    return MlRunContext.from_cli(
        request=NetAlphaTrainingRequest(
            artifact_id=artifact_id, candidate_horizon_sessions=(3, 5, 8, 10, 15, 20)
        ),
        snapshot_id=snapshot_id,
        data=data,
        cost_context=CostRunContext(
            cost_schedule_kind="base",
            cost_evidence_path="cost_evidence.json",
            cost_evidence_hash="cost-hash-1",
            has_liquidity_model=True,
        ),
        started_at=started_at or datetime(2024, 1, 1, tzinfo=UTC),
    )


def _manifest(
    artifact_id: str, model_type: str = "net_alpha_elastic_net"
) -> ModelManifest:
    return ModelManifest(
        artifact_id=artifact_id,
        asset_kind="stock",
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="schema-hash-1",
        universe_policy_hash="universe-hash-1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
        model_type=model_type,
    )


def _default_metrics(
    model_type: str = "net_alpha_elastic_net",
) -> dict[str, object]:
    return {
        "promoted": model_type != "no_trade",
        "no_trade": model_type == "no_trade",
        "model_type": model_type,
        "promotion_reasons": (
            ["no-horizon-evidence"]
            if model_type == "no_trade"
            else ["primary=3 from lower bounds {3:0.01}"]
        ),
        "gates": {
            "passed": model_type != "no_trade",
            "reasons": ["no-horizon-evidence"],
        },
        "horizon_selection": {
            "primary_horizon_sessions": None if model_type == "no_trade" else 3,
            "secondary_horizon_sessions": None,
            "lower_bounds": {"3": 0.01, "5": -0.001},
            "effective_horizon_count": 1.0,
            "selection_reasons": ["primary=3"],
            "correlation_matrix": {},
        },
        "replay": {
            "order_count": 120,
            "block_count": 4,
            "decisions": [1, 1, 1, 1],
            "period_net_returns": [0.01, 0.02, 0.03, 0.04],
        },
        "holdout": {
            "passed": True,
            "reason": "",
            "order_count": 40,
            "block_count": 3,
            "certificate": {
                "passed": True,
                "reasons": [],
                "base": {
                    "passed": True,
                    "reasons": [],
                    "cagr": 0.15,
                    "lower_cagr": 0.02,
                    "mdd": 0.08,
                    "calmar": 1.875,
                },
                "stress": {
                    "passed": True,
                    "reasons": [],
                    "cagr": 0.11,
                    "lower_cagr": 0.01,
                    "mdd": 0.10,
                    "calmar": 1.1,
                },
            },
            "cohorts": {
                "scored_sessions": 60,
                "realized_sessions": 60,
                "eligible_sessions": 12,
                "orders": 40,
                "period_count": 4,
                "observed_sessions": 24,
                "active_cohort_count": 3,
                "missing_realized_cohorts": 0,
            },
            "eligibility": {
                "eligible_from": "2024-11-01T00:00:00+00:00",
                "eligible_to": "2024-12-31T00:00:00+00:00",
            },
        },
        "run_observability": {
            "phases": [
                {
                    "name": "integrity_audit",
                    "elapsed_ms": 10,
                    "peak_rss_mib": 100.0,
                    "passed": True,
                }
            ],
            "horizons": [
                {
                    "horizon_sessions": 3,
                    "model_family": "net_alpha_elastic_net",
                    "admission": "eligible",
                    "usable_fold_count": 3,
                    "fold_rank_ics": [0.05, 0.06, 0.07],
                }
            ],
        },
    }


def _write_artifact(
    root: Path,
    artifact_id: str,
    *,
    model_type: str = "net_alpha_elastic_net",
    metrics: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
) -> None:
    artifact_dir = root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    default_manifest: dict[str, object] = {
        "artifact_id": artifact_id,
        "asset_kind": "STOCK",
        "feature_set": "stock_net_alpha_v1",
        "feature_schema_hash": "schema-hash-1",
        "universe_policy_hash": "universe-hash-1",
        "label_definition": "net_alpha_o2o",
        "label_horizon_sessions": 5,
        "eligible_from": "2024-01-01T00:00:00+00:00",
        "eligible_to": "2024-12-31T00:00:00+00:00",
        "model_type": model_type,
        "params": {},
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest or default_manifest), encoding="utf-8"
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics or _default_metrics(model_type)), encoding="utf-8"
    )


def _latest(results_root: Path) -> dict[str, object]:
    return json.loads((results_root / "ml_runs" / "latest.json").read_text())


def test_summarize_numeric_finite_only() -> None:
    result = summarize_numeric([1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 4
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["median"] == 2.5
    empty = summarize_numeric([])
    assert empty["count"] == 0
    assert empty["mean"] is None
    assert empty["std"] is None
    non_finite = summarize_numeric([float("nan"), float("inf"), 1.0, None])
    assert non_finite["count"] == 1
    assert non_finite["mean"] == 1.0


def test_observability_projects_bounded_diagnostics_summary() -> None:
    from src.stocks.ml.result_ledger import _observability_from

    telemetry = {
        "phases": [
            {"name": "feature_transform", "schema_fingerprint": "abc123"},
            {
                "name": "horizon_discovery",
                "path_evaluation_count": 48,
                "path_evaluation_bound": 72,
                "evidence_horizons": [3],
                "diagnostics_count": 6,
            },
            {
                "name": "primary_selection",
                "primary_horizon_sessions": 3,
                "rankability_reason": "challenger-skipped:no-rankability-evidence",
            },
            {
                "name": "model_comparison",
                "selected_model_type": "net_alpha_elastic_net",
                "challenger_failure_reason": "challenger-skipped:no-rankability-evidence",
            },
            {
                "name": "artifact_publish",
                "promoted": True,
                "model_type": "net_alpha_elastic_net",
            },
        ],
        "horizons": [],
    }
    observability = _observability_from({}, telemetry)
    summary = observability["summary"]
    assert isinstance(summary, dict)
    assert summary["schema_fingerprint"] == "abc123"
    assert summary["path_evaluation_count"] == 48
    assert summary["path_evaluation_bound"] == 72
    assert summary["evidence_horizons"] == [3]
    assert summary["primary_horizon_sessions"] == 3
    assert summary["rankability_reason"].startswith("challenger-skipped")
    assert summary["selected_model_type"] == "net_alpha_elastic_net"
    assert "period_net_returns" not in summary
    assert "raw" not in json.dumps(summary).lower()


def test_record_completed_projects_canonical_fields(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_completed"
    _write_artifact(registry.root, artifact_id)
    context = _context(artifact_id)
    ledger_inst = MlResultLedger(
        tmp_path / "results", clock=lambda: datetime(2024, 1, 2, tzinfo=UTC)
    )
    ledger_inst.record_completed(context, _manifest(artifact_id), registry)
    latest = _latest(tmp_path / "results")
    assert latest["schema_version"] == SCHEMA_VERSION
    assert latest["artifact_id"] == artifact_id
    assert latest["status"] == "completed"
    assert latest["started_at"] == "2024-01-01T00:00:00+00:00"
    assert latest["finished_at"] == "2024-01-02T00:00:00+00:00"
    assert latest["runtime"]["elapsed_ms"] == 86_400_000
    request = latest["input"]["request"]
    assert request["candidate_horizon_sessions"] == [3, 5, 8, 10, 15, 20]
    assert request["fold_count"] == 3
    assert "request_fingerprint" in request
    data = latest["input"]["data"]
    assert data["instrument_count"] == 4
    assert data["session_count"] == 60
    assert data["feature_rows"] > 0
    assert isinstance(data["label_definition"], str)
    assert isinstance(data["feature_schema_hash"], str)
    assert isinstance(data["universe_policy_hash"], str)
    assert any(entry["horizon_sessions"] == 3 for entry in data["horizons"])
    cost = latest["input"]["cost_context"]
    assert cost["cost_schedule_kind"] == "base"
    assert cost["cost_evidence_hash"] == "cost-hash-1"
    assert cost["liquidity_model"] is True
    outcome = latest["outcome"]
    assert outcome["promoted"] is True
    assert outcome["no_trade"] is False
    assert outcome["model_type"] == "net_alpha_elastic_net"
    assert outcome["selected_horizons"] == [3]
    assert outcome["gates"]["passed"] is True
    observability = latest["observability"]
    assert observability["phases"]
    assert observability["horizons"][0]["admission"] == "eligible"
    assert observability["replay"]["block_count"] == 4
    assert "block_log_excess" not in observability["replay"]
    assert "period_net_returns" not in observability["replay"]
    assert observability["replay"]["block_excess_summary"]["count"] == 4
    holdout = observability["holdout"]
    assert holdout["passed"] is True
    assert holdout["order_count"] == 40
    assert holdout["block_count"] == 3
    assert holdout["cohorts"]["observed_sessions"] == 24
    assert holdout["cohorts"]["active_cohort_count"] == 3
    assert holdout["cohorts"]["eligible_sessions"] == 12
    assert holdout["cohorts"]["missing_realized_cohorts"] == 0
    assert holdout["certificate"]["passed"] is True
    assert holdout["certificate"]["base"]["cagr"] == 0.15
    assert holdout["certificate"]["stress"]["lower_cagr"] == 0.01
    assert holdout["certificate"]["stress"]["mdd"] == 0.10
    assert "period_net_returns" not in holdout["certificate"]["base"]
    assert holdout["eligibility"]["eligible_from"] == "2024-11-01T00:00:00+00:00"
    assert latest["artifact"]["manifest_path"] == "manifest.json"
    assert latest["artifact"]["metrics_path"] == "metrics.json"
    manifest_bytes = (registry.root / artifact_id / "manifest.json").read_bytes()
    metrics_bytes = (registry.root / artifact_id / "metrics.json").read_bytes()
    assert latest["artifact"]["manifest_bytes"] == len(manifest_bytes)
    assert latest["artifact"]["metrics_bytes"] == len(metrics_bytes)
    assert latest["artifact"]["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert latest["artifact"]["metrics_sha256"] == hashlib.sha256(metrics_bytes).hexdigest()


def test_record_completed_no_trade(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_no_trade_record"
    _write_artifact(registry.root, artifact_id, model_type="no_trade")
    context = _context(artifact_id)
    ledger_inst = MlResultLedger(tmp_path / "results")
    ledger_inst.record_completed(context, _manifest(artifact_id, "no_trade"), registry)
    latest = _latest(tmp_path / "results")
    assert latest["outcome"]["no_trade"] is True
    assert latest["outcome"]["promoted"] is False
    reasons = latest["outcome"]["promotion_reasons_digest"]
    assert reasons["count"] == 1
    assert len(reasons["sha256"]) == 64
    assert "no-horizon-evidence" not in json.dumps(latest)
    assert latest["outcome"]["selected_horizons"] == []


def test_record_failed_writes_sanitized_failure(tmp_path) -> None:
    context = _context("na_failed")
    ledger_inst = MlResultLedger(
        tmp_path / "results", clock=lambda: datetime(2024, 1, 2, tzinfo=UTC)
    )
    exc = ValueError("boom\nsecond line " + "x" * 600)
    ledger_inst.record_failed(context, "train_net_alpha_model", exc)
    latest = _latest(tmp_path / "results")
    assert latest["status"] == "failed"
    assert latest["failure"]["phase"] == "train_net_alpha_model"
    assert latest["failure"]["exception_type"] == "ValueError"
    message = latest["failure"]["message"]
    assert isinstance(message, str)
    assert len(message) == 512
    assert message.startswith("boom second line")
    assert latest["artifact"] == {}
    assert latest["outcome"] == {}


def test_record_completed_missing_artifact_raises(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    context = _context("na_missing")
    ledger_inst = MlResultLedger(tmp_path / "results")
    with pytest.raises(ValueError, match="artifact file missing"):
        ledger_inst.record_completed(context, _manifest("na_missing"), registry)
    assert not (tmp_path / "results" / "ml_runs").exists()


def test_record_completed_manifest_id_mismatch_raises(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_dir"
    _write_artifact(
        registry.root,
        artifact_id,
        manifest={
            "artifact_id": "na_other",
            "asset_kind": "STOCK",
            "feature_set": "stock_net_alpha_v1",
            "feature_schema_hash": "schema-hash-1",
            "universe_policy_hash": "universe-hash-1",
            "label_definition": "net_alpha_o2o",
            "label_horizon_sessions": 5,
            "eligible_from": "2024-01-01T00:00:00+00:00",
            "eligible_to": "2024-12-31T00:00:00+00:00",
            "model_type": "net_alpha_elastic_net",
            "params": {},
        },
    )
    context = _context(artifact_id)
    ledger_inst = MlResultLedger(tmp_path / "results")
    with pytest.raises(ValueError, match="artifact id mismatch"):
        ledger_inst.record_completed(context, _manifest(artifact_id), registry)
    assert not (tmp_path / "results" / "ml_runs" / "recent.jsonl").exists()


def test_oversized_artifact_still_records_within_byte_contract(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_huge"
    metrics = _default_metrics()
    metrics["promotion_reasons"] = [f"reason-{index}-" + "y" * 40 for index in range(700)]
    _write_artifact(registry.root, artifact_id, metrics=metrics)
    context = _context(artifact_id)
    ledger_inst = MlResultLedger(tmp_path / "results")
    ledger_inst.record_completed(context, _manifest(artifact_id), registry)
    latest = _latest(tmp_path / "results")
    assert latest["artifact_id"] == artifact_id
    assert len(_encode(latest)) <= MAX_RECORD_BYTES
    assert latest["outcome"]["promotion_reasons_digest"]["count"] == 700
    assert "reason-0-" not in json.dumps(latest)


def test_raw_vectors_and_unknown_keys_do_not_enter_projection(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_raw"
    metrics = _default_metrics()
    metrics["replay"]["period_net_returns"] = [0.01] * 5_000
    metrics["holdout"]["cohorts"]["period_net_returns_raw"] = [0.01] * 5_000
    metrics["unknown_oversized_key"] = {"blob": "y" * 30_000}
    metrics["secret_credentials"] = {"token": "x" * 40}
    _write_artifact(registry.root, artifact_id, metrics=metrics)
    context = _context(artifact_id)
    ledger_inst = MlResultLedger(tmp_path / "results")
    ledger_inst.record_completed(context, _manifest(artifact_id), registry)
    latest = _latest(tmp_path / "results")
    assert "unknown_oversized_key" not in latest
    assert "secret_credentials" not in latest
    replay = latest["observability"]["replay"]
    assert "block_log_excess" not in replay
    assert "period_net_returns" not in replay
    assert replay["block_excess_summary"]["count"] == 5_000
    assert "period_net_returns_raw" not in latest["observability"]["holdout"]


def test_project_holdout_legacy_artifact_fallback(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_legacy_holdout"
    metrics = _default_metrics()
    metrics["holdout"] = {
        "passed": False,
        "reason": "holdout-replay-no-trade",
        "order_count": 0,
        "block_count": 0,
    }
    del metrics["replay"]["period_net_returns"]
    metrics["replay"]["block_log_excess"] = [0.01, 0.02, 0.03]
    _write_artifact(registry.root, artifact_id, metrics=metrics)
    context = _context(artifact_id)
    ledger_inst = MlResultLedger(tmp_path / "results")
    ledger_inst.record_completed(context, _manifest(artifact_id), registry)
    holdout = _latest(tmp_path / "results")["observability"]["holdout"]
    assert holdout["passed"] is False
    assert holdout["reason"] == "holdout-replay-no-trade"
    assert holdout["cohorts"]["period_count"] == 0
    assert holdout["certificate"]["passed"] is False
    assert holdout["certificate"]["base"] == {}
    assert holdout["eligibility"] == {"eligible_from": "", "eligible_to": ""}
    replay = _latest(tmp_path / "results")["observability"]["replay"]
    assert replay["block_excess_summary"]["count"] == 3


def test_retention_dedup_and_discard_metadata(tmp_path) -> None:
    base = _context("na_ret_base")
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    clock = _MutableClock(datetime(2024, 1, 1, tzinfo=UTC))
    ledger_inst = MlResultLedger(tmp_path / "results", clock=clock, retained_records=8)
    for index in range(10):
        artifact_id = f"na_ret_{index:03d}"
        _write_artifact(registry.root, artifact_id, model_type="no_trade")
        clock.advance(timedelta(seconds=1))
        ledger_inst.record_completed(
            replace(base, artifact_id=artifact_id),
            _manifest(artifact_id, "no_trade"),
            registry,
        )
    lines = (tmp_path / "results" / "ml_runs" / "recent.jsonl").read_text().splitlines()
    assert len(lines) == 8
    ids = [json.loads(line)["artifact_id"] for line in lines]
    assert "na_ret_000" not in ids
    assert "na_ret_009" in ids
    meta = json.loads(
        (tmp_path / "results" / "ml_runs" / "ledger_meta.json").read_text()
    )
    assert meta["retained_count"] == 8
    assert meta["discarded_count"] == 2
    before = (registry.root / "na_ret_000" / "manifest.json").read_bytes()
    clock.advance(timedelta(seconds=1))
    ledger_inst.record_completed(
        replace(base, artifact_id="na_ret_005"),
        _manifest("na_ret_005", "no_trade"),
        registry,
    )
    lines = (tmp_path / "results" / "ml_runs" / "recent.jsonl").read_text().splitlines()
    assert len(lines) == 8
    assert (
        sum(1 for line in lines if json.loads(line)["artifact_id"] == "na_ret_005")
        == 1
    )
    assert (registry.root / "na_ret_000" / "manifest.json").read_bytes() == before


def test_path_traversal_rejected(tmp_path) -> None:
    context = _context("../escape")
    ledger_inst = MlResultLedger(tmp_path / "results")
    with pytest.raises(ValueError, match="invalid artifact_id"):
        ledger_inst.record_failed(context, "phase", ValueError("boom"))


def test_symlink_escape_rejected_before_write(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_link"
    _write_artifact(registry.root, artifact_id)
    context = _context(artifact_id)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ledger_meta.json").write_text("{}", encoding="utf-8")
    runs = tmp_path / "results" / "ml_runs"
    runs.mkdir(parents=True)
    (runs / "ledger_meta.json").symlink_to(outside / "ledger_meta.json")
    ledger_inst = MlResultLedger(tmp_path / "results")
    with pytest.raises(ValueError, match="escapes results root"):
        ledger_inst.record_completed(context, _manifest(artifact_id), registry)
    assert not (tmp_path / "results" / "ml_runs" / "latest.json").exists()


def test_pointer_stays_within_one_kib(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_pointer"
    _write_artifact(registry.root, artifact_id)
    context = _context(artifact_id)
    ledger_inst = MlResultLedger(tmp_path / "results")
    ledger_inst.record_completed(context, _manifest(artifact_id), registry)
    pointer = tmp_path / "results" / "back-res.md"
    assert pointer.exists()
    assert pointer.stat().st_size <= 1024
    text = pointer.read_text()
    assert "Latest artifact" in text
    assert "latest.json" in text
    assert "recent.jsonl" in text
    assert "source of truth" in text
    assert "ML Result Ledger" in text


def test_deterministic_records_for_fixed_inputs(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_det"
    _write_artifact(registry.root, artifact_id)
    context = _context(artifact_id)

    def _clock() -> datetime:
        return datetime(2024, 1, 2, tzinfo=UTC)

    first = MlResultLedger(tmp_path / "results_a", clock=_clock)
    second = MlResultLedger(tmp_path / "results_b", clock=_clock)
    first.record_completed(context, _manifest(artifact_id), registry)
    second.record_completed(context, _manifest(artifact_id), registry)
    assert (tmp_path / "results_a" / "ml_runs" / "latest.json").read_bytes() == (
        tmp_path / "results_b" / "ml_runs" / "latest.json"
    ).read_bytes()
    assert (tmp_path / "results_a" / "back-res.md").read_bytes() == (
        tmp_path / "results_b" / "back-res.md"
    ).read_bytes()


def test_rebuild_from_registry_recovers_cache(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    _write_artifact(registry.root, "na_recover")
    _write_artifact(registry.root, "na_recover_no_trade", model_type="no_trade")
    ledger_inst = MlResultLedger(tmp_path / "results")
    stats = ledger_inst.rebuild_from_registry(registry)
    assert stats["scanned"] == 2
    assert stats["retained"] == 2
    lines = (tmp_path / "results" / "ml_runs" / "recent.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert {json.loads(line)["artifact_id"] for line in lines} == {
        "na_recover",
        "na_recover_no_trade",
    }
    latest = _latest(tmp_path / "results")
    assert latest["status"] == "completed"
    assert latest["outcome"]["model_type"] in {
        "net_alpha_elastic_net",
        "no_trade",
    }
    assert latest["input"]["snapshot_id"] is None


def test_record_completed_projects_no_trade_policy_frontier_digest(tmp_path) -> None:
    """Acceptance 4: dropout reasons stay in metrics.json; the ledger keeps
    only the deterministic count/digest index plus recoverable evidence."""
    from src.stocks.ml.result_ledger import _compact_policy_frontier

    dropout = {
        f"{h}:{p}": "missing-realized-vintages:0"
        for h in (3, 5, 8, 10, 15, 20)
        for p in ("legacy_overlay_5bps", "lower_bound_only")
    }
    policy_frontier = {
        "candidate_count": 0,
        "profile_ids": ["legacy_overlay_5bps", "lower_bound_only"],
        "dropout_reasons": dropout,
        "segment_sums": {"h3:legacy_overlay_5bps:s0": {"scored_sessions": 20}},
    }
    projected = _compact_policy_frontier({"policy_frontier": policy_frontier})
    assert projected["candidate_count"] == 0
    assert projected["profile_count"] == 2
    assert projected["dropout_reasons_digest"]["count"] == 12
    assert projected["dropout_reasons_digest"] == _digest_summary(dropout)
    assert projected["segment_sums_digest"]["count"] == 1
    assert "missing-realized-vintages" not in json.dumps(projected)
    assert "scores" not in json.dumps(projected)
    assert "returns" not in json.dumps(projected)

    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_frontier_ledger"
    metrics = {
        **_default_metrics("no_trade"),
        "policy_frontier": policy_frontier,
    }
    _write_artifact(registry.root, artifact_id, model_type="no_trade", metrics=metrics)
    context = _context(artifact_id)
    ledger_inst = MlResultLedger(tmp_path / "results")
    ledger_inst.record_completed(context, _manifest(artifact_id, "no_trade"), registry)
    latest = _latest(tmp_path / "results")
    frontier = latest["observability"]["policy_frontier"]
    assert frontier["candidate_count"] == 0
    assert frontier["dropout_reasons_digest"]["count"] == 12
    assert "20:lower_bound_only" not in json.dumps(latest)
    # Full evidence remains recoverable from the referenced artifact.
    stored = json.loads(
        (registry.root / artifact_id / "metrics.json").read_text(encoding="utf-8")
    )
    assert stored["policy_frontier"]["dropout_reasons"]["20:lower_bound_only"] == (
        "missing-realized-vintages:0"
    )


def test_direct_input_ledger(tmp_path) -> None:
    """LMD-06: Completed and failed ledger records contain input_ids."""
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_direct_input"
    _write_artifact(registry.root, artifact_id)
    
    df = stock_net_alpha_composed_df(n_sessions=60, n_tickers=4, audit_clean=True)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, datetime(2024, 12, 31, tzinfo=UTC), (3, 5, 8, 10, 15, 20)
    )
    
    # Create context with input_ids
    context = MlRunContext.from_cli(
        request=NetAlphaTrainingRequest(
            artifact_id=artifact_id,
            candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        ),
        snapshot_id="direct:base_2024:features_2024:labels_2024",
        data=data,
        cost_context=CostRunContext(cost_schedule_kind="base"),
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
        input_ids={
            "base_dataset_id": "base_2024",
            "feature_dataset_id": "features_2024",
            "label_dataset_id": "labels_2024",
        },
    )
    
    ledger_inst = MlResultLedger(tmp_path / "results")
    ledger_inst.record_completed(context, _manifest(artifact_id), registry)
    latest = _latest(tmp_path / "results")
    
    # Verify input_ids are present
    assert latest["input"]["input_ids"] == {
        "base_dataset_id": "base_2024",
        "feature_dataset_id": "features_2024",
        "label_dataset_id": "labels_2024",
    }
    
    # Verify snapshot_id contains direct prefix
    assert latest["input"]["snapshot_id"] == "direct:base_2024:features_2024:labels_2024"


def test_direct_input_ledger_failed(tmp_path) -> None:
    """LMD-06: Failed ledger records contain input_ids."""
    df = stock_net_alpha_composed_df(n_sessions=60, n_tickers=4, audit_clean=True)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, datetime(2024, 12, 31, tzinfo=UTC), (3, 5, 8, 10, 15, 20)
    )
    
    # Create context with input_ids
    context = MlRunContext.from_cli(
        request=NetAlphaTrainingRequest(
            artifact_id="na_direct_failed",
            candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        ),
        snapshot_id="direct:base_2024:features_2024:labels_2024",
        data=data,
        cost_context=CostRunContext(cost_schedule_kind="base"),
        started_at=datetime(2024, 1, 1, tzinfo=UTC),
        input_ids={
            "base_dataset_id": "base_2024",
            "feature_dataset_id": "features_2024",
            "label_dataset_id": "labels_2024",
        },
    )
    
    ledger_inst = MlResultLedger(tmp_path / "results")
    ledger_inst.record_failed(context, "train_net_alpha_model", ValueError("boom"))
    latest = _latest(tmp_path / "results")
    
    # Verify input_ids are present in failed record
    assert latest["input"]["input_ids"] == {
        "base_dataset_id": "base_2024",
        "feature_dataset_id": "features_2024",
        "label_dataset_id": "labels_2024",
    }


def test_request_lookback_projection_and_fingerprint_identity() -> None:
    """ROLLING_LOOKBACK_03_REQUEST_AND_LEDGER_IDENTITY.

    A 1260-session cap is accepted and projected as 1260, 251 sessions fail
    closed at request construction, and otherwise-identical None and 1260
    requests never share a request_fingerprint.
    """
    from src.stocks.ml.result_ledger import _project_request

    request = NetAlphaTrainingRequest(
        artifact_id="na_lookback", max_training_lookback_sessions=1260
    )
    projection = _project_request(request)
    assert projection["max_training_lookback_sessions"] == 1260

    with pytest.raises(ValueError, match="at least 252"):
        NetAlphaTrainingRequest(
            artifact_id="na_lookback_low", max_training_lookback_sessions=251
        )

    expanding = _project_request(NetAlphaTrainingRequest(artifact_id="na_lookback"))
    assert expanding["request_fingerprint"] != projection["request_fingerprint"]


def _oversized_metrics() -> dict[str, object]:
    """Artifact metrics carrying far more than 24 KiB of run diagnostics."""
    metrics = _default_metrics()
    dropout = {
        f"h{h}:{p}": "missing-realized-vintages:" + "d" * 180
        for h in range(100)
        for p in ("legacy_overlay_5bps", "lower_bound_only", "half_kelly", "full_kelly")
    }
    metrics["policy_frontier"] = {
        "candidate_count": 0,
        "profile_ids": ["legacy_overlay_5bps", "lower_bound_only"],
        "dropout_reasons": dropout,
        "segment_sums": {
            f"h{h}:{p}:s0": {"scored_sessions": h}
            for h in range(100)
            for p in ("legacy_overlay_5bps", "lower_bound_only")
        },
    }
    run_obs = metrics["run_observability"]
    assert isinstance(run_obs, dict)
    phases = run_obs["phases"]
    assert isinstance(phases, list)
    phases.append(
        {
            "name": "primary_selection",
            "elapsed_ms": 12,
            "selection_reasons": [
                f"reason-{index}-" + "s" * 80 for index in range(241)
            ],
        }
    )
    return metrics


def test_ml_ledger_completion_01_records_oversized_observability(tmp_path) -> None:
    """ML_LEDGER_COMPLETION_01_OVERSIZED_OBSERVABILITY.

    An artifact with more than 24 KiB of policy-frontier/selection diagnostics
    records completed; latest.json stays within MAX_RECORD_BYTES, stores the
    manifest/metrics SHA-256 and byte lengths, and exposes count/digest indexes
    instead of the raw diagnostic collections.
    """
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_oversized_obs"
    metrics = _oversized_metrics()
    _write_artifact(registry.root, artifact_id, metrics=metrics)
    context = _context(artifact_id)
    ledger_inst = MlResultLedger(
        tmp_path / "results", clock=lambda: datetime(2024, 1, 2, tzinfo=UTC)
    )
    ledger_inst.record_completed(context, _manifest(artifact_id), registry)

    latest_path = tmp_path / "results" / "ml_runs" / "latest.json"
    raw = latest_path.read_bytes()
    assert len(raw) <= MAX_RECORD_BYTES
    latest = json.loads(raw)
    assert latest["status"] == "completed"

    manifest_bytes = (registry.root / artifact_id / MANIFEST_FILENAME).read_bytes()
    metrics_bytes = (registry.root / artifact_id / METRICS_FILENAME).read_bytes()
    assert latest["artifact"]["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert latest["artifact"]["metrics_sha256"] == hashlib.sha256(metrics_bytes).hexdigest()
    assert latest["artifact"]["manifest_bytes"] == len(manifest_bytes)
    assert latest["artifact"]["metrics_bytes"] == len(metrics_bytes)

    frontier = latest["observability"]["policy_frontier"]
    expected_digest = _digest_summary(metrics["policy_frontier"]["dropout_reasons"])
    assert frontier["dropout_reasons_digest"] == expected_digest
    assert frontier["dropout_reasons_digest"]["count"] == 400
    assert frontier["segment_sums_digest"]["count"] == 200
    text = raw.decode("utf-8")
    assert "missing-realized-vintages" not in text
    assert '"selection_reasons"' not in text

    recent_lines = (
        tmp_path / "results" / "ml_runs" / "recent.jsonl"
    ).read_text().splitlines()
    assert len(recent_lines) == 1
    for line in recent_lines:
        assert len(line.encode("utf-8")) <= MAX_RECORD_BYTES


def test_ml_ledger_completion_02_rebuild_oversized_artifact(tmp_path) -> None:
    """ML_LEDGER_COMPLETION_02_REBUILD_OVERSIZED_ARTIFACT."""
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_oversized_rebuild"
    _write_artifact(registry.root, artifact_id, metrics=_oversized_metrics())
    ledger_inst = MlResultLedger(tmp_path / "results")
    stats = ledger_inst.rebuild_from_registry(registry)
    assert stats["scanned"] == 1
    assert stats["retained"] == 1

    record = json.loads(
        (tmp_path / "results" / "ml_runs" / "recent.jsonl")
        .read_text()
        .splitlines()[0]
    )
    assert record["status"] == "completed"
    assert len(_encode(record)) <= MAX_RECORD_BYTES
    manifest_bytes = (registry.root / artifact_id / MANIFEST_FILENAME).read_bytes()
    metrics_bytes = (registry.root / artifact_id / METRICS_FILENAME).read_bytes()
    assert record["artifact"]["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert record["artifact"]["metrics_sha256"] == hashlib.sha256(metrics_bytes).hexdigest()
    frontier = record["observability"]["policy_frontier"]
    assert frontier["dropout_reasons_digest"]["count"] == 400
    latest = _latest(tmp_path / "results")
    assert len(_encode(latest)) <= MAX_RECORD_BYTES


def test_ml_ledger_completion_03_byte_budget_fallback(tmp_path) -> None:
    """ML_LEDGER_COMPLETION_03_BYTE_FALLBACK.

    A deliberately oversized optional completed projection is replaced by a
    deterministic terminal fallback that retains status, the artifact
    reference, and a compaction digest within MAX_RECORD_BYTES.
    """
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_fallback"
    _write_artifact(registry.root, artifact_id)
    context = _context(artifact_id)
    report = {"sections": [{"name": f"s{i}", "blob": "y" * 2000} for i in range(40)]}

    def _clock() -> datetime:
        return datetime(2024, 1, 2, tzinfo=UTC)

    ledger_inst = MlResultLedger(tmp_path / "results", clock=_clock)
    ledger_inst.record_completed(
        context, _manifest(artifact_id), registry, diagnostic_report=report
    )
    latest = _latest(tmp_path / "results")
    assert len(_encode(latest)) <= MAX_RECORD_BYTES
    assert latest["status"] == "completed"
    compaction = latest["compaction"]
    assert len(compaction["omitted_record_sha256"]) == 64
    metrics_bytes = (registry.root / artifact_id / METRICS_FILENAME).read_bytes()
    assert latest["artifact"]["metrics_sha256"] == hashlib.sha256(metrics_bytes).hexdigest()
    assert latest["outcome"]["promoted"] is True
    assert "sections" not in json.dumps(latest)


def test_ml_ledger_completion_04_schema_refresh_on_v2_write(tmp_path) -> None:
    """ML_LEDGER_COMPLETION_04_SCHEMA_REFRESH.

    Writing a v2 terminal record replaces a stale schema.json so the metadata
    describes the compact digest-backed record shape.
    """
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = "na_schema_refresh"
    _write_artifact(registry.root, artifact_id)
    runs_root = tmp_path / "results" / "ml_runs"
    runs_root.mkdir(parents=True)
    stale = {"schema_version": SCHEMA_VERSION - 1, "record_byte_limit": 1024}
    (runs_root / SCHEMA_FILENAME).write_text(json.dumps(stale), encoding="utf-8")

    ledger_inst = MlResultLedger(tmp_path / "results")
    ledger_inst.record_completed(_context(artifact_id), _manifest(artifact_id), registry)

    schema = json.loads((runs_root / SCHEMA_FILENAME).read_text(encoding="utf-8"))
    assert schema["schema_version"] == SCHEMA_VERSION
    assert schema["record_byte_limit"] == MAX_RECORD_BYTES
    text = json.dumps(schema)
    assert "sha256" in text
    assert "digest" in text


def test_execution_frontier_bounded_ledger_projection() -> None:
    """ML_EXEC_FRONTIER_04_BOUNDED_LEDGER_PROJECTION.

    The compact ml_runs request projection contains only the frontier H/C/K lists
    and the bounded summary carries candidate_bound=60 plus the selected
    H/C/K/profile scalars; it contains no raw score, order, return, label, or
    instrument payload.
    """
    from src.stocks.ml.result_ledger import (
        _bounded_observability_summary,
        _project_request,
    )

    request = NetAlphaTrainingRequest(artifact_id="na_exec_frontier_ledger")
    projection = _project_request(request)
    frontier = projection["execution_frontier"]
    assert frontier["candidate_horizon_sessions"] == [10, 20]
    assert frontier["candidate_rebalance_frequency_sessions"] == [5, 10, 20]
    assert frontier["candidate_top_k"] == [12, 16, 20, 24]
    assert "scores" not in json.dumps(projection)
    assert "orders" not in json.dumps(projection)
    assert "labels" not in json.dumps(projection)

    telemetry = {
        "phases": [
            {
                "name": "policy_frontier",
                "candidate_count": 60,
                "candidate_bound": 60,
                "profile_ids": ["legacy_overlay_5bps", "lower_bound_only", "lower_bound_half_kelly"],
            },
            {
                "name": "primary_selection",
                "primary_horizon_sessions": 20,
                "primary_rebalance_frequency_sessions": 5,
                "primary_top_k": 12,
                "primary_profile_id": "lower_bound_only",
            },
        ],
        "horizons": [],
    }
    summary = _bounded_observability_summary(telemetry)
    assert summary["frontier_candidate_bound"] == 60
    assert summary["primary_horizon_sessions"] == 20
    assert summary["primary_rebalance_frequency_sessions"] == 5
    assert summary["primary_top_k"] == 12
    assert summary["primary_profile_id"] == "lower_bound_only"
    assert "scores" not in json.dumps(summary)
    assert "orders" not in json.dumps(summary)
    assert "labels" not in json.dumps(summary)
