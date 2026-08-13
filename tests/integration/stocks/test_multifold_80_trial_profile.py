"""Slow production-snapshot benchmark: serial 81-trial search under an RSS budget.

Run explicitly with ``pytest -m slow``; excluded from the normal
``-m "not slow"`` suite. Verifies the search reaches 81 terminal screens under
an explicit RSS budget with the deterministic 27-to-6-to-6-to-3 multi-fidelity
funnel, runs at most nine inner economic replays (one per all-positive
finalist under the single default policy), records baseline/peak RSS and
trial/fold timing telemetry in the published metrics, and publishes a terminal
promoted or ``NO_TRADE`` artifact without relaxing the promotion gates.
"""
from __future__ import annotations

import json
import time

import polars as pl
import pytest

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import METRICS_FILENAME, ModelArtifactRegistry
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import stock_v2_composed_df, stock_v2_manifest

_OPTUNA_TRIALS = 81
_BUDGET_MIB = 8000


@pytest.mark.slow
def test_multifold_80_trial_profile_completes_under_budget(tmp_path, monkeypatch) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    df = df.with_columns(
        pl.col("residual_o2o_5d").alias("residual_o2o_10d"),
        pl.col("relevance").alias("relevance_10d"),
        (pl.col("label_available_time") + pl.duration(days=5)).alias(
            "label_available_time_10d"
        ),
        pl.col("residual_o2o_5d").alias("residual_o2o_15d"),
        pl.col("relevance").alias("relevance_15d"),
        (pl.col("label_available_time") + pl.duration(days=10)).alias(
            "label_available_time_15d"
        ),
    )
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    artifact_root = tmp_path / "artifacts"
    registry = ModelArtifactRegistry(artifact_root)

    before = time.monotonic()
    train_model(
        snapshot,
        registry,
        TrainingRequest(
            artifact_id="bench_80trial",
            n_folds=3,
            optuna_trials=_OPTUNA_TRIALS,
            max_rss_mib=_BUDGET_MIB,
        ),
    )
    elapsed_seconds = time.monotonic() - before

    payload = json.loads(
        (artifact_root / "bench_80trial" / METRICS_FILENAME).read_text()
    )
    assert payload["optuna_trials"] == _OPTUNA_TRIALS
    resource = payload["resource"]
    assert resource["n_terminal_trials"] == _OPTUNA_TRIALS
    assert resource["total_terminal_screen_trials"] == _OPTUNA_TRIALS
    assert resource["configured_compounding_policy_cells"] == 1
    assert resource["selection_multiplicity_version"] == (
        "selection-multiplicity-global-count-v1"
    )
    assert payload["total_terminal_screen_trials"] == _OPTUNA_TRIALS
    assert payload["configured_compounding_policy_cells"] == 1
    assert payload["selection_multiplicity_version"] == (
        "selection-multiplicity-global-count-v1"
    )
    for row in resource["shortlist_candidate_evidence"]:
        assert row["total_terminal_screen_trials"] == _OPTUNA_TRIALS
        assert row["route_terminal_screen_trials"] == _OPTUNA_TRIALS // 3
        assert row["configured_compounding_policy_cells"] == 1
        assert row["selection_multiplicity_version"] == (
            "selection-multiplicity-global-count-v1"
        )
        assert row["policy_id"] == "default:neutral"
        assert row["exact_compounding_policy_replays"] == 1
    assert resource["peak_rss_mib"] <= _BUDGET_MIB
    assert resource["baseline_rss_mib"] > 0.0
    assert resource["trial_fold_timings_seconds"]
    assert resource["screened_trials"] >= 0
    assert resource["pruned_trials"] >= 0
    assert resource["screened_trials"] + resource["pruned_trials"] == _OPTUNA_TRIALS
    assert resource["selection_policy_version"] == "economic-selection-v5-execution-matched"
    assert resource["compute_plan_version"] == "sub10-refit-v1"
    assert resource["resolved_lgb_threads"] >= 1
    assert resource["per_route_trial_budget"] == _OPTUNA_TRIALS // 3
    assert resource["shortlisted_trials"] <= 18
    assert resource["screen_fidelity"] == "execution_matched"
    assert resource["proxy_session_stride"] == 6
    assert resource["promotion_width"] == 6
    assert resource["economic_finalist_width"] == 3
    assert resource["cache_bytes"] > 0
    assert resource["screen_seconds"] > 0.0
    assert resource["full_refit_boosting_rounds"] == 900
    assert resource["full_refit_early_stopping_rounds"] == 100
    assert resource["full_refit_seconds"] >= 0.0
    assert resource["economic_replay_seconds"] >= 0.0
    for attrs in resource["routes"].values():
        for key in (
            "context_prepare_seconds",
            "refit_train_seconds",
            "refit_predict_seconds",
            "replay_prepare_seconds",
            "economic_replay_seconds",
            "full_refit_seconds",
            "resolved_lgb_threads",
            "actual_refit_rounds",
            "actual_best_iterations",
        ):
            assert key in attrs
        assert (
            attrs["full_refit_seconds"]
            <= attrs["full_refit_seconds"]
            + attrs["replay_prepare_seconds"]
            + attrs["economic_replay_seconds"]
        )
    assert resource["selection_status"] in (
        "selected",
        "no_complete_screen_candidate",
        "no_economically_eligible_candidate",
    )
    replay_resource = resource["replay_resource"]
    assert replay_resource["inner_stress_replay"] is False
    assert replay_resource["prepared_decision_count"] >= 0
    assert replay_resource["replay_peak_rss_mib"] >= 0.0
    assert replay_resource["replay_operational_limit_mib"] > 0.0
    assert replay_resource["replay_limit_mib"] > 0.0
    inner_replays = sum(
        int(attrs.get("all_positive_finalists", 0))
        for attrs in resource["routes"].values()
    )
    assert inner_replays <= 9
    assert elapsed_seconds > 0.0
    assert payload["promoted"] is False
    assert payload["no_trade"] is True
