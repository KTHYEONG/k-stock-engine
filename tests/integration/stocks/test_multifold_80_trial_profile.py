"""Slow production-snapshot benchmark: serial 80-trial search under an RSS budget.

Run explicitly with ``pytest -m slow``; excluded from the normal
``-m "not slow"`` suite. Verifies the search reaches 80 terminal trials under
an explicit RSS budget, records baseline/peak RSS and trial/fold timing
telemetry in the published metrics, and never relaxes the promotion gates.
"""
from __future__ import annotations

import json
import time

import pytest

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import METRICS_FILENAME, ModelArtifactRegistry
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import stock_v2_composed_df, stock_v2_manifest

_OPTUNA_TRIALS = 80
_BUDGET_MIB = 2048


@pytest.mark.slow
def test_multifold_80_trial_profile_completes_under_budget(tmp_path, monkeypatch) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
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
    assert resource["peak_rss_mib"] <= _BUDGET_MIB
    assert resource["baseline_rss_mib"] > 0.0
    assert resource["trial_fold_timings_seconds"]
    assert resource["screened_trials"] >= 0
    assert resource["pruned_trials"] >= 0
    assert resource["screened_trials"] + resource["pruned_trials"] == _OPTUNA_TRIALS
    assert resource["shortlisted_trials"] <= 8
    assert resource["cache_bytes"] > 0
    assert resource["screen_seconds"] > 0.0
    assert resource["selection_status"] in (
        "selected",
        "no_complete_screen_candidate",
        "no_economically_eligible_candidate",
    )
    assert elapsed_seconds > 0.0
    assert payload["promoted"] is False
    assert payload["no_trade"] is True
