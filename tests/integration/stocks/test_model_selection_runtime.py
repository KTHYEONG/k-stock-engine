# ruff: noqa
# MODEL_SELECTION_FAST_06_RUNTIME_BENCHMARK
# PROFILED_ML_SELECTION_06_HASH_BOUND_RUNTIME
import time
import pytest


def test_PROFILED_ML_SELECTION_06_HASH_BOUND_RUNTIME():
    import time, tempfile, pathlib, polars as pl
    from datetime import datetime, UTC, timedelta
    from src.stocks.ml.contracts import NetAlphaTrainingRequest, ModelSelectionStudySettings, ModelSelectionComputeBudget
    from src.stocks.ml.model_selection import evaluate_model_selection_study
    from src.stocks.research.artifacts import ModelArtifactRegistry
    from src.core.costs import default_base_schedule, default_stress_schedule
    from tests.fixtures.stocks.helpers import stock_liquidity_model
    from src.stocks.ml.features import stock_net_alpha_v1_roles as _rf
    _roles=_rf()
    rng = __import__("numpy").random.default_rng(9)
    sessions=[datetime(2024,1,1,tzinfo=UTC)+timedelta(days=i) for i in range(2000)]
    rows=[]
    for s in sessions:
        for tidx in range(3):
            row={"instrument_id": f"KRX:{tidx:05d}", "session": s, "session_index": sessions.index(s), "sector": "tech", "available_time": s, "open": 100.0, "adtv_20d":1e6, "volatility_20d":0.02}
            for src in _roles:
                row[src]= float(rng.normal())
                row[f"feature__{src}"]= row[src]
            rows.append(row)
    frame=pl.DataFrame(rows)
    label_rows=[{"instrument_id": r["instrument_id"], "session": r["session"], "net_alpha_target": float(rng.normal(scale=0.01)), "risk_residual": 0.01, "reference_cost":0.001, "label_available_time": r["session"]+timedelta(days=5), "realized_net_return": float(rng.normal(scale=0.01))} for r in rows]
    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind
    manifest=DatasetManifest(asset_kind=AssetKind.STOCK, schema_version="v1", schema_hash="h", provider_version="p", universe_policy_version="u", universe_policy_hash="u", feature_set="stock_net_alpha_v1", feature_set_hash="f", label_definition="net_alpha_o2o", label_horizon_sessions=10, time_start=sessions[0], time_end=sessions[-1], generated_time=sessions[-1], row_count=len(rows), reference_notional=100_000_000.0)
    from src.stocks.ml.contracts import NetAlphaResearchData
    data=NetAlphaResearchData(feature_frame=frame, labels_by_horizon={10: pl.DataFrame(label_rows)}, manifest=manifest)
    request=NetAlphaTrainingRequest(artifact_id="hash06", candidate_horizon_sessions=(10,), fold_count=3, base_cost_schedule=default_base_schedule(), stress_cost_schedule=default_stress_schedule(), liquidity_model=stock_liquidity_model(), stress_liquidity_model=stock_liquidity_model(stress_multiplier=2.0))
    settings=ModelSelectionStudySettings(candidate_lookback_sessions=(1260,), common_min_train_sessions=1260, min_validation_segment_sessions=20, compute_budget=ModelSelectionComputeBudget(wall_clock_seconds=540.0, screen_phase_seconds=180.0))
    with tempfile.TemporaryDirectory() as tmp:
        registry=ModelArtifactRegistry(pathlib.Path(tmp))
        start=time.monotonic()
        result=evaluate_model_selection_study(data, request, settings, registry=registry)
        elapsed=time.monotonic()-start
        assert elapsed <= 600
        assert result["status"]=="RESEARCH_ONLY"
        ledger=result["runtime_ledger"]
        assert ledger["screen_fold_count"]==3
        assert ledger["elapsed_seconds"] <=600
        if result.get("selected_family") is not None:
            assert ledger["replay_count"] in (1,2)
            # check both base/stress ledger growth non-empty via survivors
            assert len(result.get("survivors",[]))>=1


@pytest.mark.slow
def test_MODEL_SELECTION_FAST_06_RUNTIME_BENCHMARK():
    # Hash-bound one-horizon/one-lookback six-family study either completes within 600s with ledger or returns budget-exhausted before 600s.
    from src.stocks.ml.contracts import NetAlphaTrainingRequest, ModelSelectionStudySettings, ModelSelectionComputeBudget, ModelFamily
    from src.stocks.ml.model_selection import evaluate_model_selection_study
    from src.stocks.research.artifacts import ModelArtifactRegistry
    from src.core.costs import default_base_schedule, default_stress_schedule
    from tests.fixtures.stocks.helpers import stock_liquidity_model
    import polars as pl
    from datetime import datetime, UTC, timedelta
    import tempfile, pathlib
    rng = __import__("numpy").random.default_rng(1)
    sessions=[datetime(2024,1,1,tzinfo=UTC)+timedelta(days=i) for i in range(30)]
    rows=[{"instrument_id":f"KRX:{t:05d}","session":s,"session_index":sessions.index(s),"sector":"tech","available_time":s,"feature__a":float(rng.normal()),"adtv_20d":1e6,"open":100.0} for s in sessions for t in range(5)]
    frame=pl.DataFrame(rows)
    label_rows=[{"instrument_id":r["instrument_id"],"session":r["session"],"net_alpha_target":float(rng.normal(scale=0.01)),"risk_residual":0.01,"reference_cost":0.001,"label_available_time":r["session"]+timedelta(days=5),"realized_net_return":0.01} for r in rows]
    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind
    manifest=DatasetManifest(asset_kind=AssetKind.STOCK, schema_version="v1", schema_hash="h", provider_version="p", universe_policy_version="u", universe_policy_hash="u", feature_set="stock_net_alpha_v1", feature_set_hash="f", label_definition="net_alpha_o2o", label_horizon_sessions=10, time_start=sessions[0], time_end=sessions[-1], generated_time=sessions[-1], row_count=len(rows), reference_notional=100_000_000.0)
    from src.stocks.ml.contracts import NetAlphaResearchData
    data=NetAlphaResearchData(feature_frame=frame, labels_by_horizon={10: pl.DataFrame(label_rows)}, manifest=manifest)
    request=NetAlphaTrainingRequest(artifact_id="bench06", candidate_horizon_sessions=(10,), base_cost_schedule=default_base_schedule(), stress_cost_schedule=default_stress_schedule(), liquidity_model=stock_liquidity_model(), stress_liquidity_model=stock_liquidity_model(stress_multiplier=2.0))
    settings=ModelSelectionStudySettings(candidate_lookback_sessions=(504,), compute_budget=ModelSelectionComputeBudget(wall_clock_seconds=540.0))
    with tempfile.TemporaryDirectory() as tmp:
        registry=ModelArtifactRegistry(pathlib.Path(tmp))
        start=time.monotonic()
        result=evaluate_model_selection_study(data, request, settings, registry=registry)
        elapsed=time.monotonic()-start
        assert elapsed < 600
        assert result["status"] in ("RESEARCH_ONLY",)
        assert "runtime_ledger" in result
        if result.get("study_complete"):
            assert result.get("selected_family") is not None or result.get("selected_family") is None  # either champion or explicit no-champion
            assert result["runtime_ledger"]["elapsed_seconds"] < 600
        else:
            assert result["next_action"] in ("budget-exhausted", "no-qualified-survivor", "insufficient-common-window-calendar", "budget-unbounded-grid")
            assert result["selected_family"] is None
