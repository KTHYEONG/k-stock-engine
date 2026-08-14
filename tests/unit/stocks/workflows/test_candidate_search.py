from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from src.stocks.workflows.candidate_search import run_candidate_search


def test_candidate_search_returns_typed_result_and_consumes_telemetry(monkeypatch) -> None:
    import src.stocks.research.lambdarank as lambdarank
    import src.stocks.workflows.train_model as train_model

    config = SimpleNamespace(_tuning_telemetry={"selection_status": "selected"})
    route = SimpleNamespace(horizon=5)

    def fake_tune(*_args, **_kwargs):
        return config, 7, route

    monkeypatch.setattr(train_model, "_tune_champion", fake_tune)
    monkeypatch.setattr(
        lambdarank.LambdaRankConfig,
        "_tuning_telemetry",
        {"selection_status": "selected"},
    )

    result = run_candidate_search(
        pl.DataFrame(),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        (),
        (),
        dataset_manifest=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        base_schedule=None,  # type: ignore[arg-type]
        stress_schedule=None,  # type: ignore[arg-type]
    )

    assert result.config is config
    assert result.multiplicity_count == 7
    assert result.route is route
    assert result.telemetry == {"selection_status": "selected"}
    assert config._tuning_telemetry is None
    assert lambdarank.LambdaRankConfig._tuning_telemetry is None
