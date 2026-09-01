def test_training_temporal_leaf_reexport_keeps_locked_holdout_identity() -> None:
    from src.stocks.ml import training_orchestrator as orchestrator
    from src.stocks.ml import training_temporal as temporal

    assert orchestrator._locked_holdout is temporal._locked_holdout
    assert callable(temporal._locked_holdout)
