def test_cli_exposes_and_wires_compounding_strategy() -> None:
    import inspect
    import src.data.cli as cli_mod
    from src.strategy.compounding_strategy import CompoundingStrategy

    args = cli_mod._parse_args(["run-backtest", "--strategy-id", "compounding-v1"])
    assert args.strategy_id == "compounding-v1"
    assert cli_mod.CompoundingStrategy is CompoundingStrategy
    source = inspect.getsource(cli_mod._dispatch_backtest)
    assert "strategy_id == 'compounding-v1'" in source or 'strategy_id == "compounding-v1"' in source
    assert "market_index_eligible_by_session=eligible_by_session" in source
    assert "warmup_sessions = 200" in source or "warmup = 200" in source
    assert "CompoundingStrategy(" in source
