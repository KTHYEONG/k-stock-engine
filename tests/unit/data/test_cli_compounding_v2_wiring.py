def test_cli_exposes_and_wires_compounding_v2_strategy() -> None:
    import inspect
    import src.data.cli as cli
    from src.strategy.compounding_v2_strategy import CompoundingV2Strategy

    args = cli._parse_args(['run-backtest', '--strategy-id', 'compounding-v2'])
    source = inspect.getsource(cli._dispatch_backtest)
    assert args.strategy_id == 'compounding-v2'
    assert cli.CompoundingV2Strategy is CompoundingV2Strategy
    assert "required_kinds=('universe', 'champion_scores')" in source or '"champion_scores"' in source
    assert 'warmup_sessions = 200' in source
    assert 'CompoundingV2Strategy(' in source


def test_compounding_v2_dispatch_end_to_end_with_pit_inputs(tmp_path, monkeypatch) -> None:
    import json
    from argparse import Namespace
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    import polars as pl

    import src.data.silver as silver_mod
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import SilverTable

    kst = ZoneInfo('Asia/Seoul')
    all_days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(270))
    calendar_df = pl.DataFrame({'session': list(all_days)})
    start, end = all_days[200].date().isoformat(), all_days[204].date().isoformat()
    warmup_days = all_days[0:207]
    closes_a = [10000.0 + index * 10.0 for index in range(len(warmup_days))]
    closes_b = [20000.0 + index * 5.0 for index in range(len(warmup_days))]
    market_df = pl.DataFrame({
        'session': [day for day in warmup_days for _ in ('KRX:A', 'KRX:B')],
        'instrument_id': ['KRX:A', 'KRX:B'] * len(warmup_days),
        'open': [c - 5.0 for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        'close': [c for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        'volume': [1000.0] * (2 * len(warmup_days)),
        'trading_value': [c * 1000.0 for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        'market_cap': [1e12, 5e11] * len(warmup_days),
        'available_at': [day.replace(hour=15, minute=30) for day in warmup_days for _ in ('KRX:A', 'KRX:B')],
    })
    dm_dir = tmp_path / 'dm'
    dm_dir.mkdir()
    market_df.write_parquet(dm_dir / 'daily.parquet')
    master_df = pl.DataFrame({
        'instrument_id': ['KRX:A', 'KRX:B'],
        'sector': ['Technology', 'Healthcare'],
        'valid_from': [all_days[0], all_days[0]],
        'valid_to': [all_days[269], all_days[269]],
        'available_at': [all_days[0], all_days[0]],
    })

    def fake_load_table(silver_root, table):
        if table == SilverTable.CALENDAR:
            return calendar_df
        if table == SilverTable.SECURITY_MASTER:
            return master_df
        return pl.DataFrame(schema={'instrument_id': pl.String, 'action_type': pl.String, 'effective_session': pl.Datetime(time_zone='Asia/Seoul'), 'available_at': pl.Datetime(time_zone='Asia/Seoul'), 'evidence_status': pl.String})

    import src.data.gold_artifacts as gold_artifacts_mod
    from src.data.backtest_run_manifest import build_backtest_run_manifest, write_backtest_run_manifest

    score_days = [day.replace(hour=15, minute=30) for day in all_days[200:205]]
    universe_df = pl.DataFrame({
        'decision_session': [day for day in score_days for _ in ('KRX:A', 'KRX:B')],
        'instrument_id': ['KRX:A', 'KRX:B'] * 5,
        'eligible': [True] * 10,
    })
    scores_df = pl.DataFrame({
        'decision_session': [day for day in score_days for _ in ('KRX:A', 'KRX:B')],
        'instrument_id': ['KRX:A', 'KRX:B'] * 5,
        'eligible': [True] * 10,
        'champion_score': [90.0, 80.0] * 5,
        'rank': [1, 2] * 5,
        'exclusion_reasons': [''] * 10,
        'feature_policy_version': ['champion-v1-qvef-v1'] * 10,
        'score_policy_version': ['champion-v1-scoring-v1'] * 10,
    })

    def fake_resolve(*, gold_root, dataset_id, decision_time, required_kinds):
        assert required_kinds == ('universe', 'champion_scores')
        return SimpleNamespace(dataset_id=dataset_id, universe_manifest_hash='uni-hash', champion_scores_manifest_hash='scores-hash')

    monkeypatch.setattr(silver_mod, 'load_silver_table_by_dataset_id', lambda *, root, table, dataset_id, decision_time: fake_load_table(root, table))
    monkeypatch.setattr(silver_mod, 'silver_dataset_path_by_id', lambda *, root, table, dataset_id, decision_time: dm_dir)
    monkeypatch.setattr(gold_artifacts_mod, 'resolve_gold_artifact_bundle', fake_resolve)
    monkeypatch.setattr(gold_artifacts_mod, 'load_gold_universe_and_scores', lambda *, bundle, decision_time: (universe_df, scores_df))

    run_manifest_obj = build_backtest_run_manifest(
        silver_root=tmp_path / 'silver',
        gold_root=tmp_path / 'gold',
        silver_dataset_ids={table: f'{table.value}-id' for table in SilverTable},
        gold_dataset_id='gold-test',
        validation_start=__import__('datetime').date.fromisoformat(start),
        validation_end=__import__('datetime').date.fromisoformat(end),
        strategy_id='compounding-v2',
        policy_versions={'market_inputs': 'korean-equity-market-inputs-v2'},
    )
    manifest_path = write_backtest_run_manifest(manifest=run_manifest_obj, artifact_root=tmp_path / 'artifacts')

    code = _dispatch_backtest(
        Namespace(
            silver_root=tmp_path / 'silver',
            artifact_root=tmp_path / 'artifacts',
            gold_root=tmp_path / 'gold',
            validation_start=start,
            validation_end=end,
            smoke_symbol=None,
            gold_dataset_id='gold-test',
            backtest_run_manifest=manifest_path,
            strategy_id='compounding-v2',
            initial_cash=100_000_000.0,
            scenario='base',
            ledger_id='compounding-v2-test',
        )
    )
    assert code == 0
    manifests = list((tmp_path / 'artifacts' / 'backtests').rglob('result.json'))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding='utf-8'))
    assert payload['metadata']['strategy_id'] == 'compounding-v2'
    assert payload['metadata']['selection_policy_version'] == 'compounding-v2-selection-v1'
    assert payload['metadata']['portfolio_policy_version'] == 'compounding-v2-portfolio-v1'
    assert payload['metadata']['warmup_sessions'] == 200
    assert payload['metadata']['universe_manifest_hash'] == 'uni-hash'
    assert payload['metadata']['champion_scores_manifest_hash'] == 'scores-hash'
