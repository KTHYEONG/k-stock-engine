def test_derive_historical_collection_window_keeps_full_validation_and_next_execution_session() -> None:
    from datetime import date, timedelta
    from src.data.collection_plan import derive_historical_collection_window

    sessions = tuple(date(2015, 10, 1) + timedelta(days=i) for i in range(63))
    result = derive_historical_collection_window(sessions=sessions, validation_start=sessions[60], validation_end=sessions[61], warmup_sessions=60)

    assert result.history_start == sessions[0]
    assert result.validation_start == sessions[60]
    assert result.validation_end == sessions[61]
    assert result.execution_end == sessions[62]
    assert result.sessions == sessions


def test_derive_historical_collection_window_rejects_invalid_calendar_inputs() -> None:
    from datetime import date
    import pytest
    from src.data.collection_plan import derive_historical_collection_window
    from src.data.schemas import PITDataError

    day = date(2016, 1, 4)
    with pytest.raises(PITDataError, match="non-empty"):
        derive_historical_collection_window(sessions=(), validation_start=day, validation_end=day, warmup_sessions=0)
    with pytest.raises(PITDataError, match="strictly"):
        derive_historical_collection_window(sessions=(day, day), validation_start=day, validation_end=day, warmup_sessions=0)
    with pytest.raises(PITDataError, match="strictly"):
        derive_historical_collection_window(sessions=(date(2016, 1, 5), day), validation_start=day, validation_end=day, warmup_sessions=0)
    with pytest.raises(PITDataError, match="non-negative"):
        derive_historical_collection_window(sessions=(day,), validation_start=day, validation_end=day, warmup_sessions=-1)
    with pytest.raises(PITDataError, match="after"):
        derive_historical_collection_window(sessions=(day,), validation_start=day, validation_end=date(2016, 1, 3), warmup_sessions=0)
    with pytest.raises(PITDataError, match="within"):
        derive_historical_collection_window(sessions=(day, date(2016, 1, 5)), validation_start=date(2016, 1, 3), validation_end=day, warmup_sessions=0)


def test_derive_historical_collection_window_requires_warmup_and_execution() -> None:
    from datetime import date, timedelta
    import pytest
    from src.data.collection_plan import derive_historical_collection_window
    from src.data.schemas import PITDataError

    sessions = tuple(date(2016, 1, 4) + timedelta(days=i) for i in range(3))
    with pytest.raises(PITDataError, match="warmup"):
        derive_historical_collection_window(sessions=sessions, validation_start=sessions[0], validation_end=sessions[1], warmup_sessions=1)
    with pytest.raises(PITDataError, match="execution"):
        derive_historical_collection_window(sessions=sessions[:2], validation_start=sessions[0], validation_end=sessions[1], warmup_sessions=0)


def test_historical_readiness_rejects_invalid_coverage_state_and_global_failure(tmp_path) -> None:
    from datetime import date
    import pytest
    from src.data.collection_plan import EvidenceCoverage, audit_historical_readiness, build_historical_collection_plan
    from src.data.schemas import EvidenceKind, PITDataError

    day = date(2016, 1, 4)
    plan = build_historical_collection_plan(
        sessions=(day,), universe=({'symbol': '005930', 'is_common_stock': True},),
        start=day, end=day, artifact_root=tmp_path / 'plans',
    )
    bad = EvidenceCoverage(EvidenceKind.INVESTOR_FLOW, 'KRX:005930', day, 'weird', None, 'schema')  # type: ignore[arg-type]
    with pytest.raises(PITDataError, match='unknown coverage state'):
        audit_historical_readiness(plan=plan, coverage=(bad,), usable_feature_count_by_session={day: 10}, minimum_cohort=10)
    retry = EvidenceCoverage(EvidenceKind.INVESTOR_FLOW, 'KRX:005930', day, 'retryable_failure', None, 'timeout')
    report = audit_historical_readiness(plan=plan, coverage=(retry,), usable_feature_count_by_session={day: 10}, minimum_cohort=10)
    assert report.certifiable is False


def test_rebuild_cli_fails_closed_without_security_master(tmp_path, monkeypatch) -> None:
    import sys
    from src.data import cli as cli_module
    import src.integrations.dart.xbrl as dart_module
    import src.integrations.krx.historical as krx_module
    import src.integrations.kis.investor_flow as kis_module

    monkeypatch.setattr(dart_module, 'DartXbrlCollector', lambda: object())
    monkeypatch.setattr(krx_module, 'KrxHistoricalCollector', lambda: object())
    monkeypatch.setattr(kis_module, 'KisInvestorFlowCollector', lambda symbols: object())

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'stock-data', 'rebuild-data',
            '--data-root', str(tmp_path / 'data'), '--bronze-root', str(tmp_path / 'bronze'),
            '--silver-root', str(tmp_path / 'silver'), '--gold-root', str(tmp_path / 'gold'),
            '--artifact-root', str(tmp_path / 'artifacts'), '--validation-start', '2016-01-04',
            '--validation-end', '2016-12-29', '--certification-time', '2026-09-05T00:00:00+00:00',
        ],
    )
    assert cli_module.main() == 1


def test_pipeline_log_emits_structured_phase(caplog) -> None:
    from src.data.operations import _pipeline_log

    caplog.set_level('INFO')
    _pipeline_log('test', z=2, a=1)
    assert 'phase=test a=1 z=2' in caplog.text


def test_rebuild_cli_dispatches_with_all_master_symbols(tmp_path, monkeypatch, capsys) -> None:
    import sys
    import polars as pl
    from src.data import cli as cli_module
    import src.data.operations as operations
    import src.integrations.dart.xbrl as dart_module
    import src.integrations.krx.historical as krx_module
    import src.integrations.kis.investor_flow as kis_module

    monkeypatch.setattr(dart_module, 'DartXbrlCollector', lambda: object())
    monkeypatch.setattr(krx_module, 'KrxHistoricalCollector', lambda: object())
    captured: dict[str, tuple[str, ...]] = {}
    monkeypatch.setattr(kis_module, 'KisInvestorFlowCollector', lambda symbols: captured.setdefault('symbols', symbols) or object())
    monkeypatch.setattr(cli_module, '_load_silver_table', lambda root, table: pl.DataFrame({'instrument_id': ['KRX:005930', 'KRX:000660']}))
    monkeypatch.setattr(
        operations,
        'run_historical_data_pipeline',
        lambda request, **kwargs: operations.HistoricalDataPipelineResult('p', tmp_path / 'result.json', {}, 'u', 'f', tmp_path / 'backtest.json', True),
    )
    monkeypatch.setattr(sys, 'argv', ['stock-data', 'rebuild-data', '--data-root', str(tmp_path / 'data'), '--bronze-root', str(tmp_path / 'bronze'), '--silver-root', str(tmp_path / 'silver'), '--gold-root', str(tmp_path / 'gold'), '--artifact-root', str(tmp_path / 'artifacts'), '--validation-start', '2016-01-04', '--validation-end', '2016-12-29', '--certification-time', '2026-09-05T00:00:00+00:00'])
    assert cli_module.main() == 0
    assert captured['symbols'] == ('000660', '005930')
    assert '"plan_id": "p"' in capsys.readouterr().out
