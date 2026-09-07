def test_pit_replay_reader_limits_daily_and_flow_to_required_windows() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.core.time import SessionCalendar
    from src.data.replay import PITReplayReader
    from src.features.contracts import QvefFeaturePolicy
    from src.strategy.universe import UniversePolicy

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    daily = pl.DataFrame([{'instrument_id': 'KRX:1', 'session': s, 'trading_value': 1.0, 'available_at': s} for s in sessions] + [{'instrument_id': 'KRX:F', 'session': sessions[-1], 'trading_value': 1.0, 'available_at': sessions[-1] + timedelta(days=1)}])
    flow = pl.DataFrame([{'instrument_id': 'KRX:1', 'session': s, 'foreign_net_value': 1.0, 'available_at': s} for s in sessions])
    reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=pl.DataFrame(), daily_market=daily, investor_flow=flow, financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())

    replay = reader.session_input(session=sessions[-1], decision_time=sessions[-1], universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())

    assert replay.daily_market.filter(pl.col('instrument_id') == 'KRX:1').height == 60
    assert replay.daily_market.filter(pl.col('instrument_id') == 'KRX:F').height == 0
    assert replay.investor_flow['session'].to_list() == list(sessions[-21:-1])


def test_streaming_gold_writer_rejects_incomplete_session_set(tmp_path) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.core.datasets import DatasetCertification
    from src.data.replay import StreamingGoldWriter
    from src.strategy.universe import UniverseDecision

    session = datetime(2024, 1, 2, tzinfo=UTC)
    writer = StreamingGoldWriter(root=tmp_path / 'gold', dataset_id='a' * 64, decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={'calendar': 'c', 'security_master': 'm'}, expected_sessions=(session,))
    writer.append_universe((UniverseDecision(session, 'KRX:1', True, (), 252, 3_000_000_000.0),))

    with pytest.raises(ValueError, match='incomplete'):
        writer.close()
    assert not (tmp_path / 'gold' / 'universe' / ('a' * 64)).exists()


def test_streaming_gold_writer_publishes_complete_universe(tmp_path) -> None:
    from datetime import UTC, datetime
    from src.core.datasets import DatasetCertification
    from src.data.replay import StreamingGoldWriter
    from src.strategy.universe import ExclusionReason, UniverseDecision

    session = datetime(2024, 1, 2, tzinfo=UTC)
    writer = StreamingGoldWriter(root=tmp_path / 'gold', dataset_id='b' * 64, decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={'calendar': 'c', 'security_master': 'm'}, expected_sessions=(session,))
    writer.append_universe((UniverseDecision(session, 'KRX:1', False, (ExclusionReason.MISSING_MASTER,), 0, None),))
    out = writer.close()
    assert (tmp_path / 'gold' / 'universe' / ('b' * 64)).exists()
    assert 'universe' in out


def test_latest_silver_dataset_path_and_lazy_replay(tmp_path) -> None:
    from datetime import UTC, datetime
    from src.core.time import SessionCalendar
    from src.data.replay import PITReplayReader
    from src.data.schemas import SilverTable
    from src.data.silver import SilverStore, complete_minimal_fixture, latest_silver_dataset_path
    from src.features.contracts import QvefFeaturePolicy
    from src.strategy.universe import UniversePolicy

    t0 = datetime(2024, 1, 3, tzinfo=UTC)
    tables, _, report = complete_minimal_fixture(decision_time=t0)
    now = datetime.now(UTC)
    store = SilverStore(tmp_path / 'silver')
    store.materialize_all(tables, report=report, decision_time=now)
    read_time = datetime.now(UTC)
    for table in SilverTable:
        path = latest_silver_dataset_path(root=tmp_path / 'silver', table=table, decision_time=read_time)
        assert path.exists()
    sessions = tuple(sorted(tables[SilverTable.CALENDAR]['session'].to_list()))
    reader = PITReplayReader.from_silver_root(silver_root=tmp_path / 'silver', decision_time=read_time, calendar=SessionCalendar(sessions))
    replay = reader.session_input(session=sessions[-1], decision_time=read_time, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    assert replay.daily_market.height == 1
    assert replay.security_master.height == 1


def test_materialize_backtest_inputs_bounded_loop_reaches_replay(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta
    from pathlib import Path
    import polars as pl
    import pytest
    import src.data.pipeline as pipeline
    from src.core.time import SessionCalendar
    from src.data.replay import PITReplayReader
    from src.data.schemas import BronzeReceipt, CertificationReport, EvidenceKind, SilverTable
    from src.core.datasets import DatasetCertification

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    decision = sessions[-1]
    monkeypatch.setattr(pipeline, '_require_certified_inputs', lambda *_a, **_k: None)

    def _fake_discover(*_a, **_k):
        return {
            k: (BronzeReceipt(kind=k, content_hash='h', source_path='p', retrieved_at=decision, ingested_at=decision, payload_path=Path('p'), metadata_path=Path('m')),)
            for k in EvidenceKind
        }

    import src.data.bronze_aggregation as _agg
    monkeypatch.setattr(_agg, 'discover_verified_bronze_receipts', _fake_discover)
    monkeypatch.setattr(pipeline, '_load_silver_tables', lambda _r, _d: {SilverTable.CALENDAR: pl.DataFrame({'session': list(sessions)}), SilverTable.CORPORATE_ACTIONS: pl.DataFrame()})
    monkeypatch.setattr(pipeline, 'certify_corporate_action_refresh', lambda **_k: CertificationReport(certification=DatasetCertification.RESEARCH, report_hash='r', coverage_start=sessions[0].date(), coverage_end=sessions[-1].date(), source_hashes=dict.fromkeys(EvidenceKind, 'h')))

    class _FakeReader:
        @classmethod
        def from_silver_root(cls, **_k):
            return PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=pl.DataFrame(), daily_market=pl.DataFrame(), investor_flow=pl.DataFrame(), financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())

    monkeypatch.setattr(pipeline, 'PITReplayReader', _FakeReader)
    with pytest.raises(pipeline.PITDataError, match='no PIT-complete'):
        pipeline.materialize_backtest_inputs(bronze_root=tmp_path / 'bronze', silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold', decision_time=decision)


def test_materialize_gold_window_silver_root_bounded_branch(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta
    from src.core.time import KRX_TZ
    from src.data.gold import materialize_gold_window
    from src.data.schemas import EvidenceKind, SilverTable
    from src.data.silver import SilverStore, certify_silver
    from src.data.schemas import BronzeReceipt
    from src.core.datasets import DatasetCertification
    from pathlib import Path
    import polars as pl
    from src.features.contracts import QvefFeatureRow

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    src_hash = 'h' * 64
    tables = {
        SilverTable.CALENDAR: pl.DataFrame({'session': list(sessions), 'available_at': list(sessions), 'source_hash': [src_hash] * 70}),
        SilverTable.SECURITY_MASTER: pl.DataFrame({'instrument_id': ['KRX:1'], 'ticker': ['1'], 'company_id': ['C1'], 'market': ['KOSPI'], 'sector': ['Technology'], 'listing_date': [datetime(2010, 1, 1, tzinfo=UTC)], 'delisting_date': [None], 'share_class': ['common'], 'status': ['listed'], 'valid_from': [sessions[0]], 'valid_to': [None], 'available_at': [sessions[0]], 'source_hash': [src_hash]}),
        SilverTable.DAILY_MARKET: pl.DataFrame({'session': list(sessions), 'instrument_id': ['KRX:1'] * 70, 'open': [100.0] * 70, 'high': [110.0] * 70, 'low': [90.0] * 70, 'close': [105.0] * 70, 'volume': [1000.0] * 70, 'trading_value': [1e8] * 70, 'market_cap': [1e10] * 70, 'shares_outstanding': [1e8] * 70, 'available_at': list(sessions), 'source_hash': [src_hash] * 70}),
        SilverTable.INVESTOR_FLOW: pl.DataFrame({'session': list(sessions), 'instrument_id': ['KRX:1'] * 70, 'foreign_buy_value': [1e6] * 70, 'foreign_sell_value': [5e5] * 70, 'foreign_net_value': [5e5] * 70, 'institution_net_value': [1e5] * 70, 'retail_net_value': [-6e5] * 70, 'available_at': list(sessions), 'source_hash': [src_hash] * 70}),
        SilverTable.FINANCIAL_FACTS: pl.DataFrame({'company_id': ['C1'], 'fiscal_period': ['2023Q4'], 'filing_id': ['f1'], 'fact': ['sales'], 'published_at': [sessions[0]], 'available_at': [sessions[0]], 'value': [1e9], 'unit': ['KRW'], 'consolidated': [True], 'restatement_id': ['r0'], 'source_hash': [src_hash], 'source_kind': ['opendart_standard'], 'mapping_version': ['v1'], 'raw_document_hash': [None]}),
        SilverTable.CORPORATE_ACTIONS: pl.DataFrame({'instrument_id': ['KRX:1'], 'effective_date': [sessions[0]], 'coverage_end': [sessions[0]], 'action_id': ['a1'], 'type': ['no_action'], 'factor': [1.0], 'cash_amount': [0.0], 'source': ['KRX'], 'available_at': [sessions[0]], 'source_hash': [src_hash]}),
        SilverTable.DISCLOSURES: pl.DataFrame({'company_id': ['C1'], 'filing_id': ['f1'], 'filing_type': ['annual'], 'published_at': [sessions[0]], 'available_at': [sessions[0]], 'correction_of': [None], 'source_hash': [src_hash]}),
        SilverTable.HISTORICAL_COSTS: pl.DataFrame({'market': ['KOSPI'], 'effective_date': [sessions[0]], 'cost_kind': ['commission'], 'rule_id': ['r1'], 'value': [0.00015], 'available_at': [sessions[0]], 'source_hash': [src_hash]}),
    }
    now = datetime.now(UTC)
    receipts = {k: BronzeReceipt(kind=k, content_hash=src_hash, source_path='p', retrieved_at=now, ingested_at=now, payload_path=Path('p'), metadata_path=Path('m')) for k in EvidenceKind}
    report = certify_silver(tables, receipts=receipts, coverage_start=sessions[0].date(), coverage_end=sessions[-1].date(), certification=DatasetCertification.RESEARCH)
    SilverStore(tmp_path / 'silver').materialize_all(tables, report=report, decision_time=now)
    monkeypatch.setattr(
        'src.features.qvef.build_qvef_features',
        lambda **kwargs: (QvefFeatureRow(
            decision_session=kwargs['decision_session'], instrument_id='KRX:1', sector='Technology',
            gross_profitability=None, roe=None, cfo_to_assets=None, book_to_price=None,
            earnings_to_price=None, operating_income_change=None, sales_growth=None,
            operating_margin_change=None, foreign_flow_5=None, foreign_flow_20=None,
            quality_score=None, value_score=None, earnings_score=None, foreign_flow_score=None,
            component_presence=(), source_available_at=(), policy_version=kwargs['policy'].version,
        ),),
    )
    last_date = sessions[-1].astimezone(KRX_TZ).date()
    first_date = sessions[-5].astimezone(KRX_TZ).date()
    decision_time = datetime.now(UTC)
    from src.strategy.universe import UniversePolicy
    out = materialize_gold_window(silver_root=tmp_path / 'silver', validation_start=first_date, validation_end=last_date, decision_time=decision_time, artifact_root=tmp_path / 'artifacts', gold_root=tmp_path / 'gold', universe_policy=UniversePolicy(minimum_listing_sessions=1, minimum_median_trading_value_krw=1.0))
    assert out.manifest is not None
    assert out.universe_decisions_count >= 0


def test_materialize_backtest_inputs_rejects_cert_mismatch_and_empty_order(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta
    from pathlib import Path
    import polars as pl
    import pytest
    import src.data.pipeline as pipeline
    from src.data.schemas import BronzeReceipt, CertificationReport, EvidenceKind, SilverTable
    from src.core.datasets import DatasetCertification

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    decision = sessions[-1]
    monkeypatch.setattr(pipeline, '_require_certified_inputs', lambda *_a, **_k: None)

    def _fake_discover(*_a, **_k):
        return {
            k: (BronzeReceipt(kind=k, content_hash='h', source_path='p', retrieved_at=decision, ingested_at=decision, payload_path=Path('p'), metadata_path=Path('m')),)
            for k in EvidenceKind
        }

    import src.data.bronze_aggregation as _agg
    monkeypatch.setattr(_agg, 'discover_verified_bronze_receipts', _fake_discover)
    monkeypatch.setattr(pipeline, '_load_silver_tables', lambda _r, _d: {SilverTable.CALENDAR: pl.DataFrame({'session': list(sessions)}), SilverTable.CORPORATE_ACTIONS: pl.DataFrame()})
    monkeypatch.setattr(pipeline, 'certify_corporate_action_refresh', lambda **_k: CertificationReport(certification=DatasetCertification.PRODUCTION, report_hash='r', coverage_start=sessions[0].date(), coverage_end=sessions[-1].date(), source_hashes=dict.fromkeys(EvidenceKind, 'h')))
    with pytest.raises(pipeline.PITDataError, match='certification'):
        pipeline.materialize_backtest_inputs(bronze_root=tmp_path / 'b1', silver_root=tmp_path / 's1', gold_root=tmp_path / 'g1', decision_time=decision)

    monkeypatch.setattr(pipeline, 'certify_corporate_action_refresh', lambda **_k: CertificationReport(certification=DatasetCertification.RESEARCH, report_hash='r', coverage_start=sessions[0].date(), coverage_end=sessions[-1].date(), source_hashes=dict.fromkeys(EvidenceKind, 'h')))
    with pytest.raises(pipeline.PITDataError, match='no sessions'):
        pipeline.materialize_backtest_inputs(bronze_root=tmp_path / 'b2', silver_root=tmp_path / 's2', gold_root=tmp_path / 'g2', decision_time=sessions[0] - timedelta(days=1))


def test_materialize_gold_window_requires_frames() -> None:
    from datetime import UTC, datetime
    from pathlib import Path
    import pytest
    from src.data.gold import materialize_gold_window

    with pytest.raises(ValueError, match='calendar'):
        materialize_gold_window(validation_start=datetime(2024, 1, 1, tzinfo=UTC).date(), validation_end=datetime(2024, 1, 2, tzinfo=UTC).date(), decision_time=datetime(2024, 1, 2, tzinfo=UTC), artifact_root=Path('x'))


def test_latest_silver_dataset_path_fail_closed(tmp_path) -> None:
    import json
    from datetime import UTC, datetime
    import pytest
    from src.data.schemas import SilverTable
    from src.data.silver import latest_silver_dataset_path

    with pytest.raises(ValueError, match='timezone-aware'):
        latest_silver_dataset_path(root=tmp_path, table=SilverTable.CALENDAR, decision_time=datetime(2024, 1, 1))
    with pytest.raises(ValueError, match='missing certified Silver'):
        latest_silver_dataset_path(root=tmp_path / 'nosuch', table=SilverTable.CALENDAR, decision_time=datetime(2024, 1, 1, tzinfo=UTC))
    empty_root = tmp_path / 'empty' / SilverTable.CALENDAR.value
    empty_root.mkdir(parents=True)
    with pytest.raises(ValueError, match='missing certified Silver'):
        latest_silver_dataset_path(root=tmp_path / 'empty', table=SilverTable.CALENDAR, decision_time=datetime(2024, 1, 1, tzinfo=UTC))
    bad_root = tmp_path / 'bad' / SilverTable.CALENDAR.value
    bad_root.mkdir(parents=True)
    (bad_root / 'nodata').mkdir()
    (bad_root / ' ').mkdir()
    manifest = {'asset_kind': 'STOCK', 'schema_version': 'v2', 'schema_hash': 's', 'provider_version': 'p', 'universe_policy_version': 'u', 'universe_policy_hash': 'u', 'feature_set': 'stock_pit_calendar_v1', 'feature_set_hash': 'f', 'label_definition': 'none', 'label_horizon_sessions': 1, 'time_start': '2024-01-01T00:00:00+00:00', 'time_end': '2024-01-01T00:00:00+00:00', 'generated_time': '2024-01-01T00:00:00', 'row_count': 0, 'certification': 'research', 'content_hash': 'c'}
    naive_dir = bad_root / 'naive'
    naive_dir.mkdir()
    (naive_dir / 'dataset_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    future = dict(manifest, generated_time='2030-01-01T00:00:00+00:00')
    future_dir = bad_root / 'future'
    future_dir.mkdir()
    (future_dir / 'dataset_manifest.json').write_text(json.dumps(future), encoding='utf-8')
    with pytest.raises(ValueError, match='missing certified Silver'):
        latest_silver_dataset_path(root=tmp_path / 'bad', table=SilverTable.CALENDAR, decision_time=datetime(2024, 6, 1, tzinfo=UTC))


def test_pit_replay_reader_fail_closed_inputs(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    import pytest
    from src.core.time import SessionCalendar
    from src.data.replay import PITReplayReader
    from src.features.contracts import QvefFeaturePolicy
    from src.strategy.universe import UniversePolicy

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=pl.DataFrame(), daily_market=pl.DataFrame(), investor_flow=pl.DataFrame(), financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())
    naive = datetime(2024, 1, 1)
    with pytest.raises(ValueError, match='timezone-aware'):
        reader.session_input(session=naive, decision_time=sessions[0], universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    with pytest.raises(ValueError, match='timezone-aware'):
        reader.session_input(session=sessions[0], decision_time=naive, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    with pytest.raises(ValueError, match='after decision_time'):
        reader.session_input(session=sessions[5], decision_time=sessions[0], universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    with pytest.raises(ValueError, match='calendar'):
        reader.session_input(session=sessions[-1] + timedelta(days=1), decision_time=sessions[-1] + timedelta(days=1), universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    with pytest.raises(ValueError, match='timezone-aware'):
        PITReplayReader.from_silver_root(silver_root=tmp_path, decision_time=naive, calendar=SessionCalendar(sessions))
    with pytest.raises(ValueError, match='missing certified Silver'):
        PITReplayReader.from_silver_root(silver_root=tmp_path / 'nosuch', decision_time=sessions[0], calendar=SessionCalendar(sessions))
    early = reader.session_input(session=sessions[5], decision_time=sessions[5], universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    assert early.investor_flow.height == 0
    assert early.daily_market.height == 0


def test_pit_replay_frames_edge_slices() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    import pytest
    from src.core.time import SessionCalendar
    from src.data.replay import PITReplayReader
    from src.features.contracts import QvefFeaturePolicy
    from src.strategy.universe import UniversePolicy

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    decision = sessions[-1]
    no_avail = pl.DataFrame([{'instrument_id': 'KRX:1', 'session': s} for s in sessions])
    reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=pl.DataFrame(), daily_market=no_avail, investor_flow=pl.DataFrame(), financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())
    assert reader.session_input(session=decision, decision_time=decision, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy()).daily_market.height == 0
    str_avail = pl.DataFrame([{'instrument_id': 'KRX:1', 'session': s, 'available_at': 'bad'} for s in sessions])
    bad_reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=pl.DataFrame(), daily_market=str_avail, investor_flow=pl.DataFrame(), financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())
    with pytest.raises(ValueError, match='available_at'):
        bad_reader.session_input(session=decision, decision_time=decision, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    future_daily = pl.DataFrame([{'instrument_id': 'KRX:1', 'session': s, 'available_at': decision + timedelta(days=1)} for s in sessions])
    future_reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=pl.DataFrame(), daily_market=future_daily, investor_flow=pl.DataFrame(), financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())
    assert future_reader.session_input(session=decision, decision_time=decision, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy()).daily_market.height == 0
    no_session = pl.DataFrame([{'instrument_id': 'KRX:1', 'available_at': s} for s in sessions])
    nosess_reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=pl.DataFrame(), daily_market=no_session, investor_flow=no_session, financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())
    assert nosess_reader.session_input(session=decision, decision_time=decision, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy()).daily_market.height == 0
    master_none = pl.DataFrame([
        {'instrument_id': 'KRX:1', 'valid_from': None, 'valid_to': None, 'available_at': sessions[0]},
        {'instrument_id': 'KRX:2', 'valid_from': None, 'valid_to': None, 'available_at': sessions[0]},
    ])
    none_reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=master_none, daily_market=pl.DataFrame(), investor_flow=pl.DataFrame(), financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())
    assert none_reader.session_input(session=decision, decision_time=decision, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy()).security_master.height == 0
    master_str = pl.DataFrame([
        {'instrument_id': 'KRX:1', 'valid_from': 'bad', 'valid_to': None, 'available_at': sessions[0]},
        {'instrument_id': 'KRX:2', 'valid_from': 'worse', 'valid_to': None, 'available_at': sessions[0]},
    ])
    str_reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=master_str, daily_market=pl.DataFrame(), investor_flow=pl.DataFrame(), financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())
    assert str_reader.session_input(session=decision, decision_time=decision, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy()).security_master.height == 0
    master_ok = pl.DataFrame([
        {'instrument_id': 'KRX:1', 'valid_from': sessions[0], 'valid_to': None, 'available_at': sessions[0], 'company_id': 'C1'},
    ])
    facts = pl.DataFrame([{'company_id': 'C2', 'available_at': decision}])
    m_reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=master_ok, daily_market=pl.DataFrame(), investor_flow=pl.DataFrame(), financial_facts=facts, corporate_actions=pl.DataFrame())
    replay = m_reader.session_input(session=decision, decision_time=decision, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    assert replay.security_master['instrument_id'].to_list() == ['KRX:1']
    assert replay.financial_facts.height == 0


def test_streaming_gold_writer_validation(tmp_path) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.core.datasets import DatasetCertification
    from src.data.replay import StreamingGoldWriter
    from src.features.contracts import QvefFeatureRow
    from src.strategy.scoring import ChampionScoreRow
    from src.strategy.universe import UniverseDecision

    session = datetime(2024, 1, 2, tzinfo=UTC)
    naive = datetime(2024, 1, 2)
    with pytest.raises(ValueError, match='timezone-aware'):
        StreamingGoldWriter(root=tmp_path, dataset_id='c' * 64, decision_time=naive, certification=DatasetCertification.RESEARCH, source_hashes={}, expected_sessions=(session,))
    with pytest.raises(ValueError, match='dataset_id'):
        StreamingGoldWriter(root=tmp_path, dataset_id=' ', decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={}, expected_sessions=(session,))
    with pytest.raises(ValueError, match='expected_sessions'):
        StreamingGoldWriter(root=tmp_path, dataset_id='c' * 64, decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={}, expected_sessions=())
    with pytest.raises(ValueError, match='timezone-aware'):
        StreamingGoldWriter(root=tmp_path, dataset_id='c' * 64, decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={}, expected_sessions=(naive,))
    with pytest.raises(ValueError, match='chronological'):
        StreamingGoldWriter(root=tmp_path, dataset_id='c' * 64, decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={}, expected_sessions=(session, datetime(2024, 1, 1, tzinfo=UTC)))
    writer = StreamingGoldWriter(root=tmp_path / 'w', dataset_id='d' * 64, decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={}, expected_sessions=(session,))
    with pytest.raises(ValueError, match='universe key'):
        writer.append_universe((UniverseDecision(session, ' ', True, (), 1, 1.0),))
    with pytest.raises(ValueError, match='timezone-aware'):
        writer.append_universe((UniverseDecision(naive, 'KRX:1', True, (), 1, 1.0),))
    feature = QvefFeatureRow(decision_session=naive, instrument_id='KRX:1', sector='s', gross_profitability=None, roe=None, cfo_to_assets=None, book_to_price=None, earnings_to_price=None, operating_income_change=None, sales_growth=None, operating_margin_change=None, foreign_flow_5=None, foreign_flow_20=None, quality_score=None, value_score=None, earnings_score=None, foreign_flow_score=None, component_presence=(), source_available_at=(), policy_version='v')
    with pytest.raises(ValueError, match='timezone-aware'):
        writer.append_features((feature,))
    score = ChampionScoreRow(decision_session=naive, instrument_id='KRX:1', eligible=False, champion_score=None, rank=None, exclusion_reasons=(), feature_policy_version='v', score_policy_version='v')
    with pytest.raises(ValueError, match='timezone-aware'):
        writer.append_scores((score,))
    existing = tmp_path / 'w' / 'universe' / ('e' * 64)
    existing.mkdir(parents=True)
    clogged = StreamingGoldWriter(root=tmp_path / 'w', dataset_id='e' * 64, decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={}, expected_sessions=(session,))
    from src.strategy.universe import ExclusionReason as _ER
    clogged.append_universe((UniverseDecision(session, 'KRX:1', False, (_ER.MISSING_MASTER,), 0, None),))
    with pytest.raises(ValueError, match='incomplete'):
        clogged.close()
    from datetime import timedelta as _td
    partial = StreamingGoldWriter(root=tmp_path / 'w2', dataset_id='f' * 64, decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={}, expected_sessions=(session - _td(days=1), session))
    partial.append_universe((UniverseDecision(session, 'KRX:1', False, (_ER.MISSING_MASTER,), 0, None),))
    with pytest.raises(ValueError, match='incomplete'):
        partial.close()


def test_lazy_replay_fail_closed_tables(tmp_path) -> None:
    import json
    from datetime import UTC, datetime
    import pytest
    from src.core.time import SessionCalendar
    from src.data.replay import PITReplayReader
    from src.data.schemas import SilverTable
    from src.data.silver import SilverStore, complete_minimal_fixture
    from src.features.contracts import QvefFeaturePolicy
    from src.strategy.universe import UniversePolicy

    t0 = datetime(2024, 1, 3, tzinfo=UTC)
    tables, _, report = complete_minimal_fixture(decision_time=t0)
    now = datetime.now(UTC)
    SilverStore(tmp_path / 'silver').materialize_all(tables, report=report, decision_time=now)
    read_time = datetime.now(UTC)
    sessions = tuple(sorted(tables[SilverTable.CALENDAR]['session'].to_list()))
    cal = SessionCalendar(sessions)
    nodata_root = tmp_path / 'silver' / SilverTable.DAILY_MARKET.value
    for child in nodata_root.iterdir():
        if child.is_dir():
            for part in child.rglob('*.parquet'):
                part.unlink()
    reader = PITReplayReader.from_silver_root(silver_root=tmp_path / 'silver', decision_time=read_time, calendar=cal)
    with pytest.raises(ValueError, match='daily_market'):
        reader.session_input(session=sessions[-1], decision_time=read_time, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    for child in nodata_root.iterdir():
        if child.is_dir():
            (child / 'corrupt.parquet').write_bytes(b'not parquet')
    with pytest.raises(ValueError, match='daily_market'):
        reader.session_input(session=sessions[-1], decision_time=read_time, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())
    cal_dir = tmp_path / 'silver' / SilverTable.CALENDAR.value
    first_ds = next(p for p in cal_dir.iterdir() if p.is_dir() and not p.name.startswith('.'))
    manifest = json.loads((first_ds / 'dataset_manifest.json').read_text(encoding='utf-8'))
    assert manifest['feature_set'] == 'stock_pit_calendar_v1'


def _build_eligible_silver_fixture(n_sessions: int = 255) -> tuple[tuple, dict]:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.data.schemas import SilverTable

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n_sessions))
    src_hash = 'h' * 64
    tables = {
        SilverTable.CALENDAR: pl.DataFrame({'session': list(sessions), 'available_at': list(sessions), 'source_hash': [src_hash] * n_sessions}),
        SilverTable.SECURITY_MASTER: pl.DataFrame({'instrument_id': ['KRX:1'], 'ticker': ['1'], 'company_id': ['C1'], 'market': ['KOSPI'], 'sector': ['Technology'], 'listing_date': [datetime(2010, 1, 1, tzinfo=UTC)], 'delisting_date': [None], 'share_class': ['common'], 'status': ['listed'], 'valid_from': [sessions[0]], 'valid_to': [None], 'available_at': [sessions[0]], 'source_hash': [src_hash]}),
        SilverTable.DAILY_MARKET: pl.DataFrame({'session': list(sessions), 'instrument_id': ['KRX:1'] * n_sessions, 'open': [100.0] * n_sessions, 'high': [110.0] * n_sessions, 'low': [90.0] * n_sessions, 'close': [105.0] * n_sessions, 'volume': [1000.0] * n_sessions, 'trading_value': [3e9] * n_sessions, 'market_cap': [1e10] * n_sessions, 'shares_outstanding': [1e8] * n_sessions, 'available_at': list(sessions), 'source_hash': [src_hash] * n_sessions}),
        SilverTable.INVESTOR_FLOW: pl.DataFrame({'session': list(sessions), 'instrument_id': ['KRX:1'] * n_sessions, 'foreign_buy_value': [1e6] * n_sessions, 'foreign_sell_value': [5e5] * n_sessions, 'foreign_net_value': [5e5] * n_sessions, 'institution_net_value': [1e5] * n_sessions, 'retail_net_value': [-6e5] * n_sessions, 'available_at': list(sessions), 'source_hash': [src_hash] * n_sessions}),
        SilverTable.FINANCIAL_FACTS: pl.DataFrame({'company_id': ['C1'], 'fiscal_period': ['2023Q4'], 'filing_id': ['f1'], 'fact': ['sales'], 'published_at': [sessions[0]], 'available_at': [sessions[0]], 'value': [1e9], 'unit': ['KRW'], 'consolidated': [True], 'restatement_id': ['r0'], 'source_hash': [src_hash], 'source_kind': ['opendart_standard'], 'mapping_version': ['v1'], 'raw_document_hash': [None]}),
        SilverTable.CORPORATE_ACTIONS: pl.DataFrame({'instrument_id': ['KRX:1'], 'effective_date': [sessions[0]], 'coverage_end': [sessions[-1]], 'action_id': ['a1'], 'type': ['no_action'], 'factor': [1.0], 'cash_amount': [0.0], 'source': ['KRX'], 'available_at': [sessions[0]], 'source_hash': [src_hash]}),
        SilverTable.DISCLOSURES: pl.DataFrame({'company_id': ['C1'], 'filing_id': ['f1'], 'filing_type': ['annual'], 'published_at': [sessions[0]], 'available_at': [sessions[0]], 'correction_of': [None], 'source_hash': [src_hash]}),
        SilverTable.HISTORICAL_COSTS: pl.DataFrame({'market': ['KOSPI'], 'effective_date': [sessions[0]], 'cost_kind': ['commission'], 'rule_id': ['r1'], 'value': [0.00015], 'available_at': [sessions[0]], 'source_hash': [src_hash]}),
    }
    return sessions, tables


def test_materialize_backtest_inputs_bounded_success(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    from pathlib import Path
    import src.data.pipeline as pipeline
    from src.data.schemas import BronzeReceipt, CertificationReport, EvidenceKind, SilverTable
    from src.data.silver import SilverStore, certify_silver
    from src.core.datasets import DatasetCertification

    sessions, tables = _build_eligible_silver_fixture()
    decision = sessions[-1]
    now = datetime.now(UTC)
    receipts = {k: BronzeReceipt(kind=k, content_hash='h' * 64, source_path='p', retrieved_at=now, ingested_at=now, payload_path=Path('p'), metadata_path=Path('m')) for k in EvidenceKind}
    report = certify_silver(tables, receipts=receipts, coverage_start=sessions[0].date(), coverage_end=sessions[-1].date(), certification=DatasetCertification.RESEARCH)
    SilverStore(tmp_path / 'silver').materialize_all(tables, report=report, decision_time=now)
    monkeypatch.setattr(pipeline, '_require_certified_inputs', lambda *_a, **_k: None)

    def _fake_discover(*_a, **_k):
        return {k: (receipts[k],) for k in EvidenceKind}

    import src.data.bronze_aggregation as _agg
    monkeypatch.setattr(_agg, 'discover_verified_bronze_receipts', _fake_discover)
    monkeypatch.setattr(pipeline, '_load_silver_tables', lambda _r, _d: {SilverTable.CALENDAR: tables[SilverTable.CALENDAR], SilverTable.CORPORATE_ACTIONS: tables[SilverTable.CORPORATE_ACTIONS]})
    monkeypatch.setattr(pipeline, 'certify_corporate_action_refresh', lambda **_k: CertificationReport(certification=DatasetCertification.RESEARCH, report_hash=report.report_hash, coverage_start=sessions[0].date(), coverage_end=sessions[-1].date(), source_hashes=dict.fromkeys(EvidenceKind, 'h' * 64)))
    artifact = pipeline.materialize_backtest_inputs(bronze_root=tmp_path / 'bronze', silver_root=tmp_path / 'silver', gold_root=tmp_path / 'gold', artifact_root=tmp_path / 'artifacts', decision_time=datetime.now(UTC))
    assert artifact.universe_hash
    assert artifact.qvef_hash
    assert artifact.champion_scores_hash
    assert (tmp_path / 'gold' / 'universe' / artifact.universe_hash).exists()


def test_materialize_gold_window_silver_root_eligible_branch(tmp_path) -> None:
    from datetime import UTC, datetime
    from src.core.time import KRX_TZ
    from src.data.gold import materialize_gold_window
    from src.data.schemas import BronzeReceipt, EvidenceKind
    from src.data.silver import SilverStore, certify_silver
    from src.core.datasets import DatasetCertification
    from pathlib import Path

    sessions, tables = _build_eligible_silver_fixture()
    now = datetime.now(UTC)
    receipts = {k: BronzeReceipt(kind=k, content_hash='h' * 64, source_path='p', retrieved_at=now, ingested_at=now, payload_path=Path('p'), metadata_path=Path('m')) for k in EvidenceKind}
    report = certify_silver(tables, receipts=receipts, coverage_start=sessions[0].date(), coverage_end=sessions[-1].date(), certification=DatasetCertification.RESEARCH)
    SilverStore(tmp_path / 'silver').materialize_all(tables, report=report, decision_time=now)
    last_date = sessions[-1].astimezone(KRX_TZ).date()
    first_date = sessions[-5].astimezone(KRX_TZ).date()
    out = materialize_gold_window(silver_root=tmp_path / 'silver', validation_start=first_date, validation_end=last_date, decision_time=datetime.now(UTC), artifact_root=tmp_path / 'artifacts', gold_root=None)
    assert out.eligible_decisions_count > 0
    assert out.feature_rows_count > 0


def test_frame_replay_index_returns_exact_windows_without_future_rows() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.core.time import SessionCalendar
    from src.data.replay import PITReplayReader
    from src.features.contracts import QvefFeaturePolicy
    from src.strategy.universe import UniversePolicy

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(70))
    decision = sessions[-1]
    daily = pl.DataFrame([{'instrument_id': 'KRX:1', 'session': item, 'available_at': item, 'trading_value': 1.0} for item in sessions] + [{'instrument_id': 'KRX:future', 'session': decision, 'available_at': decision + timedelta(days=1), 'trading_value': 1.0}])
    flow = pl.DataFrame([{'instrument_id': 'KRX:1', 'session': item, 'available_at': item, 'foreign_net_value': 1.0} for item in sessions])
    reader = PITReplayReader.from_frames(calendar=SessionCalendar(sessions), security_master=pl.DataFrame(), daily_market=daily, investor_flow=flow, financial_facts=pl.DataFrame(), corporate_actions=pl.DataFrame())

    replay = reader.session_input(session=decision, decision_time=decision, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())

    assert replay.daily_market.filter(pl.col('instrument_id') == 'KRX:1').height == 60
    assert replay.daily_market.filter(pl.col('instrument_id') == 'KRX:future').height == 0
    assert replay.investor_flow['session'].to_list() == list(sessions[-21:-1])


def test_streaming_gold_writer_uses_private_staging_without_legacy_mix(tmp_path) -> None:
    from datetime import UTC, datetime
    from src.core.datasets import DatasetCertification
    from src.data.replay import StreamingGoldWriter
    from src.strategy.universe import ExclusionReason, UniverseDecision

    session = datetime(2024, 1, 2, tzinfo=UTC)
    legacy = tmp_path / 'gold' / '.staging-c'
    legacy.mkdir(parents=True)
    legacy.joinpath('batch-00000.pkl').write_bytes(b'legacy')
    writer = StreamingGoldWriter(root=tmp_path / 'gold', dataset_id='c' * 64, decision_time=session, certification=DatasetCertification.RESEARCH, source_hashes={'calendar': 'c', 'security_master': 'm'}, expected_sessions=(session,), require_scores=False)

    writer.append_universe((UniverseDecision(session, 'KRX:1', False, (ExclusionReason.MISSING_MASTER,), 0, None),))

    assert legacy.joinpath('batch-00000.pkl').read_bytes() == b'legacy'
    assert writer._universe_batches[0].parent != legacy
    assert writer._universe_batches[0].name == 'batch-00000.pkl'


def test_resolve_latest_master_snapshot_uses_latest_pit_row_only() -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.data.replay import resolve_latest_master_snapshot

    session = datetime(2016, 1, 5, tzinfo=UTC)
    master = pl.DataFrame({
        'instrument_id': ['KRX:005930', 'KRX:005930'],
        'market': ['__UNKNOWN__', 'KOSPI'],
        'valid_from': [datetime(2016, 1, 4, tzinfo=UTC), session],
        'valid_to': [None, None],
        'available_at': [datetime(2016, 1, 4, tzinfo=UTC), session],
    })

    actual = resolve_latest_master_snapshot(
        master,
        session=session,
        decision_time=session,
    )

    assert actual.select(['instrument_id', 'market']).to_dicts() == [
        {'instrument_id': 'KRX:005930', 'market': 'KOSPI'}
    ]
