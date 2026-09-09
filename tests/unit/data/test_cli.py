import sys

from src.data.cli import _parse_args


def test_collect_command_requires_immutable_plan_id(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stock-data", "collect", "--plan-id", "plan-a"])

    args = _parse_args()

    assert args.command == "collect"
    assert args.plan_id == "plan-a"


def test_rebuild_data_requires_certified_master_before_kis_collection(tmp_path, monkeypatch) -> None:
    from src.data import cli as cli_module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data", "rebuild-data",
            "--data-root", str(tmp_path / "data"),
            "--bronze-root", str(tmp_path / "bronze"),
            "--silver-root", str(tmp_path / "silver"),
            "--gold-root", str(tmp_path / "gold"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--validation-start", "2016-01-04",
            "--validation-end", "2016-12-29",
            "--certification-time", "2026-09-05T00:00:00+00:00",
        ],
    )
    assert cli_module.main() == 1


def test_run_backtest_refuses_without_resolved_execution_components(tmp_path) -> None:
    from argparse import Namespace

    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="requires resolved Gold artifact"):
        _dispatch_backtest(Namespace(gold_root=tmp_path))


def test_run_backtest_validates_selected_bundle_before_execution(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import UTC, datetime

    import polars as pl
    import pytest
    import src.data.cli as cli_mod
    import src.data.gold_artifacts as gold_artifacts
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    calls: list[tuple[str, str]] = []

    def fake_resolve(*, gold_root, dataset_id, decision_time):
        calls.append(("resolve", dataset_id))
        return object()

    def fake_load(*, bundle, decision_time):
        calls.append(("load", "gold-2016"))
        scores = pl.DataFrame(
            [
                {
                    "decision_session": datetime(2016, 1, 4, 15, 30, tzinfo=UTC),
                    "instrument_id": "KRX:000001",
                    "eligible": True,
                    "champion_score": 1.0,
                    "rank": 1,
                    "exclusion_reasons": "",
                    "feature_policy_version": "champion-v1-qvef-v1",
                    "score_policy_version": "champion-v1-scoring-v1",
                }
            ]
        )
        return (object(), object(), scores)

    monkeypatch.setattr(gold_artifacts, "resolve_gold_artifact_bundle", fake_resolve)
    monkeypatch.setattr(gold_artifacts, "load_gold_artifact_frames", fake_load)

    created: dict[str, object] = {}
    real_strategy = cli_mod.ChampionStrategy

    def spy_strategy(*args, **kwargs):  # type: ignore[no-untyped-def]
        created.update(kwargs)
        return real_strategy(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "ChampionStrategy", spy_strategy)

    with pytest.raises(PITDataError, match="requires resolved Gold artifact"):
        _dispatch_backtest(
            Namespace(
                gold_root=tmp_path,
                silver_root=tmp_path / "silver",
                artifact_root=tmp_path / "artifacts",
                validation_start="2016-01-04",
                validation_end="2016-12-30",
                smoke_symbol=None,
                gold_dataset_id="gold-2016",
                strategy_id="champion-v1",
            )
        )

    assert calls == [("resolve", "gold-2016"), ("load", "gold-2016")]
    assert created == {}


def _write_cli_fact_receipt(bronze_root, payload_text) -> None:
    import hashlib
    import json

    raw = payload_text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    receipt_dir = bronze_root / "financial_facts" / digest
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_bytes(raw)
    (receipt_dir / "receipt.json").write_text(
        json.dumps(
            {
                "kind": "financial_facts",
                "content_hash": digest,
                "source_path": "cli",
                "retrieved_at": "2016-01-01T00:00:00+00:00",
                "ingested_at": "2016-01-01T00:00:00+00:00",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_normalize_dart_facts_command_parses_all_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data",
            "normalize-dart-facts",
            "--bronze-root",
            "b",
            "--silver-root",
            "s",
            "--artifact-root",
            "a",
            "--decision-time",
            "2016-12-30T00:00:00+00:00",
            "--batch-size",
            "7",
        ],
    )

    args = _parse_args()

    assert args.command == "normalize-dart-facts"
    assert args.batch_size == 7


def test_normalize_dart_facts_dispatch_publishes(tmp_path, monkeypatch, capsys) -> None:
    from src.data import cli as cli_module

    _write_cli_fact_receipt(
        tmp_path / "bronze",
        '{"records": [{"ticker": "005930", "corp_code": "00126380", "fiscal_period": "2015Q3", "filing_id": "F1", "fact": "sales", "published_at": "2015-11-16T00:00:00+00:00", "value": 10.0, "unit": "KRW"}]}',
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data",
            "normalize-dart-facts",
            "--bronze-root",
            str(tmp_path / "bronze"),
            "--silver-root",
            str(tmp_path / "silver"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--decision-time",
            "2016-12-30T00:00:00+00:00",
        ],
    )

    assert cli_module.main() == 0
    captured = capsys.readouterr()
    assert "output_hash" in captured.out
    assert (tmp_path / "silver" / "financial_facts").exists()


def test_load_silver_table_reads_latest_manifest_dataset(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.core.datasets import DatasetCertification, HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    from src.data.cli import _load_silver_table
    from src.data.schemas import SilverTable
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "session": [datetime(2015, 11, 17, tzinfo=UTC)],
            "available_at": [datetime(2015, 11, 17, tzinfo=UTC)],
            "source_hash": ["r"],
        }
    )
    content_hash = canonical_content_hash(frame, frame.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set="stock_pit_calendar_v1",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=datetime(2015, 11, 17, tzinfo=UTC),
        time_end=datetime(2015, 11, 17, tzinfo=UTC),
        provider_version="t",
        universe_policy_version="v1",
        row_count=frame.height,
        schema_version="v2",
        content_hash=content_hash,
        storage_layout=HIVE_PARTITION_LAYOUT,
        certification=DatasetCertification.RESEARCH,
    )
    ParquetDatasetStore(tmp_path / "silver" / "calendar").write_partitioned(
        frame,
        dataset_id=content_hash,
        manifest=manifest,
        expected_feature_set="stock_pit_calendar_v1",
        decision_time=decision_time,
        content_manifest={},
    )

    loaded = _load_silver_table(tmp_path / "silver", SilverTable.CALENDAR)

    assert loaded.height == 1


def test_normalize_dart_facts_dispatch_reports_failure(tmp_path, monkeypatch, capsys) -> None:
    from src.data import cli as cli_module

    receipt_dir = tmp_path / "bronze" / "financial_facts" / "bad"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text('{"records": []}', encoding="utf-8")
    (receipt_dir / "receipt.json").write_text(
        '{"kind": "financial_facts", "content_hash": "f", "retrieved_at": "2016-01-01T00:00:00+00:00", "ingested_at": "2016-01-01T00:00:00+00:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data",
            "normalize-dart-facts",
            "--bronze-root",
            str(tmp_path / "bronze"),
            "--silver-root",
            str(tmp_path / "silver"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--decision-time",
            "2016-12-30T00:00:00+00:00",
        ],
    )

    assert cli_module.main() == 1
    assert not (tmp_path / "silver" / "financial_facts").exists()


def test_cli_plan_defaults_to_kis_page_capacity(monkeypatch) -> None:
    import sys
    from src.data.cli import _parse_args

    monkeypatch.setattr(sys, 'argv', ['stock-data', 'plan', '--coverage-start', '2024-01-02', '--coverage-end', '2024-01-03', '--symbols', '005930'])
    assert _parse_args().chunk_size == 30


def test_run_backtest_requires_selected_gold_dataset_id(tmp_path) -> None:
    from argparse import Namespace

    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='gold-dataset-id'):
        _dispatch_backtest(
            Namespace(
                gold_root=tmp_path,
                silver_root=tmp_path / 'silver',
                artifact_root=tmp_path / 'artifacts',
                validation_start='2016-01-04',
                validation_end='2016-12-30',
                smoke_symbol=None,
                gold_dataset_id=None,
                strategy_id='champion-v1',
            )
        )


def test_run_backtest_core_v1_end_to_end_with_pit_inputs(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    import src.data.cli as cli_mod
    import src.data.silver as silver_mod
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import SilverTable

    kst = ZoneInfo('Asia/Seoul')
    all_days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(70))
    calendar_df = pl.DataFrame({'session': list(all_days)})
    start, end = all_days[60].date().isoformat(), all_days[64].date().isoformat()
    warmup_days = all_days[0:68]
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
        'valid_to': [all_days[69], all_days[69]],
        'available_at': [all_days[0], all_days[0]],
    })

    def fake_load_table(silver_root, table):
        if table == SilverTable.CALENDAR:
            return calendar_df
        if table == SilverTable.SECURITY_MASTER:
            return master_df
        return pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='Asia/Seoul'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='Asia/Seoul')})

    monkeypatch.setattr(cli_mod, '_load_silver_table', fake_load_table)
    monkeypatch.setattr(silver_mod, 'latest_silver_dataset_path', lambda *, root, table, decision_time: dm_dir)
    universe_dir = tmp_path / 'gold' / 'universe'
    universe_dir.mkdir(parents=True)
    universe_df = pl.DataFrame({
        'decision_session': [day.replace(hour=15, minute=30) for day in all_days[60:65] for _ in ('KRX:A', 'KRX:B')],
        'instrument_id': ['KRX:A', 'KRX:B'] * 5,
        'eligible': [True] * 10,
    })
    universe_df.write_parquet(universe_dir / 'universe.parquet')

    code = _dispatch_backtest(
        Namespace(
            silver_root=tmp_path / 'silver',
            artifact_root=tmp_path / 'artifacts',
            gold_root=tmp_path / 'gold',
            validation_start=start,
            validation_end=end,
            smoke_symbol=None,
            gold_dataset_id=None,
            strategy_id='core-v1',
            initial_cash=100_000_000.0,
            scenario='base',
            ledger_id='core-test-2016',
        )
    )
    assert code == 0
    manifests = list((tmp_path / 'artifacts' / 'backtests').rglob('result.json'))
    assert len(manifests) == 1
    import json

    payload = json.loads(manifests[0].read_text(encoding='utf-8'))
    assert payload['metadata']['strategy_id'] == 'core-v1'
    assert payload['metadata']['market_input_policy_version'] == 'korean-equity-market-inputs-v1'
    assert payload['metadata']['warmup_sessions'] == 60


def test_run_backtest_rejects_market_without_certified_columns(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    import src.data.cli as cli_mod
    import src.data.silver as silver_mod
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError, SilverTable

    kst = ZoneInfo('Asia/Seoul')
    days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(3))
    master = pl.DataFrame({'instrument_id': ['KRX:A'], 'sector': ['Technology'], 'valid_from': [days[0]], 'valid_to': [days[-1]], 'available_at': [days[0]]})
    actions = pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='Asia/Seoul'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='Asia/Seoul')})
    monkeypatch.setattr(cli_mod, '_load_silver_table', lambda silver_root, table: pl.DataFrame({'session': list(days)}) if table == SilverTable.CALENDAR else master if table == SilverTable.SECURITY_MASTER else actions)
    universe_dir = tmp_path / 'gold' / 'universe'
    universe_dir.mkdir(parents=True)
    pl.DataFrame({'decision_session': [days[0]], 'instrument_id': ['KRX:A'], 'eligible': [True]}).write_parquet(universe_dir / 'u.parquet')

    def _run_without(columns: list[str], match: str) -> None:
        dm_dir = tmp_path / f"dm_{'_'.join(columns)}"
        dm_dir.mkdir(exist_ok=True)
        base = {
            'session': list(days),
            'instrument_id': ['KRX:A'] * 3,
            'open': [100.0] * 3,
            'close': [101.0] * 3,
            'volume': [10.0] * 3,
            'trading_value': [1010.0] * 3,
            'market_cap': [1e10] * 3,
            'available_at': [day.replace(hour=15, minute=30) for day in days],
        }
        pl.DataFrame({key: base[key] for key in columns}).write_parquet(dm_dir / 'daily.parquet')
        monkeypatch.setattr(silver_mod, 'latest_silver_dataset_path', lambda *, root, table, decision_time: dm_dir)
        with pytest.raises(PITDataError, match=match):
            _dispatch_backtest(
                Namespace(
                    silver_root=tmp_path / 'silver',
                    artifact_root=tmp_path / 'artifacts',
                    gold_root=tmp_path / 'gold',
                    validation_start=days[0].date().isoformat(),
                    validation_end=days[1].date().isoformat(),
                    smoke_symbol=None,
                    gold_dataset_id=None,
                    strategy_id='core-v1',
                )
            )

    _run_without(['session', 'instrument_id', 'open', 'close', 'volume', 'trading_value', 'market_cap'], 'certified available_at')
    _run_without(['session', 'instrument_id', 'open', 'close', 'volume', 'trading_value', 'available_at'], 'market_cap')


def test_run_backtest_smoke_symbol_completes_with_empty_pit_frames(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    import src.data.cli as cli_mod
    import src.data.silver as silver_mod
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import SilverTable

    kst = ZoneInfo('Asia/Seoul')
    days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(64))
    master = pl.DataFrame({'instrument_id': ['KRX:A'], 'sector': ['Technology'], 'valid_from': [days[0]], 'valid_to': [days[-1]], 'available_at': [days[0]]})
    actions = pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='Asia/Seoul'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='Asia/Seoul')})
    monkeypatch.setattr(cli_mod, '_load_silver_table', lambda silver_root, table: pl.DataFrame({'session': list(days)}) if table == SilverTable.CALENDAR else master if table == SilverTable.SECURITY_MASTER else actions)
    dm_dir = tmp_path / 'dm_smoke'
    dm_dir.mkdir()
    pl.DataFrame({
        'session': list(days),
        'instrument_id': ['KRX:A'] * len(days),
        'open': [10000.0 + index for index in range(len(days))],
        'close': [10050.0 + index + (index % 3) * 0.1 for index in range(len(days))],
        'volume': [1000000.0] * len(days),
        'trading_value': [10050000000.0] * len(days),
        'market_cap': [1e12] * len(days),
        'available_at': [day.replace(hour=15, minute=30) for day in days],
    }).write_parquet(dm_dir / 'daily.parquet')
    monkeypatch.setattr(silver_mod, 'latest_silver_dataset_path', lambda *, root, table, decision_time: dm_dir)
    code = _dispatch_backtest(
        Namespace(
            silver_root=tmp_path / 'silver',
            artifact_root=tmp_path / 'artifacts',
            gold_root=tmp_path / 'gold',
            validation_start=days[60].date().isoformat(),
            validation_end=days[60].date().isoformat(),
            smoke_symbol='KRX:A',
            gold_dataset_id=None,
            strategy_id='core-v1',
            initial_cash=100_000_000.0,
            scenario='base',
            ledger_id='smoke-test',
        )
    )
    assert code == 0


def test_build_gold_uses_four_factor_default_policy(tmp_path, monkeypatch, capsys) -> None:
    import sys
    from types import SimpleNamespace

    import src.data.gold as gold_mod
    import src.data.gold_loader as loader_mod
    from src.data.cli import main

    captured: dict[str, object] = {}

    def fake_load(**kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            calendar=object(),
            security_master=object(),
            daily_market=object(),
            financial_facts=object(),
            corporate_actions=object(),
            investor_flow=object(),
        )

    def fake_materialize(**kwargs):  # type: ignore[no-untyped-def]
        captured['score_policy'] = kwargs['score_policy']
        manifest = SimpleNamespace(
            manifest_hash='hash',
            warmup=SimpleNamespace(warmup_ok=True, warmup_sessions_found=60),
            bar_audit=[],
            dart_eligibility=[],
            ca_excluded_instrument_ids=[],
            eligible_instrument_ids=[],
        )
        return SimpleNamespace(
            manifest=manifest,
            universe_decisions_count=0,
            eligible_decisions_count=0,
            feature_rows_count=0,
            universe_path='u',
            features_path='f',
            summary_artifact_path='s',
        )

    monkeypatch.setattr(loader_mod, 'load_gold_window_inputs', fake_load)
    monkeypatch.setattr(gold_mod, 'materialize_gold_window', fake_materialize)
    monkeypatch.setattr(sys, 'argv', [
        'stock-data', 'build-gold',
        '--silver-root', str(tmp_path),
        '--artifact-root', str(tmp_path),
        '--decision-time', '2024-01-03T00:00:00+00:00',
        '--validation-start', '2016-01-04',
        '--validation-end', '2016-12-30',
    ])
    assert main() == 0
    assert captured['score_policy'].min_required_factors == 4
    capsys.readouterr()


def test_run_backtest_parser_accepts_gold_dataset_id(monkeypatch) -> None:
    import sys

    from src.data.cli import _parse_args

    monkeypatch.setattr(
        sys,
        'argv',
        ['stock-data', 'run-backtest', '--gold-dataset-id', 'gold-2016-verified'],
    )

    args = _parse_args()

    assert args.command == 'run-backtest'
    assert args.gold_dataset_id == 'gold-2016-verified'


def test_cli_refresh_corporate_actions_wires_certified_scan(monkeypatch, tmp_path, capsys) -> None:
    from datetime import UTC, datetime
    import polars as pl
    import src.data.cli as cli
    from src.core.time import SessionCalendar

    decision = datetime(2026, 9, 9, tzinfo=UTC)
    calendar_frame = pl.DataFrame({'session': [decision]})
    captured = {}
    monkeypatch.setattr(cli, '_parse_dt', lambda _value: decision)
    monkeypatch.setattr(cli, 'load_latest_silver_table', lambda **kwargs: calendar_frame)
    monkeypatch.setattr(cli, 'load_latest_silver_market_scan', lambda **kwargs: pl.DataFrame({'session': [decision], 'instrument_id': ['KRX:A'], 'close': [1.0], 'shares_outstanding': [1.0], 'market_cap': [1.0]}).lazy())
    def refresh(**kwargs):
        captured['kwargs'] = kwargs
        return type('Report', (), {'report_hash': 'refresh-hash'})()
    monkeypatch.setattr(cli, 'refresh_corporate_action_silver', refresh)
    code = cli.main(['refresh-corporate-actions', '--bronze-root', str(tmp_path / 'bronze'), '--silver-root', str(tmp_path / 'silver'), '--artifact-root', str(tmp_path / 'artifacts'), '--decision-time', decision.isoformat()])
    assert code == 0
    assert isinstance(captured['kwargs']['calendar'], SessionCalendar)
    assert captured['kwargs']['daily_market'].collect().columns == ['session', 'instrument_id', 'close', 'shares_outstanding', 'market_cap']
    assert 'refresh-hash' in capsys.readouterr().out
