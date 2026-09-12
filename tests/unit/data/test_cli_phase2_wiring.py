def _phase2_backtest_namespace(tmp_path, monkeypatch, *, master_rows_per_session: int = 1):
    """core-v1 manifest-bound 백테스트 실행용 최소 PIT 하네스."""
    from argparse import Namespace
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    import src.data.gold_artifacts as gold_artifacts_mod
    import src.data.silver as silver_mod
    from src.data.backtest_run_manifest import build_backtest_run_manifest, write_backtest_run_manifest
    from src.data.schemas import SilverTable

    kst = ZoneInfo("Asia/Seoul")
    all_days = tuple(datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=index) for index in range(70))
    # 캘린더는 00:00 KST 앵커로 저장된 실제 Silver 인코딩을 재현한다.
    calendar_df = pl.DataFrame({"session": [d.replace(hour=0) for d in all_days]})
    start, end = all_days[60].date().isoformat(), all_days[64].date().isoformat()
    warmup_days = all_days[0:68]
    closes_a = [10000.0 + index * 10.0 for index in range(len(warmup_days))]
    closes_b = [20000.0 + index * 5.0 for index in range(len(warmup_days))]
    market_df = pl.DataFrame({
        "session": [day for day in warmup_days for _ in ("KRX:A", "KRX:B")],
        "instrument_id": ["KRX:A", "KRX:B"] * len(warmup_days),
        "open": [c - 5.0 for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        "close": [c for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        "volume": [1000.0] * (2 * len(warmup_days)),
        "trading_value": [c * 1000.0 for pair in zip(closes_a, closes_b, strict=True) for c in pair],
        "market_cap": [1e12, 5e11] * len(warmup_days),
        "available_at": [day.replace(hour=15, minute=30) for day in warmup_days for _ in ("KRX:A", "KRX:B")],
    })
    dm_dir = tmp_path / "dm"
    dm_dir.mkdir(exist_ok=True)
    market_df.write_parquet(dm_dir / "daily.parquet")

    master_days = all_days if master_rows_per_session > 0 else all_days[:1]
    master_df = pl.DataFrame({
        "instrument_id": ["KRX:A", "KRX:B"] * len(master_days),
        "sector": ["Technology", "Healthcare"] * len(master_days),
        "status": ["listed"] * (2 * len(master_days)),
        "valid_from": [day for day in master_days for _ in ("KRX:A", "KRX:B")],
        "valid_to": [day for day in master_days for _ in ("KRX:A", "KRX:B")],
        "available_at": [day for day in master_days for _ in ("KRX:A", "KRX:B")],
        "source_hash": ["h"] * (2 * len(master_days)),
    })
    empty_actions = pl.DataFrame(schema={
        "instrument_id": pl.String,
        "effective_date": pl.Datetime(time_zone="Asia/Seoul"),
        "type": pl.String,
        "factor": pl.Float64,
        "cash_amount": pl.Float64,
        "available_at": pl.Datetime(time_zone="Asia/Seoul"),
        "evidence_status": pl.String,
        "evidence_reason": pl.String,
    })

    def fake_load_by_id(*, root, table, dataset_id, decision_time):
        if table == SilverTable.CALENDAR:
            return calendar_df
        if table == SilverTable.SECURITY_MASTER:
            return master_df
        if table == SilverTable.CORPORATE_ACTIONS:
            return empty_actions
        return pl.DataFrame()

    universe_df = pl.DataFrame({
        "decision_session": [day.replace(hour=15, minute=30) for day in all_days[60:65] for _ in ("KRX:A", "KRX:B")],
        "instrument_id": ["KRX:A", "KRX:B"] * 5,
        "eligible": [True] * 10,
    })

    monkeypatch.setattr(silver_mod, "load_silver_table_by_dataset_id", fake_load_by_id)
    monkeypatch.setattr(silver_mod, "silver_dataset_path_by_id", lambda *, root, table, dataset_id, decision_time: dm_dir)
    monkeypatch.setattr(gold_artifacts_mod, "resolve_gold_artifact_bundle", lambda *, gold_root, dataset_id, decision_time: object())
    monkeypatch.setattr(gold_artifacts_mod, "load_gold_artifact_frames", lambda *, bundle, decision_time: (universe_df, pl.DataFrame(), pl.DataFrame()))

    run_manifest_obj = build_backtest_run_manifest(
        silver_root=tmp_path / "silver",
        gold_root=tmp_path / "gold",
        silver_dataset_ids={table: f"{table.value}-id" for table in SilverTable},
        gold_dataset_id="gold-test",
        validation_start=__import__("datetime").date.fromisoformat(start),
        validation_end=__import__("datetime").date.fromisoformat(end),
        strategy_id="core-v1",
        policy_versions={"market_inputs": "korean-equity-market-inputs-v2"},
    )
    manifest_path = write_backtest_run_manifest(manifest=run_manifest_obj, artifact_root=tmp_path / "artifacts")

    return Namespace(
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        gold_root=tmp_path / "gold",
        validation_start=start,
        validation_end=end,
        smoke_symbol=None,
        gold_dataset_id="gold-test",
        backtest_run_manifest=manifest_path,
        strategy_id="core-v1",
        initial_cash=100_000_000.0,
        scenario="base",
        ledger_id="core-test-2016",
    )


def test_run_backtest_builds_sessions_exactly_once(tmp_path, monkeypatch) -> None:
    """INV-SINGLE-SESSION-BUILD: 제외 결정이 1회 전처리로 확정되어 재빌드가 없어야 한다."""
    import src.data.backtest_sessions as bs_mod
    import src.data.cli as cli_mod
    from src.data.cli import _dispatch_backtest

    args = _phase2_backtest_namespace(tmp_path, monkeypatch)

    calls: list[int] = []
    real_build = bs_mod.build_backtest_sessions

    def spy_build(**kwargs):
        calls.append(1)
        return real_build(**kwargs)

    monkeypatch.setattr(cli_mod, "build_backtest_sessions", spy_build)

    code = _dispatch_backtest(args)

    assert code == 0
    assert len(calls) == 1


def test_run_backtest_records_exclusion_plan_and_time_semantics_metadata(tmp_path, monkeypatch) -> None:
    """제외 계획과 관측된 시간 의미가 런 메타데이터에 기록되어야 한다."""
    import json

    from src.data.cli import _dispatch_backtest

    args = _phase2_backtest_namespace(tmp_path, monkeypatch)

    code = _dispatch_backtest(args)

    assert code == 0
    manifests = list((tmp_path / "artifacts" / "backtests").rglob("result.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert "exclusion_reason_counts" in metadata
    assert metadata["excluded_missing_market_close_instruments"] == []
    assert metadata["excluded_unexplained_corporate_action_instruments"] == []
    semantics = metadata["silver_time_semantics"]
    assert isinstance(semantics, list)
    assert any(entry["column"] == "session" for entry in semantics)
    assert all({"table", "column", "time_zone", "hour_anchors", "canonical"} <= set(entry) for entry in semantics)


def test_run_backtest_canonicalizes_midnight_calendar_without_replace_hack(tmp_path, monkeypatch) -> None:
    """00:00 KST 로 저장된 캘린더가 canonicalize_session_keys 로 09:00 KST 앵커가 되어야 한다."""
    import src.data.cli as cli_mod
    from src.data.cli import _dispatch_backtest
    from src.data.silver_schema import canonicalize_session_keys

    args = _phase2_backtest_namespace(tmp_path, monkeypatch)

    seen: list[int] = []
    real = canonicalize_session_keys

    def spy(frame):
        result = real(frame)
        if "session" in result.columns and result.height:
            seen.extend(sorted({int(v.hour) for v in result["session"].to_list()}))
        return result

    monkeypatch.setattr(cli_mod, "canonicalize_session_keys", spy)

    code = _dispatch_backtest(args)

    assert code == 0
    assert seen
    assert set(seen) == {9}


def test_run_backtest_propagates_session_build_failure_without_regex_retry(tmp_path, monkeypatch) -> None:
    """INV-NO-ERROR-STRING-CONTROL: 빌드 실패는 메시지 파싱 재시도 없이 즉시 전파된다."""
    import pytest

    import src.data.cli as cli_mod
    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    args = _phase2_backtest_namespace(tmp_path, monkeypatch)

    attempts: list[int] = []

    def failing_build(**kwargs):
        attempts.append(1)
        raise PITDataError("unexplained price discontinuity for 'KRX:A'")

    monkeypatch.setattr(cli_mod, "build_backtest_sessions", failing_build)

    with pytest.raises(PITDataError, match="unexplained price discontinuity"):
        _dispatch_backtest(args)

    assert len(attempts) == 1
