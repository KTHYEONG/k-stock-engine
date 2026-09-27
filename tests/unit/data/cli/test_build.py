"""Build-area CLI commands (normalize, silver builds, quality, panel, benchmarks)."""
from __future__ import annotations

import sys

import json

from pathlib import Path

from src.data.cli import _parse_args
from tests.fixtures import seed_receipts
from tests.fixtures.cli_fixtures import (
    _publish_cli_fixture,
    _register_cli_current,
    _publish_cli_session_dataset,
    _write_cli_fact_receipt,
    _write_cli_gap_dataset,
    _write_cli_gap_inputs,
    _write_cli_kis_page,
    _stage_quality_facts_dataset,
    _quality_cli_args,
    _dataset_runtime,
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
    assert "quarantined_filings" in captured.out
    assert "quarantine_path" in captured.out
    assert (tmp_path / "silver").is_dir()
    assert any(path.is_dir() and path.name.startswith("financial_facts_") for path in (tmp_path / "silver").iterdir())



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

    assert cli_module.main() == 2
    assert not (tmp_path / "silver" / "financial_facts").exists()



def test_build_investor_flow_silver_command_emits_dataset_counts(tmp_path, capsys) -> None:
    import hashlib
    import json
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    import polars as pl

    universe = _publish_cli_fixture(
        runtime.workspace.silver_root,
        "ordinary_universe",
        {
            "session=2026-03-04/part.parquet": pl.DataFrame(
                {"session": [date(2026, 3, 4)], "instrument_id": ["KRX:005930"], "eligible": [True]}
            ),
            "session=2026-03-05/part.parquet": pl.DataFrame(
                {"session": [date(2026, 3, 5)], "instrument_id": ["KRX:005930"], "eligible": [True]}
            ),
        },
    )
    row = {
        "date": "20260304",
        "tjj0000": "-100", "tjj0001": "-50", "tjj0002": "-30", "tjj0003": "-20",
        "tjj0004": "-10", "tjj0005": "-10", "tjj0006": "-8", "tjj0007": "100",
        "tjj0008": "927", "tjj0009": "-800", "tjj0010": "-28", "tjj0011": "29",
        "tjj0016": "-828", "tjj0017": "129", "tjj0018": "-228",
        "close": "50000", "volume": "10000", "value": "500",
    }
    payload = {
        "provider": "LS", "endpoint": "frgr-itt", "symbol": "005930", "anchor": "2026-03-04",
        "query": {"symbol": "005930", "start": "2026-03-04", "end": "2026-03-04"},
        "rows": [row], "records": [],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    page_dir = runtime.workspace.bronze_root / "investor_flow" / digest
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "payload.json").write_bytes(raw)
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from src.core.pit import EvidenceKind as _Kind
    from src.data.receipt_catalog import BlobEntry as _Blob, ReceiptCatalog as _Catalog

    _Catalog(runtime.workspace.bronze_root / "catalog").publish(
        [],
        blobs=[
            _Blob(
                content_hash=digest, kind=_Kind.INVESTOR_FLOW, source="ls_investor_flow",
                usable=True, unusable_reason=None, retrieved_at=_dt(2026, 3, 4, tzinfo=_UTC),
                payload_path=page_dir / "payload.json",
            )
        ],
    )
    _register_cli_current(runtime, "ordinary_universe", universe)
    assert main([
        "build-investor-flow-silver",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--workers", "1",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("investor_flow_")
    assert emitted["rows"] == 1



def test_build_daily_market_silver_command_emits_dataset_counts(tmp_path, capsys) -> None:
    import hashlib
    import json

    from src.data.cli import main
    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
    from src.data.runtime import load_data_runtime
    from datetime import UTC, date, datetime

    import polars as pl

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    universe = _publish_cli_fixture(
        runtime.workspace.silver_root,
        "ordinary_universe",
        {
            "session=2026-03-04/part.parquet": pl.DataFrame(
                {"session": [date(2026, 3, 4)], "instrument_id": ["KRX:005930"], "eligible": [True]}
            ),
        },
    )
    record = {
        "ISU_CD": "KR7005930003", "ISU_SRT_CD": "005930", "MKT_NM": "KOSPI", "BAS_DD": "20260304",
        "TDD_OPNPRC": "10500", "TDD_HGPRC": "11200", "TDD_LWPRC": "10300", "TDD_CLSPRC": "11000",
        "CMPPREVDD_PRC": "1000", "FLUC_RT": "10.0", "ACC_TRDVOL": "1000", "ACC_TRDVAL": "11000000",
        "MKTCAP": "660000000000", "LIST_SHRS": "60000000",
    }
    raw = json.dumps({"session": "2026-03-04", "records": [record]}, sort_keys=True).encode("utf-8")
    page_path = runtime.workspace.bronze_root / "krx-page.json"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_bytes(raw)
    seed_receipts(
        ReceiptCatalog(runtime.workspace.bronze_root / "catalog"),
        [
            ReceiptIndexEntry(
                source="krx_daily_market",
                natural_key="2026-03-04",
                as_of=date(2026, 3, 4),
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash=hashlib.sha256(raw).hexdigest(),
                retrieved_at=datetime(2026, 3, 5, tzinfo=UTC),
                payload_path=page_path,
            )
        ],
    )
    _register_cli_current(runtime, "ordinary_universe", universe)
    assert main([
        "build-daily-market-silver",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("daily_market_")
    assert emitted["rows"] == 1



def test_build_market_panel_command_emits_dataset_counts(tmp_path, capsys) -> None:
    import json
    from datetime import date, datetime

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]

    def daily_row(session: date, ticker: str, close: int, change: int, volume: int) -> dict[str, object]:
        return {
            "session": session, "instrument_id": f"KRX:{ticker}", "ticker": ticker, "market": "KOSPI",
            "open": close, "high": close, "low": close, "close": close, "change": change,
            "base_price": close - change, "volume": volume, "trading_value": close * volume,
            "market_cap": close * 1000, "listed_shares": 1000, "price_state": "tradable",
            "invalid_reason": None,
            "available_at": datetime(session.year, session.month, session.day, 18, 0, tzinfo=KRX_TZ),
            "source_hash": "a" * 64, "policy_version": "krx-daily-market-v1",
        }

    def tickers_for(session: date) -> list[str]:
        return ["005930", "000660"] if session != sessions[2] else ["005930"]

    daily_by_session = {
        session: pl.DataFrame(
            [
                daily_row(session, ticker, 10000 if ticker == "005930" else 5000, 0,
                          0 if (ticker, session) == ("000660", sessions[1]) else 100)
                for ticker in tickers_for(session)
            ]
        ).sort("ticker")
        for session in sessions
    }
    universe_by_session = {
        session: pl.DataFrame(
            [
                {"session": session, "instrument_id": f"KRX:{ticker}", "ticker": ticker,
                 "eligible": True, "exclusion_reason": "eligible"}
                for ticker in tickers_for(session)
            ]
        ).sort("instrument_id")
        for session in sessions
    }
    daily_dir = _publish_cli_session_dataset(runtime.workspace.silver_root, "daily_market", daily_by_session)
    universe_dir = _publish_cli_session_dataset(runtime.workspace.silver_root, "ordinary_universe", universe_by_session)
    assert main([
        "build-market-panel",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--daily-market-dataset-id", daily_dir.name,
        "--universe-dataset-id", universe_dir.name,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("market_panel_")
    assert emitted["rows"] == 5
    assert emitted["exits_halted"] == 1



def test_build_market_panel_command_forwards_instrument_buckets(tmp_path, capsys) -> None:
    import json
    from datetime import date, datetime

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]

    def daily_row(session: date, ticker: str) -> dict[str, object]:
        return {
            "session": session, "instrument_id": f"KRX:{ticker}", "ticker": ticker, "market": "KOSPI",
            "open": 10000, "high": 10000, "low": 10000, "close": 10000, "change": 0,
            "base_price": 10000, "volume": 100, "trading_value": 1000000,
            "market_cap": 10000000, "listed_shares": 1000, "price_state": "tradable",
            "invalid_reason": None,
            "available_at": datetime(session.year, session.month, session.day, 18, 0, tzinfo=KRX_TZ),
            "source_hash": "a" * 64, "policy_version": "krx-daily-market-v1",
        }

    daily_by_session = {
        session: pl.DataFrame([daily_row(session, ticker) for ticker in ("005930", "000660")]).sort("ticker")
        for session in sessions
    }
    universe_by_session = {
        session: pl.DataFrame(
            [
                {"session": session, "instrument_id": f"KRX:{ticker}", "ticker": ticker,
                 "eligible": True, "exclusion_reason": "eligible"}
                for ticker in ("005930", "000660")
            ]
        ).sort("instrument_id")
        for session in sessions
    }
    daily_dir = _publish_cli_session_dataset(runtime.workspace.silver_root, "daily_market", daily_by_session)
    universe_dir = _publish_cli_session_dataset(runtime.workspace.silver_root, "ordinary_universe", universe_by_session)
    assert main([
        "build-market-panel",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--daily-market-dataset-id", daily_dir.name,
        "--universe-dataset-id", universe_dir.name,
        "--instrument-buckets", "4",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("market_panel_")
    assert Path(emitted["dataset_path"]).exists()
    assert main([
        "build-market-panel",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--daily-market-dataset-id", daily_dir.name,
        "--universe-dataset-id", universe_dir.name,
    ]) == 0
    default_emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"] == default_emitted["dataset_id"]



def test_build_reference_benchmarks_command_lists_all_ids(tmp_path, capsys) -> None:
    import json
    from datetime import date

    import polars as pl

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]
    rows = [
        {
            "session": session, "instrument_id": f"KRX:{ticker}", "eligible": True,
            "price_state": "tradable", "adtv20": 2_000_000_000.0,
            "market_cap": cap, "ret_price": 0.0 if session == sessions[0] else ret,
        }
        for session in sessions
        for ticker, cap, ret in (("005930", 300, 0.04), ("000660", 100, 0.0))
    ]
    panel_dir = _publish_cli_fixture(
        runtime.workspace.gold_root,
        "market_panel",
        {"year=2020/part.parquet": pl.DataFrame(rows).sort(["instrument_id", "session"])},
        layer="gold",
    )
    assert main([
        "build-reference-benchmarks",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--market-panel-dataset-id", panel_dir.name,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("reference_benchmarks_")
    assert sorted(emitted["benchmarks"]) == [
        "eligible_cw_pr", "eligible_ew_pr", "liquid1b_cw_pr", "liquid1b_ew_pr",
    ]



def test_compact_storage_generations_command_is_removed(tmp_path) -> None:
    import pytest

    from src.data.cli import main

    with pytest.raises(SystemExit):
        main([
            "compact-storage-generations",
            "--scope-config", "config/research/kr_swing_2019_v1.toml",
            "--data-root", str(tmp_path / "data"),
        ])



def test_build_investor_flow_kis_supplement_command_emits_coverage(tmp_path, capsys) -> None:
    import json
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    s1, s2, s3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    panel, ls_flow = _write_cli_gap_inputs(
        runtime,
        ls_cells=[(s1, "000001")],
        panel_cells=[(s1, "000001"), (s2, "000001"), (s3, "000001")],
    )
    _write_cli_kis_page(
        runtime.workspace.bronze_root,
        "000001",
        s2,
        [{
            "stck_bsop_date": "20240103",
            "prsn_ntby_qty": "100",
            "frgn_ntby_qty": "-60",
            "orgn_ntby_qty": "-30",
            "etc_ntby_qty": "-10",
        }],
    )
    assert main([
        "build-investor-flow-kis-supplement",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--ls-flow-dataset-id", ls_flow.name,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("investor_flow_kis_supplement_")
    assert emitted["filled_cells"] == 1
    assert emitted["still_missing_cells"] == 1



def test_build_investor_flow_union_command_emits_dataset(tmp_path, capsys) -> None:
    import json
    from datetime import date, datetime

    import polars as pl

    from src.core.time import KRX_TZ
    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    s1, s2 = date(2024, 1, 2), date(2024, 1, 3)

    def _row(provider, session, ticker, values):
        return {
            "session": session,
            "instrument_id": f"KRX:{ticker}",
            "ticker": ticker,
            "provider": provider,
            "individual_net_shares": values[0],
            "foreign_net_shares": values[1],
            "institution_net_shares": values[2],
            "other_net_shares": values[3],
            "available_at": datetime(session.year, session.month, session.day, 8, 0, tzinfo=KRX_TZ),
            "source_hash": f"{provider}-{ticker}",
            "policy_version": "test-v1",
        }

    ls_row = _row("LS", s1, "000001", (100, -60, -30, -10)) | {"ls_close": 50000}
    ls_flow = runtime.workspace.silver_root / "investor_flow_cli_ls"
    ls_flow = _write_cli_gap_dataset(
        ls_flow,
        pl.DataFrame(
            [ls_row],
            schema={**dict.fromkeys(ls_row, pl.String), **{
                "session": pl.Date,
                "individual_net_shares": pl.Int64,
                "foreign_net_shares": pl.Int64,
                "institution_net_shares": pl.Int64,
                "other_net_shares": pl.Int64,
                "ls_close": pl.Int64,
                "available_at": pl.Datetime("us", "Asia/Seoul"),
            }},
        ),
    )
    kis_flow = runtime.workspace.silver_root / "investor_flow_kis_supplement_cli"
    kis_flow = _write_cli_gap_dataset(
        kis_flow,
        pl.DataFrame([_row("KIS", s2, "000002", (50, -30, -10, -10))]),
    )
    assert main([
        "build-investor-flow-union",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--ls-flow-dataset-id", ls_flow.name,
        "--kis-supplement-dataset-id", kis_flow.name,
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("investor_flow_")
    assert emitted["rows"] == 2
    assert emitted["ls_dataset_id"] == ls_flow.name
    assert emitted["kis_supplement_dataset_id"] == kis_flow.name



def test_build_industry_classification_silver_command_emits_dataset(tmp_path, capsys) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.cli import main
    from src.data.runtime import load_data_runtime
    from src.core.pit import EvidenceKind
    from src.data.receipt_catalog import BlobEntry, ReceiptCatalog

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    collected_at = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)
    payload = {
        "provider": "KIS",
        "endpoint": "inquire-price",
        "symbol": "005930",
        "collected_at": collected_at.isoformat(),
        "output": {"bstp_kor_isnm": "전기·전자", "rprs_mrkt_kor_name": "KOSPI"},
        "records": [{"ticker": "005930", "industry_name": "전기·전자", "market_name": "KOSPI"}],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    import hashlib as _hashlib

    digest = _hashlib.sha256(raw).hexdigest()
    target = runtime.workspace.bronze_root / "industry" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    ReceiptCatalog(runtime.workspace.bronze_root / "catalog").publish(
        [],
        blobs=[
            BlobEntry(
                content_hash=digest, kind=EvidenceKind.INDUSTRY, source="kis_industry",
                usable=True, unusable_reason=None, retrieved_at=collected_at,
                payload_path=target / "payload.json",
            )
        ],
    )
    assert main([
        "build-industry-classification-silver",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("industry_")
    assert emitted["rows"] == 1



def test_build_financial_quality_command_publishes_dataset(tmp_path, capsys) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    decision_time = datetime(2020, 4, 1, tzinfo=UTC)
    facts_id = _stage_quality_facts_dataset(runtime.workspace.silver_root, decision_time=decision_time)

    assert main(_quality_cli_args(scope_config, runtime, facts_id, tmp_path)) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["rows"] == 3
    assert emitted["quarantine_events"] == 1
    assert emitted["manual_events"] == 1
    assert emitted["incomplete_periods"] == 2
    dataset_dir = Path(emitted["dataset_path"])
    assert dataset_dir.is_dir()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["facts"] == facts_id



def test_build_financial_quality_command_is_idempotent(tmp_path, capsys) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    decision_time = datetime(2020, 4, 1, tzinfo=UTC)
    facts_id = _stage_quality_facts_dataset(runtime.workspace.silver_root, decision_time=decision_time)

    assert main(_quality_cli_args(scope_config, runtime, facts_id, tmp_path)) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(_quality_cli_args(scope_config, runtime, facts_id, tmp_path)) == 0
    second = json.loads(capsys.readouterr().out)

    assert second["dataset_id"] == first["dataset_id"]
    quality_root = runtime.workspace.silver_root
    assert [p for p in quality_root.iterdir() if p.is_dir() and p.name.startswith("financial_quality_")] == [
        Path(first["dataset_path"])
    ]



def test_build_financial_quality_command_rejects_unknown_facts_dataset(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")

    assert main(_quality_cli_args(scope_config, runtime, "0" * 64, tmp_path)) == 2
    assert "error" in json.loads(capsys.readouterr().out)



def test_build_financial_quality_command_rejects_event_without_reason(tmp_path, capsys) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    decision_time = datetime(2020, 4, 1, tzinfo=UTC)
    facts_id = _stage_quality_facts_dataset(runtime.workspace.silver_root, decision_time=decision_time)
    argv = _quality_cli_args(scope_config, runtime, facts_id, tmp_path)
    manual_path = tmp_path / "manual.json"
    manual_path.write_text(
        json.dumps([{"company_id": "000660", "fiscal_period": "2019Q4", "filing_id": "M1",
                     "published_at": "2020-03-30T00:00:00+00:00", "available_at": "2020-03-31T00:00:00+00:00"}]),
        encoding="utf-8",
    )

    assert main(argv) == 2
    assert "reason" in json.loads(capsys.readouterr().out)["error"]



def test_build_financial_quality_command_rejects_naive_decision_time(tmp_path, capsys) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    decision_time = datetime(2020, 4, 1, tzinfo=UTC)
    facts_id = _stage_quality_facts_dataset(runtime.workspace.silver_root, decision_time=decision_time)

    argv = _quality_cli_args(scope_config, runtime, facts_id, tmp_path, decision="2026-09-24T12:00:00")
    assert main(argv) == 2
    assert "error" in json.loads(capsys.readouterr().out)



def test_build_financial_quality_command_rejects_bad_inputs(tmp_path, capsys) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    decision_time = datetime(2020, 4, 1, tzinfo=UTC)
    facts_id = _stage_quality_facts_dataset(runtime.workspace.silver_root, decision_time=decision_time)

    def run_with(quarantine_content, manual_content, decision="2020-04-01T00:00:00+00:00"):
        argv = [
            "build-financial-quality",
            "--scope-config", str(scope_config),
            "--data-root", str(tmp_path / "data"),
            "--facts-dataset-id", facts_id,
            "--decision-time", decision,
        ]
        if quarantine_content is not None:
            quarantine_path = tmp_path / "q.json"
            quarantine_path.write_text(quarantine_content, encoding="utf-8")
            argv += ["--quarantine-file", str(quarantine_path)]
        if manual_content is not None:
            manual_path = tmp_path / "m.json"
            manual_path.write_text(manual_content, encoding="utf-8")
            argv += ["--unresolved-events-file", str(manual_path)]
        assert main(argv) == 2
        return json.loads(capsys.readouterr().out)

    assert "error" in run_with(None, None, decision="not-a-date")
    missing = tmp_path / "does-not-exist.json"
    argv = [
        "build-financial-quality",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--facts-dataset-id", facts_id,
        "--quarantine-file", str(missing),
        "--decision-time", "2020-04-01T00:00:00+00:00",
    ]
    assert main(argv) == 2
    assert "error" in json.loads(capsys.readouterr().out)
    assert "error" in run_with('{"not": "a list"}', None)
    assert "error" in run_with('[1]', None)

    def manual_with(**overrides):
        entry = {
            "company_id": "000660",
            "fiscal_period": "2019Q4",
            "filing_id": "M1",
            "published_at": "2020-03-30T00:00:00+00:00",
            "available_at": "2020-03-31T00:00:00+00:00",
            "reason": "missing_source_value",
        }
        entry.update(overrides)
        return json.dumps([entry])

    assert "error" in run_with("[]", manual_with(published_at="not-a-date"))
    assert "error" in run_with("[]", manual_with(available_at="2020-03-31T00:00:00"))
    no_published = json.loads(manual_with())
    del no_published[0]["published_at"]
    assert "error" in run_with("[]", json.dumps(no_published))
    no_company = json.loads(manual_with())
    del no_company[0]["company_id"]
    assert "error" in run_with("[]", json.dumps(no_company))
    no_filing = json.loads(manual_with())
    del no_filing[0]["filing_id"]
    assert "error" in run_with("[]", json.dumps(no_filing))



def test_quality_event_timestamp_parses_datetime_objects() -> None:
    from datetime import UTC, datetime, timedelta

    from src.data.cli import _parse_quality_decision_time, _parse_quality_event_timestamp
    from src.core.pit import PITDataError
    import pytest

    moment = datetime(2020, 3, 31, tzinfo=UTC)
    assert _parse_quality_event_timestamp(moment, label="available_at") == moment
    assert _parse_quality_decision_time("2020-04-01T00:00:00+00:00") == moment + timedelta(days=1)
    with pytest.raises(PITDataError):
        _parse_quality_decision_time("2020-04-01T00:00:00")



def test_cli_normalize_scoped_and_unscoped_registration_paths(tmp_path, monkeypatch, capsys) -> None:
    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    import src.data.cli as cli_module

    assert cli_module.main(["normalize-dart-facts", "--decision-time", "2024-01-01T00:00:00"]) == 2
    assert "requires explicit roots" in json.loads(capsys.readouterr().out)["error"]
    assert cli_module.main([
        "normalize-dart-facts", "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--decision-time", "2024-01-01T00:00:00",
    ]) == 2
    assert "must be supplied together" in json.loads(capsys.readouterr().out)["error"]

    runtime = _dataset_runtime(tmp_path)
    scoped_published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity("financial_facts", DatasetLayer.SILVER, "fixture-v1", {}, {"case": "scoped"}),
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )
    unscoped_root = tmp_path / "unscoped"
    unscoped_published = publish_dataset(
        layer_root=unscoped_root / "silver" / "scope",
        identity=DatasetIdentity("financial_facts", DatasetLayer.SILVER, "fixture-v1", {}, {"case": "unscoped"}),
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )

    def _fake_normalize(**kwargs):
        selected = scoped_published if kwargs["silver_root"] == runtime.workspace.silver_root else unscoped_published
        return {
            "output_hash": "a" * 64,
            "report_hash": "b" * 64,
            "row_count": 1,
            "quarantined_filings": 0,
            "quarantine_path": str(tmp_path / "quarantine.json"),
            "dataset_path": str(selected.path),
        }

    monkeypatch.setattr("src.data.incremental_normalization.normalize_dart_facts", _fake_normalize)
    scoped_args = [
        "normalize-dart-facts", "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(runtime.workspace.root), "--decision-time", "2024-01-01T00:00:00+00:00",
    ]
    assert cli_module.main(scoped_args) == 0
    assert json.loads(capsys.readouterr().out)["dataset_id"] == scoped_published.dataset_id
    unscoped_args = [
        "normalize-dart-facts", "--bronze-root", str(tmp_path / "bronze"),
        "--silver-root", str(unscoped_root / "silver" / "scope"),
        "--artifact-root", str(tmp_path / "artifacts"),
        "--decision-time", "2024-01-01T00:00:00+00:00",
    ]
    assert cli_module.main(unscoped_args) == 0
    assert json.loads(capsys.readouterr().out)["dataset_id"] == unscoped_published.dataset_id
