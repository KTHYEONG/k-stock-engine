import json
import sys
from pathlib import Path
from typing import ClassVar

from src.data.cli import _parse_args


def _publish_cli_fixture(layer_root: Path, kind: str, partitions: dict[str, object], *, layer: str = "silver"):
    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    frames = {path: value for path, value in partitions.items() if isinstance(value, pl.DataFrame)}
    identity = DatasetIdentity(
        kind=kind,
        layer=DatasetLayer.GOLD if layer == "gold" else DatasetLayer.SILVER,
        policy_version=f"{kind}-fixture-v1",
        inputs={},
        params={},
    )
    return publish_dataset(
        layer_root=layer_root,
        identity=identity,
        partitions=frames,
    ).path




























def _register_cli_current(runtime, kind: str, dataset_path: Path) -> None:
    from src.data.dataset_registry import DatasetRegistry

    DatasetRegistry(runtime.workspace.state_root).register(kind, dataset_path.name)


def _publish_cli_session_dataset(layer_root: Path, kind: str, rows_by_session: dict[object, object]):
    return _publish_cli_fixture(
        layer_root,
        kind,
        {
            f"session={day.isoformat()}/part.parquet": rows
            for day, rows in rows_by_session.items()
        },
    )


def test_ordinary_universe_price_audit_command_emits_report_and_handles_failure(
    tmp_path, monkeypatch, capsys
) -> None:
    import src.data.ordinary_universe_price_audit as audit_mod
    from src.data.cli import main
    from src.data.ordinary_universe_price_audit import OrdinaryUniversePriceAudit
    from src.data.schemas import PITDataError

    monkeypatch.setattr(
        audit_mod,
        "audit_ordinary_universe_price_availability",
        lambda **_kwargs: OrdinaryUniversePriceAudit(
            dataset_id="audit-1",
            universe_dataset_id="universe-1",
            sessions=1,
            universe_rows=1,
            eligible_rows=1,
            price_rows=1,
            tradable_rows=1,
            missing_price_rows=0,
            invalid_price_rows=0,
            zero_volume_rows=0,
            report_hash="r" * 64,
        ),
    )
    arguments = [
        "audit-ordinary-universe-prices",
        "--universe-root",
        str(tmp_path / "universe"),
        "--bronze-root",
        str(tmp_path / "bronze"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    assert main(arguments) == 0
    assert '"dataset_id": "audit-1"' in capsys.readouterr().out

    def fail(**_kwargs):
        raise PITDataError("audit fixture failure")

    monkeypatch.setattr(audit_mod, "audit_ordinary_universe_price_availability", fail)
    assert main(arguments) == 1








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

    assert cli_module.main() == 1
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
    page_dir = runtime.workspace.bronze_root / "investor_flow" / hashlib.sha256(raw).hexdigest()
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "payload.json").write_bytes(raw)
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
    ReceiptCatalog(runtime.workspace.bronze_root / "catalog").publish([
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
    ])
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


def _write_cli_gap_dataset(directory, frame) -> Path:
    name = Path(directory).name
    if name.startswith("market_panel"):
        kind = "market_panel"
        layer_root = Path(directory).parent
    else:
        kind = "investor_flow_ls" if name.startswith("investor_flow_cli") else "investor_flow_kis_supplement"
        layer_root = Path(directory).parent
    return _publish_cli_fixture(
        layer_root,
        kind,
        {"year=2024/part.parquet": frame},
        layer="gold" if kind == "market_panel" else "silver",
    )


def _write_cli_gap_inputs(runtime, *, ls_cells, panel_cells) -> tuple:
    import polars as pl

    panel = runtime.workspace.gold_root / "market_panel_cli"
    ls_flow = runtime.workspace.silver_root / "investor_flow_cli"
    panel = _write_cli_gap_dataset(
        panel,
        pl.DataFrame(
            {
                "session": [session for session, _ in panel_cells],
                "instrument_id": [f"KRX:{ticker}" for _, ticker in panel_cells],
                "ticker": [ticker for _, ticker in panel_cells],
                "eligible": [True] * len(panel_cells),
                "price_state": ["tradable"] * len(panel_cells),
            },
            schema={
                "session": pl.Date,
                "instrument_id": pl.String,
                "ticker": pl.String,
                "eligible": pl.Boolean,
                "price_state": pl.String,
            },
        ),
    )
    ls_flow = _write_cli_gap_dataset(
        ls_flow,
        pl.DataFrame(
            {"session": [session for session, _ in ls_cells], "ticker": [ticker for _, ticker in ls_cells]},
            schema={"session": pl.Date, "ticker": pl.String},
        ),
    )
    return panel, ls_flow


def _write_cli_kis_page(bronze_root, symbol, anchor, rows) -> None:
    import hashlib
    import json

    payload = {
        "provider": "KIS",
        "endpoint": "investor-trade-by-stock-daily",
        "symbol": symbol,
        "anchor": anchor.isoformat(),
        "query": {"symbol": symbol, "anchor": anchor.isoformat()},
        "rows": rows,
        "records": [],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    page_dir = bronze_root / "investor_flow" / hashlib.sha256(raw).hexdigest()
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "payload.json").write_bytes(raw)


def test_backfill_kis_investor_flow_gap_reports_attempted_symbols(tmp_path, monkeypatch, capsys) -> None:
    import json
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    s1, s2 = date(2024, 1, 2), date(2024, 1, 3)
    panel, ls_flow = _write_cli_gap_inputs(
        runtime,
        ls_cells=[(s1, "000001"), (s1, "000002")],
        panel_cells=[(s1, "000001"), (s2, "000001"), (s1, "000002"), (s2, "000002")],
    )
    seen: dict[str, object] = {}

    class StubCollector:
        def __init__(self, symbols) -> None:
            seen["symbols"] = tuple(symbols)

        def fetch_investor_flow(self, start, end, *, bronze_root=None, retrieved_at=None, symbols=None):
            seen.setdefault("calls", []).append((start, end, symbols))
            yield {"provider": "KIS", "symbol": symbols[0], "anchor": end.isoformat(), "records": []}

    monkeypatch.setattr("src.integrations.kis.investor_flow.KisInvestorFlowCollector", StubCollector)
    assert main([
        "backfill-kis-investor-flow-gap",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--market-panel-dataset-id", panel.name,
        "--ls-flow-dataset-id", ls_flow.name,
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["symbols_attempted"] == 2
    assert emitted["target_cells"] == 2
    assert seen["symbols"] == ("000001", "000002")


def test_backfill_kis_investor_flow_gap_with_empty_gap(tmp_path, monkeypatch, capsys) -> None:
    import json
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    s1 = date(2024, 1, 2)
    panel, ls_flow = _write_cli_gap_inputs(
        runtime, ls_cells=[(s1, "000001")], panel_cells=[(s1, "000001")]
    )

    def _forbidden(symbols):
        raise AssertionError("collector must not be constructed for an empty gap")

    monkeypatch.setattr("src.integrations.kis.investor_flow.KisInvestorFlowCollector", _forbidden)
    assert main([
        "backfill-kis-investor-flow-gap",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--market-panel-dataset-id", panel.name,
        "--ls-flow-dataset-id", ls_flow.name,
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {"symbols_attempted": 0, "target_cells": 0}


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
        "--market-panel-dataset-id", panel.name,
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


def _write_cli_universe_dataset(silver_root, rows) -> object:
    import polars as pl

    dataset = silver_root / "ordinary_universe_cli"
    part_dir = dataset / "session=2024-01-02"
    part_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={"ticker": pl.String, "eligible": pl.Boolean}).write_parquet(
        part_dir / "part.parquet"
    )
    return dataset


class _StubIndustryCollector:
    calls: ClassVar[list] = []

    def __init__(self, symbols, client=None) -> None:
        self.symbols = tuple(symbols)

    def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
        type(self).calls.append(self.symbols)
        return [{"provider": "KIS", "symbol": symbol} for symbol in self.symbols]


def test_collect_industry_classification_from_symbols_file(tmp_path, monkeypatch, capsys) -> None:
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("005930\n 000660\n005930\n", encoding="utf-8")
    _StubIndustryCollector.calls = []
    monkeypatch.setattr(
        "src.integrations.kis.industry.KisIndustryCollector", _StubIndustryCollector
    )
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--symbols-from", str(symbols_file),
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {"symbols_requested": 2, "pages_collected": 2, "skipped_count": 0, "skipped": {}}
    assert _StubIndustryCollector.calls == [("005930",), ("000660",)]


def test_collect_industry_classification_defaults_to_universe(tmp_path, monkeypatch, capsys) -> None:
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_cli_universe_dataset(
        runtime.workspace.silver_root,
        [
            {"ticker": "005930", "eligible": True},
            {"ticker": "000001", "eligible": False},
            {"ticker": "000660", "eligible": True},
        ],
    )
    _StubIndustryCollector.calls = []
    monkeypatch.setattr(
        "src.integrations.kis.industry.KisIndustryCollector", _StubIndustryCollector
    )
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {"symbols_requested": 2, "pages_collected": 2, "skipped_count": 0, "skipped": {}}
    assert _StubIndustryCollector.calls == [("000660",), ("005930",)]


def test_collect_industry_classification_rejects_empty_symbols_file(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("  \n", encoding="utf-8")
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--symbols-from", str(symbols_file),
    ]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_collect_industry_classification_requires_universe(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_collect_industry_classification_requires_universe_partitions(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    (runtime.workspace.silver_root / "ordinary_universe_cli").mkdir(parents=True)
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_collect_industry_classification_requires_eligible_tickers(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_cli_universe_dataset(
        runtime.workspace.silver_root, [{"ticker": "000001", "eligible": False}]
    )
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_build_industry_classification_silver_command_emits_dataset(tmp_path, capsys) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.bronze import BronzeStore
    from src.data.cli import main
    from src.data.runtime import load_data_runtime
    from src.data.schemas import EvidenceKind

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
    BronzeStore(runtime.workspace.bronze_root).import_bytes(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        kind=EvidenceKind.INDUSTRY,
        retrieved_at=collected_at,
        source_label="KIS:inquire-price:005930:2024-01-03",
    )
    assert main([
        "build-industry-classification-silver",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("industry_")
    assert emitted["rows"] == 1


class _FailingIndustryCollector:
    calls: ClassVar[list] = []
    failing: ClassVar[str] = "000660"

    def __init__(self, symbols, client=None) -> None:
        self.symbols = tuple(symbols)

    def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
        from src.data.schemas import PITDataError

        type(self).calls.append(self.symbols)
        symbol = self.symbols[0]
        if symbol == type(self).failing:
            raise PITDataError(f"KIS industry classification missing bstp_kor_isnm for {symbol}")
        return [{"provider": "KIS", "symbol": symbol}]


class _FailingStockCollector:
    calls: ClassVar[list] = []
    failing: ClassVar[str] = "000660"

    def __init__(self, symbols, client=None) -> None:
        self.symbols = tuple(symbols)

    def fetch_stock_classification(self, *, bronze_root, retrieved_at=None):
        from src.data.schemas import PITDataError

        type(self).calls.append(self.symbols)
        symbol = self.symbols[0]
        if symbol == type(self).failing:
            raise PITDataError(f"KIS stock classification missing valid std_idst_clsf_cd for {symbol}")
        return [{"provider": "KIS", "symbol": symbol}]


def _run_classification_command(tmp_path, monkeypatch, capsys, command, stub_attr, stub_cls):
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("005930\n000660\n035420\n", encoding="utf-8")
    stub_cls.calls = []
    monkeypatch.setattr(stub_attr, stub_cls)
    code = main([
        command,
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--symbols-from", str(symbols_file),
        "--pace-seconds", "0",
    ])
    return code, json.loads(capsys.readouterr().out)


def test_collect_industry_classification_isolates_failing_ticker(tmp_path, monkeypatch, capsys) -> None:
    """Isolation: one unclassifiable ticker must not discard the rest of the run."""
    code, emitted = _run_classification_command(
        tmp_path, monkeypatch, capsys,
        "collect-industry-classification",
        "src.integrations.kis.industry.KisIndustryCollector",
        _FailingIndustryCollector,
    )
    assert code == 0
    assert emitted["symbols_requested"] == 3
    assert emitted["pages_collected"] == 2
    assert emitted["skipped_count"] == 1
    assert list(emitted["skipped"]) == ["000660"]
    assert "000660" in emitted["skipped"]["000660"]
    assert _FailingIndustryCollector.calls == [("005930",), ("000660",), ("035420",)]


def test_collect_stock_classification_isolates_failing_ticker(tmp_path, monkeypatch, capsys) -> None:
    """Stock classification mirrors the per-ticker isolation."""
    code, emitted = _run_classification_command(
        tmp_path, monkeypatch, capsys,
        "collect-stock-classification",
        "src.integrations.kis.industry.KisStockClassificationCollector",
        _FailingStockCollector,
    )
    assert code == 0
    assert emitted["symbols_requested"] == 3
    assert emitted["pages_collected"] == 2
    assert emitted["skipped_count"] == 1
    assert list(emitted["skipped"]) == ["000660"]
    assert _FailingStockCollector.calls == [("005930",), ("000660",), ("035420",)]


def test_collect_classification_with_no_pages_fails_closed(tmp_path, monkeypatch, capsys) -> None:
    """A run that collected nothing must not look like success."""
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    for command, stub_attr, stub_cls in (
        ("collect-industry-classification", "src.integrations.kis.industry.KisIndustryCollector", _FailingIndustryCollector),
        ("collect-stock-classification", "src.integrations.kis.industry.KisStockClassificationCollector", _FailingStockCollector),
    ):
        symbols_file = tmp_path / f"symbols-{command}.txt"
        symbols_file.write_text("000660\n", encoding="utf-8")
        stub_cls.calls = []
        monkeypatch.setattr(stub_attr, stub_cls)
        assert main([
            command,
            "--scope-config", str(scope_config),
            "--data-root", str(tmp_path / "data"),
            "--symbols-from", str(symbols_file),
            "--pace-seconds", "0",
        ]) == 1
        assert "error" in json.loads(capsys.readouterr().out)


def test_collect_classification_non_pit_exception_propagates(tmp_path, monkeypatch) -> None:
    """Non-PIT exceptions are not swallowed as skips."""
    import pytest

    from src.data.cli import main

    class _BoomCollector:
        def __init__(self, symbols, client=None) -> None:
            self.symbols = tuple(symbols)

        def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
            if self.symbols[0] == "000660":
                raise RuntimeError("boom")
            return [{"provider": "KIS", "symbol": self.symbols[0]}]

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("005930\n000660\n", encoding="utf-8")
    monkeypatch.setattr("src.integrations.kis.industry.KisIndustryCollector", _BoomCollector)
    with pytest.raises(RuntimeError, match="boom"):
        main([
            "collect-industry-classification",
            "--scope-config", str(scope_config),
            "--data-root", str(tmp_path / "data"),
            "--symbols-from", str(symbols_file),
            "--pace-seconds", "0",
        ])


def test_collect_stock_classification_defaults_to_universe(tmp_path, monkeypatch, capsys) -> None:
    """Stock classification defaults to the ordinary-universe tickers."""
    import json

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_cli_universe_dataset(
        runtime.workspace.silver_root,
        [
            {"ticker": "005930", "eligible": True},
            {"ticker": "000001", "eligible": False},
            {"ticker": "000660", "eligible": True},
        ],
    )
    _StubIndustryCollector.calls = []

    class _StubStockCollector:
        calls: ClassVar[list] = []

        def __init__(self, symbols, client=None) -> None:
            self.symbols = tuple(symbols)

        def fetch_stock_classification(self, *, bronze_root, retrieved_at=None):
            type(self).calls.append(self.symbols)
            return [{"provider": "KIS", "symbol": symbol} for symbol in self.symbols]

    monkeypatch.setattr(
        "src.integrations.kis.industry.KisStockClassificationCollector", _StubStockCollector
    )
    assert main([
        "collect-stock-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--pace-seconds", "0",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["symbols_requested"] == 2
    assert emitted["pages_collected"] == 2
    assert _StubStockCollector.calls == [("000660",), ("005930",)]


def test_collect_classification_logs_progress_every_hundred_symbols(tmp_path, caplog) -> None:
    """Progress logging fires on each 100-symbol boundary."""
    import logging

    from src.data.cli import _collect_classification_with_isolation

    class _StubCollector:
        def __init__(self, symbols, client=None) -> None:
            self.symbols = tuple(symbols)

        def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
            return [{"provider": "KIS", "symbol": self.symbols[0]}]

    symbols = tuple(f"{index:06d}" for index in range(100))
    with caplog.at_level(logging.INFO, logger="src.data.cli"):
        result = _collect_classification_with_isolation(
            stage="collect-industry-classification",
            collector_cls=_StubCollector,
            fetch_attr="fetch_industry_classification",
            bronze_root=tmp_path / "bronze",
            symbols=symbols,
            pace_seconds=0,
        )
    assert result["pages_collected"] == 100
    assert any("stage=collect-industry-classification" in message for message in caplog.messages)


def _stage_quality_facts_dataset(silver_root, *, decision_time):
    from datetime import UTC, datetime

    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    available = datetime(2019, 11, 14, 0, 0, tzinfo=UTC)
    rows = [
        {
            "company_id": "005930",
            "fiscal_period": "2018Q4",
            "filing_id": "F8",
            "fact": fact,
            "published_at": available,
            "available_at": available,
            "value": 100.0,
            "unit": "KRW",
            "consolidated": True,
            "restatement_id": "r0",
            "source_hash": "s",
            "source_kind": "opendart_standard",
            "mapping_version": "v1",
            "raw_document_hash": None,
            "ticker": "005930",
            "dart_corp_code": "00126380",
        }
        for fact in (
            "sales", "gross_profit", "operating_profit", "net_income",
            "assets", "equity", "operating_cash_flow",
        )
    ]
    return publish_dataset(
        layer_root=silver_root,
        identity=DatasetIdentity(
            kind="financial_facts",
            layer=DatasetLayer.SILVER,
            policy_version="dart-incremental-v1",
            inputs={},
            params={"decision_time": decision_time},
        ),
        partitions={"part-00000.parquet": pl.DataFrame(rows)},
    ).dataset_id


def _quality_cli_args(scope_config, runtime, facts_id, tmp_path, decision="2020-04-01T00:00:00+00:00", extra=()):
    import json

    quarantine = [
        {
            "company_id": "005930",
            "fiscal_period": "2019Q4",
            "filing_id": "F9",
            "published_at": "2020-03-30T00:00:00+00:00",
            "available_at": "2020-03-31T00:00:00+00:00",
        }
    ]
    manual = [
        {
            "company_id": "000660",
            "fiscal_period": "2019Q4",
            "filing_id": "M1",
            "published_at": "2020-03-30T00:00:00+00:00",
            "available_at": "2020-03-31T00:00:00+00:00",
            "reason": "missing_source_value",
        }
    ]
    quarantine_path = tmp_path / "quarantine.json"
    quarantine_path.write_text(json.dumps(quarantine), encoding="utf-8")
    manual_path = tmp_path / "manual.json"
    manual_path.write_text(json.dumps(manual), encoding="utf-8")
    return [
        "build-financial-quality",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--facts-dataset-id", facts_id,
        "--quarantine-file", str(quarantine_path),
        "--unresolved-events-file", str(manual_path),
        "--decision-time", decision,
        *extra,
    ]


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

    assert main(_quality_cli_args(scope_config, runtime, "0" * 64, tmp_path)) == 1
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

    assert main(argv) == 1
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
    assert main(argv) == 1
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
        assert main(argv) == 1
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
    assert main(argv) == 1
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
    from src.data.schemas import PITDataError
    import pytest

    moment = datetime(2020, 3, 31, tzinfo=UTC)
    assert _parse_quality_event_timestamp(moment, label="available_at") == moment
    assert _parse_quality_decision_time("2020-04-01T00:00:00+00:00") == moment + timedelta(days=1)
    with pytest.raises(PITDataError):
        _parse_quality_decision_time("2020-04-01T00:00:00")



_LIVE_SUBCOMMANDS = frozenset(
    {
        "scope-info",
        "init-workspace",
        "collect-scoped",
        "plan-scoped",
        "resume-scoped",
        "collect-dart-disclosures-scoped",
        "collect-dart-facts-scoped",
        "collect-missing-dart-facts-scoped",
        "collect-industry-classification",
        "collect-stock-classification",
        "backfill-kis-investor-flow-gap",
        "normalize-dart-facts",
        "build-financial-quality",
        "build-ordinary-universe",
        "build-dividend-events",
        "build-investor-flow-silver",
        "build-daily-market-silver",
        "build-investor-flow-kis-supplement",
        "build-investor-flow-union",
        "build-industry-classification-silver",
        "build-market-panel",
        "build-reference-benchmarks",
        "verify-datasets",
        "prune-datasets",
        "audit-ordinary-universe-prices",
    }
)


def _capture_subparsers(monkeypatch, capsys):
    import argparse

    parsers: dict[str, argparse.ArgumentParser] = {}
    real_add_parser = argparse._SubParsersAction.add_parser

    def _capture(self, name, **kwargs):
        parser = real_add_parser(self, name, **kwargs)
        parsers[name] = parser
        return parser

    monkeypatch.setattr(argparse._SubParsersAction, "add_parser", _capture)
    import pytest

    with pytest.raises(SystemExit):
        _parse_args(["--help"])
    capsys.readouterr()
    return parsers


def test_cli_registers_only_live_subcommands(monkeypatch, capsys) -> None:
    """Only live subcommands are registered."""
    parsers = _capture_subparsers(monkeypatch, capsys)
    assert set(parsers) == set(_LIVE_SUBCOMMANDS)


def test_cli_has_no_legacy_path_defaults(monkeypatch, capsys) -> None:
    """No legacy path defaults."""
    parsers = _capture_subparsers(monkeypatch, capsys)
    assert parsers
    for name, parser in parsers.items():
        for action in parser._actions:
            default = action.default
            for banned in ("stocks", "data/evidence", "data/artifacts"):
                assert banned not in str(default), f"{name}.{action.dest}={default!r}"


def _dataset_runtime(tmp_path):
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"),
        data_root=tmp_path / "data",
    )
    runtime.workspace.initialize()
    return runtime


def _publish_cli_dataset(runtime, kind, *, inputs=None, params=None):
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    import polars as pl

    identity = DatasetIdentity(
        kind=kind,
        layer=DatasetLayer.SILVER,
        policy_version="cli-fixture-v1",
        inputs=inputs or {},
        params=params or {},
    )
    return publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=identity,
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )


def _dataset_cli_args(command, runtime, *extra):
    return [
        command,
        "--scope-config",
        "config/research/kr_swing_2019_v1.toml",
        "--data-root",
        str(runtime.workspace.root),
        *extra,
    ]


def test_verify_datasets_exits_nonzero_on_tampering(tmp_path, capsys, caplog) -> None:
    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    caplog.set_level("INFO")
    runtime = _dataset_runtime(tmp_path)
    published = _publish_cli_dataset(runtime, "daily_market")
    DatasetRegistry(runtime.workspace.state_root).register("daily_market", published.dataset_id)
    partition = published.path / "part.parquet"
    partition.write_bytes(partition.read_bytes() + b"tampered")

    exit_code = main(_dataset_cli_args("verify-datasets", runtime))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert exit_code == 1
    assert any(line.get("dataset_id") == published.dataset_id and line["status"] == "failed" for line in lines)
    assert lines[-1]["type"] == "summary"
    assert lines[-1]["failed"] == 1
    assert any("[DATA] command=verify_datasets" in message for message in caplog.messages)


def test_verify_datasets_warns_but_does_not_fail_on_stale_lineage(tmp_path, capsys) -> None:
    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    runtime = _dataset_runtime(tmp_path)
    old_input = _publish_cli_dataset(runtime, "ordinary_universe", params={"generation": 1})
    current_input = _publish_cli_dataset(runtime, "ordinary_universe", params={"generation": 2})
    dependent = _publish_cli_dataset(
        runtime,
        "daily_market",
        inputs={"universe": old_input.dataset_id},
    )
    registry = DatasetRegistry(runtime.workspace.state_root)
    registry.register("ordinary_universe", current_input.dataset_id)
    registry.register("daily_market", dependent.dataset_id)

    exit_code = main(_dataset_cli_args("verify-datasets", runtime))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    dependent_line = next(line for line in lines if line.get("dataset_id") == dependent.dataset_id)

    assert exit_code == 0
    assert dependent_line["stale_inputs"] == [
        {
            "role": "universe",
            "input_kind": "ordinary_universe",
            "input_id": old_input.dataset_id,
            "registered_id": current_input.dataset_id,
        }
    ]
    assert lines[-1]["stale"] == 1


def test_verify_datasets_all_includes_unregistered_datasets(tmp_path, capsys) -> None:
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    anchor = _publish_cli_dataset(runtime, "anchor")
    from src.data.dataset_registry import DatasetRegistry
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    published = _publish_cli_dataset(runtime, "unregistered")

    exit_code = main(_dataset_cli_args("verify-datasets", runtime, "--all"))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert exit_code == 0
    assert any(line.get("dataset_id") == published.dataset_id for line in lines)


def test_prune_datasets_keeps_transitive_lineage_and_apply_removes_only_orphan(tmp_path, capsys) -> None:
    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    runtime = _dataset_runtime(tmp_path)
    root_input = _publish_cli_dataset(runtime, "source")
    middle = _publish_cli_dataset(runtime, "middle", inputs={"source": root_input.dataset_id})
    orphan = _publish_cli_dataset(runtime, "orphan")
    DatasetRegistry(runtime.workspace.state_root).register("middle", middle.dataset_id)

    plan_exit = main(_dataset_cli_args("prune-datasets", runtime))
    plan_lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    listed = {line["dataset_id"] for line in plan_lines if line.get("status") == "prunable"}

    assert plan_exit == 0
    assert listed == {orphan.dataset_id}
    assert (runtime.workspace.silver_root / root_input.dataset_id).is_dir()
    assert (runtime.workspace.silver_root / middle.dataset_id).is_dir()

    apply_exit = main(_dataset_cli_args("prune-datasets", runtime, "--apply"))

    assert apply_exit == 0
    assert not (runtime.workspace.silver_root / orphan.dataset_id).exists()
    assert (runtime.workspace.silver_root / root_input.dataset_id).is_dir()
    assert (runtime.workspace.silver_root / middle.dataset_id).is_dir()


def test_prune_datasets_never_deletes_unreadable_manifest(tmp_path, capsys) -> None:
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    state_before = sorted(path.name for path in runtime.workspace.state_root.iterdir())
    unreadable = runtime.workspace.silver_root / "broken_0123456789abcdef"
    unreadable.mkdir()

    exit_code = main(_dataset_cli_args("prune-datasets", runtime, "--apply"))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert exit_code == 1
    assert any(line.get("dataset_id") == unreadable.name and line["status"] == "unreadable_manifest" for line in lines)
    assert unreadable.is_dir()
    assert sorted(path.name for path in runtime.workspace.state_root.iterdir()) == state_before


def test_verify_datasets_accepts_retired_lineage_and_reports_missing_registration(tmp_path, capsys) -> None:
    import json as json_module

    from src.data.cli import main
    from src.data.dataset_registry import REGISTRY_NAME
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    import polars as pl

    runtime = _dataset_runtime(tmp_path)
    retired_id = "ordinary_universe_0123456789abcdef"
    identity = DatasetIdentity(
        kind="daily_market",
        layer=DatasetLayer.SILVER,
        policy_version="cli-fixture-v1",
        inputs={"universe": retired_id},
        params={},
    )
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=identity,
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )
    registry_path = runtime.workspace.state_root / REGISTRY_NAME
    registry_path.write_text(
        json_module.dumps(
            {
                "current": {"daily_market": published.dataset_id},
                "retired": {retired_id: "superseded upstream generation"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert main(_dataset_cli_args("verify-datasets", runtime)) == 0
    capsys.readouterr()
    registry_path.write_text(
        json_module.dumps(
            {
                "current": {"daily_market": "daily_market_1111111111111111"},
                "retired": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert main(_dataset_cli_args("verify-datasets", runtime)) == 1
    lines = [json_module.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any("missing or ambiguous" in failure for line in lines for failure in line.get("failures", []))


def test_dataset_commands_ignore_missing_layer_root_and_bronze_lineage(tmp_path, capsys) -> None:
    import shutil as shutil_module

    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    runtime = _dataset_runtime(tmp_path)
    source = _publish_cli_dataset(runtime, "source")
    dependent = _publish_cli_dataset(
        runtime,
        "dependent",
        inputs={"bronze": f"bronze:{'b' * 64}", "source": source.dataset_id},
    )
    DatasetRegistry(runtime.workspace.state_root).register("dependent", dependent.dataset_id)
    DatasetRegistry(runtime.workspace.state_root).register("source", source.dataset_id)
    shutil_module.rmtree(runtime.workspace.gold_root)

    assert main(_dataset_cli_args("verify-datasets", runtime)) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    dependent_line = next(line for line in lines if line["dataset_id"] == dependent.dataset_id)
    assert dependent_line["stale_inputs"] == []


def test_verify_datasets_reports_configuration_and_manifest_read_failures(tmp_path, capsys) -> None:
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    published = _publish_cli_dataset(runtime, "daily_market")
    (published.path / "manifest.json").write_text("not-json", encoding="utf-8")
    args = _dataset_cli_args("verify-datasets", runtime)
    registry_path = runtime.workspace.state_root / "datasets.json"
    registry_path.write_text(
        json.dumps(
            {
                "current": {"daily_market": published.dataset_id},
                "retired": {},
            }
        ),
        encoding="utf-8",
    )

    assert main(args) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["status"] == "failed"
    assert lines[0]["stale_inputs"] == []

    invalid_args = [
        "verify-datasets",
        "--scope-config",
        "missing-scope.toml",
        "--data-root",
        str(runtime.workspace.root),
    ]
    assert main(invalid_args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["type"] == "summary"
    assert summary["failed"] == 1


def test_prune_datasets_fails_closed_for_registered_unreadable_dataset(tmp_path, capsys) -> None:
    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    runtime = _dataset_runtime(tmp_path)
    registered = _publish_cli_dataset(runtime, "registered")
    orphan = _publish_cli_dataset(
        runtime,
        "orphan",
        inputs={"bronze": f"bronze:{'c' * 64}"},
    )
    DatasetRegistry(runtime.workspace.state_root).register("registered", registered.dataset_id)
    (registered.path / "manifest.json").write_text("not-json", encoding="utf-8")

    assert main(_dataset_cli_args("prune-datasets", runtime)) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["type"] == "summary"
    assert "current dataset manifest is invalid" in summary["error"]
    assert orphan.path.is_dir()


def test_prune_datasets_reports_configuration_failure(tmp_path, capsys) -> None:
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    args = [
        "prune-datasets",
        "--scope-config",
        "missing-scope.toml",
        "--data-root",
        str(runtime.workspace.root),
    ]

    assert main(args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["type"] == "summary"
    assert summary["failed"] == 1


def test_prune_apply_rechecks_candidate_set_before_deletion(tmp_path, capsys, monkeypatch) -> None:
    import src.data.cli as cli_module
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    orphan = _publish_cli_dataset(runtime, "orphan")
    original_plan = cli_module._prune_plan
    calls = 0

    def _no_longer_prunable(current_runtime, registry):
        nonlocal calls
        calls += 1
        return original_plan(current_runtime, registry) if calls == 1 else ([], [])

    monkeypatch.setattr(cli_module, "_prune_plan", _no_longer_prunable)
    assert main(_dataset_cli_args("prune-datasets", runtime, "--apply")) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["candidates"] == 1
    assert summary["deleted"] == 0
    assert orphan.path.is_dir()


def test_prune_apply_never_deletes_symlink_replacement(tmp_path, capsys, monkeypatch) -> None:
    import src.data.cli as cli_module
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    orphan = _publish_cli_dataset(runtime, "orphan")
    outside = tmp_path / "outside"
    outside.mkdir()
    original_plan = cli_module._prune_plan
    calls = 0

    def _replace_with_symlink(current_runtime, registry):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_plan(current_runtime, registry)
        cli_module.shutil.rmtree(orphan.path)
        orphan.path.symlink_to(outside, target_is_directory=True)
        return [orphan.path], []

    monkeypatch.setattr(cli_module, "_prune_plan", _replace_with_symlink)
    assert main(_dataset_cli_args("prune-datasets", runtime, "--apply")) == 0
    assert orphan.path.is_symlink()
    assert outside.is_dir()


def test_prune_apply_reports_delete_failure(tmp_path, capsys, monkeypatch, caplog) -> None:
    import src.data.cli as cli_module
    from src.data.cli import main

    caplog.set_level("INFO")
    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    orphan = _publish_cli_dataset(runtime, "orphan")
    monkeypatch.setattr(
        cli_module.shutil,
        "rmtree",
        lambda path, **_kwargs: (_ for _ in ()).throw(OSError(f"cannot remove {path}")),
    )

    assert main(_dataset_cli_args("prune-datasets", runtime, "--apply")) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(line.get("dataset_id") == orphan.dataset_id and line["status"] == "delete_failed" for line in lines)
    assert orphan.path.is_dir()
    assert any("action=keep status=failed" in message for message in caplog.messages)


def test_dataset_cli_directory_manifest_and_current_validation_boundaries(tmp_path, monkeypatch, capsys) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.cli import _dataset_directories, _legacy_lineage, _legacy_prunable_manifest, _prune_plan, _validated_current_directories
    from src.data.dataset_registry import REGISTRY_NAME, DatasetRegistry
    import src.data.cli as cli_module

    runtime = _dataset_runtime(tmp_path)
    silver = runtime.workspace.silver_root
    (silver / ".hidden").mkdir()
    (silver / "not-a-directory").write_text("x", encoding="utf-8")
    nested = silver / "legacy-table" / "legacy_0123456789abcdef"
    nested.mkdir(parents=True)
    (nested / "dataset_manifest.json").write_text(
        json.dumps({"inputs": {"upstream": "ordinary_universe_0123456789abcdef"}}),
        encoding="utf-8",
    )
    (nested / "content_manifest.json").write_text(
        json.dumps({"partitions": [{"path": "part.parquet"}]}), encoding="utf-8"
    )
    assert nested in _dataset_directories(runtime)
    assert _legacy_prunable_manifest(nested) is not None

    malformed_root = tmp_path / "legacy-malformed"
    malformed = malformed_root / "legacy_0123456789abcdef"
    malformed.mkdir(parents=True)
    (malformed / "manifest.json").write_text("[]", encoding="utf-8")
    assert _legacy_prunable_manifest(malformed) is None
    (malformed / "manifest.json").write_text(json.dumps({"dataset_id": "wrong", "partitions": [{"path": "x"}]}), encoding="utf-8")
    assert _legacy_prunable_manifest(malformed) is None
    (malformed / "manifest.json").write_text(json.dumps({"dataset_id": malformed.name, "partitions": []}), encoding="utf-8")
    assert _legacy_prunable_manifest(malformed) is None
    (malformed / "manifest.json").write_text(json.dumps({"dataset_id": malformed.name, "partitions": [1]}), encoding="utf-8")
    assert _legacy_prunable_manifest(malformed) is None
    content_root = tmp_path / "legacy-content"
    content_dir = content_root / "legacy_0123456789abcdef"
    content_dir.mkdir(parents=True)
    (content_dir / "dataset_manifest.json").write_text(json.dumps({"inputs": {}}), encoding="utf-8")
    (content_dir / "content_manifest.json").write_text("[]", encoding="utf-8")
    assert _legacy_prunable_manifest(content_dir) is None
    (content_dir / "content_manifest.json").write_text(json.dumps({"partitions": [{"path": "part.parquet"}]}), encoding="utf-8")
    assert _legacy_prunable_manifest(content_dir) is not None
    missing_content = tmp_path / "legacy-missing-content" / "legacy_0123456789abcdef"
    missing_content.mkdir(parents=True)
    (missing_content / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    assert _legacy_prunable_manifest(missing_content) is None

    valid_id = "ordinary_universe_0123456789abcdef"
    other_id = "daily_market_1111111111111111"
    assert _legacy_lineage(
        {
            "inputs": {"a": valid_id, "b": f"bronze:{'a' * 64}"},
            "daily_market_dataset_id": other_id,
            "universe_dataset_id": "not-an-id",
        }
    ) == {valid_id, other_id}

    empty_registry = DatasetRegistry(runtime.workspace.state_root)
    with pytest.raises(PITDataError, match="no current datasets"):
        _validated_current_directories(runtime, empty_registry)
    assert cli_module.main(_dataset_cli_args("verify-datasets", runtime)) == 1
    assert json.loads(capsys.readouterr().out)["failed"] == 1

    missing_id = "daily_market_0123456789abcdef"
    (runtime.workspace.state_root / REGISTRY_NAME).write_text(
        json.dumps({"current": {"daily_market": missing_id}, "retired": {}}), encoding="utf-8"
    )
    with pytest.raises(PITDataError, match="missing or ambiguous"):
        _validated_current_directories(runtime, DatasetRegistry(runtime.workspace.state_root))

    mismatch = _publish_cli_dataset(runtime, "daily_market")
    (runtime.workspace.state_root / REGISTRY_NAME).write_text(
        json.dumps({"current": {"market_panel": mismatch.dataset_id}, "retired": {}}), encoding="utf-8"
    )
    with pytest.raises(PITDataError, match="kind mismatch"):
        _validated_current_directories(runtime, DatasetRegistry(runtime.workspace.state_root))

    class _SnapshotRegistry:
        def snapshot(self):
            return {"market_panel": mismatch.dataset_id}

        def retired(self):
            return {}

    with pytest.raises(PITDataError, match="kind mismatch"):
        _validated_current_directories(runtime, _SnapshotRegistry())  # type: ignore[arg-type]

    failed = _publish_cli_dataset(runtime, "daily_market", params={"case": "failed"})
    registry_path = runtime.workspace.state_root / REGISTRY_NAME
    registry_path.write_text(json.dumps({"current": {"daily_market": failed.dataset_id}, "retired": {}}), encoding="utf-8")
    (failed.path / "part.parquet").write_bytes(b"tampered")
    with pytest.raises(PITDataError, match="failed verification"):
        _validated_current_directories(runtime, DatasetRegistry(runtime.workspace.state_root))

    clean = _publish_cli_dataset(runtime, "anchor")
    registry_path.write_text(json.dumps({"current": {"anchor": clean.dataset_id}, "retired": {}}), encoding="utf-8")
    dependent = _publish_cli_dataset(runtime, "dependent", inputs={"bronze": f"bronze:{'b' * 64}"})
    _prune_plan(runtime, DatasetRegistry(runtime.workspace.state_root))
    assert dependent.path.is_dir()

    legacy_orphan = silver / "legacy-table" / "orphan_0123456789abcdef"
    legacy_orphan.mkdir()
    (legacy_orphan / "dataset_manifest.json").write_text(
        json.dumps({"inputs": {"source": "daily_market_1111111111111111"}}),
        encoding="utf-8",
    )
    (legacy_orphan / "content_manifest.json").write_text(
        json.dumps({"partitions": [{"path": "part.parquet"}]}), encoding="utf-8"
    )
    candidates, unreadable = _prune_plan(runtime, DatasetRegistry(runtime.workspace.state_root))
    assert legacy_orphan in candidates
    assert not unreadable

    broken = silver / "broken_0123456789abcdef"
    broken.mkdir()
    monkeypatch.setattr(cli_module, "_validated_current_directories", lambda *_args: ({"daily_market": broken.name}, set()))
    monkeypatch.setattr(cli_module, "_dataset_directories", lambda _runtime: (broken,))
    result = _prune_plan(runtime, DatasetRegistry(runtime.workspace.state_root))
    assert result[0] == []
    assert result[1][0][0] == broken


def test_cli_scope_commands_and_builder_dispatch_paths(tmp_path, monkeypatch, capsys) -> None:
    import src.data.cli as cli_module
    import src.data.ordinary_universe as ordinary_module
    import src.data.dividend_events as dividend_module

    data_root = tmp_path / "data"
    args = [
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(data_root),
    ]
    assert cli_module.main(["scope-info", *args]) == 0
    scope_payload = json.loads(capsys.readouterr().out)
    assert scope_payload["scope_id"] == "kr_swing_2019_v1"
    assert cli_module.main(["init-workspace", *args]) == 0
    assert json.loads(capsys.readouterr().out)["scope_id"] == "kr_swing_2019_v1"

    runtime = _dataset_runtime(tmp_path)
    monkeypatch.setattr(cli_module, "_register_dataset", lambda *_args: None)
    monkeypatch.setattr(ordinary_module, "catalog_master_sessions", lambda _catalog: ())
    monkeypatch.setattr(
        ordinary_module,
        "materialize_ordinary_universe_from_catalog",
        lambda **_kwargs: tmp_path / "ordinary_universe_0123456789abcdef",
    )
    assert cli_module.main(_dataset_cli_args("build-ordinary-universe", runtime)) == 0
    capsys.readouterr()
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text(json.dumps(["2024-01-02"]), encoding="utf-8")
    assert cli_module.main(_dataset_cli_args("build-ordinary-universe", runtime, "--sessions", str(sessions_file))) == 0
    assert json.loads(capsys.readouterr().out)["sessions"] == 1
    sessions_file.write_text(json.dumps({"session": "2024-01-02"}), encoding="utf-8")
    assert cli_module.main(_dataset_cli_args("build-ordinary-universe", runtime, "--sessions", str(sessions_file))) == 1
    assert "JSON list" in json.loads(capsys.readouterr().out)["error"]

    monkeypatch.setattr(
        dividend_module,
        "materialize_dividend_events",
        lambda **_kwargs: tmp_path / "dividend_events_0123456789abcdef",
    )
    assert cli_module.main(_dataset_cli_args("build-dividend-events", runtime)) == 0
    assert json.loads(capsys.readouterr().out)["dataset_id"] == "dividend_events_0123456789abcdef"


def test_cli_normalize_scoped_and_unscoped_registration_paths(tmp_path, monkeypatch, capsys) -> None:
    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    import src.data.cli as cli_module

    assert cli_module.main(["normalize-dart-facts", "--decision-time", "2024-01-01T00:00:00"]) == 1
    assert "requires explicit roots" in json.loads(capsys.readouterr().out)["error"]
    assert cli_module.main([
        "normalize-dart-facts", "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--decision-time", "2024-01-01T00:00:00",
    ]) == 1
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

    monkeypatch.setattr(cli_module, "normalize_dart_facts", _fake_normalize)
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


def test_cli_parse_and_prune_unreadable_apply_boundaries(tmp_path, monkeypatch, capsys) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.cli import _parse_decision_time
    import src.data.cli as cli_module

    with pytest.raises(PITDataError, match="timezone-aware"):
        _parse_decision_time("2024-01-01T00:00:00")

    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    broken = runtime.workspace.silver_root / "broken_0123456789abcdef"
    broken.mkdir()
    monkeypatch.setattr(cli_module, "_prune_plan", lambda *_args: ([broken], []))
    assert cli_module.main(_dataset_cli_args("prune-datasets", runtime, "--apply")) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(line.get("status") == "delete_failed" for line in lines)
