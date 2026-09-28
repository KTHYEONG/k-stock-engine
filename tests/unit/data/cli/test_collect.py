"""Collect-area CLI commands (scoped persistence, provider jobs, classification)."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from tests.fixtures.cli_fixtures import (
    _write_cli_universe_dataset,
    _StubIndustryCollector,
    _FailingIndustryCollector,
    _FailingStockCollector,
    _run_classification_command,
    _krx_cli_stub,
    _cli_json_lines,
)


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
    ]) == 2
    assert "error" in json.loads(capsys.readouterr().out)



def test_collect_industry_classification_requires_universe(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    assert main([
        "collect-industry-classification",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 2
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
    ]) == 2
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
    ]) == 2
    assert "error" in json.loads(capsys.readouterr().out)



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
        ]) == 2
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
            from tests.fixtures.cli_fixtures import _classification_page

            type(self).calls.append(self.symbols)
            return [_classification_page(symbol, "search-stock-info") for symbol in self.symbols]

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



def _classification_writer(root):
    """The scoped writer a classification command persists its pages through."""
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.runtime import load_data_runtime
    from src.data.scoped_ingestion import ScopedBronzeWriter

    runtime = load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"), data_root=root
    )
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    return ScopedBronzeWriter(runtime=runtime, catalog=catalog), runtime


def test_collect_classification_logs_progress_every_hundred_symbols(tmp_path, caplog) -> None:
    """Progress logging fires on each 100-symbol boundary."""
    import logging

    from src.data.cli import _collect_classification_with_isolation

    class _StubCollector:
        def __init__(self, symbols, client=None) -> None:
            self.symbols = tuple(symbols)

        def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
            return [{
                "provider": "KIS", "endpoint": "inquire-price", "symbol": self.symbols[0],
                "collected_at": "2024-01-02T09:00:00+09:00", "output": {},
                "records": [{"ticker": self.symbols[0], "industry_name": "전기·전자"}],
            }]

    symbols = tuple(f"{index:06d}" for index in range(100))
    writer, runtime = _classification_writer(tmp_path / "data")
    with caplog.at_level(logging.INFO, logger="src.data.industry_collection"):
        result = _collect_classification_with_isolation(
            stage="collect-industry-classification",
            collector_cls=_StubCollector,
            fetch_attr="fetch_industry_classification",
            bronze_root=runtime.workspace.bronze_root,
            writer=writer,
            symbols=symbols,
            pace_seconds=0,
        )
    assert result["pages_collected"] == 100
    assert any("stage=collect-industry-classification" in message for message in caplog.messages)



def test_collect_classification_accepts_naive_collected_at(tmp_path) -> None:
    from src.data.cli import _collect_classification_with_isolation

    class _StubCollector:
        def __init__(self, symbols, client=None) -> None:
            self.symbols = tuple(symbols)

        def fetch_industry_classification(self, *, bronze_root=None, retrieved_at=None):
            return [{
                "provider": "KIS",
                "endpoint": "inquire-price",
                "symbol": self.symbols[0],
                "collected_at": "2024-01-03T09:00:00",
                "output": {"bstp_kor_isnm": "전기·전자"},
                "records": [{"ticker": self.symbols[0], "industry_name": "전기·전자", "market_name": ""}],
            }]

    writer, runtime = _classification_writer(tmp_path / "data")
    result = _collect_classification_with_isolation(
        stage="collect-industry-classification",
        collector_cls=_StubCollector,
        fetch_attr="fetch_industry_classification",
        bronze_root=runtime.workspace.bronze_root,
        writer=writer,
        symbols=("005930",),
        pace_seconds=0,
    )

    assert result["pages_collected"] == 1
    from src.data.receipt_catalog import ReceiptCatalog

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    assert len(list((runtime.workspace.bronze_root / "industry").rglob("payload.json"))) == 1
    assert catalog.successful_keys(source="kis_industry") == frozenset({"inquire-price:005930:2024-01-03"})



def test_collect_krx_daily_market_dry_run(tmp_path, capsys) -> None:
    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    assert main([
        "collect-krx-daily-market",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--dry-run",
    ]) == 0
    summary = _cli_json_lines(capsys)[-1]
    assert summary["job"] == "krx_daily_market"
    assert summary["status"] == "dry_run"
    assert summary["pending_left"] > 0



def test_collect_krx_security_master_executes_one_chunk(tmp_path, capsys, monkeypatch) -> None:
    from src.data.cli import main

    monkeypatch.setattr(
        "src.integrations.krx.client.build_scoped_krx_client",
        lambda **kwargs: _krx_cli_stub(),  # type: ignore[no-untyped-def]
    )
    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    assert main([
        "collect-krx-security-master",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--max-chunks", "1",
    ]) == 0
    summary = _cli_json_lines(capsys)[-1]
    assert summary["job"] == "krx_security_master"
    assert summary["status"] == "chunk_limit"
    assert summary["done"] == 5


def _write_flow_requirement_sets(runtime, sessions, tickers):  # type: ignore[no-untyped-def]
    """Register ordinary-universe and daily-market Silver covering every cell."""
    import polars as pl

    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    rows = [(day, ticker) for day in sessions for ticker in tickers]
    universe = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="ordinary_universe", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={"part.parquet": pl.DataFrame(
            {"session": [day for day, _ in rows], "ticker": [ticker for _, ticker in rows],
             "eligible": [True] * len(rows)},
            schema={"session": pl.Date, "ticker": pl.String, "eligible": pl.Boolean},
        )},
    )
    daily = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="daily_market", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={"part.parquet": pl.DataFrame(
            {"session": [day for day, _ in rows], "ticker": [ticker for _, ticker in rows],
             "price_state": ["tradable"] * len(rows)},
            schema={"session": pl.Date, "ticker": pl.String, "price_state": pl.String},
        )},
    )
    registry = DatasetRegistry(runtime.workspace.state_root)
    registry.register("ordinary_universe", universe.dataset_id)
    registry.register("daily_market", daily.dataset_id)


def _write_fresh_ls_silver(runtime, rows):  # type: ignore[no-untyped-def]
    """Register an LS Silver dataset whose bronze_flow matches the empty catalog."""
    import polars as pl

    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.receipt_catalog import ReceiptCatalog

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="investor_flow_ls", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1",
                                 inputs={"bronze_flow": catalog.blob_digest(source="ls_investor_flow")},
                                 params={}),
        partitions={"part.parquet": pl.DataFrame(
            {"session": [day for day, _ in rows], "ticker": [ticker for _, ticker in rows]},
            schema={"session": pl.Date, "ticker": pl.String},
        )},
    )
    DatasetRegistry(runtime.workspace.state_root).register("investor_flow_ls", published.dataset_id)


def _flow_ls_row(session):  # type: ignore[no-untyped-def]
    return {
        "date": session.strftime("%Y%m%d"), "close": 70100, "volume": 12_000_000, "value": 840_000_000_000,
        "tjj0000": 1, "tjj0001": 2, "tjj0002": 3, "tjj0003": 4, "tjj0004": 5, "tjj0005": 6, "tjj0006": -41,
        "tjj0007": 5, "tjj0008": -60, "tjj0009": 10, "tjj0010": 20, "tjj0011": 45,
        "tjj0016": 30, "tjj0017": 50, "tjj0018": -20,
    }


def _flow_ls_stub_client():  # type: ignore[no-untyped-def]
    class _Stub:
        def inquire_investor_trend(self, symbol, start, end):  # type: ignore[no-untyped-def]
            from datetime import timedelta

            days = []
            day = start
            while day <= end:
                days.append(day)
                day += timedelta(days=1)
            return tuple(_flow_ls_row(day) for day in days)

        def health_check(self) -> None:
            return None

    return _Stub()


def _flow_kis_stub_client():  # type: ignore[no-untyped-def]
    class _StubClient:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):  # type: ignore[no-untyped-def]
            return ({
                "stck_bsop_date": anchor.strftime("%Y%m%d"),
                "prsn_ntby_qty": "-100", "frgn_ntby_qty": "20",
                "orgn_ntby_qty": "-10", "etc_ntby_qty": "90",
            },)

    class _StubCredentials:
        @classmethod
        def from_env(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            return cls()

    return _StubClient, _StubCredentials



def test_collect_ls_investor_flow_dry_run(tmp_path, capsys) -> None:
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_flow_requirement_sets(runtime, [date(2026, 1, 5)], ["005930"])
    assert main([
        "collect-ls-investor-flow",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--dry-run",
    ]) == 0
    summary = _cli_json_lines(capsys)[-1]
    assert summary["job"] == "ls_investor_flow"
    assert summary["status"] == "dry_run"
    assert summary["pending_left"] == 1



def test_collect_ls_investor_flow_executes(tmp_path, capsys, monkeypatch) -> None:
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    monkeypatch.setattr(
        "src.integrations.ls.client.build_scoped_ls_client",
        lambda **kwargs: _flow_ls_stub_client(),  # type: ignore[no-untyped-def]
    )
    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_flow_requirement_sets(runtime, [date(2026, 1, 5)], ["005930"])
    assert main([
        "collect-ls-investor-flow",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 0
    summary = _cli_json_lines(capsys)[-1]
    assert summary["job"] == "ls_investor_flow"
    assert summary["status"] == "complete"
    assert summary["done"] == 1



def test_collect_kis_investor_flow_dry_run(tmp_path, capsys) -> None:
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_flow_requirement_sets(runtime, [date(2026, 1, 5)], ["005930"])
    _write_fresh_ls_silver(runtime, [])
    assert main([
        "collect-kis-investor-flow",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--since", "2026-01-01",
        "--dry-run",
    ]) == 0
    summary = _cli_json_lines(capsys)[-1]
    assert summary["job"] == "kis_investor_flow"
    assert summary["status"] == "dry_run"
    assert summary["pending_left"] == 1



def test_collect_kis_investor_flow_executes(tmp_path, capsys, monkeypatch) -> None:
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    stub_client, stub_creds = _flow_kis_stub_client()
    monkeypatch.setattr("src.integrations.kis.client.KisClient", stub_client)
    monkeypatch.setattr("src.integrations.kis.client.KisCredentials", stub_creds)
    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_flow_requirement_sets(runtime, [date(2026, 1, 5)], ["005930"])
    _write_fresh_ls_silver(runtime, [])
    assert main([
        "collect-kis-investor-flow",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]) == 0
    summary = _cli_json_lines(capsys)[-1]
    assert summary["job"] == "kis_investor_flow"
    assert summary["status"] == "complete"
    assert summary["done"] == 1



def test_collect_kis_investor_flow_without_ls_silver_fails_closed(tmp_path, capsys) -> None:
    import json
    from datetime import date

    from src.data.cli import main
    from src.data.runtime import load_data_runtime

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = load_data_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    _write_flow_requirement_sets(runtime, [date(2026, 1, 5)], ["005930"])
    assert main([
        "collect-kis-investor-flow",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--dry-run",
    ]) == 2
    assert "error" in json.loads(capsys.readouterr().out)


def test_collect_scoped_rejects_unreadable_payload_files(tmp_path, capsys) -> None:
    import json

    from src.data.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    base = [
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
    ]
    assert main(["collect-scoped", *base, "--payloads", str(tmp_path / "missing.json")]) == 2
    assert "error" in json.loads(capsys.readouterr().out)

    not_list = tmp_path / "not-list.json"
    not_list.write_text(json.dumps({"source": "x"}), encoding="utf-8")
    assert main(["collect-scoped", *base, "--payloads", str(not_list)]) == 2
    assert "error" in json.loads(capsys.readouterr().out)

    not_rows = tmp_path / "not-rows.json"
    not_rows.write_text(json.dumps([42]), encoding="utf-8")
    assert main(["collect-scoped", *base, "--payloads", str(not_rows)]) == 2
    assert "error" in json.loads(capsys.readouterr().out)


def test_reparse_and_fetch_documents_dry_run(tmp_path, capsys) -> None:
    from pathlib import Path

    from src.data.cli import main
    from tests.fixtures.cli_fixtures import _cli_json_lines

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    assert main([
        "reparse-dart-documents",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--dry-run",
    ]) == 0
    summary = _cli_json_lines(capsys)[-1]
    assert summary["job"] == "dart_document_reparse"
    assert summary["status"] == "dry_run"

    assert main([
        "collect-dart-documents",
        "--scope-config", str(scope_config),
        "--data-root", str(tmp_path / "data"),
        "--dry-run",
    ]) == 0
    summary = _cli_json_lines(capsys)[-1]
    assert summary["job"] == "dart_document_fetch"
    assert summary["status"] == "dry_run"
