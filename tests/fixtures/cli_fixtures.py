"""Shared CLI and scope fixtures for data-package tests.

Replaces the per-file copies of workspace, transport, and dataset helpers
previously duplicated across CLI test modules.
"""
from __future__ import annotations

from pathlib import Path

from src.data.cli import _parse_args
from typing import ClassVar

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
    from tests.fixtures import register_fact_page

    register_fact_page(bronze_root, receipt_dir)



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
    universe = _publish_cli_fixture(
        runtime.workspace.silver_root,
        "ordinary_universe",
        {
            "session=2024-01-02/part.parquet": pl.DataFrame(
                {"session": [panel_cells[0][0]], "instrument_id": ["KRX:000001"], "ticker": ["000001"], "eligible": [True]}
            ),
            "session=2024-01-03/part.parquet": pl.DataFrame(
                {"session": [panel_cells[1][0]] if len(panel_cells) > 1 else [panel_cells[0][0]],
                 "instrument_id": ["KRX:000001"], "ticker": ["000001"], "eligible": [True]}
            ),
            "session=2024-01-04/part.parquet": pl.DataFrame(
                {"session": [panel_cells[2][0]] if len(panel_cells) > 2 else [panel_cells[0][0]],
                 "instrument_id": ["KRX:000001"], "ticker": ["000001"], "eligible": [True]}
            ),
        },
    )
    daily = _publish_cli_fixture(
        runtime.workspace.silver_root,
        "daily_market",
        {
            "session=2024-01-02/part.parquet": pl.DataFrame(
                {"session": [panel_cells[0][0]], "instrument_id": ["KRX:000001"], "ticker": ["000001"],
                 "price_state": ["tradable"]}
            ),
            "session=2024-01-03/part.parquet": pl.DataFrame(
                {"session": [panel_cells[1][0]] if len(panel_cells) > 1 else [panel_cells[0][0]],
                 "instrument_id": ["KRX:000001"], "ticker": ["000001"], "price_state": ["tradable"]}
            ),
            "session=2024-01-04/part.parquet": pl.DataFrame(
                {"session": [panel_cells[2][0]] if len(panel_cells) > 2 else [panel_cells[0][0]],
                 "instrument_id": ["KRX:000001"], "ticker": ["000001"], "price_state": ["tradable"]}
            ),
        },
    )
    _register_cli_current(runtime, "ordinary_universe", universe)
    _register_cli_current(runtime, "daily_market", daily)
    _register_cli_current(runtime, "investor_flow_ls", ls_flow)
    return panel, ls_flow



def _write_cli_kis_page(bronze_root, symbol, anchor, rows) -> None:
    import hashlib
    import json
    from datetime import UTC, datetime

    from src.core.pit import EvidenceKind
    from src.data.receipt_catalog import BlobEntry, ReceiptCatalog

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
    digest = hashlib.sha256(raw).hexdigest()
    page_dir = bronze_root / "investor_flow" / digest
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "payload.json").write_bytes(raw)
    ReceiptCatalog(bronze_root / "catalog").publish(
        [],
        blobs=[
            BlobEntry(
                content_hash=digest, kind=EvidenceKind.INVESTOR_FLOW, source="kis_investor_flow",
                usable=True, unusable_reason=None,
                retrieved_at=datetime(2024, 1, 4, tzinfo=UTC),
                payload_path=page_dir / "payload.json",
            )
        ],
    )



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
        return [_classification_page(symbol, "inquire-price") for symbol in self.symbols]



def _classification_page(symbol, endpoint):
    """One KIS classification page in the shape the real collector yields."""
    return {
        "provider": "KIS",
        "endpoint": endpoint,
        "symbol": symbol,
        "collected_at": "2026-01-05T09:00:00+09:00",
        "output": {"bstp_kor_isnm": "전기·전자"},
        "records": [{"ticker": symbol, "industry_name": "전기·전자", "market_name": "KOSPI"}],
    }


class _FailingIndustryCollector:
    calls: ClassVar[list] = []
    failing: ClassVar[str] = "000660"

    def __init__(self, symbols, client=None) -> None:
        self.symbols = tuple(symbols)

    def fetch_industry_classification(self, *, bronze_root, retrieved_at=None):
        from src.core.pit import PITDataError

        type(self).calls.append(self.symbols)
        symbol = self.symbols[0]
        if symbol == type(self).failing:
            raise PITDataError(f"KIS industry classification missing bstp_kor_isnm for {symbol}")
        return [_classification_page(symbol, "inquire-price")]



class _FailingStockCollector:
    calls: ClassVar[list] = []
    failing: ClassVar[str] = "000660"

    def __init__(self, symbols, client=None) -> None:
        self.symbols = tuple(symbols)

    def fetch_stock_classification(self, *, bronze_root, retrieved_at=None):
        from src.core.pit import PITDataError

        type(self).calls.append(self.symbols)
        symbol = self.symbols[0]
        if symbol == type(self).failing:
            raise PITDataError(f"KIS stock classification missing valid std_idst_clsf_cd for {symbol}")
        return [_classification_page(symbol, "search-stock-info")]



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



def _krx_cli_stub(*, rows=None, master_rows=None):  # type: ignore[no-untyped-def]
    rows = rows if rows is not None else [{"TDD_CLSPRC": "1000", "MKTCAP": "1", "LIST_SHRS": "2"}]
    master_rows = master_rows if master_rows is not None else [{"ISU_SRT_CD": "005930", "ISU_CD": "KR7005930003"}]

    class _Stub:
        def fetch_daily_records(self, session):  # type: ignore[no-untyped-def]
            return [dict(row) for row in rows]

        def fetch_master_records(self, session):  # type: ignore[no-untyped-def]
            return [dict(row) for row in master_rows]

        def health_check(self) -> None:
            return None

    return _Stub()



def _cli_json_lines(capsys):  # type: ignore[no-untyped-def]
    import json

    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]

# ---------------------------------------------------------------------------
# R9 shared fixtures (single source in tests.fixtures; re-exported here).
# ---------------------------------------------------------------------------

from tests.fixtures import fake_transport, publish_fixture_dataset, scope_runtime  # noqa: F401,E402


_LIVE_SUBCOMMANDS = frozenset(
    {
        "scope-info",
        "init-workspace",
        "index-bronze",
        "collect-scoped",
        "collect-dart-disclosures",
        "collect-dart-corp-codes",
        "collect-dart-facts",
        "collect-dividend-decisions",
        "collect-earnings-releases",
        "reparse-dart-documents",
        "collect-dart-documents",
        "collect-dart-benchmark-documents",
        "benchmark-dart-documents",
        "collect-krx-daily-market",
        "collect-krx-security-master",
        "collect-krx-hedge-series",
        "collect-krx-trend-series",
        "collect-krx-cash-series",
        "migrate-snapshot-krx",
        "collect-kind-notices",
        "collect-kind-documents",
        "collect-ls-investor-flow",
        "collect-kis-investor-flow",
        "collect-industry-classification",
        "collect-stock-classification",
        "normalize-dart-facts",
        "build-financial-quality",
        "build-ordinary-universe",
        "build-dividend-events",
        "build-earnings-releases",
        "benchmark-earnings-releases",
        "build-market-actions",
        "build-investor-flow-silver",
        "build-daily-market-silver",
        "build-hedge-series-silver",
        "build-trend-series-silver",
        "build-cash-series-silver",
        "build-investor-flow-kis-supplement",
        "build-investor-flow-union",
        "build-industry-classification-silver",
        "build-market-panel",
        "build-reference-benchmarks",
        "verify-datasets",
        "prune-datasets",
        "refresh-scope",
        "audit-ordinary-universe-prices",
    }
)

__all__ = [
    "_LIVE_SUBCOMMANDS",
    "_FailingIndustryCollector",
    "_FailingStockCollector",
    "_StubIndustryCollector",
    "_capture_subparsers",
    "_cli_json_lines",
    "_dataset_cli_args",
    "_dataset_runtime",
    "_krx_cli_stub",
    "_publish_cli_dataset",
    "_publish_cli_fixture",
    "_publish_cli_session_dataset",
    "_quality_cli_args",
    "_register_cli_current",
    "_run_classification_command",
    "_stage_quality_facts_dataset",
    "_write_cli_fact_receipt",
    "_write_cli_gap_dataset",
    "_write_cli_gap_inputs",
    "_write_cli_kis_page",
    "_write_cli_universe_dataset",
    "fake_transport",
    "publish_fixture_dataset",
    "scope_runtime",
]
