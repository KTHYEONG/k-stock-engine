"""Invariant guards for the declarative whole-scope refresh pipeline."""

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src.core.pit import PITDataError
from src.data.dataset_registry import DatasetRegistry
from src.data.pipeline_graph import (
    SCOPE_GRAPH,
    BuildNode,
    CollectionStep,
    CollectionStepReport,
    RefreshContext,
    RefreshReport,
    build_refresh_context,
    plan_refresh,
    refresh_scope,
    run_collection_jobs,
    run_refresh,
)
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from tests.fixtures import scope_runtime as _scope_runtime
from tests.fixtures import seed_receipts

SESSIONS = tuple(date(2025, 12, 1) + timedelta(days=offset) for offset in range(25))
DECISION_TIME = datetime(2026, 9, 14, tzinfo=UTC)
EXPECTED_KINDS = (
    "ordinary_universe",
    "industry",
    "financial_facts",
    "daily_market",
    "hedge_series",
    "trend_series",
    "cash_series",
    "dividend_events",
    "earnings_releases",
    "investor_flow_ls",
    "financial_quality",
    "market_actions",
    "market_panel",
    "investor_flow_kis_supplement",
    "investor_flow",
    "reference_benchmarks",
)


def _write_bronze_receipt(bronze_root: Path, kind: str, payload: object, *, retrieved_at: datetime) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    receipt_dir = bronze_root / kind / digest
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "payload.json").write_bytes(raw)
    (receipt_dir / "receipt.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "content_hash": digest,
                "source_path": f"fixture:{kind}:{digest[:8]}",
                "retrieved_at": retrieved_at.isoformat(),
                "ingested_at": retrieved_at.isoformat(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if kind == "financial_facts":
        from tests.fixtures import register_fact_page

        register_fact_page(bronze_root, receipt_dir, natural_key=digest)
    return digest


def _publish_catalog_entry(
    catalog: ReceiptCatalog, *, source: str, key: str, as_of: date, payload_path: Path, retrieved_at: datetime
) -> str:
    digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    seed_receipts(
        catalog,
        [
            ReceiptIndexEntry(
                source=source,
                natural_key=key,
                as_of=as_of,
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash=digest,
                retrieved_at=retrieved_at,
                payload_path=payload_path,
            )
        ],
    )
    return digest


def _master_payload(session: date) -> dict[str, object]:
    return {
        "session": session.isoformat(),
        "records": [
            {
                "ISU_SRT_CD": "005930",
                "ISU_CD": "KR7005930003",
                "MKT_TP_NM": "KOSPI",
                "KIND_STKCERT_TP_NM": "보통주",
                "SECUGRP_NM": "주권",
                "ISU_NM": "삼성전자",
                "LIST_DD": "19750611",
            }
        ],
    }


def _daily_payload(session: date, *, close: int = 70000, change: int = 700) -> dict[str, object]:
    base = close - change
    fluc = change / base * 100.0
    return {
        "session": session.isoformat(),
        "records": [
            {
                "ISU_CD": "KR7005930003",
                "ISU_SRT_CD": "005930",
                "MKT_NM": "KOSPI",
                "BAS_DD": session.strftime("%Y%m%d"),
                "TDD_OPNPRC": str(close - 200),
                "TDD_HGPRC": str(close + 100),
                "TDD_LWPRC": str(close - 300),
                "TDD_CLSPRC": str(close),
                "CMPPREVDD_PRC": str(change),
                "FLUC_RT": repr(fluc),
                "ACC_TRDVOL": "1000000",
                "ACC_TRDVAL": str(close * 1000000),
                "MKTCAP": "420000000000000",
                "LIST_SHRS": "6000000000",
            }
        ],
    }


def _index_series_payload(session: date, *, index_level: float = 1000.5, inverse_close: int = 5000) -> dict[str, object]:
    stamp = session.strftime("%Y%m%d")
    return {
        "session": session.isoformat(),
        "records": [
            {"_endpoint": "index", "BAS_DD": stamp, "CLSPRC_IDX": repr(index_level)},
            {"_endpoint": "etf", "BAS_DD": stamp, "TDD_CLSPRC": str(inverse_close)},
        ],
    }


def _cash_series_payload(session: date, *, cash_close: int = 55000) -> dict[str, object]:
    return {
        "session": session.isoformat(),
        "records": [
            {"_endpoint": "etf", "ISU_CD": "153130", "BAS_DD": session.strftime("%Y%m%d"),
             "TDD_CLSPRC": str(cash_close)},
        ],
    }


def _seed_series_receipt(
    catalog: ReceiptCatalog, bronze_root: Path, *, source: str, prefix: str,
    payload: dict[str, object], session: date, retrieved_at: datetime,
) -> None:
    path = bronze_root / "catalog-payloads" / f"{prefix}-{session.isoformat()}.json"
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    _publish_catalog_entry(
        catalog, source=source, key=session.isoformat(), as_of=session,
        payload_path=path, retrieved_at=retrieved_at,
    )


def _fixture_context(tmp_path: Path) -> RefreshContext:
    runtime = _scope_runtime(tmp_path)
    bronze_root = runtime.workspace.bronze_root
    catalog = ReceiptCatalog(bronze_root / "catalog")
    for position, session in enumerate(SESSIONS):
        retrieved = datetime(2025, 12, 31, 12, 0, tzinfo=UTC)
        master_path = bronze_root / "catalog-payloads" / f"master-{session.isoformat()}.json"
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_text(json.dumps(_master_payload(session), sort_keys=True, ensure_ascii=False), encoding="utf-8")
        _publish_catalog_entry(
            catalog, source="krx_security_master", key=session.isoformat(), as_of=session,
            payload_path=master_path, retrieved_at=retrieved,
        )
        daily_path = bronze_root / "catalog-payloads" / f"daily-{session.isoformat()}.json"
        daily_path.write_text(json.dumps(_daily_payload(session), sort_keys=True, ensure_ascii=False), encoding="utf-8")
        _publish_catalog_entry(
            catalog, source="krx_daily_market", key=session.isoformat(), as_of=session,
            payload_path=daily_path, retrieved_at=retrieved,
        )
        _seed_series_receipt(
            catalog, bronze_root, source="krx_hedge_series", prefix="hedge",
            payload=_index_series_payload(session), session=session, retrieved_at=retrieved,
        )
        _seed_series_receipt(
            catalog, bronze_root, source="krx_trend_series", prefix="trend",
            payload=_index_series_payload(session, index_level=2500.25, inverse_close=12000),
            session=session, retrieved_at=retrieved,
        )
        _seed_series_receipt(
            catalog, bronze_root, source="krx_cash_series", prefix="cash",
            payload=_cash_series_payload(session), session=session, retrieved_at=retrieved,
        )
        _ = position
    _write_bronze_receipt(
        bronze_root,
        "financial_facts",
        {"records": [{"ticker": "005930", "corp_code": "00126380", "fiscal_period": "2015Q3",
                      "filing_id": "F1", "fact": "sales", "published_at": "2015-11-16T00:00:00+00:00",
                      "value": 10.0, "unit": "KRW"}]},
        retrieved_at=datetime(2025, 12, 31, 12, 0, tzinfo=UTC),
    )
    _write_bronze_receipt(
        bronze_root,
        "disclosures",
        {"records": []},
        retrieved_at=datetime(2025, 12, 31, 12, 0, tzinfo=UTC),
    )
    bridge = [{"corp_code": "00126380", "ticker": "005930"}]
    bridge_raw = json.dumps(bridge, sort_keys=True).encode("utf-8")
    from tests.fixtures import seed_corp_code_bridge

    seed_corp_code_bridge(bronze_root, bridge_raw)
    from src.data.receipt_catalog import BlobEntry as _BlobEntry
    from src.core.pit import EvidenceKind as _Kind

    def _publish_blob(digest: str, kind_dir: str, *, source: str) -> None:
        catalog.publish(
            [],
            blobs=[
                _BlobEntry(
                    content_hash=digest, kind=_Kind.INDUSTRY if kind_dir == "industry" else _Kind.INVESTOR_FLOW,
                    source=source, usable=True, unusable_reason=None,
                    retrieved_at=datetime(2025, 12, 31, 12, 0, tzinfo=UTC),
                    payload_path=bronze_root / kind_dir / digest / "payload.json",
                )
            ],
        )

    _publish_blob(
        _write_bronze_receipt(
            bronze_root,
            "industry",
            {"provider": "KIS", "endpoint": "inquire-price", "symbol": "005930",
             "collected_at": "2025-12-31T00:00:00+00:00",
             "records": [{"industry_name": "전기전자", "market_name": "KOSPI"}]},
            retrieved_at=datetime(2025, 12, 31, 12, 0, tzinfo=UTC),
        ),
        "industry", source="kis_industry",
    )
    _publish_blob(
        _write_bronze_receipt(
            bronze_root,
            "industry",
            {"provider": "KIS", "endpoint": "search-stock-info", "symbol": "005930",
             "collected_at": "2025-12-31T00:00:00+00:00",
             "records": [{"ksic_code": "261000", "ksic_name": "전자부품 제조업"}]},
            retrieved_at=datetime(2025, 12, 31, 12, 0, tzinfo=UTC),
        ),
        "industry", source="kis_industry",
    )
    _publish_blob(
        _write_bronze_receipt(
            bronze_root,
            "investor_flow",
            {"provider": "LS", "endpoint": "frgr-itt", "symbol": "005930", "anchor": "20251229",
             "query": {"symbol": "005930", "start": SESSIONS[0].isoformat(), "end": SESSIONS[0].isoformat()},
             "rows": [{"date": SESSIONS[0].strftime("%Y%m%d"),
                       "tjj0000": "-100", "tjj0001": "-50", "tjj0002": "-30", "tjj0003": "-20",
                       "tjj0004": "-10", "tjj0005": "-10", "tjj0006": "-8", "tjj0007": "100",
                       "tjj0008": "927", "tjj0009": "-800", "tjj0010": "-28", "tjj0011": "29",
                       "tjj0016": "-828", "tjj0017": "129", "tjj0018": "-228",
                       "close": "50000", "volume": "1000000", "value": "50000"}],
             "records": [{"tick": "005930"}]},
            retrieved_at=datetime(2025, 12, 31, 12, 0, tzinfo=UTC),
        ),
        "investor_flow", source="ls_investor_flow",
    )
    _publish_blob(
        _write_bronze_receipt(
            bronze_root,
            "investor_flow",
            {"provider": "KIS", "endpoint": "investor-trade-by-stock-daily", "symbol": "005930",
             "anchor": "20251202", "query": {"symbol": "005930", "anchor": "20251202"},
             "rows": [{"stck_bsop_date": "20251202", "prsn_ntby_qty": "0", "frgn_ntby_qty": "0",
                       "orgn_ntby_qty": "0", "etc_ntby_qty": "0"}],
             "records": [{"stck_bsop_date": "20251202"}]},
            retrieved_at=datetime(2025, 12, 31, 12, 0, tzinfo=UTC),
        ),
        "investor_flow", source="kis_investor_flow",
    )
    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(
            kind="disclosures", layer=DatasetLayer.SILVER, policy_version="disclosures-fixture-v1",
            inputs={}, params={},
        ),
        partitions={"part.parquet": pl.DataFrame({"filing_id": ["F0"]})},
    )
    invalid_disclosures = runtime.workspace.silver_root / "disclosures_zzzz_invalid"
    invalid_disclosures.mkdir(parents=True, exist_ok=True)
    (invalid_disclosures / "manifest.json").write_text('{"bogus": true}', encoding="utf-8")
    _seed_kind_coverage(bronze_root, catalog)
    return build_refresh_context(runtime, decision_time=DECISION_TIME)


def _seed_kind_coverage(bronze_root: Path, catalog: ReceiptCatalog) -> None:
    """Seed complete empty KIND search windows so the market-actions build sees full coverage."""
    import calendar as _calendar

    from src.config import load_provider_policy, load_runtime_config
    from src.core.pit import EvidenceKind as _EvidenceKind
    from src.core.krx_calendar import xkrx_calendar_through
    from src.core.time import KRX_TZ
    from src.data.jobs.dart import disclosure_windows
    from src.data.receipt_catalog import BlobEntry as _BlobEntry

    provider = load_provider_policy(load_runtime_config())
    decision_day = DECISION_TIME.astimezone(KRX_TZ).date()
    sessions = xkrx_calendar_through(decision_day).sessions
    start = sessions[0].astimezone(KRX_TZ).date()
    through = sessions[-1].astimezone(KRX_TZ).date()
    retrieved = datetime(2025, 12, 31, 12, 0, tzinfo=UTC)
    entries: list[ReceiptIndexEntry] = []
    for keyword in provider.kind.search_keywords:
        for window_start, window_end in disclosure_windows(start, through):
            end_month = (window_start.month - 1) // 3 * 3 + 3
            quarter_end = date(
                window_start.year, end_month,
                _calendar.monthrange(window_start.year, end_month)[1],
            )
            if quarter_end > through:
                continue
            key = f"{keyword}:{window_start.isoformat()}..{window_end.isoformat()}"
            raw = json.dumps(
                {
                    "keyword": keyword,
                    "start": window_start.isoformat(),
                    "end": window_end.isoformat(),
                    "reported_total": 0,
                    "pages": [],
                },
                sort_keys=True, ensure_ascii=False,
            ).encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            payload_path = bronze_root / "catalog-payloads" / f"kind-{digest[:12]}.json"
            payload_path.write_bytes(raw)
            entries.append(
                ReceiptIndexEntry(
                    source="kind_notice_search", natural_key=key, as_of=window_end,
                    fiscal_period=None, status=EvidenceStatus.EMPTY,
                    content_hash=digest, retrieved_at=retrieved, payload_path=payload_path,
                )
            )
    catalog.publish(
        entries,
        blobs=[
            _BlobEntry(
                content_hash=entry.content_hash, kind=_EvidenceKind.DISCLOSURES,
                source="kind_notice_search", usable=True, unusable_reason=None,
                retrieved_at=entry.retrieved_at, payload_path=entry.payload_path,
            )
            for entry in entries
        ],
    )


def _built_fixture(tmp_path: Path) -> RefreshContext:
    ctx = _fixture_context(tmp_path)
    events: list[object] = []
    report = run_refresh(ctx, dry_run=False, emit=events.append)
    assert report.status == "complete"
    assert report.planned == EXPECTED_KINDS
    assert report.built == EXPECTED_KINDS
    return ctx


def test_scope_graph_is_acyclic_and_complete() -> None:
    kinds = [node.kind for node in SCOPE_GRAPH]
    assert sorted(kinds) == sorted(EXPECTED_KINDS)
    position = {kind: index for index, kind in enumerate(kinds)}
    for node in SCOPE_GRAPH:
        for dependency in node.inputs:
            assert position[dependency] < position[node.kind]


def test_scope_graph_validator_rejects_bad_graphs() -> None:
    from src.data.pipeline_graph import _validate_scope_graph

    good = (
        BuildNode(kind="a", inputs=(), bronze_sources=(), build=lambda ctx, inputs: pytest.fail("unused")),
        BuildNode(kind="b", inputs=("a",), bronze_sources=(), build=lambda ctx, inputs: pytest.fail("unused")),
    )
    assert [node.kind for node in _validate_scope_graph(good)] == ["a", "b"]
    cyclic = (
        BuildNode(kind="a", inputs=("b",), bronze_sources=(), build=lambda ctx, inputs: pytest.fail("unused")),
        BuildNode(kind="b", inputs=("a",), bronze_sources=(), build=lambda ctx, inputs: pytest.fail("unused")),
    )
    with pytest.raises(PITDataError, match="cycle"):
        _validate_scope_graph(cyclic)
    duplicated = (good[0], good[0])
    with pytest.raises(PITDataError, match="duplicate"):
        _validate_scope_graph(duplicated)
    dangling = (
        BuildNode(kind="a", inputs=("missing",), bronze_sources=(), build=lambda ctx, inputs: pytest.fail("unused")),
    )
    with pytest.raises(PITDataError, match="unknown kind"):
        _validate_scope_graph(dangling)


def test_plan_refresh_empty_when_inputs_unchanged(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)
    assert plan_refresh(ctx) == ()


def test_plan_refresh_propagates_daily_market_lineage(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)
    bronze_root = ctx.runtime.workspace.bronze_root
    catalog = ReceiptCatalog(bronze_root / "catalog")
    session = SESSIONS[0]
    daily_path = bronze_root / "catalog-payloads" / f"daily-{session.isoformat()}.json"
    daily_path.write_text(
        json.dumps(_daily_payload(session, close=71000), sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    _publish_catalog_entry(
        catalog, source="krx_daily_market", key=session.isoformat(), as_of=session,
        payload_path=daily_path, retrieved_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
    )
    assert [node.kind for node in plan_refresh(ctx)] == [
        "daily_market",
        "dividend_events",
        "market_actions",
        "market_panel",
        "investor_flow_kis_supplement",
        "investor_flow",
        "reference_benchmarks",
    ]


def test_run_refresh_leaves_registry_untouched_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.pipeline_graph as pipeline_graph

    ctx = _built_fixture(tmp_path)
    bronze_root = ctx.runtime.workspace.bronze_root
    catalog = ReceiptCatalog(bronze_root / "catalog")
    session = SESSIONS[0]
    daily_path = bronze_root / "catalog-payloads" / f"daily-{session.isoformat()}.json"
    daily_path.write_text(
        json.dumps(_daily_payload(session, close=71000), sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    _publish_catalog_entry(
        catalog, source="krx_daily_market", key=session.isoformat(), as_of=session,
        payload_path=daily_path, retrieved_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
    )
    registry_path = ctx.runtime.workspace.state_root / "datasets.json"
    before = registry_path.read_bytes()

    def _boom(inner_ctx: RefreshContext, inputs: object) -> object:
        raise PITDataError("boom")

    patched = tuple(
        BuildNode(kind=node.kind, inputs=node.inputs, bronze_sources=node.bronze_sources,
                  build=_boom if node.kind == "market_panel" else node.build)  # type: ignore[arg-type]
        for node in pipeline_graph.SCOPE_GRAPH
    )
    monkeypatch.setattr(pipeline_graph, "SCOPE_GRAPH", patched)
    monkeypatch.setattr(pipeline_graph, "_VALIDATED_GRAPH", patched)
    with pytest.raises(PITDataError, match="boom"):
        run_refresh(ctx, dry_run=False, emit=lambda payload: None)
    assert registry_path.read_bytes() == before
    rebuilt = [path for path in ctx.runtime.workspace.silver_root.iterdir() if path.name.startswith("daily_market_")]
    assert len(rebuilt) == 2


def test_run_refresh_dry_run_makes_no_writes(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)
    registry_path = ctx.runtime.workspace.state_root / "datasets.json"
    report = run_refresh(ctx, dry_run=True, emit=lambda payload: None)
    assert report.status == "dry_run"
    assert list(report.planned) == list(EXPECTED_KINDS)
    assert report.built == ()
    assert not registry_path.exists()
    assert [path.name for path in ctx.runtime.workspace.silver_root.iterdir()
            if not path.name.startswith("disclosures")] == []
    assert list(ctx.runtime.workspace.gold_root.iterdir()) == []


def test_refresh_scope_blocks_build_when_collection_incomplete(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)

    def _ok(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        return CollectionStepReport(job="dart_disclosures", status="complete", done=1, pending_left=0, requests_used=1)

    def _exhausted(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        return CollectionStepReport(
            job="krx_daily_market", status="budget_exhausted", done=0, pending_left=3, requests_used=0
        )

    events: list[object] = []
    report = refresh_scope(
        ctx, collect=True, dry_run=False, emit=events.append,
        steps=(("dart_disclosures", _ok), ("krx_daily_market", _exhausted)),
    )
    assert report.status == "blocked"
    assert report.blocking_job == "krx_daily_market"
    assert report.built == ()
    assert report.planned == ()


def test_refresh_scope_phased_blocks_when_pre_collection_incomplete(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)

    def _ok(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        return CollectionStepReport(job="dart_disclosures", status="complete", done=1, pending_left=0, requests_used=1)

    def _exhausted(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        return CollectionStepReport(
            job="krx_daily_market", status="budget_exhausted", done=0, pending_left=3, requests_used=0
        )

    def _kis_ok(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        return CollectionStepReport(job="kis_investor_flow", status="complete", done=1, pending_left=0, requests_used=1)

    events: list[object] = []
    report = refresh_scope(
        ctx, collect=True, dry_run=False, emit=events.append,
        steps=(("dart_disclosures", _ok), ("krx_daily_market", _exhausted), ("kis_investor_flow", _kis_ok)),
    )
    assert report.status == "blocked"
    assert report.blocking_job == "krx_daily_market"
    assert report.built == ()


def test_refresh_scope_phased_blocks_when_kis_collection_incomplete(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)

    def _ok(name: str):
        def _step(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
            return CollectionStepReport(job=name, status="complete", done=1, pending_left=0, requests_used=1)

        return (name, _step)

    def _kis_exhausted(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        return CollectionStepReport(
            job="kis_investor_flow", status="budget_exhausted", done=0, pending_left=2, requests_used=0
        )

    events: list[object] = []
    report = refresh_scope(
        ctx, collect=True, dry_run=False, emit=events.append,
        steps=(
            _ok("dart_disclosures"), _ok("dart_facts"), _ok("dividend_decisions"),
            _ok("krx_daily_market"), _ok("krx_security_master"), _ok("ls_investor_flow"),
            ("kis_investor_flow", _kis_exhausted),
        ),
    )
    assert report.status == "blocked"
    assert report.blocking_job == "kis_investor_flow"


def test_run_collection_jobs_rejects_step_name_mismatch(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)

    def _misnamed(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        return CollectionStepReport(job="other", status="complete", done=0, pending_left=0, requests_used=0)

    with pytest.raises(PITDataError, match="name mismatch"):
        run_collection_jobs(ctx, dry_run=True, emit=lambda payload: None, steps=(("expected", _misnamed),))


def test_refresh_scope_collect_dry_run_reports_pending(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)
    events: list[object] = []
    report = refresh_scope(ctx, collect=True, dry_run=True, emit=events.append)
    assert report.status == "dry_run"
    assert report.planned == ()
    assert len(report.collection) == 17
    assert {str(step["job"]) for step in report.collection} == {
        "dart_corp_codes", "dart_disclosures", "dart_facts", "dividend_decisions",
        "earnings_releases",
        "krx_daily_market", "krx_security_master",
        "krx_hedge_series", "krx_trend_series", "krx_cash_series",
        "kind_notice_search", "kind_notice_documents",
        "ls_investor_flow", "kis_investor_flow",
        "dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar",
    }


def test_plan_refresh_rejects_unknown_registry_kind(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)
    registry_path = ctx.runtime.workspace.state_root / "datasets.json"
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    raw["current"]["mystery_kind"] = "mystery_kind_0123456789abcdef"
    registry_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(PITDataError, match="outside the scope graph"):
        plan_refresh(ctx)


def test_plan_refresh_marks_missing_or_corrupt_datasets_stale(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)
    registry_path = ctx.runtime.workspace.state_root / "datasets.json"
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    original_daily = raw["current"]["daily_market"]
    raw["current"]["daily_market"] = "daily_market_0123456789abcdef"
    registry_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    assert [node.kind for node in plan_refresh(ctx)] == [
        "daily_market",
        "dividend_events",
        "market_actions",
        "market_panel",
        "investor_flow_kis_supplement",
        "investor_flow",
        "reference_benchmarks",
    ]
    dividend_id = raw["current"]["dividend_events"]
    (ctx.runtime.workspace.silver_root / dividend_id / "manifest.json").write_text('{"bogus": true}', encoding="utf-8")
    raw["current"]["daily_market"] = original_daily
    registry_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    assert [node.kind for node in plan_refresh(ctx)] == ["dividend_events"]


def test_plan_refresh_succeeds_when_payloads_unreadable(tmp_path: Path) -> None:
    import shutil

    ctx = _built_fixture(tmp_path)
    shutil.rmtree(ctx.runtime.workspace.bronze_root / "industry")
    # Previews use only catalog digests, so unreadable payload files never
    # fail planning; the registered industry dataset stays fresh.
    assert "industry" not in [node.kind for node in plan_refresh(ctx)]


def test_disk_dataset_ids_skips_unreadable_entries(tmp_path: Path) -> None:
    import shutil

    import src.data.pipeline_graph as pipeline_graph

    ctx = _built_fixture(tmp_path)
    stray = ctx.runtime.workspace.silver_root / "stray.txt"
    stray.write_text("not a dataset", encoding="utf-8")
    known = pipeline_graph._disk_dataset_ids(ctx, {"extra_0000000000000000"})
    assert "extra_0000000000000000" in known
    assert set(ctx.registry.snapshot().values()) <= known
    shutil.rmtree(ctx.runtime.workspace.gold_root)
    rescanned = pipeline_graph._disk_dataset_ids(ctx, set())
    assert ctx.registry.snapshot()["ordinary_universe"] in rescanned
    assert ctx.registry.snapshot()["market_panel"] not in rescanned


def test_collection_report_blocking_job_none_when_complete() -> None:
    from src.data.pipeline_graph import CollectionReport

    report = CollectionReport(steps=(
        CollectionStepReport(job="a", status="complete", done=1, pending_left=0, requests_used=1),
    ))
    assert report.complete
    assert report.blocking_job is None
    assert not CollectionReport(steps=()).complete


def test_run_refresh_fails_closed_on_missing_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polars as pl

    import src.data.pipeline_graph as pipeline_graph
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    ctx = _fixture_context(tmp_path)
    published = publish_dataset(
        layer_root=ctx.runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="lonely", layer=DatasetLayer.SILVER,
                                 policy_version="lonely-v1", inputs={}, params={}),
        partitions={"part.parquet": pl.DataFrame({"a": [1]})},
    )
    ctx.registry.register("lonely", published.path.name)
    changed = DatasetIdentity(kind="lonely", layer=DatasetLayer.SILVER,
                              policy_version="lonely-v2", inputs={}, params={})
    monkeypatch.setitem(pipeline_graph._PREVIEW_FNS, "lonely", lambda inner_ctx: changed)
    lonely = BuildNode(kind="lonely", inputs=("ghost",), bronze_sources=(),
                       build=lambda inner_ctx, inputs: pytest.fail("unused"))
    monkeypatch.setattr(pipeline_graph, "SCOPE_GRAPH", (lonely,))
    monkeypatch.setattr(pipeline_graph, "_VALIDATED_GRAPH", (lonely,))
    with pytest.raises(PITDataError, match="not registered"):
        run_refresh(ctx, dry_run=False, emit=lambda payload: None)


def test_run_refresh_fails_closed_on_verification_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.pipeline_graph as pipeline_graph
    from src.data.datasets import PublishedDataset

    ctx = _fixture_context(tmp_path)
    phantom = ctx.runtime.workspace.silver_root / "phantom"
    phantom.mkdir(parents=True, exist_ok=True)

    def _bad_build(inner_ctx: RefreshContext, inputs: object) -> PublishedDataset:
        return PublishedDataset(dataset_id="phantom_0123456789abcdef", path=phantom, rows=0)

    node = BuildNode(kind="phantom", inputs=(), bronze_sources=(), build=_bad_build)  # type: ignore[arg-type]
    monkeypatch.setattr(pipeline_graph, "SCOPE_GRAPH", (node,))
    monkeypatch.setattr(pipeline_graph, "_VALIDATED_GRAPH", (node,))
    with pytest.raises(PITDataError, match="verification failed"):
        run_refresh(ctx, dry_run=False, emit=lambda payload: None)
    assert not (ctx.runtime.workspace.state_root / "datasets.json").exists()


def test_build_refresh_context_rejects_naive_decision_time(tmp_path: Path) -> None:
    runtime = _scope_runtime(tmp_path)
    with pytest.raises(PITDataError, match="timezone-aware"):
        build_refresh_context(runtime, decision_time=datetime(2026, 9, 14))


def test_register_many_registers_atomically(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)
    registry = DatasetRegistry(ctx.runtime.workspace.state_root)
    registry_path = ctx.runtime.workspace.state_root / "datasets.json"
    before = registry_path.read_bytes()
    snapshot = dict(registry.snapshot())
    registry.register_many({"industry": snapshot["industry"]})
    assert registry_path.read_bytes() == before
    with pytest.raises(PITDataError, match="does not exist"):
        registry.register_many(
            {"industry": snapshot["industry"], "daily_market": "daily_market_0123456789abcdef"}
        )
    assert registry_path.read_bytes() == before
    registry.register_many({})
    assert registry_path.read_bytes() == before


def test_flow_collection_steps_report_range_planned_pending(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)
    events: list[object] = []
    report = run_collection_jobs(ctx, dry_run=True, emit=events.append)
    by_job = {step.job: step for step in report.steps}

    assert by_job["ls_investor_flow"].pending_left == 1
    assert by_job["ls_investor_flow"].status == "dry_run"
    assert by_job["kis_investor_flow"].status == "dry_run"


def test_flow_collection_dry_run_without_registry_reports_kis_blocked(tmp_path: Path) -> None:
    import src.data.pipeline_graph as pipeline_graph

    ctx = _fixture_context(tmp_path)
    events: list[object] = []

    ls = pipeline_graph._run_ls_flow_collection_step(ctx, dry_run=True, emit=events.append)
    kis = pipeline_graph._run_kis_flow_collection_step(ctx, dry_run=True, emit=events.append)

    assert (ls.status, ls.pending_left) == ("dry_run", 0)
    # LS Silver 없이 KIS 대상을 셀 수 없으므로 "할 일 없음"이 아니라 차단으로 보고해야 한다.
    assert (kis.status, kis.pending_left) == ("blocked_until_ls_build", 0)


def test_kis_flow_collection_dry_run_reports_fresh_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.data.pipeline_graph as pipeline_graph
    from src.data.jobs.flow import KisInvestorFlowJob
    from src.data.jobs.runner import JobUnit

    ctx = _built_fixture(tmp_path)
    units = [
        JobUnit(source="kis_investor_flow", natural_key="005930", payload={"symbol": "005930", "windows": "[]"},
                max_requests=1)
    ]
    monkeypatch.setattr(KisInvestorFlowJob, "pending", lambda self, ctx: units)
    events: list[object] = []

    report = pipeline_graph._run_kis_flow_collection_step(ctx, dry_run=True, emit=events.append)

    assert (report.status, report.pending_left) == ("dry_run", 1)


def test_flow_collection_steps_fail_closed_when_not_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.data.pipeline_graph as pipeline_graph

    ctx = _fixture_context(tmp_path)

    class _StubLsClient:
        def inquire_investor_trend(self, symbol, start, end):  # type: ignore[no-untyped-def]
            return ()

        def health_check(self) -> None:
            return None

    monkeypatch.setattr(
        "src.integrations.ls.client.build_scoped_ls_client", lambda **kwargs: _StubLsClient()
    )

    class _StubKisClient:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

    class _StubKisCredentials:
        @classmethod
        def from_env(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            return cls()

    monkeypatch.setattr("src.integrations.kis.client.KisClient", _StubKisClient)
    monkeypatch.setattr("src.integrations.kis.client.KisCredentials", _StubKisCredentials)
    monkeypatch.setattr("src.integrations.kis.build_pool_client", lambda *args, **kwargs: _StubKisClient())
    with pytest.raises(PITDataError, match="not registered"):
        pipeline_graph._run_ls_flow_collection_step(ctx, dry_run=False, emit=lambda payload: None)
    with pytest.raises(PITDataError, match="not registered"):
        pipeline_graph._run_kis_flow_collection_step(ctx, dry_run=False, emit=lambda payload: None)


def test_refresh_scope_command_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from src.data.cli import main

    _fixture_context(tmp_path)
    assert main([
        "refresh-scope",
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(tmp_path / "data"),
        "--dry-run",
    ]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary = lines[-1]
    assert summary["type"] == "summary"
    assert summary["status"] == "dry_run"
    assert summary["planned"] == list(EXPECTED_KINDS)


def test_refresh_scope_command_reports_invalid_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.data.cli import main

    assert main([
        "refresh-scope",
        "--scope-config", "config/research/does-not-exist.toml",
        "--data-root", str(tmp_path / "data"),
        "--dry-run",
    ]) == 2
    assert json.loads(capsys.readouterr().out)["error"]


def test_refresh_scope_command_collect_dry_run_and_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.data.cli as cli_module

    _built_fixture(tmp_path)
    assert cli_module.main([
        "refresh-scope",
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(tmp_path / "data"),
        "--collect", "--dry-run",
    ]) == 0

    import src.data.pipeline_graph as pipeline_graph

    real_refresh_scope = pipeline_graph.refresh_scope

    def _blocked(
        ctx: RefreshContext, *, collect: bool, dry_run: bool, emit: object, steps: object = None,
        collect_steps: object = None,
    ) -> object:
        assert collect
        assert not dry_run
        assert collect_steps is None
        return real_refresh_scope(
            ctx, collect=False, dry_run=True, emit=emit,
        ).__class__(
            decision_time=DECISION_TIME, status="blocked", planned=(), built=(),
            datasets={}, blocking_job="krx_daily_market", collection=(), degraded_jobs=("kind_notice_search",),
        )

    monkeypatch.setattr(pipeline_graph, "refresh_scope", _blocked)
    assert cli_module.main([
        "refresh-scope",
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(tmp_path / "data"),
        "--collect",
    ]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["degraded_jobs"] == ["kind_notice_search"]


def test_build_refresh_context_reads_superseded_receipts_from_state(tmp_path: Path) -> None:
    from src.data.fact_state import SUPERSEDED_RECEIPTS_FILE

    runtime = _scope_runtime(tmp_path)
    runtime.workspace.state_root.mkdir(parents=True, exist_ok=True)
    (runtime.workspace.state_root / SUPERSEDED_RECEIPTS_FILE).write_text(json.dumps(["b" * 64]), encoding="utf-8")

    assert build_refresh_context(runtime).superseded_receipts == frozenset({"b" * 64})
    assert build_refresh_context(runtime, superseded_receipts=frozenset()).superseded_receipts == frozenset()


def test_refresh_quality_build_keeps_quarantine_and_manual_events(tmp_path: Path) -> None:
    """The refresh must feed the quality build the same evidence as ``build-financial-quality``."""
    import hashlib

    import polars as pl

    from src.data import pipeline_graph as graph
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset, read_dataset
    from src.data.fact_state import UNRESOLVED_EVENTS_FILE

    runtime = _scope_runtime(tmp_path)
    state = runtime.workspace.state_root
    state.mkdir(parents=True, exist_ok=True)
    available = datetime(2020, 3, 31, tzinfo=UTC)
    facts = pl.DataFrame([
        {
            "company_id": "A", "fiscal_period": "2018Q4", "filing_id": "A-2018Q4", "fact": fact,
            "published_at": available, "available_at": available, "value": 100.0, "unit": "KRW", "consolidated": True,
        }
        for fact in ("sales", "gross_profit", "operating_profit", "net_income", "assets", "equity", "operating_cash_flow")
    ])
    quarantine = state / "dart_fact_quarantine_fixture.json"
    quarantine.write_text(json.dumps([{
        "company_id": "A", "dart_corp_code": "00126380", "fiscal_period": "2019Q4", "filing_id": "Q1",
        "source_kind": "legacy_document", "published_at": available.isoformat(), "available_at": available.isoformat(),
    }]), encoding="utf-8")
    (state / UNRESOLVED_EVENTS_FILE).write_text(json.dumps([{
        "company_id": "B", "fiscal_period": "2019Q4", "filing_id": "M1", "published_at": available.isoformat(),
        "available_at": available.isoformat(), "reason": "missing_source_value",
    }]), encoding="utf-8")
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="financial_facts", layer=DatasetLayer.SILVER, policy_version="fixture-v1", inputs={}, params={}),
        partitions={"part-00000.parquet": facts},
        details={"quarantine_file": quarantine.name, "quarantine_sha256": hashlib.sha256(quarantine.read_bytes()).hexdigest()},
    )
    ctx = build_refresh_context(runtime, decision_time=datetime(2020, 4, 1, tzinfo=UTC))

    result = graph._build_financial_quality(ctx, {"financial_facts": published.dataset_id})

    reasons = set(read_dataset(result.path).collect()["exclusion_reason"].to_list())
    assert {"unverified_legacy_extraction", "missing_source_value"} <= reasons


def test_facts_preview_excludes_superseded_receipts_like_the_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from src.data import incremental_normalization as inc
    from src.data import pipeline_graph as graph
    from src.data.datasets import dataset_digest

    ctx = _built_fixture(tmp_path)
    monkeypatch.setattr(inc, "_discover_fact_receipts", lambda _root, **_: [{"content_hash": "a" * 64}, {"content_hash": "b" * 64}])

    plain = graph._preview_financial_facts(replace(ctx, superseded_receipts=frozenset()))
    superseded = graph._preview_financial_facts(replace(ctx, superseded_receipts=frozenset({"b" * 64})))

    assert plain.inputs["bronze_facts"] == dataset_digest(["a" * 64, "b" * 64])
    assert superseded.inputs["bronze_facts"] == dataset_digest(["a" * 64])


def test_refresh_orders_kis_collection_after_ls_build(tmp_path: Path) -> None:
    """Refresh orders KIS after LS build: LS collect, LS build, KIS collect, supplement build."""
    import src.data.pipeline_graph as pipeline_graph

    ctx = _fixture_context(tmp_path)
    calls: list[str] = []

    def _collect(name: str):
        def _step(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
            calls.append(f"collect:{name}")
            return CollectionStepReport(job=name, status="complete", done=1, pending_left=0, requests_used=0)

        return (name, _step)

    real_builders = {node.kind: node.build for node in pipeline_graph.SCOPE_GRAPH}

    def _wrap(kind: str):
        def _build(inner_ctx: RefreshContext, inputs: object) -> object:
            calls.append(f"build:{kind}")
            return real_builders[kind](inner_ctx, inputs)

        return _build

    patched = tuple(
        pipeline_graph.BuildNode(kind=node.kind, inputs=node.inputs, bronze_sources=node.bronze_sources,
                                 build=_wrap(node.kind))
        for node in pipeline_graph.SCOPE_GRAPH
    )
    ctx2 = ctx
    original_graph = pipeline_graph.SCOPE_GRAPH
    original_validated = pipeline_graph._VALIDATED_GRAPH
    try:
        pipeline_graph.SCOPE_GRAPH = patched  # type: ignore[assignment]
        pipeline_graph._VALIDATED_GRAPH = patched  # type: ignore[assignment]
        steps = (
            _collect("dart_disclosures"), _collect("dart_facts"), _collect("dividend_decisions"),
            _collect("krx_daily_market"), _collect("krx_security_master"),
            _collect("ls_investor_flow"), _collect("kis_investor_flow"),
        )
        report = pipeline_graph.refresh_scope(
            ctx2, collect=True, dry_run=False, emit=lambda payload: None, steps=steps,
        )
    finally:
        pipeline_graph.SCOPE_GRAPH = original_graph  # type: ignore[assignment]
        pipeline_graph._VALIDATED_GRAPH = original_validated  # type: ignore[assignment]
    assert report.status == "complete"
    ls_collect = calls.index("collect:ls_investor_flow")
    ls_build = calls.index("build:investor_flow_ls")
    kis_collect = calls.index("collect:kis_investor_flow")
    supplement_build = calls.index("build:investor_flow_kis_supplement")
    assert ls_collect < ls_build < kis_collect < supplement_build


def test_previews_read_no_payload(tmp_path: Path) -> None:
    """Preview reads no payload: plan_refresh succeeds with flow/industry payloads unreadable."""
    from contextlib import suppress

    ctx = _built_fixture(tmp_path)
    targets = [
        *ctx.runtime.workspace.bronze_root.rglob("investor_flow/*/payload.json"),
        *ctx.runtime.workspace.bronze_root.rglob("industry/*/payload.json"),
    ]
    for path in targets:
        with suppress(OSError):
            path.chmod(0o000)
    try:
        assert plan_refresh(ctx) == ()
    finally:
        for path in targets:
            with suppress(OSError):
                path.chmod(0o644)


def test_supplement_preview_has_no_gold_input(tmp_path: Path) -> None:
    import src.data.pipeline_graph as pipeline_graph

    ctx = _built_fixture(tmp_path)
    identity = pipeline_graph._preview_investor_flow_kis_supplement(ctx)
    assert set(identity.inputs) == {"universe", "daily_market", "ls", "bronze_kis"}
    assert identity.inputs["bronze_kis"] == ctx.catalog.blob_digest(source="kis_investor_flow")


def test_ls_preview_uses_catalog_digest(tmp_path: Path) -> None:
    import src.data.pipeline_graph as pipeline_graph

    ctx = _built_fixture(tmp_path)
    identity = pipeline_graph._preview_investor_flow_ls(ctx)
    assert identity.inputs["bronze_flow"] == ctx.catalog.blob_digest(source="ls_investor_flow")


def test_ls_parse_cache_prunes_only_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.investor_flow_silver as flow
    import src.data.pipeline_graph as graph
    from src.data.parse_cache import ParseCache

    ctx = _fixture_context(tmp_path)
    universe = graph._build_ordinary_universe(ctx, {})
    inputs = {"ordinary_universe": universe.dataset_id}
    cache_path = graph._parse_cache_path(ctx, "ls_investor_flow")
    with ParseCache(cache_path, namespace="ls_investor_flow-v1", fingerprint=flow.parse_fingerprint()) as cache:
        cache.put_many({"retired": {"value": 1}})

    original_publish = flow.publish_dataset

    def fail_publish(**kwargs: object) -> None:
        raise PITDataError("publication failed")

    monkeypatch.setattr(flow, "publish_dataset", fail_publish)
    with pytest.raises(PITDataError, match="publication failed"):
        graph._build_investor_flow_ls(ctx, inputs)
    with ParseCache(cache_path, namespace="ls_investor_flow-v1", fingerprint=flow.parse_fingerprint()) as cache:
        assert cache.get_many(["retired"]) == {"retired": {"value": 1}}

    monkeypatch.setattr(flow, "publish_dataset", original_publish)
    result = graph._build_investor_flow_ls(ctx, inputs)
    assert result.path.is_dir()
    with ParseCache(cache_path, namespace="ls_investor_flow-v1", fingerprint=flow.parse_fingerprint()) as cache:
        assert cache.get_many(["retired"]) == {}
        hashes = [blob.content_hash for blob in ctx.catalog.blobs(source="ls_investor_flow")]
        assert set(cache.get_many(hashes)) == set(hashes)


def test_build_nodes_and_phase_helpers_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.pipeline_graph as pipeline_graph
    from src.data.datasets import PublishedDataset

    ctx = _fixture_context(tmp_path)
    with pytest.raises(PITDataError, match="not registered"):
        pipeline_graph._build_nodes(
            ctx,
            (pipeline_graph.BuildNode(kind="x", inputs=("ghost",), bronze_sources=(),
                                      build=lambda _c, _i: pytest.fail("unused")),),
            {}, {}, set(), lambda _p: None,
        )

    def _bad_build(inner_ctx: RefreshContext, inputs: object) -> PublishedDataset:
        return PublishedDataset(dataset_id="phantom_0123456789abcdef", path=tmp_path / "phantom", rows=0)

    (tmp_path / "phantom").mkdir(exist_ok=True)
    with pytest.raises(PITDataError, match="verification failed"):
        pipeline_graph._build_nodes(
            ctx,
            (pipeline_graph.BuildNode(kind="phantom", inputs=(), bronze_sources=(), build=_bad_build),),  # type: ignore[arg-type]
            {}, {}, set(), lambda _p: None,
        )
    assert pipeline_graph._phase_split((),) == ((), (), ())
    with pytest.raises(PITDataError, match="no certified"):
        pipeline_graph._preview_industry(build_refresh_context(_scope_runtime(tmp_path / "empty-scope")))

    built_ctx = _built_fixture(tmp_path / "preview-fail-scope")

    def _boom(_inner_ctx: RefreshContext) -> object:
        raise PITDataError("boom-preview")

    monkeypatch.setitem(pipeline_graph._PREVIEW_FNS, "daily_market", _boom)
    assert "daily_market" in [node.kind for node in plan_refresh(built_ctx)]


def test_refresh_dry_run_reports_phased_order(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)
    events: list[object] = []

    def _collect(name: str):
        def _step(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
            return CollectionStepReport(job=name, status="dry_run", done=0, pending_left=1, requests_used=0)

        return (name, _step)

    steps = (
        _collect("dart_disclosures"), _collect("dart_facts"), _collect("dividend_decisions"),
        _collect("krx_daily_market"), _collect("krx_security_master"),
        _collect("ls_investor_flow"), _collect("kis_investor_flow"),
    )
    report = refresh_scope(ctx, collect=True, dry_run=True, emit=events.append, steps=steps)
    assert report.status == "dry_run"
    assert report.blocking_job is None
    progress = [item.get("job", item.get("kind")) for item in events if isinstance(item, dict)]
    phase_a = ["ordinary_universe", "daily_market", "investor_flow_ls"]
    phase_b = ["investor_flow_kis_supplement", "investor_flow"]
    phase_c = [kind for kind in EXPECTED_KINDS if kind not in phase_a + phase_b]
    assert progress == [name for name, _ in steps[:-1]] + phase_a + ["kis_investor_flow"] + phase_b + phase_c


def test_facts_preview_plans_without_hashing_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stale-check builds its identity without hashing any payload file."""
    from src.data import incremental_normalization as inc
    from src.data import pipeline_graph as graph

    ctx = _fixture_context(tmp_path)
    calls = 0
    real_sha = inc._sha256_file

    def _counted(path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_sha(path)

    monkeypatch.setattr(inc, "_sha256_file", _counted)
    first = graph._preview_financial_facts(ctx)
    second = graph._preview_financial_facts(ctx)
    assert calls == 0
    assert first == second
    assert first.inputs["bronze_facts"]


def test_graph_contains_earnings_releases() -> None:
    import src.data.pipeline_graph as graph_module

    node = next(item for item in SCOPE_GRAPH if item.kind == "earnings_releases")

    assert node.inputs == ()
    assert graph_module._PREVIEW_FNS["earnings_releases"] is graph_module._preview_earnings_releases


def _seed_earnings_envelope(bronze_root: Path, *, rcept_no: str, received_on: str,
                            retrieved_at: datetime | None = None,
                            sales_quarter: str = "321,223", variant: str = "") -> None:
    import base64
    import hashlib as _hashlib
    import io as _io
    import zipfile as _zipfile

    from src.data.evidence_sources import EARNINGS_RELEASE_SOURCE
    from src.data.receipt_catalog import EvidenceStatus, ReceiptIndexEntry

    rows = [
        ["1. 연결실적내용", "단위 : 백만원, %"],
        ["구분", "당기실적", "전기실적", "전기대비증감율(%)", "전년동기실적", "전년동기대비증감율(%)"],
        ["('18.4Q)", "('18.3Q)", "('17.4Q)"],
        ["매출액", "당해실적", sales_quarter, "376,466", "-14.7", "87,836", "265.7"],
        ["누계실적", "1,365,439", "1,044,216", "-", "87,836", "1,454.5"],
    ]
    body = "<table>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    ) + "</table>"
    output = _io.BytesIO()
    with _zipfile.ZipFile(output, "w") as archive:
        archive.writestr("document.xml", body)
    archive_bytes = output.getvalue()
    envelope = {
        "rcept_no": rcept_no,
        "corp_code": "00126380",
        "received_on": received_on,
        "report_nm": "연결재무제표기준영업(잠정)실적(공정공시)",
        "archive_b64": base64.b64encode(archive_bytes).decode("ascii"),
        "archive_sha256": _hashlib.sha256(archive_bytes).hexdigest(),
    }
    raw = json.dumps(envelope, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload_path = bronze_root / "catalog-payloads" / f"earnings-{rcept_no}{variant}.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(raw)
    catalog = ReceiptCatalog(bronze_root / "catalog")
    seed_receipts(
        catalog,
        [
            ReceiptIndexEntry(
                source=EARNINGS_RELEASE_SOURCE, natural_key=rcept_no,
                as_of=date.fromisoformat(received_on), fiscal_period=None,
                status=EvidenceStatus.SUCCESS, content_hash=_hashlib.sha256(raw).hexdigest(),
                retrieved_at=(retrieved_at or datetime(2026, 1, 2, 12, 0, tzinfo=UTC)),
                payload_path=payload_path,
            )
        ],
    )


def test_earnings_preview_equals_built_identity(tmp_path: Path) -> None:
    import src.data.pipeline_graph as graph_module
    from src.data.datasets import dataset_id_for
    from tests.fixtures import seed_corp_code_bridge

    runtime = _scope_runtime(tmp_path)
    seed_corp_code_bridge(
        runtime.workspace.bronze_root,
        json.dumps([{"corp_code": "00126380", "ticker": "005930"}], sort_keys=True).encode("utf-8"),
    )
    _seed_earnings_envelope(runtime.workspace.bronze_root, rcept_no="20190131000001", received_on="2019-01-31")
    _seed_earnings_envelope(runtime.workspace.bronze_root, rcept_no="20190215000002", received_on="2019-02-15")
    _seed_earnings_envelope(
        runtime.workspace.bronze_root, rcept_no="20190215000002", received_on="2019-02-15",
        retrieved_at=datetime(2026, 1, 3, 12, 0, tzinfo=UTC),
        sales_quarter="322,000", variant="-dup",
    )
    ctx = build_refresh_context(runtime)

    expected = graph_module._preview_earnings_releases(ctx)
    built = graph_module._build_earnings_releases(ctx, {})

    assert built.dataset_id == dataset_id_for(expected)


def test_series_preview_equals_built_identity(tmp_path: Path) -> None:
    """Each series preview is exactly the identity its build publishes."""
    import src.data.pipeline_graph as graph_module
    from src.data.datasets import dataset_id_for

    ctx = _fixture_context(tmp_path)
    universe_id = graph_module._build_ordinary_universe(ctx, {}).dataset_id
    ctx.registry.register("ordinary_universe", universe_id)
    for preview_fn, build_fn in (
        (graph_module._preview_hedge_series, graph_module._build_hedge_series),
        (graph_module._preview_trend_series, graph_module._build_trend_series),
        (graph_module._preview_cash_series, graph_module._build_cash_series),
    ):
        built = build_fn(ctx, {"ordinary_universe": universe_id})
        assert built.dataset_id == dataset_id_for(preview_fn(ctx))


def test_series_preview_gap_replans_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing series page fails the preview closed, so the series is rebuilt."""
    ctx = _built_fixture(tmp_path)
    real_latest = ctx.catalog.latest

    def _gapped(*, source: str, natural_keys: object) -> object:
        result = real_latest(source=source, natural_keys=natural_keys)  # type: ignore[arg-type]
        if source == "krx_hedge_series":
            result = {key: entry for key, entry in result.items() if key != SESSIONS[0].isoformat()}
        return result

    monkeypatch.setattr(ctx.catalog, "latest", _gapped)
    assert [node.kind for node in plan_refresh(ctx)] == ["hedge_series"]


def test_series_preview_conflict_replans_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A series receipt filed under the wrong session fails the preview closed."""
    from dataclasses import replace

    ctx = _built_fixture(tmp_path)
    real_latest = ctx.catalog.latest

    def _conflicted(*, source: str, natural_keys: object) -> object:
        result = dict(real_latest(source=source, natural_keys=natural_keys))  # type: ignore[arg-type]
        if source == "krx_cash_series":
            key = SESSIONS[0].isoformat()
            result[key] = replace(result[key], as_of=SESSIONS[1])
        return result

    monkeypatch.setattr(ctx.catalog, "latest", _conflicted)
    assert [node.kind for node in plan_refresh(ctx)] == ["cash_series"]


def test_refresh_scope_collect_steps_runs_only_named(
    tmp_path: Path,
) -> None:
    """A step filter runs only the named steps, in default order, and still plans."""
    ctx = _fixture_context(tmp_path)
    calls: list[str] = []

    def _stub(name: str) -> object:
        def _step(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
            calls.append(name)
            return CollectionStepReport(job=name, status="complete", done=1, pending_left=0, requests_used=0)

        return (name, _step)

    import src.data.pipeline_graph as pipeline_graph

    steps = tuple(_stub(name) for name, _ in pipeline_graph._default_collection_steps())  # type: ignore[misc]
    report = refresh_scope(
        ctx, collect=True, dry_run=True, emit=lambda payload: None,
        steps=steps,  # type: ignore[arg-type]
        collect_steps=frozenset({"krx_daily_market", "krx_security_master"}),
    )
    assert calls == ["krx_daily_market", "krx_security_master"]
    assert report.status == "dry_run"
    assert report.planned == EXPECTED_KINDS


def test_refresh_scope_collect_steps_rejects_unknown(tmp_path: Path) -> None:
    """An unknown step name fails before any collection runs."""
    import src.data.pipeline_graph as pipeline_graph

    ctx = _built_fixture(tmp_path)
    calls: list[str] = []

    def _stub(name: str) -> object:
        def _step(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
            calls.append(name)
            return CollectionStepReport(job=name, status="complete", done=1, pending_left=0, requests_used=0)

        return (name, _step)

    steps = tuple(_stub(name) for name, _ in pipeline_graph._default_collection_steps())  # type: ignore[misc]
    with pytest.raises(ValueError, match="unknown collection steps"):
        refresh_scope(
            ctx, collect=True, dry_run=True, emit=lambda payload: None,
            steps=steps,  # type: ignore[arg-type]
            collect_steps=frozenset({"nope"}),
        )
    assert calls == []


def test_refresh_scope_command_collect_steps_requires_collect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.data.cli import main

    _fixture_context(tmp_path)
    assert main([
        "refresh-scope",
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(tmp_path / "data"),
        "--collect-steps", "krx_daily_market",
    ]) == 2
    assert json.loads(capsys.readouterr().out)["error"]


def test_refresh_scope_command_collect_steps_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.data.cli import main

    _built_fixture(tmp_path)
    assert main([
        "refresh-scope",
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(tmp_path / "data"),
        "--collect", "--dry-run",
        "--collect-steps", "krx_daily_market,krx_security_master",
    ]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary = lines[-1]
    assert summary["type"] == "summary"
    assert {step["job"] for step in summary["collection"]} == {"krx_daily_market", "krx_security_master"}


@pytest.mark.parametrize("selected", [frozenset(), frozenset({"kis_investor_flow"})])
def test_filtered_empty_collection_phase_does_not_block(
    tmp_path: Path, selected: frozenset[str],
) -> None:
    ctx = _fixture_context(tmp_path)
    calls: list[str] = []

    def _kis(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        calls.append("kis_investor_flow")
        assert inner_ctx.registry.require("investor_flow_ls")
        return CollectionStepReport("kis_investor_flow", "complete", 0, 0, 0)

    report = refresh_scope(
        ctx, collect=True, dry_run=False, emit=lambda payload: None,
        steps=(("kis_investor_flow", _kis),), collect_steps=selected,
    )
    assert report.status == "complete"
    assert report.blocking_job is None
    assert report.built == EXPECTED_KINDS
    assert calls == (["kis_investor_flow"] if selected else [])


@pytest.mark.parametrize("bridge_present", [False, True])
def test_facts_preview_matches_build_with_optional_bridge(
    tmp_path: Path, bridge_present: bool,
) -> None:
    from dataclasses import replace

    from src.data import pipeline_graph as graph

    ctx = _fixture_context(tmp_path)
    if not bridge_present:
        root = ctx.runtime.workspace.bronze_root
        (root / "dart_corp_codes").rename(root / "unused_bridge")
    built = graph._build_financial_facts(ctx, {})
    later = replace(ctx, decision_time=ctx.decision_time + timedelta(days=1))
    assert graph._expected_matches_registered(graph._preview_financial_facts(later), built.path)


def test_facts_preview_skips_symlinked_disclosures(tmp_path: Path) -> None:
    from src.data import pipeline_graph as graph
    from src.data.datasets import dataset_digest, load_manifest

    ctx = _fixture_context(tmp_path)
    silver = ctx.runtime.workspace.silver_root
    candidate = next(path for path in silver.glob("disclosures_*") if "invalid" not in path.name)
    target = tmp_path / candidate.name
    candidate.rename(target)
    candidate.symlink_to(target, target_is_directory=True)
    built = graph._build_financial_facts(ctx, {})
    preview = graph._preview_financial_facts(ctx)
    assert preview.inputs["disclosures"] == dataset_digest([])
    assert preview.inputs == load_manifest(built.path).inputs


def test_facts_preview_rejects_unreadable_present_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import incremental_normalization as inc
    from src.data import pipeline_graph as graph

    ctx = _fixture_context(tmp_path)

    def _unreadable(**kwargs: object) -> object:
        raise PITDataError("bridge unreadable")

    monkeypatch.setattr(inc, "load_frozen_dart_ticker_bridge", _unreadable)
    with pytest.raises(PITDataError, match="bridge unreadable"):
        graph._preview_financial_facts(ctx)
def test_deferred_krx_nodes_are_not_scheduled(tmp_path: Path) -> None:
    from dataclasses import replace

    ctx = build_refresh_context(_scope_runtime(tmp_path))
    deferred = frozenset({"daily_market", "ordinary_universe", "hedge_series"})
    plan = plan_refresh(replace(ctx, deferred_build_kinds=deferred))
    assert not deferred.intersection(node.kind for node in plan)
    assert "financial_facts" in {node.kind for node in plan}


def _ok_step(name: str, status: str = "complete"):
    def _step(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        return CollectionStepReport(job=name, status=status, done=1, pending_left=0, requests_used=0)

    return (name, _step)


def test_optional_failure_does_not_block(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)
    events: list[object] = []
    report = run_collection_jobs(
        ctx, dry_run=False, emit=events.append,
        steps=(_ok_step("dart_disclosures"), _ok_step("kind_notice_search", "blocked"), _ok_step("ls_investor_flow")),
    )
    assert report.blocking_job is None
    assert report.degraded_jobs == ("kind_notice_search",)
    assert report.complete is True


@pytest.mark.parametrize("job", ["kind_notice_search", "kind_notice_documents"])
@pytest.mark.parametrize("error_type", [PITDataError, RuntimeError, OSError])
def test_optional_exception_is_contained(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, job: str, error_type: type[Exception],
) -> None:
    ctx = build_refresh_context(_scope_runtime(tmp_path))
    ran: list[str] = []

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise error_type("kind 403")

    def _after(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        ran.append("ls_investor_flow")
        return CollectionStepReport(job="ls_investor_flow", status="complete", done=1, pending_left=0, requests_used=0)

    events: list[object] = []
    report = run_collection_jobs(
        ctx, dry_run=False, emit=events.append,
        steps=(_ok_step("dart_disclosures"), (job, _boom), ("ls_investor_flow", _after)),
    )
    assert ran == ["ls_investor_flow"]
    assert report.steps[1] == CollectionStepReport(job, "failed", 0, 0, 0)
    assert events[1] == {
        "type": "collection", "job": job, "status": "failed",
        "done": 0, "pending_left": 0, "requests_used": 0,
    }
    assert f"job={job} status=failed error={error_type.__name__}" in caplog.text
    assert report.blocking_job is None
    assert report.degraded_jobs == (job,)


def test_required_exception_becomes_failed_and_continues(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)
    ran: list[str] = []

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise PITDataError("ls down")

    def _after(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        ran.append("ls_investor_flow")
        return CollectionStepReport(job="ls_investor_flow", status="complete", done=1, pending_left=0, requests_used=0)

    report = run_collection_jobs(
        ctx, dry_run=False, emit=lambda payload: None,
        steps=(("dart_disclosures", _boom), ("ls_investor_flow", _after)),
    )
    assert ran == ["ls_investor_flow"]
    assert [step.status for step in report.steps] == ["failed", "complete"]
    assert report.blocking_job == "dart_disclosures"
    assert report.complete is False


def test_required_block_still_blocks(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)
    report = run_collection_jobs(
        ctx, dry_run=False, emit=lambda payload: None,
        steps=(_ok_step("dart_disclosures", "pending"), _ok_step("kind_notice_search", "blocked")),
    )
    assert report.blocking_job == "dart_disclosures"
    assert report.degraded_jobs == ("kind_notice_search",)
    assert report.complete is False


def test_refresh_runs_kis_and_builds_when_kind_degraded(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)
    kis_ran: list[str] = []

    def _kis(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        kis_ran.append("kis_investor_flow")
        return CollectionStepReport(job="kis_investor_flow", status="complete", done=1, pending_left=0, requests_used=0)

    report = refresh_scope(
        ctx, collect=True, dry_run=False, emit=lambda payload: None,
        steps=(
            _ok_step("dart_disclosures"), _ok_step("kind_notice_search", "blocked"),
            _ok_step("kind_notice_documents"), _ok_step("ls_investor_flow"), ("kis_investor_flow", _kis),
        ),
    )
    assert report.status == "complete"
    assert report.built == EXPECTED_KINDS
    assert kis_ran == ["kis_investor_flow"]
    assert "kind_notice_search" in tuple(report.degraded_jobs)
    assert report.blocking_job is None


def test_optional_disclosure_coverage_exception_is_failed(tmp_path: Path) -> None:
    from src.data.jobs.dart import DisclosureCoverageError

    ctx = build_refresh_context(_scope_runtime(tmp_path))

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise DisclosureCoverageError("coverage gap", periods=("2025Q4",))

    report = run_collection_jobs(
        ctx, dry_run=False, emit=lambda payload: None, steps=(("kind_notice_search", _boom),),
    )
    assert report.steps == (CollectionStepReport("kind_notice_search", "failed", 0, 0, 0),)
    assert report.complete


@pytest.mark.parametrize("has_kis", [False, True])
@pytest.mark.parametrize("blocking", [None, "dart_disclosures", "kis_investor_flow"])
def test_refresh_preserves_degradation_across_report_paths(
    tmp_path: Path, has_kis: bool, blocking: str | None,
) -> None:
    ctx = _fixture_context(tmp_path)
    steps = (
        _ok_step("dart_disclosures", "pending" if blocking == "dart_disclosures" else "complete"),
        _ok_step("kind_notice_search", "blocked"),
        _ok_step("kind_notice_documents", "pending"),
    )
    if has_kis:
        steps += (_ok_step("kis_investor_flow", "pending" if blocking == "kis_investor_flow" else "complete"),)
    report = refresh_scope(ctx, collect=True, dry_run=False, emit=lambda payload: None, steps=steps)
    expected_block = blocking if has_kis or blocking != "kis_investor_flow" else None
    assert report.blocking_job == expected_block
    assert report.status == ("blocked" if expected_block else "complete")
    assert report.degraded_jobs == ("kind_notice_search", "kind_notice_documents")
    assert [(step["job"], step["status"]) for step in report.collection if step["job"] in report.degraded_jobs] == [
        ("kind_notice_search", "blocked"), ("kind_notice_documents", "pending"),
    ]


def test_keyboard_interrupt_propagates(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt):
        run_collection_jobs(
            ctx, dry_run=False, emit=lambda payload: None,
            steps=(("kind_notice_search", _boom),),
        )


def test_warm_daily_flags_fills_and_reuses_shared_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from src.core.time import KRX_TZ, SessionCalendar
    from src.data import market_actions, pipeline_graph as graph

    ctx = build_refresh_context(_scope_runtime(tmp_path), decision_time=DECISION_TIME)
    days = (date(2025, 12, 1), date(2025, 12, 2), date(2025, 12, 3))
    calendar = SessionCalendar(tuple(datetime(day.year, day.month, day.day, 9, tzinfo=KRX_TZ) for day in days))
    monkeypatch.setattr(graph, "_build_calendar", lambda ctx: calendar)
    ctx.runtime.workspace.bronze_root.mkdir(parents=True, exist_ok=True)
    for day in days:
        payload = ctx.runtime.workspace.bronze_root / f"daily-{day}.json"
        payload.write_text(json.dumps({"records": [{
            "ISU_CD": "005930", "SECT_TP_NM": "일반" if day == days[0] else "관리종목(소속부없음)",
        }]}))
        _publish_catalog_entry(
            ctx.catalog, source="krx_daily_market", key=day.isoformat(), as_of=day,
            payload_path=payload, retrieved_at=DECISION_TIME,
        )
    bronze_before = {path: path.read_bytes() for path in ctx.runtime.workspace.bronze_root.rglob("*") if path.is_file()}
    assert graph.warm_daily_flag_cache(ctx) == 3
    cache = ctx.runtime.workspace.state_root / "cache" / "daily_admin_flags.parquet"
    assert cache.is_file()
    assert list(ctx.runtime.workspace.silver_root.iterdir()) == []
    assert dict(ctx.registry.snapshot()) == {}
    assert bronze_before == {path: path.read_bytes() for path in ctx.runtime.workspace.bronze_root.rglob("*") if path.is_file()}
    stamp = cache.stat().st_mtime_ns

    def unexpected_parse(path: Path) -> bytes:
        pytest.fail("covered daily payload must be served from the warmed cache")

    monkeypatch.setattr(market_actions, "read_payload", unexpected_parse)
    assert graph.warm_daily_flag_cache(ctx) == 3
    assert cache.stat().st_mtime_ns == stamp
    monkeypatch.setattr("src.data.jobs.universe.read_corp_code_bridge", lambda catalog: ({}, "bridge"))
    monkeypatch.setattr("src.config.load_provider_policy", lambda config: SimpleNamespace(
        kind=SimpleNamespace(search_keywords=(), max_stale_sessions=0),
    ))
    built = graph._build_market_actions(ctx, {})
    assert built.path.is_dir()
    assert built.rows == 1
    assert cache.stat().st_mtime_ns == stamp


@pytest.mark.parametrize("error_type", [PITDataError, OSError])
def test_warm_daily_flags_propagates_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception],
) -> None:
    from src.data import pipeline_graph as graph

    ctx = build_refresh_context(_scope_runtime(tmp_path), decision_time=DECISION_TIME)

    def fail(ctx: RefreshContext) -> object:
        raise error_type("calendar unavailable")

    monkeypatch.setattr(graph, "_build_calendar", fail)
    with pytest.raises(error_type, match="calendar unavailable"):
        graph.warm_daily_flag_cache(ctx)


def _fake_proc_root(tmp_path: Path, *, status: str | None) -> Path:
    root = tmp_path / "proc"
    (root / "self").mkdir(parents=True)
    if status is not None:
        (root / "self" / "status").write_text(status, encoding="utf-8")
    return root


def test_node_peak_parses_vmhwm(tmp_path: Path) -> None:
    from src.data import pipeline_graph as graph

    root = _fake_proc_root(tmp_path, status="Name:\tpython\nVmHWM:\t    2048 kB\nVmRSS:\t    1024 kB\n")
    assert graph.node_peak_rss_mb(proc_root=root) == 2.0


def test_node_peak_missing_field_is_none(tmp_path: Path) -> None:
    from src.data import pipeline_graph as graph

    root = _fake_proc_root(tmp_path, status="Name:\tpython\nVmRSS:\t    1024 kB\n")
    assert graph.node_peak_rss_mb(proc_root=root) is None
    assert graph.node_peak_rss_mb(proc_root=tmp_path / "no-proc") is None
    assert graph.node_peak_rss_mb(proc_root=_fake_proc_root(tmp_path / "bare", status="VmHWM:\n")) is None
    assert graph.node_peak_rss_mb(proc_root=_fake_proc_root(tmp_path / "bad", status="VmHWM:\t oops kB\n")) is None


def test_reset_failure_is_swallowed(tmp_path: Path) -> None:
    from src.data import pipeline_graph as graph

    root = tmp_path / "proc-ro"
    (root / "self" / "clear_refs").mkdir(parents=True)
    assert graph.reset_peak_rss(proc_root=root) is False
    assert graph.reset_peak_rss(proc_root=tmp_path / "no-proc") is False


@pytest.mark.parametrize("value", ["nan kB", "inf kB", "-1 kB", "2048 MB", "2048", "9" * 400 + " kB"])
def test_node_peak_rejects_invalid_measurements(tmp_path: Path, value: str) -> None:
    from src.data import pipeline_graph as graph

    root = _fake_proc_root(tmp_path, status=f"VmHWM: {value}\n")
    assert graph.node_peak_rss_mb(proc_root=root) is None


def test_proc_metrics_swallow_invalid_paths() -> None:
    from src.data import pipeline_graph as graph

    root = Path("\0")
    assert graph.node_peak_rss_mb(proc_root=root) is None
    assert graph.reset_peak_rss(proc_root=root) is False


def test_reset_writes_kernel_watermark_command(tmp_path: Path) -> None:
    from src.data import pipeline_graph as graph

    root = _fake_proc_root(tmp_path, status=None)
    target = root / "self" / "clear_refs"
    target.write_text("0", encoding="ascii")
    assert graph.reset_peak_rss(proc_root=root) is True
    assert target.read_bytes() == b"5"


def test_report_carries_peaks_for_built_kinds_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polars as pl

    import src.data.pipeline_graph as graph
    from src.data.datasets import DatasetIdentity, DatasetLayer, PublishedDataset, publish_dataset

    ctx = build_refresh_context(_scope_runtime(tmp_path), decision_time=DECISION_TIME)
    identity_a = DatasetIdentity(kind="a", layer=DatasetLayer.SILVER, policy_version="v1", inputs={}, params={})
    published_a = publish_dataset(
        layer_root=ctx.runtime.workspace.silver_root, identity=identity_a,
        partitions={"part.parquet": pl.DataFrame({"a": [1]})},
    )
    ctx.registry.register("a", published_a.path.name)
    identity_b = DatasetIdentity(kind="b", layer=DatasetLayer.SILVER, policy_version="v1", inputs={}, params={})

    def _fake_build(inner_ctx: RefreshContext, inputs: object) -> PublishedDataset:
        result = publish_dataset(
            layer_root=inner_ctx.runtime.workspace.silver_root, identity=identity_b,
            partitions={"part.parquet": pl.DataFrame({"b": [1]})},
        )
        return PublishedDataset(dataset_id=result.dataset_id, path=result.path, rows=result.rows)

    patched = (
        BuildNode(kind="a", inputs=(), bronze_sources=(), build=lambda c, i: pytest.fail("fresh node must not build")),
        BuildNode(kind="b", inputs=(), bronze_sources=(), build=_fake_build),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(graph, "SCOPE_GRAPH", patched)
    monkeypatch.setattr(graph, "_VALIDATED_GRAPH", patched)
    monkeypatch.setitem(graph._PREVIEW_FNS, "a", lambda c: identity_a)
    monkeypatch.setitem(graph._PREVIEW_FNS, "b", lambda c: identity_b)
    monkeypatch.setattr(graph, "reset_peak_rss", lambda **k: True)
    monkeypatch.setattr(graph, "node_peak_rss_mb", lambda **k: 7.5)
    events: list[object] = []
    report = graph.run_refresh(ctx, dry_run=False, emit=events.append)
    assert report.built == ("b",)
    assert dict(report.node_peak_rss_mb) == {"b": 7.5}
    payloads = [item for item in events if isinstance(item, dict) and item.get("type") == "dataset"]
    assert payloads[0]["peak_rss_mb"] == 7.5


def test_unavailable_peak_omits_payload_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.pipeline_graph as graph
    from src.data.datasets import PublishedDataset

    ctx = build_refresh_context(_scope_runtime(tmp_path), decision_time=DECISION_TIME)
    target = ctx.runtime.workspace.silver_root / "solo_ds"
    target.mkdir(parents=True, exist_ok=True)
    node = BuildNode(
        kind="solo", inputs=(), bronze_sources=(),
        build=lambda c, i: PublishedDataset(dataset_id="solo_ds", path=target, rows=1),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(graph, "_build_with_peak", lambda n, c, i: (n.build(c, i), None))
    monkeypatch.setattr(graph, "verify_dataset", lambda path, known_ids=None: type("V", (), {"passed": True, "failures": []})())
    events: list[object] = []
    peaks: dict[str, float] = {}
    graph._build_nodes(ctx, (node,), {}, {}, set(), events.append, peaks)
    assert peaks == {}
    assert events == [{"type": "dataset", "kind": "solo", "dataset_id": "solo_ds", "status": "built"}]


def test_failed_build_records_no_peak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.pipeline_graph as graph

    ctx = build_refresh_context(_scope_runtime(tmp_path), decision_time=DECISION_TIME)
    monkeypatch.setattr(graph, "reset_peak_rss", lambda **k: True)

    def _boom_peak(**kwargs: object) -> float:
        pytest.fail("peak must not be read when the build raises")
        return 0.0

    monkeypatch.setattr(graph, "node_peak_rss_mb", _boom_peak)
    events: list[object] = []
    peaks: dict[str, float] = {}

    def _boom(inner_ctx: RefreshContext, inputs: object) -> object:
        raise PITDataError("boom")

    with pytest.raises(PITDataError, match="boom"):
        graph._build_nodes(
            ctx,
            (BuildNode(kind="x", inputs=(), bronze_sources=(), build=_boom),),  # type: ignore[arg-type]
            {}, {}, set(), events.append, peaks,
        )
    assert peaks == {}
    assert events == []


@pytest.mark.parametrize("kis_status", ["complete", "budget_exhausted"])
@pytest.mark.parametrize("reset_supported", [True, False])
def test_evening_split_report_merges_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    kis_status: str, reset_supported: bool,
) -> None:
    import logging
    from collections.abc import Mapping

    import polars as pl

    import src.data.pipeline_graph as graph
    from src.data.datasets import DatasetIdentity, DatasetLayer, PublishedDataset, publish_dataset

    ctx = build_refresh_context(_scope_runtime(tmp_path), decision_time=DECISION_TIME)
    values = {"ordinary_universe": 7.5, "investor_flow_ls": 2.25, "investor_flow": 3.5}
    calls: list[str] = []
    current_kind = ""

    def reset() -> bool:
        calls.append("reset")
        return reset_supported

    def peak() -> float:
        calls.append("read")
        return values[current_kind]

    def build(kind: str):
        def run(inner_ctx: RefreshContext, inputs: Mapping[str, str]) -> PublishedDataset:
            nonlocal current_kind
            current_kind = kind
            calls.append(f"build:{kind}")
            result = publish_dataset(
                layer_root=inner_ctx.runtime.workspace.silver_root,
                identity=identities[kind],
                partitions={"part.parquet": pl.DataFrame({"value": [1]})},
            )
            return PublishedDataset(dataset_id=result.dataset_id, path=result.path, rows=result.rows)

        return run

    identities = {
        kind: DatasetIdentity(kind=kind, layer=DatasetLayer.SILVER, policy_version="v1", inputs={}, params={})
        for kind in values
    }
    nodes = tuple(BuildNode(kind=kind, inputs=(), bronze_sources=(), build=build(kind)) for kind in values)
    monkeypatch.setattr(graph, "_VALIDATED_GRAPH", nodes)
    for kind in values:
        monkeypatch.setitem(graph._PREVIEW_FNS, kind, lambda c, kind=kind: identities[kind])
    monkeypatch.setattr(graph, "reset_peak_rss", reset)
    monkeypatch.setattr(graph, "node_peak_rss_mb", peak)

    def collect_ls(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        calls.append("collect:ls")
        return CollectionStepReport("ls_investor_flow", "complete", 1, 0, 0)

    def collect_kis(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        assert inner_ctx.registry.require("investor_flow_ls")
        calls.append("collect:kis")
        return CollectionStepReport("kis_investor_flow", kis_status, 1, 0, 0)

    events: list[Mapping[str, object]] = []
    with caplog.at_level(logging.INFO, logger=graph.__name__):
        report = graph.refresh_scope(
            ctx, collect=True, dry_run=False, emit=events.append,
            steps=(("ls_investor_flow", collect_ls), ("kis_investor_flow", collect_kis)),
        )
    expected = values if kis_status == "complete" else {kind: values[kind] for kind in tuple(values)[:2]}
    assert report.status == ("complete" if kis_status == "complete" else "blocked")
    assert report.built == tuple(expected)
    assert dict(report.node_peak_rss_mb) == expected
    assert set(ctx.registry.snapshot()) == set(expected)
    assert calls == [
        "collect:ls", "reset", "build:ordinary_universe", "read",
        "reset", "build:investor_flow_ls", "read", "collect:kis",
        *(["reset", "build:investor_flow", "read"] if kis_status == "complete" else []),
    ]
    payloads = [item for item in events if item["type"] == "dataset"]
    assert {item["kind"]: item["peak_rss_mb"] for item in payloads} == expected
    for payload in payloads:
        assert list(payload) == ["type", "kind", "dataset_id", "status", "peak_rss_mb"]
    for kind, mb in expected.items():
        assert f"kind={kind} dataset={report.datasets[kind]} status=built peak_rss_mb={int(mb)}" in caplog.text


def test_kind_step_uses_relay_when_env_selects_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data.kind_relay_import as relay_import
    import src.integrations.krx.kind as kind_module
    from src.data.kind_relay_import import RelayImportReport
    from src.data.pipeline_graph import _run_kind_collection_step

    ctx = build_refresh_context(_scope_runtime(tmp_path))
    monkeypatch.setenv("KSE_KIND_TRANSPORT", "relay")
    calls: list[tuple[str, bool]] = []

    def _fake_import(*, runtime, provider, job, now, dry_run):  # type: ignore[no-untyped-def]
        calls.append((job, dry_run))
        return RelayImportReport(
            job=job, status="complete", imported=2, unchanged=1,
            rejected=0, pending_left=0, fetched_at=now,
        )

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("KindClient must not be constructed in relay mode")

    monkeypatch.setattr(relay_import, "import_relay_bundle", _fake_import)
    monkeypatch.setattr(kind_module, "build_scoped_kind_client", _boom)
    events: list[object] = []
    report = _run_kind_collection_step(ctx, "kind_notice_search", dry_run=False, emit=events.append)

    assert calls == [("kind_notice_search", False)]
    assert report == CollectionStepReport("kind_notice_search", "complete", 3, 0, 0)
    (event,) = [item for item in events if isinstance(item, dict) and item.get("type") == "collection"]
    assert (event["imported"], event["unchanged"], event["rejected"]) == (2, 1, 0)
    assert isinstance(event["fetched_at"], str)


def test_kind_step_direct_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.kind_relay_import as relay_import
    import src.integrations.krx.kind as kind_module
    from src.data.pipeline_graph import _run_kind_collection_step

    ctx = build_refresh_context(_scope_runtime(tmp_path))
    monkeypatch.delenv("KSE_KIND_TRANSPORT", raising=False)

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("relay importer must not run on the direct path")

    def _client_boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("no live client is built for a dry run")

    monkeypatch.setattr(relay_import, "import_relay_bundle", _boom)
    monkeypatch.setattr(kind_module, "build_scoped_kind_client", _client_boom)
    events: list[object] = []
    report = _run_kind_collection_step(ctx, "kind_notice_search", dry_run=True, emit=events.append)

    assert report.job == "kind_notice_search"
    assert report.status == "dry_run"
    assert report.requests_used == 0


def test_invalid_transport_degrades_not_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import pipeline_graph as graph

    ctx = build_refresh_context(_scope_runtime(tmp_path))
    monkeypatch.setenv("KSE_KIND_TRANSPORT", "bogus")
    steps = (
        (
            "kind_notice_search",
            lambda inner_ctx, *, dry_run, emit: graph._run_kind_collection_step(
                inner_ctx, "kind_notice_search", dry_run=dry_run, emit=emit
            ),
        ),
        (
            "kind_notice_documents",
            lambda inner_ctx, *, dry_run, emit: graph._run_kind_collection_step(
                inner_ctx, "kind_notice_documents", dry_run=dry_run, emit=emit
            ),
        ),
    )
    report = run_collection_jobs(ctx, dry_run=False, emit=lambda payload: None, steps=steps)

    assert [step.status for step in report.steps] == ["failed", "failed"]
    assert report.blocking_job is None
    assert report.degraded_jobs == ("kind_notice_search", "kind_notice_documents")


def test_dbsec_failure_is_optional(tmp_path: Path) -> None:
    from src.integrations.errors import ProviderTerminalError

    ctx = _fixture_context(tmp_path)
    ran: list[str] = []

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise ProviderTerminalError("DBSEC down", provider="DBSEC", endpoint="kr-chart/min")

    def _after(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        ran.append("kis_investor_flow")
        return CollectionStepReport(job="kis_investor_flow", status="complete", done=1, pending_left=0, requests_used=0)

    report = run_collection_jobs(
        ctx, dry_run=False, emit=lambda payload: None,
        steps=(_ok_step("dart_disclosures"), ("dbsec_closing_bar", _boom), ("kis_investor_flow", _after)),
    )

    assert ran == ["kis_investor_flow"]
    assert report.steps[1].status == "failed"
    assert report.blocking_job is None
    assert report.degraded_jobs == ("dbsec_closing_bar",)


def test_default_collection_steps_end_with_dbsec_in_registry_order(tmp_path: Path) -> None:
    from src.data import pipeline_graph as graph

    names = [name for name, _ in graph._default_collection_steps()]

    assert names[-3:] == ["dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar"]
    assert "dbsec_ticker_master" in graph.OPTIONAL_COLLECTION_JOBS


def test_ls_build_passes_configured_bucket_count_and_keeps_preview_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.data.investor_flow_silver as flow
    import src.data.pipeline_graph as graph

    ctx = _fixture_context(tmp_path)
    universe = graph._build_ordinary_universe(ctx, {})
    inputs = {"ordinary_universe": universe.dataset_id}
    ctx.registry.register("ordinary_universe", universe.dataset_id)
    monkeypatch.setattr(
        flow, "load_investor_flow_build_config", lambda path=None: flow.InvestorFlowBuildConfig(ticker_buckets=3)
    )
    seen: dict[str, object] = {}
    original = flow.materialize_investor_flow_silver

    def _spy(**kwargs: object):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(flow, "materialize_investor_flow_silver", _spy)
    result = graph._build_investor_flow_ls(ctx, inputs)
    assert seen.get("ticker_buckets") == 3
    from src.data.datasets import dataset_id_for

    assert result.dataset_id == dataset_id_for(graph._preview_investor_flow_ls(ctx))


def test_required_exception_emits_optional_shaped_event_and_traceback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    ctx = _fixture_context(tmp_path)

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise RuntimeError("dart down")

    events: list[object] = []
    with caplog.at_level(logging.ERROR, logger="src.data.pipeline_graph"):
        report = run_collection_jobs(
            ctx, dry_run=False, emit=events.append,
            steps=(("dart_disclosures", _boom), _ok_step("ls_investor_flow")),
        )
    assert events[0] == {
        "type": "collection", "job": "dart_disclosures", "status": "failed",
        "done": 0, "pending_left": 0, "requests_used": 0,
    }
    assert report.blocking_job == "dart_disclosures"
    record = next(record for record in caplog.records if record.levelno >= logging.ERROR)
    assert "job=dart_disclosures status=failed error=RuntimeError" in record.getMessage()
    assert record.exc_info is not None


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_required_keyboard_interrupt_propagates(tmp_path: Path, error_type: type[BaseException]) -> None:
    ctx = _fixture_context(tmp_path)

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise error_type("stop")

    with pytest.raises(error_type):
        run_collection_jobs(
            ctx, dry_run=False, emit=lambda payload: None,
            steps=(("dart_disclosures", _boom),),
        )


def test_required_coverage_gap_continues(tmp_path: Path) -> None:
    from src.data.jobs.dart import DisclosureCoverageError

    ctx = _fixture_context(tmp_path)
    ran: list[str] = []

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise DisclosureCoverageError("gap", periods=("2025Q4",))

    def _after(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        ran.append("ls_investor_flow")
        return CollectionStepReport(job="ls_investor_flow", status="complete", done=1, pending_left=0, requests_used=0)

    events: list[object] = []
    report = run_collection_jobs(
        ctx, dry_run=False, emit=events.append,
        steps=(("dart_disclosures", _boom), ("ls_investor_flow", _after)),
    )
    assert ran == ["ls_investor_flow"]
    assert report.steps[0].status == "coverage_gap"
    assert report.blocking_job == "dart_disclosures"
    assert events[0]["periods"] == ["2025Q4"]  # type: ignore[index]


def test_gate_table_covers_every_node_and_only_real_steps() -> None:
    from src.data import pipeline_graph as graph

    kinds = {node.kind for node in graph.SCOPE_GRAPH}
    assert set(graph.NODE_COLLECTION_GATES) == kinds
    real_steps = {name for name, _ in graph._default_collection_steps()}
    for kind, gate in graph.NODE_COLLECTION_GATES.items():
        assert gate <= real_steps, kind
        assert not (gate & graph.OPTIONAL_COLLECTION_JOBS), kind
    assert {
        "dart_corp_codes", "dart_disclosures", "dart_facts", "dividend_decisions", "earnings_releases",
    } == graph.DART_COLLECTION_STEPS


def test_phase_split_on_real_graph() -> None:
    from src.data import pipeline_graph as graph

    plan = tuple(graph.SCOPE_GRAPH)
    first, flow_done, rest = graph._phase_split(plan)
    assert {node.kind for node in first} == {"ordinary_universe", "daily_market", "investor_flow_ls"}
    assert {node.kind for node in flow_done} == {"investor_flow_kis_supplement", "investor_flow"}
    assert {node.kind for node in first} | {node.kind for node in flow_done} | {node.kind for node in rest} == {
        node.kind for node in plan
    }
    order = [node.kind for node in graph.SCOPE_GRAPH]
    for group in (first, flow_done, rest):
        assert [node.kind for node in group] == sorted([node.kind for node in group], key=order.index)
    dart_kinds = {
        "financial_facts", "financial_quality", "dividend_events", "earnings_releases",
        "market_actions", "market_panel", "reference_benchmarks",
    }
    assert not (dart_kinds & ({node.kind for node in first} | {node.kind for node in flow_done}))


def test_blocked_kinds_transitive_and_local() -> None:
    from src.data import pipeline_graph as graph

    plan = tuple(graph.SCOPE_GRAPH)
    assert graph._blocked_kinds(plan, frozenset({"dart_disclosures"})) == frozenset({
        "financial_facts", "financial_quality", "dividend_events", "earnings_releases",
        "market_actions", "market_panel", "reference_benchmarks",
    })
    assert graph._blocked_kinds(plan, frozenset({"krx_hedge_series"})) == frozenset({"hedge_series"})
    assert graph._blocked_kinds(plan, frozenset()) == frozenset()
    assert graph._blocked_kinds(plan, frozenset({"kind_notice_search"})) == frozenset()


def _refresh_steps(names: list[str], statuses: dict[str, str] | None = None) -> tuple[CollectionStep, ...]:
    statuses = statuses or {}
    return tuple(_ok_step(name, statuses.get(name, "complete")) for name in names)


def _dart_blocked_invariants(report: RefreshReport, *, blocking: str) -> None:
    assert report.status == "blocked"
    assert report.blocking_job == blocking
    assert "investor_flow" in report.built
    assert "financial_facts" not in report.built


def test_dart_exception_still_collects_kis(tmp_path: Path) -> None:
    from src.data import pipeline_graph as graph

    ctx = _fixture_context(tmp_path)
    calls: list[str] = []
    real_builders = {node.kind: node.build for node in graph.SCOPE_GRAPH}

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise PITDataError("dart down")

    def _collect(name: str):
        def _step(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
            calls.append(f"collect:{name}")
            return CollectionStepReport(job=name, status="complete", done=1, pending_left=0, requests_used=0)

        return (name, _step)

    def _wrap(kind: str):
        def _build(inner_ctx: RefreshContext, inputs: object) -> object:
            calls.append(f"build:{kind}")
            return real_builders[kind](inner_ctx, inputs)

        return _build

    patched = tuple(
        graph.BuildNode(kind=node.kind, inputs=node.inputs, bronze_sources=node.bronze_sources, build=_wrap(node.kind))
        for node in graph.SCOPE_GRAPH
    )
    original_graph, original_validated = graph.SCOPE_GRAPH, graph._VALIDATED_GRAPH
    try:
        graph.SCOPE_GRAPH = patched  # type: ignore[assignment]
        graph._VALIDATED_GRAPH = patched  # type: ignore[assignment]
        steps = (
            ("dart_disclosures", _boom),
            _collect("krx_daily_market"), _collect("krx_security_master"),
            _collect("ls_investor_flow"), _collect("kis_investor_flow"),
        )
        report = graph.refresh_scope(ctx, collect=True, dry_run=False, emit=lambda payload: None, steps=steps)
    finally:
        graph.SCOPE_GRAPH = original_graph  # type: ignore[assignment]
        graph._VALIDATED_GRAPH = original_validated  # type: ignore[assignment]
    _dart_blocked_invariants(report, blocking="dart_disclosures")
    assert calls.index("collect:kis_investor_flow") > calls.index("build:investor_flow_ls")
    assert "build:financial_facts" not in calls


def test_dart_pending_still_collects_kis_and_skips_dart_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data.pipeline_graph as graph

    ctx = _built_fixture(tmp_path)
    before = dict(ctx.registry.snapshot())
    monkeypatch.setattr(graph, "_expected_matches_registered", lambda expected, path: False)
    built_spy: list[str] = []
    real_builders = {node.kind: node.build for node in graph.SCOPE_GRAPH}

    def _spy(kind: str):
        def _build(inner_ctx: RefreshContext, inputs: object) -> object:
            built_spy.append(kind)
            return real_builders[kind](inner_ctx, inputs)

        return _build

    patched = tuple(
        graph.BuildNode(kind=node.kind, inputs=node.inputs, bronze_sources=node.bronze_sources, build=_spy(node.kind))
        for node in graph.SCOPE_GRAPH
    )
    original_graph, original_validated = graph.SCOPE_GRAPH, graph._VALIDATED_GRAPH
    try:
        graph.SCOPE_GRAPH = patched  # type: ignore[assignment]
        graph._VALIDATED_GRAPH = patched  # type: ignore[assignment]
        steps = _refresh_steps(
            ["dart_disclosures", "dart_facts", "krx_daily_market", "krx_security_master",
             "ls_investor_flow", "kis_investor_flow"],
            {"dart_facts": "pending"},
        )
        events: list[object] = []
        report = graph.refresh_scope(ctx, collect=True, dry_run=False, emit=events.append, steps=steps)
    finally:
        graph.SCOPE_GRAPH = original_graph  # type: ignore[assignment]
        graph._VALIDATED_GRAPH = original_validated  # type: ignore[assignment]
    _dart_blocked_invariants(report, blocking="dart_facts")
    assert ctx.registry.snapshot()["investor_flow"] == report.datasets["investor_flow"]
    for kind in (
        "financial_facts", "financial_quality", "dividend_events", "earnings_releases",
        "market_actions", "market_panel", "reference_benchmarks",
    ):
        assert kind not in built_spy
        assert ctx.registry.snapshot()[kind] == before[kind]
    assert "hedge_series" in built_spy
    assert "industry" in built_spy
    skipped = {str(item.get("kind")) for item in events if isinstance(item, dict) and item.get("status") == "skipped_blocked"}
    assert skipped == {
        "financial_facts", "financial_quality", "dividend_events", "earnings_releases",
        "market_actions", "market_panel", "reference_benchmarks",
    }
    for item in events:
        if isinstance(item, dict) and item.get("status") == "skipped_blocked":
            assert item["blocking_job"] == "dart_facts"


def test_flow_chain_failure_keeps_blocked_before_build(tmp_path: Path) -> None:
    ctx = _built_fixture(tmp_path)

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise PITDataError("ls down")

    events: list[object] = []
    report = refresh_scope(
        ctx, collect=True, dry_run=False, emit=events.append,
        steps=(("ls_investor_flow", _boom), _ok_step("kis_investor_flow")),
    )
    assert report.status == "blocked"
    assert report.planned == ()
    assert report.built == ()
    assert report.blocking_job == "ls_investor_flow"
    assert not any(
        isinstance(event, dict) and event.get("job") == "kis_investor_flow" for event in events
    )


def test_kis_and_dart_blocking_precedence(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)
    steps = _refresh_steps(
        ["dart_facts", "krx_daily_market", "krx_security_master", "ls_investor_flow", "kis_investor_flow"],
        {"dart_facts": "pending", "kis_investor_flow": "budget_exhausted"},
    )
    report = refresh_scope(ctx, collect=True, dry_run=False, emit=lambda payload: None, steps=steps)
    assert report.status == "blocked"
    assert report.blocking_job == "dart_facts"
    assert report.built == ("ordinary_universe", "daily_market", "investor_flow_ls")
    assert set(ctx.registry.snapshot()) == set(report.built)


def test_complete_run_keeps_graph_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _fixture_context(tmp_path)
    calls: list[str] = []
    import src.data.pipeline_graph as graph

    registrations: list[tuple[str, ...]] = []
    original_register = DatasetRegistry.register_many

    def register(registry: DatasetRegistry, kinds: Mapping[str, str]) -> None:
        registrations.append(tuple(kinds))
        original_register(registry, kinds)

    monkeypatch.setattr(DatasetRegistry, "register_many", register)

    def emit(payload: Mapping[str, object]) -> None:
        if payload["type"] == "dataset":
            calls.append(f"build:{payload['kind']}")

    def _collect(name: str):
        def _step(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
            calls.append(f"collect:{name}")
            return CollectionStepReport(job=name, status="complete", done=1, pending_left=0, requests_used=0)

        return (name, _step)

    steps = (
        _collect("dart_disclosures"), _collect("dart_facts"), _collect("dividend_decisions"),
        _collect("krx_daily_market"), _collect("krx_security_master"),
        _collect("ls_investor_flow"), _collect("kis_investor_flow"),
    )
    report = graph.refresh_scope(ctx, collect=True, dry_run=False, emit=emit, steps=steps)
    assert report.status == "complete"
    assert report.built == EXPECTED_KINDS
    assert tuple(report.datasets) == EXPECTED_KINDS
    assert calls.index("collect:ls_investor_flow") < calls.index("collect:kis_investor_flow")
    phases = graph._phase_split(graph.SCOPE_GRAPH)
    assert registrations == [tuple(node.kind for node in phase) for phase in phases]
    assert calls.index("build:investor_flow_ls") < calls.index("collect:kis_investor_flow")
    for node in phases[2]:
        assert calls.index("build:investor_flow") < calls.index(f"build:{node.kind}")


def test_phase_c_raise_keeps_flow_registered(tmp_path: Path) -> None:
    import src.data.pipeline_graph as graph

    ctx = _fixture_context(tmp_path)
    calls: list[str] = []
    real_builders = {node.kind: node.build for node in graph.SCOPE_GRAPH}

    def _wrap(kind: str):
        def _build(inner_ctx: RefreshContext, inputs: object) -> object:
            calls.append(f"build:{kind}")
            if kind == "financial_facts":
                raise PITDataError("facts schema")
            return real_builders[kind](inner_ctx, inputs)

        return _build

    def _collect(name: str):
        def _step(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
            calls.append(f"collect:{name}")
            return CollectionStepReport(job=name, status="complete", done=1, pending_left=0, requests_used=0)

        return (name, _step)

    patched = tuple(
        graph.BuildNode(kind=node.kind, inputs=node.inputs, bronze_sources=node.bronze_sources, build=_wrap(node.kind))
        for node in graph.SCOPE_GRAPH
    )
    original_graph, original_validated = graph.SCOPE_GRAPH, graph._VALIDATED_GRAPH
    try:
        graph.SCOPE_GRAPH = patched  # type: ignore[assignment]
        graph._VALIDATED_GRAPH = patched  # type: ignore[assignment]
        steps = (
            _collect("dart_disclosures"), _collect("dart_facts"), _collect("dividend_decisions"),
            _collect("krx_daily_market"), _collect("krx_security_master"),
            _collect("ls_investor_flow"), _collect("kis_investor_flow"),
        )
        with pytest.raises(PITDataError, match="facts schema"):
            graph.refresh_scope(ctx, collect=True, dry_run=False, emit=lambda payload: None, steps=steps)
    finally:
        graph.SCOPE_GRAPH = original_graph  # type: ignore[assignment]
        graph._VALIDATED_GRAPH = original_validated  # type: ignore[assignment]
    assert calls.index("collect:kis_investor_flow") < calls.index("build:financial_facts")
    assert calls.index("build:investor_flow") < calls.index("build:financial_facts")
    snapshot = ctx.registry.snapshot()
    assert set(snapshot) == {
        "ordinary_universe", "daily_market", "investor_flow_ls", "investor_flow_kis_supplement", "investor_flow",
    }
    assert plan_refresh(ctx)
    assert "investor_flow" not in {node.kind for node in plan_refresh(ctx)}


def test_dry_run_ignores_failing_dart_gate(tmp_path: Path) -> None:
    ctx = _fixture_context(tmp_path)
    silver_before = sorted(ctx.runtime.workspace.silver_root.rglob("*"))

    def _boom(inner_ctx: RefreshContext, *, dry_run: bool, emit: object) -> CollectionStepReport:
        raise PITDataError("dart down")

    events: list[object] = []
    report = refresh_scope(
        ctx, collect=True, dry_run=True, emit=events.append,
        steps=(("dart_disclosures", _boom), _ok_step("ls_investor_flow"), _ok_step("kis_investor_flow")),
    )
    assert report.status == "dry_run"
    assert report.blocking_job is None
    assert not ctx.registry.snapshot()
    assert sorted(ctx.runtime.workspace.silver_root.rglob("*")) == silver_before
    assert not list(ctx.runtime.workspace.gold_root.iterdir())
    progress = [event.get("job", event.get("kind")) for event in events if isinstance(event, dict)]
    assert progress.index("investor_flow_ls") < progress.index("kis_investor_flow")
    assert progress.index("kis_investor_flow") < progress.index("investor_flow_kis_supplement")
    assert progress.index("investor_flow") < progress.index("financial_facts")


@pytest.mark.parametrize("job", ["dart_disclosures", "kind_notice_search"])
@pytest.mark.parametrize("from_step", [True, False])
def test_collection_emit_failure_propagates(
    tmp_path: Path, job: str, from_step: bool,
) -> None:
    ctx = build_refresh_context(_scope_runtime(tmp_path))
    failure = RuntimeError("progress sink unavailable")
    emissions: list[Mapping[str, object]] = []

    def emit(payload: Mapping[str, object]) -> None:
        emissions.append(payload)
        if len(emissions) == 1:
            raise failure

    def step(
        inner_ctx: RefreshContext, *, dry_run: bool, emit: Callable[[Mapping[str, object]], None],
    ) -> CollectionStepReport:
        if from_step:
            emit({"type": "progress", "job": job})
        return CollectionStepReport(job, "complete", 1, 0, 0)

    with pytest.raises(RuntimeError) as caught:
        run_collection_jobs(
            ctx, dry_run=False, emit=emit,
            steps=((job, step), ("ls_investor_flow", lambda *a, **k: pytest.fail("must stop"))),
        )
    assert caught.value is failure
    assert len(emissions) == 1


def test_phase_split_closes_inputs_and_tolerates_unknown_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.pipeline_graph as graph

    def unused(ctx: RefreshContext, inputs: Mapping[str, str]):
        pytest.fail("phase planning must not build")

    nodes = (
        BuildNode("custom_input", (), (), unused),
        BuildNode("ordinary_universe", ("custom_input",), (), unused),
        BuildNode("investor_flow_ls", ("ordinary_universe",), (), unused),
        BuildNode("investor_flow", ("investor_flow_ls",), (), unused),
        BuildNode("custom_rest", (), (), unused),
    )
    monkeypatch.setattr(graph, "_VALIDATED_GRAPH", nodes)
    assert graph._phase_split(nodes) == (nodes[:3], nodes[3:4], nodes[4:])
    assert graph._phase_split(nodes[2:4]) == (nodes[2:3], nodes[3:4], ())
    assert graph._blocked_kinds(nodes, frozenset({"krx_security_master"})) == {
        "ordinary_universe", "investor_flow_ls", "investor_flow",
    }
