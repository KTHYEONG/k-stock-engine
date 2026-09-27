"""Invariant guards for the declarative whole-scope refresh pipeline."""

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src.core.pit import PITDataError
from src.data.dataset_registry import DatasetRegistry
from src.data.pipeline_graph import (
    SCOPE_GRAPH,
    BuildNode,
    CollectionStepReport,
    RefreshContext,
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
    "dividend_events",
    "industry",
    "financial_facts",
    "daily_market",
    "investor_flow_ls",
    "financial_quality",
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
    return build_refresh_context(runtime, decision_time=DECISION_TIME)


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
    assert len(report.collection) == 8
    assert {str(step["job"]) for step in report.collection} == {
        "dart_corp_codes", "dart_disclosures", "dart_facts", "dividend_decisions",
        "krx_daily_market", "krx_security_master", "ls_investor_flow", "kis_investor_flow",
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
        ctx: RefreshContext, *, collect: bool, dry_run: bool, emit: object, steps: object = None
    ) -> object:
        assert collect
        assert not dry_run
        return real_refresh_scope(
            ctx, collect=False, dry_run=True, emit=emit,
        ).__class__(
            decision_time=DECISION_TIME, status="blocked", planned=(), built=(),
            datasets={}, blocking_job="krx_daily_market", collection=(),
        )

    monkeypatch.setattr(pipeline_graph, "refresh_scope", _blocked)
    assert cli_module.main([
        "refresh-scope",
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(tmp_path / "data"),
        "--collect",
    ]) == 1


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
    monkeypatch.setattr(inc, "_discover_fact_receipts", lambda _root: [{"content_hash": "a" * 64}, {"content_hash": "b" * 64}])

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
    assert pipeline_graph._phase_split((),) == ((), ())
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
    kinds = [str(item.get("kind")) for item in events if isinstance(item, dict) and item.get("type") == "plan"]
    assert "investor_flow_ls" in kinds
    jobs = [str(item.get("job")) for item in events if isinstance(item, dict) and item.get("type") == "collection"]
    assert jobs.index("ls_investor_flow") < jobs.index("kis_investor_flow")
