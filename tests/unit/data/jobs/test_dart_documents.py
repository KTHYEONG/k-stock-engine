"""Invariant scenarios for the DART document reparse and fetch jobs."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)
SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
CORP = "00126380"
CORP_OTHER = "00266961"
TICKER = "005930"
TICKER_OTHER = "000660"


def _runtime(tmp_path: Path):
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _provider():
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def _ctx(runtime, provider, *, collector=None, now=None):
    from src.data.jobs.runner import build_job_context

    return build_job_context(
        runtime=runtime, provider=provider, key_env=None, collector=collector, now=now or (lambda: NOW)
    )


def _bridge(bronze_root: Path, rows=None) -> None:
    from datetime import date as _date
    from datetime import datetime as _datetime
    from datetime import UTC as _UTC

    from src.core.pit import EvidenceKind as _Kind
    from src.data.bronze import BronzeStore as _Store
    from src.data.receipt_catalog import BlobEntry as _Blob, EvidenceStatus as _Status, ReceiptCatalog as _Catalog, ReceiptIndexEntry as _Entry

    rows = rows if rows is not None else [{"ticker": TICKER, "corp_code": CORP, "corp_name": "Test Co"}]
    raw = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    store = _Store(bronze_root)
    receipt = store.import_bytes(
        raw, kind=_Kind.SECURITY_MASTER, retrieved_at=_datetime(2026, 9, 24, 3, 0, tzinfo=_UTC),
        source_label="test:bridge",
    )
    catalog = _Catalog(bronze_root / "catalog")
    catalog.publish(
        [_Entry(source="dart_corp_codes", natural_key="dart_corp_codes", as_of=_date(2026, 9, 24),
                fiscal_period=None, status=_Status.SUCCESS, content_hash=receipt.content_hash,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path)],
        blobs=[_Blob(content_hash=receipt.content_hash, kind=_Kind.SECURITY_MASTER, source="dart_corp_codes",
                     usable=True, unusable_reason=None, retrieved_at=receipt.retrieved_at,
                     payload_path=receipt.payload_path)],
    )


def _universe_years(runtime, ticker_years: dict) -> None:
    import polars as pl

    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    rows = [
        {"session": date(int(year), 6, 30), "ticker": ticker, "eligible": True}
        for ticker, years in ticker_years.items()
        for year in years
    ]
    frame = pl.DataFrame(rows, schema={"session": pl.Date, "ticker": pl.String, "eligible": pl.Boolean})
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="ordinary_universe", layer=DatasetLayer.SILVER,
                                 policy_version="test-v1", inputs={}, params={}),
        partitions={"part.parquet": frame},
    )
    DatasetRegistry(runtime.workspace.state_root).register("ordinary_universe", published.dataset_id)


def _write_disclosure_page(bronze_root: Path, payload: dict) -> None:
    import hashlib

    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    target = bronze_root / "disclosures" / hashlib.sha256(raw).hexdigest() / "payload.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    moment = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)
    (target.parent / "receipt.json").write_text(json.dumps({"retrieved_at": moment.isoformat()}), encoding="utf-8")
    from datetime import date as _date

    from src.core.pit import EvidenceKind as _Kind
    from src.data.receipt_catalog import BlobEntry as _Blob, EvidenceStatus as _Status, ReceiptCatalog as _Catalog, ReceiptIndexEntry as _Entry

    catalog = _Catalog(bronze_root / "catalog")
    content_hash = hashlib.sha256(raw).hexdigest()
    corp = str(payload.get("corp_code") or "").strip()
    if corp:
        try:
            start = _date.fromisoformat(str(payload.get("start") or ""))
            end = _date.fromisoformat(str(payload.get("end") or ""))
        except ValueError:
            return
        key = f"{corp}:{start.isoformat()}..{end.isoformat()}"
        catalog.publish(
            [_Entry(source="dart_corp_disclosures", natural_key=key, as_of=end, fiscal_period=None,
                    status=_Status.SUCCESS if payload.get("records") else _Status.EMPTY,
                    content_hash=content_hash, retrieved_at=moment, payload_path=target)],
            blobs=[_Blob(content_hash=content_hash, kind=_Kind.DISCLOSURES, source="dart_corp_disclosures",
                         usable=True, unusable_reason=None, retrieved_at=moment, payload_path=target)],
        )


def _disclosure_record(rcept_no: str, rcept_dt: str, report_nm: str, corp: str = CORP) -> dict:
    return {"corp_code": corp, "corp_name": "Test Co", "rcept_no": rcept_no, "rcept_dt": rcept_dt, "report_nm": report_nm, "rm": ""}


def _persist_fact(ctx, page: dict, retrieved_at=None):
    from src.data.collection import dart_fact_scoped_payload

    moment = retrieved_at or NOW
    serializable = {key: value for key, value in dict(page).items() if key != "raw_archive"}
    return ctx.writer.persist(dart_fact_scoped_payload(page=serializable, retrieved_at=moment))


def _legacy_page(*, corp: str = CORP, biz_year: str = "2020", reprt_code: str = "11011",
                 filing_id: str = "20210330001234", raw_hash: str | None = None) -> dict:
    return {
        "source_kind": "legacy_document",
        "status": "013",
        "identity": {"corp_code": corp, "biz_year": biz_year, "reprt_code": reprt_code,
                     "filing_id": filing_id, "rcept_no": filing_id, "fs_div": "CFS", "published_at": "2020-03-30"},
        "records": [],
        "mapping_version": "dart-fact-map-v1",
        "diagnostics": ("legacy_fallback",),
        "raw_document_hash": raw_hash,
        "corp_code": corp, "biz_year": biz_year, "reprt_code": reprt_code,
        "filing_id": filing_id, "rcept_no": filing_id, "fs_div": "CFS", "published_at": "2020-03-30",
    }


def _store_archive(ctx, archive: bytes, rcept_no: str):
    from src.data.dart_documents import DartDocumentStore

    store = DartDocumentStore(ctx.runtime.workspace.bronze_root, catalog=ctx.catalog)
    return store.store_archive(archive, rcept_no=rcept_no, retrieved_at=NOW)


def _verified_statements():
    from datetime import date as _date

    from src.integrations.dart.document_statements import PeriodBasis, StatementFact, VerifiedStatements

    return VerifiedStatements(
        consolidated=True, period_end=_date(2020, 12, 31), report_kind="annual",
        unit_multipliers={"BS": 1},
        facts=(StatementFact(fact="assets", value=1000, basis=PeriodBasis.POINT_IN_TIME, label="자산총계"),),
        checks=("bs_balance",),
    )


def _setup_relevant(ctx, runtime, *, corp: str = CORP, ticker: str = TICKER, year: str = "2020") -> None:
    _write_disclosure_page(
        runtime.workspace.bronze_root,
        {"records": [_disclosure_record("20210330001234", "20210330", "사업보고서 (2020.12)", corp=corp)],
         "start": "2019-01-01", "end": "2021-12-31", "corp_code": corp},
    )


def _legacy_page_2020(*, corp: str = CORP, filing_id: str = "20210330001234", raw_hash: str | None = None) -> dict:
    return _legacy_page(corp=corp, biz_year="2020", reprt_code="11011", filing_id=filing_id, raw_hash=raw_hash)


def test_reparse_is_streaming_and_idempotent(tmp_path: Path, monkeypatch) -> None:
    from src.data.jobs.dart_documents import DartDocumentReparseJob
    from src.data.jobs.runner import run_job
    from src.integrations.dart.document_statements import DocumentParseResult

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe_years(runtime, {TICKER: {2020, 2021, 2022, 2023}})
    ctx = _ctx(runtime, _provider())
    archives = [f"PK-archive-{idx}".encode() for idx in range(3)]
    years = ["2020", "2021", "2022"]
    for idx, archive in enumerate(archives):
        receipt = _store_archive(ctx, archive, f"2020033{idx}00123{idx}")
        import hashlib

        digest = hashlib.sha256(archive).hexdigest()
        assert receipt.content_hash == digest
        _persist_fact(
            ctx,
            _legacy_page(biz_year=years[idx], filing_id=f"2020033{idx}00123{idx}", raw_hash=digest),
            retrieved_at=datetime(2026, 9, 23, 3, idx, tzinfo=UTC),
        )

    monkeypatch.setattr(
        "src.integrations.dart.document_statements.parse_filing_document",
        lambda archive, *, reprt_code, biz_year: DocumentParseResult(statements=_verified_statements(), diagnostics=()),
    )
    spec = DartDocumentReparseJob()
    assert len(spec.pending(ctx)) == 3

    consumed: list[str] = []
    real_entries = ctx.catalog.entries

    def _guarded_entries(*, source):
        for entry in real_entries(source=source):
            consumed.append(entry.natural_key)
            yield entry

    import types

    guarded = types.SimpleNamespace(
        entries=_guarded_entries,
        blobs=ctx.catalog.blobs,
        latest=ctx.catalog.latest,
    )
    guard_ctx = _ctx(runtime, _provider())
    object.__setattr__(guard_ctx, "catalog", guarded)
    consumed.clear()
    assert len(spec.pending(guard_ctx)) == 3
    assert set(consumed) == {"00126380:2020:11011", "00126380:2021:11011", "00126380:2022:11011"}

    emitted: list[dict] = []
    report = run_job(spec, ctx, chunk_size=10, max_chunks=None, dry_run=False, emit=emitted.append)
    assert report.done == 3
    assert spec.pending(ctx) == []


def test_reparse_skips_pages_without_an_archive(tmp_path: Path) -> None:
    from src.data.jobs.dart_documents import DartDocumentFetchJob, DartDocumentReparseJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe_years(runtime, {TICKER: {2020, 2021}})
    ctx = _ctx(runtime, _provider())
    _setup_relevant(ctx, runtime)
    _persist_fact(ctx, _legacy_page(filing_id="20210330001234", raw_hash="f" * 64))

    assert DartDocumentReparseJob().pending(ctx) == []
    assert len(DartDocumentFetchJob().pending(ctx)) == 1


def test_fetch_only_relevant_identities(tmp_path: Path) -> None:
    from src.data.jobs.dart_documents import DartDocumentFetchJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root, rows=[
        {"ticker": TICKER, "corp_code": CORP, "corp_name": "A"},
        {"ticker": TICKER_OTHER, "corp_code": CORP_OTHER, "corp_name": "B"},
    ])
    _universe_years(runtime, {TICKER: {2020, 2021}})
    ctx = _ctx(runtime, _provider())
    _write_disclosure_page(
        runtime.workspace.bronze_root,
        {"records": [_disclosure_record("20210330001234", "20210330", "사업보고서 (2020.12)", corp=CORP)],
         "start": "2019-01-01", "end": "2021-12-31", "corp_code": CORP},
    )
    _write_disclosure_page(
        runtime.workspace.bronze_root,
        {"records": [_disclosure_record("20210330009999", "20210330", "사업보고서 (2020.12)", corp=CORP_OTHER)],
         "start": "2019-01-01", "end": "2021-12-31", "corp_code": CORP_OTHER},
    )
    _persist_fact(ctx, _legacy_page(corp=CORP, filing_id="20210330001234", raw_hash=None))
    _persist_fact(ctx, _legacy_page(corp=CORP_OTHER, filing_id="20210330009999", raw_hash=None))

    units = DartDocumentFetchJob().pending(ctx)

    assert [unit.natural_key for unit in units] == [f"{CORP}:2020:11011"]


def test_fetch_stores_before_parsing(tmp_path: Path, monkeypatch) -> None:
    from src.data.dart_documents import DartDocumentStore
    from src.data.evidence_sources import DART_DOCUMENT_SOURCE
    from src.data.jobs.dart_documents import DartDocumentFetchJob
    from src.integrations.dart.document_statements import DocumentParseResult

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe_years(runtime, {TICKER: {2020, 2021}})
    ctx = _ctx(runtime, _provider())
    _setup_relevant(ctx, runtime)
    _persist_fact(ctx, _legacy_page(filing_id="20210330001234", raw_hash=None))

    archive = b"PK\x03\x04fetch-store-bytes"
    seen: dict[str, bool] = {}

    class _Collector:
        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return archive

        def health_check(self) -> None:
            return None

    fetch_ctx = _ctx(runtime, _provider(), collector=_Collector())
    (unit,) = DartDocumentFetchJob().pending(fetch_ctx)
    assert unit.max_requests == 1

    real_store = DartDocumentStore.store_archive

    def _spy_store(self, archive_bytes: bytes, *, rcept_no: str, retrieved_at):
        receipt = real_store(self, archive_bytes, rcept_no=rcept_no, retrieved_at=retrieved_at)
        seen["stored"] = True
        return receipt

    monkeypatch.setattr(DartDocumentStore, "store_archive", _spy_store)
    monkeypatch.setattr(
        "src.integrations.dart.document_statements.parse_filing_document",
        lambda data, *, reprt_code, biz_year: DocumentParseResult(statements=_verified_statements(), diagnostics=()),
    )
    (payload,) = DartDocumentFetchJob().fetch(fetch_ctx, [unit])

    assert seen.get("stored") is True
    import hashlib
    import json

    body = json.loads(payload.payload)
    assert body["raw_document_hash"] == hashlib.sha256(archive).hexdigest()
    blobs = list(fetch_ctx.catalog.blobs(source=DART_DOCUMENT_SOURCE))
    assert body["raw_document_hash"] in {blob.content_hash for blob in blobs}


def test_parser_bump_re_enables_reparse(tmp_path: Path) -> None:
    from src.data.jobs.dart_documents import DartDocumentReparseJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe_years(runtime, {TICKER: {2020, 2021}})
    ctx = _ctx(runtime, _provider())
    archive = b"PK\x03\x04old-parser-bytes"
    receipt = _store_archive(ctx, archive, "20210330001234")
    page = {
        "source_kind": "document_verified",
        "status": "000",
        "parser_version": "old",
        "checks": ["bs_balance"],
        "records": [],
        "mapping_version": "dart-fact-map-v1",
        "diagnostics": (),
        "raw_document_hash": receipt.content_hash,
        "corp_code": CORP, "biz_year": "2020", "reprt_code": "11011",
        "filing_id": "20210330001234", "rcept_no": "20210330001234",
        "fs_div": "CFS", "published_at": "2020-03-30",
    }
    _persist_fact(ctx, page)

    assert len(DartDocumentReparseJob().pending(ctx)) == 1


def test_standard_pages_ignored(tmp_path: Path) -> None:
    from src.data.jobs.dart_documents import DartDocumentFetchJob, DartDocumentReparseJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe_years(runtime, {TICKER: {2020, 2021}})
    ctx = _ctx(runtime, _provider())
    _setup_relevant(ctx, runtime)
    archive = b"PK\x03\x04standard-bytes"
    receipt = _store_archive(ctx, archive, "20210330001234")
    page = {
        "source_kind": "opendart_standard",
        "status": "000",
        "identity": {"corp_code": CORP, "biz_year": "2020", "reprt_code": "11011"},
        "records": [{"fact": "sales"}],
        "mapping_version": "test-v1",
        "diagnostics": (),
        "raw_document_hash": receipt.content_hash,
        "corp_code": CORP, "biz_year": "2020", "reprt_code": "11011",
        "filing_id": "20210330001234", "rcept_no": "20210330001234",
        "fs_div": "CFS", "published_at": "2020-03-30",
    }
    _persist_fact(ctx, page)

    assert DartDocumentReparseJob().pending(ctx) == []
    assert DartDocumentFetchJob().pending(ctx) == []


def test_relevant_empty_without_universe_or_bridge(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.jobs.dart_documents import _eligible_years_by_ticker, relevant_fact_identities

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    with pytest.raises(PITDataError):
        _eligible_years_by_ticker(ctx)
    with pytest.raises(PITDataError):
        relevant_fact_identities(ctx)


def test_eligible_years_skips_bad_rows(tmp_path: Path) -> None:
    import polars as pl

    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.jobs.dart_documents import _eligible_years_by_ticker

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    frame = pl.DataFrame(
        {
            "ticker": [TICKER, "", TICKER, "BAD"],
            "eligible": [True, True, False, True],
            "session": [date(2020, 6, 30), date(2020, 6, 30), date(2020, 6, 30), "not-a-date"],
        },
        schema={"ticker": pl.String, "eligible": pl.Boolean, "session": pl.String}, strict=False,
    )
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="ordinary_universe", layer=DatasetLayer.SILVER,
                                 policy_version="test-v1", inputs={}, params={}),
        partitions={"part.parquet": frame},
    )
    DatasetRegistry(runtime.workspace.state_root).register("ordinary_universe", published.dataset_id)
    ctx = _ctx(runtime, _provider())

    assert _eligible_years_by_ticker(ctx) == {TICKER: {2020}}


def test_document_job_edge_branches(tmp_path: Path, monkeypatch) -> None:
    from src.data.jobs.dart_documents import (
        DartDocumentFetchJob,
        DartDocumentReparseJob,
        _is_document_not_found,
        _unit_payload,
        relevant_fact_identities,
    )

    assert _is_document_not_found({"diagnostics": 42}) is False
    assert _is_document_not_found({"diagnostics": ("document_not_found",)}) is True
    assert _unit_payload("BADKEY", {})["corp_code"] == "BADKEY"

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe_years(runtime, {TICKER: {2020, 2021}})
    ctx = _ctx(runtime, _provider())
    _setup_relevant(ctx, runtime)
    _write_disclosure_page(
        runtime.workspace.bronze_root,
        {"records": [_disclosure_record("20210330007777", "20210330", "사업보고서 (2020.12)", corp="99999999")],
         "start": "2019-01-01", "end": "2021-12-31", "corp_code": "99999999"},
    )
    assert "99999999:2020:11011" not in relevant_fact_identities(ctx)

    _persist_fact(ctx, _legacy_page(filing_id="20210330001234", raw_hash=None))
    assert len(DartDocumentFetchJob().pending(ctx)) == 1

    not_found_page = _legacy_page(filing_id="20210330001234", raw_hash=None)
    not_found_page["diagnostics"] = ("document_not_found",)
    import json as _json

    from src.data.collection import dart_fact_scoped_payload

    ctx2_runtime = _runtime(tmp_path / "nf")
    _bridge(ctx2_runtime.workspace.bronze_root)
    _universe_years(ctx2_runtime, {TICKER: {2020, 2021}})
    ctx2 = _ctx(ctx2_runtime, _provider())
    _setup_relevant(ctx2, ctx2_runtime)
    ctx2.writer.persist(
        dart_fact_scoped_payload(page={k: v for k, v in not_found_page.items() if k != "raw_archive"},
                                 retrieved_at=NOW)
    )
    assert DartDocumentFetchJob().pending(ctx2) == []
    assert DartDocumentReparseJob().pending(ctx2) == []

    reparse = DartDocumentReparseJob()
    assert reparse.fetch(ctx, []) == []
    from src.data.jobs.runner import JobUnit

    assert reparse.fetch(ctx, [JobUnit(source="financial_facts", natural_key="k",
                                       payload={"corp_code": CORP}, max_requests=1)]) == []

    fetch = DartDocumentFetchJob()
    assert fetch.fetch(ctx, [JobUnit(source="financial_facts", natural_key="k",
                                     payload={"corp_code": CORP}, max_requests=1)]) == []

    class _Collector014:
        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b"<?xml version='1.0'?><result><status>014</status></result>"

        def health_check(self) -> None:
            return None

    ctx014 = _ctx(runtime, _provider(), collector=_Collector014())
    (unit014,) = DartDocumentFetchJob().pending(ctx014)
    (payload014,) = DartDocumentFetchJob().fetch(ctx014, [unit014])
    assert _json.loads(payload014.payload)["diagnostics"] == ["document_not_found"]

    reparse.health_check(ctx)
    fetch.health_check(ctx014)

    empty_runtime = _runtime(tmp_path / "empty")
    empty_ctx = _ctx(empty_runtime, _provider())
    assert tuple(DartDocumentFetchJob().pending(empty_ctx)) == ()


def test_fetch_skips_current_verified_with_archive(tmp_path: Path, monkeypatch) -> None:
    from src.data.jobs.dart_documents import DartDocumentFetchJob
    from src.integrations.dart.document_statements import (
        PARSER_VERSION,
        DocumentParseResult,
        document_verified_page,
    )

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe_years(runtime, {TICKER: {2020, 2021}})
    ctx = _ctx(runtime, _provider())
    _setup_relevant(ctx, runtime)
    archive = b"PK\x03\x04current-verified-bytes"
    receipt = _store_archive(ctx, archive, "20210330001234")
    monkeypatch.setattr(
        "src.integrations.dart.document_statements.parse_filing_document",
        lambda data, *, reprt_code, biz_year: DocumentParseResult(statements=_verified_statements(), diagnostics=()),
    )
    page = dict(
        document_verified_page(
            identity={"corp_code": CORP, "biz_year": "2020", "reprt_code": "11011",
                      "filing_id": "20210330001234", "rcept_no": "20210330001234",
                      "fs_div": "CFS", "published_at": "2021-03-30", "ticker": TICKER},
            result=DocumentParseResult(statements=_verified_statements(), diagnostics=()),
            document_hash=receipt.content_hash,
        )
    )
    assert page["parser_version"] == PARSER_VERSION
    _persist_fact(ctx, page)

    assert DartDocumentFetchJob().pending(ctx) == []


def test_fetch_skips_legacy_with_usable_archive(tmp_path: Path) -> None:
    from src.data.jobs.dart_documents import DartDocumentFetchJob, DartDocumentReparseJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe_years(runtime, {TICKER: {2020, 2021}})
    ctx = _ctx(runtime, _provider())
    _setup_relevant(ctx, runtime)
    receipt = _store_archive(ctx, b"PK\x03\x04usable-legacy-bytes", "20210330001234")
    _persist_fact(ctx, _legacy_page(filing_id="20210330001234", raw_hash=receipt.content_hash))

    assert len(DartDocumentReparseJob().pending(ctx)) == 1
    assert DartDocumentFetchJob().pending(ctx) == []
