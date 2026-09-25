from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from src.data.receipt_catalog import ReceiptCatalog
from src.data.remote_dart_inbox import fold_remote_quota, ingest_remote_bronze
from src.data.runtime import load_data_runtime
from src.data.scoped_ingestion import FACT_SOURCE, ScopedBronzeWriter, dart_fact_natural_key
from src.integrations.quota import ProviderQuotaStateStore

RETRIEVED = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def _writer(tmp_path: Path) -> tuple[ScopedBronzeWriter, Path]:
    runtime = load_data_runtime(scope_config=Path("config/research/kr_swing_2019_v1.toml"), data_root=tmp_path / "data")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    return ScopedBronzeWriter(runtime=runtime, catalog=catalog), runtime.workspace.bronze_root


def _page(corp: str = "00126380") -> dict[str, object]:
    return {
        "identity": {"corp_code": corp, "biz_year": "2016", "reprt_code": "11013", "published_at": "2016-05-16"},
        "records": [{"account": "revenue"}],
        "source_kind": "opendart_standard",
    }


def _write_fact(inbox: Path, page: dict[str, object], *, tamper: bool = False, receipt: bool = True) -> str:
    raw = json.dumps(page, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    directory = inbox / "financial_facts" / digest
    directory.mkdir(parents=True)
    (directory / "payload.json").write_bytes(raw + (b" " if tamper else b""))
    if receipt:
        (directory / "receipt.json").write_text(json.dumps({"content_hash": digest, "retrieved_at": RETRIEVED.isoformat()}), encoding="utf-8")
    return digest


def _write_document(inbox: Path, *, tamper: bool = False) -> str:
    raw = b"PK-archive-bytes"
    digest = hashlib.sha256(raw).hexdigest()
    directory = inbox / "dart_documents" / digest
    directory.mkdir(parents=True)
    (directory / "payload.zip").write_bytes(raw + (b"x" if tamper else b""))
    (directory / "receipt.json").write_text(
        json.dumps({"rcept_no": "20160516000001", "sha256": digest, "retrieved_at": RETRIEVED.isoformat()}), encoding="utf-8"
    )
    return digest


def test_ingest_registers_verified_pages_and_reports_them_deletable(tmp_path: Path) -> None:
    writer, bronze = _writer(tmp_path)
    inbox = tmp_path / "inbox"
    fact = _write_fact(inbox, _page())
    doc = _write_document(inbox)

    result = ingest_remote_bronze(inbox_bronze=inbox, bronze_root=bronze, writer=writer)

    assert set(result.accepted) == {f"financial_facts/{fact}", f"dart_documents/{doc}"}
    assert (result.fact_pages, result.document_pages, result.rejected) == (1, 1, ())
    assert (bronze / "financial_facts" / fact / "payload.json").is_file()
    assert (bronze / "dart_documents" / doc / "payload.zip").is_file()
    key = dart_fact_natural_key(corp_code="00126380", biz_year="2016", reprt_code="11013")
    entry = ReceiptCatalog(bronze / "catalog").latest(source=FACT_SOURCE, natural_keys={key})[key]
    assert entry.status.value == "success"
    assert entry.content_hash == fact


def test_ingest_rejects_tampered_pages_and_never_lists_them_deletable(tmp_path: Path) -> None:
    writer, bronze = _writer(tmp_path)
    inbox = tmp_path / "inbox"
    bad_fact = _write_fact(inbox, _page("00000001"), tamper=True)
    bad_doc = _write_document(inbox, tamper=True)

    result = ingest_remote_bronze(inbox_bronze=inbox, bronze_root=bronze, writer=writer)

    assert result.accepted == ()
    assert set(result.rejected) == {f"financial_facts/{bad_fact}", f"dart_documents/{bad_doc}"}
    assert not (bronze / "financial_facts").exists()


def test_ingest_skips_pages_still_being_written_and_is_idempotent(tmp_path: Path) -> None:
    writer, bronze = _writer(tmp_path)
    inbox = tmp_path / "inbox"
    _write_fact(inbox, _page("00000002"), receipt=False)
    ready = _write_fact(inbox, _page("00000003"))

    first = ingest_remote_bronze(inbox_bronze=inbox, bronze_root=bronze, writer=writer)
    second = ingest_remote_bronze(inbox_bronze=inbox, bronze_root=bronze, writer=writer)

    assert first.accepted == (f"financial_facts/{ready}",)
    assert second.accepted == first.accepted


def test_ingest_of_an_empty_inbox_is_a_noop(tmp_path: Path) -> None:
    writer, bronze = _writer(tmp_path)

    result = ingest_remote_bronze(inbox_bronze=tmp_path / "missing", bronze_root=bronze, writer=writer)

    assert (result.accepted, result.rejected) == ((), ())


def test_fold_remote_quota_counts_each_remote_request_once(tmp_path: Path) -> None:
    store = ProviderQuotaStateStore(tmp_path / "local")
    remote = tmp_path / "remote.json"
    marks = tmp_path / "marks.json"
    remote.write_text(json.dumps({"OpenDART#abc|fnltt": {"daily_attempt_day": "2026-09-24", "daily_attempted_requests": 30}}), encoding="utf-8")

    assert fold_remote_quota(store=store, remote_state=remote, watermark_path=marks) == 30
    assert fold_remote_quota(store=store, remote_state=remote, watermark_path=marks) == 0
    remote.write_text(json.dumps({"OpenDART#abc|fnltt": {"daily_attempt_day": "2026-09-24", "daily_attempted_requests": 45}}), encoding="utf-8")
    assert fold_remote_quota(store=store, remote_state=remote, watermark_path=marks) == 15

    now = datetime(2026, 9, 24, 3, tzinfo=UTC)
    assert store.remaining_daily_attempts(provider="OpenDART#abc", now=now, daily_limit=100) == 55
    assert fold_remote_quota(store=store, remote_state=tmp_path / "absent.json", watermark_path=marks) == 0
    remote.write_text(json.dumps({"no-separator": {"daily_attempt_day": "2026-09-24", "daily_attempted_requests": 9}}), encoding="utf-8")
    assert fold_remote_quota(store=store, remote_state=remote, watermark_path=marks) == 0


def test_ingest_ignores_unreadable_receipts_and_incomplete_documents(tmp_path: Path) -> None:
    writer, bronze = _writer(tmp_path)
    inbox = tmp_path / "inbox"
    broken = inbox / "financial_facts" / ("a" * 64)
    broken.mkdir(parents=True)
    (broken / "payload.json").write_text("{}", encoding="utf-8")
    (broken / "receipt.json").write_text("not json", encoding="utf-8")
    (inbox / "dart_documents" / ("b" * 64)).mkdir(parents=True)

    result = ingest_remote_bronze(inbox_bronze=inbox, bronze_root=bronze, writer=writer)

    assert (result.accepted, result.rejected) == ((), ())


def test_ingest_rejects_pages_that_local_serialization_would_change(tmp_path: Path) -> None:
    writer, bronze = _writer(tmp_path)
    inbox = tmp_path / "inbox"
    raw = json.dumps(_page("00000009"), indent=2).encode("utf-8")  # 해시는 맞지만 로컬 직렬화(sort_keys, 압축)와 다르다
    digest = hashlib.sha256(raw).hexdigest()
    directory = inbox / "financial_facts" / digest
    directory.mkdir(parents=True)
    (directory / "payload.json").write_bytes(raw)
    (directory / "receipt.json").write_text(json.dumps({"content_hash": digest, "retrieved_at": RETRIEVED.isoformat()}), encoding="utf-8")

    result = ingest_remote_bronze(inbox_bronze=inbox, bronze_root=bronze, writer=writer)

    assert result.rejected == (f"financial_facts/{digest}",)
    assert result.accepted == ()


def test_ingest_publishes_in_bounded_batches(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import src.data.remote_dart_inbox as module

    monkeypatch.setattr(module, "_BATCH", 2)
    writer, bronze = _writer(tmp_path)
    inbox = tmp_path / "inbox"
    for corp in ("00000011", "00000012", "00000013"):
        _write_fact(inbox, _page(corp))

    result = ingest_remote_bronze(inbox_bronze=inbox, bronze_root=bronze, writer=writer)

    assert result.fact_pages == 3
