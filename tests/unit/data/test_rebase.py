from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.data.rebase import RebaseReport, materialize_scoped_bronze
from src.data.receipt_catalog import ReceiptCatalog
from src.data.runtime import DataRuntime, load_data_runtime

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
RETRIEVED_AT = datetime(2020, 6, 1, tzinfo=UTC)


def _runtime(tmp_path: Path) -> DataRuntime:
    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _write_legacy(
    legacy_root: Path,
    *,
    kind: str,
    payload: bytes,
    retrieved_at: datetime = RETRIEVED_AT,
    source_path: str = "legacy:test",
    receipt: bool = True,
    receipt_kind: str | None = None,
    tampered: bool = False,
    corrupt_receipt: bool = False,
    blank_hash: bool = False,
    missing_retrieved_at: bool = False,
    naive_retrieved_at: bool = False,
    dirname: str | None = None,
) -> Path:
    content_hash = hashlib.sha256(payload).hexdigest()
    receipt_dir = legacy_root / "bronze" / kind / (dirname or content_hash)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    payload_path = receipt_dir / "payload.json"
    payload_path.write_bytes(payload)
    if receipt:
        if corrupt_receipt:
            (receipt_dir / "receipt.json").write_text("{not-json", encoding="utf-8")
        else:
            meta: dict[str, Any] = {
                "kind": receipt_kind or kind,
                "content_hash": "" if blank_hash else (("0" * 64) if tampered else content_hash),
                "source_path": source_path,
                "ingested_at": RETRIEVED_AT.isoformat(),
            }
            if not missing_retrieved_at:
                stamp = retrieved_at.replace(tzinfo=None).isoformat() if naive_retrieved_at else retrieved_at.isoformat()
                meta["retrieved_at"] = stamp
            (receipt_dir / "receipt.json").write_text(json.dumps(meta), encoding="utf-8")
    return payload_path


def _market(session: str, *, records: Any = "default") -> bytes:
    if records == "default":
        compact = session.replace("-", "")
        records = [{"BAS_DD": compact, "ISU_SRT_CD": "005930", "MKTCAP": 10, "LIST_SHRS": 5}]
    return json.dumps({"session": session, "records": records}).encode()


def _fact(corp: str, biz: str, reprt: str, published: str, *, identity: bool = False) -> bytes:
    body: dict[str, Any] = {"corp_code": corp, "biz_year": biz, "reprt_code": reprt, "published_at": published, "records": [1]}
    if identity:
        body = {"identity": dict(body), "records": [1]}
    return json.dumps(body).encode()


def _action(end: str, *, corp: str = "00475985", endpoint: str = "crDecsn.json") -> bytes:
    return json.dumps({"corp_code": corp, "endpoint": endpoint, "end": end, "records": [1]}).encode()


def _corp_map() -> bytes:
    return json.dumps([{"corp_code": "00126380", "corp_name": "x", "ticker": "005930"}]).encode()


def _master(session: str) -> bytes:
    return json.dumps({"as_of": session, "records": [{"ISU_SRT_CD": "005930"}]}).encode()


def _rebase(tmp_path: Path, legacy_root: Path, *, dry_run: bool = False) -> tuple[DataRuntime, RebaseReport]:
    runtime = _runtime(tmp_path)
    return runtime, materialize_scoped_bronze(runtime=runtime, legacy_data_root=legacy_root, dry_run=dry_run)


def _reasons(report: RebaseReport) -> dict[str, str]:
    return {item.natural_key or str(item.legacy_path): item.reason for item in report.decisions}


def test_materialize_scoped_bronze_retains_in_scope_daily_payload(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"))
    _write_legacy(legacy, kind="corporate_actions", payload=_action("2019-05-01"))
    runtime, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 2
    assert report.rejected_payload_count == 0
    assert report.scope_hash == runtime.scope.content_hash
    assert report.report_path.is_file()
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    assert catalog.successful_keys(source="krx_daily_market") == frozenset({"2019-06-03"})
    assert (runtime.workspace.bronze_root / "daily_market").is_dir()


def test_materialize_scoped_bronze_retains_in_scope_security_master(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="security_master", payload=_master("2019-06-03"))
    runtime, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 1
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    assert catalog.successful_keys(source="krx_security_master") == frozenset({"2019-06-03"})


def test_materialize_scoped_bronze_rejects_security_master_without_provider_date(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="security_master", payload=json.dumps({"records": []}).encode())
    _, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 0
    assert report.decisions[0].reason == "missing_provider_date"


def test_materialize_scoped_bronze_rejects_out_of_scope_and_duplicate_security_master(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="security_master", payload=_master("2018-12-31"), dirname="old")
    _write_legacy(legacy, kind="security_master", payload=_master("2019-06-03"), dirname="one")
    _write_legacy(legacy, kind="security_master", payload=_master("2019-06-03"), dirname="two")
    _write_legacy(legacy, kind="security_master", payload=json.dumps([1]).encode(), dirname="malformed")
    _, report = _rebase(tmp_path, legacy)

    assert sorted(item.reason for item in report.decisions) == [
        "duplicate_receipt", "in_scope_verified", "malformed_legacy_payload", "out_of_scope_provider_date"
    ]


def test_materialize_scoped_bronze_flushes_full_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.rebase as rebase

    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"))
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-04"))
    monkeypatch.setattr(rebase, "_REBASE_CATALOG_BATCH_SIZE", 1)
    _, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 2


def test_materialize_scoped_bronze_reads_historical_bronze_stocks_namespace(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    source = legacy / "bronze" / "stocks"
    _write_legacy(source.parent.parent, kind="daily_market", payload=_market("2019-06-03"))
    # Move the direct fixture into the on-disk layout used by the existing data root.
    source.mkdir()
    (legacy / "bronze" / "daily_market").rename(source / "daily_market")

    runtime, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 1
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    assert catalog.successful_keys(source="krx_daily_market") == frozenset({"2019-06-03"})


def test_materialize_scoped_bronze_rejects_pre_2019_daily_payload(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2018-12-31"))
    _, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 0
    assert [item.reason for item in report.decisions] == ["out_of_scope_provider_date"]
    assert all(not item.retained for item in report.decisions)


def test_materialize_scoped_bronze_applies_fiscal_floor_to_dart_facts(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="financial_facts", payload=_fact("00126380", "2018", "11011", "2020-03-30"))
    _write_legacy(legacy, kind="financial_facts", payload=_fact("00126380", "2019", "11013", "2019-05-15", identity=True))
    runtime, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 1
    assert _reasons(report) == {"00126380:2018:11011": "out_of_scope_fiscal_period", "00126380:2019:11013": "in_scope_verified"}
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    assert catalog.successful_keys(source="financial_facts", fiscal_start="2019Q1") == frozenset({"00126380:2019:11013"})


def test_materialize_scoped_bronze_rejects_post_completed_fiscal_facts(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="financial_facts", payload=_fact("00126380", "2026", "11013", "2026-05-15"))
    _, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 0
    assert _reasons(report) == {"00126380:2026:11013": "out_of_scope_fiscal_period"}


def test_materialize_scoped_bronze_retains_corp_code_map(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="dart_corp_codes", payload=_corp_map(), receipt=False)
    runtime, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 1
    assert report.decisions[0].source == "dart_corp_codes"
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    assert catalog.successful_keys(source="dart_corp_codes") == frozenset({"dart_corp_codes"})


def test_materialize_scoped_bronze_rejects_unverifiable_hash_directory(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"), receipt=False)
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-04"), tampered=True)
    _, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 0
    assert sorted(item.reason for item in report.decisions) == ["missing_legacy_receipt", "receipt_hash_mismatch"]


def test_materialize_scoped_bronze_ignores_derived_outputs(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"))
    (legacy / "silver" / "stocks").mkdir(parents=True)
    (legacy / "silver" / "stocks" / "market.parquet").write_bytes(b"silver")
    (legacy / "gold" / "stocks").mkdir(parents=True)
    (legacy / "gold" / "stocks" / "scores.parquet").write_bytes(b"gold")
    (legacy / "archive").mkdir(parents=True)
    (legacy / "archive" / "old.json").write_text("{}", encoding="utf-8")
    (legacy / "artifacts" / "collection-plans").mkdir(parents=True)
    (legacy / "artifacts" / "collection-plans" / "plan.json").write_text("{}", encoding="utf-8")
    _, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 1
    for item in report.decisions:
        text = str(item.legacy_path)
        assert "/silver/" not in text
        assert "/gold/" not in text
        assert "/archive/" not in text
        assert "/artifacts/" not in text


def test_materialize_scoped_bronze_dry_run_writes_only_report(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"))
    _write_legacy(legacy, kind="financial_facts", payload=_fact("00126380", "2019", "11013", "2019-05-15"))
    runtime, report = _rebase(tmp_path, legacy, dry_run=True)

    assert report.retained_payload_count == 2
    assert report.catalog_revision_hash == ""
    assert report.report_path.is_file()
    assert not (runtime.workspace.bronze_root / "catalog").exists()
    assert not (runtime.workspace.bronze_root / "daily_market").exists()
    assert not (runtime.workspace.bronze_root / "financial_facts").exists()


def test_materialize_scoped_bronze_is_idempotent(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"))
    runtime, first = _rebase(tmp_path, legacy)
    _, second = _rebase(tmp_path, legacy)

    assert first.content_hash == second.content_hash
    assert first.report_path == second.report_path
    assert first.catalog_revision_hash == second.catalog_revision_hash
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    assert catalog.successful_keys(source="krx_daily_market") == frozenset({"2019-06-03"})


def test_materialize_scoped_bronze_rejects_malformed_and_unsupported_receipts(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"), corrupt_receipt=True)
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-04"), missing_retrieved_at=True)
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-05"), naive_retrieved_at=True)
    _write_legacy(legacy, kind="daily_market", payload=b"{broken", dirname="bad-payload")
    _write_legacy(legacy, kind="daily_market", payload=json.dumps([1, 2]).encode(), dirname="list-payload")
    _write_legacy(legacy, kind="corporate_actions", payload=json.dumps([1]).encode(), dirname="list-action")
    _write_legacy(legacy, kind="financial_facts", payload=json.dumps([1]).encode(), dirname="list-fact")
    _write_legacy(legacy, kind="calendar", payload=json.dumps({"sessions": []}).encode())
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-06"), receipt_kind="financial_facts")
    _write_legacy(legacy, kind="financial_facts", payload=json.dumps({"records": [1]}).encode(), dirname="no-identity")
    _write_legacy(legacy, kind="daily_market", payload=json.dumps({"records": []}).encode(), dirname="no-date")
    _write_legacy(legacy, kind="daily_market", payload=json.dumps({"session": "not-a-date"}).encode(), dirname="bad-date")
    _write_legacy(
        legacy, kind="financial_facts",
        payload=json.dumps({"corp_code": "c", "biz_year": "2019", "reprt_code": "99999", "records": [1]}).encode(),
        dirname="no-fiscal",
    )
    _write_legacy(
        legacy, kind="financial_facts",
        payload=json.dumps(
            {"corp_code": "c", "biz_year": "2019", "reprt_code": "11013", "fiscal_period": "20XXQ1", "records": [1]}
        ).encode(),
        dirname="bad-fiscal",
    )
    records_only = json.dumps({"records": [{"BAS_DD": "20190603", "ISU_SRT_CD": "005930"}]}).encode()
    _write_legacy(legacy, kind="daily_market", payload=records_only, dirname="records-only")
    mixed_records = json.dumps({"records": [42, {"BAS_DD": "20190604"}]}).encode()
    _write_legacy(legacy, kind="daily_market", payload=mixed_records, dirname="mixed-records")
    bad_compact = json.dumps({"records": [{"BAS_DD": "20231345"}, {"BAS_DD": "20190606"}]}).encode()
    _write_legacy(legacy, kind="daily_market", payload=bad_compact, dirname="bad-compact")
    empty_value = json.dumps({"records": [{"BAS_DD": ""}, {"BAS_DD": "20190607"}]}).encode()
    _write_legacy(legacy, kind="daily_market", payload=empty_value, dirname="empty-value")
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-08"), blank_hash=True, dirname="blank-hash")
    _, report = _rebase(tmp_path, legacy)

    reasons = _reasons(report)
    assert reasons[str(legacy / "bronze" / "daily_market" / "bad-payload" / "payload.json")] == "malformed_legacy_payload"
    assert "unsupported_legacy_source" in set(reasons.values())
    assert "receipt_kind_mismatch" in set(reasons.values())
    assert "missing_payload_identity" in set(reasons.values())
    assert "missing_provider_date" in set(reasons.values())
    assert "missing_fiscal_period" in set(reasons.values())
    assert "invalid_fiscal_period" in set(reasons.values())
    assert "malformed_legacy_receipt" in set(reasons.values())
    assert report.retained_payload_count == 5


def test_materialize_scoped_bronze_rejects_conflicting_duplicates(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"), dirname="first")
    _write_legacy(
        legacy, kind="daily_market",
        payload=json.dumps({"session": "2019-06-03", "records": [{"BAS_DD": "20190603", "extra": 1}]}).encode(),
        dirname="second",
    )
    _write_legacy(legacy, kind="financial_facts", payload=_fact("00126380", "2019", "11013", "2019-05-15"), dirname="fact-a")
    _write_legacy(
        legacy, kind="financial_facts",
        payload=json.dumps(
            {"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11013", "published_at": "2019-05-16", "records": [2]}
        ).encode(),
        dirname="fact-b",
    )
    _write_legacy(legacy, kind="dart_corp_codes", payload=_corp_map(), receipt=False, dirname="map-a")
    _write_legacy(legacy, kind="dart_corp_codes", payload=b"[]", receipt=False, dirname="map-b")
    _, report = _rebase(tmp_path, legacy)

    reasons = sorted(item.reason for item in report.decisions)
    assert reasons.count("in_scope_verified") == 3
    assert reasons.count("duplicate_conflict") == 3


def test_materialize_scoped_bronze_handles_empty_legacy_root(tmp_path: Path) -> None:
    legacy = tmp_path / "missing"
    (tmp_path / "stray-bronze").mkdir(parents=True)
    runtime, report = _rebase(tmp_path, legacy)

    assert report.decisions == ()
    assert report.retained_payload_count == 0
    assert report.catalog_revision_hash == ""
    assert report.report_path.is_file()
    assert runtime.scope.content_hash == report.scope_hash


def test_materialize_scoped_bronze_skips_stray_files(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    (legacy / "bronze").mkdir(parents=True)
    (legacy / "bronze" / "stray.json").write_text("{}", encoding="utf-8")
    (legacy / "bronze" / "daily_market").mkdir(parents=True)
    (legacy / "bronze" / "daily_market" / "stray.json").write_text("{}", encoding="utf-8")
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"))
    _, report = _rebase(tmp_path, legacy)

    assert report.retained_payload_count == 1
    assert len(report.decisions) == 1


def test_rebase_2019_command_materializes_and_dry_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from src.data.cli import main

    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"))
    data_root = tmp_path / "data"
    args = ["--scope-config", str(SCOPE_CONFIG), "--data-root", str(data_root), "--legacy-data-root", str(legacy)]

    assert main(["rebase-2019", *args]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["retained"] == 1
    assert out["rejected"] == 0
    assert out["dry_run"] is False

    assert main(["rebase-2019", *args, "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert dry["retained"] == 1
    assert dry["dry_run"] is True
