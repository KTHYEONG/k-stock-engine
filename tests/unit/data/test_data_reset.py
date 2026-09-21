from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.data_reset import (
    LEGACY_REMOVAL_TARGETS,
    ResetVerification,
    remove_verified_legacy_data,
    verify_legacy_removal,
)
from src.data.rebase import RebaseReport, RetentionDecision, materialize_scoped_bronze
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.data.runtime import DataRuntime, load_data_runtime
from src.data.schemas import PITDataError

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
RETRIEVED_AT = datetime(2020, 6, 1, tzinfo=UTC)


def _runtime(tmp_path: Path) -> DataRuntime:
    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _write_legacy(legacy_root: Path, *, kind: str, payload: bytes) -> None:
    content_hash = hashlib.sha256(payload).hexdigest()
    receipt_dir = legacy_root / "bronze" / kind / content_hash
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "payload.json").write_bytes(payload)
    (receipt_dir / "receipt.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "content_hash": content_hash,
                "source_path": "legacy:test",
                "retrieved_at": RETRIEVED_AT.isoformat(),
                "ingested_at": RETRIEVED_AT.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _market(session: str) -> bytes:
    return json.dumps({"session": session, "records": [{"BAS_DD": session.replace("-", "")}]}).encode()


def _fact() -> bytes:
    return json.dumps(
        {"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11013", "published_at": "2019-05-15", "records": [1]}
    ).encode()


def _seed(tmp_path: Path, *, with_facts: bool = True) -> tuple[DataRuntime, RebaseReport]:
    legacy = tmp_path / "legacy"
    _write_legacy(legacy, kind="daily_market", payload=_market("2019-06-03"))
    if with_facts:
        _write_legacy(legacy, kind="financial_facts", payload=_fact())
    runtime = _runtime(tmp_path)
    report = materialize_scoped_bronze(runtime=runtime, legacy_data_root=legacy, dry_run=False)
    data_root = tmp_path / "data"
    (data_root / "archive").mkdir(parents=True, exist_ok=True)
    (data_root / "archive" / "old.json").write_text("{}", encoding="utf-8")
    (data_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (data_root / "artifacts" / "plan.json").write_text("{}", encoding="utf-8")
    (data_root / "bronze" / "stocks").mkdir(parents=True, exist_ok=True)
    (data_root / "bronze" / "stocks" / "legacy.json").write_text("{}", encoding="utf-8")
    (data_root / "silver" / "stocks").mkdir(parents=True, exist_ok=True)
    (data_root / "silver" / "stocks" / "table.parquet").write_bytes(b"silver")
    (data_root / "gold" / "stocks").mkdir(parents=True, exist_ok=True)
    (data_root / "gold" / "stocks" / "scores.parquet").write_bytes(b"gold")
    (data_root / "bronze" / "manual_note").mkdir(parents=True, exist_ok=True)
    (data_root / "bronze" / "manual_note" / "note.json").write_text("{}", encoding="utf-8")
    return runtime, report


def _publish_entry(runtime: DataRuntime, tmp_path: Path, entry: ReceiptIndexEntry) -> str:
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    catalog.publish((entry,))
    pointer = runtime.workspace.bronze_root / "catalog" / "latest.json"
    return str(json.loads(pointer.read_text(encoding="utf-8"))["revision"])[:-5]


def _entry(tmp_path: Path, *, source: str, natural_key: str, as_of: date | None, fiscal: str | None = None) -> ReceiptIndexEntry:
    body = f"{source}:{natural_key}".encode()
    payload_path = tmp_path / "blobs" / f"{natural_key}.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(body)
    return ReceiptIndexEntry(
        source=source,
        natural_key=natural_key,
        as_of=as_of,
        fiscal_period=fiscal,
        status=EvidenceStatus.SUCCESS,
        content_hash=hashlib.sha256(body).hexdigest(),
        retrieved_at=RETRIEVED_AT,
        payload_path=payload_path,
    )


def test_verify_legacy_removal_rejects_foreign_scope_report(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    foreign = dataclasses.replace(report, scope_hash="0" * 64)

    with pytest.raises(PITDataError, match="scope hash"):
        verify_legacy_removal(runtime=runtime, rebase_report=foreign, data_root=tmp_path / "data")


def test_verify_legacy_removal_blocks_uncovered_mandatory_source(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path, with_facts=False)

    with pytest.raises(PITDataError, match="mandatory raw coverage"):
        verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")


def test_remove_verified_legacy_data_plan_mode_has_no_mutation(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    verification = verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")

    planned = remove_verified_legacy_data(verification=verification, data_root=tmp_path / "data", apply=False)

    assert len(planned) == len(LEGACY_REMOVAL_TARGETS)
    for relative in LEGACY_REMOVAL_TARGETS:
        assert (tmp_path / "data" / relative).exists()


def test_remove_verified_legacy_data_apply_removes_exact_targets(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    verification = verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")

    removed = remove_verified_legacy_data(verification=verification, data_root=tmp_path / "data", apply=True)

    assert len(removed) == len(LEGACY_REMOVAL_TARGETS)
    for relative in LEGACY_REMOVAL_TARGETS:
        assert not (tmp_path / "data" / relative).exists()
    assert (tmp_path / "data" / "bronze" / "kr_swing_2019_v1").is_dir()
    assert (tmp_path / "data").is_dir()
    record = tmp_path / "data" / "state" / "kr_swing_2019_v1" / "reset" / verification.scope_hash / "removed.json"
    assert record.is_file()
    assert len(json.loads(record.read_text(encoding="utf-8"))["removed"]) == len(LEGACY_REMOVAL_TARGETS)


def test_verify_legacy_removal_symlink_target_fails_closed(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    shutil.rmtree(tmp_path / "data" / "archive")
    (tmp_path / "data" / "archive").symlink_to(tmp_path / "elsewhere")

    with pytest.raises(PITDataError, match="symlink"):
        verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")


def test_remove_verified_legacy_data_preserves_unknown_sibling(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    verification = verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")

    remove_verified_legacy_data(verification=verification, data_root=tmp_path / "data", apply=True)

    note = tmp_path / "data" / "bronze" / "manual_note" / "note.json"
    assert note.is_file()
    assert (tmp_path / "data" / "bronze" / "kr_swing_2019_v1").is_dir()


def test_remove_verified_legacy_data_records_absent_target(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    shutil.rmtree(tmp_path / "data" / "gold" / "stocks")
    verification = verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")

    removed = remove_verified_legacy_data(verification=verification, data_root=tmp_path / "data", apply=True)

    assert len(removed) == len(LEGACY_REMOVAL_TARGETS) - 1
    record = tmp_path / "data" / "state" / "kr_swing_2019_v1" / "reset" / verification.scope_hash / "removed.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["absent"] == ["gold/stocks"]
    assert len(payload["removed"]) == len(LEGACY_REMOVAL_TARGETS) - 1


def test_remove_verified_legacy_data_removes_file_target(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    shutil.rmtree(tmp_path / "data" / "artifacts")
    (tmp_path / "data" / "artifacts").write_text("stale-plan", encoding="utf-8")
    verification = verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")

    removed = remove_verified_legacy_data(verification=verification, data_root=tmp_path / "data", apply=True)

    assert len(removed) == len(LEGACY_REMOVAL_TARGETS)
    assert not (tmp_path / "data" / "artifacts").exists()


def test_verify_legacy_removal_rejects_dry_run_report(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    assert report.catalog_revision_hash
    dry = materialize_scoped_bronze(runtime=runtime, legacy_data_root=tmp_path / "legacy", dry_run=True)

    with pytest.raises(PITDataError, match="non-dry-run"):
        verify_legacy_removal(runtime=runtime, rebase_report=dry, data_root=tmp_path / "data")


def test_verify_legacy_removal_requires_matching_catalog_revision(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    bogus = dataclasses.replace(report, catalog_revision_hash="0" * 64)

    with pytest.raises(PITDataError, match="catalog revision"):
        verify_legacy_removal(runtime=runtime, rebase_report=bogus, data_root=tmp_path / "data")

    revision_path = runtime.workspace.bronze_root / "catalog" / f"{report.catalog_revision_hash}.json"
    revision_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PITDataError, match="catalog revision"):
        verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")


def test_verify_legacy_removal_rejects_undated_and_out_of_scope_entries(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    undated_hash = _publish_entry(
        runtime, tmp_path, _entry(tmp_path, source="krx_daily_market", natural_key="undated", as_of=None)
    )
    undated = dataclasses.replace(report, catalog_revision_hash=undated_hash)
    with pytest.raises(PITDataError, match="outside scope limits"):
        verify_legacy_removal(runtime=runtime, rebase_report=undated, data_root=tmp_path / "data")

    stale_hash = _publish_entry(
        runtime, tmp_path,
        _entry(tmp_path, source="financial_facts", natural_key="stale", as_of=date(2020, 1, 1), fiscal="2018Q4"),
    )
    stale = dataclasses.replace(report, catalog_revision_hash=stale_hash)
    with pytest.raises(PITDataError, match="outside scope limits"):
        verify_legacy_removal(runtime=runtime, rebase_report=stale, data_root=tmp_path / "data")


def test_verify_legacy_removal_blocks_claimed_but_missing_evidence(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    bogus = RetentionDecision(
        legacy_path=tmp_path / "ghost.json", source="krx_daily_market",
        natural_key="2099-01-01", retained=True, reason="in_scope_verified",
    )
    claimed = dataclasses.replace(report, decisions=(*report.decisions, bogus))

    with pytest.raises(PITDataError, match="cannot substitute"):
        verify_legacy_removal(runtime=runtime, rebase_report=claimed, data_root=tmp_path / "data")


def test_remove_verified_legacy_data_rejects_late_appearing_target(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    shutil.rmtree(tmp_path / "data" / "gold" / "stocks")
    verification = verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")
    (tmp_path / "data" / "gold" / "stocks").mkdir(parents=True)

    with pytest.raises(PITDataError, match="unverified target"):
        remove_verified_legacy_data(verification=verification, data_root=tmp_path / "data", apply=True)


def test_remove_verified_legacy_data_detects_incomplete_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, report = _seed(tmp_path)
    verification = verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")
    monkeypatch.setattr(shutil, "rmtree", lambda *args, **kwargs: None)

    with pytest.raises(PITDataError, match="incomplete"):
        remove_verified_legacy_data(verification=verification, data_root=tmp_path / "data", apply=True)


def test_remove_verified_legacy_data_requires_scoped_state(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    verification = verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")
    lost = dataclasses.replace(verification, rebase_report_hash="0" * 16)

    with pytest.raises(PITDataError, match="scoped state"):
        remove_verified_legacy_data(verification=lost, data_root=tmp_path / "data", apply=True)


def test_verify_legacy_removal_rejects_symlinked_parent(tmp_path: Path) -> None:
    runtime, report = _seed(tmp_path)
    outside = tmp_path / "outside"
    shutil.move(str(tmp_path / "data" / "bronze"), str(outside))
    (tmp_path / "data" / "bronze").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PITDataError, match="outside enumerated"):
        verify_legacy_removal(runtime=runtime, rebase_report=report, data_root=tmp_path / "data")


def test_reset_verification_holds_hashes() -> None:
    verification = ResetVerification(
        scope_hash="s", rebase_report_hash="r", catalog_revision_hash="c", verified_targets=()
    )

    assert verification.scope_hash == "s"


def test_remove_legacy_data_command_plans_applies_and_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.data.cli import main

    runtime, report = _seed(tmp_path)
    data_root = tmp_path / "data"
    base = [
        "--scope-config", str(SCOPE_CONFIG), "--data-root", str(data_root),
        "--rebase-report", str(report.report_path),
    ]

    assert main(["remove-legacy-data", *base]) == 0
    plan = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert plan["removable"] == len(LEGACY_REMOVAL_TARGETS)
    assert plan["removed"] == 0

    assert main(["remove-legacy-data", *base, "--apply"]) == 0
    applied = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert applied["removed"] == len(LEGACY_REMOVAL_TARGETS)
    assert applied["absent"] == 0
    assert runtime.workspace.bronze_root.is_dir()

    missing = ["--scope-config", str(SCOPE_CONFIG), "--data-root", str(data_root),
               "--rebase-report", str(tmp_path / "nope.json")]
    assert main(["remove-legacy-data", *missing]) == 1
