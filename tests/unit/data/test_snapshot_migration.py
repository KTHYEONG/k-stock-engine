"""Snapshot KRX Bronze migration invariants: scoped receipts without provider requests."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.core.pit import EvidenceKind, PITDataError

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
DAY_A = date(2017, 1, 3)
DAY_B = date(2017, 1, 4)
DAY_C = date(2017, 1, 5)
MASTER_DAY = date(2020, 10, 20)
RETRIEVED_EARLY = datetime(2026, 9, 10, 0, 0, tzinfo=UTC)
RETRIEVED_LATE = datetime(2026, 9, 11, 0, 0, tzinfo=UTC)


def _runtime(tmp_path: Path):
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    runtime.workspace.initialize()
    return runtime


def _write_blob(snapshot_root: Path, kind: EvidenceKind, *, payload: dict, source_path: str, retrieved_at: datetime) -> Path:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    blob_dir = snapshot_root / kind.value / digest
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / "payload.json").write_bytes(raw)
    (blob_dir / "receipt.json").write_text(
        json.dumps(
            {
                "source_path": source_path,
                "retrieved_at": retrieved_at.isoformat(),
                "ingested_at": retrieved_at.isoformat(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return blob_dir


def _daily_payload(day: date, *, markets: tuple[str, ...] = ("KOSPI",), bas_dd: str | None = None) -> dict:
    code = (bas_dd or day.strftime("%Y%m%d"))
    return {
        "session": day.isoformat(),
        "retrieved_at": RETRIEVED_EARLY.isoformat(),
        "records": [
            {"MKT_NM": market, "BAS_DD": code, "TDD_CLSPRC": "1000", "ISU_CD": f"KR70000{i}"}
            for i, market in enumerate(markets)
        ],
    }


def _migrate(runtime, snapshot_root: Path, **kwargs):
    from src.data.snapshot_migration import migrate_snapshot_krx

    emitted: list[dict] = []
    params = {"dry_run": False, "emit": emitted.append}
    params.update(kwargs)
    reports = migrate_snapshot_krx(runtime, snapshot_root=snapshot_root, **params)
    return reports, emitted


def test_migrated_session_counts_as_answered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.jobs.krx as krx_jobs

    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    _write_blob(
        snapshot_root, EvidenceKind.DAILY_MARKET,
        payload=_daily_payload(DAY_A), source_path=f"krx:daily-market:{DAY_A.isoformat()}",
        retrieved_at=RETRIEVED_EARLY,
    )
    reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,))
    assert reports[0].migrated == 1

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", lambda start, end: (DAY_A, DAY_B))
    from src.data.jobs.krx import build_krx_job_context
    from src.config import load_provider_policy, load_runtime_config

    ctx = build_krx_job_context(
        runtime=runtime, provider=load_provider_policy(load_runtime_config()),
        now=lambda: datetime(2017, 1, 6, 9, 1, tzinfo=UTC),
    )
    pending_keys = {unit.natural_key for unit in krx_jobs.KrxDailyMarketJob().pending(ctx)}
    assert DAY_A.isoformat() not in pending_keys
    assert DAY_B.isoformat() in pending_keys
    entry = ctx.catalog.latest(source=krx_jobs.KRX_DAILY_MARKET_SOURCE, natural_keys={DAY_A.isoformat()})[DAY_A.isoformat()]
    assert entry.retrieved_at == RETRIEVED_EARLY


def test_normalized_single_session_page_is_accepted(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    _write_blob(
        snapshot_root, EvidenceKind.DAILY_MARKET,
        payload=_daily_payload(DAY_B), source_path="normalized-provider-page:daily_market",
        retrieved_at=RETRIEVED_EARLY,
    )
    reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,))
    assert reports[0].migrated == 1
    assert reports[0].skipped_blobs == {}


def test_manifests_and_multi_date_pages_are_skipped(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    _write_blob(
        snapshot_root, EvidenceKind.DAILY_MARKET,
        payload=_daily_payload(DAY_A), source_path="manifest:daily_market",
        retrieved_at=RETRIEVED_EARLY,
    )
    mixed = _daily_payload(DAY_A)
    mixed["records"] = [
        {"MKT_NM": "KOSPI", "BAS_DD": DAY_A.strftime("%Y%m%d")},
        {"MKT_NM": "KOSPI", "BAS_DD": DAY_B.strftime("%Y%m%d")},
    ]
    _write_blob(
        snapshot_root, EvidenceKind.DAILY_MARKET,
        payload=mixed, source_path=f"krx:daily-market:{DAY_A.isoformat()}",
        retrieved_at=RETRIEVED_EARLY,
    )
    reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,))
    assert reports[0].migrated == 0
    assert sum(reports[0].skipped_blobs.values()) == 2


def test_latest_retrieval_wins_on_duplicates(tmp_path: Path) -> None:
    import json as _json

    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    early = _daily_payload(DAY_A)
    early["records"] = [{"MKT_NM": "KOSPI", "BAS_DD": DAY_A.strftime("%Y%m%d"), "TDD_CLSPRC": "1000"}]
    late = _daily_payload(DAY_A)
    late["records"] = [{"MKT_NM": "KOSPI", "BAS_DD": DAY_A.strftime("%Y%m%d"), "TDD_CLSPRC": "2000"}]
    _write_blob(snapshot_root, EvidenceKind.DAILY_MARKET, payload=early,
                 source_path=f"krx:daily-market:{DAY_A.isoformat()}", retrieved_at=RETRIEVED_EARLY)
    _write_blob(snapshot_root, EvidenceKind.DAILY_MARKET, payload=late,
                 source_path=f"KRX:daily-market:{DAY_A.isoformat()}", retrieved_at=RETRIEVED_LATE)
    reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,))
    assert reports[0].migrated == 1
    from src.data.jobs.krx import KRX_DAILY_MARKET_SOURCE
    from src.data.receipt_catalog import ReceiptCatalog

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    entry = catalog.latest(source=KRX_DAILY_MARKET_SOURCE, natural_keys={DAY_A.isoformat()})[DAY_A.isoformat()]
    stored = _json.loads(Path(entry.payload_path).read_bytes())
    assert stored["records"][0]["TDD_CLSPRC"] == "2000"


def test_konex_rows_are_dropped(tmp_path: Path) -> None:
    import json as _json

    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    _write_blob(
        snapshot_root, EvidenceKind.DAILY_MARKET,
        payload=_daily_payload(DAY_A, markets=("KOSPI", "KONEX")),
        source_path=f"krx:daily-market:{DAY_A.isoformat()}", retrieved_at=RETRIEVED_EARLY,
    )
    reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,))
    assert reports[0].migrated == 1
    from src.data.jobs.krx import KRX_DAILY_MARKET_SOURCE
    from src.data.receipt_catalog import ReceiptCatalog

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    entry = catalog.latest(source=KRX_DAILY_MARKET_SOURCE, natural_keys={DAY_A.isoformat()})[DAY_A.isoformat()]
    stored = _json.loads(Path(entry.payload_path).read_bytes())
    assert {row["MKT_NM"] for row in stored["records"]} == {"KOSPI"}


def test_rerun_is_idempotent_and_never_overwrites(tmp_path: Path) -> None:
    import json as _json

    from src.data.jobs.krx import KRX_DAILY_MARKET_SOURCE, krx_daily_market_scoped_payload

    runtime = _runtime(tmp_path)
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.scoped_ingestion import ScopedBronzeWriter

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    live = krx_daily_market_scoped_payload(
        records=[{"MKT_NM": "KOSPI", "BAS_DD": DAY_A.strftime("%Y%m%d"), "TDD_CLSPRC": "9999"}],
        session=DAY_A, retrieved_at=RETRIEVED_EARLY,
    )
    writer.persist(live)
    before = _json.loads(Path(catalog.latest(source=KRX_DAILY_MARKET_SOURCE, natural_keys={DAY_A.isoformat()})[DAY_A.isoformat()].payload_path).read_bytes())

    snapshot_root = tmp_path / "snapshot"
    _write_blob(
        snapshot_root, EvidenceKind.DAILY_MARKET,
        payload=_daily_payload(DAY_A), source_path=f"krx:daily-market:{DAY_A.isoformat()}",
        retrieved_at=RETRIEVED_LATE,
    )
    reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,))
    assert reports[0].already_answered == 1
    assert reports[0].migrated == 0
    after = _json.loads(Path(catalog.latest(source=KRX_DAILY_MARKET_SOURCE, natural_keys={DAY_A.isoformat()})[DAY_A.isoformat()].payload_path).read_bytes())
    assert after == before
    reports2, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,))
    assert reports2[0].migrated == 0


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    blob = _write_blob(
        snapshot_root, EvidenceKind.DAILY_MARKET,
        payload=_daily_payload(DAY_A), source_path=f"krx:daily-market:{DAY_A.isoformat()}",
        retrieved_at=RETRIEVED_EARLY,
    )
    (blob / "payload.json").write_bytes(b'{"records": []}')
    with pytest.raises(PITDataError):
        _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,))
    from src.data.receipt_catalog import ReceiptCatalog

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    assert catalog.latest(source="krx_daily_market", natural_keys={DAY_A.isoformat()}) == {}


def test_dry_run_persists_nothing(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    for day in (DAY_A, DAY_B):
        _write_blob(
            snapshot_root, EvidenceKind.DAILY_MARKET,
            payload=_daily_payload(day), source_path=f"krx:daily-market:{day.isoformat()}",
            retrieved_at=RETRIEVED_EARLY,
        )
    dry_reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,), dry_run=True)
    from src.data.receipt_catalog import ReceiptCatalog

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    assert catalog.latest(source="krx_daily_market", natural_keys={DAY_A.isoformat(), DAY_B.isoformat()}) == {}
    wet_reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,), dry_run=False)
    assert dry_reports[0].migrated == wet_reports[0].migrated == 2


def test_security_master_as_of_mapping(tmp_path: Path) -> None:
    import json as _json

    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    payload = {
        "as_of": MASTER_DAY.isoformat(),
        "endpoint": "sto/stk_isu_base_info",
        "provider": "KRX",
        "records": [{"ISU_SRT_CD": "005930", "ISU_CD": "KR7005930003", "MKT_TP_NM": "KOSPI"}],
    }
    _write_blob(
        snapshot_root, EvidenceKind.SECURITY_MASTER, payload=payload,
        source_path=f"KRX:historical-master:{MASTER_DAY.isoformat()}", retrieved_at=RETRIEVED_EARLY,
    )
    reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.SECURITY_MASTER,))
    assert reports[0].migrated == 1
    from src.data.receipt_catalog import ReceiptCatalog

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    entry = catalog.latest(source="krx_security_master", natural_keys={MASTER_DAY.isoformat()})[MASTER_DAY.isoformat()]
    assert entry.as_of == MASTER_DAY
    stored = _json.loads(Path(entry.payload_path).read_bytes())
    assert stored["session"] == MASTER_DAY.isoformat()
    assert stored["records"][0]["ISU_SRT_CD"] == "005930"


def _write_raw_blob(snapshot_root: Path, kind: EvidenceKind, *, raw: bytes, source_path: str, retrieved_at: object) -> Path:
    digest = hashlib.sha256(raw).hexdigest()
    blob_dir = snapshot_root / kind.value / digest
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / "payload.json").write_bytes(raw)
    (blob_dir / "receipt.json").write_text(
        json.dumps({"source_path": source_path, "retrieved_at": str(retrieved_at), "ingested_at": str(retrieved_at)}, sort_keys=True),
        encoding="utf-8",
    )
    return blob_dir


def test_edge_classification_and_validation(tmp_path: Path) -> None:
    from src.data.snapshot_migration import _extract_records, _page_session, _parse_day_text, _session_universe

    assert _parse_day_text("") is None
    assert _parse_day_text("not-a-date") is None
    assert _parse_day_text("20171301") is None
    assert _parse_day_text(DAY_A.strftime("%Y%m%d")) == DAY_A
    assert _parse_day_text(DAY_A.isoformat()) == DAY_A
    assert _extract_records([]) is None
    assert _extract_records({"records": "nope"}) is None
    assert _extract_records({"OutBlock_1": [{"BAS_DD": DAY_A.strftime("%Y%m%d")}]}) is not None
    assert _page_session(EvidenceKind.DAILY_MARKET, []) is None
    assert _page_session(EvidenceKind.DAILY_MARKET, {"records": []}) is None
    assert _page_session(EvidenceKind.DAILY_MARKET, {"records": [{"BAS_DD": "bad"}]}) is None
    assert _page_session(EvidenceKind.SECURITY_MASTER, {"records": [{"a": 1}]}) is None
    assert _session_universe(evidence_start=date(2020, 1, 1), last=date(2019, 1, 1)) == frozenset()

    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    kind_root = snapshot_root / EvidenceKind.DAILY_MARKET.value
    kind_root.mkdir(parents=True)
    (kind_root / "orphan").mkdir()
    bad_receipt = kind_root / "badreceipt"
    bad_receipt.mkdir()
    (bad_receipt / "receipt.json").write_text("not json", encoding="utf-8")
    (bad_receipt / "payload.json").write_bytes(b"{}")
    _write_raw_blob(snapshot_root, EvidenceKind.DAILY_MARKET, raw=b"not json",
                     source_path=f"krx:daily-market:{DAY_A.isoformat()}", retrieved_at=RETRIEVED_EARLY.isoformat())
    _write_raw_blob(snapshot_root, EvidenceKind.DAILY_MARKET,
                     raw=json.dumps({"records": []}, sort_keys=True).encode(),
                     source_path="KRX:warmup:daily_market", retrieved_at=RETRIEVED_EARLY.isoformat())
    _write_raw_blob(snapshot_root, EvidenceKind.DAILY_MARKET,
                     raw=json.dumps({"records": [{"MKT_NM": "KOSPI", "BAS_DD": DAY_A.strftime("%Y%m%d")}], "_tag": "aggregate"}, sort_keys=True).encode(),
                     source_path="data/evidence/aggregate.json", retrieved_at=RETRIEVED_EARLY.isoformat())
    _write_raw_blob(snapshot_root, EvidenceKind.DAILY_MARKET,
                     raw=json.dumps({"records": [{"MKT_NM": "KOSPI", "BAS_DD": DAY_A.strftime("%Y%m%d")}], "_tag": "badtime"}, sort_keys=True).encode(),
                     source_path=f"krx:daily-market:{DAY_A.isoformat()}", retrieved_at="not-a-time")
    _write_blob(snapshot_root, EvidenceKind.DAILY_MARKET, payload=_daily_payload(date(2015, 12, 30)),
                 source_path="krx:daily-market:2015-12-30", retrieved_at=RETRIEVED_EARLY)
    saturday = date(2017, 1, 7)
    _write_blob(snapshot_root, EvidenceKind.DAILY_MARKET, payload=_daily_payload(saturday),
                 source_path=f"krx:daily-market:{saturday.isoformat()}", retrieved_at=RETRIEVED_EARLY)
    naive = _daily_payload(DAY_B)
    naive_dir = _write_blob(snapshot_root, EvidenceKind.DAILY_MARKET, payload=naive,
                             source_path=f"krx:daily-market:{DAY_B.isoformat()}", retrieved_at=RETRIEVED_EARLY)
    (naive_dir / "receipt.json").write_text(
        json.dumps({"source_path": f"krx:daily-market:{DAY_B.isoformat()}", "retrieved_at": "2026-09-10T00:00:00",
                     "ingested_at": "2026-09-10T00:00:00"}, sort_keys=True), encoding="utf-8",
    )
    iso_payload = {"session": DAY_C.isoformat(), "records": [{"MKT_NM": "KOSPI", "BAS_DD": DAY_C.isoformat()}]}
    _write_blob(snapshot_root, EvidenceKind.DAILY_MARKET, payload=iso_payload,
                 source_path=f"krx:daily-market:{DAY_C.isoformat()}", retrieved_at=RETRIEVED_EARLY)
    legacy_payload = {"OutBlock_1": [{"MKT_NM": "KOSPI", "BAS_DD": DAY_A.strftime("%Y%m%d")}]}
    _write_raw_blob(snapshot_root, EvidenceKind.DAILY_MARKET,
                     raw=json.dumps(legacy_payload, sort_keys=True).encode(),
                     source_path=f"krx:daily-market:{DAY_A.isoformat()}", retrieved_at=RETRIEVED_EARLY.isoformat())
    reports, _ = _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,), batch_sessions=1)
    assert reports[0].migrated == 2
    assert reports[0].skipped_blobs["invalid_retrieved_at"] == 2
    assert reports[0].skipped_blobs["outside_scope_sessions"] == 2


def test_guards_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="batch_sessions"):
        _migrate(runtime, snapshot_root, kinds=(EvidenceKind.DAILY_MARKET,), batch_sessions=0)
    with pytest.raises(ValueError, match="unsupported kind"):
        _migrate(runtime, snapshot_root, kinds=(EvidenceKind.INVESTOR_FLOW,))  # type: ignore[list-item]
    with pytest.raises(PITDataError, match="scoped Bronze root"):
        _migrate(runtime, runtime.workspace.bronze_root, kinds=(EvidenceKind.DAILY_MARKET,))
    empty_reports, _ = _migrate(tmp_path / "nosuch" if False else runtime, snapshot_root, kinds=(EvidenceKind.SECURITY_MASTER,))
    assert empty_reports[0].migrated == 0

    _write_blob(snapshot_root, EvidenceKind.DAILY_MARKET, payload=_daily_payload(DAY_A),
                 source_path=f"krx:daily-market:{DAY_A.isoformat()}", retrieved_at=RETRIEVED_EARLY)
    _write_blob(snapshot_root, EvidenceKind.SECURITY_MASTER,
                 payload={"as_of": MASTER_DAY.isoformat(), "endpoint": "sto/stk_isu_base_info",
                          "provider": "KRX", "records": [
                              {"ISU_SRT_CD": "005930", "ISU_CD": "KR7005930003", "MKT_TP_NM": "KOSPI"},
                              {"ISU_SRT_CD": "000000", "ISU_CD": "KR7000000000", "MKT_TP_NM": "KONEX"},
                          ]},
                 source_path=f"KRX:historical-master:{MASTER_DAY.isoformat()}", retrieved_at=RETRIEVED_EARLY)
    from src.data.cli import main

    args = ["migrate-snapshot-krx", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"),
            "--snapshot-root", str(snapshot_root), "--kinds", "daily_market", "security_master"]
    assert main(args) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert any(line.get("kind") == "daily_market" for line in lines)
    assert main([*args, "--dry-run"]) == 0

    from src.data.snapshot_migration import _persist_batch
    from src.data.scoped_ingestion import ScopedBronzeWriter
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.jobs.krx import krx_daily_market_scoped_payload

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    assert _persist_batch(writer, [], rejected={}, dry_run=False) == 0
    good = krx_daily_market_scoped_payload(
        records=[{"MKT_NM": "KOSPI", "BAS_DD": DAY_B.strftime("%Y%m%d")}], session=DAY_B, retrieved_at=RETRIEVED_EARLY)
    rejected: dict[str, str] = {}
    assert _persist_batch(writer, [good], rejected=rejected, dry_run=True) == 1
    monkeypatch.setattr(writer, "persist_many", lambda _batch: (_ for _ in ()).throw(PITDataError("batch boom")))
    rejected2: dict[str, str] = {}
    assert _persist_batch(writer, [], rejected=rejected2, dry_run=True) == 0
    real_many = ScopedBronzeWriter.persist_many.__get__(writer, ScopedBronzeWriter)
    calls = {"n": 0}

    def _flaky(batch):
        calls["n"] += 1
        if len(tuple(batch)) > 1 or calls["n"] == 1:
            raise PITDataError("batch boom")
        return real_many(batch)

    monkeypatch.setattr(writer, "persist_many", _flaky)
    assert _persist_batch(writer, [good], rejected=rejected2, dry_run=False) == 1
    assert rejected2 == {}
    from src.core.pit import PITDataError as _PIT

    def _fail(_payload):
        raise _PIT("single boom")

    monkeypatch.setattr(writer, "persist", _fail)
    monkeypatch.setattr(writer, "persist_many", lambda _batch: (_ for _ in ()).throw(_PIT("batch boom")))
    rejected3: dict[str, str] = {}
    bad = krx_daily_market_scoped_payload(
        records=[{"MKT_NM": "KOSPI", "BAS_DD": DAY_C.strftime("%Y%m%d")}], session=DAY_C, retrieved_at=RETRIEVED_EARLY)
    assert _persist_batch(writer, [bad], rejected=rejected3, dry_run=False) == 0
    assert DAY_C.isoformat() in rejected3


def test_closed_branches(tmp_path: Path) -> None:
    from src.data.snapshot_migration import _persist_batch, _read_verified

    runtime = _runtime(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    blob = _write_blob(snapshot_root, EvidenceKind.DAILY_MARKET, payload=_daily_payload(DAY_A),
                        source_path=f"krx:daily-market:{DAY_A.isoformat()}", retrieved_at=RETRIEVED_EARLY)
    with pytest.raises(PITDataError, match="hash mismatch"):
        _read_verified(blob / "payload.json", "0" * 64)
    only_old = tmp_path / "old_snapshot"
    _write_blob(only_old, EvidenceKind.DAILY_MARKET, payload=_daily_payload(date(2015, 12, 30)),
                source_path="krx:daily-market:2015-12-30", retrieved_at=RETRIEVED_EARLY)
    reports, _ = _migrate(runtime, only_old, kinds=(EvidenceKind.DAILY_MARKET,))
    assert reports[0].migrated == 0
    assert sum(reports[0].skipped_blobs.values()) == 1
    konex_only = tmp_path / "konex_snapshot"
    _write_blob(konex_only, EvidenceKind.DAILY_MARKET, payload=_daily_payload(DAY_A, markets=("KONEX",)),
                source_path=f"krx:daily-market:{DAY_A.isoformat()}", retrieved_at=RETRIEVED_EARLY)
    reports, _ = _migrate(runtime, konex_only, kinds=(EvidenceKind.DAILY_MARKET,))
    assert reports[0].migrated == 0
    assert reports[0].rejected_sessions[DAY_A.isoformat()] == "no_live_market_records"

    from src.data.jobs.krx import krx_daily_market_scoped_payload
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.scoped_ingestion import ScopedBronzeWriter

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    invalid = krx_daily_market_scoped_payload(records=[], session=DAY_B, retrieved_at=RETRIEVED_EARLY)
    object.__setattr__(invalid, "payload", b"")
    rejected: dict[str, str] = {}
    assert _persist_batch(writer, [invalid], rejected=rejected, dry_run=True) == 0
    assert DAY_B.isoformat() in rejected
