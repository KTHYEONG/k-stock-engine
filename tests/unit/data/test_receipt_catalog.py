from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.data.schemas import PITDataError


def _payload_file(tmp_path: Path, name: str, data: bytes) -> tuple[str, Path]:
    path = tmp_path / "payloads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), path


def _entry(
    tmp_path: Path,
    *,
    source: str = "krx_daily_market",
    natural_key: str = "2024-01-02",
    as_of: date | None = date(2024, 1, 2),
    fiscal_period: str | None = None,
    status: EvidenceStatus = EvidenceStatus.SUCCESS,
    body: bytes = b'{"records": [1]}',
    retrieved_at: datetime = datetime(2024, 1, 3, tzinfo=UTC),
) -> ReceiptIndexEntry:
    content_hash, payload_path = _payload_file(tmp_path, f"{source}-{natural_key}-{retrieved_at.isoformat()}.json", body)
    return ReceiptIndexEntry(
        source=source,
        natural_key=natural_key,
        as_of=as_of,
        fiscal_period=fiscal_period,
        status=status,
        content_hash=content_hash,
        retrieved_at=retrieved_at,
        payload_path=payload_path,
    )


def test_successful_receipt_covers_exact_key(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entry = _entry(tmp_path)
    revision = catalog.publish((entry,))

    assert revision.row_count == 1
    found = catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])
    assert found["2024-01-02"].content_hash == entry.content_hash
    assert catalog.latest(source="krx_daily_market", natural_keys=["2024-01-03"]) == {}
    assert catalog.latest(source="other_source", natural_keys=["2024-01-02"]) == {}


def test_unsuccessful_receipt_is_not_coverage(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    for index, status in enumerate(
        (
            EvidenceStatus.EMPTY,
            EvidenceStatus.PROVIDER_UNAVAILABLE,
            EvidenceStatus.PROVIDER_ERROR,
            EvidenceStatus.EXTRACTION_FAILED,
        )
    ):
        catalog.publish(
            (_entry(tmp_path, natural_key=f"key-{index}", status=status, body=f"b{index}".encode()),)
        )

    assert catalog.successful_keys(source="krx_daily_market") == frozenset()
    assert catalog.latest(source="krx_daily_market", natural_keys=["key-0"])["key-0"].status == EvidenceStatus.EMPTY


def test_newer_correction_supersedes_current_state(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    first = _entry(tmp_path, body=b"v1", retrieved_at=datetime(2024, 1, 3, tzinfo=UTC))
    second = _entry(tmp_path, body=b"v22", retrieved_at=datetime(2024, 1, 4, tzinfo=UTC))
    catalog.publish((first,))
    revision = catalog.publish((second,))

    assert catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])["2024-01-02"].content_hash == second.content_hash
    assert revision.row_count == 1
    assert Path(first.payload_path).is_file()
    assert Path(second.payload_path).is_file()

    catalog.publish((first,))
    assert catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])["2024-01-02"].content_hash == second.content_hash


def test_timestamp_conflict_fails_closed(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    moment = datetime(2024, 1, 3, tzinfo=UTC)
    catalog.publish((_entry(tmp_path, body=b"v1", retrieved_at=moment),))

    with pytest.raises(PITDataError, match="conflict"):
        catalog.publish((_entry(tmp_path, body=b"v2-other", retrieved_at=moment),))

    revision = catalog.publish((_entry(tmp_path, body=b"v1", retrieved_at=moment),))
    assert revision.row_count == 1


def test_fiscal_filter_uses_period(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    catalog.publish(
        (
            _entry(
                tmp_path,
                source="financial_facts",
                natural_key="00126380:2018:11011",
                as_of=date(2019, 1, 2),
                fiscal_period="2018Q4",
                retrieved_at=datetime(2019, 1, 3, tzinfo=UTC),
            ),
        )
    )
    catalog.publish(
        (
            _entry(
                tmp_path,
                source="financial_facts",
                natural_key="00126380:2019:11013",
                as_of=date(2019, 5, 16),
                fiscal_period="2019Q1",
                retrieved_at=datetime(2019, 5, 17, tzinfo=UTC),
            ),
            _entry(
                tmp_path,
                source="financial_facts",
                natural_key="no-period",
                as_of=date(2019, 5, 16),
                fiscal_period=None,
                retrieved_at=datetime(2019, 5, 17, tzinfo=UTC),
            ),
        )
    )

    assert catalog.successful_keys(source="financial_facts", fiscal_start="2019Q1") == frozenset(
        {"00126380:2019:11013"}
    )
    assert catalog.successful_keys(source="financial_facts") == frozenset(
        {"00126380:2018:11011", "00126380:2019:11013", "no-period"}
    )


def test_publish_rejects_unvalidated_entries(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    valid = _entry(tmp_path)

    with pytest.raises(PITDataError, match="source and natural key"):
        catalog.publish((_entry(tmp_path, natural_key="   "),))
    with pytest.raises(PITDataError, match="missing"):
        catalog.publish(
            (
                ReceiptIndexEntry(
                    source="krx_daily_market",
                    natural_key="absent",
                    as_of=date(2024, 1, 2),
                    fiscal_period=None,
                    status=EvidenceStatus.SUCCESS,
                    content_hash="0" * 64,
                    retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
                    payload_path=tmp_path / "no-such-file.json",
                ),
            )
        )
    content_hash, payload_path = _payload_file(tmp_path, "real.json", b"real")
    with pytest.raises(PITDataError, match="hash mismatch"):
        catalog.publish(
            (
                ReceiptIndexEntry(
                    source="krx_daily_market",
                    natural_key="tampered",
                    as_of=date(2024, 1, 2),
                    fiscal_period=None,
                    status=EvidenceStatus.SUCCESS,
                    content_hash="f" * 64,
                    retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
                    payload_path=payload_path,
                ),
            )
        )
    assert content_hash
    assert valid.content_hash
    assert catalog.latest(source="krx_daily_market", natural_keys=["tampered"]) == {}


def test_snapshot_failures_raise(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    catalog.publish((_entry(tmp_path),))

    (tmp_path / "catalog" / "latest.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(PITDataError, match="unreadable"):
        catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])

    root = tmp_path / "catalog2"
    root.mkdir(parents=True)
    (root / "rev.json").write_text('{"not": "a list"}', encoding="utf-8")
    (root / "latest.json").write_text('{"revision": "rev.json"}', encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid schema"):
        ReceiptCatalog(root).latest(source="s", natural_keys=["k"])

    root3 = tmp_path / "catalog3"
    root3.mkdir(parents=True)
    (root3 / "rev.json").write_text('[{"source": "s"}]', encoding="utf-8")
    (root3 / "latest.json").write_text('{"revision": "rev.json"}', encoding="utf-8")
    with pytest.raises(PITDataError, match="retrieved_at"):
        ReceiptCatalog(root3).latest(source="s", natural_keys=["k"])

    root4 = tmp_path / "catalog4"
    root4.mkdir(parents=True)
    (root4 / "rev.json").write_text('["stray"]', encoding="utf-8")
    (root4 / "latest.json").write_text('{"revision": "rev.json"}', encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid schema"):
        ReceiptCatalog(root4).latest(source="s", natural_keys=["k"])


def _publish_worker(root: str, payload_dir: str, worker_index: int, count: int) -> None:
    import hashlib
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    from src.data.receipt_catalog import EvidenceStatus as _Status
    from src.data.receipt_catalog import ReceiptCatalog as _Catalog
    from src.data.receipt_catalog import ReceiptIndexEntry as _Entry

    catalog = _Catalog(_Path(root))
    for offset in range(count):
        key = f"worker-{worker_index}-key-{offset:03d}"
        payload_path = _Path(payload_dir) / f"{key}.json"
        catalog.publish((
            _Entry(
                source="concurrent",
                natural_key=key,
                as_of=None,
                fiscal_period=None,
                status=_Status.SUCCESS,
                content_hash=hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
                payload_path=payload_path,
            ),
        ))


def test_concurrent_publishes_keep_all_entries(tmp_path: Path) -> None:
    import multiprocessing

    root = tmp_path / "catalog"
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    per_worker = 50
    for worker_index in (0, 1):
        for offset in range(per_worker):
            (payload_dir / f"worker-{worker_index}-key-{offset:03d}.json").write_bytes(b'{"records": []}')

    with multiprocessing.Pool(processes=2) as pool:
        pool.starmap(
            _publish_worker,
            [(str(root), str(payload_dir), 0, per_worker), (str(root), str(payload_dir), 1, per_worker)],
        )

    catalog = ReceiptCatalog(root)
    keys = [f"worker-{w}-key-{o:03d}" for w in (0, 1) for o in range(per_worker)]
    assert len(catalog.latest(source="concurrent", natural_keys=keys)) == 2 * per_worker


def test_lock_released_after_failure(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    catalog.publish((_entry(tmp_path),))

    content_hash, payload_path = _payload_file(tmp_path, "real.json", b"real")
    assert content_hash
    with pytest.raises(PITDataError, match="hash mismatch"):
        catalog.publish((
            ReceiptIndexEntry(
                source="krx_daily_market",
                natural_key="tampered",
                as_of=date(2024, 1, 2),
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash="f" * 64,
                retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
                payload_path=payload_path,
            ),
        ))

    (tmp_path / "catalog" / "latest.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(PITDataError, match="unreadable"):
        catalog.publish((_entry(tmp_path, natural_key="after-corruption"),))

    (tmp_path / "catalog" / "latest.json").unlink()
    follow_up = catalog.publish((_entry(tmp_path, natural_key="recovered"),))

    assert follow_up.row_count == 1
    assert catalog.latest(source="krx_daily_market", natural_keys=["recovered"])


def test_no_stray_temp_files_after_publish(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    catalog.publish((_entry(tmp_path),))
    catalog.publish((_entry(tmp_path, natural_key="second"),))
    _hash, payload_path = _payload_file(tmp_path, "real.json", b"real")
    assert _hash
    with pytest.raises(PITDataError, match="hash mismatch"):
        catalog.publish((
            ReceiptIndexEntry(
                source="krx_daily_market",
                natural_key="tampered",
                as_of=date(2024, 1, 2),
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash="f" * 64,
                retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
                payload_path=payload_path,
            ),
        ))

    assert list((tmp_path / "catalog").glob("*.tmp")) == []
    assert list((tmp_path / "catalog").glob(".*.tmp")) == []


def test_identical_input_yields_identical_revision(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entries = (_entry(tmp_path), _entry(tmp_path, natural_key="second"))

    first = catalog.publish(entries)
    second = catalog.publish(entries)

    assert second.content_hash == first.content_hash
    assert second.path.read_text(encoding="utf-8") == first.path.read_text(encoding="utf-8")
