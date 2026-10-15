"""Catalog-driven Silver investor-flow tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.core.pit import EvidenceKind, PITDataError
from src.data.investor_flow_silver import REVISION as INVESTOR_FLOW_SILVER_REVISION
from src.data.investor_flow_silver import (
    InvestorFlowSilverPolicy,
    materialize_investor_flow_silver,
)
from src.data.receipt_catalog import BlobEntry, ReceiptCatalog

SESSIONS = (date(2026, 3, 4), date(2026, 3, 5), date(2026, 3, 6))


def _write_universe(universe_root: Path) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    return publish_dataset(
        layer_root=Path(universe_root),
        identity=DatasetIdentity(
            kind="ordinary_universe",
            layer=DatasetLayer.SILVER,
            policy_version="krx-ordinary-equity-v1",
            inputs={},
            params={"calendar": ",".join(day.isoformat() for day in SESSIONS)},
        ),
        partitions={
            f"session={day.isoformat()}/part.parquet": pl.DataFrame(
                {"session": [day], "instrument_id": ["KRX:005930"], "ticker": ["005930"], "eligible": [True]}
            )
            for day in SESSIONS
        },
    ).path


def _row(session: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": session,
        "tjj0000": "-100", "tjj0001": "-50", "tjj0002": "-30", "tjj0003": "-20",
        "tjj0004": "-10", "tjj0005": "-10", "tjj0006": "-8", "tjj0007": "100",
        "tjj0008": "927", "tjj0009": "-800", "tjj0010": "-28", "tjj0011": "29",
        "tjj0016": "-828", "tjj0017": "129", "tjj0018": "-228",
        "close": "50000", "volume": "10000", "value": "500",
    }
    row.update(overrides)
    return row


def _ls_payload(symbol: str, rows: list[dict[str, object]], *, start: str, end: str) -> dict[str, object]:
    return {
        "provider": "LS",
        "query": {"symbol": symbol, "start": start, "end": end},
        "rows": rows,
    }


def _ls_envelope(symbol: str, rows: list[dict[str, object]], *, start: str, end: str) -> dict[str, object]:
    return {
        "envelope": "raw-rows-v1",
        "provider": "LS",
        "endpoint": "t1702",
        "query": {"symbol": symbol, "start": start, "end": end},
        "rows": rows,
    }


def _seed_blob(
    catalog: ReceiptCatalog, bronze_root: Path, payload: dict[str, object], *,
    source: str = "ls_investor_flow", usable: bool = True, reason: str | None = None,
) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze_root / "investor_flow" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    catalog.publish(
        [],
        blobs=[
            BlobEntry(
                content_hash=digest, kind=EvidenceKind.INVESTOR_FLOW, source=source,
                usable=usable, unusable_reason=(None if usable else (reason or "seeded")),
                retrieved_at=datetime(2026, 3, 6, tzinfo=UTC),
                payload_path=target / "payload.json",
            )
        ],
    )
    return digest


def _roots(tmp_path: Path) -> tuple[Path, Path, Path, ReceiptCatalog]:
    bronze_root = tmp_path / "bronze"
    universe_root = tmp_path / "silver"
    silver_root = tmp_path / "silver"
    _write_universe(universe_root)
    return bronze_root, universe_root, silver_root, ReceiptCatalog(bronze_root / "catalog")


def test_ignores_rows_outside_query_range(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(
        catalog, bronze_root,
        _ls_payload(
            "005930", [_row("20260304"), _row("20260305"), _row("20260306")],
            start="2026-03-04", end="2026-03-05",
        ),
    )
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 2


def _frame(result_path: Path) -> pl.DataFrame:
    return pl.read_parquet(result_path / "year=2026" / "part.parquet")


def test_builds_from_usable_catalog_blobs_only(tmp_path: Path) -> None:
    """Unusable blobs never read: one usable plus one unusable file that would fail parsing."""
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(catalog, bronze_root, _ls_payload("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    bad_raw = b"not json"
    bad_digest = hashlib.sha256(bad_raw).hexdigest()
    bad_dir = bronze_root / "investor_flow" / bad_digest
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "payload.json").write_bytes(bad_raw)
    catalog.publish(
        [],
        blobs=[
            BlobEntry(
                content_hash=bad_digest, kind=EvidenceKind.INVESTOR_FLOW, source="ls_investor_flow",
                usable=False, unusable_reason="seeded", retrieved_at=datetime(2026, 3, 6, tzinfo=UTC),
                payload_path=bad_dir / "payload.json",
            )
        ],
    )
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 1
    assert result.raw_pages == 1
    assert INVESTOR_FLOW_SILVER_REVISION == "ls-t1702-net-shares-v2"


def test_uncatalogued_files_ignored(tmp_path: Path) -> None:
    """Uncatalogued files ignored: an extra valid page on disk not in the catalog."""
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(catalog, bronze_root, _ls_payload("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    extra = {"provider": "LS", "query": {"symbol": "999999", "start": "2026-03-04", "end": "2026-03-04"},
             "rows": [_row("20260304")]}
    raw = json.dumps(extra, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze_root / "investor_flow" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 1
    assert "999999" not in _frame(result.dataset_path)["ticker"].to_list()


def test_both_layouts_parse_identically(tmp_path: Path) -> None:
    """Both layouts parse identically: historical page vs RAW_ROWS_V1 envelope."""
    bronze_a, universe_a, silver_a, _ = _roots(tmp_path / "a")
    bronze_b, universe_b, silver_b, _ = _roots(tmp_path / "b")
    # separate catalogs per roots
    from src.data.receipt_catalog import ReceiptCatalog as _Catalog

    catalog_a = _Catalog(bronze_a / "catalog")
    catalog_b = _Catalog(bronze_b / "catalog")
    rows = [_row("20260304")]
    _seed_blob(catalog_a, bronze_a, _ls_payload("005930", rows, start="2026-03-04", end="2026-03-06"))
    _seed_blob(catalog_b, bronze_b, _ls_envelope("005930", rows, start="2026-03-04", end="2026-03-06"))
    first = materialize_investor_flow_silver(
        catalog=catalog_a, universe_root=universe_a, silver_root=silver_a, workers=1
    )
    second = materialize_investor_flow_silver(
        catalog=catalog_b, universe_root=universe_b, silver_root=silver_b, workers=1
    )
    left = _frame(first.dataset_path).drop("source_hash")
    right = _frame(second.dataset_path).drop("source_hash")
    assert left.equals(right)


def test_isolates_identity_violations(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    bad = _row("20260304", tjj0000="-99")
    _seed_blob(catalog, bronze_root, _ls_payload("005930", [bad, _row("20260305")], start="2026-03-04", end="2026-03-06"))
    _seed_blob(catalog, bronze_root, _ls_payload("005930", [_row("20260304")], start="2026-03-04", end="2026-03-04"))
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.identity_violation_cells == 1
    assert _frame(result.dataset_path)["session"].to_list() == [date(2026, 3, 5)]


def test_assigns_next_session_availability_and_tail(tmp_path: Path) -> None:
    from src.core.time import KRX_TZ as _TZ

    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(catalog, bronze_root, _ls_payload("005930", [_row("20260304", volume="100")], start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    from datetime import datetime as _dt

    assert _frame(result.dataset_path)["available_at"].to_list() == [_dt(2026, 3, 5, 8, 0, tzinfo=_TZ)]
    assert result.retail_exceeds_volume_rows == 1


def test_excludes_calendar_tail(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(catalog, bronze_root, _ls_payload("005930", [_row("20260306")], start="2026-03-06", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 0
    assert result.unavailable_tail_cells == 1


def test_collapses_identical_duplicates_but_not_conflicts(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    rows = [_row("20260304")]
    _seed_blob(catalog, bronze_root, _ls_payload("005930", rows, start="2026-03-04", end="2026-03-06"))
    _seed_blob(catalog, bronze_root, dict(_ls_envelope("005930", rows, start="2026-03-04", end="2026-03-06"), extra="a"))
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 1
    assert result.conflict_cells == 0


def test_rejects_hash_mismatch_and_bad_rows(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    digest = _seed_blob(catalog, bronze_root, _ls_payload("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    (bronze_root / "investor_flow" / digest / "payload.json").write_bytes(b"tampered")
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_skips_zero_flow_dateless_row(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    zero = dict.fromkeys(
        ("tjj0000", "tjj0001", "tjj0002", "tjj0003", "tjj0004", "tjj0005", "tjj0006",
         "tjj0007", "tjj0008", "tjj0009", "tjj0010", "tjj0011", "tjj0016", "tjj0017", "tjj0018"), "0")
    rows = [_row("20260304"), _row("", **zero)]
    _seed_blob(catalog, bronze_root, _ls_payload("005930", rows, start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.dateless_rows == 1
    assert result.rows == 1


def test_manifest_has_no_legacy_counters(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(catalog, bronze_root, _ls_payload("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy_version"] == "ls-t1702-net-shares-v2"
    for key in ("ignored_records_only_pages", "foreign_provider_pages", "negative_cells"):
        assert key not in manifest
        assert key not in manifest.get("inputs", {})


def test_rejects_invalid_policy_and_workers(tmp_path: Path) -> None:
    _, universe_root, silver_root, catalog = _roots(tmp_path)
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=0
        )
    with pytest.raises(PITDataError, match="availability lag"):
        materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_root,
            policy=InvestorFlowSilverPolicy(available_session_lag=0),
        )


def test_fail_closed_on_unreadable_and_tampered_blobs(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    digest = _seed_blob(catalog, bronze_root, _ls_payload("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    path = bronze_root / "investor_flow" / digest / "payload.json"
    path.chmod(0o000)
    try:
        with pytest.raises(PITDataError):
            materialize_investor_flow_silver(
                catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
            )
    finally:
        path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_fail_closed_on_calendar_and_dateless_violations(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(catalog, bronze_root, _ls_payload("005930", [_row("20260307")], start="2026-03-04", end="2026-03-07"))
    with pytest.raises(PITDataError, match="outside certified calendar"):
        materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
        )
    bronze2, universe2, silver2, catalog2 = _roots(tmp_path / "dateless")
    _seed_blob(catalog2, bronze2, _ls_payload("005930", [_row("")], start="2026-03-04", end="2026-03-06"))
    with pytest.raises(PITDataError, match="dateless row carries flow values"):
        materialize_investor_flow_silver(
            catalog=catalog2, universe_root=universe2, silver_root=silver2, workers=1
        )
    with pytest.raises(PITDataError, match="no usable"):
        materialize_investor_flow_silver(
            catalog=ReceiptCatalog(tmp_path / "empty" / "catalog"),
            universe_root=universe_root, silver_root=silver_root, workers=1,
        )


def test_fail_closed_on_malformed_blobs(tmp_path: Path) -> None:
    cases = [
        b"not json",
        b"[1, 2]",
        json.dumps({"provider": "KIS", "query": {"symbol": "005930", "start": "2026-03-04", "end": "2026-03-04"}, "rows": [{"date": "20260304"}]}).encode(),
        json.dumps({"provider": "", "query": {}, "rows": []}).encode(),
        json.dumps({"provider": "LS", "query": {"symbol": "", "start": "2026-03-04", "end": "2026-03-04"}, "rows": [{"date": "20260304"}]}).encode(),
        json.dumps({"provider": "LS", "query": {"symbol": "005930", "start": "2026-03-04", "end": "2026-03-04"}, "rows": []}).encode(),
        json.dumps({"provider": "LS", "query": {"symbol": "005930", "start": "2026-03-04", "end": "2026-03-04"}, "rows": ["nope"]}).encode(),
    ]
    for index, raw in enumerate(cases):
        bronze_root, universe_root, silver_root, catalog = _roots(tmp_path / f"bad-{index}")
        digest = hashlib.sha256(raw).hexdigest()
        target = bronze_root / "investor_flow" / digest
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.json").write_bytes(raw)
        catalog.publish(
            [],
            blobs=[
                BlobEntry(
                    content_hash=digest, kind=EvidenceKind.INVESTOR_FLOW, source="ls_investor_flow",
                    usable=True, unusable_reason=None,
                    retrieved_at=datetime(2026, 3, 6, tzinfo=UTC),
                    payload_path=target / "payload.json",
                )
            ],
        )
        with pytest.raises(PITDataError):
            materialize_investor_flow_silver(
                catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
            )


def test_packed_bronze_builds_identical_flow(tmp_path: Path) -> None:
    from src.data.datasets import load_manifest
    from src.data.payload_store import pack_loose

    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(
        catalog, bronze_root,
        _ls_payload(
            "005930", [_row("20260304"), _row("20260305"), _row("20260306")],
            start="2026-03-04", end="2026-03-05",
        ),
    )
    first = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    pack_loose(bronze_root, kinds=["investor_flow"], offsite_paths=None, remove=True)
    silver2 = tmp_path / "silver2"
    _write_universe(silver2)
    second = materialize_investor_flow_silver(
        catalog=catalog, universe_root=silver2, silver_root=silver2, workers=1
    )
    assert first.dataset_id == second.dataset_id
    assert [
        (partition.path, partition.sha256) for partition in load_manifest(first.dataset_path).partitions
    ] == [
        (partition.path, partition.sha256) for partition in load_manifest(second.dataset_path).partitions
    ]


def test_multi_year_build_publishes_paths_not_frames(tmp_path: Path, monkeypatch) -> None:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    sessions = (date(2025, 12, 29), date(2025, 12, 30), date(2026, 1, 2), date(2026, 1, 5))
    bronze_root = tmp_path / "bronze"
    universe_root = tmp_path / "silver"
    silver_root = tmp_path / "silver"
    publish_dataset(
        layer_root=universe_root,
        identity=DatasetIdentity(
            kind="ordinary_universe",
            layer=DatasetLayer.SILVER,
            policy_version="krx-ordinary-equity-v1",
            inputs={},
            params={"calendar": ",".join(day.isoformat() for day in sessions)},
        ),
        partitions={
            f"session={day.isoformat()}/part.parquet": pl.DataFrame(
                {"session": [day], "instrument_id": ["KRX:005930"], "ticker": ["005930"], "eligible": [True]}
            )
            for day in sessions
        },
    )
    catalog = ReceiptCatalog(bronze_root / "catalog")
    _seed_blob(
        catalog, bronze_root,
        _ls_payload(
            "005930",
            [_row("20251229"), _row("20251230"), _row("20260102"), _row("20260105")],
            start="2025-12-29", end="2026-01-05",
        ),
    )
    import src.data.investor_flow_silver as flow_module

    captured: dict[str, object] = {}
    original = flow_module.publish_dataset

    def _spy(*, layer_root, identity, partitions, details=None, checks=()):  # type: ignore[no-untyped-def]
        captured.update(partitions)
        return original(
            layer_root=layer_root, identity=identity, partitions=partitions,
            details=details, checks=checks,
        )

    monkeypatch.setattr(flow_module, "publish_dataset", _spy)
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    from src.data.datasets import load_manifest

    assert result.rows == 3
    assert set(captured) == {"year=2025/part.parquet", "year=2026/part.parquet"}
    assert all(isinstance(value, Path) for value in captured.values())
    manifest = load_manifest(result.dataset_path)
    assert [(part.path, part.rows) for part in manifest.partitions] == [
        ("year=2025/part.parquet", 2), ("year=2026/part.parquet", 1),
    ]
    assert dict(manifest.details) == {
        "universe_dataset_id": next(universe_root.glob("ordinary_universe_*")).name,
        "partitions": [{"year": 2025, "rows": 2}, {"year": 2026, "rows": 1}],
        "raw_pages": 1,
        "conflict_cells": 0,
        "identity_violation_cells": 0,
        "identity_violation_keys": [],
        "unavailable_tail_cells": 1,
        "retail_exceeds_volume_rows": 0,
        "dateless_rows": 0,
        "tickers": 1,
    }


def test_rebuild_is_republish_noop(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(
        catalog, bronze_root,
        _ls_payload("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"),
    )
    first = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    before = {
        path.relative_to(first.dataset_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in first.dataset_path.rglob("*") if path.is_file()
    }
    second = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert second.dataset_id == first.dataset_id
    assert second.dataset_path == first.dataset_path
    assert {
        path.relative_to(second.dataset_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in second.dataset_path.rglob("*") if path.is_file()
    } == before


def test_all_rows_outside_query_range_publishes_empty(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    _seed_blob(
        catalog, bronze_root,
        _ls_payload("005930", [_row("20260306")], start="2026-03-04", end="2026-03-05"),
    )
    result = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 0


def _seed_pages(catalog: ReceiptCatalog, bronze_root: Path, count: int) -> list[str]:
    digests: list[str] = []
    for idx in range(count):
        symbol = f"{100000 + idx:06d}"
        digests.append(
            _seed_blob(
                catalog, bronze_root,
                _ls_payload(symbol, [_row("20260304"), _row("20260305")], start="2026-03-04", end="2026-03-06"),
            )
        )
    return digests


def _manifest_details(dataset_path: Path) -> dict[str, object]:
    return dict(json.loads((dataset_path / "manifest.json").read_text(encoding="utf-8")).get("details", {}))


def test_cold_warm_partial_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.datasets as datasets
    from src.data.investor_flow_silver import parse_fingerprint
    from src.data.parse_cache import ParseCache

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 6, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(datasets, "datetime", FixedDatetime)
    bronze_root, universe_root, _, catalog = _roots(tmp_path)
    _seed_pages(catalog, bronze_root, 4)
    silver_cold = tmp_path / "silver-cold"
    silver_empty = tmp_path / "silver-empty"
    silver_warm = tmp_path / "silver-warm"
    silver_partial = tmp_path / "silver-partial"
    for path in (silver_cold, silver_empty, silver_warm, silver_partial):
        _write_universe(path)
    cold = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_cold, workers=1
    )
    cache_path = tmp_path / "parse.sqlite"
    with ParseCache(cache_path, namespace="ls_investor_flow-v1", fingerprint=parse_fingerprint()) as cache:
        empty = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_empty, workers=1, parse_cache=cache
        )
        warmed = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_warm, workers=1, parse_cache=cache
        )
        hashes = [blob.content_hash for blob in catalog.blobs(source="ls_investor_flow", usable=True)]
        cache.prune(set(hashes[:2]))
        partial = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_partial, workers=1, parse_cache=cache
        )
    assert cold.dataset_id == empty.dataset_id == warmed.dataset_id == partial.dataset_id
    expected_files = {
        path.relative_to(cold.dataset_path): path.read_bytes()
        for path in cold.dataset_path.rglob("*") if path.is_file()
    }
    for result in (empty, warmed, partial):
        assert {
            path.relative_to(result.dataset_path): path.read_bytes()
            for path in result.dataset_path.rglob("*") if path.is_file()
        } == expected_files
        assert result.raw_pages == cold.raw_pages == 4


def test_warm_run_does_not_read_bronze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.investor_flow_silver as flow_module
    from src.data.investor_flow_silver import parse_fingerprint
    from src.data.parse_cache import ParseCache

    bronze_root, universe_root, _, catalog = _roots(tmp_path)
    _seed_pages(catalog, bronze_root, 2)
    silver_a = tmp_path / "silver-a"
    silver_b = tmp_path / "silver-b"
    _write_universe(silver_a)
    _write_universe(silver_b)
    cache_path = tmp_path / "parse.sqlite"
    with ParseCache(cache_path, namespace="ls_investor_flow-v1", fingerprint=parse_fingerprint()) as cache:
        first = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_a, workers=1, parse_cache=cache
        )

        def _boom(path: Path) -> bytes:
            raise AssertionError("bronze must not be read on full hit")

        monkeypatch.setattr(flow_module, "read_payload", _boom)
        second = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_b, workers=1, parse_cache=cache
        )
    assert first.dataset_id == second.dataset_id


def test_cache_cold_path_still_fail_closed(tmp_path: Path) -> None:
    from src.data.investor_flow_silver import parse_fingerprint
    from src.data.parse_cache import ParseCache

    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    digest = _seed_blob(
        catalog, bronze_root,
        _ls_payload("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"),
    )
    (bronze_root / "investor_flow" / digest / "payload.json").write_bytes(b"tampered")
    with ParseCache(tmp_path / "parse.sqlite", namespace="n", fingerprint=parse_fingerprint()) as cache:
        with pytest.raises(PITDataError):
            materialize_investor_flow_silver(
                catalog=catalog, universe_root=universe_root, silver_root=silver_root, workers=1, parse_cache=cache
            )
        assert cache.get_many([digest]) == {}


def test_calendar_guard_on_hit(tmp_path: Path) -> None:
    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.investor_flow_silver import parse_fingerprint
    from src.data.parse_cache import ParseCache

    bronze_root, universe_root, _, catalog = _roots(tmp_path)
    _seed_blob(
        catalog, bronze_root,
        _ls_payload("005930", [_row("20260306")], start="2026-03-06", end="2026-03-06"),
    )
    silver_warm = tmp_path / "silver-warm"
    _write_universe(silver_warm)
    with ParseCache(tmp_path / "parse.sqlite", namespace="n", fingerprint=parse_fingerprint()) as cache:
        materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_warm, workers=1, parse_cache=cache
        )
        narrow_sessions = (date(2026, 3, 4), date(2026, 3, 5))
        narrow_root = tmp_path / "silver-narrow"
        publish_dataset(
            layer_root=narrow_root,
            identity=DatasetIdentity(
                kind="ordinary_universe", layer=DatasetLayer.SILVER, policy_version="krx-ordinary-equity-v1",
                inputs={}, params={"calendar": ",".join(day.isoformat() for day in narrow_sessions)},
            ),
            partitions={
                f"session={day.isoformat()}/part.parquet": pl.DataFrame(
                    {"session": [day], "instrument_id": ["KRX:005930"], "ticker": ["005930"], "eligible": [True]}
                )
                for day in narrow_sessions
            },
        )
        with pytest.raises(PITDataError, match="outside certified calendar"):
            materialize_investor_flow_silver(
                catalog=catalog, universe_root=narrow_root, silver_root=tmp_path / "silver-fail",
                workers=1, parse_cache=cache,
            )


def test_identity_violations_preserved_cold_vs_warm(tmp_path: Path) -> None:
    from src.data.investor_flow_silver import parse_fingerprint
    from src.data.parse_cache import ParseCache

    bronze_root, universe_root, _, catalog = _roots(tmp_path)
    bad = _row("20260304", tjj0000="-99")
    _seed_blob(
        catalog, bronze_root,
        _ls_payload("005930", [bad, _row("20260305")], start="2026-03-04", end="2026-03-06"),
    )
    silver_cold = tmp_path / "silver-cold"
    silver_warm = tmp_path / "silver-warm"
    silver_warm2 = tmp_path / "silver-warm2"
    for path in (silver_cold, silver_warm, silver_warm2):
        _write_universe(path)
    cold = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_cold, workers=1
    )
    with ParseCache(tmp_path / "parse.sqlite", namespace="n", fingerprint=parse_fingerprint()) as cache:
        materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_warm, workers=1, parse_cache=cache
        )
        warm = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_warm2, workers=1, parse_cache=cache
        )
    assert cold.dataset_id == warm.dataset_id
    assert _manifest_details(cold.dataset_path)["identity_violation_keys"] == _manifest_details(
        warm.dataset_path
    )["identity_violation_keys"]


def test_order_independence_of_catalog_tail(tmp_path: Path) -> None:
    from src.data.investor_flow_silver import parse_fingerprint
    from src.data.parse_cache import ParseCache

    bronze_root, universe_root, _, catalog = _roots(tmp_path)
    _seed_pages(catalog, bronze_root, 3)
    silver_seed = tmp_path / "silver-seed"
    _write_universe(silver_seed)
    with ParseCache(tmp_path / "parse.sqlite", namespace="n", fingerprint=parse_fingerprint()) as cache:
        materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_seed, workers=1, parse_cache=cache
        )
        previous_hashes = [blob.content_hash for blob in catalog.blobs(source="ls_investor_flow")]
        for index in range(100):
            digest = _seed_blob(
                catalog, bronze_root,
                _ls_payload(f"{200000 + index:06d}", [_row("20260304")], start="2026-03-04", end="2026-03-06"),
            )
            if previous_hashes[0] < digest < previous_hashes[-1]:
                break
        else:
            pytest.fail("fixture must add a page inside the existing hash order")
        assert len(list(catalog.blobs(source="ls_investor_flow"))) > len(previous_hashes)
        silver_warm = tmp_path / "silver-warm"
        silver_cold = tmp_path / "silver-cold"
        _write_universe(silver_warm)
        _write_universe(silver_cold)
        warm = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_warm, workers=1, parse_cache=cache
        )
        cold = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_cold, workers=1
        )
    assert warm.dataset_id == cold.dataset_id


def test_corrupted_cache_value_reparses(tmp_path: Path) -> None:
    import json as _json
    import sqlite3
    import zlib

    from src.data.investor_flow_silver import parse_fingerprint
    from src.data.parse_cache import ParseCache

    bronze_root, universe_root, _, catalog = _roots(tmp_path)
    _seed_pages(catalog, bronze_root, 2)
    silver_seed = tmp_path / "silver-seed"
    silver_warm = tmp_path / "silver-warm"
    silver_cold = tmp_path / "silver-cold"
    for path in (silver_seed, silver_warm, silver_cold):
        _write_universe(path)
    cache_path = tmp_path / "parse.sqlite"
    with ParseCache(cache_path, namespace="n", fingerprint=parse_fingerprint()) as cache:
        materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_seed, workers=1, parse_cache=cache
        )
    connection = sqlite3.connect(str(cache_path))
    try:
        row = connection.execute("SELECT content_hash, value FROM entries LIMIT 1").fetchone()
        assert row is not None
        target, blob = row
        value = _json.loads(zlib.decompress(blob))
        value["cells"][0][1] = value["cells"][0][1][:5]
        connection.execute(
            "UPDATE entries SET value=? WHERE content_hash=?", (zlib.compress(_json.dumps(value).encode()), target)
        )
        connection.commit()
    finally:
        connection.close()
    with ParseCache(cache_path, namespace="n", fingerprint=parse_fingerprint()) as cache:
        warm = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_warm, workers=1, parse_cache=cache
        )
    cold = materialize_investor_flow_silver(
        catalog=catalog, universe_root=universe_root, silver_root=silver_cold, workers=1
    )
    assert warm.dataset_id == cold.dataset_id


def test_encode_decode_round_trip_preserves_types_and_order() -> None:
    from src.data.investor_flow_silver import _decode_page, _encode_page

    sessions = frozenset(SESSIONS)
    cells = [
        (date(2026, 3, 4), tuple(range(15)), 50000, 10000, 500),
        (date(2026, 3, 5), tuple([-x for x in range(15)]), 1, 2, 3),
    ]
    violations = [date(2026, 3, 6)]
    symbol, (out_cells, out_dateless, out_violations) = _decode_page(
        _encode_page("005930", (cells, 2, violations)), sessions, page_label="label"
    )
    assert symbol == "005930"
    assert out_cells == cells
    assert all(type(session) is date for session, _, _, _, _ in out_cells)
    assert all(type(amount) is int for _, groups, _, _, _ in out_cells for amount in groups)
    assert out_dateless == 2
    assert out_violations == violations


def test_decode_rejects_malformed_shapes() -> None:
    from src.data.investor_flow_silver import _decode_page

    sessions = frozenset(SESSIONS)
    good_cells = [[date(2026, 3, 4).toordinal(), list(range(15)), 1, 2, 3]]
    base = {"symbol": "005930", "dateless": 0, "violations": [], "cells": good_cells}
    malformed = [
        None, [], "x",
        dict(base, symbol=""), dict(base, symbol=1), dict(base, symbol=True),
        dict(base, dateless=-1), dict(base, dateless=True), dict(base, dateless="0"),
        dict(base, violations={}), dict(base, cells={}),
        dict(base, violations=["x"]), dict(base, violations=[True]),
        dict(base, violations=[999999999]),
        dict(base, cells=[["x", list(range(15)), 1, 2, 3]]),
        dict(base, cells=[[date(2026, 3, 4).toordinal(), list(range(5)), 1, 2, 3]]),
        dict(base, cells=[[date(2026, 3, 4).toordinal(), [1] * 14 + ["x"], 1, 2, 3]]),
        dict(base, cells=[[date(2026, 3, 4).toordinal(), list(range(15)), 1.5, 2, 3]]),
        dict(base, cells=[[999999999, list(range(15)), 1, 2, 3]]),
        dict(base, cells=[["short"]]),
    ]
    for value in malformed:
        with pytest.raises((TypeError, ValueError)):
            _decode_page(value, sessions, page_label="p")


def test_decode_calendar_guard_for_violation_and_cell() -> None:
    from src.data.investor_flow_silver import _decode_page

    sessions = frozenset({date(2026, 3, 4)})
    with pytest.raises(PITDataError, match="outside certified calendar"):
        _decode_page(
            {"symbol": "s", "dateless": 0, "violations": [date(2026, 3, 5).toordinal()], "cells": []},
            sessions, page_label="p",
        )
    with pytest.raises(PITDataError, match="outside certified calendar"):
        _decode_page(
            {"symbol": "s", "dateless": 0, "violations": [],
             "cells": [[date(2026, 3, 5).toordinal(), [0] * 15, 1, 2, 3]]},
            sessions, page_label="p",
        )


def test_parse_fingerprint_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.investor_flow_silver as flow_module

    monkeypatch.setattr(flow_module, "_PARSE_FINGERPRINT", None)
    first = flow_module.parse_fingerprint()
    assert first == flow_module._PARSE_FINGERPRINT
    assert flow_module.parse_fingerprint() == first


def test_malformed_hit_is_reparsed_before_calendar_validation(tmp_path: Path) -> None:
    from src.data.parse_cache import ParseCache

    bronze_root, universe_root, silver_root, catalog = _roots(tmp_path)
    digest = _seed_blob(
        catalog, bronze_root,
        _ls_payload("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"),
    )
    with ParseCache(tmp_path / "parse.sqlite", namespace="n", fingerprint="f") as cache:
        cache.put_many({digest: {
            "symbol": "005930", "dateless": 0,
            "violations": [date(2026, 3, 7).toordinal()],
            "cells": [[date(2026, 3, 4).toordinal(), [0], 1, 2, 3]],
        }})
        result = materialize_investor_flow_silver(
            catalog=catalog, universe_root=universe_root, silver_root=silver_root,
            workers=1, parse_cache=cache,
        )
        assert result.rows == 1
        assert cache.get_many([digest])[digest]["violations"] == []
