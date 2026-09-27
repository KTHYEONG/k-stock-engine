"""Catalog-driven Silver investor-flow tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.core.pit import EvidenceKind, PITDataError
from src.data.investor_flow_silver import (
    POLICY_VERSION,
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
    assert POLICY_VERSION == "ls-t1702-net-shares-v2"


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
