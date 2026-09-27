"""Catalog-driven KIS supplement tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.core.pit import EvidenceKind, PITDataError
from src.data.receipt_catalog import BlobEntry, ReceiptCatalog

S1 = date(2024, 1, 2)
S2 = date(2024, 1, 3)
S3 = date(2024, 1, 4)
S4 = date(2024, 1, 5)


def _write_dataset(directory: Path, frame: pl.DataFrame, *, kind: str) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    layer = DatasetLayer.SILVER
    return publish_dataset(
        layer_root=directory.parent,
        identity=DatasetIdentity(kind=kind, layer=layer, policy_version="test-v1", inputs={}, params={}),
        partitions={"year=2024/part.parquet": frame},
    ).path


def _write_universe_daily(root: Path, cells: list[tuple[date, str]]) -> tuple[str, str]:
    uni = pl.DataFrame(
        {"session": [s for s, _ in cells], "ticker": [t for _, t in cells], "eligible": [True] * len(cells)},
        schema={"session": pl.Date, "ticker": pl.String, "eligible": pl.Boolean},
    )
    daily = pl.DataFrame(
        {"session": [s for s, _ in cells], "ticker": [t for _, t in cells], "price_state": ["tradable"] * len(cells)},
        schema={"session": pl.Date, "ticker": pl.String, "price_state": pl.String},
    )
    uni_path = _write_dataset(root / "silver" / "uni", uni, kind="ordinary_universe")
    daily_path = _write_dataset(root / "silver" / "daily", daily, kind="daily_market")
    return uni_path.name, daily_path.name


def _write_ls(root: Path, cells: list[tuple[date, str]]) -> Path:
    frame = pl.DataFrame(
        {"session": [s for s, _ in cells], "ticker": [t for _, t in cells]},
        schema={"session": pl.Date, "ticker": pl.String},
    )
    return _write_dataset(root / "silver" / "flow", frame, kind="investor_flow_ls")


def _targets(universe_id: str, daily_id: str, cells: list[tuple[date, str]]) -> object:
    from src.data.flow_targets import FlowTargets

    frame = pl.DataFrame(
        {"session": [s for s, _ in cells], "ticker": [t for _, t in cells]},
        schema={"session": pl.Date, "ticker": pl.String},
    )
    return FlowTargets(universe_dataset_id=universe_id, daily_market_dataset_id=daily_id, cells=frame)


def _kis_row(session: date, prsn: Any, frgn: Any, orgn: Any, etc: Any) -> dict[str, Any]:
    return {
        "stck_bsop_date": session.strftime("%Y%m%d"),
        "prsn_ntby_qty": prsn, "frgn_ntby_qty": frgn, "orgn_ntby_qty": orgn, "etc_ntby_qty": etc,
    }


def _seed_kis(catalog: ReceiptCatalog, bronze_root: Path, symbol: str, anchor: date, rows: list[dict[str, Any]], *, usable: bool = True) -> str:
    payload = {
        "provider": "KIS",
        "query": {"symbol": symbol, "anchor": anchor.isoformat()},
        "rows": rows,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze_root / "investor_flow" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    catalog.publish(
        [],
        blobs=[BlobEntry(content_hash=digest, kind=EvidenceKind.INVESTOR_FLOW, source="kis_investor_flow",
                          usable=usable, unusable_reason=(None if usable else "kis_mapped_only"),
                          retrieved_at=datetime(2024, 1, 5, tzinfo=UTC),
                          payload_path=target / "payload.json")],
    )
    return digest


def _materialize(root: Path, catalog: ReceiptCatalog, targets: object, ls_path: Path):
    from src.data.investor_flow_kis_supplement import materialize_investor_flow_kis_supplement

    return materialize_investor_flow_kis_supplement(
        catalog=catalog, targets=targets, ls_flow_silver_path=ls_path, silver_root=root / "silver",
    )


def _cells(path: Path) -> pl.DataFrame:
    return pl.scan_parquet(sorted(path.glob("year=*/part.parquet"))).collect()


def test_supplement_has_no_gold_input(tmp_path: Path) -> None:
    """Supplement has no Gold input: identity inputs are exactly the Silver set."""
    uni_id, daily_id = _write_universe_daily(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000002")])
    ls_path = _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze / "catalog")
    _seed_kis(catalog, bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])
    result = _materialize(tmp_path, catalog, _targets(uni_id, daily_id, [(S1, "000001"), (S2, "000001"), (S3, "000001")]), ls_path)
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["inputs"]) == {"universe", "daily_market", "ls", "bronze_kis"}
    assert "market_panel" not in manifest["inputs"]
    assert result.filled_cells == 1


def test_both_kis_layouts_parse_identically(tmp_path: Path) -> None:
    uni_id, daily_id = _write_universe_daily(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000002")])
    ls_path = _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze / "catalog")
    rows = [_kis_row(S2, "100", "-60", "-30", "-10")]
    _seed_kis(catalog, bronze, "000001", S2, rows)
    result = _materialize(tmp_path, catalog, _targets(uni_id, daily_id, [(S1, "000001"), (S2, "000001")]), ls_path)
    assert result.filled_cells == 1
    frame = _cells(result.dataset_path)
    assert frame["provider"].to_list() == ["KIS"]
    # RAW_ROWS_V1 envelope with the same rows parses the same way.
    bronze2 = tmp_path / "bronze2"
    catalog2 = ReceiptCatalog(bronze2 / "catalog")
    envelope = {"envelope": "raw-rows-v1", "provider": "KIS", "endpoint": "investor-trade-by-stock-daily",
                "query": {"symbol": "000001", "anchor": S2.isoformat()}, "rows": rows}
    raw = json.dumps(envelope, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze2 / "investor_flow" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    catalog2.publish([], blobs=[BlobEntry(content_hash=digest, kind=EvidenceKind.INVESTOR_FLOW,
                                           source="kis_investor_flow", usable=True, unusable_reason=None,
                                           retrieved_at=datetime(2024, 1, 5, tzinfo=UTC),
                                           payload_path=target / "payload.json")])
    from src.data.investor_flow_kis_supplement import _parse_kis_blob

    sessions = frozenset([S1, S2, S3])
    first = _parse_kis_blob(next(catalog.blobs(source="kis_investor_flow")), sessions)
    second = _parse_kis_blob(next(catalog2.blobs(source="kis_investor_flow")), sessions)
    assert first[1:] == second[1:]


def test_kis_page_without_quantity_rows_not_used(tmp_path: Path) -> None:
    """KIS page without quantity rows not used: kis_mapped_only blob leaves the cell missing."""
    uni_id, daily_id = _write_universe_daily(tmp_path, [(S1, "000001"), (S2, "000001")])
    ls_path = _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze / "catalog")
    payload = {"provider": "KIS", "query": {"symbol": "000001", "anchor": S2.isoformat()}, "rows": []}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze / "investor_flow" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    catalog.publish([], blobs=[BlobEntry(content_hash=digest, kind=EvidenceKind.INVESTOR_FLOW,
                                          source="kis_investor_flow", usable=False,
                                          unusable_reason="kis_mapped_only",
                                          retrieved_at=datetime(2024, 1, 5, tzinfo=UTC),
                                          payload_path=target / "payload.json")])
    result = _materialize(tmp_path, catalog, _targets(uni_id, daily_id, [(S1, "000001"), (S2, "000001")]), ls_path)
    assert result.filled_cells == 0
    assert result.still_missing_cells == 1


def test_uncatalogued_kis_files_ignored(tmp_path: Path) -> None:
    uni_id, daily_id = _write_universe_daily(tmp_path, [(S1, "000001"), (S2, "000001")])
    ls_path = _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze / "catalog")
    _seed_kis(catalog, bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])
    extra = {"provider": "KIS", "query": {"symbol": "000001", "anchor": S1.isoformat()},
             "rows": [_kis_row(S1, "1", "0", "-1", "0")]}
    raw = json.dumps(extra, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze / "investor_flow" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    result = _materialize(tmp_path, catalog, _targets(uni_id, daily_id, [(S1, "000001"), (S2, "000001")]), ls_path)
    assert result.filled_cells == 0 or _cells(result.dataset_path)["session"].to_list() == [S2]


def test_isolates_violation_and_conflicts(tmp_path: Path) -> None:
    uni_id, daily_id = _write_universe_daily(tmp_path, [(S1, "000001"), (S2, "000001")])
    ls_path = _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze / "catalog")
    _seed_kis(catalog, bronze, "000001", S2, [_kis_row(S2, "1", "1", "1", "1")])
    result = _materialize(tmp_path, catalog, _targets(uni_id, daily_id, [(S1, "000001"), (S2, "000001")]), ls_path)
    assert result.identity_violation_cells == 1
    assert result.filled_cells == 0


def test_rejects_invalid_policy(tmp_path: Path) -> None:
    from src.data.investor_flow_kis_supplement import (
        InvestorFlowKisSupplementPolicy,
        materialize_investor_flow_kis_supplement,
    )

    uni_id, daily_id = _write_universe_daily(tmp_path, [(S1, "000001")])
    ls_path = _write_ls(tmp_path, [(S1, "000001")])
    catalog = ReceiptCatalog(tmp_path / "bronze" / "catalog")
    with pytest.raises(PITDataError, match="availability lag"):
        materialize_investor_flow_kis_supplement(
            catalog=catalog, targets=_targets(uni_id, daily_id, [(S1, "000001")]),
            ls_flow_silver_path=ls_path, silver_root=tmp_path / "silver",
            policy=InvestorFlowKisSupplementPolicy(available_session_lag=0),
        )


def test_fail_closed_on_unreadable_blob(tmp_path: Path) -> None:
    uni_id, daily_id = _write_universe_daily(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000002")])
    ls_path = _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze / "catalog")
    digest = _seed_kis(catalog, bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])
    path = bronze / "investor_flow" / digest / "payload.json"
    path.chmod(0o000)
    try:
        from src.data.investor_flow_kis_supplement import materialize_investor_flow_kis_supplement

        with pytest.raises(PITDataError):
            materialize_investor_flow_kis_supplement(
                catalog=catalog,
                targets=_targets(uni_id, daily_id, [(S1, "000001"), (S2, "000001")]),
                ls_flow_silver_path=ls_path, silver_root=tmp_path / "silver",
            )
    finally:
        path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PITDataError):
        materialize_investor_flow_kis_supplement(
            catalog=catalog,
            targets=_targets(uni_id, daily_id, [(S1, "000001"), (S2, "000001")]),
            ls_flow_silver_path=ls_path, silver_root=tmp_path / "silver",
        )


def test_fail_closed_on_malformed_blobs(tmp_path: Path) -> None:
    uni_id, daily_id = _write_universe_daily(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000002")])
    ls_path = _write_ls(tmp_path, [(S1, "000001")])
    cases = [
        b"not json",
        b"[1, 2]",
        json.dumps({"provider": "LS", "query": {"symbol": "000001", "anchor": S2.isoformat()},
                    "rows": [{"stck_bsop_date": "20240103"}]}).encode(),
        json.dumps({"provider": "KIS", "query": {"symbol": "", "anchor": S2.isoformat()},
                    "rows": [{"stck_bsop_date": "20240103", "prsn_ntby_qty": "1",
                              "frgn_ntby_qty": "0", "orgn_ntby_qty": "-1", "etc_ntby_qty": "0"}]}).encode(),
        json.dumps({"provider": "KIS", "query": {"symbol": "000001", "anchor": S2.isoformat()},
                    "rows": []}).encode(),
        json.dumps({"provider": "KIS", "query": {"symbol": "000001", "anchor": S2.isoformat()},
                    "rows": ["nope"]}).encode(),
    ]
    for index, raw in enumerate(cases):
        bronze = tmp_path / f"bronze-bad-{index}"
        catalog = ReceiptCatalog(bronze / "catalog")
        digest = hashlib.sha256(raw).hexdigest()
        target = bronze / "investor_flow" / digest
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.json").write_bytes(raw)
        catalog.publish(
            [],
            blobs=[BlobEntry(content_hash=digest, kind=EvidenceKind.INVESTOR_FLOW,
                              source="kis_investor_flow", usable=True, unusable_reason=None,
                              retrieved_at=datetime(2024, 1, 5, tzinfo=UTC),
                              payload_path=target / "payload.json")],
        )
        from src.data.investor_flow_kis_supplement import materialize_investor_flow_kis_supplement

        with pytest.raises(PITDataError):
            materialize_investor_flow_kis_supplement(
                catalog=catalog,
                targets=_targets(uni_id, daily_id, [(S1, "000001"), (S2, "000001")]),
                ls_flow_silver_path=ls_path, silver_root=tmp_path / "silver",
            )


def test_rejects_empty_daily_calendar(tmp_path: Path) -> None:
    from src.data.investor_flow_kis_supplement import _load_daily_calendar

    empty = _write_dataset(tmp_path / "gold" / "empty", pl.DataFrame({"value": [1]}), kind="market_panel")
    with pytest.raises(PITDataError, match="no session"):
        _load_daily_calendar(empty)
