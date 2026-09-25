"""Invariant guards for the provider-tagged KIS supplement of the LS gap."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

S1 = date(2024, 1, 2)
S2 = date(2024, 1, 3)
S3 = date(2024, 1, 4)
S4 = date(2024, 1, 5)


def _write_dataset(directory: Path, frame: pl.DataFrame) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    is_gold = "gold" in directory.parts
    kind = "market_panel" if is_gold else (
        "investor_flow_kis_supplement" if "kis" in directory.name else "investor_flow_ls"
    )
    return publish_dataset(
        layer_root=directory.parent,
        identity=DatasetIdentity(
            kind=kind,
            layer=DatasetLayer.GOLD if is_gold else DatasetLayer.SILVER,
            policy_version="test-v1",
            inputs={},
            params={},
        ),
        partitions={"year=2024/part.parquet": frame},
    ).path


def _write_panel(root: Path, cells: list[tuple[date, str]]) -> Path:
    frame = pl.DataFrame(
        {
            "session": [session for session, _ in cells],
            "instrument_id": [f"KRX:{ticker}" for _, ticker in cells],
            "ticker": [ticker for _, ticker in cells],
            "eligible": [True] * len(cells),
            "price_state": ["tradable"] * len(cells),
        },
        schema={
            "session": pl.Date,
            "instrument_id": pl.String,
            "ticker": pl.String,
            "eligible": pl.Boolean,
            "price_state": pl.String,
        },
    )
    return _write_dataset(root / "gold" / "market_panel_test", frame)


def _write_ls(root: Path, cells: list[tuple[date, str]]) -> Path:
    frame = pl.DataFrame(
        {"session": [session for session, _ in cells], "ticker": [ticker for _, ticker in cells]},
        schema={"session": pl.Date, "ticker": pl.String},
    )
    return _write_dataset(root / "silver" / "investor_flow_test", frame)


def _kis_row(session: date, prsn: Any, frgn: Any, orgn: Any, etc: Any, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stck_bsop_date": session.strftime("%Y%m%d"),
        "prsn_ntby_qty": prsn,
        "frgn_ntby_qty": frgn,
        "orgn_ntby_qty": orgn,
        "etc_ntby_qty": etc,
    }
    row.update(extra)
    return row


def _write_page(bronze_root: Path, payload: dict[str, Any]) -> Path:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    page_dir = bronze_root / "investor_flow" / hashlib.sha256(raw).hexdigest()
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "payload.json").write_bytes(raw)
    return page_dir


def _write_kis_page(
    bronze_root: Path, symbol: str, anchor: date, rows: list[dict[str, Any]]
) -> Path:
    return _write_page(
        bronze_root,
        {
            "provider": "KIS",
            "endpoint": "investor-trade-by-stock-daily",
            "symbol": symbol,
            "anchor": anchor.isoformat(),
            "query": {"symbol": symbol, "anchor": anchor.isoformat()},
            "rows": rows,
            "records": [],
        },
    )


def _write_raw_page(bronze_root: Path, raw: bytes) -> Path:
    page_dir = bronze_root / "investor_flow" / hashlib.sha256(raw).hexdigest()
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "payload.json").write_bytes(raw)
    return page_dir


def _materialize(root: Path, bronze_root: Path | None = None):
    from src.data.investor_flow_kis_supplement import materialize_investor_flow_kis_supplement

    panel = next(path for path in (root / "gold").iterdir() if path.is_dir() and path.name.startswith("market_panel_"))
    ls_flow = next(path for path in (root / "silver").iterdir() if path.is_dir() and path.name.startswith("investor_flow_ls_"))
    return materialize_investor_flow_kis_supplement(
        bronze_root=bronze_root or root / "bronze",
        market_panel_path=panel,
        ls_flow_silver_path=ls_flow,
        silver_root=root / "silver",
    )


def _output_cells(dataset_path: Path) -> pl.DataFrame:
    return pl.scan_parquet(sorted((dataset_path).glob("year=*/part.parquet"))).collect()


def test_materialize_writes_only_target_gap_cells(tmp_path: Path) -> None:
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(
        bronze,
        "000001",
        S2,
        [
            _kis_row(S1, "100", "-60", "-30", "-10"),
            _kis_row(S2, "100", "-60", "-30", "-10"),
        ],
    )

    result = _materialize(tmp_path, bronze)

    assert result.filled_cells == 1
    assert result.still_missing_cells == 1
    frame = _output_cells(result.dataset_path)
    assert frame.select("session", "ticker").sort("session").to_dicts() == [
        {"session": S2, "ticker": "000001"}
    ]
    assert frame["provider"].to_list() == ["KIS"]
    assert frame["instrument_id"].to_list() == ["KRX:000001"]


def test_materialize_ignores_ls_bronze_pages(tmp_path: Path) -> None:
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000001")])
    _write_ls(tmp_path, [(S1, "000001"), (S3, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])
    _write_page(
        bronze,
        {
            "provider": "LS",
            "endpoint": "frgr-itt",
            "symbol": "000001",
            "query": {"symbol": "000001", "start": "2024-01-03", "end": "2024-01-03"},
            "rows": [{"date": "20240103", "tjj0008": "5"}],
            "records": [],
        },
    )
    _write_page(bronze, {"provider": "KIS", "symbol": "000001", "rows": []})

    result = _materialize(tmp_path, bronze)

    assert result.filled_cells == 1
    assert _output_cells(result.dataset_path).height == 1


def test_materialize_isolates_identity_violation_row(tmp_path: Path) -> None:
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(
        bronze,
        "000001",
        S3,
        [
            _kis_row(S2, "100", "-60", "-30", "-10"),
            _kis_row(S3, "1", "1", "1", "1"),
        ],
    )

    result = _materialize(tmp_path, bronze)

    assert result.identity_violation_cells == 1
    assert result.filled_cells == 1
    frame = _output_cells(result.dataset_path)
    assert frame["session"].to_list() == [S2]


def test_materialize_reads_institution_from_orgn_directly(tmp_path: Path) -> None:
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000001")])
    _write_ls(tmp_path, [(S1, "000001"), (S3, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(
        bronze,
        "000001",
        S2,
        [
            _kis_row(
                S2,
                "100",
                "-60",
                "-30",
                "-10",
                scrt_ntby_qty="0",
                ivtr_ntby_qty="0",
                pe_fund_ntby_vol="0",
                bank_ntby_qty="0",
                insu_ntby_qty="0",
                mrbn_ntby_qty="0",
                fund_ntby_qty="0",
            )
        ],
    )

    result = _materialize(tmp_path, bronze)

    frame = _output_cells(result.dataset_path)
    assert frame["institution_net_shares"].to_list() == [-30]
    assert frame["individual_net_shares"].to_list() == [100]
    assert frame["foreign_net_shares"].to_list() == [-60]
    assert frame["other_net_shares"].to_list() == [-10]


def test_materialize_excludes_cell_on_last_panel_session(tmp_path: Path) -> None:
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(
        bronze,
        "000001",
        S3,
        [
            _kis_row(S2, "100", "-60", "-30", "-10"),
            _kis_row(S3, "100", "-60", "-30", "-10"),
        ],
    )

    result = _materialize(tmp_path, bronze)

    assert result.filled_cells == 1
    assert result.still_missing_cells == 1
    assert _output_cells(result.dataset_path)["session"].to_list() == [S2]


def test_materialize_reports_coverage_counts(tmp_path: Path) -> None:
    _write_panel(
        tmp_path,
        [
            (S1, "000001"),
            (S2, "000001"),
            (S3, "000001"),
            (S1, "000002"),
            (S2, "000002"),
            (S3, "000002"),
            (S4, "000002"),
        ],
    )
    _write_ls(tmp_path, [(S1, "000001"), (S1, "000002")])
    bronze = tmp_path / "bronze"
    _write_kis_page(
        bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")]
    )
    _write_kis_page(
        bronze,
        "000002",
        S3,
        [
            _kis_row(S2, "50", "-30", "-10", "-10"),
            _kis_row(S3, "50", "-30", "-10", "-10"),
        ],
    )

    result = _materialize(tmp_path, bronze)

    assert result.target_cells == 5
    assert result.filled_cells == 3
    assert result.still_missing_cells == 2
    assert result.rows == 3


def test_materialize_excludes_conflicting_duplicate_cell(tmp_path: Path) -> None:
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000001")])
    _write_ls(tmp_path, [(S1, "000001"), (S3, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "50", "-30", "-10", "-10")])

    result = _materialize(tmp_path, bronze)

    assert result.filled_cells == 0
    assert result.still_missing_cells == 1
    assert result.identity_violation_cells == 0


def test_materialize_collapses_identical_duplicate_pages(tmp_path: Path) -> None:
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000001")])
    _write_ls(tmp_path, [(S1, "000001"), (S3, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])
    _write_kis_page(bronze, "000001", S3, [_kis_row(S2, "100", "-60", "-30", "-10")])

    result = _materialize(tmp_path, bronze)

    assert result.filled_cells == 1
    assert _output_cells(result.dataset_path).height == 1


def test_materialize_is_deterministic_across_runs(tmp_path: Path) -> None:
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000001")])
    _write_ls(tmp_path, [(S1, "000001"), (S3, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])

    first = _materialize(tmp_path, bronze)
    manifest_before = (first.dataset_path / "manifest.json").read_bytes()
    second = _materialize(tmp_path, bronze)

    assert second.dataset_id == first.dataset_id
    assert second.dataset_path == first.dataset_path
    assert (second.dataset_path / "manifest.json").read_bytes() == manifest_before


def test_materialize_manifest_pins_source_datasets(tmp_path: Path) -> None:
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000001")])
    _write_ls(tmp_path, [(S1, "000001"), (S3, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])

    result = _materialize(tmp_path, bronze)
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["inputs"]["ls"] == result.ls_dataset_id
    assert manifest["inputs"]["market_panel"].startswith("market_panel_")
    assert result.ls_dataset_id.startswith("investor_flow_ls_")


def test_materialize_rejects_tampered_bronze_payload(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    page_dir = _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])
    (page_dir / "payload.json").write_bytes(b"tampered")

    with pytest.raises(PITDataError, match="hash mismatch"):
        _materialize(tmp_path, bronze)


def test_materialize_rejects_malformed_bronze_json(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    _write_raw_page(tmp_path / "bronze", b"not json{")

    with pytest.raises(PITDataError, match="invalid investor-flow Bronze JSON"):
        _materialize(tmp_path, tmp_path / "bronze")


def test_materialize_rejects_non_dict_bronze_root(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    _write_raw_page(tmp_path / "bronze", b"[1, 2]")

    with pytest.raises(PITDataError, match="invalid investor-flow Bronze root"):
        _materialize(tmp_path, tmp_path / "bronze")


def test_materialize_rejects_page_without_symbol(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_page(bronze, {"provider": "KIS", "rows": [_kis_row(S2, "1", "0", "-1", "0")]})

    with pytest.raises(PITDataError, match="lacks query symbol"):
        _materialize(tmp_path, bronze)


def test_materialize_rejects_non_object_row(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, ["not-a-dict"])  # type: ignore[list-item]

    with pytest.raises(PITDataError, match="must be an object"):
        _materialize(tmp_path, bronze)


def test_materialize_rejects_malformed_session(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    row = _kis_row(S2, "1", "0", "-1", "0")
    row["stck_bsop_date"] = "2024-3-3"
    _write_kis_page(bronze, "000001", S2, [row])

    with pytest.raises(PITDataError, match="invalid date"):
        _materialize(tmp_path, bronze)


def test_materialize_rejects_impossible_session(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    row = _kis_row(S2, "1", "0", "-1", "0")
    row["stck_bsop_date"] = "20241301"
    _write_kis_page(bronze, "000001", S2, [row])

    with pytest.raises(PITDataError, match="invalid date"):
        _materialize(tmp_path, bronze)


def test_materialize_rejects_missing_quantity(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    row = _kis_row(S2, "1", "0", "-1", "0")
    del row["prsn_ntby_qty"]
    _write_kis_page(bronze, "000001", S2, [row])

    with pytest.raises(PITDataError, match="missing prsn_ntby_qty"):
        _materialize(tmp_path, bronze)


def test_materialize_rejects_non_integral_quantity(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "1.5", "0", "-1", "-0.5")])

    with pytest.raises(PITDataError, match="non-integral"):
        _materialize(tmp_path, bronze)


def test_materialize_rejects_invalid_quantity(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "abc", "0", "-1", "0")])

    with pytest.raises(PITDataError, match="invalid prsn_ntby_qty"):
        _materialize(tmp_path, bronze)


def test_materialize_skips_session_outside_calendar(tmp_path: Path) -> None:
    # KIS's anchor-walk returns a fixed 30-row window that can spill past the
    # certified range's edge (real, observed on the live backfill); this must
    # be dropped silently, not treated as a data defect.
    # S3 extends the certified calendar past S2 (via a different ticker) so
    # S2 still has a following session for the availability-lag check.
    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001"), (S3, "000002")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(
        bronze,
        "000001",
        S2,
        [
            _kis_row(date(2024, 1, 10), "1", "0", "-1", "0"),
            _kis_row(S2, "1", "0", "-1", "0"),
        ],
    )

    result = _materialize(tmp_path, bronze)

    assert result.filled_cells == 1
    assert result.target_cells == 2


def test_materialize_rejects_differing_existing_dataset(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])

    result = _materialize(tmp_path, bronze)
    manifest_path = result.dataset_path / "manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(PITDataError, match="differs"):
        _materialize(tmp_path, bronze)


def test_materialize_rejects_unreadable_existing_manifest(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError

    _write_panel(tmp_path, [(S1, "000001"), (S2, "000001")])
    _write_ls(tmp_path, [(S1, "000001")])
    bronze = tmp_path / "bronze"
    _write_kis_page(bronze, "000001", S2, [_kis_row(S2, "100", "-60", "-30", "-10")])

    result = _materialize(tmp_path, bronze)
    (result.dataset_path / "manifest.json").unlink()

    with pytest.raises(PITDataError, match="unreadable"):
        _materialize(tmp_path, bronze)


def test_kis_supplement_rejects_invalid_policy_and_empty_panel_calendar(tmp_path: Path) -> None:
    import pytest

    from src.data.investor_flow_kis_supplement import (
        InvestorFlowKisSupplementPolicy,
        _load_panel_calendar,
        materialize_investor_flow_kis_supplement,
    )
    from src.data.schemas import PITDataError

    panel = _write_dataset(tmp_path / "gold" / "market_panel_empty", pl.DataFrame({"value": [1]}))
    with pytest.raises(PITDataError, match="no session partitions"):
        _load_panel_calendar(panel)

    with pytest.raises(PITDataError, match="availability lag"):
        materialize_investor_flow_kis_supplement(
            bronze_root=tmp_path / "bronze",
            market_panel_path=panel,
            ls_flow_silver_path=tmp_path / "silver" / "missing",
            silver_root=tmp_path / "silver",
            policy=InvestorFlowKisSupplementPolicy(available_session_lag=0),
        )
    with pytest.raises(PITDataError, match="available_time"):
        materialize_investor_flow_kis_supplement(
            bronze_root=tmp_path / "bronze",
            market_panel_path=panel,
            ls_flow_silver_path=tmp_path / "silver" / "missing",
            silver_root=tmp_path / "silver",
            policy=InvestorFlowKisSupplementPolicy(available_time=object()),  # type: ignore[arg-type]
        )
