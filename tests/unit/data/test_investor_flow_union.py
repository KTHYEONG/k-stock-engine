"""Invariant guards for the certified LS plus KIS investor-flow union."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

S1 = date(2024, 1, 2)
S2 = date(2024, 1, 3)

EXPECTED_COLUMNS = (
    "session",
    "instrument_id",
    "ticker",
    "provider",
    "individual_net_shares",
    "foreign_net_shares",
    "institution_net_shares",
    "other_net_shares",
    "available_at",
    "source_hash",
    "policy_version",
)


def _available_at(session: date) -> datetime:
    from src.core.time import KRX_TZ

    return datetime(session.year, session.month, session.day, 8, 0, tzinfo=KRX_TZ)


def _common_row(
    provider: str, session: date, ticker: str, values: tuple[int, int, int, int]
) -> dict[str, Any]:
    return {
        "session": session,
        "instrument_id": f"KRX:{ticker}",
        "ticker": ticker,
        "provider": provider,
        "individual_net_shares": values[0],
        "foreign_net_shares": values[1],
        "institution_net_shares": values[2],
        "other_net_shares": values[3],
        "available_at": _available_at(session),
        "source_hash": f"{provider.lower()}-hash-{ticker}-{session.isoformat()}",
        "policy_version": f"{provider.lower()}-policy-v1",
    }


def _ls_row(session: date, ticker: str, values: tuple[int, int, int, int]) -> dict[str, Any]:
    return _common_row("LS", session, ticker, values) | {
        "tjj0000_net_shares": values[0] + values[1],
        "tjj0011_net_shares": values[2] + values[3],
        "ls_close": 50000,
        "ls_volume": 1000,
        "ls_value_mkrw": 50,
    }


def _write_dataset(directory: Path, rows: list[dict[str, Any]]) -> Path:
    frame = pl.DataFrame(rows)
    directory.mkdir(parents=True, exist_ok=True)
    rel = Path("year=2024") / "part.parquet"
    out_path = directory / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": directory.name,
                "policy_version": "test-v1",
                "partitions": [
                    {
                        "path": str(rel),
                        "row_count": frame.height,
                        "parquet_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                        "year": 2024,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return directory


def _write_inputs(root: Path, ls_rows: list[dict[str, Any]], kis_rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    silver = root / "silver"
    ls_path = _write_dataset(silver / "investor_flow_ls_test", ls_rows)
    kis_path = _write_dataset(silver / "investor_flow_kis_supplement_test", kis_rows)
    return ls_path, kis_path, silver


def _materialize(ls_path: Path, kis_path: Path, silver: Path):
    from src.data.investor_flow_union import materialize_investor_flow_union

    return materialize_investor_flow_union(
        ls_flow_silver_path=ls_path,
        kis_supplement_silver_path=kis_path,
        silver_root=silver,
    )


def _output_frame(dataset_path: Path) -> pl.DataFrame:
    return pl.scan_parquet(sorted(dataset_path.glob("year=*/part.parquet"))).collect()


def test_materialize_unions_disjoint_datasets(tmp_path: Path) -> None:
    """Disjoint LS and KIS inputs concatenate to one dataset with all rows."""
    ls_path, kis_path, silver = _write_inputs(
        tmp_path,
        [_ls_row(S1, "000001", (100, -60, -30, -10))],
        [_common_row("KIS", S2, "000002", (50, -30, -10, -10))],
    )

    result = _materialize(ls_path, kis_path, silver)

    assert result.rows == 2
    frame = _output_frame(result.dataset_path).sort("session", "ticker")
    assert frame.select("session", "ticker").to_dicts() == [
        {"session": S1, "ticker": "000001"},
        {"session": S2, "ticker": "000002"},
    ]
    assert result.ls_dataset_id == "investor_flow_ls_test"
    assert result.kis_supplement_dataset_id == "investor_flow_kis_supplement_test"


def test_materialize_rejects_conflicting_key(tmp_path: Path) -> None:
    """A (session, ticker) key in both inputs fails closed without writing."""
    import pytest

    from src.data.schemas import PITDataError

    ls_path, kis_path, silver = _write_inputs(
        tmp_path,
        [_ls_row(S1, "000001", (100, -60, -30, -10))],
        [_common_row("KIS", S1, "000001", (50, -30, -10, -10))],
    )
    before = {entry.name for entry in silver.iterdir()}

    with pytest.raises(PITDataError, match="000001"):
        _materialize(ls_path, kis_path, silver)

    assert {entry.name for entry in silver.iterdir()} == before


def test_materialize_drops_ls_only_columns(tmp_path: Path) -> None:
    """LS breakdown columns stay in LS; common values carry over exactly."""
    ls_path, kis_path, silver = _write_inputs(
        tmp_path,
        [_ls_row(S1, "000001", (100, -60, -30, -10))],
        [_common_row("KIS", S2, "000002", (50, -30, -10, -10))],
    )

    result = _materialize(ls_path, kis_path, silver)

    frame = _output_frame(result.dataset_path)
    assert tuple(frame.columns) == EXPECTED_COLUMNS
    ls_out = frame.filter(pl.col("provider") == "LS").to_dicts()
    assert len(ls_out) == 1
    assert [ls_out[0][column] for column in (
        "individual_net_shares",
        "foreign_net_shares",
        "institution_net_shares",
        "other_net_shares",
    )] == [100, -60, -30, -10]


def test_materialize_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Identical inputs reproduce the same id and reuse it as a no-op."""
    ls_path, kis_path, silver = _write_inputs(
        tmp_path,
        [_ls_row(S1, "000001", (100, -60, -30, -10))],
        [_common_row("KIS", S2, "000002", (50, -30, -10, -10))],
    )

    first = _materialize(ls_path, kis_path, silver)
    manifest_before = (first.dataset_path / "manifest.json").read_bytes()
    second = _materialize(ls_path, kis_path, silver)

    assert second.dataset_id == first.dataset_id
    assert second.dataset_path == first.dataset_path
    assert (second.dataset_path / "manifest.json").read_bytes() == manifest_before


def test_materialize_rejects_differing_existing_dataset(tmp_path: Path) -> None:
    """A stale directory at the computed id is never silently overwritten."""
    import pytest

    from src.data.schemas import PITDataError

    ls_path, kis_path, silver = _write_inputs(
        tmp_path,
        [_ls_row(S1, "000001", (100, -60, -30, -10))],
        [_common_row("KIS", S2, "000002", (50, -30, -10, -10))],
    )
    result = _materialize(ls_path, kis_path, silver)
    manifest_path = result.dataset_path / "manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    tampered = manifest_path.read_bytes()

    with pytest.raises(PITDataError, match="differs"):
        _materialize(ls_path, kis_path, silver)

    assert manifest_path.read_bytes() == tampered


def test_materialize_rejects_corrupt_input_partition(tmp_path: Path) -> None:
    """A tampered LS partition hash fails closed before any output is written."""
    import pytest

    from src.data.schemas import PITDataError

    ls_path, kis_path, silver = _write_inputs(
        tmp_path,
        [_ls_row(S1, "000001", (100, -60, -30, -10))],
        [_common_row("KIS", S2, "000002", (50, -30, -10, -10))],
    )
    part = ls_path / "year=2024" / "part.parquet"
    with part.open("ab") as handle:
        handle.write(b"tampered")
    before = {entry.name for entry in silver.iterdir()}

    with pytest.raises(PITDataError, match="hash mismatch"):
        _materialize(ls_path, kis_path, silver)

    assert {entry.name for entry in silver.iterdir()} == before


def test_materialize_preserves_provider_attribution(tmp_path: Path) -> None:
    """Each output row keeps its source provider tag without rewriting."""
    ls_path, kis_path, silver = _write_inputs(
        tmp_path,
        [_ls_row(S1, "000001", (100, -60, -30, -10))],
        [_common_row("KIS", S2, "000002", (50, -30, -10, -10))],
    )

    result = _materialize(ls_path, kis_path, silver)

    frame = _output_frame(result.dataset_path).sort("session", "ticker")
    assert frame["provider"].to_list() == ["LS", "KIS"]


def test_materialize_rejects_unreadable_existing_manifest(tmp_path: Path) -> None:
    """A directory at the computed id without a manifest fails closed."""
    import pytest

    from src.data.schemas import PITDataError

    ls_path, kis_path, silver = _write_inputs(
        tmp_path,
        [_ls_row(S1, "000001", (100, -60, -30, -10))],
        [_common_row("KIS", S2, "000002", (50, -30, -10, -10))],
    )

    result = _materialize(ls_path, kis_path, silver)
    (result.dataset_path / "manifest.json").unlink()

    with pytest.raises(PITDataError, match="unreadable"):
        _materialize(ls_path, kis_path, silver)
