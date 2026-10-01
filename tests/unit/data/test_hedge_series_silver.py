"""Silver hedge-series materialization tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time
from pathlib import Path

import polars as pl
import pytest

from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.data.hedge_series_silver import (
    POLICY_VERSION,
    HedgeSeriesConfig,
    load_hedge_series_config,
    materialize_hedge_series_silver,
)
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from tests.fixtures import seed_receipts

DAY1 = date(2026, 3, 4)
DAY2 = date(2026, 3, 5)
DAY3 = date(2026, 3, 6)


def _config(**overrides: object) -> HedgeSeriesConfig:
    values: dict[str, object] = {
        "collection_start": DAY1,
        "index_name": "코스닥 150",
        "inverse_ticker": "251340",
        "available_time": time(18, 0),
    }
    values.update(overrides)
    return HedgeSeriesConfig.model_validate(values)


def _universe(silver_root: Path, sessions: tuple[date, ...]) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    partitions = {
        f"session={day.isoformat()}/part.parquet": pl.DataFrame(
            {"session": [day], "instrument_id": ["KRX:005930"], "ticker": ["005930"], "eligible": [True]}
        )
        for day in sessions
    }
    return publish_dataset(
        layer_root=Path(silver_root),
        identity=DatasetIdentity(
            kind="ordinary_universe",
            layer=DatasetLayer.SILVER,
            policy_version="krx-ordinary-equity-v1",
            inputs={},
            params={"calendar": ",".join(day.isoformat() for day in sessions)},
        ),
        partitions=partitions,
    ).path


def _index_record(session: date, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "_endpoint": "index",
        "IDX_CLSS": "KOSDAQ",
        "IDX_NM": "코스닥 150",
        "BAS_DD": session.strftime("%Y%m%d"),
        "CLSPRC_IDX": "1000.5",
    }
    record.update(overrides)
    return record


def _etf_record(session: date, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "_endpoint": "etf",
        "ISU_CD": "251340",
        "ISU_SRT_CD": "251340",
        "BAS_DD": session.strftime("%Y%m%d"),
        "TDD_CLSPRC": "5000",
    }
    record.update(overrides)
    return record


def _publish(
    catalog_root: Path,
    payload_dir: Path,
    session: date,
    records: object,
    *,
    status: EvidenceStatus = EvidenceStatus.SUCCESS,
    as_of: date | None = None,
    raw_bytes: bytes | None = None,
) -> str:
    payload_dir.mkdir(parents=True, exist_ok=True)
    if raw_bytes is None:
        raw = json.dumps({"session": session.isoformat(), "records": records}, sort_keys=True).encode("utf-8")
    else:
        raw = raw_bytes
    digest = hashlib.sha256(raw).hexdigest()
    payload_path = payload_dir / f"{session.isoformat()}.json"
    payload_path.write_bytes(raw)
    seed_receipts(
        ReceiptCatalog(catalog_root),
        [
            ReceiptIndexEntry(
                source="krx_hedge_series",
                natural_key=session.isoformat(),
                as_of=session if as_of is None else as_of,
                fiscal_period=None,
                status=status,
                content_hash=digest,
                retrieved_at=datetime(2026, 3, 7, tzinfo=UTC),
                payload_path=payload_path,
            )
        ],
    )
    return digest


def _setup(tmp_path: Path, sessions: tuple[date, ...]) -> tuple[Path, Path]:
    catalog_root = tmp_path / "catalog"
    silver_root = tmp_path / "silver"
    _universe(silver_root, sessions)
    return catalog_root, silver_root


def _pages(tmp_path: Path) -> Path:
    return tmp_path / "pages"


def _frame(dataset_path: Path, session: date) -> pl.DataFrame:
    return pl.read_parquet(dataset_path / f"session={session.isoformat()}" / "part.parquet")


def test_happy_path_null_before_listing(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    digest1 = _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    digest3 = _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    result = materialize_hedge_series_silver(
        catalog=ReceiptCatalog(catalog_root),
        universe_root=silver_root,
        silver_root=silver_root,
        config=_config(),
    )
    assert result.sessions == 3
    assert result.inverse_listing_session == DAY3
    assert _frame(result.dataset_path, DAY1)["inverse_close"].to_list() == [None]
    assert _frame(result.dataset_path, DAY2)["inverse_close"].to_list() == [None]
    assert _frame(result.dataset_path, DAY3)["inverse_close"].to_list() == [5000]
    assert _frame(result.dataset_path, DAY3)["index_level"].to_list() == [1000.5]
    assert _frame(result.dataset_path, DAY1)["source_hash"].to_list() == [digest1]
    assert _frame(result.dataset_path, DAY3)["source_hash"].to_list() == [digest3]
    assert _frame(result.dataset_path, DAY1)["available_at"].to_list() == [
        datetime(2026, 3, 4, 18, 0, tzinfo=KRX_TZ)
    ]
    assert result.dataset_id.startswith("hedge_series_")


def test_missing_session_receipt_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    with pytest.raises(PITDataError):
        materialize_hedge_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_corrupted_payload_hash_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    payload_path = pages / f"{DAY2.isoformat()}.json"
    payload_path.write_bytes(payload_path.read_bytes() + b" ")
    with pytest.raises(PITDataError):
        materialize_hedge_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_etf_missing_after_listing_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2), _etf_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3)])
    with pytest.raises(PITDataError):
        materialize_hedge_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_non_positive_index_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1, CLSPRC_IDX="0")])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    with pytest.raises(PITDataError):
        materialize_hedge_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_page_date_mismatch_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    with pytest.raises(PITDataError):
        materialize_hedge_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_materialize_is_idempotent(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    first = materialize_hedge_series_silver(
        catalog=ReceiptCatalog(catalog_root),
        universe_root=silver_root,
        silver_root=silver_root,
        config=_config(),
    )
    second = materialize_hedge_series_silver(
        catalog=ReceiptCatalog(catalog_root),
        universe_root=silver_root,
        silver_root=silver_root,
        config=_config(),
    )
    assert first.dataset_id == second.dataset_id
    assert second.dataset_path == first.dataset_path
    assert len([p for p in Path(silver_root).iterdir() if p.name.startswith("hedge_series_")]) == 1
    _ = POLICY_VERSION


def test_config_validation_rejects_bad_documents(tmp_path: Path) -> None:
    good = tmp_path / "good.toml"
    good.write_text(
        'collection_start = 2017-01-02\nindex_name = "코스닥 150"\n'
        'inverse_ticker = "251340"\navailable_time = "18:00:00"\n',
        encoding="utf-8",
    )
    assert load_hedge_series_config(good).collection_start == date(2017, 1, 2)
    assert load_hedge_series_config(Path("config/data/hedge_series.toml")).inverse_ticker == "251340"
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(good.read_text(encoding="utf-8") + 'extra_key = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hedge-series config"):
        load_hedge_series_config(unknown)
    missing = tmp_path / "missing.toml"
    missing.write_text(
        'collection_start = 2017-01-02\nindex_name = "코스닥 150"\n' 'available_time = "18:00:00"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hedge-series config"):
        load_hedge_series_config(missing)
    bad_time = tmp_path / "bad_time.toml"
    bad_time.write_text(
        'collection_start = 2017-01-02\nindex_name = "코스닥 150"\n'
        'inverse_ticker = "251340"\navailable_time = "not-a-time"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hedge-series config"):
        load_hedge_series_config(bad_time)
    bad_ticker = tmp_path / "bad_ticker.toml"
    bad_ticker.write_text(
        'collection_start = 2017-01-02\nindex_name = "코스닥 150"\n'
        'inverse_ticker = "123"\navailable_time = "18:00:00"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hedge-series config"):
        load_hedge_series_config(bad_ticker)
    with pytest.raises(ValueError, match="hedge-series config"):
        load_hedge_series_config(tmp_path / "absent.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("collection_start = [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hedge-series config"):
        load_hedge_series_config(broken)
    with pytest.raises(ValueError, match="non-empty string"):
        HedgeSeriesConfig.model_validate(
            {
                "collection_start": DAY1,
                "index_name": "  ",
                "inverse_ticker": "251340",
                "available_time": time(18, 0),
            }
        )


def _materialize(catalog_root: Path, silver_root: Path, **overrides: object) -> object:
    return materialize_hedge_series_silver(
        catalog=ReceiptCatalog(catalog_root),
        universe_root=silver_root,
        silver_root=silver_root,
        config=_config(**overrides),
    )


def test_universe_autoresolve_requires_exactly_one_dataset(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    empty_root = tmp_path / "empty-silver"
    empty_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PITDataError):
        materialize_hedge_series_silver(
            catalog=catalog, universe_root=empty_root, silver_root=tmp_path / "out", config=_config()
        )


def test_unreadable_payload_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    (pages / f"{DAY1.isoformat()}.json").unlink()
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []


@pytest.mark.parametrize("raw_bytes", [b"not-json", b"[1, 2]", b'{"session": "2026-03-04"}'])
def test_malformed_payloads_fail_closed(tmp_path: Path, raw_bytes: bytes) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)], raw_bytes=raw_bytes)
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []


@pytest.mark.parametrize(
    "close",
    [None, "", True, 10.5, "abc", "1.5", "0"],
)
def test_malformed_inverse_close_fails_closed(tmp_path: Path, close: object) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3, TDD_CLSPRC=close)])
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []


@pytest.mark.parametrize("close", [5000, 5000.0, "5,000"])
def test_numeric_inverse_close_forms_accepted(tmp_path: Path, close: object) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3, TDD_CLSPRC=close)])
    result = _materialize(catalog_root, silver_root)
    assert _frame(result.dataset_path, DAY3)["inverse_close"].to_list() == [5000]


@pytest.mark.parametrize("bas_dd", ["2026-03-04", "20261301", "", None])
def test_invalid_bas_dd_fails_closed(tmp_path: Path, bas_dd: object) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1, BAS_DD=bas_dd)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_no_sessions_after_collection_start_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    with pytest.raises(PITDataError):
        materialize_hedge_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(collection_start=date(2027, 1, 4)),
        )


def test_catalog_date_conflict_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)], as_of=DAY2)
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []


@pytest.mark.parametrize("case", ["missing", "duplicated"])
def test_missing_and_duplicated_index_rows_fail_closed(tmp_path: Path, case: str) -> None:
    if case == "missing":
        records: object = [_etf_record(DAY1)]
    else:
        records = [_index_record(DAY1), _index_record(DAY1)]
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, records)
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_duplicated_etf_row_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(
        catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3), _etf_record(DAY3)]
    )
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_unparsable_index_level_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1, CLSPRC_IDX="abc")])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3)])
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_etf_page_date_conflict_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(
        catalog_root, pages, DAY3, [_index_record(DAY3), _etf_record(DAY3, BAS_DD="20260305")]
    )
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []


def test_never_listed_etf_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_index_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_index_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_index_record(DAY3)])
    with pytest.raises(PITDataError):
        _materialize(catalog_root, silver_root)
    assert list(Path(silver_root).glob("hedge_series_*")) == []
