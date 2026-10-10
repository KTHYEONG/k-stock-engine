"""Silver daily-market materialization tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.core.time import KRX_TZ
from src.data.daily_market_silver import REVISION as DAILY_MARKET_SILVER_REVISION
from src.data.daily_market_silver import DailyMarketSilverPolicy, materialize_daily_market_silver
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from tests.fixtures import seed_receipts
from src.core.pit import PITDataError

DAY1 = date(2026, 3, 4)
DAY2 = date(2026, 3, 5)


def _universe(silver_root: Path, sessions: tuple[date, ...], *, name: str = "ordinary_universe_testfix") -> Path:
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


def _rec(session: date, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "ISU_CD": "KR7005930003",
        "ISU_SRT_CD": "005930",
        "MKT_NM": "KOSPI",
        "BAS_DD": session.strftime("%Y%m%d"),
        "TDD_OPNPRC": "10500",
        "TDD_HGPRC": "11200",
        "TDD_LWPRC": "10300",
        "TDD_CLSPRC": "11000",
        "CMPPREVDD_PRC": "1000",
        "FLUC_RT": "10.0",
        "ACC_TRDVOL": "1000",
        "ACC_TRDVAL": "11000000",
        "MKTCAP": "660000000000",
        "LIST_SHRS": "60000000",
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
                source="krx_daily_market",
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


def _setup(tmp_path: Path, sessions: tuple[date, ...]) -> tuple[ReceiptCatalog, Path, Path]:
    catalog_root = tmp_path / "catalog"
    payload_dir = tmp_path / "pages"
    silver_root = tmp_path / "silver"
    _universe(silver_root, sessions)
    return ReceiptCatalog(catalog_root), catalog_root, silver_root


def _payload_dir(tmp_path: Path) -> Path:
    return tmp_path / "pages"


def _frame(dataset_path: Path, session: date) -> pl.DataFrame:
    return pl.read_parquet(dataset_path / f"session={session.isoformat()}" / "part.parquet")


def test_materialize_preserves_base_price(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1)])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.rows == 1
    assert result.tradable_rows == 1
    frame = _frame(result.dataset_path, DAY1)
    assert frame["base_price"].to_list() == [10000]
    assert frame["price_state"].to_list() == ["tradable"]
    assert frame["invalid_reason"].to_list() == [None]
    assert frame["instrument_id"].to_list() == ["KRX:005930"]
    assert frame["ticker"].to_list() == ["005930"]


def test_materialize_keeps_corporate_action_base_and_counts_adjusted(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    pages = _payload_dir(tmp_path)
    _publish(catalog_root, pages, DAY1, [_rec(DAY1, TDD_CLSPRC="20000", CMPPREVDD_PRC="0", FLUC_RT="0.0",
        TDD_OPNPRC="19900", TDD_HGPRC="20100", TDD_LWPRC="19800", ACC_TRDVOL="500")])
    _publish(catalog_root, pages, DAY2, [
        _rec(DAY2, TDD_CLSPRC="10500", CMPPREVDD_PRC="500", FLUC_RT="5.0",
            TDD_OPNPRC="10400", TDD_HGPRC="10600", TDD_LWPRC="10300", ACC_TRDVOL="800"),
        _rec(DAY2, ISU_CD="KR7005830000", ISU_SRT_CD="000020", TDD_CLSPRC="5000", CMPPREVDD_PRC="0",
            FLUC_RT="0.0", TDD_OPNPRC="4950", TDD_HGPRC="5050", TDD_LWPRC="4900", ACC_TRDVOL="300"),
    ])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.base_adjusted_rows == 1
    frame = _frame(result.dataset_path, DAY2)
    assert frame.filter(pl.col("ticker") == "005930")["close"].to_list() == [10500]
    assert frame.filter(pl.col("ticker") == "005930")["base_price"].to_list() == [10000]


def test_materialize_marks_fluc_mismatch_invalid_but_kept(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(
        DAY1, TDD_CLSPRC="10300", CMPPREVDD_PRC="300", FLUC_RT="5.0",
        TDD_OPNPRC="10200", TDD_HGPRC="10400", TDD_LWPRC="10100", ACC_TRDVOL="500")])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.rows == 1
    assert result.invalid_rows == 1
    frame = _frame(result.dataset_path, DAY1)
    assert frame["price_state"].to_list() == ["invalid"]
    assert frame["invalid_reason"].to_list() == ["fluc_mismatch"]


def test_materialize_marks_zero_volume_state(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(
        DAY1, TDD_OPNPRC="0", TDD_HGPRC="0", TDD_LWPRC="0", TDD_CLSPRC="10000",
        CMPPREVDD_PRC="0", FLUC_RT="0.0", ACC_TRDVOL="0", ACC_TRDVAL="0")])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.zero_volume_rows == 1
    frame = _frame(result.dataset_path, DAY1)
    assert frame["price_state"].to_list() == ["zero_volume"]
    assert frame["invalid_reason"].to_list() == [None]


def test_materialize_marks_ohlc_inconsistency_invalid(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1, TDD_HGPRC="10900")])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.invalid_rows == 1
    frame = _frame(result.dataset_path, DAY1)
    assert frame["invalid_reason"].to_list() == ["ohlc_inconsistent"]


@pytest.mark.parametrize(
    "record",
    [
        _rec(DAY1, TDD_CLSPRC="0", CMPPREVDD_PRC="0", FLUC_RT="0.0",
             TDD_OPNPRC="0", TDD_HGPRC="0", TDD_LWPRC="0", ACC_TRDVOL="100"),
        _rec(DAY1, TDD_CLSPRC="100", CMPPREVDD_PRC="200", FLUC_RT="0.0"),
    ],
)
def test_materialize_marks_non_positive_close_or_base_invalid(tmp_path: Path, record: dict[str, object]) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [record])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.invalid_rows == 1
    frame = _frame(result.dataset_path, DAY1)
    assert frame["invalid_reason"].to_list() == ["non_positive_close_or_base"]


def test_materialize_excludes_konex_rows(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [
        _rec(DAY1),
        _rec(DAY1, ISU_CD="KR4000110007", ISU_SRT_CD="011000", MKT_NM="KONEX"),
    ])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.rows == 1
    frame = _frame(result.dataset_path, DAY1)
    assert frame["ticker"].to_list() == ["005930"]


def test_materialize_fails_without_session_page(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1)])
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_fails_on_non_success_entry(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1)], status=EvidenceStatus.EMPTY)
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_fails_on_catalog_date_conflict(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1)], as_of=DAY2)
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_fails_on_page_date_conflict(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY2)])
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_fails_on_duplicate_ticker(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1), _rec(DAY1)])
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_sets_availability_at_session_close(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    pages = _payload_dir(tmp_path)
    _publish(catalog_root, pages, DAY1, [_rec(DAY1)])
    _publish(catalog_root, pages, DAY2, [_rec(DAY2)])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.sessions == 2
    assert _frame(result.dataset_path, DAY1)["available_at"].to_list() == [datetime(2026, 3, 4, 18, 0, tzinfo=KRX_TZ)]
    assert _frame(result.dataset_path, DAY2)["available_at"].to_list() == [datetime(2026, 3, 5, 18, 0, tzinfo=KRX_TZ)]


def test_materialize_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1)])
    first = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    manifest_path = first.dataset_path / "manifest.json"
    before = manifest_path.read_bytes()
    second = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert first.dataset_id == second.dataset_id
    assert second.dataset_path == first.dataset_path
    assert manifest_path.read_bytes() == before
    assert first.dataset_id.startswith("daily_market_")


def test_materialize_rejects_differing_existing_dataset(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1)])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    (result.dataset_path / "manifest.json").write_text('{"dataset_id": "tampered"}', encoding="utf-8")
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_rejects_unreadable_existing_dataset(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1)])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    (result.dataset_path / "manifest.json").unlink()
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_rejects_hash_mismatch(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    pages = _payload_dir(tmp_path)
    _publish(catalog_root, pages, DAY1, [_rec(DAY1)])
    payload_path = pages / f"{DAY1.isoformat()}.json"
    payload_path.write_bytes(payload_path.read_bytes() + b" ")
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_rejects_unreadable_payload(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    pages = _payload_dir(tmp_path)
    _publish(catalog_root, pages, DAY1, [_rec(DAY1)])
    (pages / f"{DAY1.isoformat()}.json").unlink()
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


@pytest.mark.parametrize("raw_bytes", [b"not-json", b"[1, 2]", b'{"session": "2026-03-04"}'])
def test_materialize_rejects_malformed_payloads(tmp_path: Path, raw_bytes: bytes) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [], raw_bytes=raw_bytes)
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_rejects_non_mapping_record(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, ["nope"])
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_rejects_missing_identity(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    record = _rec(DAY1)
    del record["ISU_CD"]
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [record])
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_falls_back_to_isu_cd_ticker(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    record = _rec(DAY1)
    del record["ISU_SRT_CD"]
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [record])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert _frame(result.dataset_path, DAY1)["ticker"].to_list() == ["KR7005930003"]


@pytest.mark.parametrize("bad_date", ["2026-03-04", "20261301", "", None])
def test_materialize_rejects_invalid_bas_dd(tmp_path: Path, bad_date: object) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1, BAS_DD=bad_date)])
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


@pytest.mark.parametrize("bad_value", ["abc", "1.5", 10.5, True, None, ""])
def test_materialize_rejects_malformed_numerics(tmp_path: Path, bad_value: object) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1, TDD_CLSPRC=bad_value)])
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


@pytest.mark.parametrize("bad_fluc", ["abc", "inf", None, ""])
def test_materialize_rejects_malformed_fluc_rate(tmp_path: Path, bad_fluc: object) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1, FLUC_RT=bad_fluc)])
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def test_materialize_accepts_int_and_integral_float_numerics(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    record = _rec(DAY1, TDD_CLSPRC=11000, CMPPREVDD_PRC=1000, ACC_TRDVOL=1000.0, FLUC_RT=10)
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [record])
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.rows == 1
    assert result.tradable_rows == 1


def test_materialize_rejects_broken_universe(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty-silver"
    empty_root.mkdir(parents=True, exist_ok=True)
    catalog = ReceiptCatalog(tmp_path / "catalog")
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=catalog, universe_root=empty_root, silver_root=tmp_path / "out"
        )
    broken = tmp_path / "broken-silver"
    dataset = broken / "ordinary_universe_broken"
    dataset.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=catalog, universe_root=broken, silver_root=tmp_path / "out"
        )
    (dataset / "manifest.json").write_text('{"dataset_id": "other", "partitions": []}', encoding="utf-8")
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=catalog, universe_root=broken, silver_root=tmp_path / "out"
        )
    two_root = tmp_path / "two-silver"
    for name in ("ordinary_universe_a", "ordinary_universe_b"):
        day_dir = two_root / name
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "manifest.json").write_text(
            json.dumps({"dataset_id": name, "partitions": [{"session": "2026-03-04"}]}), encoding="utf-8"
        )
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=catalog, universe_root=two_root, silver_root=tmp_path / "out"
        )


@pytest.mark.parametrize(
    "manifest",
    [
        {"dataset_id": "ordinary_universe_bad", "partitions": ["nope"]},
        {"dataset_id": "ordinary_universe_bad", "partitions": [{"session": "xx"}]},
        {"dataset_id": "ordinary_universe_bad", "partitions": [{"session": "2026-03-05"}, {"session": "2026-03-04"}]},
    ],
)
def test_materialize_rejects_malformed_universe_partitions(tmp_path: Path, manifest: dict[str, object]) -> None:
    root = tmp_path / "upart"
    dataset = root / "ordinary_universe_bad"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PITDataError):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(tmp_path / "catalog"), universe_root=root, silver_root=tmp_path / "out"
        )


def test_materialize_uses_single_catalog_lookup(tmp_path: Path) -> None:
    sessions = (DAY1, DAY2, date(2026, 3, 6))
    _, catalog_root, silver_root = _setup(tmp_path, sessions)
    pages = _payload_dir(tmp_path)
    for session in sessions:
        _publish(catalog_root, pages, session, [_rec(session)])
    calls: list[frozenset[str]] = []
    inner = ReceiptCatalog(catalog_root)

    class CountingCatalog(ReceiptCatalog):
        def latest(self, *, source: str, natural_keys) -> object:  # type: ignore[override]
            calls.append(frozenset(natural_keys))
            return inner.latest(source=source, natural_keys=natural_keys)

    result = materialize_daily_market_silver(
        catalog=CountingCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert result.sessions == 3
    assert len(calls) == 1
    assert calls[0] == frozenset(session.isoformat() for session in sessions)


def test_materialize_missing_page_fails_before_writing(tmp_path: Path) -> None:
    sessions = (DAY1, DAY2, date(2026, 3, 6))
    _, catalog_root, silver_root = _setup(tmp_path, sessions)
    pages = _payload_dir(tmp_path)
    _publish(catalog_root, pages, DAY1, [_rec(DAY1)])
    _publish(catalog_root, pages, sessions[2], [_rec(sessions[2])])
    with pytest.raises(PITDataError, match="missing"):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )
    assert list(Path(silver_root).glob("daily_market_*")) == []
    assert [p for p in Path(silver_root).iterdir() if p.name.startswith(".daily-market-silver-")] == []


def test_materialize_identity_unchanged_by_batching(tmp_path: Path) -> None:
    sessions = (DAY1, DAY2)
    _, catalog_root, silver_root = _setup(tmp_path, sessions)
    pages = _payload_dir(tmp_path)
    digests = {session: _publish(catalog_root, pages, session, [_rec(session)]) for session in sessions}
    result = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    from src.data.datasets import DatasetIdentity, dataset_id_for, load_manifest

    manifest = load_manifest(result.dataset_path)
    identity = DatasetIdentity(
        kind=manifest.kind,
        layer=manifest.layer,
        policy_version=manifest.policy_version,
        inputs=manifest.inputs,
        params=manifest.params,
    )
    assert result.dataset_id == dataset_id_for(identity)
    assert manifest.kind == "daily_market"
    assert manifest.params["available_time"] == "18:00:00"
    assert manifest.params["fluc_tolerance_pct"] == 0.01
    assert {entry["session"] for entry in manifest.details["partitions"]} == {
        session.isoformat() for session in sessions
    }
    assert {digests[session] for session in sessions} == {
        entry["source_hash"] for entry in manifest.details["partitions"]
    }
    _ = (hashlib, DAILY_MARKET_SILVER_REVISION, DailyMarketSilverPolicy)


def test_materialize_parses_payload_once_per_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as json_module

    sessions = (DAY1, DAY2)
    _, catalog_root, silver_root = _setup(tmp_path, sessions)
    pages = _payload_dir(tmp_path)
    for session in sessions:
        _publish(catalog_root, pages, session, [_rec(session)])
    real_loads = json_module.loads
    calls = {"count": 0}

    def counting_loads(s: object, *args: object, **kwargs: object) -> object:
        if isinstance(s, (bytes, bytearray)):
            calls["count"] += 1
        return real_loads(s, *args, **kwargs)  # type: ignore[arg-type]

    import src.data.daily_market_silver as dms

    monkeypatch.setattr(dms.json, "loads", counting_loads)
    materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
    )
    assert calls["count"] == len(sessions)


def test_materialize_rejects_invalid_policy_values(tmp_path: Path) -> None:
    catalog, _catalog_root, silver_root = _setup(tmp_path, (DAY1,))

    with pytest.raises(PITDataError, match="available_time"):
        materialize_daily_market_silver(
            catalog=catalog,
            universe_root=silver_root,
            silver_root=silver_root,
            policy=DailyMarketSilverPolicy(available_time=object()),  # type: ignore[arg-type]
        )
    with pytest.raises(PITDataError, match="tolerance"):
        materialize_daily_market_silver(
            catalog=catalog,
            universe_root=silver_root,
            silver_root=silver_root,
            policy=DailyMarketSilverPolicy(fluc_tolerance_pct=float("nan")),
        )


def test_materialize_rejects_duplicate_exchange_ticker(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(
        catalog_root,
        _payload_dir(tmp_path),
        DAY1,
        [_rec(DAY1), _rec(DAY1, ISU_CD="KR7005930004")],
    )
    with pytest.raises(PITDataError, match="duplicate ticker"):
        materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root, silver_root=silver_root
        )


def _daily_fingerprint(path: Path) -> object:
    import json

    from src.data.datasets import load_manifest

    manifest = load_manifest(path)
    return (
        manifest.dataset_id,
        tuple((partition.path, partition.sha256) for partition in manifest.partitions),
        json.dumps(manifest.details, sort_keys=True, default=str),
    )


def _republish_session(catalog_root: Path, payload_dir: Path, session: date, records: object) -> None:
    raw = json.dumps({"session": session.isoformat(), "records": records}, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    payload_path = payload_dir / f"{session.isoformat()}.json"
    payload_path.write_bytes(raw)
    seed_receipts(
        ReceiptCatalog(catalog_root),
        [
            ReceiptIndexEntry(
                source="krx_daily_market",
                natural_key=session.isoformat(),
                as_of=session,
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash=digest,
                retrieved_at=datetime(2026, 3, 8, tzinfo=UTC),
                payload_path=payload_path,
            )
        ],
    )


def test_reuse_equals_full_rebuild(tmp_path: Path) -> None:
    from src.data.datasets import load_manifest

    DAY3 = date(2026, 3, 6)
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    for day in (DAY1, DAY2):
        _publish(catalog_root, _payload_dir(tmp_path), day, [_rec(day)])
    first = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    _universe(silver_root, (DAY1, DAY2, DAY3))
    universe_id = _universe(tmp_path / "silver-b", (DAY1, DAY2, DAY3)).name
    _publish(catalog_root, _payload_dir(tmp_path), DAY3, [_rec(DAY3)])
    catalog = ReceiptCatalog(catalog_root)
    sessions = (DAY1, DAY2, DAY3)
    reused = materialize_daily_market_silver(
        catalog=catalog, universe_root=tmp_path / "silver-b", silver_root=tmp_path / "silver-reuse",
        universe_dataset_id=universe_id, reuse_from=first.dataset_path,
    )
    full = materialize_daily_market_silver(
        catalog=catalog, universe_root=tmp_path / "silver-b", silver_root=tmp_path / "silver-full",
        universe_dataset_id=universe_id,
    )
    assert _daily_fingerprint(reused.dataset_path) == _daily_fingerprint(full.dataset_path)
    assert {entry["session"] for entry in load_manifest(reused.dataset_path).details["partitions"]} == {
        day.isoformat() for day in sessions
    }


def test_reused_sessions_read_no_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.daily_market_silver as daily_module

    DAY3 = date(2026, 3, 6)
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    payload_dir = _payload_dir(tmp_path)
    for day in (DAY1, DAY2):
        _publish(catalog_root, payload_dir, day, [_rec(day)])
    first = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    universe_id = _universe(tmp_path / "silver-b", (DAY1, DAY2, DAY3)).name
    _publish(catalog_root, payload_dir, DAY3, [_rec(DAY3)])
    catalog = ReceiptCatalog(catalog_root)
    full = materialize_daily_market_silver(
        catalog=catalog, universe_root=tmp_path / "silver-b", silver_root=tmp_path / "silver-full",
        universe_dataset_id=universe_id,
    )
    (payload_dir / f"{DAY1.isoformat()}.json").unlink()
    (payload_dir / f"{DAY2.isoformat()}.json").unlink()
    calls = 0
    real_read = daily_module.read_payload

    def _counted(path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_read(path)

    monkeypatch.setattr(daily_module, "read_payload", _counted)
    reused = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=tmp_path / "silver-b",
        silver_root=tmp_path / "silver-reuse", universe_dataset_id=universe_id, reuse_from=first.dataset_path,
    )
    assert calls == 1
    assert _daily_fingerprint(reused.dataset_path) == _daily_fingerprint(full.dataset_path)


def test_changed_source_hash_is_rebuilt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.daily_market_silver as daily_module

    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    payload_dir = _payload_dir(tmp_path)
    for day in (DAY1, DAY2):
        _publish(catalog_root, payload_dir, day, [_rec(day)])
    first = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    _republish_session(catalog_root, payload_dir, DAY2, [_rec(DAY2, TDD_CLSPRC="12000")])
    calls = 0
    real_read = daily_module.read_payload

    def _counted(path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_read(path)

    monkeypatch.setattr(daily_module, "read_payload", _counted)
    catalog = ReceiptCatalog(catalog_root)
    reused = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-reuse", reuse_from=first.dataset_path
    )
    assert calls == 1
    full = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-full"
    )
    assert _daily_fingerprint(reused.dataset_path) == _daily_fingerprint(full.dataset_path)


def test_corrupt_prior_partition_is_not_trusted(tmp_path: Path) -> None:
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    payload_dir = _payload_dir(tmp_path)
    for day in (DAY1, DAY2):
        _publish(catalog_root, payload_dir, day, [_rec(day)])
    first = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    with (first.dataset_path / f"session={DAY1.isoformat()}" / "part.parquet").open("ab") as handle:
        handle.write(b"\x00" * 64)
    catalog = ReceiptCatalog(catalog_root)
    reused = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-reuse", reuse_from=first.dataset_path
    )
    full = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-full"
    )
    assert _daily_fingerprint(reused.dataset_path) == _daily_fingerprint(full.dataset_path)


def test_universe_change_invalidates_the_daily_market_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.data.daily_market_silver as daily_module
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    payload_dir = _payload_dir(tmp_path)
    for day in (DAY1, DAY2):
        _publish(catalog_root, payload_dir, day, [_rec(day)])
    first = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    shifted = publish_dataset(
        layer_root=tmp_path / "silver-b",
        identity=DatasetIdentity(
            kind="ordinary_universe",
            layer=DatasetLayer.SILVER,
            policy_version="krx-ordinary-equity-v1",
            inputs={},
            params={"calendar": ",".join(day.isoformat() for day in (DAY1, DAY2))},
        ),
        partitions={
            f"session={DAY1.isoformat()}/part.parquet": pl.DataFrame(
                {"session": [DAY1], "instrument_id": ["KRX:005930"], "ticker": ["005930"], "eligible": [True]}
            ),
            f"session={DAY2.isoformat()}/part.parquet": pl.DataFrame(
                {"session": [DAY2], "instrument_id": ["KRX:000660"], "ticker": ["000660"], "eligible": [True]}
            ),
        },
    ).path
    calls = 0
    real_read = daily_module.read_payload

    def _counted(path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_read(path)

    monkeypatch.setattr(daily_module, "read_payload", _counted)
    catalog = ReceiptCatalog(catalog_root)
    reused = materialize_daily_market_silver(
        catalog=catalog, universe_root=tmp_path / "silver-b", silver_root=tmp_path / "silver-reuse",
        universe_dataset_id=shifted.name, reuse_from=first.dataset_path,
    )
    assert calls == 1
    full = materialize_daily_market_silver(
        catalog=catalog, universe_root=tmp_path / "silver-b", silver_root=tmp_path / "silver-full",
        universe_dataset_id=shifted.name,
    )
    assert _daily_fingerprint(reused.dataset_path) == _daily_fingerprint(full.dataset_path)


def test_cross_session_counters_survive_reuse(tmp_path: Path) -> None:
    from src.data.datasets import load_manifest

    DAY3 = date(2026, 3, 6)
    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    payload_dir = _payload_dir(tmp_path)
    _publish(catalog_root, payload_dir, DAY1, [_rec(DAY1)])
    _publish(catalog_root, payload_dir, DAY2, [_rec(DAY2)])
    first = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    universe_id = _universe(tmp_path / "silver-b", (DAY1, DAY2, DAY3)).name
    _publish(catalog_root, payload_dir, DAY3, [_rec(DAY3, TDD_CLSPRC="12000", CMPPREVDD_PRC="500")])
    catalog = ReceiptCatalog(catalog_root)
    reused = materialize_daily_market_silver(
        catalog=catalog, universe_root=tmp_path / "silver-b", silver_root=tmp_path / "silver-reuse",
        universe_dataset_id=universe_id, reuse_from=first.dataset_path,
    )
    full = materialize_daily_market_silver(
        catalog=catalog, universe_root=tmp_path / "silver-b", silver_root=tmp_path / "silver-full",
        universe_dataset_id=universe_id,
    )
    assert load_manifest(reused.dataset_path).details["base_adjusted_rows"] == 2
    assert _daily_fingerprint(reused.dataset_path) == _daily_fingerprint(full.dataset_path)


def test_packed_bronze_builds_identical_daily_market(tmp_path: Path) -> None:
    from src.data.payload_store import pack_loose

    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    payload_dir = _payload_dir(tmp_path)
    for day in (DAY1, DAY2):
        _publish(catalog_root, payload_dir, day, [_rec(day)])
    first = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    pack_loose(tmp_path, kinds=["pages"], offsite_paths=None, remove=True)
    second = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root,
        silver_root=tmp_path / "silver-b",
    )
    assert _daily_fingerprint(first.dataset_path) == _daily_fingerprint(second.dataset_path)


def test_stale_or_foreign_prior_dataset_falls_back_to_full_rebuild(tmp_path: Path) -> None:
    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    payload_dir = _payload_dir(tmp_path)
    for day in (DAY1, DAY2):
        _publish(catalog_root, payload_dir, day, [_rec(day)])
    foreign = publish_dataset(
        layer_root=tmp_path / "silver-foreign",
        identity=DatasetIdentity(
            kind="ordinary_universe", layer=DatasetLayer.SILVER, policy_version="krx-ordinary-equity-v1",
            inputs={}, params={},
        ),
        partitions={"part-00000.parquet": pl.DataFrame({"a": [1]})},
    ).path
    expected = _daily_fingerprint(
        materialize_daily_market_silver(
            catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-full"
        ).dataset_path
    )
    for reuse_from in (tmp_path / "missing-dir", foreign):
        rebuilt = materialize_daily_market_silver(
            catalog=ReceiptCatalog(catalog_root), universe_root=silver_root,
            silver_root=tmp_path / "silver-fallback", reuse_from=reuse_from,
        )
        assert _daily_fingerprint(rebuilt.dataset_path) == expected
    shifted_policy = DailyMarketSilverPolicy(fluc_tolerance_pct=5.0)
    prior = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-prior",
        policy=shifted_policy,
    )
    rebuilt = materialize_daily_market_silver(
        catalog=ReceiptCatalog(catalog_root), universe_root=silver_root,
        silver_root=tmp_path / "silver-fallback", reuse_from=prior.dataset_path,
    )
    assert _daily_fingerprint(rebuilt.dataset_path) == expected


def test_empty_session_page_builds_empty_partition(tmp_path: Path) -> None:
    from src.data.datasets import load_manifest

    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    payload_dir = _payload_dir(tmp_path)
    _publish(catalog_root, payload_dir, DAY1, [_rec(DAY1)])
    _publish(catalog_root, payload_dir, DAY2, [])
    result = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    assert result.rows == 1
    manifest = load_manifest(result.dataset_path)
    assert manifest.details["sessions"] == 2
    assert _frame(result.dataset_path, DAY2).height == 0


def test_universe_partition_without_session_marker_is_ignored(tmp_path: Path) -> None:
    import polars as pl

    from src.data.daily_market_silver import _universe_partition_shas
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    _publish(catalog_root, _payload_dir(tmp_path), DAY1, [_rec(DAY1)])
    universe_id = _universe(silver_root, (DAY1,)).name
    extra = publish_dataset(
        layer_root=tmp_path / "silver-extra",
        identity=DatasetIdentity(
            kind="ordinary_universe", layer=DatasetLayer.SILVER, policy_version="krx-ordinary-equity-v1",
            inputs={}, params={},
        ),
        partitions={
            f"session={DAY1.isoformat()}/part.parquet": pl.DataFrame(
                {"session": [DAY1], "instrument_id": ["KRX:005930"], "ticker": ["005930"], "eligible": [True]}
            ),
            "meta/info.parquet": pl.DataFrame({"note": ["extra"]}),
        },
    ).path
    shas = _universe_partition_shas(tmp_path / "silver-extra", extra.name)
    assert set(shas) == {DAY1.isoformat()}
    result = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    assert result.sessions == 1


def test_malformed_or_partial_prior_dataset_falls_back_to_full_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import shutil

    import polars as pl

    catalog, catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    payload_dir = _payload_dir(tmp_path)
    for day in (DAY1, DAY2):
        _publish(catalog_root, payload_dir, day, [_rec(day)])
    first = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-a"
    )
    expected = _daily_fingerprint(first.dataset_path)
    catalog = ReceiptCatalog(catalog_root)

    for details in (
        {},
        {"partitions": [{"session": DAY1.isoformat()}]},
        {"partitions": "nope"},
        {"partitions": [{"session": DAY1.isoformat(), "source_hash": "x", "universe_sha": 5}]},
    ):
        variant = tmp_path / "silver-variant"
        if variant.exists():
            shutil.rmtree(variant)
        shutil.copytree(first.dataset_path, variant / first.dataset_id)
        manifest_path = variant / first.dataset_id / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {**json.loads(manifest_path.read_text(encoding="utf-8")), "details": details},
                indent=2, sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        rebuilt = materialize_daily_market_silver(
            catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-fallback",
            reuse_from=variant / first.dataset_id,
        )
        assert _daily_fingerprint(rebuilt.dataset_path) == expected

    partial = tmp_path / "silver-partial"
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(first.dataset_path, partial / first.dataset_id)
    (partial / first.dataset_id / f"session={DAY2.isoformat()}" / "part.parquet").unlink()
    partial_manifest = partial / first.dataset_id / "manifest.json"
    raw_manifest = json.loads(partial_manifest.read_text(encoding="utf-8"))
    raw_manifest["partitions"] = [
        item for item in raw_manifest["partitions"] if DAY2.isoformat() not in str(item.get("path"))
    ]
    raw_manifest["rows"] = sum(int(item["rows"]) for item in raw_manifest["partitions"])
    partial_manifest.write_text(json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    calls = 0
    real_read = pl.read_parquet

    def _flaky(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        if first.dataset_id in str(path) and str(path).endswith(f"session={DAY1.isoformat()}/part.parquet"):
            calls += 1
            raise OSError("transient read race")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", _flaky)
    rebuilt = materialize_daily_market_silver(
        catalog=catalog, universe_root=silver_root, silver_root=tmp_path / "silver-reuse",
        reuse_from=partial / first.dataset_id,
    )
    monkeypatch.undo()
    assert calls == 1
    assert _daily_fingerprint(rebuilt.dataset_path) == expected
