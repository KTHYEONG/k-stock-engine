"""Silver investor-flow materialization tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.core.time import KRX_TZ
from src.data.investor_flow_silver import (
    InvestorFlowSilverPolicy,
    materialize_investor_flow_silver,
)
from src.data.schemas import PITDataError

SESSIONS = (date(2026, 3, 4), date(2026, 3, 5), date(2026, 3, 6))


def _write_universe(universe_root: Path) -> Path:
    dataset = Path(universe_root) / "ordinary_universe_testfix"
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": dataset.name,
        "policy_version": "krx-ordinary-equity-v1",
        "partitions": [{"session": day.isoformat()} for day in SESSIONS],
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


def _row(session: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": session,
        "tjj0000": "-100",
        "tjj0001": "-50",
        "tjj0002": "-30",
        "tjj0003": "-20",
        "tjj0004": "-10",
        "tjj0005": "-10",
        "tjj0006": "-8",
        "tjj0007": "100",
        "tjj0008": "927",
        "tjj0009": "-800",
        "tjj0010": "-28",
        "tjj0011": "29",
        "tjj0016": "-828",
        "tjj0017": "129",
        "tjj0018": "-228",
        "close": "50000",
        "volume": "10000",
        "value": "500",
    }
    row.update(overrides)
    return row


def _raw_page(symbol: str, rows: list[dict[str, object]], *, start: str, end: str, anchor: str = "2026-03-06") -> dict[str, object]:
    return {
        "provider": "LS",
        "endpoint": "frgr-itt",
        "symbol": symbol,
        "anchor": anchor,
        "query": {"symbol": symbol, "start": start, "end": end},
        "rows": rows,
        "records": [],
    }


def _write_page(bronze_root: Path, payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = Path(bronze_root) / "investor_flow" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    return digest


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    bronze_root = tmp_path / "bronze"
    universe_root = tmp_path / "silver"
    silver_root = tmp_path / "silver"
    _write_universe(universe_root)
    return bronze_root, universe_root, silver_root


def _frame(result_path: Path) -> pl.DataFrame:
    return pl.read_parquet(result_path / "year=2026" / "part.parquet")


def test_materialize_uses_raw_rows_only_and_ignores_records_only(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    _write_page(bronze_root, {
        "provider": "LS",
        "symbol": "005930",
        "records": [{
            "ticker": "005930",
            "session": "2026-03-04",
            "individual_net_shares": 927000000,
            "foreign_net_shares": -828000000,
            "institution_net_shares": -228000000,
            "other_net_shares": 129000000,
        }],
    })
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 1
    assert result.ignored_records_only_pages == 1
    assert result.raw_pages == 1
    frame = _frame(result.dataset_path)
    assert frame.filter(pl.col("session") == date(2026, 3, 4))["individual_net_shares"].to_list() == [927]
    assert frame["foreign_net_shares"].to_list() == [-828]
    assert frame["institution_net_shares"].to_list() == [-228]
    assert frame["other_net_shares"].to_list() == [129]
    assert frame["instrument_id"].to_list() == ["KRX:005930"]
    assert frame["provider"].to_list() == ["LS"]


@pytest.mark.parametrize(
    "bad_row",
    [
        _row("20260304", tjj0000="-99"),
        _row("20260304", tjj0009="-799"),
        _row("20260304", tjj0007="101"),
        _row("20260304", tjj0008="928", tjj0000="-99", tjj0001="-51"),
    ],
)
def test_materialize_isolates_aggregate_identity_violations(tmp_path: Path, bad_row: dict[str, object]) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [bad_row, _row("20260305")], start="2026-03-04", end="2026-03-06"))
    # 다른 페이지의 정상 값이 있어도 위반 셀은 복구하지 않는다.
    _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-04"))
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.identity_violation_cells == 1
    assert _frame(result.dataset_path)["session"].to_list() == [date(2026, 3, 5)]
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["identity_violation_keys"] == ["2026-03-04:005930"]


def test_materialize_preserves_foreign_total_and_subgroups(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    row = _row(
        "20260304",
        tjj0000="-2", tjj0001="-1", tjj0002="-1", tjj0003="-1", tjj0004="-1",
        tjj0005="-1", tjj0006="-1", tjj0007="3", tjj0008="9", tjj0009="-5",
        tjj0010="-1", tjj0011="2", tjj0016="-6", tjj0017="5", tjj0018="-8",
    )
    _write_page(bronze_root, _raw_page("005930", [row], start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    frame = _frame(result.dataset_path)
    assert frame["foreign_net_shares"].to_list() == [-6]
    assert frame["tjj0009_net_shares"].to_list() == [-5]
    assert frame["tjj0010_net_shares"].to_list() == [-1]
    assert frame["tjj0011_net_shares"].to_list() == [2]


def test_materialize_assigns_next_session_availability(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260304", volume="100")], start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    frame = _frame(result.dataset_path)
    assert frame["available_at"].to_list() == [datetime(2026, 3, 5, 8, 0, tzinfo=KRX_TZ)]
    assert result.retail_exceeds_volume_rows == 1


def test_materialize_excludes_calendar_tail(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260306")], start="2026-03-06", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 0
    assert result.unavailable_tail_cells == 1
    assert result.tickers == 0


def test_materialize_collapses_identical_duplicates(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    first = _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06", anchor="2026-03-04"))
    second = _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06", anchor="2026-03-05"))
    assert first != second
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=2
    )
    assert result.rows == 1
    assert result.conflict_cells == 0
    frame = _frame(result.dataset_path)
    assert frame["source_hash"].to_list() == [min(first, second)]


def test_materialize_excludes_conflicting_duplicates(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06", anchor="2026-03-04"))
    _write_page(
        bronze_root,
        _raw_page(
            "005930",
            [_row("20260304", tjj0008="928", tjj0000="-101", tjj0018="-229")],
            start="2026-03-04",
            end="2026-03-06",
            anchor="2026-03-05",
        ),
    )
    _write_page(bronze_root, _raw_page("005930", [_row("20260305")], start="2026-03-05", end="2026-03-05", anchor="2026-03-05"))
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.conflict_cells == 1
    frame = _frame(result.dataset_path)
    assert frame["session"].to_list() == [date(2026, 3, 5)]
    assert result.rows == 1


def test_materialize_counts_negative_pages_without_zero_fill(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, {
        "provider": "ls",
        "endpoint": "frgr-itt",
        "symbol": "005930",
        "sessions": ["2026-03-04", "2026-03-06"],
        "status": "missing_sessions",
        "missing_sessions": ["2026-03-04", "2026-03-06"],
    })
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.negative_cells == 2
    assert result.rows == 0


def test_materialize_counts_provider_errors_net_of_raw_coverage(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-04"))
    _write_page(bronze_root, {
        "provider": "ls",
        "endpoint": "frgr-itt",
        "symbol": "005930",
        "sessions": ["2026-03-04", "2026-03-05"],
        "status": "provider_error",
    })
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.negative_cells == 1
    assert result.rows == 1


def test_materialize_rejects_hash_mismatch(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    digest = _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    payload_path = bronze_root / "investor_flow" / digest / "payload.json"
    payload_path.write_bytes(payload_path.read_bytes() + b" ")
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    first = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    second = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert first.dataset_id == second.dataset_id
    assert second.dataset_path == first.dataset_path
    manifest = json.loads((first.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == first.dataset_id
    assert manifest["policy_version"] == "ls-t1702-net-shares-v1"
    assert manifest["universe_dataset_id"] == "ordinary_universe_testfix"
    assert manifest["rows"] == 1
    assert manifest["partitions"][0]["row_count"] == 1
    assert len(manifest["partitions"][0]["parquet_sha256"]) == 64


def test_materialize_rejects_differing_existing_dataset(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    manifest_path = result.dataset_path / "manifest.json"
    manifest_path.write_text('{"dataset_id": "tampered"}', encoding="utf-8")
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_rejects_unreadable_existing_dataset(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    (result.dataset_path / "manifest.json").unlink()
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_ignores_rows_outside_query_range(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(
        bronze_root,
        _raw_page("005930", [_row("20260304"), _row("20260305"), _row("20260306")], start="2026-03-04", end="2026-03-05"),
    )
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 2


def test_materialize_rejects_unknown_session_outside_calendar(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260307")], start="2026-03-04", end="2026-03-07"))
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


@pytest.mark.parametrize(
    "bad_value",
    ["1.5", 927.5, True, "abc", None, ""],
)
def test_materialize_rejects_non_integral_and_malformed_quantities(tmp_path: Path, bad_value: object) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(
        bronze_root,
        _raw_page("005930", [_row("20260304", tjj0008=bad_value)], start="2026-03-04", end="2026-03-06"),
    )
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_accepts_int_and_integral_float_quantities(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    row = _row("20260304", tjj0008=927, tjj0016=-828, close=50000.0, volume=10000, value=500)
    _write_page(bronze_root, _raw_page("005930", [row], start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.rows == 1


def test_materialize_rejects_malformed_rows_and_pages(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", ["not-a-mapping"], start="2026-03-04", end="2026-03-06"))  # type: ignore[list-item]
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_rejects_invalid_row_dates(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("2026-03-04")], start="2026-03-04", end="2026-03-06"))
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_rejects_impossible_row_dates(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20261301")], start="2026-01-01", end="2026-12-31"))
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_rejects_raw_page_without_query_symbol(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, {"rows": [_row("20260304")], "records": []})
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_rejects_raw_page_with_invalid_query_range(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="not-a-date", end="2026-03-06"))
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_rejects_invalid_page_bytes(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    raw = b"not json"
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze_root / "investor_flow" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_rejects_non_mapping_page(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    raw = b"[1, 2]"
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze_root / "investor_flow" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )


def test_materialize_rejects_bad_workers(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=0
        )


def test_materialize_rejects_ambiguous_or_broken_universe(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    bronze_root.mkdir(parents=True, exist_ok=True)
    empty_root = tmp_path / "empty-silver"
    empty_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=empty_root, silver_root=tmp_path / "out", workers=1
        )
    broken = tmp_path / "broken-silver"
    dataset = broken / "ordinary_universe_broken"
    dataset.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=broken, silver_root=tmp_path / "out", workers=1
        )
    (dataset / "manifest.json").write_text('{"dataset_id": "other", "partitions": []}', encoding="utf-8")
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=broken, silver_root=tmp_path / "out", workers=1
        )
    two_root = tmp_path / "two-silver"
    for name in ("ordinary_universe_a", "ordinary_universe_b"):
        day_dir = two_root / name
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "manifest.json").write_text(
            json.dumps({"dataset_id": name, "partitions": [{"session": "2026-03-04"}]}), encoding="utf-8"
        )
    with pytest.raises(PITDataError):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=two_root, silver_root=tmp_path / "out", workers=1
        )


def test_materialize_rejects_malformed_universe_partitions(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    bronze_root.mkdir(parents=True, exist_ok=True)
    cases = [
        {"dataset_id": "ordinary_universe_bad", "partitions": ["nope"]},
        {"dataset_id": "ordinary_universe_bad", "partitions": [{"session": "xx"}]},
        {
            "dataset_id": "ordinary_universe_bad",
            "partitions": [{"session": "2026-03-05"}, {"session": "2026-03-04"}],
        },
    ]
    for index, manifest in enumerate(cases):
        root = tmp_path / f"upart-{index}"
        dataset = root / "ordinary_universe_bad"
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(PITDataError):
            materialize_investor_flow_silver(
                bronze_root=bronze_root, universe_root=root, silver_root=tmp_path / "out", workers=1
            )


def test_materialize_uses_default_policy_and_workers(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("20260304")], start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root)
    assert result.rows == 1
    assert InvestorFlowSilverPolicy().available_session_lag == 1


_ZERO_FLOWS = dict.fromkeys((
    "tjj0000", "tjj0001", "tjj0002", "tjj0003", "tjj0004", "tjj0005", "tjj0006",
    "tjj0007", "tjj0008", "tjj0009", "tjj0010", "tjj0011", "tjj0016", "tjj0017", "tjj0018",
), "0")


def test_materialize_skips_zero_flow_dateless_provider_row(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    rows = [_row("20260304"), _row("", **_ZERO_FLOWS)]
    _write_page(bronze_root, _raw_page("005930", rows, start="2026-03-04", end="2026-03-06"))
    result = materialize_investor_flow_silver(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
    )
    assert result.dateless_rows == 1
    assert result.rows == 1
    assert _frame(result.dataset_path)["individual_net_shares"].to_list() == [927]
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dateless_rows"] == 1


def test_materialize_rejects_dateless_row_with_flow_values(tmp_path: Path) -> None:
    bronze_root, universe_root, silver_root = _roots(tmp_path)
    _write_page(bronze_root, _raw_page("005930", [_row("")], start="2026-03-04", end="2026-03-06"))
    with pytest.raises(PITDataError, match="dateless row carries flow values"):
        materialize_investor_flow_silver(
            bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, workers=1
        )
