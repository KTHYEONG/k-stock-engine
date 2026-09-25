"""Gold reference-benchmark materialization tests."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.data.reference_benchmarks import (
    BenchmarkDefinition,
    Weighting,
    load_benchmark_definitions,
    materialize_reference_benchmarks,
)
from src.data.schemas import PITDataError

DAY0 = date(2020, 1, 6)
DAY1 = date(2020, 1, 7)
DAY2 = date(2020, 1, 8)
DAY3 = date(2020, 1, 9)

EW = BenchmarkDefinition(benchmark_id="ew_pr", weighting=Weighting.EQUAL, min_adtv20_krw=None)
CW = BenchmarkDefinition(benchmark_id="cw_pr", weighting=Weighting.CAP, min_adtv20_krw=None)
LIQ = BenchmarkDefinition(benchmark_id="liq_pr", weighting=Weighting.EQUAL, min_adtv20_krw=1_000_000_000.0)

_PANEL_SCHEMA: dict[str, object] = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "eligible": pl.Boolean,
    "price_state": pl.String,
    "adtv20": pl.Float64,
    "market_cap": pl.Int64,
    "ret_price": pl.Float64,
}


def _prow(
    session: date,
    instrument_id: str,
    *,
    eligible: bool = True,
    price_state: str = "tradable",
    adtv20: float | None = 2_000_000_000.0,
    market_cap: int = 1000,
    ret: float | None = 0.0,
) -> dict[str, object]:
    return {
        "session": session,
        "instrument_id": instrument_id,
        "eligible": eligible,
        "price_state": price_state,
        "adtv20": adtv20,
        "market_cap": market_cap,
        "ret_price": ret,
    }


def _write_panel(root: Path, name: str, rows: list[dict[str, object]]) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    by_year: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        session = row["session"]
        assert isinstance(session, date)
        by_year.setdefault(session.year, []).append(row)
    partitions = {
        f"year={year}/part.parquet": pl.DataFrame(by_year[year], schema=_PANEL_SCHEMA)
        for year in sorted(by_year)
    }
    return publish_dataset(
        layer_root=root,
        identity=DatasetIdentity(
            kind="market_panel",
            layer=DatasetLayer.GOLD,
            policy_version="krx-market-panel-v2",
            inputs={},
            params={},
        ),
        partitions=partitions,
    ).path


def _inputs(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    panel_path = _write_panel(tmp_path / "gold", "market_panel_test", rows)
    return panel_path, tmp_path / "gold"


def _frame(result_path: Path) -> pl.DataFrame:
    return pl.read_parquet(result_path / "benchmarks.parquet")


def _cell(frame: pl.DataFrame, benchmark_id: str, session: date, column: str) -> object:
    values = frame.filter(
        (pl.col("benchmark_id") == benchmark_id) & (pl.col("session") == session)
    )[column].to_list()
    assert len(values) == 1
    return values[0]


def test_materialize_equal_weight_return(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A"),
        _prow(DAY0, "KRX:B"),
        _prow(DAY0, "KRX:C", eligible=False),
        _prow(DAY1, "KRX:A", ret=0.10),
        _prow(DAY1, "KRX:B", ret=-0.10),
        _prow(DAY1, "KRX:C", eligible=False, ret=0.50),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    frame = _frame(result.dataset_path)
    assert _cell(frame, "ew_pr", DAY1, "ret") == pytest.approx(0.0)
    assert _cell(frame, "ew_pr", DAY1, "constituents") == 2
    assert _cell(frame, "ew_pr", DAY1, "dropped_exit_weight") == pytest.approx(0.0)


def test_materialize_cap_weights_from_prior_session(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A", market_cap=300),
        _prow(DAY0, "KRX:B", market_cap=100),
        _prow(DAY1, "KRX:A", market_cap=330, ret=0.04),
        _prow(DAY1, "KRX:B", market_cap=100, ret=0.0),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(CW,), definitions_version="test-v1", gold_root=gold_root
    )
    assert _cell(_frame(result.dataset_path), "cw_pr", DAY1, "ret") == pytest.approx(0.03)


def test_materialize_constituents_fixed_at_prior_session(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A"),
        _prow(DAY0, "KRX:B"),
        _prow(DAY0, "KRX:C", eligible=False),
        _prow(DAY1, "KRX:A", ret=0.01),
        _prow(DAY1, "KRX:B", ret=0.01),
        _prow(DAY1, "KRX:C", ret=0.01),
        _prow(DAY2, "KRX:A", ret=0.01),
        _prow(DAY2, "KRX:B", ret=0.01),
        _prow(DAY2, "KRX:C", ret=0.01),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    frame = _frame(result.dataset_path)
    assert _cell(frame, "ew_pr", DAY1, "constituents") == 2
    assert _cell(frame, "ew_pr", DAY2, "constituents") == 3


def test_materialize_liquidity_threshold_uses_prior_adtv(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A", adtv20=500_000_000.0),
        _prow(DAY0, "KRX:B", adtv20=2_000_000_000.0),
        _prow(DAY1, "KRX:A", adtv20=2_000_000_000.0, ret=0.02),
        _prow(DAY1, "KRX:B", adtv20=2_000_000_000.0, ret=0.02),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(LIQ,), definitions_version="test-v1", gold_root=gold_root
    )
    assert _cell(_frame(result.dataset_path), "liq_pr", DAY1, "constituents") == 1


def test_materialize_exit_drops_and_renormalizes(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A"),
        _prow(DAY0, "KRX:B"),
        _prow(DAY1, "KRX:B", ret=0.06),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    frame = _frame(result.dataset_path)
    assert _cell(frame, "ew_pr", DAY1, "dropped_exit_weight") == pytest.approx(0.5)
    assert _cell(frame, "ew_pr", DAY1, "ret") == pytest.approx(0.06)
    assert _cell(frame, "ew_pr", DAY1, "constituents") == 2
    assert result.dropped_exit_weight_max == pytest.approx(0.5)


def test_materialize_index_chaining(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A"),
        _prow(DAY1, "KRX:A", ret=0.10),
        _prow(DAY2, "KRX:A", ret=-0.05),
        _prow(DAY3, "KRX:A", ret=0.02),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    frame = _frame(result.dataset_path).filter(pl.col("benchmark_id") == "ew_pr").sort("session")
    rets = frame["ret"].to_list()
    assert rets[0] is None
    expected = 1.0
    for session, ret in zip((DAY1, DAY2, DAY3), rets[1:], strict=True):
        expected *= 1 + ret
        assert _cell(frame, "ew_pr", session, "index_level") == pytest.approx(expected)
    assert _cell(frame, "ew_pr", DAY0, "index_level") == pytest.approx(1.0)
    assert math.isclose(frame["index_level"].to_list()[-1], 1.1 * 0.95 * 1.02, rel_tol=1e-12)


def test_materialize_invalid_prior_state_excluded(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A", price_state="invalid", ret=None),
        _prow(DAY0, "KRX:B"),
        _prow(DAY1, "KRX:A", ret=0.10),
        _prow(DAY1, "KRX:B", ret=0.10),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    assert _cell(_frame(result.dataset_path), "ew_pr", DAY1, "constituents") == 1


def test_load_benchmark_definitions_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "benchmarks.toml"
    path.write_text(
        'version = "reference-benchmarks-v1"\n'
        '[[benchmarks]]\nbenchmark_id = "a"\nweighting = "equal"\n'
        '[[benchmarks]]\nbenchmark_id = "a"\nweighting = "cap"\n',
        encoding="utf-8",
    )
    with pytest.raises(PITDataError):
        load_benchmark_definitions(path)


@pytest.mark.parametrize(
    "body",
    [
        '[[benchmarks]]\nbenchmark_id = "a"\nweighting = "weird"\n',
        '[[benchmarks]]\nbenchmark_id = "a"\nweighting = "equal"\nmin_adtv20_krw = 0\n',
        '[[benchmarks]]\nbenchmark_id = "a"\nweighting = "equal"\nmin_adtv20_krw = -5\n',
        'version = "other"\n[[benchmarks]]\nbenchmark_id = "a"\nweighting = "equal"\n',
    ],
)
def test_load_benchmark_definitions_rejects_bad_values(tmp_path: Path, body: str) -> None:
    path = tmp_path / "benchmarks.toml"
    path.write_text('version = "reference-benchmarks-v1"\n' + body if "version" not in body else body,
                    encoding="utf-8")
    with pytest.raises(PITDataError):
        load_benchmark_definitions(path)


def test_load_benchmark_definitions_reads_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "benchmarks.toml"
    path.write_text(
        'version = "reference-benchmarks-v1"\n'
        '[[benchmarks]]\nbenchmark_id = "ew"\nweighting = "equal"\n'
        '[[benchmarks]]\nbenchmark_id = "liq"\nweighting = "cap"\nmin_adtv20_krw = 1000000000\n',
        encoding="utf-8",
    )
    version, definitions = load_benchmark_definitions(path)
    assert version == "reference-benchmarks-v1"
    assert definitions == (
        BenchmarkDefinition(benchmark_id="ew", weighting=Weighting.EQUAL, min_adtv20_krw=None),
        BenchmarkDefinition(benchmark_id="liq", weighting=Weighting.CAP, min_adtv20_krw=1_000_000_000.0),
    )


def test_materialize_future_perturbation_invariance(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A"),
        _prow(DAY0, "KRX:B"),
        _prow(DAY1, "KRX:A", ret=0.01),
        _prow(DAY1, "KRX:B", ret=0.02),
        _prow(DAY2, "KRX:A", ret=0.03),
        _prow(DAY2, "KRX:B", ret=0.04),
        _prow(DAY3, "KRX:A", ret=0.05),
        _prow(DAY3, "KRX:B", ret=0.06),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    first = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW, CW), definitions_version="test-v1", gold_root=gold_root
    )
    before = _frame(first.dataset_path)
    perturbed = [dict(row, ret=0.50) if row["session"] == DAY3 else dict(row) for row in rows]
    panel_path2 = _write_panel(tmp_path / "gold2", "market_panel_test", perturbed)
    second = materialize_reference_benchmarks(
        market_panel_path=panel_path2, definitions=(EW, CW), definitions_version="test-v1",
        gold_root=tmp_path / "gold2",
    )
    after = _frame(second.dataset_path)
    left = before.filter(pl.col("session") <= DAY2).sort(["benchmark_id", "session"])
    right = after.filter(pl.col("session") <= DAY2).sort(["benchmark_id", "session"])
    assert left.equals(right)


def test_materialize_year_boundary_continuity(tmp_path: Path) -> None:
    dec30 = date(2025, 12, 30)
    dec31 = date(2025, 12, 31)
    jan02 = date(2026, 1, 2)
    rows = [
        _prow(dec30, "KRX:A"),
        _prow(dec30, "KRX:B", market_cap=100),
        _prow(dec31, "KRX:A", market_cap=300, ret=0.01),
        _prow(dec31, "KRX:B", market_cap=100, ret=0.01),
        _prow(jan02, "KRX:A", market_cap=330, ret=0.04),
        _prow(jan02, "KRX:B", market_cap=100, ret=0.0),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(CW,), definitions_version="test-v1", gold_root=gold_root
    )
    frame = _frame(result.dataset_path)
    assert _cell(frame, "cw_pr", jan02, "ret") == pytest.approx(0.03)
    assert _cell(frame, "cw_pr", jan02, "index_level") == pytest.approx(
        _cell(frame, "cw_pr", dec31, "index_level") * 1.03
    )


def test_materialize_invalid_current_row_dropped_like_exit(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A"),
        _prow(DAY0, "KRX:B"),
        _prow(DAY1, "KRX:A", price_state="invalid", ret=None),
        _prow(DAY1, "KRX:B", ret=0.08),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    frame = _frame(result.dataset_path)
    assert _cell(frame, "ew_pr", DAY1, "dropped_exit_weight") == pytest.approx(0.5)
    assert _cell(frame, "ew_pr", DAY1, "ret") == pytest.approx(0.08)


def test_materialize_gap_instrument_excluded(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A"),
        _prow(DAY0, "KRX:B"),
        _prow(DAY1, "KRX:B", ret=0.01),
        _prow(DAY2, "KRX:A", ret=0.05),
        _prow(DAY2, "KRX:B", ret=0.05),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    assert _cell(_frame(result.dataset_path), "ew_pr", DAY2, "constituents") == 1


def test_materialize_rejects_empty_constituents_after_first(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A", eligible=False),
        _prow(DAY1, "KRX:A", ret=0.01),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    with pytest.raises(PITDataError):
        materialize_reference_benchmarks(
            market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
        )


def test_materialize_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A"),
        _prow(DAY1, "KRX:A", ret=0.02),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    first = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    before = (first.dataset_path / "manifest.json").read_bytes()
    second = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    assert first.dataset_id == second.dataset_id
    assert second.dataset_path == first.dataset_path
    assert (first.dataset_path / "manifest.json").read_bytes() == before
    assert first.dataset_id.startswith("reference_benchmarks_")
    assert first.sessions == 2
    assert first.benchmarks == ("ew_pr",)


@pytest.mark.parametrize(
    "body",
    [
        "not toml [[[\n",
        '[[benchmarks]]\nbenchmark_id = "a"\nweighting = "equal"\n',
        'version = "reference-benchmarks-v1"\nbenchmarks = "nope"\n',
        'version = "reference-benchmarks-v1"\nbenchmarks = [1]\n',
        'version = "reference-benchmarks-v1"\n[[benchmarks]]\nbenchmark_id = "a"\nweighting = "equal"\nextra = 1\n',
        'version = "reference-benchmarks-v1"\n[[benchmarks]]\nweighting = "equal"\n',
    ],
)
def test_load_benchmark_definitions_rejects_malformed_files(tmp_path: Path, body: str) -> None:
    path = tmp_path / "benchmarks.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(PITDataError):
        load_benchmark_definitions(path)


def test_load_benchmark_definitions_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        load_benchmark_definitions(tmp_path / "missing.toml")


def test_materialize_rejects_bad_panel_manifest(tmp_path: Path) -> None:
    with pytest.raises(PITDataError):
        materialize_reference_benchmarks(
            market_panel_path=tmp_path / "missing",
            definitions=(EW,),
            definitions_version="test-v1",
            gold_root=tmp_path / "gold",
        )
    rows = [_prow(DAY0, "KRX:A"), _prow(DAY1, "KRX:A", ret=0.01)]
    panel_path = _write_panel(tmp_path / "gold", "market_panel_test", rows)
    (panel_path / "manifest.json").write_text('{"dataset_id": "other", "partitions": []}', encoding="utf-8")
    with pytest.raises(PITDataError):
        materialize_reference_benchmarks(
            market_panel_path=panel_path,
            definitions=(EW,),
            definitions_version="test-v1",
            gold_root=tmp_path / "gold",
        )


def test_materialize_rejects_tampered_panel_partition(tmp_path: Path) -> None:
    rows = [_prow(DAY0, "KRX:A"), _prow(DAY1, "KRX:A", ret=0.01)]
    panel_path = _write_panel(tmp_path / "gold", "market_panel_test", rows)
    part_path = panel_path / "year=2020" / "part.parquet"
    part_path.write_bytes(part_path.read_bytes() + b" ")
    with pytest.raises(PITDataError):
        materialize_reference_benchmarks(
            market_panel_path=panel_path,
            definitions=(EW,),
            definitions_version="test-v1",
            gold_root=tmp_path / "gold",
        )


def test_materialize_all_dropped_constituents_raise(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A"),
        _prow(DAY0, "KRX:B"),
        _prow(DAY1, "KRX:A", price_state="invalid", ret=None),
        _prow(DAY1, "KRX:B", price_state="invalid", ret=None),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    with pytest.raises(PITDataError):
        materialize_reference_benchmarks(
            market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
        )


@pytest.mark.parametrize(
    "partitions",
    [
        ["nope"],
        [{"year": 2020, "path": "p"}],
        [{"year": "2020", "path": "p", "parquet_sha256": "s"}],
    ],
)
def test_materialize_rejects_malformed_partitions(tmp_path: Path, partitions: list[object]) -> None:
    panel_path = tmp_path / "gold" / "market_panel_test"
    panel_path.mkdir(parents=True, exist_ok=True)
    (panel_path / "manifest.json").write_text(
        json.dumps({"dataset_id": "market_panel_test", "partitions": partitions}), encoding="utf-8"
    )
    with pytest.raises(PITDataError):
        materialize_reference_benchmarks(
            market_panel_path=panel_path,
            definitions=(EW,),
            definitions_version="test-v1",
            gold_root=tmp_path / "gold",
        )


def test_materialize_rejects_unreadable_partition(tmp_path: Path) -> None:
    rows = [_prow(DAY0, "KRX:A"), _prow(DAY1, "KRX:A", ret=0.01)]
    panel_path = _write_panel(tmp_path / "gold", "market_panel_test", rows)
    (panel_path / "year=2020" / "part.parquet").unlink()
    with pytest.raises(PITDataError):
        materialize_reference_benchmarks(
            market_panel_path=panel_path,
            definitions=(EW,),
            definitions_version="test-v1",
            gold_root=tmp_path / "gold",
        )


def test_materialize_rejects_differing_existing_dataset(tmp_path: Path) -> None:
    rows = [_prow(DAY0, "KRX:A"), _prow(DAY1, "KRX:A", ret=0.01)]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    (result.dataset_path / "manifest.json").write_text('{"dataset_id": "tampered"}', encoding="utf-8")
    with pytest.raises(PITDataError):
        materialize_reference_benchmarks(
            market_panel_path=panel_path,
            definitions=(EW,),
            definitions_version="test-v1",
            gold_root=gold_root,
        )


def test_materialize_rejects_unreadable_existing_dataset(tmp_path: Path) -> None:
    rows = [_prow(DAY0, "KRX:A"), _prow(DAY1, "KRX:A", ret=0.01)]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
    )
    (result.dataset_path / "manifest.json").unlink()
    with pytest.raises(PITDataError):
        materialize_reference_benchmarks(
            market_panel_path=panel_path,
            definitions=(EW,),
            definitions_version="test-v1",
            gold_root=gold_root,
        )


def test_materialize_liquidity_index_starts_after_warmup(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A", adtv20=None),
        _prow(DAY1, "KRX:A", adtv20=2_000_000_000.0, ret=0.01),
        _prow(DAY2, "KRX:A", adtv20=2_000_000_000.0, ret=0.05),
        _prow(DAY3, "KRX:A", adtv20=2_000_000_000.0, ret=-0.02),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=(LIQ,), definitions_version="test-v1", gold_root=gold_root
    )
    frame = _frame(result.dataset_path).sort("session")
    # 창이 차기 전(DAY0) 구성 불가 → 기준일은 DAY1, 첫 수익률은 DAY2
    assert frame["session"].to_list() == [DAY1, DAY2, DAY3]
    assert frame["ret"].to_list()[0] is None
    assert _cell(frame, "liq_pr", DAY2, "ret") == pytest.approx(0.05)
    assert _cell(frame, "liq_pr", DAY3, "index_level") == pytest.approx(1.05 * 0.98)


def test_materialize_rejects_empty_constituents_after_inception(tmp_path: Path) -> None:
    rows = [
        _prow(DAY0, "KRX:A", ret=0.0),
        _prow(DAY1, "KRX:A", eligible=False, ret=0.01),
        _prow(DAY2, "KRX:A", ret=0.02),
    ]
    panel_path, gold_root = _inputs(tmp_path, rows)
    with pytest.raises(PITDataError, match="empty constituent set"):
        materialize_reference_benchmarks(
            market_panel_path=panel_path, definitions=(EW,), definitions_version="test-v1", gold_root=gold_root
        )


def test_reference_benchmark_panel_reader_boundaries(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from src.data.datasets import PITDataError
    from src.data.reference_benchmarks import _load_panel_frame, materialize_reference_benchmarks
    import src.data.reference_benchmarks as benchmark_module

    panel_path, gold_root = _inputs(tmp_path, [_prow(DAY0, "KRX:A"), _prow(DAY1, "KRX:A")])
    monkeypatch.setattr(
        benchmark_module,
        "dataset_partition_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PITDataError("bad paths")),
    )
    with pytest.raises(PITDataError, match="input manifest"):
        _load_panel_frame(panel_path)
    monkeypatch.undo()

    monkeypatch.setattr(benchmark_module, "load_manifest", lambda _path: (_ for _ in ()).throw(PITDataError("bad manifest")))
    assert _load_panel_frame(panel_path)[0] == panel_path.name
    monkeypatch.undo()

    monkeypatch.setattr(
        benchmark_module,
        "load_manifest",
        lambda _path: SimpleNamespace(kind="daily_market"),
    )
    with pytest.raises(PITDataError, match="input manifest"):
        _load_panel_frame(panel_path)
    monkeypatch.undo()

    monkeypatch.setattr(benchmark_module.pl, "read_parquet", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")))
    with pytest.raises(PITDataError, match="input manifest"):
        _load_panel_frame(panel_path)
    monkeypatch.undo()

    with pytest.raises(PITDataError, match="at least one definition"):
        materialize_reference_benchmarks(
            market_panel_path=panel_path, definitions=(), definitions_version="test-v1", gold_root=gold_root
        )
