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
from src.core.pit import PITDataError

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


def _multi_definition_rows() -> list[dict[str, object]]:
    return [
        _prow(DAY0, "KRX:A", adtv20=2_000_000_000.0, market_cap=1000, ret=None),
        _prow(DAY0, "KRX:B", adtv20=500_000_000.0, market_cap=3000, ret=None),
        _prow(DAY0, "KRX:C", eligible=False, market_cap=500, ret=None),
        _prow(DAY0, "KRX:D", adtv20=None, market_cap=700, ret=None),
        _prow(DAY1, "KRX:A", adtv20=2_000_000_000.0, market_cap=1100, ret=0.0123457),
        _prow(DAY1, "KRX:B", adtv20=600_000_000.0, market_cap=2900, ret=-0.0456789),
        _prow(DAY1, "KRX:C", eligible=False, market_cap=500, ret=0.5),
        _prow(DAY1, "KRX:D", adtv20=1_500_000_000.0, market_cap=720, ret=0.0271828),
        _prow(DAY2, "KRX:A", adtv20=2_100_000_000.0, market_cap=1050, ret=0.031415),
        _prow(DAY2, "KRX:B", adtv20=1_200_000_000.0, market_cap=3100, ret=0.00001),
        _prow(DAY2, "KRX:C", adtv20=3_000_000_000.0, market_cap=2222, ret=-0.007),
        _prow(DAY2, "KRX:D", price_state="halted", adtv20=1_500_000_000.0, market_cap=715, ret=0.1),
        _prow(DAY3, "KRX:A", adtv20=2_000_000_000.0, market_cap=1020, ret=-0.0123),
        _prow(DAY3, "KRX:B", adtv20=1_300_000_000.0, market_cap=3300, ret=0.0777),
        _prow(DAY3, "KRX:C", adtv20=2_900_000_000.0, market_cap=2100, ret=0.0333),
        _prow(DAY3, "KRX:D", adtv20=2_000_000_000.0, market_cap=999, ret=0.0002),
    ]


def _legacy_benchmark_frame(
    *,
    base: pl.DataFrame,
    calendar: pl.DataFrame,
    sessions: list[date],
    definition: BenchmarkDefinition,
) -> pl.DataFrame:
    """Pre-hoist per-definition algorithm, kept as the bit-identity oracle."""
    from src.data.reference_benchmarks import _OUTPUT_SCHEMA

    weight = (
        pl.lit(1.0)
        if definition.weighting is Weighting.EQUAL
        else pl.col("market_cap").cast(pl.Float64)
    )
    candidates = (
        base.with_columns(
            w=weight,
            next_ret=pl.col("ret_price").shift(-1).over("instrument_id"),
            next_pos=pl.col("pos").shift(-1).over("instrument_id"),
            target_pos=pl.col("pos") + 1,
        )
        .join(
            calendar.select(pl.col("session").alias("t"), pl.col("pos").alias("target_pos")),
            on="target_pos",
            how="left",
        )
        .filter(
            pl.col("eligible")
            & (pl.col("price_state") == "tradable")
            & (
                pl.lit(True)
                if definition.min_adtv20_krw is None
                else pl.col("adtv20").is_not_null() & (pl.col("adtv20") >= definition.min_adtv20_krw)
            )
            & pl.col("t").is_not_null()
        )
        .select(
            "t",
            "w",
            "next_ret",
            kept=(pl.col("next_pos") == pl.col("target_pos")) & pl.col("next_ret").is_not_null(),
        )
    )
    grouped = (
        candidates.group_by("t", maintain_order=True)
        .agg(
            n_total=pl.len(),
            total_w=pl.col("w").sum(),
            kept_n=pl.col("kept").sum(),
            kept_w=pl.when(pl.col("kept")).then(pl.col("w")).otherwise(0.0).sum(),
            ret_num=pl.when(pl.col("kept")).then(pl.col("w") * pl.col("next_ret")).otherwise(0.0).sum(),
        )
        .sort("t")
    )
    inception_index = sessions.index(grouped["t"][0]) - 1
    inception = sessions[inception_index]
    expected = sessions[inception_index + 1 :]
    formed = set(grouped["t"].to_list())
    assert all(session in formed for session in expected)
    body = grouped.with_columns(
        session=pl.col("t"),
        benchmark_id=pl.lit(definition.benchmark_id),
        ret=pl.col("ret_num") / pl.col("kept_w"),
        constituents=pl.col("n_total").cast(pl.Int32),
        dropped_exit_weight=(pl.col("total_w") - pl.col("kept_w")) / pl.col("total_w"),
    ).select("session", "benchmark_id", "ret", "constituents", "dropped_exit_weight")
    first = pl.DataFrame(
        {
            "session": [inception],
            "benchmark_id": [definition.benchmark_id],
            "ret": [None],
            "constituents": [0],
            "dropped_exit_weight": [0.0],
        },
        schema={
            "session": pl.Date,
            "benchmark_id": pl.String,
            "ret": pl.Float64,
            "constituents": pl.Int32,
            "dropped_exit_weight": pl.Float64,
        },
    )
    return (
        pl.concat([first, body], how="vertical_relaxed")
        .sort("session")
        .with_columns(index_level=(pl.lit(1.0) + pl.col("ret").fill_null(0.0)).cum_prod())
        .select(list(_OUTPUT_SCHEMA))
    )


def test_benchmark_output_bit_identical_after_hoist(tmp_path: Path) -> None:
    from polars.testing import assert_frame_equal

    from src.data.reference_benchmarks import _load_panel_frame

    definitions = (EW, CW, LIQ)
    panel_path, gold_root = _inputs(tmp_path, _multi_definition_rows())
    result = materialize_reference_benchmarks(
        market_panel_path=panel_path, definitions=definitions, definitions_version="test-v1", gold_root=gold_root
    )
    _, frame, sessions = _load_panel_frame(panel_path)
    calendar = frame.select("session").unique().sort("session").with_row_index("pos")
    base = frame.join(calendar, on="session", how="left").sort(["instrument_id", "session"])
    expected = pl.concat(
        [
            _legacy_benchmark_frame(base=base, calendar=calendar, sessions=sessions, definition=definition)
            for definition in definitions
        ],
        how="vertical_relaxed",
    ).sort(["benchmark_id", "session"])

    assert_frame_equal(_frame(result.dataset_path), expected, check_exact=True)


def test_hoisted_prepared_not_mutated_between_definitions(tmp_path: Path) -> None:
    from polars.testing import assert_frame_equal

    rows = _multi_definition_rows()
    solo_panel, solo_gold = _inputs(tmp_path / "solo", rows)
    solo = materialize_reference_benchmarks(
        market_panel_path=solo_panel, definitions=(CW,), definitions_version="test-v1", gold_root=solo_gold
    )
    paired_panel, paired_gold = _inputs(tmp_path / "paired", rows)
    paired = materialize_reference_benchmarks(
        market_panel_path=paired_panel,
        definitions=(EW, CW),
        definitions_version="test-v1",
        gold_root=paired_gold,
    )

    assert_frame_equal(
        _frame(solo.dataset_path).filter(pl.col("benchmark_id") == "cw_pr"),
        _frame(paired.dataset_path).filter(pl.col("benchmark_id") == "cw_pr"),
        check_exact=True,
    )
