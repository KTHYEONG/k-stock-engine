"""Run registry append-only and determinism invariants."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, UTC
from pathlib import Path

import numpy as np
import pytest

from src.core.pit import PITDataError
from src.research.registry import RunRegistry, RunReturns

_NOW = datetime(2026, 9, 29, tzinfo=UTC)


def _sole_return_id(root: Path) -> str:
    files = sorted((root / "returns").glob("*.parquet"))
    assert len(files) == 1
    return files[0].stem


def _returns(n: int = 5, benches: tuple[str, ...] = ()) -> RunReturns:
    sessions = tuple(date(2023, 1, 2 + i) for i in range(n))
    net = np.arange(n, dtype=np.float64) * 0.001
    benchmarks = {name: np.full(n, 0.0005, dtype=np.float64) for name in benches}
    return RunReturns(sessions=sessions, net=net, benchmarks=benchmarks)


def _record(registry: RunRegistry, **overrides: object) -> object:
    returns = _returns()
    assert isinstance(overrides, dict)
    params = {
        "family": "a",
        "spec_hash": "hash-a",
        "spec_json": '{"a": 1}',
        "run_id": "run0123456789abcdef",
        "scenario": "base",
        "capital_krw": 100_000_000,
        "sim_config_json": '{"scenario": "base"}',
        "cube_id": "cube-1",
        "protocol_hash": "p" * 64,
        "engine_config_hash": "e" * 64,
        "report_digest": "d" * 64,
        "returns": returns,
        "metrics": {"objective_j": 0.1},
        "now": _NOW,
    }
    params.update(overrides)  # type: ignore[arg-type]
    return registry.record(**params)  # type: ignore[arg-type]


def test_recording_is_idempotent(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    first = _record(registry)
    second = _record(registry)
    assert first == second
    assert len((tmp_path / "runs" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 1


def test_scenarios_share_run_id(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    _record(registry, scenario="base")
    _record(registry, scenario="stress_slippage")
    assert len(registry.runs(run_id="run0123456789abcdef")) == 2


def test_non_determinism_rejected(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    _record(registry)
    altered = RunReturns(
        sessions=tuple(date(2023, 1, 2 + i) for i in range(5)),
        net=np.full(5, 0.09, dtype=np.float64),
        benchmarks={},
    )
    with pytest.raises(PITDataError):
        registry.record(
            family="a",
            spec_hash="hash-a",
            spec_json='{"a": 1}',
            run_id="run0123456789abcdef",
            scenario="base",
            capital_krw=100_000_000,
            sim_config_json='{"scenario": "base"}',
            cube_id="cube-1",
            protocol_hash="p" * 64,
            engine_config_hash="e" * 64,
            report_digest="d" * 64,
            returns=altered,
            metrics={"objective_j": 0.1},
            now=_NOW,
        )
    assert len((tmp_path / "runs" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 1


def test_round_trip_preserves_returns(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    returns = _returns(6, ("eq", "cap"))
    record = registry.record(
        family="a",
        spec_hash="hash-a",
        spec_json='{"a": 1}',
        run_id="run1",
        scenario="base",
        capital_krw=250_000_000,
        sim_config_json="{}",
        cube_id="cube-1",
        protocol_hash="p",
        engine_config_hash="e",
        report_digest="d",
        returns=returns,
        metrics={},
        now=_NOW,
    )
    stored = registry.returns(_sole_return_id(tmp_path / "runs"))
    assert record.report_digest == "d"
    assert record.capital_krw == 250_000_000
    assert stored.sessions == returns.sessions
    assert np.array_equal(stored.net, returns.net)
    assert set(stored.benchmarks) == {"eq", "cap"}
    assert np.array_equal(stored.benchmarks["eq"], returns.benchmarks["eq"])


def test_filters_by_run_and_family(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    registry.record(
        family="a", spec_hash="h1", spec_json='{"v": 1}', run_id="r1", scenario="base",
        capital_krw=100_000_000, sim_config_json="{}", cube_id="c1", protocol_hash="p", engine_config_hash="e",
        report_digest="d", returns=_returns(), metrics={}, now=_NOW,
    )
    registry.record(
        family="b", spec_hash="h2", spec_json='{"v": 2}', run_id="r1", scenario="stress_slippage",
        capital_krw=100_000_000,
        sim_config_json="{}", cube_id="c1", protocol_hash="p", engine_config_hash="e",
        report_digest="d", returns=_returns(), metrics={}, now=_NOW,
    )
    registry.record(
        family="a", spec_hash="h3", spec_json='{"v": 3}', run_id="r2", scenario="base",
        capital_krw=100_000_000, sim_config_json="{}", cube_id="c1", protocol_hash="p", engine_config_hash="e",
        report_digest="d", returns=_returns(), metrics={}, now=_NOW,
    )
    assert len(registry.runs(run_id="r1", family="a")) == 1
    assert len(registry.runs(run_id="r1")) == 2


def test_family_filter_without_run(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    registry.record(
        family="a", spec_hash="h1", spec_json='{"v": 1}', run_id="r1", scenario="base",
        capital_krw=100_000_000, sim_config_json="{}", cube_id="c1", protocol_hash="p", engine_config_hash="e",
        report_digest="d", returns=_returns(), metrics={}, now=_NOW,
    )
    registry.record(
        family="b", spec_hash="h2", spec_json='{"v": 2}', run_id="r2", scenario="base",
        capital_krw=100_000_000, sim_config_json="{}", cube_id="c1", protocol_hash="p", engine_config_hash="e",
        report_digest="d", returns=_returns(), metrics={}, now=_NOW,
    )
    assert len(registry.runs(family="a")) == 1


def test_torn_index_line_fails_closed(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    _record(registry)
    index = tmp_path / "runs" / "index.jsonl"
    with index.open("a", encoding="utf-8") as handle:
        handle.write('{"record_id": "torn"\n')
    with pytest.raises(PITDataError):
        registry.runs()


def test_returns_missing_file_fails_closed(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    with pytest.raises(PITDataError):
        registry.returns("absent0123456789abcdef")


def test_returns_invalid_columns_fail_closed(tmp_path: Path) -> None:
    import polars as pl

    registry = RunRegistry(tmp_path / "runs")
    returns_dir = tmp_path / "runs" / "returns"
    returns_dir.mkdir(parents=True)
    pl.DataFrame({"session": [date(2023, 1, 2)], "other": [1.0]}).write_parquet(
        returns_dir / "badcolumns0123456789ab.parquet"
    )
    with pytest.raises(PITDataError):
        registry.returns("badcolumns0123456789ab")


def test_check_arrays_rejects_bad_shapes_and_nonfinite(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    sessions = tuple(date(2023, 1, 2 + i) for i in range(3))
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", run_id="r", scenario="base",
            capital_krw=100_000_000, sim_config_json="{}", cube_id="c", protocol_hash="p", engine_config_hash="e",
            report_digest="d",
            returns=RunReturns(sessions=(), net=np.array([], dtype=np.float64), benchmarks={}),
            metrics={}, now=_NOW,
        )
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", run_id="r", scenario="base",
            capital_krw=100_000_000, sim_config_json="{}", cube_id="c", protocol_hash="p", engine_config_hash="e",
            report_digest="d",
            returns=RunReturns(sessions=sessions, net=np.ones(2, dtype=np.float64), benchmarks={}),
            metrics={}, now=_NOW,
        )
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", run_id="r", scenario="base",
            capital_krw=100_000_000, sim_config_json="{}", cube_id="c", protocol_hash="p", engine_config_hash="e",
            report_digest="d",
            returns=RunReturns(
                sessions=sessions, net=np.ones(3, dtype=np.float64),
                benchmarks={"b": np.ones(2, dtype=np.float64)},
            ),
            metrics={}, now=_NOW,
        )
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", run_id="r", scenario="base",
            capital_krw=100_000_000, sim_config_json="{}", cube_id="c", protocol_hash="p", engine_config_hash="e",
            report_digest="d",
            returns=RunReturns(
                sessions=sessions, net=np.array([0.1, np.inf, 0.1], dtype=np.float64), benchmarks={},
            ),
            metrics={}, now=_NOW,
        )
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", run_id="r", scenario="base",
            capital_krw=100_000_000, sim_config_json="{}", cube_id="c", protocol_hash="p", engine_config_hash="e",
            report_digest="d",
            returns=RunReturns(
                sessions=sessions, net=np.ones(3, dtype=np.float64),
                benchmarks={"b": np.array([0.1, np.nan, 0.1], dtype=np.float64)},
            ),
            metrics={}, now=_NOW,
        )


def test_unreadable_index_fails_closed(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    index_dir = tmp_path / "runs" / "index.jsonl"
    index_dir.mkdir(parents=True)
    with pytest.raises(PITDataError):
        registry.runs()


def test_invalid_index_object_fails_closed(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs" / "index.jsonl").write_text(
        json.dumps({"schema": 2, "family": "a"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(PITDataError):
        registry.runs()


def test_old_schema_lines_are_skipped_and_preserved(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    registry = RunRegistry(tmp_path / "runs")
    index = tmp_path / "runs" / "index.jsonl"
    index.parent.mkdir(parents=True)
    legacy = json.dumps({"record_id": "legacy1", "segment": "holdout", "run_id": "old"}) + "\n"
    index.write_text(legacy + "\n", encoding="utf-8")
    record = _record(registry)
    written = index.read_text(encoding="utf-8")
    assert written.startswith(legacy)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.research.registry"):
        found = registry.runs()
    assert [item.run_id for item in found] == [record.run_id]
    assert sum("older schema" in message for message in caplog.messages) == 1
    assert index.read_text(encoding="utf-8") == written


def test_index_lines_carry_schema_two(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    _record(registry)
    lines = (tmp_path / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["schema"] for line in lines] == [2]


def test_existing_returns_file_reused_when_index_lost(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    record = _record(registry)
    assert record is not None
    (tmp_path / "runs" / "index.jsonl").unlink()
    rerun = registry.record(
        family="a", spec_hash="hash-a", spec_json='{"a": 1}', run_id="run0123456789abcdef",
        scenario="base", capital_krw=100_000_000, sim_config_json='{"scenario": "base"}', cube_id="cube-1",
        protocol_hash="p" * 64, engine_config_hash="e" * 64, report_digest="d" * 64,
        returns=_returns(), metrics={"objective_j": 0.1}, now=_NOW,
    )
    assert rerun == record
