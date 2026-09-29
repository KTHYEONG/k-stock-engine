"""Trial registry append-only and determinism invariants."""
from __future__ import annotations

from datetime import date, datetime, UTC
from pathlib import Path

import numpy as np
import pytest

from src.core.pit import PITDataError
from src.data.research_protocol import Segment
from src.research.registry import TrialRegistry, TrialReturns

_NOW = datetime(2026, 9, 29, tzinfo=UTC)


def _returns(n: int = 5, benches: tuple[str, ...] = ()) -> TrialReturns:
    sessions = tuple(date(2023, 1, 2 + i) for i in range(n))
    net = np.arange(n, dtype=np.float64) * 0.001
    benchmarks = {name: np.full(n, 0.0005, dtype=np.float64) for name in benches}
    return TrialReturns(sessions=sessions, net=net, benchmarks=benchmarks)


def _record(registry: TrialRegistry, **overrides: object) -> object:
    returns = _returns()
    assert isinstance(overrides, dict)
    params = {
        "family": "a",
        "spec_hash": "hash-a",
        "spec_json": '{"a": 1}',
        "segment": Segment.DISCOVERY,
        "sim_config_json": '{"sim": 1}',
        "cube_id": "cube-1",
        "returns": returns,
        "metrics": {"sharpe": 1.0},
        "now": _NOW,
    }
    params.update(overrides)  # type: ignore[arg-type]
    return registry.record(**params)  # type: ignore[arg-type]


def test_recording_is_idempotent(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    first = _record(registry)
    second = _record(registry)
    assert first == second
    assert len((tmp_path / "trials" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 1


def test_non_determinism_rejected(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    _record(registry)
    altered = TrialReturns(
        sessions=tuple(date(2023, 1, 2 + i) for i in range(5)),
        net=np.full(5, 0.09, dtype=np.float64),
        benchmarks={},
    )
    with pytest.raises(PITDataError):
        registry.record(
            family="a",
            spec_hash="hash-a",
            spec_json='{"a": 1}',
            segment=Segment.DISCOVERY,
            sim_config_json='{"sim": 1}',
            cube_id="cube-1",
            returns=altered,
            metrics={"sharpe": 1.0},
            now=_NOW,
        )
    assert len((tmp_path / "trials" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 1


def test_round_trip_preserves_returns(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    returns = _returns(6, ("eq", "cap"))
    record = registry.record(
        family="a",
        spec_hash="hash-a",
        spec_json='{"a": 1}',
        segment=Segment.DISCOVERY,
        sim_config_json='{"sim": 1}',
        cube_id="cube-1",
        returns=returns,
        metrics={},
        now=_NOW,
    )
    stored = registry.returns(record.trial_id)  # type: ignore[attr-defined]
    assert stored.sessions == returns.sessions
    assert np.array_equal(stored.net, returns.net)
    assert set(stored.benchmarks) == {"eq", "cap"}
    assert np.array_equal(stored.benchmarks["eq"], returns.benchmarks["eq"])


def test_filters_by_segment_and_family(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    registry.record(
        family="a", spec_hash="h1", spec_json='{"v": 1}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 1}', cube_id="c1", returns=_returns(), metrics={}, now=_NOW,
    )
    registry.record(
        family="b", spec_hash="h2", spec_json='{"v": 2}', segment=Segment.HOLDOUT,
        sim_config_json='{"s": 1}', cube_id="c1", returns=_returns(), metrics={}, now=_NOW,
    )
    registry.record(
        family="a", spec_hash="h3", spec_json='{"v": 3}', segment=Segment.HOLDOUT,
        sim_config_json='{"s": 1}', cube_id="c1", returns=_returns(), metrics={}, now=_NOW,
    )
    assert len(registry.trials(segment=Segment.DISCOVERY, family="a")) == 1
    assert len(registry.trials(segment=Segment.HOLDOUT)) == 2


def test_torn_index_line_fails_closed(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    _record(registry)
    index = tmp_path / "trials" / "index.jsonl"
    with index.open("a", encoding="utf-8") as handle:
        handle.write('{"trial_id": "torn"\n')
    with pytest.raises(PITDataError):
        registry.trials()


def test_family_filter_without_segment(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    registry.record(
        family="a", spec_hash="h1", spec_json='{"v": 1}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 1}', cube_id="c1", returns=_returns(), metrics={}, now=_NOW,
    )
    registry.record(
        family="b", spec_hash="h2", spec_json='{"v": 2}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 2}', cube_id="c1", returns=_returns(), metrics={}, now=_NOW,
    )
    assert len(registry.trials(family="a")) == 1


def test_returns_missing_file_fails_closed(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    with pytest.raises(PITDataError):
        registry.returns("absent0123456789abcdef")


def test_returns_invalid_columns_fail_closed(tmp_path: Path) -> None:
    import polars as pl

    registry = TrialRegistry(tmp_path / "trials")
    returns_dir = tmp_path / "trials" / "returns"
    returns_dir.mkdir(parents=True)
    pl.DataFrame({"session": [date(2023, 1, 2)], "other": [1.0]}).write_parquet(
        returns_dir / "badcolumns0123456789ab.parquet"
    )
    with pytest.raises(PITDataError):
        registry.returns("badcolumns0123456789ab")


def test_check_arrays_rejects_bad_shapes_and_nonfinite(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    sessions = tuple(date(2023, 1, 2 + i) for i in range(3))
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", segment=Segment.DISCOVERY,
            sim_config_json="{}", cube_id="c",
            returns=TrialReturns(sessions=(), net=np.array([], dtype=np.float64), benchmarks={}),
            metrics={}, now=_NOW,
        )
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", segment=Segment.DISCOVERY,
            sim_config_json="{}", cube_id="c",
            returns=TrialReturns(sessions=sessions, net=np.ones(2, dtype=np.float64), benchmarks={}),
            metrics={}, now=_NOW,
        )
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", segment=Segment.DISCOVERY,
            sim_config_json="{}", cube_id="c",
            returns=TrialReturns(
                sessions=sessions, net=np.ones(3, dtype=np.float64),
                benchmarks={"b": np.ones(2, dtype=np.float64)},
            ),
            metrics={}, now=_NOW,
        )
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", segment=Segment.DISCOVERY,
            sim_config_json="{}", cube_id="c",
            returns=TrialReturns(
                sessions=sessions, net=np.array([0.1, np.inf, 0.1], dtype=np.float64), benchmarks={},
            ),
            metrics={}, now=_NOW,
        )
    with pytest.raises(PITDataError):
        registry.record(
            family="a", spec_hash="h", spec_json="{}", segment=Segment.DISCOVERY,
            sim_config_json="{}", cube_id="c",
            returns=TrialReturns(
                sessions=sessions, net=np.ones(3, dtype=np.float64),
                benchmarks={"b": np.array([0.1, np.nan, 0.1], dtype=np.float64)},
            ),
            metrics={}, now=_NOW,
        )


def test_unreadable_index_fails_closed(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    index_dir = tmp_path / "trials" / "index.jsonl"
    index_dir.mkdir(parents=True)
    with pytest.raises(PITDataError):
        registry.trials()


def test_invalid_index_object_fails_closed(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    (tmp_path / "trials").mkdir(parents=True, exist_ok=True)
    (tmp_path / "trials" / "index.jsonl").write_text('{"family": "a"}\n', encoding="utf-8")
    with pytest.raises(PITDataError):
        registry.trials()


def test_existing_returns_file_reused_when_index_lost(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials")
    record = _record(registry)
    assert record is not None
    (tmp_path / "trials" / "index.jsonl").unlink()
    rerun = registry.record(
        family="a", spec_hash="hash-a", spec_json='{"a": 1}', segment=Segment.DISCOVERY,
        sim_config_json='{"sim": 1}', cube_id="cube-1", returns=_returns(), metrics={"sharpe": 1.0},
        now=_NOW,
    )
    assert rerun.trial_id == record.trial_id  # type: ignore[attr-defined]
