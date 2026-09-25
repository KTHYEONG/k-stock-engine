"""Run persistence: idempotent writes, distinct ids, and manifest integrity."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.backtest.manifest import _code_version, write_run
from src.backtest.metrics import PerformanceSummary, summarize
from src.backtest.strategy import EqualWeightLiquid
from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.market import MarketArrays, load_market_arrays
from src.backtest.events import build_engine_events
from tests.unit.backtest.test_engine import _config, _flat_row, _rules, _sessions
from tests.unit.backtest.test_events import _write_exits
from tests.unit.backtest.test_market import _write_panel


def _mini_run(
    tmp_path: Path, max_names: int
) -> tuple[BacktestResult, PerformanceSummary, MarketArrays, dict[str, Any]]:
    sessions = _sessions(3)
    rows = [_flat_row(day, "KRX:A", 100) for day in sessions]
    rows += [_flat_row(day, "KRX:B", 200) for day in sessions]
    panel = _write_panel(tmp_path / "gold", "market_panel_manifest", rows)
    _write_exits(panel, [])
    arrays = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "cache")
    events = build_engine_events(arrays=arrays, panel_dir=panel, dividends=None)
    strategy = EqualWeightLiquid(min_adtv20_krw=0, max_names=max_names)
    config = _config(initial_cash=1_000_000, impact_k=0.0, commission="0")
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=strategy,
        config=config,
        deposits={},
        asof_tables={},
        rules=_rules(),
        start=sessions[0],
        end=sessions[-1],
    )
    summary = summarize(result=result, arrays=arrays, sessions_per_year=252)
    inputs = {"market_panel": arrays.dataset_id}
    return result, summary, arrays, {"inputs": inputs, "config": config, "strategy": strategy}


def _write(tmp_path: Path, max_names: int, run_root: Path) -> tuple[Path, BacktestResult, PerformanceSummary, dict[str, Any]]:
    result, summary, arrays, parts = _mini_run(tmp_path, max_names)
    path = write_run(
        run_root=run_root,
        result=result,
        summary=summary,
        inputs=parts["inputs"],
        config=parts["config"],
        strategy=parts["strategy"],
        deposits={},
    )
    return path, result, summary, parts


def test_idempotent_rewrite(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    first, _, _, _ = _write(tmp_path, 20, run_root)
    second, _, _, _ = _write(tmp_path, 20, run_root)
    assert first == second
    assert [p for p in run_root.iterdir() if p.is_dir()] == [first]
    for name in ("manifest.json", "nav.parquet", "fills.parquet", "rejects.parquet", "journal.parquet", "summary.json"):
        assert (first / name).is_file()
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
    assert manifest["trial"]["strategy"] == "equal_weight_liquid"
    trials = (run_root / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(trials) == 1
    assert json.loads(trials[0])["strategy"] == "equal_weight_liquid"


def test_changed_parameter_changes_run_id(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    path20, _, _, _ = _write(tmp_path, 20, run_root)
    path21, _, _, _ = _write(tmp_path, 21, run_root)
    assert path20 != path21


def test_differing_run_with_same_id_fails(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    path, result, summary, parts = _write(tmp_path, 20, run_root)
    other_summary = dataclasses.replace(summary, twr_total=summary.twr_total + 1.0)
    with pytest.raises(ValueError):
        write_run(
            run_root=run_root,
            result=result,
            summary=other_summary,
            inputs=parts["inputs"],
            config=parts["config"],
            strategy=parts["strategy"],
            deposits={},
        )
    assert path.is_dir()


def test_unreadable_existing_run_fails(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    path, result, summary, parts = _write(tmp_path, 20, run_root)
    (path / "manifest.json").unlink()
    with pytest.raises(ValueError):
        write_run(
            run_root=run_root,
            result=result,
            summary=summary,
            inputs=parts["inputs"],
            config=parts["config"],
            strategy=parts["strategy"],
            deposits={},
        )


def test_code_version_falls_back_to_unknown(monkeypatch: Any) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert _code_version() == "unknown"
