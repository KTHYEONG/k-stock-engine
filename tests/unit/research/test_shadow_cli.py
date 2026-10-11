"""Research shadow-run wiring: tail context, both variants and the JSON envelope."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.core.time import KRX_TZ

INSTRS = ["KRX:000001", "KRX:000002"]
SPEC_FIXTURE = Path("tests/fixtures/strategies/ml_sleeve_hedge.toml")


def _pipe(tmp_path: Path, sessions: list[date]) -> Any:
    from src.core.market_rules import load_krx_market_rules
    from src.data.research_protocol import WindowGuard, load_research_protocol
    from src.data.research_scope import load_research_scope
    from src.research.hedge import HedgeInputs
    from src.research.ledger_bridge import run_ledger
    from src.research.pipeline import Pipeline, PipelineContext
    from src.research.registry import RunRegistry
    from tests.fixtures.synthetic_panel import synthetic_cube, synthetic_panel_row, write_synthetic_panel

    protocol = load_research_protocol(
        Path("config/research/protocol.toml"), load_research_scope(Path("config/research/kr_swing_2019_v1.toml"))
    )
    cube = synthetic_cube(sessions, INSTRS)
    panel = write_synthetic_panel(
        tmp_path / "gold", "market_panel_test",
        [synthetic_panel_row(day, inst) for day in sessions for inst in INSTRS],
    )
    ctx = PipelineContext(
        protocol=protocol, cube=cube, registry=RunRegistry(tmp_path / "runs"),
        guard=WindowGuard(protocol=protocol, last_session=sessions[-1]), panel_dir=panel,
        dividends=pl.DataFrame(),
        hedge_inputs=HedgeInputs(sessions=tuple(sessions), index_level=np.full(len(sessions), 1500.0), inverse_close=np.full(len(sessions), 8000.0)),
        rules=load_krx_market_rules(Path("config/market/krx_market_rules.toml")),
        engine_config_path=Path("config/backtest/default_engine.toml"), market_cache_root=tmp_path / "mcache",
        reports_root=tmp_path / "reports", scores_root=tmp_path / "scores", ledger_runner=run_ledger,
        now=lambda: datetime(2026, 9, 30, tzinfo=KRX_TZ),
        dataset_ids={"market_panel": "a", "dividend_events": "b", "hedge_series": "c", "cash_series": "d"},
        cash_returns=np.zeros(len(sessions), dtype=np.float64),
    )
    return Pipeline(ctx)


def _chain(sessions: list[date], rows: list[int]) -> tuple[Any, ...]:
    from src.research.operate import SealedDecision

    out = []
    for row in rows:
        data, execution = sessions[row], sessions[row + 1]
        out.append(SealedDecision(
            data_session=data, execution_session=execution,
            sealed_at=datetime(execution.year, execution.month, execution.day, 8, 0, tzinfo=KRX_TZ),
            spec_hash="s", models_id="m", inputs={}, targets={INSTRS[row % 2]: 0.5},
            reference={}, overlay={}, state_digest="d", prev_hash="", hash="",
        ))
    return tuple(out)


def test_shadow_run_appends_both_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import src.research.cli as cli

    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(300, start=date(2019, 1, 5))
    pipe = _pipe(tmp_path, sessions)
    chain = _chain(sessions, [260, 261])
    monkeypatch.setattr(cli, "_decide_context", lambda args, tail: SimpleNamespace(pipeline=pipe, context=pipe._ctx, cube=pipe._ctx.cube, inputs=None))
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    code = cli.main(["shadow-run", "--spec", str(SPEC_FIXTURE), "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "shadow")])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["appended"] == {"official": 39, "ideal": 39}
    assert payload["late_sessions"] == []
    assert payload["last_session"] == sessions[-1].isoformat()
    assert payload["nav_krw"] > 0
    assert (tmp_path / "shadow" / "official" / "nav.parquet").is_file()
    assert (tmp_path / "shadow" / "ideal" / "nav.parquet").is_file()


def test_shadow_run_rejects_small_tail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import src.research.cli as cli

    code = cli.main(["shadow-run", "--spec", str(SPEC_FIXTURE), "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "shadow"), "--tail-sessions", "100"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "tail-sessions" in payload["error"]


def _patched_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pipe: Any, chain: tuple[Any, ...]) -> Any:
    import src.research.cli as cli

    monkeypatch.setattr(cli, "_decide_context", lambda args, tail: SimpleNamespace(pipeline=pipe, context=pipe._ctx, cube=pipe._ctx.cube, inputs=None))
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    return cli


def test_shadow_run_empty_chain_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json as _json

    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(300, start=date(2019, 1, 5))
    pipe = _pipe(tmp_path, sessions)
    cli = _patched_main(monkeypatch, tmp_path, pipe, ())
    code = cli.main(["shadow-run", "--spec", str(SPEC_FIXTURE), "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "shadow")])
    assert code == 1
    assert "no sealed decisions" in _json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]


def test_shadow_run_start_outside_cube_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json as _json
    from datetime import timedelta as _td

    from src.research.operate import SealedDecision
    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(300, start=date(2019, 1, 5))
    pipe = _pipe(tmp_path, sessions)
    far = SealedDecision(
        data_session=sessions[-1], execution_session=sessions[-1] + _td(days=30),
        sealed_at=datetime(2019, 12, 1, tzinfo=KRX_TZ), spec_hash="s", models_id="m", inputs={},
        targets={}, reference={}, overlay={}, state_digest="d", prev_hash="", hash="",
    )
    cli = _patched_main(monkeypatch, tmp_path, pipe, (far,))
    code = cli.main(["shadow-run", "--spec", str(SPEC_FIXTURE), "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "shadow")])
    assert code == 1
    assert "not in the tail cube" in _json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]


def test_shadow_run_short_history_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json as _json

    from tests.fixtures.synthetic_panel import synthetic_sessions

    sessions = synthetic_sessions(300, start=date(2019, 1, 5))
    pipe = _pipe(tmp_path, sessions)
    cli = _patched_main(monkeypatch, tmp_path, pipe, _chain(sessions, [5, 6]))
    code = cli.main(["shadow-run", "--spec", str(SPEC_FIXTURE), "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "shadow")])
    assert code == 1
    assert "253 sessions" in _json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
