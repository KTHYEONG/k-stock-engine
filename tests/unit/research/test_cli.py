"""Account-engine evaluation CLI envelope invariants."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _write_spec(path: Path, *, n: int = 2, hedge_ratio: float = 1.0) -> Path:
    path.write_text(
        f'[policy]\nfamily = "ml_trend_cash"\nn = {n}\nrebalance_every_sessions = 5\n'
        "[policy.universe]\nmin_adtv20_krw = 0\nmin_price_krw = 0\n"
        "[scorer]\nfirst_test_year = 2019\nmin_train_rows = 1\nmin_cross_section = 1\n"
        "num_boost_round = 2\nmin_data_in_leaf = 2\nnum_threads = 1\n"
        "[book]\nsleeves = 5\nstock_capital_fraction = 0.75\n"
        f"[hedge]\nhedge_ratio = {hedge_ratio}\nbeta_window_sessions = 10\nbeta_min_sessions = 2\nbeta_cap = 2.0\n"
        "rebalance_every_sessions = 5\nuse_futures = true\ncontract_multiplier_krw = 10000\n"
        "initial_margin_rate = 0.2175\nmargin_buffer_rate = 0.10\nmargin_topup_trigger_fraction = 0.75\n"
        "futures_cost_rate = 0.0003\ninverse_cost_rate = 0.0007\nresize_sell_cost_rate = 0.0025\n"
        "resize_buy_cost_rate = 0.0005\nfutures_tax_rate = 0.11\n"
        "futures_annual_deduction_krw = 2500000\ninverse_tax_rate = 0.154\n",
        encoding="utf-8",
    )
    return path


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, pipeline: object) -> None:
    from src.research import cli as cli_module

    monkeypatch.setattr(cli_module, "_pipeline_for", lambda args: pipeline)


def _fake_run() -> object:
    report = SimpleNamespace(
        spec_hash="h" * 64,
        run_id="r" * 20,
        passed=True,
        objective_j=0.12,
        metrics={"g": 0.2, "mdd": -0.1},
    )
    return SimpleNamespace(report=report)


class _FakePipeline:
    def __init__(self, calls: list[str], reports_root: Path) -> None:
        self._calls = calls
        self._ctx = SimpleNamespace(reports_root=reports_root)

    def evaluate(self, spec: object, *, capital_krw: int | None = None) -> object:
        self._calls.append("evaluate")
        return _fake_run()


def test_evaluate_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.research.cli import main

    calls: list[str] = []
    _patch_pipeline(monkeypatch, _FakePipeline(calls, tmp_path))
    spec = _write_spec(tmp_path / "spec.toml")
    assert main(["evaluate", "--spec", str(spec)]) == 0
    assert calls == ["evaluate"]
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["spec_hash"] == "h" * 64
    assert payload["run_id"] == "r" * 20
    assert payload["passed"] is True
    assert payload["objective_j"] == 0.12
    assert payload["g"] == 0.2
    assert payload["mdd"] == -0.1
    assert "report_path" in payload


def test_cli_error_json_on_pit_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.pit import PITDataError
    from src.research import cli as cli_module
    from src.research.cli import main

    def _boom(args: object) -> object:
        raise PITDataError("bad panel")

    monkeypatch.setattr(cli_module, "_pipeline_for", _boom)
    spec = _write_spec(tmp_path / "spec.toml")
    code = main(["evaluate", "--spec", str(spec)])
    assert code == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines
    assert "error" in json.loads(lines[-1])


def test_window_error_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.research_protocol import WindowError
    from src.research import cli as cli_module
    from src.research.cli import main

    class _OutOfRange:
        def evaluate(self, spec: object, *, capital_krw: int | None = None) -> object:
            raise WindowError("run window is outside certified data")

    monkeypatch.setattr(cli_module, "_pipeline_for", lambda args: _OutOfRange())
    spec = _write_spec(tmp_path / "spec.toml")
    assert main(["evaluate", "--spec", str(spec)]) == 1
    assert "outside" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]


def test_removed_commands_are_gone(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.research.cli import main

    spec = _write_spec(tmp_path / "spec.toml")
    for command in ("backtest", "register", "holdout", "trials"):
        with pytest.raises(SystemExit) as exc:
            main([command, "--spec", str(spec)])
        assert exc.value.code == 2


def test_build_cube_without_registry_fails_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.research.cli import main

    code = main(["build-cube", "--scope-config", "config/research/kr_swing_2019_v1.toml", "--data-root", str(tmp_path)])
    assert code == 1
    assert "error" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_spec_identity(tmp_path: Path) -> None:
    from src.research.pipeline import load_strategy_spec

    first = _write_spec(tmp_path / "a.toml", n=2)
    second = _write_spec(tmp_path / "b.toml", n=3)
    assert load_strategy_spec(first).spec_hash != load_strategy_spec(second).spec_hash
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[policy]\nfamily = "ml_trend_cash"\nrebalance_every_sessions = 5\n[unknown]\nx = 1\n[scorer]\n'
        "[book]\nsleeves = 5\nstock_capital_fraction = 0.75\n[hedge]\nhedge_ratio = 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown"):
        load_strategy_spec(bad)


def _patch_store(monkeypatch: pytest.MonkeyPatch, root: Path) -> Any:
    from src.research import cli as cli_module
    from src.research.champion import ChampionStore

    store = ChampionStore(root)
    monkeypatch.setattr(cli_module, "_champion_store", lambda args: store)
    return store


def _ledger_by_breadth(sessions: list[Any]) -> Any:
    """Fake account engine whose net drift rises with the breadth of the book, so ``n`` is a real knob."""
    from src.research.ledger_bridge import LedgerOutcome

    def _fake(**kwargs: Any) -> LedgerOutcome:
        index = {day: pos for pos, day in enumerate(sessions)}
        window = tuple(sessions[index[kwargs["start"]] : index[kwargs["end"]] + 1])
        breadth = max(
            (int(np.count_nonzero(np.asarray(weights) > 0.0)) for weights in kwargs["targets"].values()),
            default=0,
        )
        drift = 0.0004 + 0.0002 * breadth
        if kwargs.get("extra_slippage"):
            drift -= 0.0002
        if kwargs.get("auction_slippage_ticks"):
            drift -= 0.0001 * float(kwargs["auction_slippage_ticks"])
        return LedgerOutcome(
            capital_krw=kwargs["capital_krw"],
            halted_exit_policy=kwargs["halted_exit_policy"].value,
            sessions=window,
            log_returns=np.full(len(window), drift),
            stock_book_returns=np.full(len(window), drift * 0.9),
            reject_counts={},
            journal_totals_krw={"commission": -1000},
            avg_stock_exposure=0.8,
            avg_margin_share=0.1,
            avg_inverse_share=0.05,
            turnover_per_year=2.0,
            ledger_hash="h",
        )

    return _fake


def test_champion_command_on_empty_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.research.cli import main

    _patch_store(monkeypatch, tmp_path / "champion")
    assert main(["champion"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload == {"champion": None, "history": []}


def test_challenge_without_champion_fails_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.research.cli import main

    _patch_store(monkeypatch, tmp_path / "champion")
    spec = _write_spec(tmp_path / "spec.toml")
    assert main(["challenge", "--spec", str(spec)]) == 1
    assert "no champion" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]


def test_champion_store_resolves_the_state_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.research.cli import main

    assert main(["champion", "--data-root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {"champion": None, "history": []}


def test_champion_lifecycle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bootstrap the incumbent, challenge it with knob neighbors, promote, then read the store back."""
    from src.research.cli import main
    from tests.unit.research.test_pipeline import _clean_scores, _context, _sessions

    sessions = _sessions()
    pipeline = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    object.__setattr__(pipeline._ctx, "ledger_runner", _ledger_by_breadth(sessions))
    _patch_pipeline(monkeypatch, pipeline)
    store = _patch_store(monkeypatch, tmp_path / "champion")
    caplog.set_level(logging.INFO, logger="src.research.cli")

    champion_toml = _write_spec(tmp_path / "champion.toml", n=2)
    challenger_toml = _write_spec(tmp_path / "challenger.toml", n=4)
    lower = _write_spec(tmp_path / "neighbor_low.toml", n=3)
    upper = _write_spec(tmp_path / "neighbor_high.toml", n=5)

    assert main(["promote", "--spec", str(challenger_toml)]) == 1
    assert "pass --bootstrap" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]

    assert main(["promote", "--spec", str(champion_toml), "--bootstrap"]) == 0
    bootstrapped = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert bootstrapped["reason"] == "bootstrap"
    assert bootstrapped["decision_digest"] is None

    assert main(["promote", "--spec", str(challenger_toml)]) == 1
    assert "no saved promotable decision" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]

    argv = ["challenge", "--spec", str(challenger_toml), "--neighbors", str(lower), str(upper)]
    assert main(argv) == 0
    challenged = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert challenged["promotable"] is True
    assert challenged["reasons"] == []
    assert set(challenged) == {
        "promotable",
        "reasons",
        "delta_mean",
        "delta_lower",
        "challenger_j",
        "champion_j",
        "alpha_effective",
        "paired_horizon_sessions",
        "decision_path",
    }
    assert challenged["delta_mean"] > challenged["delta_lower"] > 0.0
    assert challenged["challenger_j"] > challenged["champion_j"]
    decision_path = Path(challenged["decision_path"])
    assert decision_path.is_file()
    assert json.loads(decision_path.read_text(encoding="utf-8"))["knob_changes"] == ["policy.n"]
    assert store.current() == store.history()[-1]

    assert main(["promote", "--spec", str(challenger_toml)]) == 0
    promoted = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert promoted["reason"] == "challenge"
    assert promoted["decision_digest"] == decision_path.stem

    assert main(["champion"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["champion"]["spec_hash"] == promoted["spec_hash"]
    assert payload["champion"]["decision_digest"] == promoted["decision_digest"]
    assert len(payload["history"]) == 2
    assert [row["reason"] for row in payload["history"]] == ["bootstrap", "challenge"]

    portfolio = [record.getMessage() for record in caplog.records if "[PORTFOLIO]" in record.getMessage()]
    assert len(portfolio) == 2
    assert "champion promoted" in portfolio[-1]


def test_promote_refuses_ambiguous_decisions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.research.cli import main
    from tests.unit.research.test_pipeline import _clean_scores, _context, _sessions

    sessions = _sessions()
    pipeline = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    object.__setattr__(pipeline._ctx, "ledger_runner", _ledger_by_breadth(sessions))
    _patch_pipeline(monkeypatch, pipeline)
    _patch_store(monkeypatch, tmp_path / "champion")

    champion_toml = _write_spec(tmp_path / "champion.toml", n=2)
    challenger_toml = _write_spec(tmp_path / "challenger.toml", n=4)
    lower = _write_spec(tmp_path / "neighbor_low.toml", n=3)
    upper = _write_spec(tmp_path / "neighbor_high.toml", n=5)

    assert main(["promote", "--spec", str(champion_toml), "--bootstrap"]) == 0
    capsys.readouterr()
    for neighbors in ([lower], [lower, upper]):
        assert main(["challenge", "--spec", str(challenger_toml), "--neighbors", *map(str, neighbors)]) == 0
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["promotable"] is True

    assert main(["promote", "--spec", str(challenger_toml)]) == 1
    assert "2 saved decisions match" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
