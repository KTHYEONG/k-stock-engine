"""Account-engine evaluation CLI envelope invariants."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _write_spec(path: Path, *, n: int = 2, hedge_ratio: float = 1.0, seeds: bool = True) -> Path:
    scorer_seeds = "seeds = [11, 12, 13]\n" if seeds else ""
    path.write_text(
        f'[policy]\nfamily = "ml_trend_cash"\nn = {n}\nrebalance_every_sessions = 5\n'
        "[policy.universe]\nmin_adtv20_krw = 0\nmin_price_krw = 0\n"
        "[scorer]\nfirst_test_year = 2019\nmin_train_rows = 1\nmin_cross_section = 1\n"
        f"num_boost_round = 2\nmin_data_in_leaf = 2\nnum_threads = 1\n{scorer_seeds}"
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
    champion_file = root.parent / "champion.toml"
    monkeypatch.setattr(cli_module, "_champion_file_path", lambda args: champion_file)
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
    assert payload == {"champion": None, "history": [], "champion_file_state": "missing"}


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
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {
        "champion": None, "history": [], "champion_file_state": "missing"
    }


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
        "path",
        "reasons",
        "baseline_hash",
        "delta_mean",
        "delta_lower",
        "tail_mean",
        "tail_lower",
        "noninferiority_margin",
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


def test_policy_override_flag_adopts_a_saved_non_promotable_decision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--policy-override selects the non-promotable decision and hands the rationale to the store."""
    import src.research.cli as cli_module
    from tests.unit.research.test_pipeline import _clean_scores, _context, _sessions

    sessions = _sessions()
    pipeline = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    object.__setattr__(pipeline._ctx, "ledger_runner", _ledger_by_breadth(sessions))
    _patch_pipeline(monkeypatch, pipeline)
    store = _patch_store(monkeypatch, tmp_path / "champion")
    champion_toml = _write_spec(tmp_path / "champion.toml", n=2)
    challenger_toml = _write_spec(tmp_path / "challenger.toml", n=4)
    assert cli_module.main(["promote", "--spec", str(champion_toml), "--bootstrap"]) == 0
    capsys.readouterr()

    assert cli_module.main(["promote", "--spec", str(challenger_toml), "--policy-override", "risk limit"]) == 1
    assert "no saved non-promotable decision" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]

    seen: dict[str, object] = {}
    current = store.current()

    def fake_saved(_store: object, **kwargs: object) -> str:
        seen["promotable"] = kwargs["promotable"]
        return "decision"

    def fake_adopt(**kwargs: object) -> object:
        seen["rationale"] = kwargs["rationale"]
        seen["decision"] = kwargs["decision"]
        return current

    monkeypatch.setattr(cli_module, "_saved_decision", fake_saved)
    monkeypatch.setattr(store, "adopt_policy", fake_adopt)
    assert cli_module.main(["promote", "--spec", str(challenger_toml), "--policy-override", "risk limit"]) == 0
    assert seen == {"promotable": False, "rationale": "risk limit", "decision": "decision"}


def test_challenge_evaluates_the_reseeded_champion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparator runs on with_seeds(stored) while the saved decision keeps the stored identity."""
    from src.research.champion import with_seeds
    from src.research.cli import main
    from src.research.pipeline import load_strategy_spec
    from tests.unit.research.test_pipeline import _clean_scores, _context, _sessions

    sessions = _sessions()
    pipeline = _context(tmp_path, sessions)
    _clean_scores(monkeypatch)
    object.__setattr__(pipeline._ctx, "ledger_runner", _ledger_by_breadth(sessions))
    assert tuple(pipeline._ctx.protocol.champion.champion_seeds) == (11, 12, 13)

    seen: list[str] = []
    inner = pipeline.evaluate

    class _Recording:
        def __init__(self, ctx: object) -> None:
            self._ctx = ctx

        def evaluate(self, spec: Any, *, capital_krw: int | None = None) -> Any:
            seen.append(spec.spec_hash)
            return inner(spec, capital_krw=capital_krw)

    _patch_pipeline(monkeypatch, _Recording(pipeline._ctx))
    store = _patch_store(monkeypatch, tmp_path / "champion")

    champion_toml = _write_spec(tmp_path / "champion.toml", n=2, seeds=False)
    challenger_toml = _write_spec(tmp_path / "challenger.toml", n=4, seeds=True)
    assert main(["promote", "--spec", str(champion_toml), "--bootstrap"]) == 0
    capsys.readouterr()
    seen.clear()

    assert main(["challenge", "--spec", str(challenger_toml)]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    stored = load_strategy_spec(champion_toml)
    challenger = load_strategy_spec(challenger_toml)
    reseeded = with_seeds(stored, (11, 12, 13))
    assert reseeded.spec_hash != stored.spec_hash
    assert seen == [challenger.spec_hash, reseeded.spec_hash]
    assert payload["baseline_hash"] == reseeded.spec_hash
    saved = store.decisions()
    assert len(saved) == 1
    assert saved[0].champion_hash == stored.spec_hash
    assert saved[0].baseline_hash == reseeded.spec_hash


def _bootstrap_record(tmp_path: Path, spec_path: Path) -> Any:
    from datetime import UTC, datetime

    from src.research.champion import ChampionStore
    from src.research.pipeline import load_strategy_spec

    spec = load_strategy_spec(spec_path)
    store = ChampionStore(tmp_path / "champion")
    run = _fake_run_for(spec)
    record = store.bootstrap(
        run=run, spec=spec, spec_path=spec_path, now=datetime(2026, 9, 30, tzinfo=UTC)
    )
    return store, spec, record


def _fake_run_for(spec: Any) -> Any:
    from types import SimpleNamespace

    report = SimpleNamespace(
        spec_hash=spec.spec_hash,
        run_id="r" * 20,
        passed=True,
        objective_j=0.12,
        digest="d" * 64,
        metrics={"g": 0.2, "mdd": -0.1},
    )
    return SimpleNamespace(report=report)


def _sync_helpers(
    monkeypatch: pytest.MonkeyPatch, store: Any, champion_file: Path
) -> None:
    import src.research.cli as cli_module

    monkeypatch.setattr(cli_module, "_champion_store", lambda args: store)
    monkeypatch.setattr(cli_module, "_champion_file_path", lambda args: champion_file)


def test_champion_file_fresh_is_synced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research.cli import _champion_file_state, main

    store, spec, record = _bootstrap_record(tmp_path, _write_spec(tmp_path / "c.toml", n=2))
    champion_file = tmp_path / "champion.toml"
    futures = Path("config/market/futures.toml")
    _sync_helpers(monkeypatch, store, champion_file)
    assert _champion_file_state(record, tmp_path / "absent.toml", futures) == "missing"
    assert main(["champion", "--sync-file"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["state"] == "synced"
    assert payload["champion_file"].endswith("champion.toml")
    assert _champion_file_state(record, champion_file, futures) == "synced"
    assert main(["champion"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["champion_file_state"] == "synced"


def test_champion_file_drift_guards_challenge_and_promote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research.cli import _champion_file_state, main
    from src.research.pipeline import load_strategy_spec

    champion_toml = _write_spec(tmp_path / "c.toml", n=2)
    store, spec, _ = _bootstrap_record(tmp_path, champion_toml)
    champion_file = tmp_path / "champion.toml"
    futures = Path("config/market/futures.toml")
    _sync_helpers(monkeypatch, store, champion_file)
    assert main(["champion", "--sync-file"]) == 0
    capsys.readouterr()
    text = champion_file.read_text(encoding="utf-8")
    champion_file.write_text(text.replace("n = 2", "n = 4"), encoding="utf-8")
    current = store.current()
    assert current is not None
    assert _champion_file_state(current, champion_file, futures) == "drift"
    challenger_toml = _write_spec(tmp_path / "challenger.toml", n=4)
    _patch_pipeline(monkeypatch, _FakePipeline([], tmp_path))
    assert main(["challenge", "--spec", str(challenger_toml)]) == 1
    error = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
    assert current.spec_hash in error
    assert load_strategy_spec(champion_file).spec_hash in error
    assert "champion --sync-file" in error
    assert main(["promote", "--spec", str(challenger_toml)]) == 1
    assert "champion --sync-file" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
    assert main(["evaluate", "--spec", str(challenger_toml)]) == 0


def test_champion_file_missing_then_repaired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research.cli import _champion_file_state, main

    store, _, record = _bootstrap_record(tmp_path, _write_spec(tmp_path / "c.toml", n=2))
    champion_file = tmp_path / "champion.toml"
    futures = Path("config/market/futures.toml")
    _sync_helpers(monkeypatch, store, champion_file)
    assert _champion_file_state(record, champion_file, futures) == "missing"
    assert main(["champion"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["champion_file_state"] == "missing"
    assert main(["champion", "--sync-file"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["state"] == "synced"
    assert _champion_file_state(record, champion_file, futures) == "synced"


@pytest.mark.parametrize("policy_override", [False, True])
@pytest.mark.parametrize("write_fails", [False, True])
def test_promotion_rewrites_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
    policy_override: bool, write_fails: bool,
) -> None:
    import os

    from src.research.cli import _champion_file_state, _write_champion_file, main
    from src.research.pipeline import load_strategy_spec
    from src.research.strategy_file import render_strategy_toml
    from tests.unit.research.test_champion import _promotable, _waivable

    store, challenger, run, decision = _promotable(tmp_path)
    if policy_override:
        decision = _waivable(decision)
    store.save_decision(decision)
    first = store.current()
    assert first is not None
    champion_file = tmp_path / "champion.toml"
    futures = Path("config/market/futures.toml")
    _write_champion_file(first, champion_file, futures)
    before_file = champion_file.read_bytes()
    challenger_file = tmp_path / "challenger.toml"
    challenger_file.write_text(render_strategy_toml(challenger, futures_constants=futures), encoding="utf-8")
    _sync_helpers(monkeypatch, store, champion_file)
    _patch_pipeline(
        monkeypatch, SimpleNamespace(_ctx=SimpleNamespace(now=lambda: first.promoted_at), evaluate=lambda spec: run)
    )
    argv = ["promote", "--spec", str(challenger_file)]
    if policy_override:
        argv.extend(["--policy-override", "risk limit"])
    replace = os.replace

    def fail_file_replace(src: Any, dst: Any) -> None:
        if Path(dst) == champion_file:
            raise OSError("disk full")
        replace(src, dst)

    with monkeypatch.context() as patch:
        if write_fails:
            patch.setattr(os, "replace", fail_file_replace)
        assert main(argv) == int(write_fails)
    capsys.readouterr()
    second = store.current()
    assert second is not None
    assert second.spec_hash == challenger.spec_hash != first.spec_hash
    assert second.reason == ("policy_override" if policy_override else "challenge")
    assert second.spec_json == challenger.canonical_json()
    assert store.history()[-1] == second
    before_store = (tmp_path / "champion" / "current.json").read_bytes()
    if write_fails:
        assert champion_file.read_bytes() == before_file
        assert list(tmp_path.glob(".*.tmp")) == []
        assert _champion_file_state(second, champion_file, futures) == "drift"
        assert main(["challenge", "--spec", str(challenger_file)]) == 1
        assert "champion --sync-file" in json.loads(capsys.readouterr().out)["error"]
        assert main(["champion", "--sync-file"]) == 0
        capsys.readouterr()
    assert (tmp_path / "champion" / "current.json").read_bytes() == before_store
    assert _champion_file_state(second, champion_file, futures) == "synced"
    assert load_strategy_spec(champion_file).spec_hash == second.spec_hash


def test_sync_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from src.research.cli import _write_champion_file

    _, _, record = _bootstrap_record(tmp_path, _write_spec(tmp_path / "c.toml", n=2))
    champion_file = tmp_path / "champion.toml"
    futures = Path("config/market/futures.toml")
    _write_champion_file(record, champion_file, futures)
    before = champion_file.read_text(encoding="utf-8")

    def _boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        _write_champion_file(record, champion_file, futures)
    assert champion_file.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob(".*.tmp")) == []


def test_store_identity_unaffected(tmp_path: Path) -> None:
    import json as _json
    from src.research.cli import _write_champion_file

    store, spec, _ = _bootstrap_record(tmp_path, _write_spec(tmp_path / "c.toml", n=2))
    current = store.current()
    assert current is not None
    before = (tmp_path / "champion" / "current.json").read_bytes()
    _write_champion_file(current, tmp_path / "champion.toml", Path("config/market/futures.toml"))
    assert (tmp_path / "champion" / "current.json").read_bytes() == before
    raw = _json.loads((tmp_path / "champion" / "current.json").read_text(encoding="utf-8"))
    assert set(raw) == {
        "decision_digest",
        "objective_j",
        "promoted_at",
        "reason",
        "report_digest",
        "run_id",
        "spec_hash",
        "spec_json",
        "spec_path",
    }
    assert raw["spec_json"] == spec.canonical_json()


def test_sync_writes_use_distinct_temporary_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from src.research.cli import _write_champion_file
    from src.research.pipeline import load_strategy_spec

    _, _, record = _bootstrap_record(tmp_path, _write_spec(tmp_path / "c.toml", n=2))
    path = tmp_path / "champion.toml"
    futures = Path("config/market/futures.toml")
    replace = os.replace
    temporary_paths: list[Path] = []

    def interleave(src: Any, dst: Any) -> None:
        temporary_paths.append(Path(src))
        if len(temporary_paths) == 1:
            _write_champion_file(record, path, futures)
        replace(src, dst)

    monkeypatch.setattr(os, "replace", interleave)
    _write_champion_file(record, path, futures)
    assert len(set(temporary_paths)) == 2
    assert list(tmp_path.glob(".*.tmp")) == []
    assert load_strategy_spec(path).spec_hash == record.spec_hash


def test_sync_reports_repo_relative_path_outside_repo_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config.runtime import load_runtime_config
    from src.research.cli import _repo_relative_champion

    path = load_runtime_config().champion_file
    monkeypatch.chdir(tmp_path)
    assert _repo_relative_champion(path) == "config/research/champion.toml"


def test_champion_file_state_edge_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research.cli import _champion_file_state, _file_spec_hash, _repo_relative_champion

    _, _, record = _bootstrap_record(tmp_path, _write_spec(tmp_path / "c.toml", n=2))
    futures = Path("config/market/futures.toml")
    champion_dir = tmp_path / "champion_dir"
    champion_dir.mkdir()
    assert _champion_file_state(record, champion_dir, futures) == "drift"
    broken = tmp_path / "broken.toml"
    broken.write_text("[[[", encoding="utf-8")
    assert _champion_file_state(record, broken, futures) == "drift"
    assert _file_spec_hash(broken, futures) == "unreadable"
    assert _repo_relative_champion(Path("/nowhere/x.toml")) == "/nowhere/x.toml"


def test_sync_file_without_champion_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research.cli import main

    _patch_store(monkeypatch, tmp_path / "champion")
    assert main(["champion", "--sync-file"]) == 1
    assert "no champion" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]


def test_champion_paths_resolve_from_runtime_config() -> None:
    from argparse import Namespace

    import src.research.cli as cli_module

    args = Namespace(scope_config=None, data_root=None)
    assert cli_module._champion_file_path(args).name == "champion.toml"
    assert cli_module._champion_futures_constants(args).name == "futures.toml"


def test_missing_file_guard_names_the_champion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research.cli import main

    store, _, _ = _bootstrap_record(tmp_path, _write_spec(tmp_path / "c.toml", n=2))
    _sync_helpers(monkeypatch, store, tmp_path / "champion.toml")
    challenger_toml = _write_spec(tmp_path / "challenger.toml", n=4)
    _patch_pipeline(monkeypatch, _FakePipeline([], tmp_path))
    assert main(["challenge", "--spec", str(challenger_toml)]) == 1
    error = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
    current = store.current()
    assert current is not None
    assert "missing" in error
    assert current.spec_hash in error


def _rebaseline_pipeline(monkeypatch: pytest.MonkeyPatch, *, differ: bool = False) -> Any:
    import src.research.cli as cli_module
    from tests.unit.research.test_champion import _rebaseline_run, _sessions as _build_sessions

    sessions = _build_sessions()
    calls: list[str] = []

    class _Pipeline:
        _ctx = SimpleNamespace(now=lambda: datetime(2026, 10, 9, tzinfo=UTC))

        def evaluate(self, spec: Any, *, capital_krw: int | None = None) -> Any:
            calls.append(spec.spec_hash)
            run = _rebaseline_run(spec, sessions)
            if differ and len(calls) == 2:
                run = replace(run, evidence=replace(
                    run.evidence, base=replace(run.evidence.base, ledger_hash="different-ledger"),
                ))
            return run

    pipeline = _Pipeline()
    monkeypatch.setattr(cli_module, "_pipeline_for", lambda args: pipeline)
    return pipeline


def test_promote_rebaseline_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research.cli import _champion_file_state, main
    from src.research.pipeline import load_strategy_spec

    champion_toml = _write_spec(tmp_path / "c.toml", n=2, seeds=False)
    store, _, _ = _bootstrap_record(tmp_path, champion_toml)
    champion_file = tmp_path / "champion.toml"
    futures = Path("config/market/futures.toml")
    _sync_helpers(monkeypatch, store, champion_file)
    assert main(["champion", "--sync-file"]) == 0
    capsys.readouterr()
    successor = _write_spec(tmp_path / "successor.toml", n=2, seeds=True)
    successor.write_text(
        successor.read_text(encoding="utf-8").replace("num_threads = 1", "num_threads = 4"),
        encoding="utf-8",
    )
    _rebaseline_pipeline(monkeypatch)
    assert main(["promote", "--spec", str(successor), "--rebaseline", "remove seed luck"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["reason"] == "rebaseline"
    assert payload["spec_hash"] == load_strategy_spec(successor).spec_hash
    current = store.current()
    assert current is not None
    assert current.reason == "rebaseline"
    assert "replaces" in (current.rationale or "")
    assert _champion_file_state(current, champion_file, futures) == "synced"
    assert load_strategy_spec(champion_file).spec_hash == current.spec_hash


def test_promote_rebaseline_refusal_leaves_store_and_file_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research.cli import main

    champion_toml = _write_spec(tmp_path / "c.toml", n=2, seeds=False)
    store, _, _ = _bootstrap_record(tmp_path, champion_toml)
    champion_file = tmp_path / "champion.toml"
    _sync_helpers(monkeypatch, store, champion_file)
    assert main(["champion", "--sync-file"]) == 0
    capsys.readouterr()
    before_store = (tmp_path / "champion" / "current.json").read_bytes()
    before_file = champion_file.read_bytes()
    successor = _write_spec(tmp_path / "successor.toml", n=2, seeds=True)
    _rebaseline_pipeline(monkeypatch, differ=True)
    assert main(["promote", "--spec", str(successor), "--rebaseline", "x"]) == 1
    assert "base: ledger_hash mismatch" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
    assert (tmp_path / "champion" / "current.json").read_bytes() == before_store
    assert champion_file.read_bytes() == before_file
    assert len(store.history()) == 1


def test_freeze_models_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.research.cli import main

    calls: list[int | None] = []

    class _FreezePipeline:
        def freeze_models(self, spec: object, *, year: int | None = None) -> Path:
            calls.append(year)
            resolved = int(year) if year is not None else 2021
            target = tmp_path / "models" / str(resolved)
            target.mkdir(parents=True, exist_ok=True)
            return target

    _patch_pipeline(monkeypatch, _FreezePipeline())
    spec = _write_spec(tmp_path / "spec.toml")
    assert main(["freeze-models", "--spec", str(spec), "--year", "2020"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["models_dir"].endswith("2020")
    assert payload["year"] == 2020
    assert payload["boosters"] == 9
    assert main(["freeze-models", "--spec", str(spec)]) == 0
    defaulted = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert defaulted["year"] == 2021
    assert defaulted["boosters"] == 9
    assert calls == [2020, None]


def test_promote_rebaseline_flags_are_exclusive(tmp_path: Path) -> None:
    from src.research.cli import main

    spec = _write_spec(tmp_path / "spec.toml")
    with pytest.raises(SystemExit) as exc:
        main(["promote", "--spec", str(spec), "--bootstrap", "--rebaseline", "x"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc2:
        main(["promote", "--spec", str(spec), "--policy-override", "x", "--rebaseline", "y"])
    assert exc2.value.code == 2


def _parity_fake_pipeline(sessions: list[Any], instruments: list[str], combined: dict[int, Any]) -> Any:
    class _FakeParity:
        def __init__(self) -> None:
            self._ctx = SimpleNamespace(
                protocol=SimpleNamespace(primary_capital_krw=10_000_000),
                cube=SimpleNamespace(sessions=tuple(sessions), instrument_ids=tuple(instruments)),
            )

        def targets_through_last_row(self, spec: object, *, capital_krw: int) -> tuple[object, object]:
            return combined, None

    return _FakeParity()


def test_parity_detects_a_divergence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date

    import numpy as np

    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.operate import SealedDecision, seal

    sessions = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    instruments = ["KRX:000001", "KRX:000002"]
    weights = np.zeros(2, dtype=np.float64)
    weights[0] = 0.5
    combined = {1: np.ascontiguousarray(weights)}
    monkeypatch.setattr(cli_module, "_pipeline_for", lambda args: _parity_fake_pipeline(sessions, instruments, combined))
    decisions = tmp_path / "decisions"
    record = SealedDecision(
        data_session=sessions[1],
        execution_session=sessions[2],
        sealed_at=datetime(2021, 1, 6, 9, 0, tzinfo=UTC),
        spec_hash="s" * 64,
        models_id="m",
        inputs={"cube": "c"},
        targets={"KRX:000001": 0.9},
        reference={"KRX:000001": 5000.0},
        overlay={"regime_fraction": 0.0, "trend_signal": 0.0, "trend_vol_scale": 1.0},
        state_digest="d" * 64,
        prev_hash="",
        hash="",
    )
    seal(decisions, record)
    spec = _write_spec(tmp_path / "spec.toml")
    code = main(["parity", "--spec", str(spec), "--decisions-dir", str(decisions)])
    assert code == 1
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["diverged"] is True
    entry = next(item for item in payload["sessions"] if item["data_session"] == "2021-01-05")
    assert entry["max_abs_weight_diff"] > 1e-9


def test_decide_is_idempotent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date, timedelta

    import numpy as np

    from src.research import cli as cli_module
    from src.research import cube as cube_mod
    from src.research.book import BookSpec, build_sleeve_targets, sleeve_capital_krw
    from src.research.cli import main
    from src.research.hedge import HedgeInputs
    from src.research.model import FrozenScorer, ScorerConfig, ScoreMatrix, score_rows
    from src.research.operate import bootstrap_state, load_state
    from src.research.panel import build_panel, feature_names_for
    from src.research.policy import TrendCashPolicy, UniverseRule, decision_rows, select_row, universe_mask

    base = date(2021, 1, 4)
    sessions = []
    day = base
    while len(sessions) < 20:
        if day.weekday() < 5:
            sessions.append(day)
        day += timedelta(days=1)

    from src.backtest.market import MarketArrays

    shape = (len(sessions), 2)
    rng = np.random.default_rng(9)
    close_i = np.ascontiguousarray(100 + np.cumsum(rng.integers(0, 3, size=shape), axis=0), dtype=np.int64)
    fake = MarketArrays(
        dataset_id="market_panel_0123456789abcdef",
        sessions=tuple(sessions),
        instrument_ids=("KRX:000001", "KRX:000002"),
        int_fields={
            "open": close_i, "high": close_i, "low": close_i, "close": close_i,
            "base_price": np.maximum(close_i - 1, 1), "volume": np.ones(shape, dtype=np.int64),
            "tick_size": np.zeros(shape, dtype=np.int64),
            "upper_limit": np.zeros(shape, dtype=np.int64),
            "lower_limit": np.zeros(shape, dtype=np.int64),
        },
        float_fields={
            "sell_tax_rate": np.zeros(shape), "adtv20": np.full(shape, 1e8),
            "ret_vol60": np.full(shape, 0.01), "share_factor": np.ones(shape),
        },
        bool_fields={
            "present": np.ones(shape, dtype=bool),
            "eligible": np.ones(shape, dtype=bool),
            "open_at_upper": np.zeros(shape, dtype=bool), "open_at_lower": np.zeros(shape, dtype=bool),
            "entry_blocked": np.zeros(shape, dtype=bool),
        },
        market=np.ones(shape, dtype=np.int8),
    )
    monkeypatch.setattr(cube_mod, "load_market_arrays", lambda panel_dir, cache_root: fake)
    monkeypatch.setattr(
        cube_mod, "_panel_float_columns",
        lambda _dir, names, sess, ids: {n: np.full((len(sess), len(ids)), 5.0) for n in names},
    )
    monkeypatch.setattr(cube_mod, "_require_kind", lambda _path, _kind: None)
    import polars as pl

    krx = __import__("zoneinfo", fromlist=["ZoneInfo"]).ZoneInfo("Asia/Seoul")
    from datetime import datetime as _dtm

    def _at(d: Any) -> Any:
        return _dtm(d.year, d.month, d.day, 9, 0, tzinfo=krx)

    facts = pl.DataFrame([
        {"company_id": "KRX:000001", "ticker": "000001", "fiscal_period": "2020Q4", "fact": "equity",
         "value": 100.0, "available_at": _at(sessions[5]), "consolidated": True},
        {"company_id": "KRX:000001", "ticker": "000001", "fiscal_period": "2020Q4", "fact": "sales",
         "value": 50.0, "available_at": _at(sessions[5]), "consolidated": True},
        {"company_id": "KRX:000002", "ticker": "000002", "fiscal_period": "2020Q4", "fact": "equity",
         "value": 200.0, "available_at": _at(sessions[5]), "consolidated": True},
        {"company_id": "KRX:000002", "ticker": "000002", "fiscal_period": "2020Q4", "fact": "sales",
         "value": 60.0, "available_at": _at(sessions[5]), "consolidated": True},
    ])
    dividends = pl.DataFrame([
        {"instrument_id": "KRX:000001", "ex_session": sessions[8], "dps_krw": 2, "available_at": _at(sessions[7])},
    ])
    flows = pl.DataFrame(
        [
            {"instrument_id": "KRX:000001", "session": sessions[6], "foreign_net_shares": 10,
             "institution_net_shares": 5, "individual_net_shares": -3, "available_at": _at(sessions[7])},
        ],
        schema={"instrument_id": pl.String, "session": pl.Date, "foreign_net_shares": pl.Float64,
                "institution_net_shares": pl.Float64, "individual_net_shares": pl.Float64,
                "available_at": pl.Datetime("us", "Asia/Seoul")},
    )
    releases = pl.DataFrame([
        {"instrument_id": "KRX:000001", "fiscal_period": "2020Q4", "metric": "sales",
         "span": "quarter", "value_krw": 50.0, "prior_year_value_krw": 40.0,
         "available_at": _at(sessions[7]), "basis": "consolidated", "release_kind": "preliminary"},
    ])

    def _frame(path: Path, columns: object = None) -> pl.DataFrame:
        name = Path(path).name
        if "financial_facts" in name:
            return facts
        if "dividend" in name:
            return dividends
        if "investor_flow" in name:
            return flows
        return releases

    monkeypatch.setattr(cube_mod, "_read_frame", _frame)
    monkeypatch.setattr(
        cube_mod, "_exit_arrays",
        lambda _panel, _sess, _insts: (np.full(len(_insts), -1, dtype=np.int64), np.zeros(len(_insts), dtype=np.bool_)),
    )
    from src.core.market_rules import load_krx_market_rules

    rules = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))
    inputs = cube_mod.CubeInputs(
        market_panel=tmp_path, dividend_events=tmp_path / "dividend",
        financial_facts=tmp_path / "financial_facts", investor_flow=tmp_path / "investor_flow",
        earnings_releases=tmp_path / "releases", market_rules=rules,
        dividend_withholding_rate=__import__("decimal", fromlist=["Decimal"]).Decimal("0.1"),
    )
    cube = cube_mod.build_research_cube(inputs)
    panel = build_panel(cube, last_row=len(sessions) - 1)
    policy = TrendCashPolicy(
        family="ml_trend_cash", n=1, keep_rank_multiple=3.0, rebalance_every_sessions=2,
        universe=UniverseRule(min_adtv20_krw=0, min_price_krw=0),
        trend_min_dev_ma20=None, trend_min_ret21=None, min_units_per_slot=0,
    )
    book = BookSpec(sleeves=2, stock_capital_fraction=0.75)
    config = ScorerConfig(horizons=(5,), min_cross_section=1, min_train_rows=1)

    class _Stub:
        def model_to_string(self) -> str:
            return "stub"

        def predict(self, data: np.ndarray) -> np.ndarray:
            mat = np.asarray(data, dtype=np.float64)
            if mat.shape[0] == 0 or mat.shape[1] == 0:
                return np.zeros(mat.shape[0], dtype=np.float64)
            return np.ascontiguousarray(mat[:, 0], dtype=np.float64)

    scorer = FrozenScorer(config=config, year=2021, boosters={(int(config.seed), 5): _Stub()})  # type: ignore[dict-item]
    uni = np.ascontiguousarray(np.asarray(universe_mask(cube, policy.universe), dtype=bool))
    mats = [np.asarray(panel.features[name]) for name in feature_names_for("full62")]
    scored = np.asarray(score_rows(mats, uni, list(range(len(sessions))), scorer), dtype=np.float64)
    lo, last = 4, 12
    capital = 10_000_000
    buffer = 0.0
    matrix = ScoreMatrix(scores=np.ascontiguousarray(scored, dtype=np.float32), test_years=(), config_hash="t", last_row=len(sessions) - 1)
    view = SimpleNamespace(arrays={"close": np.asarray(cube.arrays["close"], dtype=np.float64)})
    sleeves = build_sleeve_targets(
        policy, view, panel, matrix, uni, sessions=list(cube.sessions), sleeves=2, lo=lo,
        hi=len(sessions), sleeve_capital_krw=sleeve_capital_krw(capital, book), cash_buffer=buffer,
    )
    slot = float(sleeve_capital_krw(capital, book)) / float(policy.n)
    close = np.asarray(cube.arrays["close"], dtype=np.float64)
    dev = np.asarray(panel.features["dev_ma20"], dtype=np.float64)
    ret = np.asarray(panel.features["ret_21"], dtype=np.float64)
    incumb: list[frozenset[int]] = []
    for phase in range(2):
        carried: frozenset[int] = frozenset()
        for row in decision_rows(list(cube.sessions), lo=lo, hi=len(sessions), every=2, phase=phase):
            _, carried = select_row(
                policy, score_row=scored[row], universe_row=uni[row], close_row=close[row],
                dev_row=dev[row], ret_row=ret[row], incumbents=carried, slot_krw=slot,
            )
        incumb.append(carried)
    from src.research.pipeline import load_strategy_spec

    spec_toml = _write_spec(tmp_path / "spec.toml", n=1)
    spec_toml.write_text(
        '[policy]\nfamily = "ml_trend_cash"\nn = 1\nrebalance_every_sessions = 2\nmin_units_per_slot = 0\n'
        "[policy.universe]\nmin_adtv20_krw = 0\nmin_price_krw = 0\n"
        "[scorer]\nfirst_test_year = 2019\nmin_train_rows = 1\nmin_cross_section = 1\n"
        "num_boost_round = 2\nmin_data_in_leaf = 2\nnum_threads = 1\nhorizons = [5]\n"
        "[book]\nsleeves = 2\nstock_capital_fraction = 0.75\n"
        "[hedge]\nhedge_ratio = 1.0\nbeta_window_sessions = 10\nbeta_min_sessions = 2\nbeta_cap = 2.0\n"
        "rebalance_every_sessions = 5\nuse_futures = true\ncontract_multiplier_krw = 10000\n"
        "initial_margin_rate = 0.2175\nmargin_buffer_rate = 0.10\nmargin_topup_trigger_fraction = 0.75\n"
        "futures_cost_rate = 0.0003\ninverse_cost_rate = 0.0007\nresize_sell_cost_rate = 0.0025\n"
        "resize_buy_cost_rate = 0.0005\nfutures_tax_rate = 0.11\n"
        "futures_annual_deduction_krw = 2500000\ninverse_tax_rate = 0.154\n",
        encoding="utf-8",
    )
    spec = load_strategy_spec(spec_toml)
    state = bootstrap_state(spec, cube, sleeves, incumb, lo=lo, last_row=last)
    state_path = tmp_path / "state.json"
    from src.research.operate import save_state

    save_state(state_path, state)
    hedge = HedgeInputs(
        sessions=tuple(sessions),
        index_level=np.ascontiguousarray(np.full(len(sessions), 400.0)),
        inverse_close=np.ascontiguousarray(np.full(len(sessions), 10000.0)),
    )
    ctx = SimpleNamespace(
        protocol=SimpleNamespace(primary_capital_krw=capital),
        cash_returns=np.ascontiguousarray(np.full(len(sessions), 0.0001)),
        hedge_inputs=hedge,
        trend_inputs=None,
        dataset_ids={"market_panel": "m"},
        now=lambda: datetime(2021, 1, 6, 9, 0, tzinfo=UTC),
    )
    pipeline = SimpleNamespace(_sim_config=lambda cap: SimpleNamespace(cash_buffer=0.0))
    _stub_flow_gate(monkeypatch, [])
    monkeypatch.setattr(
        cli_module, "_decide_context",
        lambda args, tail: SimpleNamespace(
            pipeline=pipeline, context=ctx, cube=cube, inputs=inputs,
            runtime=SimpleNamespace(marker="tiny-runtime"),
        ),
    )
    monkeypatch.setattr(cli_module, "_scorer_for", lambda models_dir, year, expected: scorer)
    models = tmp_path / "models"
    models.mkdir()
    decisions = tmp_path / "decisions"
    argv = ["decide", "--spec", str(spec_toml), "--state", str(state_path), "--models-dir", str(models),
            "--decisions-dir", str(decisions)]
    assert main(argv) == 0
    first_out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert len(first_out["sealed"]) == len(sessions) - 1 - last
    assert load_state(state_path).last_session == sessions[-1]
    assert main(argv) == 0
    second_out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert second_out["sealed"] == []


def test_operate_bootstrap_writes_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date

    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.operate import OperateState, SleeveState, load_state

    state = OperateState(
        spec_hash="s" * 64,
        anchor_session=date(2021, 1, 4),
        last_session=date(2021, 1, 5),
        sessions_since_anchor=1,
        sleeves=(
            SleeveState(phase=0, incumbents=(), weights={}, decided_session=None),
            SleeveState(phase=1, incumbents=(), weights={}, decided_session=None),
        ),
    )

    class _FakeBoot:
        def __init__(self) -> None:
            self._ctx = SimpleNamespace(protocol=SimpleNamespace(primary_capital_krw=10_000_000))

        def targets_through_last_row(self, spec: object, *, capital_krw: int) -> tuple[object, object]:
            assert capital_krw == 10_000_000
            return {}, state

    monkeypatch.setattr(cli_module, "_pipeline_for", lambda args: _FakeBoot())
    spec = _write_spec(tmp_path / "spec.toml")
    out = tmp_path / "state.json"
    assert main(["operate-bootstrap", "--spec", str(spec), "--out", str(out)]) == 0
    assert load_state(out) == state
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["last_session"] == "2021-01-05"
    assert payload["out"] == str(out)


def test_scorer_for_missing_and_prefix_cube(tmp_path: Path) -> None:
    from datetime import date, timedelta

    import numpy as np

    from src.research import cli as cli_module

    with pytest.raises(ValueError, match="no frozen scorer"):
        cli_module._scorer_for(tmp_path, 2021, "abc")
    sessions = [date(2021, 1, 4) + timedelta(days=i) for i in range(5)]
    shape = (5, 2)
    arrays = {
        "close": np.ascontiguousarray(np.full(shape, 100.0)),
        "present": np.ones(shape, dtype=bool),
    }
    from src.research.cube import ResearchCube

    cube = ResearchCube.from_arrays(
        cube_id="c", sessions=sessions, instrument_ids=["KRX:000001", "KRX:000002"],
        arrays=arrays, exit_at=np.array([5, -1], dtype=np.int64), exit_halted=np.zeros(2, dtype=np.bool_),
    )
    prefix = cli_module._prefix_cube(cube, 2)
    assert tuple(prefix.sessions) == tuple(sessions[:3])
    assert list(prefix.exit_at) == [-1, -1]
    assert np.asarray(prefix.arrays["close"]).shape == (3, 2)


def test_parity_clean_and_missing_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date

    import numpy as np

    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.operate import SealedDecision, seal

    sessions = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    instruments = ["KRX:000001", "KRX:000002"]
    weights = np.zeros(2, dtype=np.float64)
    weights[0] = 0.5
    combined = {1: np.ascontiguousarray(weights)}
    monkeypatch.setattr(cli_module, "_pipeline_for", lambda args: _parity_fake_pipeline(sessions, instruments, combined))
    decisions = tmp_path / "decisions"
    seal(
        decisions,
        SealedDecision(
            data_session=sessions[1], execution_session=sessions[2],
            sealed_at=datetime(2021, 1, 6, 9, 0, tzinfo=UTC), spec_hash="s" * 64, models_id="m",
            inputs={"cube": "c"}, targets={"KRX:000001": 0.5}, reference={"KRX:000001": 5000.0},
            overlay={"regime_fraction": 0.0, "trend_signal": 0.0, "trend_vol_scale": 1.0},
            state_digest="d" * 64, prev_hash="", hash="",
        ),
    )
    spec = _write_spec(tmp_path / "spec.toml")
    assert main(["parity", "--spec", str(spec), "--decisions-dir", str(decisions)]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["diverged"] is False
    seal(
        decisions,
        SealedDecision(
            data_session=date(2022, 5, 4), execution_session=date(2022, 5, 6),
            sealed_at=datetime(2021, 1, 6, 9, 0, tzinfo=UTC), spec_hash="s" * 64, models_id="m",
            inputs={"cube": "c"}, targets={"KRX:000001": 0.5}, reference={"KRX:000001": 5000.0},
            overlay={"regime_fraction": 0.0, "trend_signal": 0.0, "trend_vol_scale": 1.0},
            state_digest="d" * 64, prev_hash="", hash="",
        ),
    )
    assert main(["parity", "--spec", str(spec), "--decisions-dir", str(decisions)]) == 1
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["diverged"] is True


def test_peak_rss_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from src.research import cli as cli_module

    monkeypatch.setitem(sys.modules, "resource", None)
    assert cli_module._peak_rss_mb() == 0.0


def _tiny_decide_cube(sessions: list[Any]) -> Any:
    import numpy as np

    from src.research.cube import ResearchCube

    shape = (len(sessions), 2)
    return ResearchCube.from_arrays(
        cube_id="c",
        sessions=sessions,
        instrument_ids=["KRX:000001", "KRX:000002"],
        arrays={
            "close": np.ascontiguousarray(np.full(shape, 5000.0)),
            "present": np.ones(shape, dtype=bool),
            "eligible": np.ones(shape, dtype=bool),
            "entry_blocked": np.zeros(shape, dtype=bool),
            "volume": np.full(shape, 100000.0),
            "adtv20": np.full(shape, 1e9),
            "ret_vol60": np.full(shape, 0.02),
        },
        exit_at=np.full(2, -1, dtype=np.int64),
        exit_halted=np.zeros(2, dtype=np.bool_),
    )


def _tiny_decide_state(tmp_path: Path, spec: Any, cube: Any, last: Any) -> Path:
    from src.research.operate import bootstrap_state, save_state

    state = bootstrap_state(spec, cube, ({}, {}, {}, {}, {}), (frozenset(), frozenset(), frozenset(), frozenset(), frozenset()), lo=1, last_row=0)
    import dataclasses

    state = dataclasses.replace(
        state, last_session=last,
        sessions_since_anchor=cube.sessions.index(last) if last in cube.sessions else 1,
    )
    path = tmp_path / "state.json"
    save_state(path, state)
    return path


def _tiny_decide_ctx(sessions: list[Any], *, cash: float = 0.0001, hedge: float = 400.0) -> Any:
    import numpy as np

    from src.research.hedge import HedgeInputs

    return SimpleNamespace(
        protocol=SimpleNamespace(primary_capital_krw=10_000_000),
        cash_returns=np.ascontiguousarray(np.full(len(sessions), cash)),
        hedge_inputs=HedgeInputs(
            sessions=tuple(sessions),
            index_level=np.ascontiguousarray(np.full(len(sessions), hedge)),
            inverse_close=np.ascontiguousarray(np.full(len(sessions), 10000.0)),
        ),
        trend_inputs=None,
        dataset_ids={"market_panel": "m"},
        now=lambda: datetime(2021, 1, 6, 9, 0, tzinfo=UTC),
    )


def _tiny_decide_resolved(cube: Any, ctx: Any) -> Any:
    pipeline = SimpleNamespace(_sim_config=lambda cap: SimpleNamespace(cash_buffer=0.0))
    return SimpleNamespace(
        pipeline=pipeline,
        context=ctx,
        cube=cube,
        inputs=SimpleNamespace(investor_flow=Path("investor_flow")),
        runtime=SimpleNamespace(marker="tiny-runtime"),
    )


def _stub_flow_gate(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]) -> None:
    import src.data.investor_flow_freshness as freshness_module

    def _check(**kwargs: Any) -> tuple[()]:
        calls.append(kwargs)
        return ()

    monkeypatch.setattr(freshness_module, "check_investor_flow_freshness", _check)


def test_decide_rejects_tail_sessions_below_700(tmp_path: Path) -> None:
    from src.research import cli as cli_module

    args = SimpleNamespace(scope_config=None, data_root=None)
    with pytest.raises(ValueError, match="tail-sessions"):
        cli_module._decide_context(args, 699)


def test_decide_spec_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date

    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.pipeline import load_strategy_spec

    sessions = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    cube = _tiny_decide_cube(sessions)
    spec = load_strategy_spec(_write_spec(tmp_path / "spec.toml"))
    other = load_strategy_spec(_write_spec(tmp_path / "other.toml", n=3))
    ctx = _tiny_decide_ctx(sessions)
    monkeypatch.setattr(cli_module, "_decide_context", lambda args, tail: _tiny_decide_resolved(cube, ctx))
    state_path = _tiny_decide_state(tmp_path, other, cube, sessions[0])
    code = main(["decide", "--spec", str(tmp_path / "spec.toml"), "--state", str(state_path),
                 "--models-dir", str(tmp_path), "--decisions-dir", str(tmp_path / "d")])
    assert code == 1


def test_decide_unknown_session_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date

    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.pipeline import load_strategy_spec

    sessions = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    cube = _tiny_decide_cube(sessions)
    spec = load_strategy_spec(_write_spec(tmp_path / "spec.toml"))
    ctx = _tiny_decide_ctx(sessions)
    monkeypatch.setattr(cli_module, "_decide_context", lambda args, tail: _tiny_decide_resolved(cube, ctx))
    state_path = _tiny_decide_state(tmp_path, spec, cube, date(2022, 12, 31))
    code = main(["decide", "--spec", str(tmp_path / "spec.toml"), "--state", str(state_path),
                 "--models-dir", str(tmp_path), "--decisions-dir", str(tmp_path / "d")])
    assert code == 1


def test_decide_nan_cash_seals_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import date

    import numpy as np

    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.pipeline import load_strategy_spec

    sessions = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    cube = _tiny_decide_cube(sessions)
    spec = load_strategy_spec(_write_spec(tmp_path / "spec.toml"))
    ctx = _tiny_decide_ctx(sessions)
    ctx.cash_returns[1] = np.nan
    monkeypatch.setattr(cli_module, "_decide_context", lambda args, tail: _tiny_decide_resolved(cube, ctx))
    state_path = _tiny_decide_state(tmp_path, spec, cube, sessions[0])
    decisions = tmp_path / "d"
    code = main(["decide", "--spec", str(tmp_path / "spec.toml"), "--state", str(state_path),
                 "--models-dir", str(tmp_path), "--decisions-dir", str(decisions)])
    assert code == 1
    assert not decisions.exists() or not list(decisions.iterdir())


def test_decide_nan_hedge_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import date

    import numpy as np

    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.pipeline import load_strategy_spec

    sessions = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    cube = _tiny_decide_cube(sessions)
    spec = load_strategy_spec(_write_spec(tmp_path / "spec.toml"))
    ctx = _tiny_decide_ctx(sessions)
    ctx.hedge_inputs = SimpleNamespace(
        sessions=tuple(sessions),
        index_level=np.ascontiguousarray(np.full(len(sessions), np.nan)),
        inverse_close=np.ascontiguousarray(np.full(len(sessions), 10000.0)),
    )
    monkeypatch.setattr(cli_module, "_decide_context", lambda args, tail: _tiny_decide_resolved(cube, ctx))
    state_path = _tiny_decide_state(tmp_path, spec, cube, sessions[0])
    code = main(["decide", "--spec", str(tmp_path / "spec.toml"), "--state", str(state_path),
                 "--models-dir", str(tmp_path), "--decisions-dir", str(tmp_path / "d")])
    assert code == 1


def test_decide_missing_trend_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date

    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.pipeline import load_strategy_spec

    sessions = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    cube = _tiny_decide_cube(sessions)
    spec = load_strategy_spec(_write_spec(tmp_path / "spec.toml"))
    trending = SimpleNamespace(
        policy=spec.policy, book=spec.book, scorer=spec.scorer, spec_hash=spec.spec_hash,
        trend_overlay=object(), regime_hedge=None,
    )
    monkeypatch.setattr(cli_module, "_load_spec", lambda path: trending)
    ctx = _tiny_decide_ctx(sessions)
    monkeypatch.setattr(cli_module, "_decide_context", lambda args, tail: _tiny_decide_resolved(cube, ctx))
    state_path = _tiny_decide_state(tmp_path, spec, cube, sessions[0])
    code = main(["decide", "--spec", str(tmp_path / "spec.toml"), "--state", str(state_path),
                 "--models-dir", str(tmp_path), "--decisions-dir", str(tmp_path / "d")])
    assert code == 1


def test_decide_calendar_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date

    import numpy as np

    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.model import FrozenScorer, ScorerConfig
    from src.research.panel import FeaturePanel
    from src.research.pipeline import load_strategy_spec

    sessions = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    cube = _tiny_decide_cube(sessions)
    spec = load_strategy_spec(_write_spec(tmp_path / "spec.toml", n=1))

    class _Stub:
        def model_to_string(self) -> str:
            return "stub"

        def predict(self, data: np.ndarray) -> np.ndarray:
            return np.zeros(np.asarray(data).shape[0], dtype=np.float64)

    config = ScorerConfig(horizons=(5,), min_cross_section=1, min_train_rows=1)
    scorer = FrozenScorer(config=config, year=2021, boosters={(int(config.seed), 5): _Stub()})  # type: ignore[dict-item]
    from src.research.panel import feature_names_for as _names_for

    panel = FeaturePanel(
        features={
            name: np.zeros((3, 2), dtype=np.float32) for name in _names_for("full62")
        },
        labels={},
        last_row=2,
    )
    monkeypatch.setattr(cli_module, "_scorer_for", lambda models_dir, year, expected: scorer)
    import src.research.operate as operate_module

    real_advance = operate_module._advance_with_states

    def _one_day(state: Any, spec: Any, cube: Any, panel: Any, scorer: Any, *, capital_krw: int, cash_buffer: float,
                 series: Any) -> Any:
        final, _, snapshots = real_advance(state, spec, cube, panel, scorer, capital_krw=capital_krw,
                                           cash_buffer=cash_buffer, series=series)
        return final, ((sessions[1], {"KRX:000001": 0.5}),), snapshots[:1]

    monkeypatch.setattr(operate_module, "_advance_with_states", _one_day)
    import src.core.krx_calendar as calendar_module

    original_calendar = calendar_module.xkrx_calendar_through

    def calendar_for(day: Any) -> Any:
        if day == sessions[-1]:
            return original_calendar(day)
        raise ValueError("no calendar")

    monkeypatch.setattr(calendar_module, "xkrx_calendar_through", calendar_for)
    from src.research import panel as panel_module

    monkeypatch.setattr(panel_module, "build_panel", lambda cube, last_row: panel)
    ctx = _tiny_decide_ctx(sessions)
    _stub_flow_gate(monkeypatch, [])
    monkeypatch.setattr(cli_module, "_decide_context", lambda args, tail: _tiny_decide_resolved(cube, ctx))
    state_path = _tiny_decide_state(tmp_path, spec, cube, sessions[0])
    code = main(["decide", "--spec", str(tmp_path / "spec.toml"), "--state", str(state_path),
                 "--models-dir", str(tmp_path), "--decisions-dir", str(tmp_path / "d")])
    assert code == 1


def test_decide_flow_gate_blocks_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from datetime import date

    from src.config.live_gate import load_live_gate_policy
    from src.config.runtime import load_runtime_config
    from src.data.investor_flow_freshness import FlowFreshness, StaleInvestorFlowError
    from src.research import cli as cli_module
    from src.research.cli import main
    from src.research.pipeline import load_strategy_spec
    import src.data.investor_flow_freshness as freshness_module

    sessions = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    cube = _tiny_decide_cube(sessions)
    spec = load_strategy_spec(_write_spec(tmp_path / "spec.toml"))
    ctx = _tiny_decide_ctx(sessions)
    calls: list[dict[str, Any]] = []

    def _stale(**kwargs: Any) -> tuple[()]:
        calls.append(kwargs)
        raise StaleInvestorFlowError(
            (FlowFreshness(sessions[0], 100, 90, 10, 10, 0.10),)
        )

    monkeypatch.setattr(freshness_module, "check_investor_flow_freshness", _stale)
    monkeypatch.setattr(cli_module, "_decide_context", lambda args, tail: _tiny_decide_resolved(cube, ctx))
    state_path = _tiny_decide_state(tmp_path, spec, cube, sessions[0])
    code = main(["decide", "--spec", str(tmp_path / "spec.toml"), "--state", str(state_path),
                 "--models-dir", str(tmp_path), "--decisions-dir", str(tmp_path / "d")])
    assert code == 1
    assert "stale" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
    assert len(calls) == 1
    assert calls[0]["flow_dataset_dir"] == Path("investor_flow")
    assert calls[0]["source_sessions"] == [sessions[0], sessions[1]]
    assert calls[0]["seal_time"] == ctx.now()
    assert calls[0]["runtime"] is not None
    expected = load_live_gate_policy(load_runtime_config().live_gate)
    assert calls[0]["policy"] == expected


def test_decide_multi_year_catch_up(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date, timedelta

    import numpy as np
    import polars as pl

    from src.research import cli as cli_module
    from src.research import cube as cube_mod
    from src.research.book import BookSpec, build_sleeve_targets, sleeve_capital_krw
    from src.research.cli import main
    from src.research.hedge import HedgeInputs
    from src.research.model import FrozenScorer, ScorerConfig, ScoreMatrix, score_rows
    from src.research.operate import bootstrap_state, load_state, save_state
    from src.research.panel import build_panel, feature_names_for
    from src.research.policy import TrendCashPolicy, UniverseRule, decision_rows, select_row, universe_mask

    base = date(2021, 12, 20)
    sessions = []
    day = base
    while len(sessions) < 21:
        if day.weekday() < 5 and day != date(2021, 12, 31):
            sessions.append(day)
        day += timedelta(days=1)
    sessions = sessions[:20]
    assert sessions[0].year == 2021
    assert sessions[-1].year == 2022

    from src.backtest.market import MarketArrays

    shape = (len(sessions), 2)
    rng = np.random.default_rng(9)
    close_i = np.ascontiguousarray(100 + np.cumsum(rng.integers(0, 3, size=shape), axis=0), dtype=np.int64)
    fake = MarketArrays(
        dataset_id="market_panel_0123456789abcdef",
        sessions=tuple(sessions),
        instrument_ids=("KRX:000001", "KRX:000002"),
        int_fields={
            "open": close_i, "high": close_i, "low": close_i, "close": close_i,
            "base_price": np.maximum(close_i - 1, 1), "volume": np.ones(shape, dtype=np.int64),
            "tick_size": np.zeros(shape, dtype=np.int64),
            "upper_limit": np.zeros(shape, dtype=np.int64),
            "lower_limit": np.zeros(shape, dtype=np.int64),
        },
        float_fields={
            "sell_tax_rate": np.zeros(shape), "adtv20": np.full(shape, 1e8),
            "ret_vol60": np.full(shape, 0.01), "share_factor": np.ones(shape),
        },
        bool_fields={
            "present": np.ones(shape, dtype=bool),
            "eligible": np.ones(shape, dtype=bool),
            "open_at_upper": np.zeros(shape, dtype=bool), "open_at_lower": np.zeros(shape, dtype=bool),
            "entry_blocked": np.zeros(shape, dtype=bool),
        },
        market=np.ones(shape, dtype=np.int8),
    )
    monkeypatch.setattr(cube_mod, "load_market_arrays", lambda panel_dir, cache_root: fake)
    monkeypatch.setattr(
        cube_mod, "_panel_float_columns",
        lambda _dir, names, sess, ids: {n: np.full((len(sess), len(ids)), 5.0) for n in names},
    )
    monkeypatch.setattr(cube_mod, "_require_kind", lambda _path, _kind: None)
    krx = __import__("zoneinfo", fromlist=["ZoneInfo"]).ZoneInfo("Asia/Seoul")
    from datetime import datetime as _dtm

    def _at(d: Any) -> Any:
        return _dtm(d.year, d.month, d.day, 9, 0, tzinfo=krx)

    facts = pl.DataFrame([
        {"company_id": "KRX:000001", "ticker": "000001", "fiscal_period": "2021Q3", "fact": "equity",
         "value": 100.0, "available_at": _at(sessions[5]), "consolidated": True},
        {"company_id": "KRX:000001", "ticker": "000001", "fiscal_period": "2021Q3", "fact": "sales",
         "value": 50.0, "available_at": _at(sessions[5]), "consolidated": True},
        {"company_id": "KRX:000002", "ticker": "000002", "fiscal_period": "2021Q3", "fact": "equity",
         "value": 200.0, "available_at": _at(sessions[5]), "consolidated": True},
        {"company_id": "KRX:000002", "ticker": "000002", "fiscal_period": "2021Q3", "fact": "sales",
         "value": 60.0, "available_at": _at(sessions[5]), "consolidated": True},
    ])
    dividends = pl.DataFrame([
        {"instrument_id": "KRX:000001", "ex_session": sessions[8], "dps_krw": 2, "available_at": _at(sessions[7])},
    ])
    flows = pl.DataFrame(
        [
            {"instrument_id": "KRX:000001", "session": sessions[6], "foreign_net_shares": 10,
             "institution_net_shares": 5, "individual_net_shares": -3, "available_at": _at(sessions[7])},
        ],
        schema={"instrument_id": pl.String, "session": pl.Date, "foreign_net_shares": pl.Float64,
                "institution_net_shares": pl.Float64, "individual_net_shares": pl.Float64,
                "available_at": pl.Datetime("us", "Asia/Seoul")},
    )
    releases = pl.DataFrame([
        {"instrument_id": "KRX:000001", "fiscal_period": "2021Q3", "metric": "sales",
         "span": "quarter", "value_krw": 50.0, "prior_year_value_krw": 40.0,
         "available_at": _at(sessions[7]), "basis": "consolidated", "release_kind": "preliminary"},
    ])

    def _frame(path: Path, columns: object = None) -> pl.DataFrame:
        name = Path(path).name
        if "financial_facts" in name:
            return facts
        if "dividend" in name:
            return dividends
        if "investor_flow" in name:
            return flows
        return releases

    monkeypatch.setattr(cube_mod, "_read_frame", _frame)
    monkeypatch.setattr(
        cube_mod, "_exit_arrays",
        lambda _panel, _sess, _insts: (np.full(len(_insts), -1, dtype=np.int64), np.zeros(len(_insts), dtype=np.bool_)),
    )
    from src.core.market_rules import load_krx_market_rules

    rules = load_krx_market_rules(Path("config/market/krx_market_rules.toml"))
    inputs = cube_mod.CubeInputs(
        market_panel=tmp_path, dividend_events=tmp_path / "dividend",
        financial_facts=tmp_path / "financial_facts", investor_flow=tmp_path / "investor_flow",
        earnings_releases=tmp_path / "releases", market_rules=rules,
        dividend_withholding_rate=__import__("decimal", fromlist=["Decimal"]).Decimal("0.1"),
    )
    cube = cube_mod.build_research_cube(inputs)
    panel = build_panel(cube, last_row=len(sessions) - 1)
    policy = TrendCashPolicy(
        family="ml_trend_cash", n=1, keep_rank_multiple=3.0, rebalance_every_sessions=2,
        universe=UniverseRule(min_adtv20_krw=0, min_price_krw=0),
        trend_min_dev_ma20=None, trend_min_ret21=None, min_units_per_slot=0,
    )
    book = BookSpec(sleeves=2, stock_capital_fraction=0.75)
    config = ScorerConfig(horizons=(5,), min_cross_section=1, min_train_rows=1)

    class _Stub:
        def model_to_string(self) -> str:
            return "stub"

        def predict(self, data: np.ndarray) -> np.ndarray:
            mat = np.asarray(data, dtype=np.float64)
            if mat.shape[0] == 0 or mat.shape[1] == 0:
                return np.zeros(mat.shape[0], dtype=np.float64)
            return np.ascontiguousarray(mat[:, 0], dtype=np.float64)

    scorers = {
        year: FrozenScorer(config=config, year=year, boosters={(int(config.seed), 5): _Stub()})  # type: ignore[dict-item]
        for year in (2021, 2022)
    }
    uni = np.ascontiguousarray(np.asarray(universe_mask(cube, policy.universe), dtype=bool))
    mats = [np.asarray(panel.features[name]) for name in feature_names_for("full62")]
    scored = np.asarray(score_rows(mats, uni, list(range(len(sessions))), scorers[2021]), dtype=np.float64)
    lo, last = 4, 7
    capital = 10_000_000
    matrix = ScoreMatrix(scores=np.ascontiguousarray(scored, dtype=np.float32), test_years=(), config_hash="t", last_row=len(sessions) - 1)
    view = SimpleNamespace(arrays={"close": np.asarray(cube.arrays["close"], dtype=np.float64)})
    sleeves = build_sleeve_targets(
        policy, view, panel, matrix, uni, sessions=list(cube.sessions), sleeves=2, lo=lo,
        hi=len(sessions), sleeve_capital_krw=sleeve_capital_krw(capital, book), cash_buffer=0.0,
    )
    slot = float(sleeve_capital_krw(capital, book)) / float(policy.n)
    close = np.asarray(cube.arrays["close"], dtype=np.float64)
    dev = np.asarray(panel.features["dev_ma20"], dtype=np.float64)
    ret = np.asarray(panel.features["ret_21"], dtype=np.float64)
    incumb: list[frozenset[int]] = []
    for phase in range(2):
        carried: frozenset[int] = frozenset()
        for row in decision_rows(list(cube.sessions), lo=lo, hi=len(sessions), every=2, phase=phase):
            _, carried = select_row(
                policy, score_row=scored[row], universe_row=uni[row], close_row=close[row],
                dev_row=dev[row], ret_row=ret[row], incumbents=carried, slot_krw=slot,
            )
        incumb.append(carried)
    spec_toml = tmp_path / "spec.toml"
    spec_toml.write_text(
        '[policy]\nfamily = "ml_trend_cash"\nn = 1\nrebalance_every_sessions = 2\nmin_units_per_slot = 0\n'
        "[policy.universe]\nmin_adtv20_krw = 0\nmin_price_krw = 0\n"
        "[scorer]\nfirst_test_year = 2019\nmin_train_rows = 1\nmin_cross_section = 1\n"
        "num_boost_round = 2\nmin_data_in_leaf = 2\nnum_threads = 1\nhorizons = [5]\n"
        "[book]\nsleeves = 2\nstock_capital_fraction = 0.75\n"
        "[hedge]\nhedge_ratio = 1.0\nbeta_window_sessions = 10\nbeta_min_sessions = 2\nbeta_cap = 2.0\n"
        "rebalance_every_sessions = 5\nuse_futures = true\ncontract_multiplier_krw = 10000\n"
        "initial_margin_rate = 0.2175\nmargin_buffer_rate = 0.10\nmargin_topup_trigger_fraction = 0.75\n"
        "futures_cost_rate = 0.0003\ninverse_cost_rate = 0.0007\nresize_sell_cost_rate = 0.0025\n"
        "resize_buy_cost_rate = 0.0005\nfutures_tax_rate = 0.11\n"
        "futures_annual_deduction_krw = 2500000\ninverse_tax_rate = 0.154\n",
        encoding="utf-8",
    )
    from src.research.pipeline import load_strategy_spec

    spec = load_strategy_spec(spec_toml)
    state_path = tmp_path / "state.json"
    save_state(state_path, bootstrap_state(spec, cube, sleeves, incumb, lo=lo, last_row=last))
    hedge = HedgeInputs(
        sessions=tuple(sessions),
        index_level=np.ascontiguousarray(np.full(len(sessions), 400.0)),
        inverse_close=np.ascontiguousarray(np.full(len(sessions), 10000.0)),
    )
    ctx = SimpleNamespace(
        protocol=SimpleNamespace(primary_capital_krw=capital),
        cash_returns=np.ascontiguousarray(np.full(len(sessions), 0.0001)),
        hedge_inputs=hedge,
        trend_inputs=None,
        dataset_ids={"market_panel": "m"},
        now=lambda: datetime(2021, 1, 6, 9, 0, tzinfo=UTC),
    )
    pipeline = SimpleNamespace(_sim_config=lambda cap: SimpleNamespace(cash_buffer=0.0))
    _stub_flow_gate(monkeypatch, [])
    monkeypatch.setattr(
        cli_module, "_decide_context",
        lambda args, tail: SimpleNamespace(
            pipeline=pipeline, context=ctx, cube=cube, inputs=inputs,
            runtime=SimpleNamespace(marker="tiny-runtime"),
        ),
    )
    monkeypatch.setattr(cli_module, "_scorer_for", lambda models_dir, year, expected: scorers[year])
    models = tmp_path / "models"
    models.mkdir()
    decisions = tmp_path / "decisions"
    prior_state = state_path.read_text(encoding="utf-8")

    def missing_year(models_dir: Path, year: int, expected: str) -> Any:
        if year == 2022:
            raise ValueError("missing 2022 model")
        return scorers[year]

    monkeypatch.setattr(cli_module, "_scorer_for", missing_year)
    assert main(["decide", "--spec", str(spec_toml), "--state", str(state_path), "--models-dir", str(models),
                 "--decisions-dir", str(decisions)]) == 1
    assert not decisions.exists()
    assert state_path.read_text(encoding="utf-8") == prior_state
    capsys.readouterr()
    monkeypatch.setattr(cli_module, "_scorer_for", lambda models_dir, year, expected: scorers[year])
    assert main(["decide", "--spec", str(spec_toml), "--state", str(state_path), "--models-dir", str(models),
                 "--decisions-dir", str(decisions)]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert len(out["sealed"]) == len(sessions) - 1 - last
    chained = {
        json.loads(line)["data_session"]
        for line in (decisions / "chain.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert chained == set(out["sealed"])
    assert load_state(state_path).last_session == sessions[-1]
    chain_before = (decisions / "chain.jsonl").read_bytes()
    state_path.write_text(prior_state, encoding="utf-8")
    assert main(["decide", "--spec", str(spec_toml), "--state", str(state_path), "--models-dir", str(models),
                 "--decisions-dir", str(decisions)]) == 0
    replay = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replay["sealed"] == []
    assert (decisions / "chain.jsonl").read_bytes() == chain_before
    assert load_state(state_path).last_session == sessions[-1]


def test_scorer_for_loads_year_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import cli as cli_module
    import src.research.model as model_module

    year_dir = tmp_path / "2021"
    year_dir.mkdir()
    (year_dir / "manifest.json").write_text("{}", encoding="utf-8")
    seen: list[str] = []
    monkeypatch.setattr(
        model_module, "load_frozen_scorer",
        lambda directory, expected_config_hash: seen.append(str(directory)) or "scorer",
    )
    assert cli_module._scorer_for(tmp_path, 2021, "abc") == "scorer"
    assert seen == [str(year_dir)]


@pytest.mark.parametrize("tail", [0, 699, -1])
def test_decide_validates_tail_before_resolving_data(tail: int) -> None:
    from src.research.cli import _run_decide

    with pytest.raises(ValueError, match="tail-sessions"):
        _run_decide(SimpleNamespace(tail_sessions=tail))


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0])
def test_parity_refuses_invalid_tolerance(tolerance: float, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import cli

    monkeypatch.setattr(cli, "_pipeline_for", lambda args: object())
    monkeypatch.setattr(cli, "_load_spec", lambda path: object())
    with pytest.raises(ValueError, match="tolerance"):
        cli._run_parity(SimpleNamespace(spec=Path("spec.toml"), tolerance=tolerance))
