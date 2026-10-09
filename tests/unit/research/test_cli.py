"""Account-engine evaluation CLI envelope invariants."""

from __future__ import annotations

import json
import logging
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
