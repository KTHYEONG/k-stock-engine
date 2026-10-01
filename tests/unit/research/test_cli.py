"""ML trend-cash CLI envelope invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_spec(path: Path, **overrides: object) -> Path:
    policy_n = overrides.get("n", 2)
    path.write_text(
        f'[policy]\nfamily = "ml_trend_cash"\nn = {policy_n}\nrebalance_every_sessions = 5\n'
        "[policy.universe]\nmin_adtv20_krw = 0\nmin_price_krw = 0\n"
        "[scorer]\nfirst_test_year = 2019\nmin_train_rows = 1\nmin_cross_section = 1\n"
        "num_boost_round = 2\nmin_data_in_leaf = 2\nnum_threads = 1\n"
        "[book]\nsleeves = 5\nstock_capital_fraction = 0.75\n"
        "[hedge]\nhedge_ratio = 1.0\nbeta_window_sessions = 10\nbeta_min_sessions = 2\nbeta_cap = 2.0\n"
        "rebalance_every_sessions = 5\nuse_futures = true\ncontract_multiplier_krw = 10000\n"
        "initial_margin_rate = 0.2175\nmargin_buffer_rate = 0.10\nmargin_topup_trigger_fraction = 0.75\n"
        "futures_cost_rate = 0.0003\ninverse_cost_rate = 0.0007\nresize_sell_cost_rate = 0.0025\n"
        "resize_buy_cost_rate = 0.0005\nfutures_tax_rate = 0.11\nfutures_annual_deduction_krw = 2500000\n"
        "inverse_tax_rate = 0.154\n",
        encoding="utf-8",
    )
    return path


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, pipeline: object) -> None:
    from src.research import cli as cli_module

    monkeypatch.setattr(cli_module, "_pipeline_for", lambda args: pipeline)


class _FakeReport:
    def __init__(self) -> None:
        self.spec_hash = "h" * 64
        self.passed = True
        self.digest = "d" * 64


class _FakePipeline:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        from src.research.registry import TrialRegistry  # noqa: F401

    def evaluate_discovery(self, spec: object) -> object:
        self._calls.append("backtest")
        return _FakeReport()

    def register_finalist(self, spec: object) -> None:
        self._calls.append("register")

    def holdout(self, spec: object) -> object:
        self._calls.append("holdout")
        return _FakeReport()


def test_backtest_register_holdout_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.research.cli import main

    calls: list[str] = []
    _patch_pipeline(monkeypatch, _FakePipeline(calls))
    spec = _write_spec(tmp_path / "spec.toml")

    class _Ctx:
        registry = None

    assert main(["backtest", "--spec", str(spec)]) == 0
    assert main(["register", "--spec", str(spec)]) == 0
    assert main(["holdout", "--spec", str(spec)]) == 0
    assert calls == ["backtest", "register", "holdout"]
    assert len(capsys.readouterr().out.strip().splitlines()) == 3


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
    code = main(["backtest", "--spec", str(spec)])
    assert code == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines
    assert "error" in json.loads(lines[-1])


def test_holdout_refused_while_sealed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.research_protocol import LockboxError
    from src.research import cli as cli_module
    from src.research.cli import main

    class _Sealed:
        def evaluate_discovery(self, spec: object) -> object:
            raise LockboxError("holdout is sealed")

        def register_finalist(self, spec: object) -> None:
            raise LockboxError("holdout is sealed")

        def holdout(self, spec: object) -> object:
            raise LockboxError("holdout is sealed")

    monkeypatch.setattr(cli_module, "_pipeline_for", lambda args: _Sealed())
    spec = _write_spec(tmp_path / "spec.toml")
    assert main(["holdout", "--spec", str(spec)]) == 1
    assert "sealed" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]


def test_trials_prints_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from src.research.cli import main

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    assert main(["trials", "--scope-config", str(scope_config), "--data-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["raw_count"] == 0
    assert "effective_count" in payload
    assert "prior_trials" in payload
    assert "prior_effective_trials" in payload
    assert payload["n_for_dsr"] == pytest.approx(payload["effective_count"] + payload["prior_effective_trials"])


def test_trials_with_mixed_registry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import UTC, datetime

    import numpy as np

    from src.data.research_protocol import Segment
    from src.research.cli import main
    from src.research.registry import TrialRegistry, TrialReturns

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    registry = TrialRegistry(tmp_path / "state" / "kr_swing_2019_v1" / "research" / "trials_ml")
    sessions = tuple(__import__("datetime").date(2023, 1, 2 + i) for i in range(30))
    rng = np.random.default_rng(11)
    for key in ("a", "b"):
        registry.record(
            family="f", spec_hash=f"hash-{key}", spec_json=f'{{"v": "{key}"}}',
            segment=Segment.DISCOVERY, sim_config_json='{"s": 1}', cube_id="cube",
            returns=TrialReturns(sessions=sessions, net=rng.normal(0.001, 0.01, size=30), benchmarks={}),
            metrics={}, now=datetime(2026, 9, 30, tzinfo=UTC),
        )
    flat = registry.record(
        family="f", spec_hash="flat", spec_json='{"v": "flat"}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 2}', cube_id="cube",
        returns=TrialReturns(sessions=sessions, net=np.zeros(30), benchmarks={}),
        metrics={}, now=datetime(2026, 9, 30, tzinfo=UTC),
    )
    (tmp_path / "state" / "kr_swing_2019_v1" / "research" / "trials_ml" / "returns" / f"{flat.trial_id}.parquet").unlink()
    assert main(["trials", "--scope-config", str(scope_config), "--data-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["raw_count"] == 3
    assert payload["effective_count"] >= 1.0


def test_trials_single_flat_trial(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import UTC, datetime

    import numpy as np

    from src.data.research_protocol import Segment
    from src.research.cli import main
    from src.research.registry import TrialRegistry, TrialReturns

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    registry = TrialRegistry(tmp_path / "state" / "kr_swing_2019_v1" / "research" / "trials_ml")
    sessions = tuple(__import__("datetime").date(2023, 1, 2 + i) for i in range(30))
    registry.record(
        family="f", spec_hash="only", spec_json='{"v": "only"}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 1}', cube_id="cube",
        returns=TrialReturns(sessions=sessions, net=np.zeros(30), benchmarks={}),
        metrics={}, now=datetime(2026, 9, 30, tzinfo=UTC),
    )
    assert main(["trials", "--scope-config", str(scope_config), "--data-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["raw_count"] == 1
    assert payload["effective_count"] == pytest.approx(1.0)


def test_trials_multi_flat_falls_back_to_count(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import UTC, datetime

    import numpy as np

    from src.data.research_protocol import Segment
    from src.research.cli import main
    from src.research.registry import TrialRegistry, TrialReturns

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    registry = TrialRegistry(tmp_path / "state" / "kr_swing_2019_v1" / "research" / "trials_ml")
    sessions = tuple(__import__("datetime").date(2023, 1, 2 + i) for i in range(30))
    for key in ("f1", "f2"):
        registry.record(
            family="f", spec_hash=key, spec_json=f'{{"v": "{key}"}}', segment=Segment.DISCOVERY,
            sim_config_json='{"s": 1}', cube_id="cube",
            returns=TrialReturns(sessions=sessions, net=np.zeros(30), benchmarks={}),
            metrics={}, now=datetime(2026, 9, 30, tzinfo=UTC),
        )
    assert main(["trials", "--scope-config", str(scope_config), "--data-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["raw_count"] == 2
    assert payload["effective_count"] == pytest.approx(2.0)


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
