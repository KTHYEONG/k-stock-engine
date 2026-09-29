"""Research CLI grid expansion and error-envelope invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_spec(path: Path) -> Path:
    path.write_text(
        'family = "pead"\nn = 20\nkeep_rank_multiple = 2.0\nrebalance = "M"\n'
        '[universe]\nmin_adtv20_krw = 500000000\nmin_price_krw = 1000\n'
        '[score]\nsue_op = 1.0\nsue_ni = 1.0\n',
        encoding="utf-8",
    )
    return path


def test_grid_expansion_is_cartesian_and_validated(tmp_path: Path) -> None:
    """A 2x3 grid expands to six distinct validated specs."""
    from src.research.cli import expand_grid

    base = _write_spec(tmp_path / "base.toml")
    grid = tmp_path / "grid.toml"
    grid.write_text(
        f'base = "{base}"\n[axes]\nn = [10, 20]\nkeep_rank_multiple = [1.0, 2.0, 3.0]\n',
        encoding="utf-8",
    )
    specs = expand_grid(grid)
    assert len(specs) == 6
    assert len({spec.spec_hash for spec in specs}) == 6
    bad = tmp_path / "bad.toml"
    bad.write_text(f'base = "{base}"\n[axes]\nn = [0]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        expand_grid(bad)


def test_cli_error_json_on_missing_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing spec path exits 1 with an error JSON line."""
    from src.research.cli import main

    code = main(["screen", "--spec", str(tmp_path / "absent.toml")])
    assert code == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines
    assert "error" in json.loads(lines[-1])


def test_holdout_refused_while_sealed(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Holdout with no finalists exits 1 with a lockbox error."""
    from src.research import cli as cli_module
    from src.research.cli import main

    spec = _write_spec(tmp_path / "spec.toml")

    class _SealedLockbox:
        def authorize(self, **kwargs: object) -> object:
            from src.data.research_protocol import LockboxError

            raise LockboxError("holdout is sealed; register a finalist to open it")

    class _Evaluator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._lockbox = _SealedLockbox()

        def holdout(self, spec: object) -> object:
            from src.data.research_protocol import LockboxError

            raise LockboxError("holdout is sealed; register a finalist to open it")

    monkeypatch.setattr(cli_module, "_evaluator_for", lambda args: _Evaluator())
    code = main(["holdout", "--spec", str(spec)])
    assert code == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert "sealed" in json.loads(lines[-1])["error"]


def test_pead_grid_expands_to_160_members() -> None:
    """The pre-registered pead grid has the declared 160 members."""
    from src.research.cli import expand_grid

    specs = expand_grid(Path("config/research/strategies/pead_grid.toml"))
    assert len(specs) == 160


def test_grid_rejects_duplicate_members(tmp_path: Path) -> None:
    """A grid with identical axes values fails closed."""
    from src.research.cli import expand_grid

    base = _write_spec(tmp_path / "base.toml")
    grid = tmp_path / "dup.toml"
    grid.write_text(f'base = "{base}"\n[axes]\nn = [10, 10]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        expand_grid(grid)


def _fake_record() -> object:
    from datetime import datetime, UTC

    from src.data.research_protocol import Segment
    from src.research.registry import TrialRecord

    return TrialRecord(
        trial_id="t" * 20, family="pead", spec_hash="h" * 64, spec_json="{}",
        segment=Segment.DISCOVERY, start=__import__("datetime").date(2020, 1, 6),
        end=__import__("datetime").date(2020, 1, 7), sim_config_json="{}",
        cube_id="cube", created_at=datetime(2026, 9, 29, tzinfo=UTC),
        metrics={"g": 0.01, "active_g_uew": 0.005},
    )


def _fake_report() -> object:
    from src.data.research_protocol import Segment
    from src.research.gates import GateCheck, GateReport

    return GateReport(
        spec_hash="h" * 64, trial_id="t" * 20, segment=Segment.DISCOVERY,
        protocol_version="v", checks=(GateCheck(gate="G1", name="G1.perturbation", value=0.0,
                                                threshold=0.0, passed=True, detail="ok"),),
    )


class _FakeEvaluator:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def screen(self, spec: object) -> object:
        self._calls.append("screen")
        return _fake_record()

    def screen_family(self, specs: object) -> object:
        self._calls.append("family")
        return (_fake_record(),)

    def validate(self, spec: object, family: object) -> object:
        self._calls.append("validate")
        return _fake_report()

    def register_finalists(self, specs: object) -> None:
        self._calls.append("register")

    def holdout(self, spec: object) -> object:
        self._calls.append("holdout")
        return _fake_report()

    def forward(self, spec: object) -> object:
        self._calls.append("forward")
        return _fake_record()


def _patch_evaluator(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    from src.research import cli as cli_module

    monkeypatch.setattr(cli_module, "_evaluator_for", lambda args: _FakeEvaluator(calls))


def test_screen_command_prints_trial(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Screen prints one JSON line with the trial id."""
    from src.research.cli import main

    calls: list[str] = []
    _patch_evaluator(monkeypatch, calls)
    spec = _write_spec(tmp_path / "spec.toml")
    assert main(["screen", "--spec", str(spec)]) == 0
    assert calls == ["screen"]
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["trial_id"] == "t" * 20


def test_family_validate_register_holdout_forward_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Family, validate, register, holdout and forward each emit one JSON line."""
    from src.research.cli import main

    calls: list[str] = []
    _patch_evaluator(monkeypatch, calls)
    spec = _write_spec(tmp_path / "spec.toml")
    grid = tmp_path / "grid.toml"
    grid.write_text(f'base = "{spec}"\n[axes]\nn = [10, 20]\n', encoding="utf-8")
    assert main(["family", "--grid", str(grid)]) == 0
    assert main(["validate", "--spec", str(spec), "--grid", str(grid)]) == 0
    assert main(["register-finalists", "--spec", str(spec)]) == 0
    assert main(["holdout", "--spec", str(spec)]) == 0
    assert main(["forward", "--spec", str(spec)]) == 0
    assert calls == ["family", "validate", "register", "holdout", "forward"]
    assert len(capsys.readouterr().out.strip().splitlines()) == 5


def test_trials_command_counts_registry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Trials reports raw and effective counts from a scratch workspace."""
    import numpy as np

    from src.data.research_protocol import Segment
    from src.research.cli import main
    from src.research.registry import TrialRegistry, TrialReturns

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    code = main(["trials", "--scope-config", str(scope_config), "--data-root", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["raw_count"] == 0
    assert payload["n_for_dsr"] == payload["effective_count"] + payload["prior_trials"]
    from datetime import datetime, UTC

    registry = TrialRegistry(tmp_path / "state" / "kr_swing_2019_v1" / "research" / "trials")
    rng = np.random.default_rng(11)
    sessions = tuple(__import__("datetime").date(2023, 1, 2 + i) for i in range(30))
    for key in ("a", "b"):
        registry.record(
            family="f", spec_hash=f"hash-{key}", spec_json=f'{{"v": "{key}"}}',
            segment=Segment.DISCOVERY, sim_config_json='{"s": 1}', cube_id="cube",
            returns=TrialReturns(sessions=sessions, net=rng.normal(0.001, 0.01, size=30), benchmarks={}),
            metrics={}, now=datetime(2026, 9, 29, tzinfo=UTC),
        )
    code = main(["trials", "--scope-config", str(scope_config), "--data-root", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["raw_count"] == 2


def test_grid_rejects_bad_tables(tmp_path: Path) -> None:
    """Grids without base or axes fail closed."""
    from src.research.cli import expand_grid

    base = _write_spec(tmp_path / "base.toml")
    missing_base = tmp_path / "n griff.toml"
    missing_base.write_text('[axes]\nn = [10]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="base"):
        expand_grid(missing_base)
    empty_axes = tmp_path / "empty.toml"
    empty_axes.write_text(f'base = "{base}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="axes"):
        expand_grid(empty_axes)
    empty_values = tmp_path / "values.toml"
    empty_values.write_text(f'base = "{base}"\n[axes]\nn = []\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        expand_grid(empty_values)


def test_build_cube_without_registry_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Build-cube on an empty workspace exits 1 with an error line."""
    from src.research.cli import main

    code = main(["build-cube", "--scope-config", "config/research/kr_swing_2019_v1.toml",
                 "--data-root", str(tmp_path)])
    assert code == 1
    assert "error" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_grid_with_unknown_nested_axis_fails(tmp_path: Path) -> None:
    """An axis under an unknown table still reaches validation and fails."""
    from src.research.cli import expand_grid

    base = _write_spec(tmp_path / "base.toml")
    grid = tmp_path / "nested.toml"
    grid.write_text(f'base = "{base}"\n[axes]\n"nope.sub" = [1]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"nope|unknown|extra|forbidden"):
        expand_grid(grid)


def test_trials_skips_corrupt_and_degenerate_members(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Corrupt parquet is skipped and flat members fall back to raw counts."""
    import numpy as np

    from src.research.cli import main
    from src.research.registry import TrialRegistry, TrialReturns

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    from datetime import datetime, UTC

    from src.data.research_protocol import Segment

    registry = TrialRegistry(tmp_path / "state" / "kr_swing_2019_v1" / "research" / "trials")
    sessions = tuple(__import__("datetime").date(2023, 1, 2 + i) for i in range(30))
    flat_net = np.zeros(30)
    for key in ("c1", "c2"):
        registry.record(
            family="f", spec_hash=f"flat-{key}", spec_json=f'{{"v": "{key}"}}',
            segment=Segment.DISCOVERY, sim_config_json='{"s": 1}', cube_id="cube",
            returns=TrialReturns(sessions=sessions, net=flat_net, benchmarks={}),
            metrics={}, now=datetime(2026, 9, 29, tzinfo=UTC),
        )
    gone = registry.record(
        family="f", spec_hash="gone", spec_json='{"v": "gone"}', segment=Segment.DISCOVERY,
        sim_config_json='{"s": 1}', cube_id="cube",
        returns=TrialReturns(sessions=sessions, net=flat_net, benchmarks={}),
        metrics={}, now=datetime(2026, 9, 29, tzinfo=UTC),
    )
    (tmp_path / "state" / "kr_swing_2019_v1" / "research" / "trials" / "returns" / f"{gone.trial_id}.parquet").unlink()
    code = main(["trials", "--scope-config", str(scope_config), "--data-root", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["raw_count"] == 3
    assert payload["effective_count"] == 2.0
