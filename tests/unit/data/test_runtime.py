from __future__ import annotations

from pathlib import Path

CANONICAL = Path("config/research/kr_swing_2019_v1.toml")


def test_runtime_binds_one_scope_once(tmp_path: Path) -> None:
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=CANONICAL, data_root=tmp_path / "data")
    assert runtime.scope.scope_id == "kr_swing_2019_v1"
    assert runtime.workspace.scope.scope_id == runtime.scope.scope_id
    for root in (
        runtime.workspace.bronze_root,
        runtime.workspace.silver_root,
        runtime.workspace.gold_root,
        runtime.workspace.state_root,
        runtime.workspace.runs_root,
    ):
        assert root.name == "kr_swing_2019_v1"
        assert root.is_absolute()


def test_runtime_construction_does_not_mutate_storage(tmp_path: Path) -> None:
    from src.data.runtime import load_data_runtime

    data_root = tmp_path / "fresh"
    assert not data_root.exists()
    runtime = load_data_runtime(scope_config=CANONICAL, data_root=data_root)
    assert not data_root.exists()
    assert not runtime.workspace.bronze_root.exists()
    runtime.workspace.initialize()
    assert runtime.workspace.bronze_root.is_dir()


def test_scoped_commands_bind_runtime_without_layer_roots(tmp_path: Path) -> None:
    from src.data.cli import _parse_args, main

    args = _parse_args(["scope-info", "--scope-config", str(CANONICAL), "--data-root", str(tmp_path / "data")])
    assert args.scope_config == CANONICAL
    assert str(args.data_root) == str(tmp_path / "data")
    assert not hasattr(args, "bronze_root")
    assert not hasattr(args, "coverage_start")
    assert main(["scope-info", "--scope-config", str(CANONICAL), "--data-root", str(tmp_path / "data")]) == 0


def test_init_workspace_creates_scope_directories(tmp_path: Path) -> None:
    from src.data.cli import main

    data_root = tmp_path / "data"
    assert main(["init-workspace", "--scope-config", str(CANONICAL), "--data-root", str(data_root)]) == 0
    assert (data_root / "bronze" / "kr_swing_2019_v1").is_dir()
    assert (data_root / "silver" / "kr_swing_2019_v1").is_dir()
    assert (data_root / "gold" / "kr_swing_2019_v1").is_dir()
    assert (data_root / "state" / "kr_swing_2019_v1").is_dir()
    assert (data_root / "runs" / "kr_swing_2019_v1").is_dir()
