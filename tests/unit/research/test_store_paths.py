"""Private strategy store feature sets, meta blocks and production gate edge cases."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import ConfigError
from src.research.panel import FEATURE_NAMES, feature_names_for, known_feature_set_names


def _store(root: Path, sets_text: str | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if sets_text is not None:
        (root / "k-stock-engine").mkdir(exist_ok=True)
        (root / "k-stock-engine" / "feature_sets.toml").write_text(sets_text, encoding="utf-8")
    return root


def test_missing_feature_sets_file_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSE_STRATEGY_STORE", str(_store(tmp_path / "empty", None)))
    with pytest.raises(ValueError, match="missing"):
        [redacted]
    assert known_feature_set_names() == ("full62",)


def test_invalid_feature_sets_toml_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSE_STRATEGY_STORE", str(_store(tmp_path / "bad", "[sets\n")))
    with pytest.raises(ValueError, match="invalid"):
        [redacted]


def test_feature_sets_without_table_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSE_STRATEGY_STORE", str(_store(tmp_path / "notable", "other = 1\n")))
    with pytest.raises(ValueError, match="non-empty"):
        [redacted]


def test_non_list_feature_set_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSE_STRATEGY_STORE", str(_store(tmp_path / "nonlist", '[sets]\nx = "ret_1d"\n')))
    with pytest.raises(ValueError, match="must be a list"):
        feature_names_for("x")


def test_store_edit_is_reloaded_by_mtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _store(tmp_path / "edit", '[sets]\nmine = ["ret_1d", "ret_5"]\n')
    path = root / "k-stock-engine" / "feature_sets.toml"
    monkeypatch.setenv("KSE_STRATEGY_STORE", str(root))
    assert feature_names_for("mine") == ("ret_1d", "ret_5")
    path.write_text('[sets]\nmine = ["ret_10", "ret_21"]\n', encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert feature_names_for("mine") == ("ret_10", "ret_21")


def test_broken_store_still_lists_built_in_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSE_STRATEGY_STORE", str(_store(tmp_path / "broken", "[sets\n")))
    assert known_feature_set_names() == ("full62",)
    assert feature_names_for("full62") == FEATURE_NAMES


def test_split_meta_block_without_separator_is_all_meta() -> None:
    from src.research.strategy_file import split_meta_block

    assert split_meta_block("[meta]\nexample = false") == ("[meta]\nexample = false", "")
    assert split_meta_block("[policy]\n") == ("", "[policy]\n")


def test_production_gate_rejects_invalid_and_long_chains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research.strategy_file import require_production_spec

    monkeypatch.setenv("KSE_STRATEGY_STORE", str(tmp_path))
    broken = tmp_path / "broken.toml"
    broken.write_bytes(b"[[[\n")
    with pytest.raises(ConfigError):
        require_production_spec(broken)
    for index in range(5):
        nxt = tmp_path / f"link{index + 1}.toml"
        (tmp_path / f"link{index}.toml").write_text(f'extends = "link{index + 1}.toml"\n', encoding="utf-8")
        if index == 4:
            nxt.write_text("[policy]\nn = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="exceeds"):
        require_production_spec(tmp_path / "link0.toml")


def test_production_gate_accepts_relative_extends_into_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research.strategy_file import require_production_spec

    store = Path("tests/fixtures/strategy_store/k-stock-engine/kr_swing_2019_v1")
    monkeypatch.setenv("KSE_STRATEGY_STORE", str(store.resolve()))
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = child_dir / "run.toml"
    child.write_text(f'extends = "{os.path.relpath(store.resolve() / "champion.toml", child_dir)}"\n', encoding="utf-8")
    require_production_spec(child)
