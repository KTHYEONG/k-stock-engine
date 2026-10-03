"""tools/verify 단위 검증 및 회귀 가드 테스트."""

from __future__ import annotations

import json
from pathlib import Path

from tools import verify


def test_scaffolding_leak_guard_detects_recipe_and_todo(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    dirty_py = src_dir / "dirty.py"
    dirty_py.write_text(
        'def target():\n    """\n    [STEP-BY-STEP RECIPE FOR IMPLEMENTER]:\n    Step 1. Do something\n    """\n    # TODO: fix this later\n    return 42\n',
        encoding="utf-8",
    )
    clean_py = src_dir / "clean.py"
    clean_py.write_text(
        'def clean_func() -> int:\n    """Standard Google-style docstring."""\n    return 42\n',
        encoding="utf-8",
    )

    leaks = verify._check_scaffolding_leaks([str(dirty_py), str(clean_py)])
    assert len(leaks) >= 2
    leak_msgs = [d["error"] for d in leaks]
    assert any("Recipe directive leaked" in m for m in leak_msgs)
    assert any("TODO/FIXME placeholder" in m for m in leak_msgs)

    # Clean file produces 0 leaks
    assert len(verify._check_scaffolding_leaks([str(clean_py)])) == 0


def test_find_test_files_direct_convention(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "src" / "mhs" / "engine"
    src_dir.mkdir(parents=True)
    source_file = src_dir / "matcher.py"
    source_file.write_text("def match(): pass\n", encoding="utf-8")

    tests_dir = tmp_path / "tests" / "unit" / "mhs" / "engine"
    tests_dir.mkdir(parents=True)
    test_file = tests_dir / "test_matcher.py"
    test_file.write_text("def test_match(): pass\n", encoding="utf-8")

    matched, unmapped = verify._find_test_files(["src/mhs/engine/matcher.py"])
    assert "tests/unit/mhs/engine/test_matcher.py" in matched
    assert unmapped == []


def test_available_memory_gb_returns_positive_float() -> None:
    mem = verify._available_memory_gb()
    assert isinstance(mem, float)
    assert mem > 0.0


def test_emit_json_format() -> None:
    pass_json = verify._emit_json("PASS", "all", [], coverage=100)
    data = json.loads(pass_json)
    assert data["status"] == "PASS"
    assert data["exit_code"] == 0
    assert data["coverage"] == 100
    assert data["diagnostics"] == []

    fail_json = verify._emit_json("FAIL", "pytest", [{"error": "fail"}])
    fail_data = json.loads(fail_json)
    assert fail_data["status"] == "FAIL"
    assert fail_data["exit_code"] == 1


def test_find_test_files_nested_package_mapping(tmp_path: Path, monkeypatch) -> None:
    """src/data/cli/build.py maps to tests/unit/data/cli/test_build.py."""
    monkeypatch.chdir(tmp_path)
    nested = tmp_path / "tests" / "unit" / "data" / "cli"
    nested.mkdir(parents=True)
    (nested / "test_build.py").write_text("def test_build(): pass\n", encoding="utf-8")

    matched, unmapped = verify._find_test_files(["src/data/cli/build.py"])
    assert "tests/unit/data/cli/test_build.py" in matched
    assert unmapped == []


def test_find_test_files_unmapped_module_is_reported(tmp_path: Path, monkeypatch) -> None:
    """A module without tests appears in the unmapped diagnostics."""
    monkeypatch.chdir(tmp_path)
    matched, unmapped = verify._find_test_files(["src/data/cli/phantom.py"])
    assert matched == []
    assert unmapped == ["src/data/cli/phantom.py"]
