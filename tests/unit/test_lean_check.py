"""lean_check 단위 검증 및 회귀 가드 테스트."""

from __future__ import annotations

import json
from pathlib import Path

from tools.agent_skills import lean_check


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

    leaks = lean_check._check_scaffolding_leaks([str(dirty_py), str(clean_py)])
    assert len(leaks) >= 2
    leak_msgs = [d["error"] for d in leaks]
    assert any("Recipe directive leaked" in m for m in leak_msgs)
    assert any("TODO/FIXME placeholder" in m for m in leak_msgs)

    # Clean file produces 0 leaks
    assert len(lean_check._check_scaffolding_leaks([str(clean_py)])) == 0


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

    matched = lean_check._find_test_files(["src/mhs/engine/matcher.py"])
    assert "tests/unit/mhs/engine/test_matcher.py" in matched


def test_find_test_files_spec_markdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tests_dir = tmp_path / "tests" / "unit"
    tests_dir.mkdir(parents=True)
    test_file = tests_dir / "test_spec_feature.py"
    test_file.write_text("def test_it(): pass\n", encoding="utf-8")

    spec_file = tmp_path / "feature_spec.md"
    spec_file.write_text(
        "# Feature Spec\n\n## Test Suite: tests/unit/test_spec_feature.py\n",
        encoding="utf-8",
    )

    matched = lean_check._find_test_files([], spec_path=str(spec_file))
    assert "tests/unit/test_spec_feature.py" in matched


def test_available_memory_gb_returns_positive_float() -> None:
    mem = lean_check._available_memory_gb()
    assert isinstance(mem, float)
    assert mem > 0.0


def test_emit_json_format() -> None:
    pass_json = lean_check._emit_json("PASS", "all", [], coverage=100)
    data = json.loads(pass_json)
    assert data["status"] == "PASS"
    assert data["exit_code"] == 0
    assert data["coverage"] == 100
    assert data["diagnostics"] == []

    fail_json = lean_check._emit_json("FAIL", "pytest", [{"error": "fail"}])
    fail_data = json.loads(fail_json)
    assert fail_data["status"] == "FAIL"
    assert fail_data["exit_code"] == 1


def test_find_test_files_spec_markdown_invariant_scenarios(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tests_dir = tmp_path / "tests" / "unit"
    tests_dir.mkdir(parents=True)
    test_file = tests_dir / "test_invariant_feature.py"
    test_file.write_text("def test_invariant(): pass\n", encoding="utf-8")

    spec_file = tmp_path / "feature_spec.md"
    spec_file.write_text(
        "# Feature Spec\n\n## Invariant Scenarios: tests/unit/test_invariant_feature.py\n",
        encoding="utf-8",
    )

    matched = lean_check._find_test_files([], spec_path=str(spec_file))
    assert "tests/unit/test_invariant_feature.py" in matched


def test_check_pre_impl_spec_valid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    caller_file = src_dir / "caller.py"
    caller_file.write_text("def run():\n    anchor_symbol()\n", encoding="utf-8")

    tests_dir = tmp_path / "tests" / "unit"
    tests_dir.mkdir(parents=True)

    spec_file = tmp_path / "valid_spec.md"
    spec_file.write_text(
        "## Target: src/pkg/target.py\n\n"
        "## Wiring: src/pkg/caller.py\n"
        "- Anchor: anchor_symbol\n\n"
        "## Invariant Scenarios: tests/unit/test_target.py\n",
        encoding="utf-8",
    )

    code, diags = lean_check._check_pre_impl_spec(str(spec_file))
    assert code == 0
    assert diags == []


def test_check_pre_impl_spec_invalid_anchor_and_missing_caller(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    caller_file = src_dir / "caller.py"
    caller_file.write_text("def run():\n    pass\n", encoding="utf-8")

    spec_file = tmp_path / "invalid_spec.md"
    spec_file.write_text(
        "## Target: non_existent_dir/sub/target.py\n\n"
        "## Wiring: src/pkg/caller.py\n"
        "- Anchor: missing_anchor\n\n"
        "## Invariant Scenarios: bad_dir/test_target.py\n",
        encoding="utf-8",
    )

    code, diags = lean_check._check_pre_impl_spec(str(spec_file))
    assert code == 1
    assert len(diags) == 3
    errors = [d["error"] for d in diags]
    assert any("Target parent directory does not exist" in e for e in errors)
    assert any("Wiring anchor 'missing_anchor' not found" in e for e in errors)
    assert any("Test suite directory does not exist" in e for e in errors)

