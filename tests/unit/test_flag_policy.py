"""Config flag policy tripwire.

Discovers toggles from bool fields of config models and from literal env reads in src/, then checks them
against docs/flags.toml. There is no deploy baseline file in this repo, so the deploy-baseline check of the
shared contract is skipped by design; it is not a missing check.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "docs" / "flags.toml"
CONFIG_ROOT = REPO_ROOT / "src" / "config"
SOURCE_ROOT = REPO_ROOT / "src"
BUDGET_TOOL = REPO_ROOT / "tools" / "instruction_budget.py"
RULE_FILE = ".agents/rules/config-flags.md"
TOGGLE_NAME_RE = re.compile(r"(?i)(enable|disable|resend|_on_|allow|auto_adopt|force_)")
KINDS = frozenset({"env_binding", "safety_interlock", "component", "experiment"})
REVIEW_GRACE_DAYS = 14


def _config_model_classes() -> list[type[BaseModel]]:
    classes: list[type[BaseModel]] = []
    for path in sorted(CONFIG_ROOT.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = importlib.import_module(f"src.config.{path.stem}")
        classes += [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, BaseModel) and cls.__module__ == module.__name__
        ]
    return classes


def discover_model_toggles(classes: Iterable[type[BaseModel]] | None = None) -> dict[str, bool]:
    """Map 'Model.field' to its code default for bool fields whose name matches TOGGLE_NAME_RE."""
    toggles: dict[str, bool] = {}
    for cls in _config_model_classes() if classes is None else classes:
        for field_name, field in cls.model_fields.items():
            if field.annotation is bool and TOGGLE_NAME_RE.search(field_name):
                toggles[f"{cls.__name__}.{field_name}"] = bool(field.default)
    return toggles


def _is_environ(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ" and _is_name(node.value, "os")


def _is_name(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _env_literal(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and TOGGLE_NAME_RE.search(node.value):
        return node.value
    return None


def scan_env_toggles(source: str, filename: str = "<source>") -> dict[str, str]:
    """Map env names read by literal os.environ.get / os.getenv / os.environ[...] to 'file:line'."""
    found: dict[str, str] = {}
    for node in ast.walk(ast.parse(source, filename=filename)):
        name: str | None = None
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            is_get = isinstance(func, ast.Attribute) and func.attr == "get" and _is_environ(func.value)
            is_getenv = isinstance(func, ast.Attribute) and func.attr == "getenv" and _is_name(func.value, "os")
            if is_get or is_getenv:
                name = _env_literal(node.args[0])
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            name = _env_literal(node.slice)
        if name is not None:
            found.setdefault(name, f"{filename}:{node.lineno}")
    return found


def discover_env_toggles(root: Path = SOURCE_ROOT) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        found.update(scan_env_toggles(path.read_text(encoding="utf-8"), relative))
    return found


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        rows = tomllib.load(handle).get("flag", [])
    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["name"]
        if name in registry:
            raise ValueError(f"duplicate flag registry row: {name}")
        registry[name] = row
    return registry


def missing_rows(toggles: Iterable[str], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: toggle has no row in docs/flags.toml ({RULE_FILE})" for name in sorted(toggles) if name not in registry
    ]


def stale_rows(toggles: Iterable[str], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    known = set(toggles)
    return [
        f"{name}: registry row names no existing toggle ({RULE_FILE})" for name in sorted(registry) if name not in known
    ]


def kind_violations(registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: kind {row.get('kind')!r} is not one of {sorted(KINDS)} ({RULE_FILE})"
        for name, row in sorted(registry.items())
        if row.get("kind") not in KINDS
    ]


def default_violations(toggles: Mapping[str, bool], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: registry default {row.get('default')!r} differs from code default {toggles[name]!r} ({RULE_FILE})"
        for name, row in sorted(registry.items())
        if name in toggles and row.get("default") != toggles[name]
    ]


def component_violations(registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: component must default to True, found {row.get('default')!r} ({RULE_FILE})"
        for name, row in sorted(registry.items())
        if row.get("kind") == "component" and row.get("default") is not True
    ]


def experiment_violations(registry: Mapping[str, Mapping[str, Any]], today: date) -> list[str]:
    violations: list[str] = []
    for name, row in sorted(registry.items()):
        if row.get("kind") != "experiment":
            continue
        if not str(row.get("exit_condition", "")).strip():
            violations.append(f"{name}: experiment needs a non-empty exit_condition ({RULE_FILE})")
        try:
            review_by = date.fromisoformat(str(row.get("review_by", "")))
        except ValueError:
            violations.append(f"{name}: experiment review_by must be YYYY-MM-DD ({RULE_FILE})")
            continue
        if today > review_by + timedelta(days=REVIEW_GRACE_DAYS):
            violations.append(
                f"{name}: experiment review_by {review_by.isoformat()} is more than {REVIEW_GRACE_DAYS} days past; "
                f"delete the flag and its OFF branch ({RULE_FILE})"
            )
    return violations


def test_no_unregistered_toggles() -> None:
    toggles = {*discover_model_toggles(), *discover_env_toggles()}

    assert missing_rows(toggles, load_registry()) == []


def test_registry_has_no_stale_rows() -> None:
    toggles = {*discover_model_toggles(), *discover_env_toggles()}

    assert stale_rows(toggles, load_registry()) == []


def test_policy_detects_new_default_off_flag() -> None:
    class _Synthetic(BaseModel):
        NEW_FEATURE_ENABLED: bool = False

    model_toggles = discover_model_toggles([_Synthetic])
    env_toggles = scan_env_toggles('import os\nvalue = os.environ.get("NEW_RESEND_ON_FAILURE")\n', "synthetic.py")

    assert model_toggles == {"_Synthetic.NEW_FEATURE_ENABLED": False}
    assert env_toggles == {"NEW_RESEND_ON_FAILURE": "synthetic.py:2"}
    unregistered = missing_rows({*model_toggles, *env_toggles}, load_registry())
    assert f"_Synthetic.NEW_FEATURE_ENABLED: toggle has no row in docs/flags.toml ({RULE_FILE})" in unregistered
    assert f"NEW_RESEND_ON_FAILURE: toggle has no row in docs/flags.toml ({RULE_FILE})" in unregistered


def test_env_scan_covers_getenv_and_subscript_only_for_toggle_names() -> None:
    source = (
        "import os\n"
        'a = os.getenv("SHARD_WATCHDOG_DISABLE")\n'
        'b = os.environ["LEGACY_FORCE_ON"]\n'
        'c = os.environ.get("KSE_KIND_TRANSPORT", "direct")\n'
        'd = os.environ.get(name_variable)\n'
    )

    assert scan_env_toggles(source, "synthetic.py") == {
        "SHARD_WATCHDOG_DISABLE": "synthetic.py:2",
        "LEGACY_FORCE_ON": "synthetic.py:3",
    }


def test_experiment_requires_deadline_and_exit() -> None:
    today = date(2026, 11, 11)
    base = {"kind": "experiment", "default": False, "reason": "r"}
    missing_exit = {**base, "exit_condition": "", "review_by": today.isoformat()}
    missing_date = {**base, "exit_condition": "metric holds", "review_by": ""}
    stale = {**base, "exit_condition": "metric holds", "review_by": (today - timedelta(days=15)).isoformat()}
    within_grace = {**base, "exit_condition": "metric holds", "review_by": (today - timedelta(days=13)).isoformat()}

    assert experiment_violations({"X_ENABLED": missing_exit}, today) != []
    assert experiment_violations({"X_ENABLED": missing_date}, today) != []
    assert experiment_violations({"X_ENABLED": stale}, today) != []
    assert experiment_violations({"X_ENABLED": within_grace}, today) == []


def test_registry_experiments_are_not_expired() -> None:
    assert experiment_violations(load_registry(), date.today()) == []


def test_registry_kind_is_valid() -> None:
    assert kind_violations(load_registry()) == []
    assert kind_violations({"X_ENABLED": {"kind": "rollout", "default": True}}) == [
        f"X_ENABLED: kind 'rollout' is not one of {sorted(KINDS)} ({RULE_FILE})"
    ]


def test_registry_default_matches_code() -> None:
    assert default_violations(discover_model_toggles(), load_registry()) == []


def test_component_defaults_on() -> None:
    assert component_violations(load_registry()) == []
    assert component_violations({"X_ENABLED": {"kind": "component", "default": False}}) != []


def test_policy_ignores_policy_data_booleans() -> None:
    discovered = discover_model_toggles()
    policy_data = {"DartKeyPolicy.default", "DocumentParserPolicy.trusted", "KisPolicy.require_shared_admission"}

    assert policy_data.isdisjoint(discovered)


def test_deploy_baseline_check_is_skipped_by_design() -> None:
    assert "deploy baseline" in (__doc__ or "").lower()
    assert not (REPO_ROOT / "docs" / "deploy_baseline.json").exists()


def test_config_flags_rule_within_budget() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(BUDGET_TOOL)], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "violations=0" in result.stdout
