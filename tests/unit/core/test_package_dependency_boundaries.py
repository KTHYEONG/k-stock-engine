"""AST-level package dependency boundaries for ``src``.

Every import statement is inspected, including function-local (lazy) imports
that line-anchored regex checks cannot see. ``ALLOWED`` is the target layering;
``KNOWN_VIOLATIONS`` is a ratchet: existing violations are pinned so no new one
can appear, and an entry that no longer occurs must be deleted so the list only
shrinks.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"

ALLOWED: dict[str, frozenset[str]] = {
    "core": frozenset(),
    "storage": frozenset({"core"}),
    "execution": frozenset({"core"}),
    "integrations": frozenset({"core", "storage"}),
    "data": frozenset({"core", "storage", "integrations"}),
    "backtest": frozenset({"core", "storage", "data"}),
}

# (importing file relative to repo root, imported top-level package)
KNOWN_VIOLATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("src/integrations/dart/client.py", "data"),
        ("src/integrations/kis/industry.py", "data"),
        ("src/integrations/kis/investor_flow.py", "data"),
    }
)


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


def _observed_violations() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in SRC.rglob("*.py"):
        package = path.relative_to(SRC).parts[0]
        if package not in ALLOWED:
            continue
        for module in _imported_modules(ast.parse(path.read_text(encoding="utf-8"))):
            parts = module.split(".")
            if parts[0] != "src" or len(parts) < 2 or parts[1] == package:
                continue
            if parts[1] not in ALLOWED[package]:
                found.add((path.relative_to(SRC.parent).as_posix(), parts[1]))
    return found


def test_no_new_package_dependency_violations() -> None:
    new = _observed_violations() - KNOWN_VIOLATIONS
    assert not new, f"imports outside the allowed package layering: {sorted(new)}"


def test_known_violations_list_only_shrinks() -> None:
    stale = KNOWN_VIOLATIONS - _observed_violations()
    assert not stale, f"remove resolved entries from KNOWN_VIOLATIONS: {sorted(stale)}"


def test_every_source_package_has_a_declared_layer() -> None:
    packages = {p.name for p in SRC.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert packages <= set(ALLOWED), f"undeclared packages: {sorted(packages - set(ALLOWED))}"


def test_layering_detects_lazy_imports() -> None:
    tree = ast.parse("def f():\n    from src.strategy.x import y\n")
    assert _imported_modules(tree) == ["src.strategy.x"]
