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
    "config": frozenset({"core"}),
    "execution": frozenset({"core"}),
    "integrations": frozenset({"core", "config"}),
    "data": frozenset({"core", "config", "integrations"}),
    "backtest": frozenset({"core", "config", "data"}),
    "research": frozenset({"core", "config", "data", "backtest"}),
}

# (importing file relative to repo root, imported top-level package)
KNOWN_VIOLATIONS: frozenset[tuple[str, str]] = frozenset()


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


# Folded live assertions from the removed-architecture boundary tests
# (test_active_archive_boundary, test_architecture_consolidation,
# test_import_boundaries): the retired-prefix ban and the legacy quarantine
# layout are still live because ``legacy/`` exists.

_RETIRED_PREFIXES = (
    "src.legacy",
    "src.stocks",
    "src.etfs",
    "legacy.stocks",
    "legacy.etfs",
    "src.backtest.strategy",
    "src.backtest.view",
    "src.backtest.manifest",
    "src.backtest.metrics",
    "src.backtest.cli",
    "src.research.features",
    "src.research.strategy",
    "src.research.gates",
    "src.research.evaluator",
)


def test_no_retired_prefix_references() -> None:
    import re

    root = SRC.parent
    pat = re.compile(r"(?:{})".format("|".join(re.escape(prefix) for prefix in _RETIRED_PREFIXES)))
    active_files = [
        p
        for p in [*SRC.rglob("*.py"), *(root / "tests").rglob("*.py")]
        if p.name != Path(__file__).name
    ]
    for path in active_files:
        text = path.read_text(encoding="utf-8")
        assert not pat.search(text), f"{path} contains retired prefix"


def test_quarantined_legacy_layout() -> None:
    root = SRC.parent
    legacy = root / "legacy"
    assert (legacy / "stocks").is_dir()
    assert (legacy / "etfs").is_dir()
    assert (legacy / "live_yeti_v1").is_dir()
    assert not (SRC / "stocks").exists()
    assert not (SRC / "etfs").exists()
    assert not (SRC / "legacy").exists()


def test_no_reexport_only_modules() -> None:
    for pkg in ("core", "execution", "integrations"):
        base = SRC / pkg
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in str(path) or path.name == "__init__.py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            defines = any(
                isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                or (
                    isinstance(node, (ast.Assign, ast.AnnAssign))
                    and not (
                        isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                    )
                )
                for node in tree.body
            )
            assert defines, f"{path} is a re-export-only module"


# The Bronze writer is reachable only through the scoped writer. The two pinned
# modules below still reach it directly: C1 moves the KIS gap backfill and the
# Bronze aggregation over to catalog reads, after which this list must be empty.

BRONZE_STORE_IMPORTERS: frozenset[str] = frozenset(
    {"scoped_ingestion.py", "dart_documents.py"}
)
PINNED_BRONZE_STORE_IMPORTERS: frozenset[str] = frozenset()


def _bronze_store_importers() -> set[str]:
    found: set[str] = set()
    for path in (SRC / "data").rglob("*.py"):
        for module in _imported_modules(ast.parse(path.read_text(encoding="utf-8"))):
            if module == "src.data.bronze":
                found.add(path.name)
    return found


def test_bronze_store_is_imported_only_by_allowed_modules() -> None:
    observed = _bronze_store_importers()
    unallowed = observed - BRONZE_STORE_IMPORTERS - PINNED_BRONZE_STORE_IMPORTERS
    assert not unallowed, f"BronzeStore imported outside the writer boundary: {sorted(unallowed)}"


def test_pinned_bronze_store_importers_only_shrink() -> None:
    stale = PINNED_BRONZE_STORE_IMPORTERS - _bronze_store_importers()
    assert not stale, f"remove migrated modules from PINNED_BRONZE_STORE_IMPORTERS: {sorted(stale)}"
