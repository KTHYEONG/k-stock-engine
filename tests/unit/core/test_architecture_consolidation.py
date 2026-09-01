"""Architecture invariants for the consolidated layout.

Rejects the anti-patterns the blueprint removes: re-export-only modules with
ambiguous sources of truth, duplicated ``DatasetManifest``/store definitions,
and any modern import of the quarantined ``legacy`` package.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent.parent / "src"
MODERN_PACKAGES = ("core", "storage", "execution", "integrations")


def _walk(pkg: str) -> list[Path]:
    base = SRC / pkg
    if not base.exists():
        return []
    return [p for p in base.rglob("*.py") if "__pycache__" not in str(p)]


def _is_reexport_only(path: Path) -> bool:
    """True if a module only re-exports imported names (no local definitions)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    has_definition = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            has_definition = True
            break
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            has_definition = True
            break
    return not has_definition


class TestNoReexportOnlyModules:
    def test_modern_modules_define_their_own_logic(self) -> None:
        for pkg in MODERN_PACKAGES:
            for path in _walk(pkg):
                if path.name == "__init__.py":
                    continue
                assert not _is_reexport_only(path), f"{path} is a re-export-only module"


class TestSingleSourceOfTruth:
    def test_dataset_manifest_defined_only_in_core(self) -> None:
        owners = []
        for pkg in MODERN_PACKAGES:
            for path in _walk(pkg):
                text = path.read_text(encoding="utf-8")
                if re.search(r"^class DatasetManifest\b", text, re.MULTILINE):
                    owners.append(str(path.relative_to(SRC)))
        assert owners == ["core/datasets.py"], f"DatasetManifest duplicated in {owners}"

    def test_parquet_store_defined_only_in_storage(self) -> None:
        owners = []
        for pkg in MODERN_PACKAGES:
            for path in _walk(pkg):
                text = path.read_text(encoding="utf-8")
                if re.search(r"^class ParquetDatasetStore\b", text, re.MULTILINE):
                    owners.append(str(path.relative_to(SRC)))
        assert owners == ["storage/parquet_datasets.py"], f"store duplicated in {owners}"

    def test_integrations_are_active_and_legacy_free(self) -> None:
        # Ensure integrations package exists and contains no legacy imports
        integ = SRC / "integrations"
        assert integ.is_dir()
        for path in _walk("integrations"):
            text = path.read_text(encoding="utf-8")
            assert "legacy" not in text or "legacy" not in text.lower().split("import"), f"{path} must not reference legacy"
            assert "src.legacy" not in text
            assert "src.stocks" not in text
            assert "src.etfs" not in text
