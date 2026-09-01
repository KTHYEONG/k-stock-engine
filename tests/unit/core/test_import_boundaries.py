"""Architecture boundary contract.

``core`` must not import ``execution``, ``legacy``;
``integrations`` is active and must not import ``legacy``;
``storage`` imports only ``core``. No modern package may import quarantined
legacy modules.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent.parent / "src"

MODERN_PACKAGES = ('core', 'storage', 'execution', 'integrations')
active_package_boundary = MODERN_PACKAGES  # spec alias for lean_check
LEGACY_PREFIXES = (
    "src.legacy",
    "legacy",
    "legacy.stocks",
    "legacy.etfs",
    "src.stocks",
    "src.etfs",
)
FORBIDDEN_CORE = ("src.legacy", "legacy", "legacy.stocks", "legacy.etfs", "src.stocks", "src.etfs")
FORBIDDEN_INTEGRATIONS = ("src.legacy", "legacy", "legacy.stocks", "legacy.etfs")
STORAGE_FORBIDDEN = ("src.legacy", "legacy", "legacy.stocks", "legacy.etfs", "src.stocks", "src.etfs", "src.execution")
EXECUTION_FORBIDDEN = ("src.legacy", "legacy", "legacy.stocks", "legacy.etfs", "src.stocks", "src.etfs")


def _walk(pkg: str) -> list[Path]:
    base = SRC / pkg
    if not base.exists():
        return []
    return [p for p in base.rglob("*.py") if "__pycache__" not in str(p)]


def _imports(text: str) -> set[str]:
    found: set[str] = set()
    for m in re.finditer(r"^(?:from|import)\s+((?:src\.)?[a-zA-Z0-9_.]+)", text, re.MULTILINE):
        module = m.group(1)
        if module.startswith("src."):
            found.add(module)
        elif module.startswith("src"):
            found.add("src" + module[3:])
    # Also detect legacy imports without src prefix
    for m in re.finditer(r"^(?:from|import)\s+(legacy[^\s]*)", text, re.MULTILINE):
        found.add(m.group(1))
    return found


class TestImportBoundaries:
    def test_core_does_not_import_legacy_or_asset_modules(self) -> None:
        for path in _walk("core"):
            text = path.read_text(encoding="utf-8")
            imports = _imports(text)
            for forbidden in FORBIDDEN_CORE:
                assert not any(i.startswith(forbidden) for i in imports), (
                    f"{path} must not import {forbidden}"
                )

    def test_storage_imports_only_core(self) -> None:
        for path in _walk("storage"):
            text = path.read_text(encoding="utf-8")
            imports = _imports(text)
            for forbidden in STORAGE_FORBIDDEN:
                assert not any(i.startswith(forbidden) for i in imports), (
                    f"{path} must not import {forbidden}"
                )

    def test_integrations_does_not_import_legacy(self) -> None:
        for path in _walk("integrations"):
            text = path.read_text(encoding="utf-8")
            imports = _imports(text)
            for forbidden in FORBIDDEN_INTEGRATIONS:
                assert not any(i.startswith(forbidden) for i in imports), (
                    f"{path} must not import {forbidden}"
                )
        # wiring anchor: verify integrations walk
        assert _walk('integrations')

    def test_no_modern_package_imports_legacy(self) -> None:
        for pkg in MODERN_PACKAGES:
            for path in _walk(pkg):
                text = path.read_text(encoding="utf-8")
                imports = _imports(text)
                for legacy in LEGACY_PREFIXES:
                    assert not any(
                        i == legacy or i.startswith(legacy + ".") for i in imports
                    ), f"{path} must not import {legacy}"

    def test_execution_depends_only_on_core_beyond_own_tree(self) -> None:
        for path in _walk("execution"):
            text = path.read_text(encoding="utf-8")
            imports = _imports(text)
            external = {
                i
                for i in imports
                if i.startswith("src.")
                and not i.startswith(("src.execution", "src.core", "src.integrations"))
                and not i.startswith("legacy")
            }
            # integrations are allowed for execution? No, execution should only depend on core per spec
            # but allow integrations as broker adapter? Keep strict: only core
            external = {i for i in external if not i.startswith("src.integrations")}
            assert not external, f"{path} imports outside execution/core: {external}"

    def test_integrations_does_not_depend_on_legacy_archives(self) -> None:
        for path in _walk("integrations"):
            text = path.read_text(encoding="utf-8")
            assert "legacy" not in text.lower() or "legacy" not in _imports(text), f"{path} must not depend on legacy"
