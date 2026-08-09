"""Architecture boundary contract.

``core`` must not import ``stocks``, ``etfs``, ``execution``, or ``legacy``;
``stocks`` and ``etfs`` must not import one another or ``legacy``;
``storage`` imports only ``core``. No modern package may import quarantined
legacy modules.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "src"

MODERN_PACKAGES = ("core", "storage", "stocks", "etfs", "execution")
LEGACY_PREFIXES = (
    "src.legacy",
    "src.etf",
    "src.training",
    "src.evaluation",
    "src.filters",
    "src.data.etf_manager",
    "src.execution.kis_client",
    "src.execution.yeti",
)
FORBIDDEN_CORE = ("src.stocks", "src.etfs", "src.execution", "src.storage", "src.legacy")
FORBIDDEN_STOCK_ETF_CROSS = {"stocks": "src.etfs", "etfs": "src.stocks"}
STORAGE_FORBIDDEN = ("src.stocks", "src.etfs", "src.execution", "src.legacy")


def _walk(pkg: str) -> list[Path]:
    base = SRC / pkg
    return [p for p in base.rglob("*.py") if "__pycache__" not in str(p)]


def _imports(text: str) -> set[str]:
    found: set[str] = set()
    for m in re.finditer(r"^(?:from|import)\s+((?:src\.)?[a-zA-Z0-9_.]+)", text, re.MULTILINE):
        module = m.group(1)
        if module.startswith("src."):
            found.add(module)
        elif module.startswith("src"):
            found.add("src" + module[3:])
    return found


class TestImportBoundaries:
    def test_core_does_not_import_asset_or_execution_modules(self) -> None:
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

    def test_stocks_and_etfs_do_not_import_each_other(self) -> None:
        for pkg, forbidden in FORBIDDEN_STOCK_ETF_CROSS.items():
            for path in _walk(pkg):
                text = path.read_text(encoding="utf-8")
                imports = _imports(text)
                assert not any(i.startswith(forbidden) for i in imports), (
                    f"{path} must not import {forbidden}"
                )

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
                and not i.startswith(("src.execution", "src.core"))
            }
            assert not external, f"{path} imports outside execution/core: {external}"
