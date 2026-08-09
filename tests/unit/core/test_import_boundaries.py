"""Phase 1 exit gate: static import-boundary contract.

``core`` must not import ``stocks``, ``etfs``, or ``execution``; ``stocks`` and
``etfs`` must not import one another; ``execution`` may depend on ``core`` only.
No newly written code may import legacy ``training``/``evaluation``/``filters``/
``etf`` modules.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "src"

FORBIDDEN_CORE = ("src.stocks", "src.etfs", "src.execution", "src.etf", "src.training", "src.evaluation", "src.filters")
FORBIDDEN_STOCK_ETF_CROSS = {"stocks": "src.etfs", "etfs": "src.stocks"}
FORBIDDEN_ALL = ("src.training", "src.evaluation", "src.filters", "src.etf", "src.data.etf_manager")


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

    def test_stocks_and_etfs_do_not_import_each_other(self) -> None:
        for pkg, forbidden in FORBIDDEN_STOCK_ETF_CROSS.items():
            for path in _walk(pkg):
                text = path.read_text(encoding="utf-8")
                imports = _imports(text)
                assert not any(i.startswith(forbidden) for i in imports), (
                    f"{path} must not import {forbidden}"
                )

    def test_no_legacy_imports_in_new_code(self) -> None:
        for pkg in ("core", "stocks", "etfs", "execution"):
            for path in _walk(pkg):
                text = path.read_text(encoding="utf-8")
                imports = _imports(text)
                for forbidden in FORBIDDEN_ALL:
                    assert not any(i == forbidden or i.startswith(forbidden + ".") for i in imports), (
                        f"{path} must not import {forbidden}"
                    )

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
