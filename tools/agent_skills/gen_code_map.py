#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from tools.agent_skills import lean_check  # noqa: E402


def _matching_tests(source_file: str, test_files: list[str]) -> list[str]:
    """Return every repository test that covers ``source_file``.

    Exact mirrored ``tests/<category>/<dir>/test_<module>.py`` paths are the fast
    path; otherwise the lean-check AST semantic reference matcher is reused so
    feature-named CLI/workflow tests remain linked.
    """
    parts = source_file.split("/")
    module_name = parts[-1]
    test_name = f"test_{module_name}"
    exact = {
        f"tests/{category}/{'/'.join(parts[1:-1])}/{test_name}" if parts[1:-1]
        else f"tests/{category}/{test_name}"
        for category in ("unit", "integration", "e2e")
    }
    matched = [tp for tp in test_files if tp in exact]
    if matched:
        return matched
    return [tp for tp in test_files if lean_check._test_references_source(tp, source_file)]


def main() -> None:
    py_files: list[str] = []
    for root, dirs, files in os.walk("src"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        py_files.extend(
            os.path.join(root, filename)
            for filename in sorted(files)
            if filename.endswith(".py")
        )
    py_files = sorted(py_files)
    test_files = lean_check._repository_test_files()

    code_map: dict[str, object] = {}
    for source_file in py_files:
        if source_file.endswith("__init__.py"):
            continue
        matched = _matching_tests(source_file, test_files)
        entry: dict[str, object] = {}
        if matched:
            entry["testing"] = matched[0] if len(matched) == 1 else matched
        code_map[source_file] = entry

    with open("docs/code_map.json", "w", encoding="utf-8") as handle:
        json.dump(code_map, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"regenerated docs/code_map.json with {len(code_map)} canonical sources")


if __name__ == "__main__":
    main()
