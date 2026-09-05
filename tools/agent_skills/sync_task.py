#!/usr/bin/env python3
"""Sync: Task Metadata + Code Map + Cleanup Protocol."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from datetime import datetime


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _path_exists(path: str) -> bool:
    return os.path.exists(path)


def _resolve_test_path(source_file: str) -> str | None:
    if not source_file.startswith("src/") or source_file.endswith("__init__.py"):
        return None
    parts = source_file.split("/")
    module_name = parts[-1]
    test_name = f"test_{module_name}"
    # Root-level tests (repo convention: tests/test_<module>.py)
    if os.path.exists(f"tests/{test_name}"):
        return f"tests/{test_name}"
    for category in ["unit", "integration", "e2e"]:
        sub = "/".join(parts[1:-1])
        test_dir = f"tests/{category}/{sub}" if sub else f"tests/{category}"
        tp = f"{test_dir}/{test_name}"
        if os.path.exists(tp):
            return tp
    return None


_MEMORY_FIELD_CHAR_CAP = 300


def _cap_field(text: str) -> str:
    """Hard backstop on why/what/impact -- these get echoed into every future
    spec_init.py match, so an unbounded field taxes every later spec run, not
    just this one. Prompt guidance (sync/SKILL.md) is the first line; this is
    the fail-closed script-level one.
    """
    text = text.strip()
    if len(text) <= _MEMORY_FIELD_CHAR_CAP:
        return text
    return text[:_MEMORY_FIELD_CHAR_CAP].rstrip() + "..."


def _update_decisions_json(
    task: str,
    title: str,
    why: str,
    what: str,
    impact: str,
    domain: str,
    failed_hypothesis: str | None = None,
    failure_reason: str | None = None,
    archive_path: str | None = None,
) -> str:
    # Tolerate absent active task_index.json; do not recreate archived records under docs/
    # Archived decisions remain in legacy/docs/decisions
    if os.path.exists("legacy/docs/decisions/task_index.json") and not os.path.exists("docs/decisions"):
        # Active decisions are archived; skip recreation
        date_str = datetime.now().strftime("%Y-%m-%d")
        adr_date = datetime.now().strftime("%Y%m%d")
        return f"ADR_{adr_date}_{task.replace('TASK_', '')}"
    date_str = datetime.now().strftime("%Y-%m-%d")
    adr_date = datetime.now().strftime("%Y%m%d")
    adr_id = f"ADR_{adr_date}_{task.replace('TASK_', '')}"

    tasks_json_path = "docs/decisions/task_index.json"

    # 1. Update task_index.json
    index_data: dict[str, list[dict[str, str]]] = {"tasks": []}
    if _path_exists(tasks_json_path):
        with contextlib.suppress(Exception):
            parsed = json.loads(_read_file(tasks_json_path))
            if isinstance(parsed, dict) and "tasks" in parsed and isinstance(parsed["tasks"], list):
                index_data = parsed

    new_task_entry = {
        "task_id": task,
        "date": date_str,
        "adr_id": adr_id,
        "domain": domain,
        "title": title,
        "why": _cap_field(why),
        "resolution": _cap_field(what),
        "impact": _cap_field(impact),
    }
    if archive_path:
        # Pointer, not content -- spec_init.py surfaces this path so a future
        # spec can Read the full design_rationale/performance_budget on demand,
        # instead of the archived contract being an unreachable dead file.
        new_task_entry["archive_path"] = archive_path

    # Prepend new task entry
    index_data["tasks"] = [new_task_entry] + [t for t in index_data.get("tasks", []) if t.get("task_id") != task]
    # Keep max 100 entries in task_index.json
    index_data["tasks"] = index_data["tasks"][:100]
    _write_file(tasks_json_path, json.dumps(index_data, indent=2, ensure_ascii=False) + "\n")

    return adr_id


def _update_index(source_file: str, test_file: str | None, doc_file: str | None) -> None:
    # Tolerate absent active code_map.json; do not recreate archived records under docs/
    if os.path.exists("legacy/docs/code_map.json") and not os.path.exists("docs/code_map.json"):
        return
    code_map_path = "docs/code_map.json"
    data: dict[str, dict[str, str | list[str]]] = {}

    if _path_exists(code_map_path):
        try:
            data = json.loads(_read_file(code_map_path))
            if not isinstance(data, dict):
                data = {}
        except (json.JSONDecodeError, ValueError):
            shutil.copyfile(code_map_path, f"{code_map_path}.bak")
            data = {}

    if source_file not in data:
        data[source_file] = {}

    entry = data[source_file]
    if doc_file is not None:
        entry["architecture"] = doc_file
    if test_file is not None:
        cur = entry.get("testing")
        if cur is None:
            entry["testing"] = test_file
        elif isinstance(cur, str) and cur != test_file:
            entry["testing"] = [cur, test_file]
        elif isinstance(cur, list) and test_file not in cur:
            cur.append(test_file)

    _write_file(code_map_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


EXCLUDED_DIRS = frozenset({".git", ".venv", ".mypy_cache", ".ruff_cache", "__pycache__", "node_modules"})


def _wipe_temp_artifacts() -> int:
    count = 0
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for f in files:
            if f.endswith((".tmp", ".bak")):
                fpath = os.path.join(root, f)
                try:
                    os.remove(fpath)
                    count += 1
                except OSError:
                    pass
    return count


def _clean_scratch_dir() -> int:
    scratch_dir = "scratch"
    if not os.path.exists(scratch_dir):
        return 0
    count = 0
    for root, dirs, files in os.walk(scratch_dir, topdown=False):
        for f in files:
            if f == ".gitignore":
                continue
            fpath = os.path.join(root, f)
            try:
                os.remove(fpath)
                count += 1
            except OSError:
                pass
        for d in dirs:
            dpath = os.path.join(root, d)
            with contextlib.suppress(OSError):
                os.rmdir(dpath)
    return count


def _clean_tmp_dir() -> int:
    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        return 0
    count = 0
    for root, dirs, files in os.walk(tmp_dir, topdown=False):
        for f in files:
            if f == ".gitignore":
                continue
            fpath = os.path.join(root, f)
            try:
                os.remove(fpath)
                count += 1
            except OSError:
                pass
        for d in dirs:
            dpath = os.path.join(root, d)
            with contextlib.suppress(OSError):
                os.rmdir(dpath)
    return count


def _clean_logs_dir() -> int:
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return 0
    count = 0
    for root, dirs, files in os.walk(logs_dir, topdown=False):
        for f in files:
            if f == ".gitignore":
                continue
            fpath = os.path.join(root, f)
            try:
                os.remove(fpath)
                count += 1
            except OSError:
                pass
        for d in dirs:
            dpath = os.path.join(root, d)
            with contextlib.suppress(OSError):
                os.rmdir(dpath)
    return count


def _clean_specs(task_id: str, remove_specs: list[str] | None = None) -> tuple[int, list[str]]:
    specs_dir = "docs/specs"
    if not _path_exists(specs_dir):
        return (0, [])

    target_prefixes: set[str] = set()
    if remove_specs:
        for item in remove_specs:
            base = item.replace(".md", "").replace("_contract.json", "").replace("contract.json", "").replace("docs/specs/", "").strip()
            if base:
                target_prefixes.add(base.lower())

    archive_dir = os.path.join("docs/decisions/archive", task_id)
    count = 0
    archived: list[str] = []
    for fname in os.listdir(specs_dir):
        if fname.endswith((".md", "_contract.json", "contract.json")):
            if fname == "00_architecture.md":
                continue
            if target_prefixes:
                fname_base = fname.replace(".md", "").replace("_contract.json", "").replace("contract.json", "").lower()
                if fname_base not in target_prefixes and fname.lower() not in target_prefixes:
                    continue

            fpath = os.path.join(specs_dir, fname)
            try:
                if fname.endswith("_contract.json") or fname == "contract.json":
                    # 계약(설계 근거)은 삭제 대신 아카이브 — spec 재사용 및 감사 추적성 보존
                    os.makedirs(archive_dir, exist_ok=True)
                    dest = os.path.join(archive_dir, fname)
                    shutil.move(fpath, dest)
                    archived.append(dest)
                else:
                    os.remove(fpath)
                count += 1
            except OSError:
                pass
    return (count, archived)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync: Task Registry + Code Map + Anti-Pattern Registry + Cleanup.")
    parser.add_argument("--task", required=True, help="Task ID (e.g. TASK_L0_MTF_FUSION)")
    parser.add_argument("--title", required=True, help="Decision title")
    parser.add_argument("--why", required=True, help="Context/Why")
    parser.add_argument("--what", required=True, help="Resolution/What")
    parser.add_argument("--impact", required=True, help="Impact")
    parser.add_argument("--source", default=None, help="Modified source file path (auto-detected if omitted)")
    parser.add_argument("--domain", default="general", help="Domain category (e.g. signal, risk, execution)")
    parser.add_argument("--failed-hypothesis", default=None, help="Failed hypothesis if applicable")
    parser.add_argument("--failure-reason", default=None, help="Reason why hypothesis failed")
    parser.add_argument("--test", default=None, help="Test file path")
    parser.add_argument("--doc", default=None, help="Architecture doc path")
    parser.add_argument("--remove-specs", nargs="*", default=[], help="Spec files to remove")
    args = parser.parse_args()

    logs: list[str] = []
    errors: list[str] = []

    # Auto-detect source file if omitted
    source_file = args.source
    if not source_file:
        try:
            import subprocess
            diff_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=10, check=False
            )
            for line in diff_res.stdout.splitlines():
                fp = line[3:].strip()
                if fp.startswith("src/") and fp.endswith(".py"):
                    source_file = fp
                    break
        except Exception:
            pass
    if not source_file:
        source_file = "src/main.py"

    # 1. Spec Cleanup (archive contracts first so their path can be linked
    #    into the task_index.json entry created in step 2)
    archived_contract_paths: list[str] = []
    try:
        cleaned, archived_contract_paths = _clean_specs(task_id=args.task, remove_specs=args.remove_specs)
        if cleaned > 0:
            logs.append(f"Archived/cleaned {cleaned} spec files")
    except Exception as e:
        errors.append(f"Spec cleanup failed: {e}")

    # 2. Update Smart JSON Registries (task_index.json & anti_patterns.json)
    try:
        adr_id = _update_decisions_json(
            task=args.task,
            title=args.title,
            why=args.why,
            what=args.what,
            impact=args.impact,
            domain=args.domain,
            failed_hypothesis=args.failed_hypothesis,
            failure_reason=args.failure_reason,
            archive_path=archived_contract_paths[0] if archived_contract_paths else None,
        )
        logs.append(f"Task Registry updated ({adr_id})")
    except Exception as e:
        errors.append(f"JSON Registry update failed: {e}")
        adr_id = "N/A"

    # 3. Code Map Update for files
    test_file = args.test or _resolve_test_path(source_file)
    try:
        _update_index(source_file, test_file, args.doc)
        logs.append(f"Code map updated for {source_file}")
    except Exception as e:
        errors.append(f"Code map update failed: {e}")

    # Auto-regenerate code map
    with contextlib.suppress(Exception):
        from tools.agent_skills import gen_code_map
        gen_code_map.main()


    # 4. Temp & Scratch & Logs Wipe
    try:
        wiped = _wipe_temp_artifacts()
        scratch_wiped = _clean_scratch_dir()
        tmp_wiped = _clean_tmp_dir()
        logs_wiped = _clean_logs_dir()
        total_cleaned = wiped + scratch_wiped + tmp_wiped + logs_wiped
        if total_cleaned > 0:
            logs.append(
                f"Wiped {wiped} temp, {scratch_wiped} scratch, {tmp_wiped} tmp, {logs_wiped} logs files"
            )
    except Exception as e:
        errors.append(f"Temp wipe failed: {e}")

    # 5. Summary
    status = "OK" if not errors else "PARTIAL"
    summary = f"### 🏁 [SYNC:{status}] [{adr_id}] | {' | '.join(logs)}"
    print(summary)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
