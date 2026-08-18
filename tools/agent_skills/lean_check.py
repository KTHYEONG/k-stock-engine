#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import functools
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

JsonDiag = dict[str, Any]

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def _emit_json(
    status: str,
    phase: str,
    diagnostics: list[JsonDiag],
    coverage: int | None = None,
) -> str:
    return json.dumps(
        {
            "status": status,
            "phase": phase,
            "exit_code": 0 if status == "PASS" else 1,
            "coverage": coverage,
            "diagnostics": diagnostics,
        }
    )


def _fail_exit_many(phase: str, header: str, diags: list[JsonDiag]) -> None:
    print(header)
    for d in diags:
        print(f"FAIL | {d.get('error', '')}")
    print(_emit_json("FAIL", phase, diags), file=sys.stderr)
    sys.exit(1)


def _fail_exit(phase: str, msg: str, diag: JsonDiag) -> None:
    print(msg)
    print(_emit_json("FAIL", phase, [diag]), file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    # Strip unnecessary 'uv run' prefix when already running inside virtualenv to avoid double env setup overhead
    if (
        len(cmd) >= 3
        and cmd[0] == "uv"
        and cmd[1] == "run"
        and os.environ.get("VIRTUAL_ENV")
    ):
        cmd = cmd[2:]
    env = os.environ.copy()
    env["COVERAGE_NO_CTRACE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        return subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, shell=False, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout="",
            stderr=f"Error: timed out after {timeout}s.",
        )


def _repo_relative(path: str) -> str:
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, os.getcwd())
    except ValueError:
        return path


@functools.lru_cache(maxsize=1)
def _get_src_files_contents() -> tuple[tuple[str, str], ...]:
    results: list[tuple[str, str]] = []
    if os.path.exists("src"):
        for root, dirs, files in os.walk("src"):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn_name in files:
                if fn_name.endswith(".py"):
                    fp = os.path.join(root, fn_name)
                    try:
                        with open(fp, encoding="utf-8", errors="ignore") as f:
                            results.append((fp, f.read()))
                    except OSError:
                        continue
    return tuple(results)


@functools.lru_cache(maxsize=1)
def _get_tests_files_contents() -> tuple[tuple[str, str], ...]:
    results: list[tuple[str, str]] = []
    if os.path.exists("tests"):
        for root, _dirs, fnames in os.walk("tests"):
            for fn in fnames:
                if fn.endswith(".py"):
                    fp = os.path.join(root, fn)
                    try:
                        with open(fp, encoding="utf-8", errors="ignore") as f:
                            results.append((fp, f.read()))
                    except OSError:
                        continue
    return tuple(results)


@functools.cache
def _repository_test_files() -> list[str]:
    """Return test modules in deterministic order for semantic source matching."""
    test_files: list[str] = []
    for root, _dirs, files in os.walk("tests"):
        test_files.extend(
            os.path.join(root, filename)
            for filename in sorted(files)
            if filename.startswith("test_") and filename.endswith(".py")
        )
    return sorted(test_files)


@functools.cache
def _load_test_ast(test_file: str) -> ast.AST | None:
    """Parse a test file once for repeated semantic source checks."""
    try:
        with open(test_file, encoding="utf-8") as handle:
            return ast.parse(handle.read(), filename=test_file)
    except (OSError, SyntaxError):
        return None


@functools.cache
def _imported_source_modules(test_file: str) -> frozenset[str]:
    """Return imported module paths from a cached test AST."""
    tree = _load_test_ast(test_file)
    if tree is None:
        return frozenset()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return frozenset(modules)


def _test_references_source(test_file: str, source_file: str) -> bool:
    """Match a test to a source module through its imports."""
    source_module = source_file[:-3].replace("/", ".")
    return source_module in _imported_source_modules(test_file)


def _check_orphaned_implementations(fh: str, kind: str, name: str) -> list[JsonDiag]:
    if kind == "field" or not fh.startswith("src"):
        return []
    leaf = name.rpartition(".")[2] if "." in name else name
    if not leaf:
        return []
    ref_pat = re.compile(rf"\b{re.escape(leaf)}\b")
    def_pat = re.compile(rf"^\s*(?:def|class)\s+{re.escape(leaf)}\b")
    found_caller = False
    for _fp, content in _get_src_files_contents():
        for line in content.splitlines():
            if ref_pat.search(line) and not def_pat.match(line):
                found_caller = True
                break
        if found_caller:
            break
    if found_caller:
        return []
    return [
        {
            "file": fh,
            "line": 0,
            "error": f"Spec: {kind} '{name}' has no callers in src/ outside its own definition (orphaned implementation)",
            "fix_hint": f"Wire {name} into its caller per the spec's wiring plan -- it currently does nothing in production",
        }
    ]


def _is_stub_node(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = node.body
    filtered_body = [
        stmt
        for stmt in body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
        and not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and getattr(getattr(stmt.value, "func", None), "attr", "")
            in ("debug", "info", "warning", "error", "critical")
        )
    ]
    if not filtered_body:
        return True
    if len(filtered_body) == 1:
        single = filtered_body[0]
        if isinstance(single, ast.Pass):
            return True
        if (
            isinstance(single, ast.Expr)
            and isinstance(single.value, ast.Constant)
            and single.value.value == Ellipsis
        ):
            return True
        if isinstance(single, ast.Raise):
            if (
                isinstance(single.exc, ast.Call)
                and getattr(single.exc.func, "id", None) == "NotImplementedError"
            ):
                return True
            if (
                isinstance(single.exc, ast.Name)
                and single.exc.id == "NotImplementedError"
            ):
                return True
        if isinstance(single, ast.Return):
            if single.value is None:
                return True
            if isinstance(single.value, ast.Constant) and (
                single.value.value in (None, "", 0, False, True)
                or isinstance(single.value.value, (int, float, str))
            ):
                return True
            if isinstance(
                single.value, (ast.List, ast.Dict, ast.Tuple, ast.Set)
            ) and not getattr(
                single.value, "elts", getattr(single.value, "keys", None)
            ):
                return True
    return False


def _iter_contract_entries(contract: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = list(contract.get("contracts", []))
    default_target = contract.get("target_file", "")
    for change in contract.get("changes", []) + contract.get("symbols", []):
        symbol = change.get("symbol") or change.get("name", "")
        target = change.get("target_file") or default_target
        entries.append(
            {
                "file_hint": _repo_relative(target),
                "kind": change.get("kind")
                or ("class" if symbol and symbol[0].isupper() else "function"),
                "name": symbol,
            }
        )
    return entries


def _check_spec_compliance(spec_path: str, pre_impl: bool = False) -> tuple[int, list[JsonDiag]]:
    diagnostics: list[JsonDiag] = []
    try:
        with open(spec_path) as f:
            contract = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return (
            1,
            [
                {
                    "file": spec_path,
                    "line": 0,
                    "error": f"Spec file error: {e}",
                    "fix_hint": "",
                }
            ],
        )

    for c in _iter_contract_entries(contract):
        fh: str = c.get("file_hint", "") or c.get("file", "")
        kind: str = c.get("kind", "function")
        raw_name: str = c.get("name", "") or c.get("symbol", "")
        name: str = raw_name.split()[0] if raw_name else ""
        if not fh or not name:
            continue
        if pre_impl:
            # During pre-implementation check, verify target path/parent directory validity rather than existing symbol
            parent_dir = os.path.dirname(fh)
            if parent_dir and not os.path.exists(parent_dir):
                d = {
                    "file": fh,
                    "line": 0,
                    "error": f"Spec target parent directory not found: {parent_dir}",
                    "fix_hint": f"Ensure valid directory path for {fh}",
                }
                diagnostics.append(d)
            continue
        if not os.path.exists(fh):
            d = {
                "file": fh,
                "line": 0,
                "error": f"Spec: file not found ({kind} {name})",
                "fix_hint": f"Create {fh}",
            }
            diagnostics.append(d)
            continue

        with open(fh) as sf:
            sf_content = sf.read()
            if kind == "field":
                field_name = name.split(".")[-1] if "." in name else name
                pat = rf"\b{re.escape(field_name)}[\"']?\s*(?::|=)"
                if not re.search(pat, sf_content, re.MULTILINE):
                    msg = f"Spec: {kind} '{name}' not implemented"
                    d = {
                        "file": fh,
                        "line": 0,
                        "error": msg,
                        "fix_hint": f"Implement {kind} {name} in {fh}",
                    }
                    diagnostics.append(d)
            else:
                owner, _, leaf = name.rpartition(".")
                target_node: ast.AST | None = None
                found_impl = False
                try:
                    tree = ast.parse(sf_content, filename=fh)
                    if owner:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef) and node.name == owner:
                                for member in node.body:
                                    if (
                                        isinstance(
                                            member,
                                            (ast.FunctionDef, ast.AsyncFunctionDef),
                                        )
                                        and member.name == leaf
                                    ):
                                        target_node = member
                                        found_impl = True
                    else:
                        for node in ast.walk(tree):
                            if (
                                isinstance(
                                    node,
                                    (
                                        ast.FunctionDef,
                                        ast.AsyncFunctionDef,
                                        ast.ClassDef,
                                    ),
                                )
                                and node.name == name
                            ):
                                target_node = node
                                found_impl = True
                except Exception:
                    pat = rf"^(?:class|def)\s+{re.escape(name)}\b"
                    found_impl = bool(re.search(pat, sf_content, re.MULTILINE))

                if not found_impl:
                    msg = f"Spec: {kind} '{name}' not implemented"
                    d = {
                        "file": fh,
                        "line": 0,
                        "error": msg,
                        "fix_hint": f"Implement {kind} {name} in {fh}",
                    }
                    diagnostics.append(d)
                elif target_node is not None and _is_stub_node(target_node):
                    msg = f"Spec: {kind} '{name}' is a stub implementation"
                    d = {
                        "file": fh,
                        "line": getattr(target_node, "lineno", 0),
                        "error": msg,
                        "fix_hint": f"Implement real logic in {name}",
                    }
                    diagnostics.append(d)
                else:
                    diagnostics.extend(_check_orphaned_implementations(fh, kind, name))

    if not pre_impl:
        for s in contract.get("scenarios", []):
            test_name: str = s.get("name", "") or s.get("scenario_id", "")
            if not test_name:
                continue
            if s.get("scenario_id"):
                parts = test_name.split("-")
                reference = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]
            else:
                reference = test_name

            target_test_file: str = _repo_relative(s.get("target_test_file", ""))
            found = False
            ref_pattern = re.compile(rf"\b{re.escape(reference)}\b")
            if target_test_file and os.path.exists(target_test_file):
                with open(target_test_file) as tf:
                    content = tf.read()
                found = bool(ref_pattern.search(content)) or bool(
                    re.search(
                        rf"^[ \t]*def\s+{re.escape(test_name)}\b", content, re.MULTILINE
                    )
                )
            if not found:
                for _fp, content in _get_tests_files_contents():
                    if bool(ref_pattern.search(content)) or re.search(
                        rf"^[ \t]*def\s+{re.escape(test_name)}\b", content, re.MULTILINE
                    ):
                        found = True
                        break
            if not found:
                fix_hint = (
                    f"Write a test referencing {test_name} in {target_test_file}"
                    if target_test_file
                    else f"Write {test_name}"
                )
                d = {
                    "file": target_test_file,
                    "line": 0,
                    "error": f"Spec: missing test '{test_name}'",
                    "fix_hint": fix_hint,
                }
                diagnostics.append(d)

    wirings: list[dict[str, Any]] = []
    if "wiring" in contract and isinstance(contract["wiring"], list):
        wirings.extend(contract["wiring"])
    elif "wiring" in contract and isinstance(contract["wiring"], dict):
        wirings.append(contract["wiring"])
    wirings.extend(
        c["wiring"]
        for c in contract.get("contracts", [])
        if "wiring" in c and isinstance(c["wiring"], dict)
    )

    if not wirings:
        diagnostics.append(
            {
                "file": spec_path,
                "line": 0,
                "error": "Spec: contract.json missing mandatory 'wiring' section",
                "fix_hint": "Add 'wiring' to contract.json",
            }
        )

    for w in wirings:
        wf: str = _repo_relative(
            w.get("file", "") or w.get("target", "") or w.get("caller_file", "")
        )
        anchor: str = w.get("anchor", "")
        import_symbol: str = w.get("import_symbol", "") or w.get("callee", "") or w.get("symbol", "")
        invocation_expr: str = w.get("invocation_expression", "") or w.get("invocation_symbol", "")
        if not wf or not os.path.exists(wf):
            if wf:
                diagnostics.append(
                    {
                        "file": wf,
                        "line": 0,
                        "error": f"Spec wiring target file not found: {wf}",
                        "fix_hint": f"Create {wf}",
                    }
                )
            continue
        with open(wf) as f:
            wf_content = f.read()
            if anchor:
                found_anchor = anchor in wf_content or any(
                    t in wf_content
                    for t in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", anchor)
                    if t not in ("step", "main", "when")
                )
                if not found_anchor:
                    diagnostics.append(
                        {
                            "file": wf,
                            "line": 0,
                            "error": f"Spec wiring: missing anchor '{anchor}'",
                            "fix_hint": f"Add ref to {anchor} in {wf}",
                        }
                    )
            if not pre_impl:
                if import_symbol and import_symbol not in wf_content:
                    diagnostics.append(
                        {
                            "file": wf,
                            "line": 0,
                            "error": f"Spec wiring: missing reference to '{import_symbol}'",
                            "fix_hint": f"Import {import_symbol} in {wf}",
                        }
                    )
                if invocation_expr and invocation_expr not in wf_content:
                    diagnostics.append(
                        {
                            "file": wf,
                            "line": 0,
                            "error": f"Spec wiring: missing invocation of '{invocation_expr}'",
                            "fix_hint": f"Invoke {invocation_expr} in {wf}",
                        }
                    )

    return (1 if diagnostics else 0, diagnostics)


def _find_test_files(py_files: list[str], impact_level: int = 1) -> list[str]:
    test_files = [f for f in py_files if f.startswith("tests/") or "test_" in f]
    source_files = [f for f in py_files if not (f.startswith("tests/") or "test_" in f)]
    repository_files = _repository_test_files()
    for sf in source_files:
        if sf.startswith("src/") and not sf.endswith("__init__.py"):
            parts = sf.split("/")
            module_name = parts[-1]
            test_name = f"test_{module_name}"
            found_direct = False
            for category in ["unit", "integration", "e2e", "contract"]:
                sub_path = "/".join(parts[1:-1])
                td = f"tests/{category}/{sub_path}" if sub_path else f"tests/{category}"
                tp = f"{td}/{test_name}"
                if tp in test_files:
                    found_direct = True
                    break
                if os.path.exists(tp):
                    test_files.append(tp)
                    found_direct = True
                    break
            # Wider AST reverse lookup is only used if NO direct test exists for this module
            if not found_direct:
                for tp in repository_files:
                    if tp not in test_files and _test_references_source(tp, sf):
                        test_files.append(tp)
    return test_files


def _analyze_impact_level(py_files: list[str]) -> tuple[int, str]:
    """Analyze change scope and return (impact_level, reason)."""
    if not py_files:
        return (1, "No python files modified")

    core_keywords = ("config", "base", "core", "schema", "contract")
    is_core_modified = any(
        any(kw in f.lower() for kw in core_keywords) for f in py_files
    )
    if is_core_modified or len(py_files) >= 5:
        return (3, "Core module or large multi-file change detected")

    src_files = [f for f in py_files if f.startswith("src/")]
    if not src_files:
        return (1, "Only test or tool files modified")

    # Check if modified source files are heavily referenced across src/
    ref_count = 0
    src_contents = _get_src_files_contents()
    for sf in src_files:
        leaf_name = os.path.splitext(os.path.basename(sf))[0]
        if leaf_name == "__init__":
            continue
        pat = re.compile(rf"\b{re.escape(leaf_name)}\b")
        for fp, content in src_contents:
            if _repo_relative(fp) != sf and pat.search(content):
                ref_count += 1

    if ref_count > 3:
        return (
            2,
            f"Module imported across multiple components ({ref_count} references)",
        )
    return (1, "Leaf/isolated module change")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Selective Lean Check with JSON diagnostics."
    )
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--spec", default=None, help="Path to spec contract JSON")
    parser.add_argument("--skip-lint", action="store_true", help="Skip Ruff linting")
    parser.add_argument(
        "--skip-mypy", action="store_true", help="Skip Mypy static check"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip pytest and run fast static checks only",
    )
    parser.add_argument(
        "--smart",
        action="store_true",
        default=True,
        help="Enable Smart Selective Verification (Impact Level targeting)",
    )
    parser.add_argument(
        "--spec-only", action="store_true", help="Run ONLY spec-compliance and exit"
    )
    parser.add_argument(
        "--pre-impl",
        action="store_true",
        help="Run spec-compliance in pre-implementation validation mode (validates schema, paths, and anchors only)",
    )
    parser.add_argument(
        "--deselect", nargs="*", default=[], help="Pytest node ids to deselect"
    )
    parser.add_argument(
        "--pytest-timeout", type=int, default=None, help="Seconds for pytest step"
    )
    args = parser.parse_args()

    if args.spec_only or args.pre_impl:
        if not args.spec:
            print("FAIL | --spec-only and --pre-impl require --spec")
            sys.exit(2)
        ec, diags = _check_spec_compliance(args.spec, pre_impl=args.pre_impl)
        if ec != 0:
            _fail_exit_many(
                "spec-compliance",
                f"FAIL | Spec compliance failed with {len(diags)} error(s)",
                diags,
            )
        print("PASS | Spec compliance verified" + (" (pre-impl)" if args.pre_impl else ""))
        print(_emit_json("PASS", "spec-compliance", []), file=sys.stderr)
        sys.exit(0)

    if not args.files:
        try:
            diff_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            git_files = [
                line[3:].strip()
                for line in diff_res.stdout.splitlines()
                if "D" not in line[:2]
                and line[3:].strip().endswith(".py")
                and not line[3:].strip().startswith("tools/")
                and not line[3:].strip().endswith("conftest.py")
                and os.path.exists(line[3:].strip())
            ]
            args.files = git_files
        except Exception:
            args.files = []

    if not args.spec and os.path.exists("docs/specs"):
        spec_candidates = [
            os.path.join("docs/specs", f)
            for f in os.listdir("docs/specs")
            if f.endswith("_contract.json") or f == "contract.json"
        ]
        if spec_candidates:
            spec_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            args.spec = spec_candidates[0]

    py_files = [f for f in args.files if f.endswith(".py")]
    if not py_files and not args.spec:
        print("ALLCHECKS:PASS | No modified .py files detected")
        sys.exit(0)

    # 1. Co-modification Check & Test Discovery
    impact_level, impact_reason = _analyze_impact_level(py_files)
    print(f"INFO | Impact Level: {impact_level} ({impact_reason})")
    test_files = _find_test_files(py_files, impact_level=impact_level)

    for pf in py_files:
        if (
            pf.startswith("src/")
            and not pf.endswith("__init__.py")
            and not pf.startswith("tools/")
        ):
            parts = pf.split("/")
            test_name = f"test_{parts[-1]}"
            has_test = any(test_name in tf for tf in test_files)
            if not has_test:
                d = {
                    "file": pf,
                    "line": 0,
                    "error": f"No matching test for {pf}",
                    "fix_hint": f"Create test for {pf}",
                }
                _fail_exit("co-modification", f"FAIL | {pf}: test file missing", d)

    # 2. Parallel Static Checks (Spec, Print-check, Ruff, Mypy)
    def check_spec_task() -> tuple[str, int, list[JsonDiag], str]:
        if not args.spec:
            return ("spec-compliance", 0, [], "")
        ec, diags = _check_spec_compliance(args.spec)
        return ("spec-compliance", ec, diags, f"FAIL | Spec compliance failed with {len(diags)} error(s)")

    def check_print_task() -> tuple[str, int, list[JsonDiag], str]:
        if args.skip_lint:
            return ("print-check", 0, [], "")
        print_re = re.compile(r"(?<!#)\bprint\s*\(")
        for pf in py_files:
            if pf.startswith("tools/"):
                continue
            with open(pf, encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if print_re.search(line):
                        d = {
                            "file": pf,
                            "line": idx,
                            "error": "Unsanctioned print() detected",
                            "fix_hint": "Use logging module instead of print()",
                        }
                        return ("print-check", 1, [d], f"FAIL | {pf}:{idx} print() detected")
        return ("print-check", 0, [], "")

    def check_ruff_task() -> tuple[str, int, list[JsonDiag], str]:
        if args.skip_lint or not py_files:
            return ("ruff", 0, [], "")
        ruff_res = run_cmd(["uv", "run", "ruff", "check", *py_files, "--quiet"])
        if ruff_res.returncode != 0:
            out_sliced = "\n".join(
                (ruff_res.stdout or ruff_res.stderr).strip().splitlines()[:10]
            )
            d = {
                "file": py_files[0],
                "line": 0,
                "error": out_sliced,
                "fix_hint": "Fix ruff lint errors",
            }
            return ("ruff", 1, [d], "FAIL | Ruff Lint Failed")
        return ("ruff", 0, [], "")

    def check_mypy_task() -> tuple[str, int, list[JsonDiag], str]:
        if args.skip_mypy or not py_files:
            return ("mypy", 0, [], "")
        mypy_res = run_cmd(["uv", "run", "mypy", *py_files, "--ignore-missing-imports"])
        if mypy_res.returncode != 0:
            out_sliced = "\n".join(
                (mypy_res.stdout or mypy_res.stderr).strip().splitlines()[:10]
            )
            d = {
                "file": py_files[0],
                "line": 0,
                "error": out_sliced,
                "fix_hint": "Fix mypy type errors",
            }
            return ("mypy", 1, [d], "FAIL | Mypy Type Check Failed")
        return ("mypy", 0, [], "")

    with ThreadPoolExecutor(max_workers=4) as executor:
        f_spec = executor.submit(check_spec_task)
        f_print = executor.submit(check_print_task)
        f_ruff = executor.submit(check_ruff_task)
        f_mypy = executor.submit(check_mypy_task)

        # Collect results
        tasks = [f_spec, f_print, f_ruff, f_mypy]
        for f in tasks:
            phase, code, diags, msg = f.result()
            if code != 0:
                if len(diags) > 1:
                    _fail_exit_many(phase, msg, diags)
                else:
                    _fail_exit(phase, msg, diags[0] if diags else {})

    if args.spec:
        print("PASS | Spec compliance verified")

    if args.fast:
        print("PASS | Fast Check Passed (Spec, Mapping, Print, Ruff, Mypy verified)")
        print(_emit_json("PASS", "fast-check", [], None), file=sys.stderr)
        return

    # 5. Pytest
    if not test_files:
        print("PASS | Lint & Type check passed (no tests to run)")
        print(_emit_json("PASS", "all", [], None), file=sys.stderr)
        return

    deselect_args = [f"--deselect={node}" for node in args.deselect]
    core_cmd = [
        "uv",
        "run",
        "pytest",
        "-m",
        "not slow",
        *test_files,
        *deselect_args,
        "-q",
        "--tb=line",
    ]
    pytest_timeout = args.pytest_timeout or max(300, min(1200, 240 * len(test_files)))
    pt_res = run_cmd(core_cmd, timeout=pytest_timeout)

    if pt_res.returncode == 0:
        print("PASS | All checks passed (Lint, Type, Tests verified)")
        print(_emit_json("PASS", "all", [], None), file=sys.stderr)
    else:
        last_err = [
            line
            for line in (pt_res.stdout or "").splitlines()
            if any(x in line for x in ("FAIL", "Error", "AssertionError"))
        ]
        cause = (
            last_err[-1]
            if last_err
            else (pt_res.stderr or "Check pytest output.").strip()
        )
        cause_sliced = "\n".join(cause.splitlines()[:10])
        d = {
            "file": "",
            "line": 0,
            "error": cause_sliced,
            "fix_hint": "Fix failing pytest assertions",
        }
        _fail_exit("pytest", f"FAIL | Pytest Failed: {cause_sliced}", d)


if __name__ == "__main__":
    main()
