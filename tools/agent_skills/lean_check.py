#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    env["POLARS_MAX_THREADS"] = "2"
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
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
def _get_src_cache() -> tuple[tuple[str, str, frozenset[str]], ...]:
    results: list[tuple[str, str, frozenset[str]]] = []
    if os.path.exists("src"):
        for root, dirs, files in os.walk("src"):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn_name in files:
                if fn_name.endswith(".py"):
                    fp = os.path.join(root, fn_name)
                    try:
                        with open(fp, encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                            tokens = frozenset(re.findall(r"[A-Za-z_]\w*", text))
                            results.append((fp, text, tokens))
                    except OSError:
                        continue
    return tuple(results)


def _get_src_files_contents() -> tuple[tuple[str, str], ...]:
    return tuple((fp, text) for fp, text, _ in _get_src_cache())


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


def _is_dead_anchor(node: ast.stmt) -> bool:
    """True for `_ = <anything>` assignments (single Name target with id '_')."""
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_"
    )


def _first_identifier(expression: str) -> str:
    match = re.match(r"\s*([A-Za-z_]\w*)", expression)
    return match.group(1) if match else ""


def _invocation_identifier(expression: str) -> str:
    """Resolve the called/referenced symbol of an invocation expression.

    Handles `ok = func(...)` (assignment-wrapped calls), `TABLE[key](ctx)`
    (subscript-called tables), `TABLE[key]` (subscript reads) and plain
    `from m import name` statements, falling back to the first identifier.
    """
    try:
        tree = ast.parse(expression, mode="exec")
    except SyntaxError:
        return _first_identifier(expression)
    if not tree.body:
        return _first_identifier(expression)
    node = tree.body[0]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        if node.names:
            return node.names[0].name.split(".")[0]
        return _first_identifier(expression)
    target: ast.AST = node
    if isinstance(node, (ast.Assign, ast.Expr, ast.Return)) and getattr(node, "value", None) is not None:
        target = node.value  # type: ignore[assignment]
    while isinstance(target, ast.Call):
        target = target.func
    while isinstance(target, (ast.Subscript, ast.Starred)):
        target = target.value
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return _first_identifier(expression)


def verify_wiring(
    caller_file: Path | str, import_symbol: str, invocation_expression: str
) -> tuple[bool, str]:
    """AST-based wiring check replacing the legacy substring containment test.

    PASS requires BOTH: (a) import_symbol appears as an ast.Import /
    ast.ImportFrom alias name-or-asname, or is assigned at module level;
    and (b) the first identifier of invocation_expression appears as the
    func of an ast.Call node, or as an ast.Attribute / ast.Name inside a
    non-dead-anchor statement. A name occurring only inside an ast.Constant
    string, or only as the value of a `_ = ...` assignment, does NOT count.
    """
    path = Path(caller_file)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return (False, f"cannot read {caller_file}: {exc}")
    try:
        tree = ast.parse(source, filename=str(caller_file))
    except SyntaxError as exc:
        return (False, f"cannot parse {caller_file}: {exc}")

    norm_sym = (import_symbol or "").strip().lower()
    is_sentinel = not norm_sym or norm_sym in ("(none)", "none", "n/a") or "none" in norm_sym

    if not is_sentinel:
        if re.match(r"^[A-Za-z_]\w*$", import_symbol):
            has_import = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(
                        alias.name.split(".")[0] == import_symbol or alias.asname == import_symbol
                        for alias in node.names
                    ):
                        has_import = True
                        break
                elif isinstance(node, ast.ImportFrom) and any(
                    alias.name == import_symbol or alias.asname == import_symbol for alias in node.names
                ):
                    has_import = True
                    break
            if not has_import:
                for stmt in tree.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if stmt.name == import_symbol:
                            has_import = True
                            break
                        continue
                    targets: list[ast.expr] = []
                    if isinstance(stmt, ast.Assign):
                        targets = list(stmt.targets)
                    elif isinstance(stmt, ast.AnnAssign):
                        targets = [stmt.target]
                    if any(isinstance(t, ast.Name) and t.id == import_symbol for t in targets):
                        has_import = True
                        break
            if not has_import:
                return (False, f"{import_symbol} is never imported (or module-level assigned) in {caller_file}")
        else:
            if import_symbol not in source:
                return (False, f"missing reference to {import_symbol} in {caller_file}")

    if invocation_expression:
        first = _invocation_identifier(invocation_expression)
        if not first:
            return (False, f"no identifier found in invocation expression {invocation_expression!r}")

        def _live_nodes(root: ast.AST) -> list[ast.AST]:
            found: list[ast.AST] = []

            def _visit(node: ast.AST) -> None:
                if isinstance(node, ast.stmt) and _is_dead_anchor(node):
                    return
                found.append(node)
                for child in ast.iter_child_nodes(node):
                    _visit(child)

            _visit(root)
            return found

        has_invocation = False
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            for node in _live_nodes(stmt):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == first:
                        has_invocation = True
                        break
                    if isinstance(func, ast.Attribute) and func.attr == first:
                        has_invocation = True
                        break
                elif (
                    (isinstance(node, ast.Attribute) and node.attr == first)
                    or (
                        isinstance(node, ast.Name)
                        and node.id == first
                        and isinstance(node.ctx, ast.Load)
                    )
                    or (
                        isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and (node.value == invocation_expression or node.value == first)
                    )
                ):
                    has_invocation = True
                    break
            if has_invocation:
                break

        if not has_invocation and invocation_expression in source:
            has_invocation = any(
                not _is_dead_anchor(s) and invocation_expression in (ast.unparse(s) if hasattr(ast, "unparse") else source)
                for s in tree.body
            )

        if not has_invocation:
            return (False, f"{first} is never called outside dead `_ = ...` anchors in {caller_file}")

    return (True, f"ok: {import_symbol} wired in {caller_file}")


def _check_orphaned_implementations(fh: str, kind: str, name: str) -> list[JsonDiag]:
    # Exclude non-src, fields, configs, CLI flags, entrypoint functions, and internal helpers
    leaf = name.rpartition(".")[2] if "." in name else name
    if (
        kind in ("field", "cli_argument", "pydantic_computed_field", "module_reexport", "entrypoint")
        or not fh.startswith("src")
        or leaf in ("main", "run", "cli", "app", "handler", "__init__", "__call__")
        or leaf.startswith("_")
    ):
        return []
    src_cache = _get_src_cache()
    if kind == "registry_entry":
        entry_match = re.match(r"^\w+\[(['\"])(.+?)\1\]$", name)
        if entry_match is None:
            return []
        raw_key = entry_match.group(2)
        const_names = {
            m.group(1)
            for _, text, _ in src_cache
            if raw_key in text
            for m in re.finditer(
                rf"^([A-Z0-9_]+)\s*(?::\s*[^=\n]+)?=\s*['\"]{re.escape(raw_key)}['\"]",
                text,
                re.MULTILINE,
            )
        }
        target_tokens = {raw_key, *const_names}
        for fp, _, tokens in src_cache:
            if fp != fh and (target_tokens & tokens):
                return []
        return [
            {
                "file": fh,
                "line": 0,
                "error": f"Spec: {kind} '{name}' has no callers in src/ outside its own definition (orphaned implementation)",
                "fix_hint": f"Wire {name} into its caller per the spec's wiring plan -- it currently does nothing in production",
            }
        ]
    if not leaf:
        return []
    ref_pat = re.compile(rf"\b{re.escape(leaf)}\b")
    def_pat = re.compile(rf"^\s*(?:def|class)\s+{re.escape(leaf)}\b")
    for fp, text, tokens in src_cache:
        if leaf not in tokens:
            continue
        for line in text.splitlines():
            if ref_pat.search(line) and not def_pat.match(line):
                if os.path.normpath(fp) == os.path.normpath(fh) and re.match(
                    rf"^\s*{re.escape(leaf)}\s*[:=]", line
                ):
                    continue
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
            # Empty collections or literal None
            if (
                isinstance(single.value, ast.Constant)
                and single.value.value is None
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
        target = (
            change.get("target_file")
            or change.get("file_hint")
            or change.get("file")
            or default_target
        )
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
        with open(spec_path, encoding="utf-8") as f:
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
            if kind in ("field", "dataclass_field", "pydantic_computed_field"):
                field_name = name.split(".")[-1] if "." in name else name
                pat = rf"\b{re.escape(field_name)}\b"
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
                    if kind in ("constant", "type alias", "module_constant"):
                        # 모듈 수준 상수/타입 별칭(AnnAssign/Assign 타깃)를 인식한다.
                        for node in ast.walk(tree):
                            if (
                                isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and node.target.id == name
                            ):
                                found_impl = True
                                break
                            if isinstance(node, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == name
                                for t in node.targets
                            ):
                                found_impl = True
                                break
                    elif kind in ("reexport", "module_reexport"):
                        # module_reexport names may have _export suffix; strip it for lookup
                        lookup_name = name.removesuffix("_export") if name.endswith("_export") else name
                        # consider either ImportFrom or assignment (e.g. FOO = settings.FOO)
                        imported = any(
                            isinstance(node, ast.ImportFrom)
                            and any(
                                alias.name == lookup_name or alias.asname == lookup_name
                                for alias in node.names
                            )
                            for node in ast.walk(tree)
                        )
                        assigned = bool(re.search(rf"\b{re.escape(lookup_name)}\s*=", sf_content))
                        # 재수출 계약은 __all__ 등재까지 요구한다(인용 문자열 검색).
                        found_impl = (imported or assigned) and f'"{lookup_name}"' in sf_content
                    elif kind == "cli_argument":
                        found_impl = bool(
                            re.search(
                                rf"add_argument\(\s*['\"]{re.escape(name)}['\"]",
                                sf_content,
                            )
                        )
                    elif kind == "parameter_add" and owner and "." in name:
                        # parameter_add: verify the owner function exists and
                        # the leaf parameter is present in its signature.
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == owner:
                                target_node = node
                                arg_names = [a.arg for a in node.args.args]
                                found_impl = leaf in arg_names
                                break
                    elif kind == "registry_entry":
                        # 예: NAME['key'] / NAME["key"] 형태의 레지스트리 엔트리.
                        # 구조(NAME 모듈 수준 dict 할당)와 키 리터럴을 함께 확인한다.
                        # Plain registry dict name (e.g. _ALTDATA_PANELS)도 지원한다.
                        entry_match = re.match(
                            r"^(?P<owner>\w+)\[(?P<q>['\"])(?P<key>.+?)(?P=q)\]$",
                            name,
                        )
                        if entry_match is None:
                            # Plain registry dict: check assignment exists
                            found_impl = any(
                                isinstance(node, (ast.Assign, ast.AnnAssign))
                                and any(
                                    isinstance(t, ast.Name) and t.id == name
                                    for t in (
                                        node.targets
                                        if isinstance(node, ast.Assign)
                                        else [node.target]
                                    )
                                )
                                for node in ast.walk(tree)
                            )
                            # also accept via regex fallback
                            if not found_impl:
                                found_impl = bool(re.search(rf"\b{re.escape(name)}\b", sf_content))
                        else:
                            reg_owner = entry_match.group("owner")
                            key_literal = entry_match.group("key")
                            has_registry = any(
                                (
                                    isinstance(node, (ast.Assign, ast.AnnAssign))
                                    and any(
                                        isinstance(t, ast.Name) and t.id == reg_owner
                                        for t in (
                                            node.targets
                                            if isinstance(node, ast.Assign)
                                            else [node.target]
                                        )
                                    )
                                )
                                or (
                                    isinstance(
                                        node,
                                        (ast.FunctionDef, ast.AsyncFunctionDef),
                                    )
                                    and node.name == reg_owner
                                )
                                for node in ast.walk(tree)
                            )
                            if not has_registry:
                                has_registry = bool(
                                    re.search(rf"\b{re.escape(reg_owner)}\b", sf_content)
                                )
                            # 키 리터럴은 파일의 인용 스타일(' 또는 ")과 무관하게 인정.
                            has_key = bool(
                                re.search(
                                    rf"['\"]{re.escape(key_literal)}['\"]",
                                    sf_content,
                                )
                            )
                            if not has_key:
                                # 상수로 참조된 경우 (예: IDs/상수 모듈에서 정의된 변수명으로 참조)
                                for _src_fp, src_c, _ in _get_src_cache():
                                    if key_literal in src_c:
                                        const_names = re.findall(
                                            rf"^([A-Z0-9_]+)\s*(?::\s*[^=\n]+)?=\s*['\"]{re.escape(key_literal)}['\"]",
                                            src_c,
                                            re.MULTILINE,
                                        )
                                        if any(
                                            re.search(rf"\b{re.escape(c)}\b", sf_content)
                                            for c in const_names
                                        ):
                                            has_key = True
                                            break
                            found_impl = has_registry and has_key
                    elif owner:
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

    scenarios = contract.get("scenarios", []) or contract.get("tests", [])
    if pre_impl:
        for s in scenarios:
            s_id = s.get("scenario_id") or s.get("name", "")
            skeleton = s.get("test_skeleton") or s.get("code", "")
            if not skeleton or not skeleton.strip():
                diagnostics.append({
                    "file": spec_path,
                    "line": 0,
                    "error": f"Spec: scenario '{s_id}' missing mandatory executable 'test_skeleton' (or 'code')",
                    "fix_hint": f"Provide complete def {s_id}() test function in contract.json",
                })
            else:
                try:
                    tree = ast.parse(skeleton)
                    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    if not funcs:
                        diagnostics.append({
                            "file": spec_path,
                            "line": 0,
                            "error": f"Spec: 'test_skeleton' for '{s_id}' must define a valid test function",
                            "fix_hint": f"Include 'def {s_id}():' in test_skeleton",
                        })
                    elif any(_is_stub_node(fn) for fn in funcs):
                        diagnostics.append({
                            "file": spec_path,
                            "line": 0,
                            "error": f"Spec: 'test_skeleton' for '{s_id}' is a stub (pass/empty/raise NotImplementedError)",
                            "fix_hint": f"Provide complete assertions and execution body for {s_id}",
                        })
                except SyntaxError as syn_err:
                    diagnostics.append({
                        "file": spec_path,
                        "line": 0,
                        "error": f"Spec: 'test_skeleton' syntax error in '{s_id}': {syn_err}",
                        "fix_hint": f"Fix python syntax in test_skeleton for {s_id}",
                    })
    else:
        for s in scenarios:
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
            def_pattern = re.compile(rf"^[ \t]*def\s+{re.escape(test_name)}\b", re.MULTILINE)
            func_pattern = re.compile(rf"^[ \t]*def\s+[a-zA-Z0-9_]*{re.escape(reference)}[a-zA-Z0-9_]*\b", re.MULTILINE)
            if target_test_file and os.path.isfile(target_test_file):
                with open(target_test_file, encoding="utf-8", errors="ignore") as tf:
                    content = tf.read()
                found = bool(ref_pattern.search(content)) or bool(def_pattern.search(content)) or bool(func_pattern.search(content))
            elif target_test_file and os.path.isdir(target_test_file):
                for root, _dirs, fnames in os.walk(target_test_file):
                    for fn in fnames:
                        if fn.endswith(".py"):
                            fp = os.path.join(root, fn)
                            try:
                                with open(fp, encoding="utf-8", errors="ignore") as tf:
                                    content = tf.read()
                                if bool(ref_pattern.search(content)) or bool(def_pattern.search(content)) or bool(func_pattern.search(content)):
                                    found = True
                                    break
                            except OSError:
                                continue
                    if found:
                        break
            if not found:
                for _fp, content in _get_tests_files_contents():
                    if bool(ref_pattern.search(content)) or bool(def_pattern.search(content)) or bool(func_pattern.search(content)):
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
        with open(wf, encoding="utf-8", errors="ignore") as f:
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
                ok, reason = verify_wiring(wf, import_symbol, invocation_expr)
                if not ok:
                    diagnostics.append(
                        {
                            "file": wf,
                            "line": 0,
                            "error": f"Spec wiring: {reason}",
                            "fix_hint": f"Import {import_symbol} and invoke {invocation_expr} in {wf}",
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
            # Wider AST reverse lookup: if no direct test exists, or if impact_level >= 2,
            # include reverse-referencing tests up to a reasonable cap (max 5 per module)
            # to avoid explosive full-suite runs and timeout deadlocks.
            if not found_direct or impact_level >= 2:
                matched_count = 0
                for tp in repository_files:
                    if tp not in test_files and _test_references_source(tp, sf):
                        test_files.append(tp)
                        matched_count += 1
                        if found_direct and matched_count >= 5:
                            break
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
    src_cache = _get_src_cache()
    for sf in src_files:
        leaf_name = os.path.splitext(os.path.basename(sf))[0]
        if leaf_name == "__init__":
            continue
        for fp, _, tokens in src_cache:
            if _repo_relative(fp) != sf and leaf_name in tokens:
                ref_count += 1

    if ref_count > 3:
        return (
            2,
            f"Module imported across multiple components ({ref_count} references)",
        )
    return (1, "Leaf/isolated module change")


def _git_diff_added_lines(file: str) -> set[int] | None:
    """1-indexed line numbers this working-tree diff adds to `file`.

    Returns None for an untracked (new) file -- caller treats every
    coverage-measured line as "added" in that case.
    """
    status_res = subprocess.run(  # noqa: S603
        ["git", "status", "--porcelain", "--", file],
        capture_output=True, text=True, timeout=10,
    )
    if status_res.stdout.strip().startswith("??"):
        return None
    diff_res = subprocess.run(  # noqa: S603
        ["git", "diff", "--unified=0", "HEAD", "--", file],
        capture_output=True, text=True, timeout=10,
    )
    added: set[int] = set()
    cur_line = 0
    for line in diff_res.stdout.splitlines():
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            if m:
                cur_line = int(m.group(1))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.add(cur_line)
            cur_line += 1
        elif not line.startswith("-"):
            cur_line += 1
    return added


def _check_diff_coverage(
    src_files: list[str], cov_json_path: str
) -> tuple[list[JsonDiag], int | None]:
    """Fail-closed check: every line this diff *adds* to a touched src file
    must actually execute during the test run. Deliberately diff-scoped
    (not a flat % threshold) -- a magic coverage percentage can pass while
    the exact lines this change introduced sit untested; this can't.
    """
    if not os.path.exists(cov_json_path):
        return ([], None)
    try:
        with open(cov_json_path, encoding="utf-8") as f:
            cov_data = json.load(f)
    except Exception:
        return ([], None)

    files_data = cov_data.get("files", {})
    diags: list[JsonDiag] = []
    total_added = 0
    total_covered = 0
    for sf in src_files:
        entry = files_data.get(sf) or files_data.get(sf.replace("/", os.sep))
        if not entry:
            continue
        missing = set(entry.get("missing_lines", []))
        executed = set(entry.get("executed_lines", []))
        added = _git_diff_added_lines(sf)
        if added is None:
            added = executed | missing
        added &= executed | missing  # only lines coverage.py actually measured
        if not added:
            continue
        total_added += len(added)
        total_covered += len(added - missing)
        uncovered_new = sorted(added & missing)
        if uncovered_new:
            shown = uncovered_new[:10]
            diags.append({
                "file": sf,
                "line": shown[0],
                "error": f"{len(uncovered_new)} newly-added line(s) not executed by any test: {shown}",
                "fix_hint": "Add/extend a scenario test exercising these lines, or simplify if genuinely unreachable",
            })
    pct = round(100 * total_covered / total_added) if total_added else None
    return (diags, pct)


def _parse_diag_line(
    raw_output: str,
    default_file: str,
    pattern: re.Pattern[str],
    fix_hint: str,
    error_override: str | None = None,
) -> JsonDiag:
    lines = [l for l in raw_output.strip().splitlines() if l.strip()]
    out_sliced = error_override if error_override is not None else "\n".join(lines[:10])
    err_file, err_line = default_file, 0
    for l in lines:
        m = pattern.search(l)
        if m:
            err_file = m.group(1)
            err_line = int(m.group(2))
            break
    return {
        "file": err_file,
        "line": err_line,
        "error": out_sliced,
        "fix_hint": fix_hint,
    }


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
    parser.add_argument(
        "--test-timeout", type=int, default=120,
        help="Per-test wall-clock limit in seconds via pytest-timeout. 0 disables.",
    )
    parser.add_argument(
        "--no-cov", action="store_true",
        help="Disable the diff-coverage gate (every line this diff adds to a "
             "touched src/ file must execute during the test run)",
    )
    parser.add_argument(
        "--no-xdist", action="store_true",
        help="Force serial execution (-p no:cacheprovider -n0)",
    )
    args = parser.parse_args()

    if args.pre_impl:
        if not args.spec:
            print("FAIL | --pre-impl requires --spec")
            sys.exit(2)
        ec, diags = _check_spec_compliance(args.spec, pre_impl=True)
        if ec != 0:
            _fail_exit_many(
                "spec-compliance",
                f"FAIL | Spec compliance failed with {len(diags)} error(s)",
                diags,
            )
        print("PASS | Spec compliance verified (pre-impl)")
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

    # 0. Self-heal code_map.json before test_code_map.py can see it stale.
    # Regenerating here (idempotent, index-only) means a genuinely broken
    # registration surfaces as a real Tier 1 failure instead of requiring
    # `check` to special-case "only test_code_map.py failed" as a soft pass.
    if py_files:
        try:
            from tools.agent_skills import gen_code_map

            gen_code_map.main()
        except Exception as e:
            print(f"INFO | code_map self-heal skipped: {e}")

    # 1. Co-modification Check & Test Discovery
    impact_level, impact_reason = _analyze_impact_level(py_files)
    print(f"INFO | Impact Level: {impact_level} ({impact_reason})")
    test_files = _find_test_files(py_files, impact_level=impact_level)

    spec_target_files: set[str] = set()
    # Ingest target_test_file from spec contract if available
    if args.spec and os.path.isfile(args.spec):
        with contextlib.suppress(Exception):
            with open(args.spec, encoding="utf-8") as sf:
                spec_data = json.load(sf)
            for sc in spec_data.get("scenarios", []) or spec_data.get("tests", []):
                ttf = _repo_relative(sc.get("target_test_file", ""))
                if ttf and os.path.exists(ttf) and ttf not in test_files:
                    test_files.append(ttf)
            if "target_file" in spec_data:
                spec_target_files.add(_repo_relative(spec_data["target_file"]))
            for change in spec_data.get("changes", []) + spec_data.get("symbols", []) + spec_data.get("units", []):
                t_f = change.get("target_file") or change.get("file_hint") or change.get("file")
                if t_f:
                    spec_target_files.add(_repo_relative(t_f))

    for pf in py_files:
        if (
            pf.startswith("src/")
            and not pf.endswith("__init__.py")
            and not pf.startswith("tools/")
        ):
            parts = pf.split("/")
            test_name = f"test_{parts[-1]}"
            has_test = (
                any(test_name in tf for tf in test_files)
                or (pf in spec_target_files and len(test_files) > 0)
                or any(_test_references_source(tf, pf) for tf in test_files)
            )
            if not has_test:
                d = {
                    "file": pf,
                    "line": 0,
                    "error": f"No matching test for {pf}",
                    "fix_hint": f"Create test for {pf}",
                }
                _fail_exit("co-modification", f"FAIL | {pf}: test file missing", d)

    # 2. Parallel Static Checks (Spec, Ruff, Mypy)
    def check_spec_task() -> tuple[str, int, list[JsonDiag], str]:
        if not args.spec:
            return ("spec-compliance", 0, [], "")
        ec, diags = _check_spec_compliance(args.spec)
        return ("spec-compliance", ec, diags, f"FAIL | Spec compliance failed with {len(diags)} error(s)")

    def check_ruff_task() -> tuple[str, int, list[JsonDiag], str]:
        if args.skip_lint or not py_files:
            return ("ruff", 0, [], "")
        ruff_res = run_cmd(["uv", "run", "ruff", "check", *py_files, "--quiet"])
        if ruff_res.returncode != 0:
            d = _parse_diag_line(
                ruff_res.stdout or ruff_res.stderr,
                py_files[0],
                re.compile(r"^(.+?):(\d+):(?:\d+:)?"),
                "Fix ruff lint errors",
            )
            return ("ruff", 1, [d], "FAIL | Ruff Lint Failed")
        return ("ruff", 0, [], "")

    def check_mypy_task() -> tuple[str, int, list[JsonDiag], str]:
        if args.skip_mypy or not py_files:
            return ("mypy", 0, [], "")
        # Only run mypy on src files and active tests to avoid cold-start overhead
        target_mypy = [f for f in py_files if f.startswith("src/")] or py_files
        mypy_res = run_cmd(["uv", "run", "mypy", *target_mypy, "--ignore-missing-imports"])
        if mypy_res.returncode != 0:
            d = _parse_diag_line(
                mypy_res.stdout or mypy_res.stderr,
                target_mypy[0],
                re.compile(r"^(.+?):(\d+):(?:\d+:)?\s*(?:error|note):"),
                "Fix mypy type errors",
            )
            return ("mypy", 1, [d], "FAIL | Mypy Type Check Failed")
        return ("mypy", 0, [], "")

    with ThreadPoolExecutor(max_workers=3) as executor:
        f_spec = executor.submit(check_spec_task)
        f_ruff = executor.submit(check_ruff_task)
        f_mypy = executor.submit(check_mypy_task)

        # Collect results
        tasks = [f_spec, f_ruff, f_mypy]
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
        print("PASS | Fast Check Passed (Spec, Mapping, Ruff, Mypy verified)")
        print(_emit_json("PASS", "fast-check", [], None), file=sys.stderr)
        return

    # 3. Pytest
    if not test_files:
        print("PASS | Lint & Type check passed (no tests to run)")
        print(_emit_json("PASS", "all", [], None), file=sys.stderr)
        return

    deselect_args = [f"--deselect={node}" for node in args.deselect]
    timeout_args = (
        [f"--timeout={args.test_timeout}", "--timeout-method=thread"]
        if args.test_timeout > 0
        else []
    )
    xdist_args = ["-p", "no:cacheprovider", "-n", "0"] if args.no_xdist else []
    src_files = [f for f in py_files if f.startswith("src/")]
    cov_json_path = "tmp/lean_check_coverage.json"
    cov_args: list[str] = []
    if src_files and not args.no_cov:
        os.makedirs("tmp", exist_ok=True)
        # Remove any stale report from a prior run before invoking pytest-cov.
        # If pytest-cov silently fails to write a fresh one (seen with a bad
        # --cov form, or any other collection error), _check_diff_coverage
        # must see "file absent" and skip the gate -- not read old data whose
        # line numbers no longer match this file's current content.
        with contextlib.suppress(OSError):
            os.remove(cov_json_path)
        # Multiple per-module --cov flags re-import numpy via exchange_calendars
        # during conftest collection; one src root is enough for diff coverage.
        cov_args = [
            "--cov=src",
            f"--cov-report=json:{cov_json_path}",
        ]
    core_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not slow",
        *test_files,
        *deselect_args,
        *timeout_args,
        *xdist_args,
        *cov_args,
        "-q",
        "--tb=line",
    ]
    pytest_timeout = args.pytest_timeout or max(120, min(600, 120 * len(test_files)))
    pt_res = run_cmd(core_cmd, timeout=pytest_timeout)

    # Nested-fork hazard fallback
    if pt_res.returncode == 124 and not args.no_xdist:
        print(
            f"INFO | pytest timed out after {pytest_timeout}s under xdist "
            "(possible fork-in-fork deadlock); retrying serially with -n0"
        )
        serial_cmd = [*core_cmd, "-p", "no:cacheprovider", "-n", "0"]
        pt_res = run_cmd(serial_cmd, timeout=pytest_timeout)

    if pt_res.returncode == 0:
        cov_diags, cov_pct = (
            _check_diff_coverage(src_files, cov_json_path) if cov_args else ([], None)
        )
        if cov_diags:
            _fail_exit_many(
                "coverage",
                f"FAIL | Diff Coverage: {len(cov_diags)} file(s) with untested new lines",
                cov_diags,
            )
        cov_suffix = f", Diff-Coverage {cov_pct}%" if cov_pct is not None else ""
        print(f"PASS | All checks passed (Lint, Type, Tests{cov_suffix} verified)")
        print(_emit_json("PASS", "all", [], cov_pct), file=sys.stderr)
    else:
        raw_output = (pt_res.stdout or "") + "\n" + (pt_res.stderr or "")
        lines = [l for l in raw_output.splitlines() if l.strip()]
        last_err = [
            line
            for line in lines
            if any(x in line for x in ("FAIL", "Error", "AssertionError"))
        ]
        cause = (
            last_err[-1]
            if last_err
            else (pt_res.stderr or "Check pytest output.").strip()
        )
        cause_sliced = "\n".join(cause.splitlines()[:10])
        d = _parse_diag_line(
            raw_output,
            "",
            re.compile(r"(tests/[a-zA-Z0-9_\/\.]+\.py):(\d+):"),
            "Fix failing pytest assertions",
            error_override=cause_sliced,
        )
        _fail_exit("pytest", f"FAIL | Pytest Failed: {cause_sliced}", d)


if __name__ == "__main__":
    main()
