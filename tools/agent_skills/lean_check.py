#!/usr/bin/env python3
"""Smart Selective Lean Check: Fast, token-efficient mechanical audit gate."""

from __future__ import annotations

import argparse
import contextlib
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


def _exit_with_diags(phase: str, header: str, diags: list[JsonDiag], exit_code: int = 1) -> None:
    print(header)
    for d in diags:
        err = d.get("error", "")
        if err:
            print(f"FAIL | {err}")
    print(_emit_json("FAIL", phase, diags), file=sys.stderr)
    sys.exit(exit_code)


def run_cmd(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    # Strip unnecessary 'uv run' prefix when already running inside virtualenv
    if len(cmd) >= 3 and cmd[0] == "uv" and cmd[1] == "run" and os.environ.get("VIRTUAL_ENV"):
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


def _available_memory_gb() -> float:
    """Return available system RAM in gigabytes using Linux procfs or sysconf."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        pass
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")) / (1024**3)
    except (ValueError, OSError, AttributeError):
        return 8.0


# ---------------------------------------------------------------------------
# Scaffolding Leak Guard: Block temporary spec/recipe text from production code
# ---------------------------------------------------------------------------

_SCAFFOLDING_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\[(?:STEP-BY-STEP\s+)?RECIPE", re.IGNORECASE),
        "Recipe directive leaked into code/docstring",
    ),
    (
        re.compile(r"\[ALGORITHM\s+RECIPE\]", re.IGNORECASE),
        "Algorithm recipe placeholder leaked into code",
    ),
    (
        re.compile(r"^\s*(?:#|/{2})?\s*Step\s+\d+\.\s+[A-Z]", re.MULTILINE),
        "Spec Step-by-step numbering leaked into code/comment",
    ),
    (
        re.compile(r"\b(?:TODO|FIXME)\b\s*:", re.IGNORECASE),
        "TODO/FIXME placeholder found in modified code",
    ),
)


def _check_scaffolding_leaks(py_files: list[str]) -> list[JsonDiag]:
    """Verify that no temporary spec recipes or placeholders remain in production code."""
    diags: list[JsonDiag] = []
    src_files = [f for f in py_files if (f.startswith("src/") or "/src/" in f) and os.path.isfile(f)]

    for fpath in src_files:
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            continue

        for idx, line in enumerate(lines, start=1):
            for pat, desc in _SCAFFOLDING_PATTERNS:
                if pat.search(line):
                    diags.append(
                        {
                            "file": fpath,
                            "line": idx,
                            "error": f"Scaffolding Leak: {desc} -> '{line.strip()}'",
                            "fix_hint": "Remove temporary spec/recipe directives and write clean production code/docstring",
                        }
                    )
                    break
    return diags


# ---------------------------------------------------------------------------
# Direct Test Matching (Predictable, zero-cascade convention mapping)
# ---------------------------------------------------------------------------


def _find_test_files(py_files: list[str], spec_path: str | None = None) -> list[str]:
    """Find direct unit tests corresponding to modified source files."""
    test_files = [f for f in py_files if f.startswith("tests/") or "test_" in f]
    source_files = [f for f in py_files if f.startswith("src/") and not f.endswith("__init__.py")]

    # 1. Direct path convention: src/path/module.py -> tests/unit/path/test_module.py
    for sf in source_files:
        rel = sf[4:]  # strip 'src/'
        parts = rel.split("/")
        mod_name = parts[-1]
        test_name = f"test_{mod_name}"
        sub_path = "/".join(parts[:-1])

        candidates = [
            f"tests/unit/{sub_path}/{test_name}" if sub_path else f"tests/unit/{test_name}",
            f"tests/unit/{test_name}",
            f"tests/contract/{sub_path}/{test_name}" if sub_path else f"tests/contract/{test_name}",
        ]
        for cand in candidates:
            if cand in test_files:
                break
            if os.path.isfile(cand):
                test_files.append(cand)
                break

    # 2. Spec test suites if provided
    if spec_path and os.path.isfile(spec_path):
        with contextlib.suppress(OSError):
            with open(spec_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            matches = re.findall(
                r"(?m)^##\s+(?:Test\s+Suite|Invariant\s+Scenarios):\s*`?([^\n`]+)`?",
                content,
            )
            for m in matches:
                tf = m.strip().strip("`").strip()
                if os.path.isfile(tf) and tf not in test_files:
                    test_files.append(tf)

    return sorted(dict.fromkeys(test_files))


def _check_pre_impl_spec(spec_path: str) -> tuple[int, list[JsonDiag]]:
    """Validate spec blueprint paths, targets, caller files, and anchors before implementation."""
    diags: list[JsonDiag] = []
    if not os.path.isfile(spec_path):
        return 1, [
            {
                "file": spec_path,
                "line": 0,
                "error": f"Spec file not found: {spec_path}",
                "fix_hint": "Check spec file path",
            }
        ]

    try:
        with open(spec_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError as e:
        return 1, [
            {
                "file": spec_path,
                "line": 0,
                "error": f"Cannot read spec file: {e}",
                "fix_hint": "Check file permissions",
            }
        ]

    current_caller: str | None = None
    target_found = False

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Check Target
        m_target = re.match(r"^##\s+Target:\s*`?([^\n`]+)`?", stripped)
        if m_target:
            target_file = m_target.group(1).strip().strip("`").strip()
            target_found = True
            parent = os.path.dirname(target_file)
            if parent and not os.path.isdir(parent):
                diags.append(
                    {
                        "file": spec_path,
                        "line": idx,
                        "error": f"Target parent directory does not exist: '{parent}' for target '{target_file}'",
                        "fix_hint": f"Create directory {parent} or fix path in spec",
                    }
                )

        # Check Wiring caller file
        m_wiring = re.match(r"^##\s+Wiring:\s*`?([^\n`]+)`?", stripped)
        if m_wiring:
            current_caller = m_wiring.group(1).strip().strip("`").strip()
            if not os.path.isfile(current_caller):
                diags.append(
                    {
                        "file": spec_path,
                        "line": idx,
                        "error": f"Wiring caller file does not exist: '{current_caller}'",
                        "fix_hint": f"Verify caller file path in {spec_path}",
                    }
                )

        # Check Anchor in caller file
        m_anchor = re.match(r"^-\s*(?:Anchor|anchor):\s*`?([^\n`]+)`?", stripped)
        if m_anchor and current_caller and os.path.isfile(current_caller):
            anchor = m_anchor.group(1).strip().strip("`").strip()
            try:
                with open(current_caller, encoding="utf-8", errors="ignore") as cf:
                    caller_content = cf.read()
                if anchor not in caller_content:
                    diags.append(
                        {
                            "file": current_caller,
                            "line": 0,
                            "error": f"Wiring anchor '{anchor}' not found in caller file '{current_caller}'",
                            "fix_hint": f"Ensure anchor '{anchor}' matches an existing symbol or line in {current_caller}",
                        }
                    )
            except OSError:
                pass

        # Check Invariant Scenarios / Test Suite test file
        m_test = re.match(r"^##\s+(?:Invariant\s+Scenarios|Test\s+Suite):\s*`?([^\n`]+)`?", stripped)
        if m_test:
            test_file = m_test.group(1).strip().strip("`").strip()
            parent = os.path.dirname(test_file)
            if parent and not os.path.isdir(parent):
                diags.append(
                    {
                        "file": spec_path,
                        "line": idx,
                        "error": f"Test suite directory does not exist: '{parent}' for '{test_file}'",
                        "fix_hint": f"Create directory {parent} or fix path in spec",
                    }
                )

    if not target_found:
        diags.append(
            {
                "file": spec_path,
                "line": 0,
                "error": "Spec missing mandatory '## Target: <path>' section",
                "fix_hint": "Add '## Target: <relative_path>' section to spec",
            }
        )

    return (1 if diags else 0), diags


# ---------------------------------------------------------------------------
# Diff Coverage Gate
# ---------------------------------------------------------------------------


def _git_diff_added_lines(file: str) -> set[int] | None:
    """1-indexed line numbers this working-tree diff adds to `file`."""
    status_res = subprocess.run(  # noqa: S603
        ["git", "status", "--porcelain", "--", file],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if status_res.stdout.strip().startswith("??"):
        return None
    diff_res = subprocess.run(  # noqa: S603
        ["git", "diff", "--unified=0", "HEAD", "--", file],
        capture_output=True,
        text=True,
        timeout=10,
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


def _check_diff_coverage(src_files: list[str], cov_json_path: str) -> tuple[list[JsonDiag], int | None]:
    """Verify that every line added to touched src/ files is executed by tests."""
    if not os.path.exists(cov_json_path):
        return [], None
    try:
        with open(cov_json_path, encoding="utf-8") as f:
            cov_data = json.load(f)
    except Exception:
        return [], None

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
        added &= executed | missing
        if not added:
            continue
        total_added += len(added)
        total_covered += len(added - missing)
        uncovered_new = sorted(added & missing)
        if uncovered_new:
            shown = uncovered_new[:10]
            diags.append(
                {
                    "file": sf,
                    "line": shown[0],
                    "error": f"{len(uncovered_new)} newly-added line(s) not executed by tests: {shown}",
                    "fix_hint": "Add or update tests exercising these lines",
                }
            )
    pct = round(100 * total_covered / total_added) if total_added else None
    return diags, pct


# ---------------------------------------------------------------------------
# Main CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Selective Lean Check: Tier 1 Mechanical Gate.")
    parser.add_argument("--files", nargs="*", default=[], help="Explicit files to check")
    parser.add_argument("--spec", default=None, help="Path to markdown/JSON spec file")
    parser.add_argument("--fast", action="store_true", help="Run static checks only (scaffolding, ruff, mypy)")
    parser.add_argument("--skip-lint", action="store_true", help="Skip Ruff linting")
    parser.add_argument("--skip-mypy", action="store_true", help="Skip Mypy static check")
    parser.add_argument("--no-cov", action="store_true", help="Disable diff-coverage gate")
    parser.add_argument("--no-xdist", action="store_true", help="Force serial pytest execution (-n 0)")
    parser.add_argument("--timeout", type=int, default=None, help="Pytest timeout in seconds")
    parser.add_argument(
        "--pre-impl",
        action="store_true",
        help="Validate spec blueprint paths and wiring anchors before implementation",
    )
    args = parser.parse_args()

    # 0. Pre-implementation Spec Blueprint Gate
    if args.pre_impl:
        if not args.spec:
            _exit_with_diags(
                "pre-impl",
                "FAIL | --pre-impl requires --spec <spec_file>",
                [
                    {
                        "file": "",
                        "line": 0,
                        "error": "--pre-impl requires --spec argument",
                        "fix_hint": "Pass --spec docs/specs/<feature>_spec.md",
                    }
                ],
            )
        code, diags = _check_pre_impl_spec(args.spec)
        if code != 0:
            _exit_with_diags(
                "pre-impl",
                f"FAIL | Pre-impl spec validation failed ({len(diags)} error(s))",
                diags,
            )
        print("PASS | Spec blueprint paths and wiring anchors verified (pre-impl)")
        print(_emit_json("PASS", "pre-impl", []), file=sys.stderr)
        return

    # 1. File discovery from git if not explicitly passed
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

    py_files = [f for f in args.files if f.endswith(".py")]
    if not py_files:
        print("ALLCHECKS:PASS | No modified .py files detected")
        sys.exit(0)

    # 2. Scaffolding Leak Guard
    scaffolding_diags = _check_scaffolding_leaks(py_files)
    if scaffolding_diags:
        _exit_with_diags(
            "scaffolding-guard",
            f"FAIL | Scaffolding Leak: {len(scaffolding_diags)} temporary spec/recipe artifact(s) found in code",
            scaffolding_diags,
        )

    # 3. Parallel Static Checks (Ruff, Mypy)
    def check_ruff() -> tuple[str, int, list[JsonDiag], str]:
        if args.skip_lint or not py_files:
            return "ruff", 0, [], ""
        res = run_cmd(["uv", "run", "ruff", "check", *py_files, "--quiet"])
        if res.returncode != 0:
            out = "\n".join((res.stdout or res.stderr).strip().splitlines()[:10])
            return "ruff", 1, [{"file": py_files[0], "line": 0, "error": out, "fix_hint": "Fix ruff lint errors"}], "FAIL | Ruff Lint Failed"
        return "ruff", 0, [], ""

    def check_mypy() -> tuple[str, int, list[JsonDiag], str]:
        if args.skip_mypy or not py_files:
            return "mypy", 0, [], ""
        target_mypy = [f for f in py_files if f.startswith("src/")] or py_files
        res = run_cmd(["uv", "run", "mypy", *target_mypy, "--ignore-missing-imports"])
        if res.returncode != 0:
            out = "\n".join((res.stdout or res.stderr).strip().splitlines()[:10])
            return "mypy", 1, [{"file": target_mypy[0], "line": 0, "error": out, "fix_hint": "Fix mypy type errors"}], "FAIL | Mypy Type Check Failed"
        return "mypy", 0, [], ""

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_ruff = executor.submit(check_ruff)
        f_mypy = executor.submit(check_mypy)
        for f in [f_ruff, f_mypy]:
            phase, code, diags, msg = f.result()
            if code != 0:
                _exit_with_diags(phase, msg, diags)

    if args.fast:
        print("PASS | Fast Check Passed (Scaffolding, Ruff, Mypy verified)")
        print(_emit_json("PASS", "fast-check", [], None), file=sys.stderr)
        return

    # 4. Direct Test Discovery
    test_files = _find_test_files(py_files, spec_path=args.spec)
    if not test_files:
        print("PASS | Lint & Type check passed (no tests to run)")
        print(_emit_json("PASS", "all", [], None), file=sys.stderr)
        return

    # 5. Smart Pytest Execution (Resource Safety Guard)
    env_workers = os.environ.get("LEAN_CHECK_WORKERS")
    avail_mem_gb = _available_memory_gb()

    if (
        args.no_xdist
        or len(test_files) <= 5
        or (env_workers and env_workers in ("0", "1"))
        or avail_mem_gb < 2.0
    ):
        xdist_args = ["-p", "no:cacheprovider", "-n", "0"]
    else:
        target_workers = int(env_workers) if env_workers and env_workers.isdigit() else 2
        worker_count = min(target_workers, os.cpu_count() or 2, len(test_files))
        xdist_args = ["-p", "no:cacheprovider", "-n", str(worker_count)]

    src_files = [f for f in py_files if f.startswith("src/")]
    cov_json_path = "tmp/lean_check_coverage.json"
    cov_args: list[str] = []

    if src_files and not args.no_cov:
        os.makedirs("tmp", exist_ok=True)
        with contextlib.suppress(OSError):
            os.remove(cov_json_path)
        pkgs = {f.split("/")[1] for f in src_files if len(f.split("/")) >= 2}
        cov_pkgs = [f"--cov=src/{p}" for p in sorted(pkgs)] if pkgs else ["--cov=src"]
        cov_args = [*cov_pkgs, f"--cov-report=json:{cov_json_path}"]

    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not slow",
        *test_files,
        *xdist_args,
        *cov_args,
        "-q",
        "--tb=line",
    ]
    pytest_timeout = args.timeout or max(60, min(240, 20 * len(test_files)))
    pt_res = run_cmd(pytest_cmd, timeout=pytest_timeout)

    if pt_res.returncode == 124:
        _exit_with_diags(
            "pytest-timeout",
            f"FAIL | Pytest Timed Out ({pytest_timeout}s)",
            [{
                "file": "",
                "line": 0,
                "error": f"pytest timed out after {pytest_timeout}s across {len(test_files)} file(s).",
                "fix_hint": "Use --files to scope checks, investigate slow tests, or pass --timeout with a larger value.",
            }],
        )

    if pt_res.returncode == 0:
        cov_diags, cov_pct = _check_diff_coverage(src_files, cov_json_path) if cov_args else ([], None)
        if cov_diags:
            _exit_with_diags(
                "coverage",
                f"FAIL | Diff Coverage: {len(cov_diags)} file(s) with untested new lines",
                cov_diags,
            )
        cov_suffix = f", Diff-Coverage {cov_pct}%" if cov_pct is not None else ""
        print(f"PASS | All checks passed (Scaffolding-Clean, Lint, Type, Tests{cov_suffix})")
        print(_emit_json("PASS", "all", [], cov_pct), file=sys.stderr)
    else:
        last_err = [
            line
            for line in (pt_res.stdout or "").splitlines()
            if any(x in line for x in ("FAIL", "Error", "AssertionError"))
        ]
        cause = last_err[-1] if last_err else (pt_res.stderr or "Check pytest output.").strip()
        cause_sliced = "\n".join(cause.splitlines()[:10])
        _exit_with_diags(
            "pytest",
            f"FAIL | Pytest Failed: {cause_sliced}",
            [{"file": "", "line": 0, "error": cause_sliced, "fix_hint": "Fix failing pytest assertions"}],
        )


if __name__ == "__main__":
    main()
