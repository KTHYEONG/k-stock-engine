"""Source hygiene: no path, host, or secret literals outside the config layer."""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
TOOLS = REPO / "tools"
SECRETS_MODULE = SRC / "config" / "secrets.py"
# Reference-only until the parser spec deletes the tool; never executed.
EXCLUDED = {REPO / "tools" / "legacy_recovery", REPO / "tools" / "agent_skills"}

_PATH_PREFIXES = ("config/", "data/", "/home/")


def _iter_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in str(path) and not any(path.is_relative_to(exc) for exc in EXCLUDED)
    )


def _read_tree(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    files.extend(_iter_files(SRC))
    files.extend(_iter_files(TOOLS))
    return files


def _string_parts(tree: ast.AST) -> list[str]:
    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            parts.extend(
                value.value
                for value in node.values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            )
    return parts


def _env_accesses(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"getenv", "environ"}:
            target = node.value
            if isinstance(target, ast.Name) and target.id == "os":
                found.append(node.attr)
    return found


def test_no_path_host_or_secret_literals() -> None:
    offenders: list[str] = []
    for path in _scanned_files():
        if path == SECRETS_MODULE:
            continue
        tree = _read_tree(path)
        if tree is None:
            continue
        offenders.extend(
            f"{path.relative_to(REPO)}: {part[:60]!r}"
            for part in _string_parts(tree)
            if part.lstrip().startswith(_PATH_PREFIXES)
        )
    assert not offenders, f"path/host literals must live in config/runtime.toml: {offenders}"


def test_no_env_access_outside_secrets_module() -> None:
    offenders: list[str] = []
    for path in _scanned_files():
        if path == SECRETS_MODULE:
            continue
        tree = _read_tree(path)
        if tree is None:
            continue
        if _env_accesses(tree):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"os environment access must live in src/config/secrets.py: {offenders}"


def test_no_secret_env_name_literals() -> None:
    from src.config import load_provider_policy, load_runtime_config

    provider = load_provider_policy(load_runtime_config())
    names = {
        provider.kis.app_key_env,
        provider.kis.app_secret_env,
        provider.kis.account_no_env,
        provider.kis.account_product_code_env,
        *provider.dart.keys,
    }
    offenders: list[str] = []
    for path in _scanned_files():
        if path == SECRETS_MODULE or (SRC / "config") in path.parents:
            continue
        tree = _read_tree(path)
        if tree is None:
            continue
        offenders.extend(
            f"{path.relative_to(REPO)}: {part!r}" for part in _string_parts(tree) if part.strip() in names
        )
    assert not offenders, f"secret env names must live in config/providers.toml: {offenders}"
