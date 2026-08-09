---
trigger:
  - on_file_path_regex: "src/.*\\.py"
  - on_file_path_regex: "tests/.*\\.py"
priority: 9
---

# Python Architecture & Coding Standards

## 1. Environment & Package Execution
- **Environment Manager:** Project relies on `uv` for dependencies and environment.
- **Execution Rule:** All linting, type checking, and testing MUST run with `uv run` prefix (`uv run ruff check .`, `uv run mypy .`, `uv run pytest`).
- **Dependencies:** Inspect `pyproject.toml` before adding external dependencies (`uv add [package_name]`).

## 2. Python Standards
- **Typing Standard:** Preserve the repository's existing typing level; strengthen type annotations in changed interfaces.
- **Modern Syntax:** Use modern Python 3.11+ syntax (`asyncio.TaskGroup`, `|` union type, `Self`, etc.) when it improves code clarity or correctness.
- **Configuration:** Manage settings via environment variables (`.env`) and `pydantic-settings`.
- **Directory Isolation**: All production logic must reside in `src/` and test suites in `tests/`.

## 3. Structural Design & Scope Hygiene
- **Modularity:** Target cohesive modules. Split when distinct architectural layers (e.g. DB, business logic, DTO mapping) become mixed.
- **Single Source of Truth:** Code contracts (`Protocol`, `dataclass`, `Pydantic`) strictly supersede external doc markdown files.
- **Scope Control & Dead Code:** Remove dead code only when directly related to the current task and behavior-preserving. Avoid sweeping refactoring outside modified scope.
- **Constants:** Extract domain or configuration constants, not obvious local literals.
