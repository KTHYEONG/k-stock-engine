"""Registry invariants for the area-based data CLI: surface, laziness, envelope, uniqueness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SNAPSHOTS = Path(__file__).parent / "snapshots"


def _registered_names() -> list[str]:
    from src.data.cli import _ensure_registered
    from src.data.cli.registry import commands

    _ensure_registered()
    return [command.name for command in commands()]


def test_help_surface_matches_pre_split_snapshot() -> None:
    """Every registered command renders the --help stored before the split."""
    for name in _registered_names():
        result = subprocess.run(  # noqa: S603 - fixed interpreter and argv in tests
            [sys.executable, "-m", "src.data.cli", name, "--help"],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, name
        assert result.stdout == (SNAPSHOTS / f"{name}.txt").read_text(encoding="utf-8"), name
    top = subprocess.run(  # noqa: S603 - fixed interpreter and argv in tests
        [sys.executable, "-m", "src.data.cli", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert top.returncode == 0
    assert top.stdout == (SNAPSHOTS / "__main__.txt").read_text(encoding="utf-8")


def test_help_imports_no_heavy_modules() -> None:
    """A fresh interpreter running main(['--help']) never imports polars."""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and inline script in tests
        [sys.executable, "-c",
         "from src.data.cli import main\n"
         "try:\n"
         "    main(['--help'])\n"
         "except SystemExit as exc:\n"
         "    print(f'exit={exc.code}')\n"
         "import sys\n"
         "print(f\"polars={('polars' in sys.modules)}\")"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "exit=0" in result.stdout
    assert "polars=False" in result.stdout


def test_exit_codes_follow_error_envelope(capsys, monkeypatch) -> None:
    """PITDataError/ConfigError exit 2, ProviderError exits 3, each with one JSON error line."""
    import pytest

    from src.config.errors import ConfigError
    from src.core.pit import PITDataError
    from src.data.cli import main
    from src.data.cli.common import CommandFailed
    from src.data.cli.registry import Command, register
    from src.data.cli import registry as registry_module
    from src.integrations.errors import ProviderError

    saved = list(registry_module._COMMANDS)
    try:
        def add(name, error):
            def fail(_args):
                raise error

            register(Command(name=name, help="envelope probe",
                             add_arguments=lambda parser: None, run=fail))

        add("probe-pit", PITDataError("pit boom"))
        add("probe-config", ConfigError("config boom"))
        add("probe-provider", ProviderError("provider boom"))
        add("probe-value", ValueError("value boom"))
        register(Command(name="probe-ok", help="envelope probe",
                         add_arguments=lambda parser: None, run=lambda _args: {"ok": True}))
        register(Command(name="probe-failed", help="envelope probe",
                         add_arguments=lambda parser: None,
                         run=lambda _args: (_ for _ in ()).throw(
                             CommandFailed(1, {"type": "summary", "failed": 1}))))

        assert main(["probe-ok"]) == 0
        assert json.loads(capsys.readouterr().out) == {"ok": True}

        for name in ("probe-pit", "probe-config", "probe-value"):
            assert main([name]) == 2, name
            line = capsys.readouterr().out.strip().splitlines()
            assert len(line) == 1, name
            assert "boom" in json.loads(line[0])["error"], name

        assert main(["probe-provider"]) == 3
        line = capsys.readouterr().out.strip().splitlines()
        assert len(line) == 1
        assert "boom" in json.loads(line[0])["error"]

        assert main(["probe-failed"]) == 1
        assert json.loads(capsys.readouterr().out) == {"type": "summary", "failed": 1}
    finally:
        registry_module._COMMANDS.clear()
        registry_module._COMMANDS.extend(saved)

    with pytest.raises(ImportError):
        from src.data.cli import this_name_does_not_exist  # noqa: F401


def test_duplicate_command_names_are_rejected() -> None:
    """Two registrations of one name fail, at registration and at parser build."""
    import pytest

    from src.data.cli import registry as registry_module
    from src.data.cli.registry import Command, build_parser, register

    saved = list(registry_module._COMMANDS)
    try:
        register(Command(name="probe-dup", help="first",
                         add_arguments=lambda parser: None, run=lambda _args: {}))
        register(Command(name="probe-dup", help="second",
                         add_arguments=lambda parser: None, run=lambda _args: {}))
        with pytest.raises(ValueError, match="duplicate"):
            build_parser()
    finally:
        registry_module._COMMANDS.clear()
        registry_module._COMMANDS.extend(saved)
