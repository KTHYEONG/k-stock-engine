"""Area command registry for the data CLI: names, parsers and dispatch without domain imports."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Command:
    """One CLI command: argument registration plus a lazy domain run function."""

    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], Mapping[str, object]]


_COMMANDS: list[Command] = []


def register(command: Command) -> Command:
    """Register one command; duplicate names fail when the parser is built."""
    _COMMANDS.append(command)
    return command


def commands() -> tuple[Command, ...]:
    """Registered commands in registration order."""
    return tuple(_COMMANDS)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser from the registry, rejecting duplicate names."""
    # Fixed prog keeps --help bytes identical across the old module file and the new package entry points.
    parser = argparse.ArgumentParser(description="PIT dataset foundation CLI", prog="cli.py")
    sub = parser.add_subparsers(dest="command", required=True)
    seen: set[str] = set()
    for command in commands():
        if command.name in seen:
            raise ValueError(f"duplicate data CLI command: {command.name!r}")
        seen.add(command.name)
        child = sub.add_parser(command.name, help=command.help)
        command.add_arguments(child)
    return parser
