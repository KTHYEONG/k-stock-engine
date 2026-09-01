# mypy: ignore-errors
"""Cost contexts, direct training execution, memory estimation."""
from __future__ import annotations

from collections.abc import Callable


def _run_direct_training(runner: Callable[[], int]) -> int:
    if not callable(runner):
        raise TypeError("runner must be callable")
    result = runner()
    if not isinstance(result, int):
        raise TypeError("runner must return an integer exit code")
    return result
__all__ = ["_run_direct_training"]
