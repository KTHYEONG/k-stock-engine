# mypy: ignore-errors
"""All research-only study handlers."""
from __future__ import annotations

from collections.abc import Callable


def run_research_only_model_selection_study(study: Callable[[], int]) -> int:
    if not callable(study):
        raise TypeError("study must be callable")
    result = study()
    if not isinstance(result, int):
        raise TypeError("study must return an integer exit code")
    return result
__all__ = ["run_research_only_model_selection_study"]
