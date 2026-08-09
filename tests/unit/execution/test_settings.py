"""Execution settings contract tests."""
from __future__ import annotations

from src.execution.settings import DEFAULT_EXECUTION, ExecutionSettings


class TestExecutionSettings:
    def test_paper_is_the_only_default_mode(self) -> None:
        assert ExecutionSettings().default_mode == "paper"

    def test_default_instance_is_frozen_default(self) -> None:
        assert ExecutionSettings() == DEFAULT_EXECUTION
