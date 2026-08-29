from __future__ import annotations

from src.stocks.config.runtime import StockRuntimeSettings
from src.core.paths import PROJECT_ROOT


def test_runtime_settings_do_not_define_financial_limits() -> None:
    settings = StockRuntimeSettings()

    assert settings.diagnostics_enabled is False


def test_runtime_results_root_defaults_to_docs_results() -> None:
    assert StockRuntimeSettings().results_root == PROJECT_ROOT / "docs" / "results"
