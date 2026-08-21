from __future__ import annotations

from src.stocks.config.runtime import StockRuntimeSettings


def test_runtime_settings_do_not_define_financial_limits() -> None:
    settings = StockRuntimeSettings()

    assert settings.diagnostics_enabled is False
