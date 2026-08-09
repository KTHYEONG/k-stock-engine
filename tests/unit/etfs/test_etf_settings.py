"""ETF settings contract tests."""
from __future__ import annotations

from src.etfs.settings import DEFAULT_ETF, EtfSettings


class TestEtfSettings:
    def test_defaults_match_engine_contract(self) -> None:
        settings = EtfSettings()
        assert settings.strategy_name == "IndexSwitchV1"
        assert settings.initial_balance == 10_000_000.0
        assert settings.fee_rate == 0.00015
        assert settings.capital_use == 0.99

    def test_default_instance_is_frozen_default(self) -> None:
        assert EtfSettings() == DEFAULT_ETF
