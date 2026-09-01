"""Stock settings contract tests."""
from __future__ import annotations

from legacy.stocks.settings import DEFAULT_STOCK_ALPHA, StockAlphaSettings


class TestStockAlphaSettings:
    def test_defaults_match_baseline_pipeline(self) -> None:
        settings = StockAlphaSettings()
        assert settings.feature_set == "stock_alpha_v1"
        assert settings.label_definition == "fwd_ret_5d"
        assert settings.n_folds == 3
        assert settings.top_k == 20
        assert settings.max_single_weight == 0.08
        assert settings.max_exposure == 0.90

    def test_default_instance_is_frozen_default(self) -> None:
        assert StockAlphaSettings() == DEFAULT_STOCK_ALPHA
