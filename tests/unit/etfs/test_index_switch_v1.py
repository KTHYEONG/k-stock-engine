"""ETF unit tests: universe kind, signal determinism, simulation config."""
from __future__ import annotations

import pytest

from src.core.instruments import AssetKind
from src.etfs.domain.universe import KOSPI_ETF_UNIVERSE
from src.etfs.backtesting.engine import EtfBacktester, EtfSimulationConfig
from src.etfs.strategies.index_switch_v1 import IndexSwitchParams, IndexSwitchV1
from tests.fixtures.etfs.helpers import make_etf_fixture, preprocess_index


class TestEtfDomain:
    def test_etf_instruments_carry_explicit_asset_kind(self) -> None:
        for instrument in KOSPI_ETF_UNIVERSE.instruments():
            assert instrument.asset_kind is AssetKind.ETF
        assert KOSPI_ETF_UNIVERSE.bull_1x == "069500"

    def test_etf_simulation_config_is_declared(self) -> None:
        config = EtfSimulationConfig()
        assert config.fee_rate == 0.00015
        assert config.capital_use == 0.99


class TestIndexSwitchV1:
    def test_signal_requires_ohlc(self) -> None:
        from polars.exceptions import ColumnNotFoundError

        index_df, _ = make_etf_fixture(n_days=10)
        bad = index_df.select(["ticker", "date"])
        with pytest.raises(ColumnNotFoundError):
            IndexSwitchV1().generate_signal(bad)

    def test_signal_is_deterministic(self) -> None:
        index_df, _ = make_etf_fixture(n_days=60)
        preprocessed = preprocess_index(index_df)
        a = IndexSwitchV1().generate_signal(preprocessed)
        b = IndexSwitchV1().generate_signal(preprocessed)
        assert a["signal_trigger"].to_list() == b["signal_trigger"].to_list()


class TestEtfBacktester:
    def test_backtest_with_default_params_runs(self) -> None:
        index_df, etf_df = make_etf_fixture(n_days=90)
        bt = EtfBacktester(index_df, etf_df)
        results = bt.run(KOSPI_ETF_UNIVERSE, target_market="KOSPI")
        assert len(results) == 1
        assert results[0].total_trades >= 0

    def test_warmup_is_declared(self) -> None:
        assert IndexSwitchParams().required_warmup == 140
