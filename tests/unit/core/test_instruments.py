"""Canonical Instrument identity contract, including lot size."""
from __future__ import annotations

import pytest

from src.core.instruments import AssetKind, Instrument, InstrumentResolver, ProviderSymbol


def make_instrument(**overrides: object) -> Instrument:
    values: dict[str, object] = {
        "instrument_id": "KRX:005930",
        "asset_kind": AssetKind.STOCK,
        "exchange": "KRX",
        "symbol": "005930",
        "currency": "KRW",
    }
    values.update(overrides)
    return Instrument(**values)


class TestInstrument:
    def test_default_lot_size_is_one(self) -> None:
        instrument = make_instrument()
        assert instrument.lot_size == 1

    def test_lot_size_is_carried(self) -> None:
        instrument = make_instrument(lot_size=100)
        assert instrument.lot_size == 100

    def test_rejects_zero_or_negative_lot_size(self) -> None:
        with pytest.raises(ValueError, match="lot_size"):
            make_instrument(lot_size=0)
        with pytest.raises(ValueError, match="lot_size"):
            make_instrument(lot_size=-1)

    def test_rejects_empty_instrument_id(self) -> None:
        with pytest.raises(ValueError, match="instrument_id"):
            make_instrument(instrument_id="")

    def test_rejects_empty_symbol(self) -> None:
        with pytest.raises(ValueError, match="symbol"):
            make_instrument(symbol="")

class TestAssetKindIsMandatory:
    def test_instrument_requires_explicit_asset_kind(self) -> None:
        inst = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
        assert inst.asset_kind is AssetKind.STOCK

    def test_instrument_is_frozen_and_slotted(self) -> None:
        inst = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
        assert inst.__slots__
        with pytest.raises(AttributeError):
            inst.asset_kind = AssetKind.ETF  # type: ignore[misc]

    def test_constructing_without_asset_kind_raises(self) -> None:
        with pytest.raises(TypeError):
            Instrument("KRX:005930", "KRX", "005930", "KRW")  # type: ignore[call-arg]

    def test_resolver_never_infers_kind_from_symbol(self) -> None:
        stock = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
        resolver = InstrumentResolver({("krx", "005930"): stock})
        resolved = resolver.resolve("krx", "005930")
        assert resolved.asset_kind is AssetKind.STOCK

    def test_routing_stock_instrument_to_etf_service_raises(self) -> None:
        stock = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")

        def etf_only_service(instrument: Instrument) -> None:
            if instrument.asset_kind is not AssetKind.ETF:
                raise ValueError(
                    f"ETF service received non-ETF instrument {instrument.instrument_id}"
                )

        with pytest.raises(ValueError, match="ETF service"):
            etf_only_service(stock)

    def test_resolver_unknown_symbol_raises(self) -> None:
        resolver = InstrumentResolver({})
        with pytest.raises(ValueError, match="Unknown provider symbol"):
            resolver.resolve("krx", "nope")

    def test_provider_symbol_holds_raw_input(self) -> None:
        ps = ProviderSymbol("krx", "005930")
        assert ps.symbol == "005930"
