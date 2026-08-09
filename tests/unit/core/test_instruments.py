"""Canonical Instrument identity contract, including lot size."""
from __future__ import annotations

import pytest

from src.core.instruments import AssetKind, Instrument


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
