"""PLAN-01-ASSET-KIND-IS-MANDATORY: AssetKind is mandatory on every instrument."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.core.instruments import AssetKind, Instrument, InstrumentResolver, ProviderSymbol
from src.core.time import KRX_DAILY, KRX_TZ, PointInTime, SessionCalendar, TemporalViolationError


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


class TestPointInTime:
    def test_valid_ordering_accepted(self) -> None:
        pit = PointInTime(
            observation_time=datetime(2024, 1, 2, 14, 0, tzinfo=KRX_TZ),
            available_time=datetime(2024, 1, 2, 15, 30, tzinfo=KRX_TZ),
            decision_time=datetime(2024, 1, 3, 8, 50, tzinfo=KRX_TZ),
            execution_time=datetime(2024, 1, 3, 9, 5, tzinfo=KRX_TZ),
        )
        assert pit.available_time <= pit.decision_time

    def test_available_after_decision_is_rejected(self) -> None:
        with pytest.raises(TemporalViolationError):
            PointInTime(
                observation_time=datetime(2024, 1, 2, 14, 0, tzinfo=KRX_TZ),
                available_time=datetime(2024, 1, 3, 8, 50, tzinfo=KRX_TZ),
                decision_time=datetime(2024, 1, 3, 8, 0, tzinfo=KRX_TZ),
                execution_time=datetime(2024, 1, 3, 9, 5, tzinfo=KRX_TZ),
            )


class TestSession:
    def test_krx_session_hours(self) -> None:
        assert KRX_DAILY.open_time.hour == 9
        assert KRX_DAILY.close_time.hour == 15

    def test_calendar_requires_monotonic_sessions(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            SessionCalendar(
                sessions=(
                    datetime(2024, 1, 2, tzinfo=KRX_TZ),
                    datetime(2024, 1, 2, tzinfo=KRX_TZ),
                )
            )

    def test_calendar_sessions_between(self) -> None:
        cal = SessionCalendar(
            sessions=(
                datetime(2024, 1, 2, tzinfo=KRX_TZ),
                datetime(2024, 1, 3, tzinfo=KRX_TZ),
                datetime(2024, 1, 4, tzinfo=KRX_TZ),
                datetime(2024, 1, 5, tzinfo=KRX_TZ),
            )
        )
        got = cal.sessions_between(
            datetime(2024, 1, 3, tzinfo=KRX_TZ), datetime(2024, 1, 4, tzinfo=KRX_TZ)
        )
        assert len(got) == 1
