"""KIS investor-flow transport invariants: raw rows, query record, empty answer."""
from __future__ import annotations

from datetime import date


def _row(**overrides):  # type: ignore[no-untyped-def]
    row = {
        "stck_bsop_date": "20240102",
        "prsn_ntby_qty": "-100",
        "frgn_ntby_qty": "20",
        "orgn_ntby_qty": "-10",
        "etc_ntby_qty": "90",
    }
    row.update(overrides)
    return row


class _StubClient:
    def __init__(self, rows=(), *, error=None):  # type: ignore[no-untyped-def]
        self._rows = tuple(rows)
        self._error = error
        self.calls: list[tuple] = []

    def inquire_investor_trade_by_stock_daily(self, symbol, anchor):  # type: ignore[no-untyped-def]
        self.calls.append((symbol, anchor))
        if self._error is not None:
            raise self._error
        return self._rows

    def health_check(self) -> None:
        return None


def _collector(rows=(), **kwargs):  # type: ignore[no-untyped-def]
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    return KisInvestorFlowCollector(client=_StubClient(rows, **kwargs))


def test_raw_rows_preserved() -> None:
    row = _row(extra_field="kept")
    response = _collector((row,)).fetch("000001", date(2024, 1, 2))

    assert response.rows == (row,)


def test_query_recorded() -> None:
    response = _collector((_row(),)).fetch("000001", date(2024, 1, 2))

    assert dict(response.query) == {"symbol": "000001", "anchor": "2024-01-02"}


def test_no_rows_is_not_an_error() -> None:
    response = _collector(()).fetch("000001", date(2024, 1, 2))

    assert response.rows == ()


def test_transport_failure_raises_provider_error() -> None:
    import pytest

    from src.integrations.errors import ProviderError, ProviderRetryableError

    with pytest.raises(ProviderRetryableError):
        _collector(error=ProviderRetryableError("down")).fetch("000001", date(2024, 1, 2))
    with pytest.raises(ProviderError, match="transport failed"):
        _collector(error=RuntimeError("boom")).fetch("000001", date(2024, 1, 2))


def test_fetch_validates_inputs() -> None:
    import pytest

    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    collector = _collector((_row(),))
    with pytest.raises(ValueError, match="symbol"):
        collector.fetch("  ", date(2024, 1, 2))
    with pytest.raises(ValueError, match="anchor"):
        collector.fetch("000001", "2024-01-02")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one symbol"):
        KisInvestorFlowCollector(())


def test_constructor_keeps_declared_symbols() -> None:
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    collector = KisInvestorFlowCollector((" 000001 ", "000001", "000002"), client=_StubClient())

    assert collector._symbols == ("000001", "000002")


def test_health_check_delegates_to_client() -> None:
    checks: list[str] = []

    class _HealthyClient(_StubClient):
        def health_check(self) -> None:
            checks.append("ok")

    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    KisInvestorFlowCollector(client=_HealthyClient()).health_check()
    KisInvestorFlowCollector(client=object()).health_check()

    assert checks == ["ok"]
