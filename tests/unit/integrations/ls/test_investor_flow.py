"""LS investor-flow transport invariants: raw rows, query record, placeholder rule."""
from __future__ import annotations

from datetime import date


def _balanced(**overrides):  # type: ignore[no-untyped-def]
    row = {
        "date": "20260105",
        "close": "70100",
        "volume": "12000000",
        "value": "840000000000",
        "tjj0008": "-60",
        "tjj0009": "10",
        "tjj0010": "20",
        "tjj0016": "30",
        "tjj0007": "5",
        "tjj0011": "45",
        "tjj0017": "50",
        "tjj0000": "1",
        "tjj0001": "2",
        "tjj0002": "3",
        "tjj0003": "4",
        "tjj0004": "5",
        "tjj0005": "6",
        "tjj0006": "-41",
        "tjj0018": "-20",
    }
    row.update(overrides)
    return row


class _StubClient:
    def __init__(self, rows=(), *, error=None):  # type: ignore[no-untyped-def]
        self._rows = tuple(rows)
        self._error = error
        self.calls: list[tuple] = []

    def inquire_investor_trend(self, symbol, start, end):  # type: ignore[no-untyped-def]
        self.calls.append((symbol, start, end))
        if self._error is not None:
            raise self._error
        return self._rows

    def health_check(self) -> None:
        return None


def _collector(rows=(), **kwargs):  # type: ignore[no-untyped-def]
    from src.integrations.ls.investor_flow import LsInvestorFlowCollector

    return LsInvestorFlowCollector(client=_StubClient(rows, **kwargs))


def test_raw_rows_preserved() -> None:
    row = _balanced(extra_field="kept", tjj0008="nonsense-but-raw")
    response = _collector((row,)).fetch("005930", date(2026, 1, 5), date(2026, 1, 5))

    assert response.rows == (row,)
    assert dict(response.query) == {"symbol": "005930", "start": "2026-01-05", "end": "2026-01-05"}


def test_no_rows_is_not_an_error() -> None:
    response = _collector(()).fetch("005930", date(2026, 1, 5), date(2026, 1, 5))

    assert response.rows == ()


def test_placeholder_dropped_only_when_all_zero() -> None:
    zeroed = {f"tjj{i:04d}": 0 for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18)}
    placeholder = {"date": "f", "close": 116, "volume": 5004, **zeroed}
    valued = _balanced(date="f")

    response = _collector((placeholder, valued)).fetch("005930", date(2026, 1, 5), date(2026, 1, 5))

    assert response.rows == (valued,)


def test_transport_failure_raises_provider_error() -> None:
    import pytest

    from src.integrations.errors import ProviderError, ProviderRetryableError

    with pytest.raises(ProviderRetryableError):
        _collector(error=ProviderRetryableError("down")).fetch("005930", date(2026, 1, 5), date(2026, 1, 5))
    with pytest.raises(ProviderError, match="transport failed"):
        _collector(error=RuntimeError("boom")).fetch("005930", date(2026, 1, 5), date(2026, 1, 5))


def test_fetch_validates_inputs() -> None:
    import pytest

    from src.integrations.ls.investor_flow import LsInvestorFlowCollector

    collector = _collector((_balanced(),))
    with pytest.raises(ValueError, match="coverage_start"):
        collector.fetch("005930", date(2026, 1, 6), date(2026, 1, 5))
    with pytest.raises(ValueError, match="symbol"):
        collector.fetch("  ", date(2026, 1, 5), date(2026, 1, 5))
    with pytest.raises(ValueError, match="at least one symbol"):
        LsInvestorFlowCollector(())


def test_constructor_keeps_declared_symbols() -> None:
    from src.integrations.ls.investor_flow import LsInvestorFlowCollector

    collector = LsInvestorFlowCollector((" 005930 ", "005930", "000660"), client=_StubClient())

    assert collector._symbols == ("005930", "000660")


def test_health_check_delegates_to_client() -> None:
    checks: list[str] = []

    class _HealthyClient(_StubClient):
        def health_check(self) -> None:
            checks.append("ok")

    from src.integrations.ls.investor_flow import LsInvestorFlowCollector

    LsInvestorFlowCollector(client=_HealthyClient()).health_check()
    LsInvestorFlowCollector(client=object()).health_check()

    assert checks == ["ok"]


def test_empty_date_and_iso_date_rows_are_kept() -> None:
    zeroed = {f"tjj{i:04d}": 0 for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18)}
    dateless = {"date": "", "close": 0, "volume": 0, **zeroed}
    iso = _balanced(date="2026-01-05")

    response = _collector((dateless, iso)).fetch("005930", date(2026, 1, 5), date(2026, 1, 5))

    assert response.rows == (dateless, iso)
