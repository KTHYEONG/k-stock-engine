from datetime import date

import pytest

from src.data.schemas import PITDataError
from src.integrations.krx.historical import KrxHistoricalCollector


def test_krx_investor_flow_does_not_fallback_to_trade_client() -> None:
    collector = KrxHistoricalCollector(api_key="key", request_json=lambda *_: {})
    with pytest.raises(PITDataError, match="certification blocked"):
        collector.fetch_investor_flow(date(2024, 1, 2), date(2024, 1, 2))
