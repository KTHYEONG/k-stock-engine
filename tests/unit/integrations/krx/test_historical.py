from datetime import date

import pytest

from src.data.schemas import PITDataError
from src.integrations.krx.historical import KrxHistoricalCollector


def test_krx_investor_flow_does_not_fallback_to_trade_client() -> None:
    collector = KrxHistoricalCollector(api_key="key", request_json=lambda *_: {})
    with pytest.raises(PITDataError, match="certification blocked"):
        collector.fetch_investor_flow(date(2024, 1, 2), date(2024, 1, 2))


def test_krx_daily_market_rejects_missing_capitalisation_fields() -> None:
    from datetime import date
    from types import SimpleNamespace

    import pytest

    from src.data.schemas import PITDataError
    from src.integrations.krx.historical import KrxHistoricalCollector

    collector = KrxHistoricalCollector(api_key='key', request_json=lambda *_: {})
    collector._client = SimpleNamespace(fetch_trade_records=lambda _day: [{'ISU_SRT_CD': '005930'}])

    with pytest.raises(PITDataError, match=r'MKTCAP.*LIST_SHRS'):
        tuple(collector.fetch_daily_market(date(2016, 1, 4), date(2016, 1, 4)))


def test_krx_historical_collector_calls_only_planned_sessions() -> None:
    from datetime import date
    from types import SimpleNamespace
    from src.integrations.krx.historical import KrxHistoricalCollector

    called = []
    collector = KrxHistoricalCollector(api_key='key', request_json=lambda *_args: {})
    collector._client = SimpleNamespace(fetch_trade_records=lambda day: called.append(day) or [{'BAS_DD': day.strftime('%Y%m%d'), 'ISU_SRT_CD': '005930', 'TDD_OPNPRC': '10', 'TDD_HGPRC': '11', 'TDD_LWPRC': '9', 'TDD_CLSPRC': '10', 'ACC_TRDVOL': '1', 'ACC_TRDVAL': '10', 'MKTCAP': '100', 'LIST_SHRS': '10'}])

    tuple(collector.fetch_daily_market(date(2016, 1, 8), date(2016, 1, 11), sessions=(date(2016, 1, 8), date(2016, 1, 11))))

    assert called == [date(2016, 1, 8), date(2016, 1, 11)]


def test_krx_historical_collector_validates_ranges_and_master_plans() -> None:
    from types import SimpleNamespace

    collector = KrxHistoricalCollector(api_key='key', request_json=lambda *_args: {})
    with pytest.raises(PITDataError, match='coverage_start'):
        tuple(collector.fetch_daily_market(date(2016, 1, 5), date(2016, 1, 4), sessions=()))
    collector._client = SimpleNamespace(
        fetch_master_records=lambda day: [{'ticker': '005930', 'session': day.isoformat()}]
    )
    with pytest.raises(PITDataError, match='dates only'):
        tuple(collector.fetch_master_lineage(date(2016, 1, 4), date(2016, 1, 4), sessions=('bad',)))
    with pytest.raises(PITDataError, match='outside'):
        tuple(collector.fetch_master_lineage(date(2016, 1, 4), date(2016, 1, 4), sessions=(date(2016, 1, 5),)))
    with pytest.raises(PITDataError, match='dates only'):
        tuple(collector.fetch_daily_market(date(2016, 1, 4), date(2016, 1, 4), sessions=('bad',)))
    with pytest.raises(PITDataError, match='outside'):
        tuple(collector.fetch_daily_market(date(2016, 1, 4), date(2016, 1, 4), sessions=(date(2016, 1, 5),)))
    pages = tuple(
        collector.fetch_master_lineage(
            date(2016, 1, 4), date(2016, 1, 5), sessions=(date(2016, 1, 5), date(2016, 1, 4))
        )
    )
    assert [page['session'] for page in pages] == ['2016-01-04', '2016-01-05']
    pages = tuple(collector.fetch_master_lineage(date(2016, 1, 4), date(2016, 1, 4)))
    assert pages[0]['session'] == '2016-01-04'
