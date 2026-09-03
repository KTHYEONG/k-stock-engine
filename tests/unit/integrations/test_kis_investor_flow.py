from datetime import date

import pytest

from src.data.schemas import PITDataError
from src.integrations.kis.investor_flow import KisInvestorFlowCollector


def test_kis_investor_flow_maps_transaction_values() -> None:
    class Client:
        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            return (
                {
                    'stck_bsop_date': '20160129',
                    'frgn_shnu_tr_pbmn': '100',
                    'frgn_seln_tr_pbmn': '40',
                    'frgn_ntby_tr_pbmn': '60',
                    'orgn_ntby_tr_pbmn': '-20',
                    'prsn_ntby_tr_pbmn': '-40',
                },
            )

    rows = tuple(
        KisInvestorFlowCollector(('005930',), client=Client()).fetch_investor_flow(
            date(2016, 1, 29), date(2016, 1, 29)
        )
    )

    assert rows[0]['records'][0] == {
        'session': '2016-01-29',
        'ticker': '005930',
        'foreign_buy_value': 100.0,
        'foreign_sell_value': 40.0,
        'foreign_net_value': 60.0,
        'institution_net_value': -20.0,
        'retail_net_value': -40.0,
    }


def test_kis_probe_rejects_missing_requested_session() -> None:
    class Client:
        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            return (
                {
                    'stck_bsop_date': '20160201',
                    'frgn_shnu_tr_pbmn': '100',
                    'frgn_seln_tr_pbmn': '40',
                    'frgn_ntby_tr_pbmn': '60',
                    'orgn_ntby_tr_pbmn': '-20',
                    'prsn_ntby_tr_pbmn': '-40',
                },
            )

    with pytest.raises(PITDataError, match='requested session'):
        KisInvestorFlowCollector(('005930',), client=Client()).probe('005930', date(2016, 1, 29))
