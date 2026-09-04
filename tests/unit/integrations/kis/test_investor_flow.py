from datetime import date

from src.integrations.kis.investor_flow import KisInvestorFlowCollector


def test_kis_collection_can_limit_a_call_to_one_planned_symbol() -> None:
    class Client:
        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            return (
                {
                    "stck_bsop_date": anchor.strftime("%Y%m%d"),
                    "frgn_shnu_tr_pbmn": "1",
                    "frgn_seln_tr_pbmn": "0",
                    "frgn_ntby_tr_pbmn": "1",
                    "orgn_ntby_tr_pbmn": "0",
                    "prsn_ntby_tr_pbmn": "-1",
                },
            )

    pages = tuple(
        KisInvestorFlowCollector(("000001", "000002"), client=Client()).fetch_investor_flow(
            date(2024, 1, 2), date(2024, 1, 2), symbols=("000002",)
        )
    )

    assert pages[0]["symbol"] == "000002"
