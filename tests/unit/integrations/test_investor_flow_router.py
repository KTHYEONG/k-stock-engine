import pytest

from src.data.schemas import PITDataError
from src.integrations.investor_flow_router import resolve_investor_flow_collector
from src.integrations.kis.investor_flow import KisInvestorFlowCollector
from src.integrations.kiwoom.investor_flow import KiwoomInvestorFlowCollector
from src.integrations.ls.investor_flow import LsInvestorFlowCollector


def test_resolve_investor_flow_collector_routing() -> None:
    c_ls = resolve_investor_flow_collector("ls", ("005930",), client=object())
    assert isinstance(c_ls, LsInvestorFlowCollector)

    c_kw = resolve_investor_flow_collector("kiwoom", ("005930",), client=object())
    assert isinstance(c_kw, KiwoomInvestorFlowCollector)

    c_kis = resolve_investor_flow_collector("kis", ("005930",), client=object())
    assert isinstance(c_kis, KisInvestorFlowCollector)

    with pytest.raises(PITDataError, match="unsupported investor flow provider"):
        resolve_investor_flow_collector("invalid_broker", ("005930",))
