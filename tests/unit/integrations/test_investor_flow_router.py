import pytest

from src.data.schemas import PITDataError
from src.integrations.investor_flow_router import resolve_investor_flow_collector
from src.integrations.kiwoom.investor_flow import KiwoomInvestorFlowCollector
from src.integrations.ls.investor_flow import LsInvestorFlowCollector


def test_resolve_investor_flow_collector_routing() -> None:
    c_ls = resolve_investor_flow_collector("ls", ("005930",), client=object())
    assert isinstance(c_ls, LsInvestorFlowCollector)

    c_kw = resolve_investor_flow_collector("kiwoom", ("005930",), client=object())
    assert isinstance(c_kw, KiwoomInvestorFlowCollector)

    with pytest.raises(PITDataError, match="unsupported investor flow provider"):
        resolve_investor_flow_collector("kis", ("005930",), client=object())

    with pytest.raises(PITDataError, match="unsupported investor flow provider"):
        resolve_investor_flow_collector("invalid_broker", ("005930",))


def test_router_excludes_kis_and_routes_only_verified_providers() -> None:
    import pytest
    from src.data.schemas import PITDataError
    from src.integrations.investor_flow_router import SUPPORTED_INVESTOR_FLOW_PROVIDERS, resolve_investor_flow_collector
    from src.integrations.kiwoom.investor_flow import KiwoomInvestorFlowCollector
    from src.integrations.ls.investor_flow import LsInvestorFlowCollector

    assert frozenset({'ls', 'kiwoom'}) == SUPPORTED_INVESTOR_FLOW_PROVIDERS
    assert isinstance(resolve_investor_flow_collector(' LS ', ('005930',), client=object()), LsInvestorFlowCollector)
    assert isinstance(resolve_investor_flow_collector('kiwoom', ('005930',), client=object()), KiwoomInvestorFlowCollector)
    with pytest.raises(PITDataError, match='unsupported investor flow provider'):
        resolve_investor_flow_collector('kis', ('005930',), client=object())
