"""Unified routing for multi-broker investor-flow collectors."""
from __future__ import annotations

from typing import Any

from src.data.schemas import PITDataError
from src.integrations.kis.investor_flow import KisInvestorFlowCollector
from src.integrations.kiwoom.investor_flow import KiwoomInvestorFlowCollector
from src.integrations.ls.investor_flow import LsInvestorFlowCollector


def resolve_investor_flow_collector(
    provider: str,
    symbols: tuple[str, ...],
    *,
    client: Any | None = None,
) -> Any:
    norm_provider = str(provider).strip().lower()
    if norm_provider == "ls":
        return LsInvestorFlowCollector(symbols, client=client)
    if norm_provider == "kiwoom":
        return KiwoomInvestorFlowCollector(symbols, client=client)
    if norm_provider == "kis":
        return KisInvestorFlowCollector(symbols, client=client)
    raise PITDataError(f"unsupported investor flow provider: {provider!r}")
