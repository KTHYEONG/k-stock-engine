"""Unified routing for multi-broker investor-flow collectors."""
from __future__ import annotations

from typing import Any

from src.core.pit import PITDataError
from src.integrations.kiwoom.investor_flow import KiwoomInvestorFlowCollector
from src.integrations.ls.investor_flow import LsInvestorFlowCollector

SUPPORTED_INVESTOR_FLOW_PROVIDERS: frozenset[str] = frozenset({"ls", "kiwoom"})


def resolve_investor_flow_collector(
    provider: str,
    symbols: tuple[str, ...],
    *,
    client: Any | None = None,
) -> LsInvestorFlowCollector | KiwoomInvestorFlowCollector:
    norm_provider = str(provider).strip().lower()
    if norm_provider == "ls":
        return LsInvestorFlowCollector(symbols, client=client)
    if norm_provider == "kiwoom":
        return KiwoomInvestorFlowCollector(symbols, client=client)
    raise PITDataError(f"unsupported investor flow provider: {provider!r}")
