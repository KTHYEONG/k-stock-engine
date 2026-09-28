"""One fact vocabulary for every DART account label and element id.

The tables below are the single place where a statement label or an XBRL
element id becomes a standardized fact name.  Every DART source (XBRL,
filing document) maps through here so the same account never means two
different things depending on where it was read.
"""
from __future__ import annotations

import re
from typing import Final

MAPPING_VERSION: Final = "dart-fact-map-v1"

_LABEL_TO_FACT: dict[str, str] = {
    "매출액": "sales",
    "매출": "sales",
    "영업수익": "sales",
    "수익": "sales",
    "매출총이익": "gross_profit",
    "매출총손익": "gross_profit",
    "영업이익": "operating_profit",
    "영업손익": "operating_profit",
    "당기순이익": "net_income",
    "당기순손익": "net_income",
    "분기순이익": "net_income",
    "분기순손익": "net_income",
    "반기순이익": "net_income",
    "지배기업소유주지분순이익": "net_income",
    "자산총계": "assets",
    "자산총액": "assets",
    "자산계": "assets",
    "자본총계": "equity",
    "자본총액": "equity",
    "자본계": "equity",
    "부채총계": "debt",
    "부채총액": "debt",
    "부채계": "debt",
    "현금및현금성자산": "cash",
    "기말현금및현금성자산": "cash",
    "영업활동현금흐름": "operating_cash_flow",
    "영업활동으로인한현금흐름": "operating_cash_flow",
    "영업활동으로인한순현금흐름": "operating_cash_flow",
    "유형자산취득": "capex",
    "자본적지출": "capex",
    "설비투자": "capex",
    "유형자산의취득": "capex",
    "유무형자산취득": "capex",
}

_ID_TO_FACT: dict[str, str] = {
    "ifrs-full_Revenue": "sales",
    "ifrs-full_RevenueFromContractsWithCustomers": "sales",
    "ifrs-full_SalesRevenue": "sales",
    "ifrs-full_GrossProfit": "gross_profit",
    "ifrs-full_OperatingProfit": "operating_profit",
    "ifrs-full_OperatingIncome": "operating_profit",
    "ifrs-full_ProfitLoss": "net_income",
    "ifrs-full_ProfitLossAttributableToOwnersOfParent": "net_income",
    "ifrs-full_ProfitLossAttributableToOwners": "net_income",
    "ifrs-full_ComprehensiveIncome": "net_income",
    "ifrs-full_Assets": "assets",
    "ifrs-full_Equity": "equity",
    "ifrs-full_EquityAttributableToOwnersOfParent": "equity",
    "ifrs-full_Liabilities": "debt",
    "ifrs-full_CashAndCashEquivalents": "cash",
    "ifrs-full_CashFlowsFromOperatingActivities": "operating_cash_flow",
    "ifrs-full_CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
    "ifrs-full_NetCashFlowsFromOperatingActivities": "operating_cash_flow",
    "ifrs-full_PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "ifrs-full_PaymentsToAcquireIntangibleAssets": "capex",
    "dart_OperatingIncome": "operating_profit",
    "dart_Revenue": "sales",
}

# Leading row enumerators: roman numeral, arabic digit or Korean syllable, always
# followed by "." or ")" so that "자산총계" keeps its first syllable.
_ENUMERATOR_RE = re.compile(r"^[\(\[]?(?:[IVXLCDMⅠ-Ⅻ]+|\d+|[가-하])[.\)]")  # noqa: RUF001
_NOTE_REFERENCE_RE = re.compile(r"\(주석[^)]*\)$")


def map_standardized_account(*, account_id: str = "", account_nm: str = "") -> str | None:
    """Map an XBRL element id or a statement label to its standardized fact name."""
    aid = (account_id or "").strip()
    if aid and aid in _ID_TO_FACT:
        return _ID_TO_FACT[aid]
    label = re.sub(r"\([^)]*\)", "", (account_nm or "")).replace(" ", "").strip()
    if label and label in _LABEL_TO_FACT:
        return _LABEL_TO_FACT[label]
    return None


def normalize_statement_label(raw: str) -> str:
    """Reduce one source row label to its bare account name.

    Removes whitespace, a trailing note reference such as ``(주석 29)``, a
    leading enumerator (``I.``, ``1.``, ``(1)``, ``가.``) and any brackets
    left around the name, so that a spaced ``"자 산 총 계"`` with a
    roman-numeral prefix becomes ``"자산총계"`` for fact lookup.
    """
    label = "".join(raw.split())
    label = _NOTE_REFERENCE_RE.sub("", label)
    label = _ENUMERATOR_RE.sub("", label)
    return label.strip("()[]")
