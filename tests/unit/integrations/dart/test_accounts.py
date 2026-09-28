"""Shared DART fact vocabulary tests (label and element-id mapping)."""
from __future__ import annotations

STANDARD_PAIRS = {
    ("ifrs-full_Revenue", "매출액"): "sales",
    ("ifrs-full_Assets", "자산총계"): "assets",
    ("ifrs-full_OperatingProfit", "영업이익"): "operating_profit",
    ("ifrs-full_Liabilities", "부채총계"): "debt",
    ("ifrs-full_Equity", "자본총계"): "equity",
    ("ifrs-full_CashAndCashEquivalents", "현금및현금성자산"): "cash",
    ("ifrs-full_NetCashFlowsFromOperatingActivities", "영업활동현금흐름"): "operating_cash_flow",
    ("ifrs-full_PaymentsToAcquirePropertyPlantAndEquipment", "유형자산취득"): "capex",
    ("", "당기순이익"): "net_income",
    ("", "매출총이익"): "gross_profit",
    ("unknown_xyz", "???"): None,
    ("", ""): None,
}


def test_standard_mapping_pairs_resolve_to_the_same_facts() -> None:
    from src.integrations.dart.accounts import MAPPING_VERSION, map_standardized_account

    assert MAPPING_VERSION == "dart-fact-map-v1"
    for (account_id, account_nm), expected in STANDARD_PAIRS.items():
        assert map_standardized_account(account_id=account_id, account_nm=account_nm) == expected


def test_normalize_statement_label_strips_enumerators_spacing_and_notes() -> None:
    from src.integrations.dart.accounts import normalize_statement_label

    assert normalize_statement_label("Ⅰ. 자 산 총 계") == "자산총계"  # noqa: RUF001
    assert normalize_statement_label("1. 현금및예치금") == "현금및예치금"
    assert normalize_statement_label("이자수익 (주석 29)") == "이자수익"
    assert normalize_statement_label("(1) 매출액") == "매출액"


def test_normalize_statement_label_keeps_the_account_name_intact() -> None:
    from src.integrations.dart.accounts import normalize_statement_label, map_standardized_account

    assert normalize_statement_label("자산총계") == "자산총계"
    assert normalize_statement_label("  부채 및 자본 총계 ") == "부채및자본총계"
    assert normalize_statement_label("가. 매출액") == "매출액"
    assert map_standardized_account(account_nm=normalize_statement_label("Ⅱ. 자 산 총 계")) == "assets"
