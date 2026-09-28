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


def test_enumerated_parenthesized_revenue_label_maps_to_sales() -> None:
    from src.integrations.dart.accounts import map_standardized_account, normalize_statement_label

    for raw in ("Ⅰ.수익(매출액)", "I. 수익(매출액)", "1.수익(매출액)"):  # noqa: RUF001
        assert map_standardized_account(account_nm=normalize_statement_label(raw)) == "sales"
    assert map_standardized_account(account_nm=normalize_statement_label("I. 영업수익")) == "sales"


def test_whole_label_bracket_wrapper_is_removed() -> None:
    from src.integrations.dart.accounts import normalize_statement_label

    assert normalize_statement_label("(매출액)") == "매출액"
    assert normalize_statement_label("(1)영업이익") == "영업이익"


def test_inner_parenthetical_survives_normalization() -> None:
    from src.integrations.dart.accounts import normalize_statement_label

    assert normalize_statement_label("수익(매출액)") == "수익(매출액)"


def test_normalization_is_idempotent() -> None:
    from src.integrations.dart.accounts import normalize_statement_label

    labels = [
        "Ⅰ. 자 산 총 계",  # noqa: RUF001
        "1. 현금및예치금",
        "이자수익 (주석 29)",
        "(1) 매출액",
        "자산총계",
        "  부채 및 자본 총계 ",
        "가. 매출액",
        "Ⅰ.수익(매출액)",  # noqa: RUF001
        "(매출액)",
        "(1)영업이익",
        "수익(매출액)",
    ]
    for raw in labels:
        once = normalize_statement_label(raw)
        assert normalize_statement_label(once) == once


def test_normalize_statement_label_bracket_boundaries() -> None:
    from src.integrations.dart.accounts import normalize_statement_label

    assert normalize_statement_label(")매출액") == "매출액"
    assert normalize_statement_label("[매출액") == "매출액"
    assert normalize_statement_label("매출액]") == "매출액"
    assert normalize_statement_label("(매출액)(영업이익)") == "(매출액)(영업이익)"
    assert normalize_statement_label("((매출액)") == "매출액"
    assert normalize_statement_label("(매출액") == "매출액"
    assert normalize_statement_label("수익(매출액") == "수익매출액"


def test_loss_only_labels_map_to_their_facts() -> None:
    from src.integrations.dart.accounts import map_loss_only_account

    assert map_loss_only_account(account_nm="영업손실") == "operating_profit"
    assert map_loss_only_account(account_nm="당기순손실") == "net_income"
    assert map_loss_only_account(account_nm="분기순손실") == "net_income"
    assert map_loss_only_account(account_nm="반기순손실") == "net_income"
    assert map_loss_only_account(account_nm="연결당기순손실") == "net_income"
    assert map_loss_only_account(account_nm="분기연결순손실") == "net_income"
    assert map_loss_only_account(account_nm="반기연결순손실") == "net_income"


def test_standard_vocabulary_leaves_loss_only_labels_unmapped() -> None:
    from src.integrations.dart.accounts import map_standardized_account

    assert map_standardized_account(account_nm="영업손실") is None
    assert map_standardized_account(account_nm="당기순손실") is None


def test_support_lines_map_to_consistency_inputs() -> None:
    from src.integrations.dart.accounts import map_income_support_account

    assert map_income_support_account(account_nm="매출원가") == "cost_of_sales"
    assert map_income_support_account(account_nm="판매비와관리비") == "sga"
    assert map_income_support_account(account_nm="판매비및관리비") == "sga"


def test_new_mapping_functions_ignore_unrelated_labels() -> None:
    from src.integrations.dart.accounts import map_income_support_account, map_loss_only_account

    assert map_loss_only_account(account_nm="영업이익") is None
    assert map_loss_only_account(account_nm="자산총계") is None
    assert map_income_support_account(account_nm="영업이익") is None
    assert map_income_support_account(account_nm="자산총계") is None
