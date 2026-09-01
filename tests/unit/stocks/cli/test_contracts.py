from __future__ import annotations


def test_research_study_kind_exposes_stock_only_factor_study() -> None:
    from src.stocks.cli.contracts import ResearchStudyKind

    assert ResearchStudyKind.stock_only_factor_study == "stock_only_factor_study"
    assert ResearchStudyKind("stock_only_factor_study") is ResearchStudyKind.stock_only_factor_study
