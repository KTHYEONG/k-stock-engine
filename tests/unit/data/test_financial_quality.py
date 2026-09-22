from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from src.data.financial_quality import (
    FinancialQualityEvent,
    FinancialQualityPolicy,
    build_financial_quality_events,
    eligible_companies_from_quality,
    load_latest_financial_quality,
    materialize_financial_quality,
)
from src.core.datasets import DatasetCertification
from src.data.schemas import PITDataError


def _facts(*, company_id: str, period: str, available_at: datetime, missing: frozenset[str] = frozenset()) -> list[dict[str, object]]:
    values = {
        "sales": 100.0,
        "gross_profit": 20.0,
        "operating_profit": 10.0,
        "net_income": 8.0,
        "assets": 200.0,
        "equity": 100.0,
        "operating_cash_flow": 12.0,
    }
    return [
        {
            "company_id": company_id,
            "fiscal_period": period,
            "filing_id": f"{company_id}-{period}",
            "fact": fact,
            "published_at": available_at,
            "available_at": available_at,
            "value": value,
            "unit": "KRW",
            "consolidated": True,
        }
        for fact, value in values.items()
        if fact not in missing
    ]


def test_financial_quality_unresolved_newest_period_blocks_company() -> None:
    available = datetime(2020, 3, 31, 9, tzinfo=UTC)
    event = FinancialQualityEvent(
        company_id="205290",
        fiscal_period="2019Q4",
        filing_id="20200330004717",
        published_at=available - timedelta(days=1),
        available_at=available,
        reason="missing_source_value",
    )
    quality = build_financial_quality_events(
        pl.DataFrame(_facts(company_id="205290", period="2019Q3", available_at=available - timedelta(days=90))),
        unresolved_events=(event,),
        decision_time=available,
    )

    assert eligible_companies_from_quality(quality, decision_time=available - timedelta(seconds=1), company_ids={"205290"}) == frozenset({"205290"})
    assert eligible_companies_from_quality(quality, decision_time=available, company_ids={"205290"}) == frozenset()


def test_financial_quality_later_complete_correction_only_applies_after_availability() -> None:
    failed_at = datetime(2020, 3, 31, 9, tzinfo=UTC)
    corrected_at = failed_at + timedelta(days=5)
    event = FinancialQualityEvent(
        company_id="205290",
        fiscal_period="2019Q4",
        filing_id="failed",
        published_at=failed_at - timedelta(days=1),
        available_at=failed_at,
        reason="missing_source_value",
    )
    quality = build_financial_quality_events(
        pl.DataFrame(_facts(company_id="205290", period="2019Q4", available_at=corrected_at)),
        unresolved_events=(event,),
        decision_time=corrected_at,
    )

    assert eligible_companies_from_quality(quality, decision_time=failed_at, company_ids={"205290"}) == frozenset()
    assert eligible_companies_from_quality(quality, decision_time=corrected_at, company_ids={"205290"}) == frozenset({"205290"})


def test_financial_quality_prefers_complete_consolidated_and_allows_complete_separate() -> None:
    available = datetime(2024, 5, 1, tzinfo=UTC)
    rows = _facts(company_id="A", period="2024Q1", available_at=available)
    rows.extend(
        {
            **row,
            "company_id": "B",
            "filing_id": row["filing_id"].replace("A", "B"),
            "consolidated": False,
        }
        for row in _facts(company_id="A", period="2024Q1", available_at=available)
    )
    rows.extend(_facts(company_id="B", period="2024Q1", available_at=available, missing=frozenset({"sales"})))
    quality = build_financial_quality_events(pl.DataFrame(rows), unresolved_events=(), decision_time=available)

    assert eligible_companies_from_quality(quality, decision_time=available, company_ids={"A", "B"}) == frozenset({"A", "B"})
    a = quality.filter((pl.col("company_id") == "A") & (pl.col("financial_complete"))).select("accounting_basis").to_series().to_list()
    assert a == ["consolidated"]


def test_financial_quality_marks_missing_non_krw_and_non_finite_fact_incomplete() -> None:
    available = datetime(2024, 5, 1, tzinfo=UTC)
    rows = _facts(company_id="A", period="2024Q1", available_at=available)
    rows[0]["unit"] = "USD"
    rows[1]["value"] = float("nan")
    quality = build_financial_quality_events(pl.DataFrame(rows), unresolved_events=(), decision_time=available)

    row = quality.filter(pl.col("company_id") == "A").tail(1).to_dicts()[0]
    assert row["financial_complete"] is False
    assert json.loads(row["missing_facts_json"]) == ["gross_profit", "sales"]


def test_materialized_financial_quality_preserves_source_dataset_lineage(tmp_path) -> None:
    available = datetime(2024, 5, 1, tzinfo=UTC)
    quality = build_financial_quality_events(
        pl.DataFrame(_facts(company_id="A", period="2024Q1", available_at=available)),
        unresolved_events=(),
        decision_time=available,
    )

    path = materialize_financial_quality(
        quality,
        root=tmp_path,
        dataset_id="quality-v1",
        decision_time=available,
        source_dataset_id="financial-v1",
        certification=DatasetCertification.RESEARCH,
    )

    assert path.name == "quality-v1"
    manifest = json.loads((path / "content_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_financial_dataset_id"] == "financial-v1"
    assert load_latest_financial_quality(root=tmp_path, decision_time=available).equals(quality)
    assert load_latest_financial_quality(
        root=tmp_path, decision_time=available - timedelta(days=1)
    ).equals(quality)


def test_financial_quality_rejects_invalid_contract_inputs(tmp_path) -> None:
    available = datetime(2024, 5, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="required facts"):
        FinancialQualityPolicy(required_facts=("sales", "sales"))
    with pytest.raises(ValueError, match="identity"):
        FinancialQualityEvent("", "2024Q1", "f", available, available, "missing")
    with pytest.raises(PITDataError, match="timezone-aware"):
        build_financial_quality_events(
            pl.DataFrame(), unresolved_events=(), decision_time=available.replace(tzinfo=None)
        )
    with pytest.raises(PITDataError, match="missing certified"):
        load_latest_financial_quality(root=tmp_path, decision_time=available)
    empty = build_financial_quality_events(pl.DataFrame(), unresolved_events=(), decision_time=available)
    assert empty.is_empty()
    assert eligible_companies_from_quality(empty, decision_time=available, company_ids={"A"}) == frozenset()
