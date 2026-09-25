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
    )

    assert path.name.startswith("financial_quality_")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["facts"].startswith("financial_facts_")
    assert manifest["details"]["decision_time"] == available.isoformat()
    assert load_latest_financial_quality(root=tmp_path, decision_time=available).equals(quality)
    assert load_latest_financial_quality(
        root=tmp_path, decision_time=available - timedelta(days=1)
    ).is_empty()


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


def _quarantine_record(*, company_id: str, period: str, filing_id: str, published_at: datetime, available_at: datetime) -> dict[str, object]:
    return {
        "company_id": company_id,
        "dart_corp_code": "00126380",
        "fiscal_period": period,
        "filing_id": filing_id,
        "source_kind": "legacy_document",
        "published_at": published_at.isoformat(),
        "available_at": available_at.isoformat(),
    }


def test_quarantine_events_mark_period_incomplete_from_availability() -> None:
    from src.data.financial_quality import UNVERIFIED_LEGACY_REASON, build_financial_quality_events, quarantine_events

    available = datetime(2020, 3, 31, 0, 0, tzinfo=UTC)
    events = quarantine_events([
        _quarantine_record(
            company_id="005930", period="2019Q4", filing_id="F9",
            published_at=available - timedelta(days=1), available_at=available,
        )
    ])
    quality = build_financial_quality_events(
        pl.DataFrame(_facts(company_id="005930", period="2018Q4", available_at=available - timedelta(days=90))),
        unresolved_events=events,
        decision_time=available,
    )

    row = quality.filter(pl.col("fiscal_period") == "2019Q4").to_dicts()[0]
    assert row["financial_complete"] is False
    assert row["exclusion_reason"] == UNVERIFIED_LEGACY_REASON == "unverified_legacy_extraction"
    assert row["available_at"] == available


def test_quarantined_period_does_not_fall_back_to_prior_period() -> None:
    from src.data.financial_quality import eligible_companies_from_quality, build_financial_quality_events, quarantine_events

    complete_at = datetime(2019, 11, 14, 0, 0, tzinfo=UTC)
    quarantined_at = datetime(2020, 3, 31, 0, 0, tzinfo=UTC)
    events = quarantine_events([
        _quarantine_record(
            company_id="005930", period="2019Q4", filing_id="F9",
            published_at=quarantined_at - timedelta(days=1), available_at=quarantined_at,
        )
    ])
    quality = build_financial_quality_events(
        pl.DataFrame(_facts(company_id="005930", period="2018Q4", available_at=complete_at)),
        unresolved_events=events,
        decision_time=quarantined_at,
    )

    assert eligible_companies_from_quality(quality, decision_time=quarantined_at - timedelta(seconds=1), company_ids={"005930"}) == frozenset({"005930"})
    assert eligible_companies_from_quality(quality, decision_time=quarantined_at, company_ids={"005930"}) == frozenset()


def test_quarantine_events_reject_malformed_records() -> None:
    import pytest

    from src.data.financial_quality import quarantine_events

    available = datetime(2020, 3, 31, 0, 0, tzinfo=UTC)
    good = _quarantine_record(
        company_id="005930", period="2019Q4", filing_id="F9",
        published_at=available - timedelta(days=1), available_at=available,
    )
    missing_filing = {k: v for k, v in good.items() if k != "filing_id"}
    with pytest.raises(PITDataError):
        quarantine_events([missing_filing])
    naive = dict(good)
    naive["available_at"] = "2020-03-31T00:00:00"
    with pytest.raises(PITDataError):
        quarantine_events([naive])
    bad_period = dict(good)
    bad_period["fiscal_period"] = "2019H2"
    with pytest.raises(PITDataError):
        quarantine_events([bad_period])
    precedes = dict(good)
    precedes["available_at"] = (available - timedelta(days=2)).isoformat()
    with pytest.raises(PITDataError):
        quarantine_events([precedes])


def test_quarantine_events_ordering_deterministic() -> None:
    from src.data.financial_quality import quarantine_events

    available = datetime(2020, 3, 31, 0, 0, tzinfo=UTC)
    records = [
        _quarantine_record(
            company_id="005930", period="2019Q4", filing_id="B",
            published_at=available - timedelta(days=1), available_at=available,
        ),
        _quarantine_record(
            company_id="000660", period="2019Q4", filing_id="A",
            published_at=available - timedelta(days=1), available_at=available,
        ),
    ]
    assert quarantine_events(records) == quarantine_events(list(reversed(records)))
    assert [e.company_id for e in quarantine_events(records)] == ["000660", "005930"]


def test_implausible_balance_flags() -> None:
    from src.data.financial_quality import implausible_balance_flags

    assert implausible_balance_flags(assets=185, equity=5e11) == ("assets_below_floor", "equity_exceeds_assets")
    assert implausible_balance_flags(assets=-10, equity=-10) == ("assets_nonpositive",)
    assert implausible_balance_flags(assets=3e11, equity=1e11) == ()
    assert implausible_balance_flags(assets=None, equity=5) == ()


def test_quarantine_events_accept_datetime_and_reject_bad_shapes() -> None:
    import pytest

    from src.data.financial_quality import quarantine_events
    from src.data.schemas import PITDataError

    available = datetime(2020, 3, 31, 0, 0, tzinfo=UTC)
    published = available - timedelta(days=1)
    with pytest.raises(PITDataError):
        quarantine_events(12345)  # type: ignore[arg-type]
    with pytest.raises(PITDataError):
        quarantine_events(["not-a-mapping"])  # type: ignore[list-item]
    good = _quarantine_record(
        company_id="005930", period="2019Q4", filing_id="F9",
        published_at=published, available_at=available,
    )
    as_datetimes = dict(good)
    as_datetimes["published_at"] = published
    as_datetimes["available_at"] = available
    events = quarantine_events([as_datetimes])
    assert [e.filing_id for e in events] == ["F9"]
    missing_company = {k: v for k, v in good.items() if k != "company_id"}
    with pytest.raises(PITDataError):
        quarantine_events([missing_company])
    bad_timestamp = dict(good)
    bad_timestamp["published_at"] = "not-a-date"
    with pytest.raises(PITDataError):
        quarantine_events([bad_timestamp])
    missing_timestamp = {k: v for k, v in good.items() if k != "published_at"}
    with pytest.raises(PITDataError):
        quarantine_events([missing_timestamp])


def test_financial_quality_materialization_and_loader_boundaries(tmp_path) -> None:
    from src.data.financial_quality import (
        build_financial_quality_events,
        load_latest_financial_quality,
        materialize_financial_quality,
    )

    available = datetime(2024, 5, 1, tzinfo=UTC)
    quality = build_financial_quality_events(
        pl.DataFrame(_facts(company_id="A", period="2024Q1", available_at=available)),
        unresolved_events=(),
        decision_time=available,
    )
    with pytest.raises(PITDataError, match="facts dataset id"):
        materialize_financial_quality(
            quality,
            layer_root=tmp_path / "silver",
            decision_time=available,
            facts_dataset_id="",
        )
    with pytest.raises(PITDataError, match="requires events"):
        materialize_financial_quality(
            pl.DataFrame(),
            layer_root=tmp_path / "silver",
            decision_time=available,
            facts_dataset_id="financial_facts_0123456789abcdef",
        )
    with pytest.raises(PITDataError, match="unexpected schema"):
        materialize_financial_quality(
            pl.DataFrame({"unexpected": [1]}),
            layer_root=tmp_path / "silver",
            decision_time=available,
            facts_dataset_id="financial_facts_0123456789abcdef",
        )
    with pytest.raises(PITDataError, match="after decision_time"):
        materialize_financial_quality(
            quality,
            layer_root=tmp_path / "silver",
            decision_time=available - timedelta(seconds=1),
            facts_dataset_id="financial_facts_0123456789abcdef",
        )
    with pytest.raises(PITDataError, match="Silver layer root"):
        materialize_financial_quality(
            quality,
            decision_time=available,
            facts_dataset_id="financial_facts_0123456789abcdef",
        )

    root = tmp_path / "quality-root"
    path = materialize_financial_quality(
        quality,
        root=root,
        decision_time=available,
        facts_dataset_id="financial_facts_0123456789abcdef",
    )
    (root / ".hidden").mkdir()
    (root / "broken_0123456789abcdef").mkdir()
    (root / "broken_0123456789abcdef" / "manifest.json").write_text("not-json", encoding="utf-8")
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    publish_dataset(
        layer_root=root,
        identity=DatasetIdentity("daily_market", DatasetLayer.SILVER, "fixture-v1", {}, {}),
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )
    with pytest.raises(PITDataError, match="timezone-aware"):
        load_latest_financial_quality(root=root, decision_time=available.replace(tzinfo=None))
    loaded = load_latest_financial_quality(root=root, decision_time=available)
    assert loaded.equals(quality)
    assert path.is_dir()
