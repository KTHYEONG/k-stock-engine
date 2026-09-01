"""Stock data quality hardening: classification, invariants, coverage, availability."""
from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.core.datasets import DatasetCertification
from legacy.stocks.data.quality import (
    CorporateActionInterval,
    CorporateActionSnapshot,
    InstrumentMasterRecord,
    InstrumentMasterSnapshot,
    KRXSessionCalendar,
    StockDataQualityPolicy,
    validate_canonical_stock_panel,
)
from legacy.stocks.research.datasets import (
    QUALITY_REASON_COLUMN,
)

SESSIONS = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5))
MASTER_ID = "KRX:000050"


def bar(day: date, ticker: str = "000050", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": day,
        "ticker": ticker,
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "volume": 1_000_000.0,
        "trading_value": 1.05e8,
        "market_cap": 1e12,
        "sector": "S1",
    }
    row.update(overrides)
    return row


def panel(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def master(*, common: tuple[str, ...] = ("000050",), non_equity: tuple[str, ...] = ()) -> InstrumentMasterSnapshot:
    records = [
        InstrumentMasterRecord(
            source_identifier=ticker,
            instrument_id=f"KRX:{ticker}",
            asset_type="common_stock",
            is_common_stock=True,
            listed_from=date(2024, 1, 1),
            tradable_from=date(2024, 1, 1),
            tradable_to=date(2024, 12, 31),
            available_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
        for ticker in common
    ]
    records.extend(
        InstrumentMasterRecord(
            source_identifier=ticker,
            instrument_id=f"KRX:{ticker}",
            asset_type="index",
            is_common_stock=False,
            listed_from=date(2024, 1, 1),
            available_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
        for ticker in non_equity
    )
    return InstrumentMasterSnapshot(
        version="test-master", records=tuple(records), generated_time=datetime(2026, 1, 1, tzinfo=UTC)
    )


def calendar() -> KRXSessionCalendar:
    return KRXSessionCalendar(
        version="test-calendar", sessions=SESSIONS, generated_time=datetime(2026, 1, 1, tzinfo=UTC)
    )


def no_action_actions() -> CorporateActionSnapshot:
    intervals = tuple(
        CorporateActionInterval(
            instrument_id=MASTER_ID,
            previous_session=SESSIONS[i - 1],
            session=SESSIONS[i],
            action_code="no_action",
            adjustment_factor=1.0,
        )
        for i in range(1, len(SESSIONS))
    )
    return CorporateActionSnapshot(
        version="test-actions", intervals=intervals, generated_time=datetime(2026, 1, 1, tzinfo=UTC)
    )


def provisional_policy(**overrides) -> StockDataQualityPolicy:
    return StockDataQualityPolicy(certification=DatasetCertification.PROVISIONAL, **overrides)


def research_policy(**overrides) -> StockDataQualityPolicy:
    base = {
        "certification": DatasetCertification.RESEARCH,
        "calendar": calendar(),
        "feature_availability": (),
    }
    base.update(overrides)
    return StockDataQualityPolicy(**base)


class TestScenario01UnclassifiedIdentifier:
    def test_alphanumeric_identifier_is_quarantined_as_unclassified(self) -> None:
        frame = panel(
            [
                bar(SESSIONS[0], ticker="0001A0"),
                bar(SESSIONS[1], ticker="000050"),
            ]
        )
        report = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        assert report.quarantined_row_count == 1
        quarantined = report.quarantined.filter(
            pl.col("instrument_id") == "KRX:0001A0"
        )
        assert quarantined[QUALITY_REASON_COLUMN].to_list() == ["unclassified_instrument"]
        assert report.eligible_row_count == 1
        assert "KRX:0001A0" in report.affected_identifiers["unclassified_instrument"]

    def test_master_only_permits_research_eligibility(self) -> None:
        frame = panel(
            [
                bar(SESSIONS[0], ticker="0001A0"),
                bar(SESSIONS[1], ticker="000050"),
            ]
        )
        m = master(common=("000050", "0001A0"))
        report = validate_canonical_stock_panel(
            frame, m, no_action_actions(), research_policy()
        )
        assert report.eligible_row_count == 2
        assert report.quarantined_row_count == 0

    def test_research_without_master_fails(self) -> None:
        frame = panel([bar(SESSIONS[0])])
        with pytest.raises(ValueError, match="InstrumentMasterSnapshot"):
            validate_canonical_stock_panel(frame, None, no_action_actions(), research_policy())


class TestScenario02MarketInvariants:
    def test_zero_open_high_low_is_quarantined_not_repaired(self) -> None:
        frame = panel([bar(SESSIONS[0], open=0.0, high=0.0, low=0.0, close=47700.0)])
        report = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        assert report.quarantined_row_count == 1
        assert report.quarantined[QUALITY_REASON_COLUMN].to_list() == ["non_positive_or_missing_ohlc"]
        assert report.eligible_row_count == 0

    def test_ohlc_ordering_violation_is_reason_coded(self) -> None:
        frame = panel(
            [
                bar(SESSIONS[0], low=120.0, close=105.0),
                bar(SESSIONS[1]),
            ]
        )
        report = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        assert report.quarantined_row_count == 1
        assert report.quarantined[QUALITY_REASON_COLUMN].to_list() == ["invalid_ohlc_ordering"]
        assert report.eligible_row_count == 1

    def test_non_executable_bar_is_reason_coded(self) -> None:
        frame = panel([bar(SESSIONS[0], volume=0.0), bar(SESSIONS[1])])
        report = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        assert report.quarantined_row_count == 1
        assert report.quarantined[QUALITY_REASON_COLUMN].to_list() == ["non_executable_bar"]

    def test_negative_market_cap_is_reason_coded(self) -> None:
        frame = panel([bar(SESSIONS[0], market_cap=-5.0), bar(SESSIONS[1])])
        report = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        assert report.quarantined_row_count == 1
        assert report.quarantined[QUALITY_REASON_COLUMN].to_list() == ["negative_capitalization"]

    def test_valid_rows_remain_eligible(self) -> None:
        frame = panel([bar(day) for day in SESSIONS])
        report = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        assert report.eligible_row_count == 4
        assert report.quarantined_row_count == 0


class TestScenario03ActionIntervals:
    def test_unexplained_discontinuity_is_quarantined_for_derived_returns(self) -> None:
        frame = panel([bar(day) for day in SESSIONS])
        # No interval record between session 1 and 2.
        partial = CorporateActionSnapshot(
            version="partial",
            intervals=(
                CorporateActionInterval(
                    instrument_id=MASTER_ID,
                    previous_session=SESSIONS[2],
                    session=SESSIONS[3],
                    action_code="no_action",
                ),
            ),
            generated_time=datetime(2026, 1, 1, tzinfo=UTC),
        )
        report = validate_canonical_stock_panel(
            frame,
            master(),
            partial,
            research_policy(certification=DatasetCertification.PRODUCTION),
        )
        assert report.reason_counts["uncovered_action_interval"] >= 2
        uncovered = report.quarantined.filter(
            pl.col(QUALITY_REASON_COLUMN) == "uncovered_action_interval"
        )
        assert uncovered.height == report.reason_counts["uncovered_action_interval"]

    def test_no_action_records_keep_all_rows_eligible(self) -> None:
        frame = panel([bar(day) for day in SESSIONS])
        report = validate_canonical_stock_panel(
            frame, master(), no_action_actions(), research_policy()
        )
        assert report.eligible_row_count == 4
        assert report.quarantined_row_count == 0
        assert report.action_coverage["uncovered"] == 0
        assert report.action_coverage["covered"] == 3

    def test_invalid_adjustment_factor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="adjustment_factor"):
            CorporateActionInterval(
                instrument_id=MASTER_ID,
                previous_session=SESSIONS[0],
                session=SESSIONS[1],
                action_code="split",
                adjustment_factor=0.0,
            )


class TestScenario04FeatureAvailability:
    def test_fully_null_column_is_omitted_and_reported(self) -> None:
        frame = panel(
            [
                bar(SESSIONS[0], disclosure_date=None, roe=0.1),
                bar(SESSIONS[1], disclosure_date=None, roe=0.2),
            ]
        )
        report = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        assert "disclosure_date" in report.fully_null_columns
        assert report.column_null_rates["disclosure_date"] == 1.0
        assert report.column_null_rates["roe"] == 0.0

    def test_research_requires_feature_availability_evidence(self) -> None:
        frame = panel([bar(day) for day in SESSIONS])
        with pytest.raises(ValueError, match="KRXSessionCalendar"):
            validate_canonical_stock_panel(
                frame, master(), no_action_actions(),
                StockDataQualityPolicy(certification=DatasetCertification.RESEARCH),
            )

    def test_scenario_research_public_price_panel_does_not_require_actions(self) -> None:
        """SCENARIO_RESEARCH_PUBLIC_PRICE_PANEL: public-price research is not adjusted."""
        frame = panel([bar(day) for day in SESSIONS])

        report = validate_canonical_stock_panel(
            frame,
            master(),
            None,
            research_policy(),
        )

        assert report.eligible_row_count == frame.height
        assert report.quarantined_row_count == 0
        assert report.eligible["action_interval_covered"].null_count() == frame.height

    def test_research_rejects_undocumented_feature_column(self) -> None:
        frame = panel([bar(day, roe=0.1) for day in SESSIONS])
        with pytest.raises(ValueError, match="FeatureAvailabilityRecord"):
            validate_canonical_stock_panel(
                frame,
                master(),
                no_action_actions(),
                research_policy(),
            )

    def test_research_quarantines_non_calendar_session(self) -> None:
        frame = panel([bar(date(2024, 1, 6)), bar(SESSIONS[0])])
        actions = CorporateActionSnapshot(
            version="non-session-actions",
            intervals=(
                CorporateActionInterval(
                    instrument_id=MASTER_ID,
                    previous_session=SESSIONS[0],
                    session=date(2024, 1, 6),
                    action_code="no_action",
                ),
            ),
            generated_time=datetime(2026, 1, 1, tzinfo=UTC),
        )
        report = validate_canonical_stock_panel(
            frame,
            master(),
            actions,
            research_policy(),
        )
        assert report.reason_counts["non_calendar_session"] == 1
        assert report.eligible_row_count == 1


class TestCertificationEvidence:
    def test_scenario_production_action_gate_requires_master_actions_calendar(self) -> None:
        """SCENARIO_PRODUCTION_ACTION_GATE: adjusted evidence remains fail-closed."""
        frame = panel([bar(day) for day in SESSIONS])
        with pytest.raises(ValueError, match="CorporateActionSnapshot"):
            validate_canonical_stock_panel(
                frame, master(), None,
                StockDataQualityPolicy(
                    certification=DatasetCertification.PRODUCTION, calendar=calendar()
                ),
            )

    def test_report_hashes_are_deterministic(self) -> None:
        frame = panel([bar(day) for day in SESSIONS])
        first = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        second = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        assert first.hashes["quality_report"] == second.hashes["quality_report"]
        assert first.to_json_dict() == second.to_json_dict()
        assert first.eligible.equals(second.eligible)


class TestBenchmarkRouting:
    def test_index_rows_are_routed_to_non_equity(self) -> None:
        frame = panel(
            [
                bar(SESSIONS[0], ticker="KOSPI"),
                bar(SESSIONS[1], ticker="000050"),
            ]
        )
        report = validate_canonical_stock_panel(frame, None, None, provisional_policy())
        assert report.non_equity_row_count == 1
        assert report.non_equity[QUALITY_REASON_COLUMN].to_list() == ["index_instrument"]
        assert report.benchmark_routed_identifiers == ("KRX:KOSPI",)
        assert report.eligible_row_count == 1
