"""Outcome-evidence materialization and publication tests."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from legacy.stocks.data.outcome_evidence import (
    OUTCOME_EVIDENCE_COLUMNS,
    RECONCILIATION_ACTION_COLUMN,
    RECONCILIATION_SOURCE_UNAVAILABLE,
    RECONCILIATION_UNRECONCILED_NO_BAR,
    RECONCILIATION_VERIFIED_DELISTING_OR_SETTLEMENT,
    RESOLUTION_CONFIRMED_NO_BAR,
    RESOLUTION_DEFERRED_OPEN,
    RESOLUTION_PARTIAL_TAIL,
    RESOLUTION_SCHEDULED_OPEN,
    RESOLUTION_SOURCE_UNAVAILABLE,
    RESOLUTION_UNSUPPORTED_CORPORATE_ACTION,
    build_missing_exit_reconciliation_report,
    build_outcome_status_sidecar,
    build_partitioned_outcome_evidence,
    publish_outcome_evidence_dataset,
    resolve_policy_outcome,
)
from legacy.stocks.data.quality import KRXSessionCalendar
from legacy.stocks.domain.execution_policy import (
    SCHEDULED_OPEN_V1,
    ExecutionOutcomePolicy,
)
from legacy.stocks.ml.contracts import OUTCOME_STATUS_COLUMN


def _calendar(n: int = 15, start: datetime | None = None) -> KRXSessionCalendar:
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    sessions = tuple((start + timedelta(days=i)).date() for i in range(n))
    return KRXSessionCalendar(version="fixture", sessions=sessions, generated_time=start)


def _base(
    n_sessions: int = 15,
    *,
    missing_open_sessions: set[int] | None = None,
    instrument: str = "KRX:00001",
) -> pl.DataFrame:
    missing = missing_open_sessions or set()
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "instrument_id": [instrument] * n_sessions,
            "session": [start + timedelta(days=i) for i in range(n_sessions)],
            "open": [None if i in missing else 100.0 for i in range(n_sessions)],
            "sector": ["S1"] * n_sessions,
            "adtv": [1.0e8] * n_sessions,
            "market_cap": [1.0e11] * n_sessions,
            "beta": [1.0] * n_sessions,
            "volatility": [0.02] * n_sessions,
        }
    )


def test_scheduled_resolution_emits_full_evidence_schema() -> None:
    evidence = resolve_policy_outcome(
        _base(), _calendar(), horizon_sessions=3, policy=SCHEDULED_OPEN_V1
    )
    assert list(evidence.columns) == list(OUTCOME_EVIDENCE_COLUMNS)
    assert evidence["policy_id"].unique().to_list() == ["scheduled_open_v1"]
    assert (
        evidence["policy_hash"].unique().to_list()
        == [SCHEDULED_OPEN_V1.canonical_hash]
    )
    realized = evidence.filter(pl.col("outcome_status") == "REALIZED")
    assert realized["resolution_kind"].unique().to_list() == [RESOLUTION_SCHEDULED_OPEN]
    assert realized["entry_delay_sessions"].max() == 0
    assert realized["exit_delay_sessions"].max() == 0
    assert realized.filter(pl.col("label_available_time").is_null()).height == 0


def test_scheduled_missing_exit_open_is_confirmed_no_bar() -> None:
    # decision at session 0: exit at session 4; remove the open at session 4.
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["outcome_status"] == "MISSING_EXIT_PRICE"
    assert row["resolution_kind"] == RESOLUTION_CONFIRMED_NO_BAR
    assert row["actual_exit_session"] is None
    assert row["label_available_time"] is None


def test_source_unavailable_distinguishes_collection_failure() -> None:
    unavailable = {date(2024, 1, 5)}
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        unavailable_sessions=unavailable,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["resolution_kind"] == RESOLUTION_SOURCE_UNAVAILABLE
    assert row["outcome_status"] == "MISSING_EXIT_PRICE"


def test_partial_tail_beyond_calendar_is_tagged() -> None:
    evidence = resolve_policy_outcome(
        _base(n_sessions=5),
        _calendar(n=5),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
    )
    tail = evidence.filter(pl.col("resolution_kind") == RESOLUTION_PARTIAL_TAIL)
    assert not tail.is_empty()
    assert set(tail["outcome_status"].unique().to_list()) == {"PARTIAL_TAIL"}


def test_deferred_policy_fills_first_valid_open_within_bound() -> None:
    deferred = ExecutionOutcomePolicy(
        policy_id="first_tradable_open_v1",
        max_entry_delay_sessions=2,
        max_exit_delay_sessions=2,
    )
    # decision at session 2: scheduled entry at session 3 is the missing open,
    # first valid open is session 4 (delay 1).
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={3}),
        _calendar(),
        horizon_sessions=3,
        policy=deferred,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 3)).row(0, named=True)
    assert row["outcome_status"] == "REALIZED"
    assert row["resolution_kind"] == RESOLUTION_DEFERRED_OPEN
    assert row["entry_delay_sessions"] == 1
    assert row["actual_entry_session"] == date(2024, 1, 5)
    assert row["label_available_time"] is not None


def test_deferred_policy_never_looks_backwards() -> None:
    deferred = ExecutionOutcomePolicy(
        policy_id="first_tradable_open_v1", max_entry_delay_sessions=1
    )
    # missing open at session 3; first valid open at session 4 within bound.
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={3}),
        _calendar(),
        horizon_sessions=3,
        policy=deferred,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 3)).row(0, named=True)
    assert row["actual_entry_session"] == date(2024, 1, 5)
    assert row["resolution_kind"] == RESOLUTION_DEFERRED_OPEN


def test_deferred_policy_expiry_is_unresolved_not_zero_return() -> None:
    deferred = ExecutionOutcomePolicy(
        policy_id="first_tradable_open_v1", max_exit_delay_sessions=1
    )
    # missing exit open at sessions 4 and 5; session 5 is beyond the 1-session bound.
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={4, 5}),
        _calendar(),
        horizon_sessions=3,
        policy=deferred,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["outcome_status"] == "MISSING_EXIT_PRICE"
    assert row["resolution_kind"] == RESOLUTION_CONFIRMED_NO_BAR


def test_unsupported_corporate_action_is_never_synthetic() -> None:
    action_key = {("KRX:00001", date(2024, 1, 1))}
    evidence = resolve_policy_outcome(
        _base(),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        corporate_action_keys=action_key,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["outcome_status"] == "UNSUPPORTED_CORPORATE_ACTION"
    assert row["resolution_kind"] == RESOLUTION_UNSUPPORTED_CORPORATE_ACTION
    assert row["actual_exit_session"] is None


def test_bar_evidence_source_is_authoritative() -> None:
    bars = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"] * 5,
            "price_date": [(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)).date() for i in range(5)],
            "open": [None, 100.0, 101.0, 102.0, None],
        }
    )
    base = _base(n_sessions=3)
    with pytest.raises(ValueError, match="non-positive or non-finite opens"):
        resolve_policy_outcome(
            base, _calendar(n=5), horizon_sessions=1, policy=SCHEDULED_OPEN_V1,
            bar_evidence=bars,
        )


def test_bar_evidence_overlays_matching_base_key_without_dropping_base() -> None:
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        bar_evidence=pl.DataFrame(
            {
                "instrument_id": ["KRX:00001"],
                "price_date": [date(2024, 1, 5)],
                "open": [105.0],
            }
        ),
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["outcome_status"] == "REALIZED"
    assert row["actual_entry_session"] == date(2024, 1, 2)
    assert row["actual_exit_session"] == date(2024, 1, 5)


def test_settled_cash_uses_official_consideration_not_ohlc() -> None:
    """SCENARIO_SETTLED_CASH: a verified settlement realizes a filled position."""
    settlements = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "entitlement_date": [date(2024, 1, 5)],
            "per_share_consideration": [1_000.0],
            "settlement_source_id": ["settle-1"],
        }
    )
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        settlements=settlements,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["outcome_status"] == "REALIZED"
    assert row["resolution_kind"] == "SETTLED_CASH"
    assert row["actual_exit_session"] == date(2024, 1, 5)
    assert row["exit_disposition"] == "SETTLED_CASH"
    assert row["corporate_action_event_id"] == "settle-1"
    assert row["label_available_time"] is not None


def test_settled_cash_requires_positive_official_consideration() -> None:
    settlements = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "entitlement_date": [date(2024, 1, 5)],
            "per_share_consideration": [0.0],
            "settlement_source_id": ["settle-bad"],
        }
    )
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        resolve_policy_outcome(
            _base(missing_open_sessions={4}),
            _calendar(),
            horizon_sessions=3,
            policy=SCHEDULED_OPEN_V1,
            settlements=settlements,
        )


def test_settled_cash_never_overrides_a_verified_exit_open() -> None:
    settlements = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "entitlement_date": [date(2024, 1, 5)],
            "per_share_consideration": [900.0],
            "settlement_source_id": ["settle-2"],
        }
    )
    evidence = resolve_policy_outcome(
        _base(),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        settlements=settlements,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["outcome_status"] == "REALIZED"
    assert row["resolution_kind"] == RESOLUTION_SCHEDULED_OPEN
    assert row["exit_disposition"] == "FILLED"
    assert row["corporate_action_event_id"] is None


def test_hard_exclusion_event_emits_unexecutable_exit() -> None:
    """SCENARIO_UNEXECUTABLE_EXIT: halt/delisting yields a typed non-realized exit."""
    events = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "published_at": [datetime(2024, 1, 1, 12, 0, tzinfo=UTC)],
            "effective_session": [date(2024, 1, 5)],
            "tradability_state": ["DELISTING_OR_SETTLEMENT"],
        }
    )
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        tradability_events=events,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["outcome_status"] == "UNEXECUTABLE_EXIT"
    assert row["resolution_kind"] == "UNEXECUTABLE_EXIT"
    assert row["actual_exit_session"] is None
    assert row["exit_disposition"] == "UNEXECUTABLE_EXIT"
    assert row["label_available_time"] is None


def test_event_published_after_decision_cutoff_does_not_modify() -> None:
    """SCENARIO_PRE_ENTRY_TIMESTAMP_BOUNDARY: a late disclosure never alters a decision."""
    events = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "published_at": [datetime(2024, 1, 3, 12, 0, tzinfo=UTC)],
            "effective_session": [date(2024, 1, 5)],
            "tradability_state": ["DELISTING_OR_SETTLEMENT"],
        }
    )
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        tradability_events=events,
    )
    # The halt was published after the Jan 1 decision, so it must not modify it.
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["outcome_status"] == "MISSING_EXIT_PRICE"
    assert row["resolution_kind"] == RESOLUTION_CONFIRMED_NO_BAR


def test_event_published_before_decision_but_after_exit_is_not_unexecutable() -> None:
    events = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "published_at": [datetime(2024, 1, 6, 12, 0, tzinfo=UTC)],
            "effective_session": [date(2024, 1, 5)],
            "tradability_state": ["ACTIVE_HALT"],
        }
    )
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        tradability_events=events,
    )
    # published after the exit date; cannot make a historical exit unexecutable.
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["resolution_kind"] == RESOLUTION_CONFIRMED_NO_BAR


def test_settlement_takes_precedence_over_hard_exclusion() -> None:
    settlements = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "entitlement_date": [date(2024, 1, 5)],
            "per_share_consideration": [1_000.0],
            "settlement_source_id": ["settle-3"],
        }
    )
    events = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "published_at": [datetime(2024, 1, 1, 12, 0, tzinfo=UTC)],
            "effective_session": [date(2024, 1, 5)],
            "tradability_state": ["DELISTING_OR_SETTLEMENT"],
        }
    )
    evidence = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        tradability_events=events,
        settlements=settlements,
    )
    row = evidence.filter(pl.col("session") == date(2024, 1, 1)).row(0, named=True)
    assert row["outcome_status"] == "REALIZED"
    assert row["resolution_kind"] == "SETTLED_CASH"


def test_settlements_and_events_reject_duplicate_keys() -> None:
    dup_settlements = pl.concat(
        [
            pl.DataFrame(
                {
                    "instrument_id": ["KRX:00001"],
                    "entitlement_date": [date(2024, 1, 5)],
                    "per_share_consideration": [1_000.0],
                    "settlement_source_id": ["a"],
                }
            ),
            pl.DataFrame(
                {
                    "instrument_id": ["KRX:00001"],
                    "entitlement_date": [date(2024, 1, 5)],
                    "per_share_consideration": [900.0],
                    "settlement_source_id": ["b"],
                }
            ),
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        resolve_policy_outcome(
            _base(missing_open_sessions={4}),
            _calendar(),
            horizon_sessions=3,
            policy=SCHEDULED_OPEN_V1,
            settlements=dup_settlements,
        )
    dup_events = pl.concat(
        [
            pl.DataFrame(
                {
                    "instrument_id": ["KRX:00001"],
                    "published_at": [datetime(2024, 1, 1, 12, 0, tzinfo=UTC)],
                    "effective_session": [date(2024, 1, 5)],
                    "tradability_state": ["ACTIVE_HALT"],
                }
            ),
            pl.DataFrame(
                {
                    "instrument_id": ["KRX:00001"],
                    "published_at": [datetime(2024, 1, 1, 12, 0, tzinfo=UTC)],
                    "effective_session": [date(2024, 1, 5)],
                    "tradability_state": ["DELISTING_OR_SETTLEMENT"],
                }
            ),
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        resolve_policy_outcome(
            _base(missing_open_sessions={4}),
            _calendar(),
            horizon_sessions=3,
            policy=SCHEDULED_OPEN_V1,
            tradability_events=dup_events,
        )


def test_partitioned_evidence_and_status_sidecar_align() -> None:
    evidence = build_partitioned_outcome_evidence(
        _base(n_sessions=15),
        _calendar(),
        horizon_sessions=(3, 5),
        policy=SCHEDULED_OPEN_V1,
    )
    assert set(evidence["horizon_sessions"].unique().to_list()) == {3, 5}
    status = build_outcome_status_sidecar(evidence)
    assert set(status.columns) == {"instrument_id", "session", "horizon_sessions", OUTCOME_STATUS_COLUMN}
    assert status.height == evidence.height
    duplicate = status.group_by(["instrument_id", "session", "horizon_sessions"]).len().filter(
        pl.col("len") > 1
    )
    assert duplicate.height == 0


def test_publish_pins_policy_hash_and_rejects_foreign(tmp_path) -> None:
    evidence = build_partitioned_outcome_evidence(
        _base(n_sessions=15),
        _calendar(),
        horizon_sessions=(3,),
        policy=SCHEDULED_OPEN_V1,
    )
    result = publish_outcome_evidence_dataset(
        evidence,
        destination_root=tmp_path / "labels",
        dataset_id="na_outcome_evidence",
        base_panel_hash="base-hash",
        calendar_hash="cal-hash",
        horizon_sessions=(3,),
        policy=SCHEDULED_OPEN_V1,
        generated_time=datetime(2024, 2, 1, tzinfo=UTC),
    )
    assert result.manifest.content_hash
    foreign = evidence.with_columns(
        pl.lit("foreign-hash").alias("policy_hash")
    )
    with pytest.raises(ValueError, match="foreign policy hash"):
        publish_outcome_evidence_dataset(
            foreign,
            destination_root=tmp_path / "labels2",
            dataset_id="na_outcome_evidence",
            base_panel_hash="base-hash",
            calendar_hash="cal-hash",
            horizon_sessions=(3,),
            policy=SCHEDULED_OPEN_V1,
            generated_time=datetime(2024, 2, 1, tzinfo=UTC),
        )


def test_rejects_non_calendar_decision_sessions() -> None:
    base = _base(n_sessions=5).with_columns(
        pl.col("session").cast(pl.Date).alias("session")
    )
    base = base.with_columns(
        pl.when(pl.col("instrument_id").is_not_null())
        .then(pl.lit(None))
        .otherwise(pl.col("session"))
        .alias("session")
    )
    bad = _base(n_sessions=5)
    bad = bad.with_columns(
        (pl.col("session") + timedelta(days=40)).alias("session")
    )
    with pytest.raises(ValueError, match="non-calendar sessions"):
        resolve_policy_outcome(bad, _calendar(), horizon_sessions=1, policy=SCHEDULED_OPEN_V1)


def test_build_missing_exit_reconciliation_report_classifies_and_groups() -> None:
    """SCENARIO_UNRESOLVED_OUTCOME_IS_DIAGNOSTIC: source gap is a backfill request."""
    unavailable = {date(2024, 1, 5)}
    # Decision 0 (Jan 1) exits at Jan 5 (unavailable); decision 2 (Jan 3) is an
    # unsupported corporate action; decision 6 (Jan 7) exits at Jan 11 (missing
    # open). Each classification uses a distinct decision key, so the merged
    # evidence is duplicate-free.
    source_unavailable = resolve_policy_outcome(
        _base(missing_open_sessions={4, 10}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        unavailable_sessions=unavailable,
        corporate_action_keys={("KRX:00001", date(2024, 1, 3))},
    )
    evidence = source_unavailable.filter(
        pl.col("outcome_status").is_in(("MISSING_EXIT_PRICE", "UNSUPPORTED_CORPORATE_ACTION"))
    )
    assert not evidence.is_empty()

    report = build_missing_exit_reconciliation_report(evidence, (3,))

    assert report.source_unavailable_count == 1
    assert report.unreconciled_no_bar_count == 1
    assert report.verified_delisting_or_settlement_count == 1
    assert report.official_open_backfill_count == 0
    assert report.verified_trading_halt_count == 0
    rows = report.rows
    assert set(rows[RECONCILIATION_ACTION_COLUMN].unique().to_list()) == {
        RECONCILIATION_SOURCE_UNAVAILABLE,
        RECONCILIATION_UNRECONCILED_NO_BAR,
        RECONCILIATION_VERIFIED_DELISTING_OR_SETTLEMENT,
    }
    unavailable_row = rows.filter(
        pl.col(RECONCILIATION_ACTION_COLUMN) == RECONCILIATION_SOURCE_UNAVAILABLE
    ).row(0, named=True)
    assert unavailable_row["count"] == 1
    assert unavailable_row["policy_hash"] == SCHEDULED_OPEN_V1.canonical_hash
    assert unavailable_row["outcome_status"] == "MISSING_EXIT_PRICE"
    assert unavailable_row["entry_partition_hash"] is None
    no_bar_row = rows.filter(
        pl.col(RECONCILIATION_ACTION_COLUMN) == RECONCILIATION_UNRECONCILED_NO_BAR
    ).row(0, named=True)
    assert no_bar_row["count"] == 1
    action_row = rows.filter(
        pl.col(RECONCILIATION_ACTION_COLUMN)
        == RECONCILIATION_VERIFIED_DELISTING_OR_SETTLEMENT
    ).row(0, named=True)
    assert action_row["outcome_status"] == "UNSUPPORTED_CORPORATE_ACTION"
    assert action_row["entry_disposition"] == "FILLED"

    # Exact source-unavailable backfill requests are emitted separately.
    requests = report.source_unavailable_requests
    assert {"instrument_id", "scheduled_exit_session", "horizon_sessions"} <= set(
        requests.columns
    )
    assert requests.height == 1
    assert requests["scheduled_exit_session"].to_list() == [date(2024, 1, 5)]


def test_reconciliation_official_open_backfill_and_verified_events() -> None:
    """SCENARIO_SNAPSHOT_PINS_OUTCOME_EVIDENCE: halt and settlement stay distinct."""
    confirmed_no_bar = resolve_policy_outcome(
        _base(missing_open_sessions={5}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
    )
    # Only the filled-entry missing exit (decision session 1, exit Jan 6) is a
    # reconciliation candidate; the entry-side MISSING_ENTRY_PRICE row is not.
    official = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "price_date": [date(2024, 1, 6)],
            "open": [110.0],
        }
    )
    halt = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "session": [date(2024, 1, 6)],
            "event_kind": ["TRADING_HALT"],
            "corporate_action_event_id": ["halt-1"],
        }
    )
    backfill_report = build_missing_exit_reconciliation_report(
        confirmed_no_bar, (3,), official_bars=official
    )
    assert backfill_report.official_open_backfill_count == 1
    assert backfill_report.unreconciled_no_bar_count == 0
    assert backfill_report.source_unavailable_count == 0

    halt_report = build_missing_exit_reconciliation_report(
        confirmed_no_bar, (3,), corporate_events=halt
    )
    assert halt_report.verified_trading_halt_count == 1
    assert halt_report.unreconciled_no_bar_count == 0

    settlement = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "session": [date(2024, 1, 6)],
            "event_kind": ["SETTLEMENT"],
            "corporate_action_event_id": ["settle-1"],
        }
    )
    settle_report = build_missing_exit_reconciliation_report(
        confirmed_no_bar, (3,), corporate_events=settlement
    )
    assert settle_report.verified_delisting_or_settlement_count == 1
    row = settle_report.rows.filter(
        pl.col(RECONCILIATION_ACTION_COLUMN)
        == RECONCILIATION_VERIFIED_DELISTING_OR_SETTLEMENT
    ).row(0, named=True)
    assert row["corporate_action_event_id"] == "settle-1"


def test_reconciliation_rejects_duplicate_or_non_positive_evidence() -> None:
    """SCENARIO_SNAPSHOT_PINS_OUTCOME_EVIDENCE: malformed inputs fail closed."""
    evidence = build_partitioned_outcome_evidence(
        _base(n_sessions=15),
        _calendar(),
        horizon_sessions=(3, 5),
        policy=SCHEDULED_OPEN_V1,
    )
    with pytest.raises(ValueError, match="horizon partitions"):
        build_missing_exit_reconciliation_report(evidence, (3,))
    with pytest.raises(ValueError, match="unknown resolution kinds"):
        build_missing_exit_reconciliation_report(
            evidence.with_columns(pl.lit("BOGUS_KIND").alias("resolution_kind")),
            (3, 5),
        )
    with pytest.raises(ValueError, match="missing reconciliation columns"):
        build_missing_exit_reconciliation_report(
            evidence.drop("scheduled_entry_session"), (3, 5)
        )
    missing = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
    )
    missing5 = resolve_policy_outcome(
        _base(missing_open_sessions={6}),
        _calendar(),
        horizon_sessions=5,
        policy=SCHEDULED_OPEN_V1,
    )
    sample = pl.concat(
        [
            missing.head(1),
            missing5.head(1),
        ]
    )
    doubled = pl.concat([sample, sample])
    with pytest.raises(ValueError, match="duplicate decision keys"):
        build_missing_exit_reconciliation_report(doubled, (3, 5))


def test_reconciliation_rejects_duplicate_official_bar_and_bad_open() -> None:
    """Only a finite positive official open at the exact key may backfill."""
    confirmed = resolve_policy_outcome(
        _base(missing_open_sessions={5}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
    )
    dup_bars = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001", "KRX:00001"],
            "price_date": [date(2024, 1, 6), date(2024, 1, 6)],
            "open": [100.0, 101.0],
        }
    )
    with pytest.raises(ValueError, match=r"duplicate .* keys"):
        build_missing_exit_reconciliation_report(
            confirmed, (3,), official_bars=dup_bars
        )
    bad_open = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "price_date": [date(2024, 1, 6)],
            "open": [0.0],
        }
    )
    with pytest.raises(ValueError, match="non-positive or non-finite opens"):
        build_missing_exit_reconciliation_report(
            confirmed, (3,), official_bars=bad_open
        )
    bad_event = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "session": [date(2024, 1, 6)],
            "event_kind": ["BOGUS_EVENT"],
            "corporate_action_event_id": ["e-1"],
        }
    )
    with pytest.raises(ValueError, match="unknown event kinds"):
        build_missing_exit_reconciliation_report(
            confirmed, (3,), corporate_events=bad_event
        )


def test_build_missing_exit_reconciliation_report_rejects_unknown_disposition() -> None:
    evidence = build_partitioned_outcome_evidence(
        _base(n_sessions=15),
        _calendar(),
        horizon_sessions=(3,),
        policy=SCHEDULED_OPEN_V1,
    )
    with pytest.raises(ValueError, match="unknown entry/exit dispositions"):
        build_missing_exit_reconciliation_report(
            evidence.with_columns(
                pl.lit("BOGUS_DISP").alias("exit_disposition")
            ),
            (3,),
        )
