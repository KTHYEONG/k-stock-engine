"""Outcome-evidence materialization and publication tests."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src.stocks.data.outcome_evidence import (
    OUTCOME_EVIDENCE_COLUMNS,
    RESOLUTION_CONFIRMED_NO_BAR,
    RESOLUTION_DEFERRED_OPEN,
    RESOLUTION_PARTIAL_TAIL,
    RESOLUTION_SCHEDULED_OPEN,
    RESOLUTION_SOURCE_UNAVAILABLE,
    RESOLUTION_UNSUPPORTED_CORPORATE_ACTION,
    build_outcome_recovery_report,
    build_outcome_status_sidecar,
    build_partitioned_outcome_evidence,
    publish_outcome_evidence_dataset,
    resolve_policy_outcome,
)
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.domain.execution_policy import (
    SCHEDULED_OPEN_V1,
    ExecutionOutcomePolicy,
)
from src.stocks.ml.contracts import OUTCOME_STATUS_COLUMN


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


def test_build_outcome_recovery_report_classifies_and_groups() -> None:
    """Acceptance 2: SOURCE_UNAVAILABLE is backfill work, no-bar is structural."""
    unavailable = {date(2024, 1, 5)}
    source_unavailable = resolve_policy_outcome(
        _base(missing_open_sessions={4}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        unavailable_sessions=unavailable,
    )
    confirmed_no_bar = resolve_policy_outcome(
        _base(missing_open_sessions={5}),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
    )
    unsupported_action = resolve_policy_outcome(
        _base(),
        _calendar(),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        corporate_action_keys={("KRX:00001", date(2024, 1, 3))},
    )
    evidence = pl.concat(
        [source_unavailable, confirmed_no_bar, unsupported_action]
    ).sort(["instrument_id", "session", "horizon_sessions"])

    report = build_outcome_recovery_report(evidence, (3,))

    # Session index 4 feeds the scheduled exit of decision session 0 and the
    # scheduled entry of decision session 3, so both legs are recoverable
    # backfill; session index 5 likewise yields two confirmed no-bars.
    assert report.recoverable_backfill_count == 2
    assert report.confirmed_no_bar_count == 2
    assert report.unsupported_corporate_action_count == 1
    rows = report.rows
    assert set(rows["resolution_kind"].unique().to_list()) == {
        RESOLUTION_SOURCE_UNAVAILABLE,
        RESOLUTION_CONFIRMED_NO_BAR,
        RESOLUTION_UNSUPPORTED_CORPORATE_ACTION,
    }
    unavailable_rows = rows.filter(
        pl.col("resolution_kind") == RESOLUTION_SOURCE_UNAVAILABLE
    )
    assert unavailable_rows["count"].sum() == report.recoverable_backfill_count
    unavailable_row = unavailable_rows.filter(
        pl.col("scheduled_exit_session") == date(2024, 1, 5)
    ).row(0, named=True)
    assert unavailable_row["policy_hash"] == SCHEDULED_OPEN_V1.canonical_hash
    assert unavailable_row["outcome_status"] == "MISSING_EXIT_PRICE"
    assert unavailable_row["entry_partition_hash"] is None
    no_bar_row = rows.filter(
        pl.col("resolution_kind") == RESOLUTION_CONFIRMED_NO_BAR
    ).filter(pl.col("scheduled_exit_session") == date(2024, 1, 6)).row(0, named=True)
    assert no_bar_row["count"] == 1
    action_row = rows.filter(
        pl.col("resolution_kind") == RESOLUTION_UNSUPPORTED_CORPORATE_ACTION
    ).row(0, named=True)
    assert action_row["outcome_status"] == "UNSUPPORTED_CORPORATE_ACTION"
    assert action_row["entry_disposition"] == "FILLED"


def test_build_outcome_recovery_report_rejects_malformed_evidence() -> None:
    evidence = build_partitioned_outcome_evidence(
        _base(n_sessions=15),
        _calendar(),
        horizon_sessions=(3, 5),
        policy=SCHEDULED_OPEN_V1,
    )
    with pytest.raises(ValueError, match="horizon partitions"):
        build_outcome_recovery_report(evidence, (3,))
    with pytest.raises(ValueError, match="unknown resolution kinds"):
        build_outcome_recovery_report(
            evidence.with_columns(pl.lit("BOGUS_KIND").alias("resolution_kind")),
            (3, 5),
        )
    with pytest.raises(ValueError, match="missing recovery columns"):
        build_outcome_recovery_report(evidence.drop("scheduled_entry_session"), (3, 5))


def test_build_outcome_recovery_report_rejects_unknown_disposition() -> None:
    evidence = build_partitioned_outcome_evidence(
        _base(n_sessions=15),
        _calendar(),
        horizon_sessions=(3,),
        policy=SCHEDULED_OPEN_V1,
    )
    with pytest.raises(ValueError, match="unknown entry/exit dispositions"):
        build_outcome_recovery_report(
            evidence.with_columns(
                pl.lit("BOGUS_DISP").alias("exit_disposition")
            ),
            (3,),
        )
