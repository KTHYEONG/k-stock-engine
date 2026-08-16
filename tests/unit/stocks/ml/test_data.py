"""NetAlphaResearchData composition: realized-outcome retention and clean predictors."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.data import compose_net_alpha_training_data
from src.stocks.ml.training import _build_label_join
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)

_NARROW_LABEL_COLUMNS = {
    "instrument_id",
    "session",
    "net_alpha_target",
    "label_available_time",
    "risk_residual",
    "reference_cost",
}


def _decision_time() -> datetime:
    return datetime(2024, 12, 31, tzinfo=UTC)


def _assert_narrow_labels_restored(data, horizon: int) -> None:
    label_frame = data.labels_by_horizon[horizon]
    assert set(label_frame.columns) == _NARROW_LABEL_COLUMNS
    join = _build_label_join(data, horizon)
    assert {
        "instrument_id",
        "session",
        "net_alpha_target",
        "label_available_time",
        "risk_residual",
        "reference_cost",
        "open",
        "adtv_20d",
        "volatility_20d",
        "realized_net_return",
    } <= set(join.columns)
    realized = (join["risk_residual"] - join["reference_cost"]).to_list()
    assert join["realized_net_return"].to_list() == pytest.approx(realized)


def test_wide_composition_retains_decimal_realized_outcomes() -> None:
    df = stock_net_alpha_composed_df(n_sessions=30, n_tickers=4, audit_clean=True)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    for horizon in (3, 5):
        _assert_narrow_labels_restored(data, horizon)


def test_long_composition_retains_decimal_realized_outcomes() -> None:
    wide = stock_net_alpha_composed_df(
        n_sessions=10, n_tickers=2, candidate_horizon_sessions=(3, 5)
    )
    parts = [
        wide.select(
            pl.col("session_index"),
            pl.col("session"),
            pl.col("instrument_id"),
            pl.col("open"),
            pl.col("adtv_20d"),
            pl.col("volatility_20d"),
            pl.lit(horizon).alias("horizon_sessions"),
            pl.col(f"net_alpha_{horizon}d_target").alias("net_alpha_target"),
            pl.col(f"label_available_time_{horizon}d").alias("label_available_time"),
            pl.col(f"risk_residual_{horizon}d").alias("risk_residual"),
            pl.col(f"reference_cost_{horizon}d").alias("reference_cost"),
            pl.lit(0.0).alias("gross_return"),
        )
        for horizon in (3, 5)
    ]
    long_frame = pl.concat(parts)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=long_frame.columns),
        frame=long_frame,
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    for horizon in (3, 5):
        _assert_narrow_labels_restored(data, horizon)


def test_feature_frame_is_target_free() -> None:
    df = stock_net_alpha_composed_df(
        n_sessions=10, n_tickers=2, audit_clean=True, label_scale=50.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    for column in data.feature_frame.columns:
        assert not column.startswith(
            ("net_alpha_", "label_available_time_", "risk_residual_", "reference_cost_")
        )
        assert column not in ("horizon_sessions", "net_alpha_target", "risk_residual", "reference_cost")


def test_horizon_universes_are_independent() -> None:
    df = stock_net_alpha_composed_df(n_sessions=40, n_tickers=6, audit_clean=True)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    assert set(data.labels_by_horizon) == {3, 5}
    assert data.labels_by_horizon[3].height > 0
    assert data.labels_by_horizon[5].height > 0


def test_compose_builds_status_coverage_and_join_evidence() -> None:
    df = stock_net_alpha_composed_df(
        n_sessions=40, n_tickers=6, audit_clean=True
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    assert set(data.coverage_by_horizon) == {3, 5}
    for horizon in (3, 5):
        coverage = data.coverage_by_horizon[horizon]
        assert coverage.horizon_sessions == horizon
        assert coverage.decision_rows == data.feature_frame.height
        assert coverage.realized_rows == len(data.labels_by_horizon[horizon])
        assert coverage.status_counts.realized > 0
        assert "outcome_status" in coverage.status_projection.columns
        # Every decision key resolves to exactly one typed state.
        assert (
            coverage.status_counts.realized
            + coverage.status_counts.partial_tail
            + coverage.status_counts.unresolved
            == coverage.decision_rows
        )
    evidence = {e.horizon_sessions: e for e in data.join_evidence}
    for horizon in (3, 5):
        assert evidence[horizon].decision_rows == data.feature_frame.height
        assert evidence[horizon].realized_rows == len(data.labels_by_horizon[horizon])
        assert evidence[horizon].status_counts is not None
        assert evidence[horizon].status_counts.realized == evidence[horizon].realized_rows


def test_feature_frame_never_carries_outcome_status() -> None:
    df = stock_net_alpha_composed_df(
        n_sessions=10, n_tickers=2, audit_clean=True
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    assert "outcome_status" not in data.feature_frame.columns
    for frame in data.labels_by_horizon.values():
        assert "outcome_status" not in frame.columns


def test_future_label_or_status_never_changes_decision_eligibility() -> None:
    """Acceptance 4: removing a future label/status row cannot change the decision frame."""
    df = stock_net_alpha_composed_df(
        n_sessions=30, n_tickers=4, audit_clean=True
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    decision_time = _decision_time()
    full = compose_net_alpha_training_data(snapshot, decision_time, (3, 5))
    # Drop the last session's label/status rows for every instrument: those are
    # the newest (future) rows relative to the decision-time panel of earlier
    # sessions. The decision feature universe and per-horizon realised labels
    # must be identical because decision eligibility never depends on future rows.
    last_session = df["session"].max()
    trimmed = df.filter(pl.col("session") != last_session)
    trimmed_snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=trimmed.columns),
        frame=trimmed,
    )
    trimmed_data = compose_net_alpha_training_data(
        trimmed_snapshot, decision_time, (3, 5)
    )
    assert set(full.labels_by_horizon) == set(trimmed_data.labels_by_horizon)
    for horizon in (3, 5):
        full_labels = full.labels_by_horizon[horizon]
        trimmed_labels = trimmed_data.labels_by_horizon[horizon]
        assert (
            full_labels.filter(pl.col("session") != last_session)
            .sort(["instrument_id", "session"])["session"]
            .to_list()
            == trimmed_labels.sort(["instrument_id", "session"])["session"].to_list()
        )
    # The decision feature universe is identical once the newest session is
    # excluded from both panels.
    assert (
        full.feature_frame.filter(pl.col("session") != last_session)
        .sort(["instrument_id", "session"])["session"]
        .to_list()
        == trimmed_data.feature_frame.sort(["instrument_id", "session"])["session"].to_list()
    )


def test_horizon_outcome_coverage_maps_score_keys_and_segments() -> None:
    from datetime import timedelta

    from src.stocks.ml.data import HorizonOutcomeCoverage

    start = datetime(2024, 1, 1, tzinfo=UTC)
    sessions = [start + timedelta(days=i) for i in range(6)]
    score_keys = pl.DataFrame(
        {
            "instrument_id": [f"KRX:{t:05d}" for t in range(3)] * 6,
            "session": [s for s in sessions for _ in range(3)],
            "oof_segment_id": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    status_frame = pl.DataFrame(
        {
            "instrument_id": [f"KRX:{t:05d}" for t in range(3)] * 6,
            "session": [s for s in sessions for _ in range(3)],
            "outcome_status": ["REALIZED"] * 17 + ["MISSING_EXIT_PRICE"],
        }
    )
    coverage = HorizonOutcomeCoverage.build(
        3, score_keys, status_frame, segment_column="oof_segment_id"
    )
    assert coverage.decision_rows == 18
    assert coverage.realized_rows == 17
    assert coverage.status_counts.realized == 17
    assert coverage.status_counts.unresolved == 1
    assert len(coverage.segment_projection) == 2
    segment_counts = {s.segment_id: s.counts for s in coverage.segment_projection}
    assert segment_counts[0].realized == 9
    assert segment_counts[1].unresolved == 1
    assert "outcome_status" in coverage.status_projection.columns
    assert coverage.to_json()["realized_rows"] == 17


def test_horizon_outcome_coverage_fails_closed_on_unclassified_key() -> None:
    from datetime import timedelta

    from src.stocks.ml.data import HorizonOutcomeCoverage

    start = datetime(2024, 1, 1, tzinfo=UTC)
    score_keys = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001", "KRX:00002"],
            "session": [start, start + timedelta(days=1)],
        }
    )
    # KRX:00002 has no status row: the sidecar must cover the whole universe.
    status_frame = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "session": [start],
            "outcome_status": ["REALIZED"],
        }
    )
    with pytest.raises(ValueError, match="absent from the outcome status sidecar"):
        HorizonOutcomeCoverage.build(3, score_keys, status_frame)


def _readiness_data():
    """Composed wide net-alpha data over a readiness-clean fixture panel."""
    df = stock_net_alpha_composed_df(
        n_sessions=30, n_tickers=4, audit_clean=True
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    return compose_net_alpha_training_data(snapshot, _decision_time(), (3, 5))


def _pin_readiness_data(data):
    """Attach a hash-bound evidence spine, mapping status to resolution kind."""
    from dataclasses import replace

    evidence_by_horizon: dict[int, pl.DataFrame] = {}
    for horizon, status in data.status_by_horizon.items():
        evidence_by_horizon[horizon] = status.select(
            pl.col("instrument_id"),
            pl.col("session"),
            pl.lit(horizon, dtype=pl.Int64).alias("horizon_sessions"),
            pl.lit("policy-hash-v1").alias("policy_hash"),
            pl.when(pl.col("outcome_status") == "REALIZED")
            .then(pl.lit("SCHEDULED_OPEN"))
            .when(pl.col("outcome_status") == "PARTIAL_TAIL")
            .then(pl.lit("PARTIAL_TAIL"))
            .otherwise(pl.lit("CONFIRMED_NO_BAR"))
            .alias("resolution_kind"),
        )
    return replace(
        data,
        evidence_by_horizon=evidence_by_horizon,
        status_provenance="pinned",
    )


def _set_evidence_resolution(data, horizon, key, resolution):
    """Override one key's evidence resolution kind in a pinned readiness data."""
    from dataclasses import replace

    instrument, session = key
    evidence = data.evidence_by_horizon[horizon].with_columns(
        pl.when(
            (pl.col("instrument_id") == instrument)
            & (pl.col("session") == session)
        )
        .then(pl.lit(resolution))
        .otherwise(pl.col("resolution_kind"))
        .alias("resolution_kind")
    )
    return replace(
        data, evidence_by_horizon={**data.evidence_by_horizon, horizon: evidence}
    )


def _early_key(data, tail_count: int = 3) -> tuple[str, object]:
    sessions = data.feature_frame["session"].unique().sort().to_list()
    tail = set(sessions[-tail_count:])
    early = (
        data.feature_frame.filter(~pl.col("session").is_in(sorted(tail)))
        .sort("session")
        .limit(1)
    )
    return str(early["instrument_id"][0]), early["session"][0]


def test_snapshot_outcome_readiness_rejects_source_unavailable() -> None:
    """SCENARIO_UNRESOLVED_OUTCOME_IS_DIAGNOSTIC: a source gap fails the report."""
    from dataclasses import replace

    from src.stocks.ml.contracts import OUTCOME_MISSING_EXIT_PRICE
    from src.stocks.ml.data import assess_snapshot_outcome_readiness

    data = _readiness_data()
    early_id, early_session = _early_key(data)
    broken = data.status_by_horizon[3].with_columns(
        pl.when(
            (pl.col("instrument_id") == early_id)
            & (pl.col("session") == early_session)
        )
        .then(pl.lit(OUTCOME_MISSING_EXIT_PRICE))
        .otherwise(pl.col("outcome_status"))
        .alias("outcome_status")
    )
    data = replace(data, status_by_horizon={**data.status_by_horizon, 3: broken})
    data = _pin_readiness_data(data)
    data = _set_evidence_resolution(
        data, 3, (early_id, early_session), "SOURCE_UNAVAILABLE"
    )
    report = assess_snapshot_outcome_readiness(data, (3, 5))
    assert not report.passed
    horizon3 = next(h for h in report.horizon_results if h.horizon_sessions == 3)
    assert not horizon3.passed
    assert horizon3.source_unavailable_rows == 1
    assert horizon3.unresolved_status_counts.count(OUTCOME_MISSING_EXIT_PRICE) == 1
    assert horizon3.earliest_unresolved_session == early_session
    assert horizon3.realized_rows == horizon3.decision_rows - 1
    horizon5 = next(h for h in report.horizon_results if h.horizon_sessions == 5)
    assert horizon5.passed
    assert report.to_json()["passed"] is False
    assert len(report.to_json()["horizons"]) == 2


def test_snapshot_outcome_readiness_allows_confirmed_no_bar() -> None:
    """A verified structural no-bar stays visible and does not fail the gate."""
    from dataclasses import replace

    from src.stocks.ml.contracts import OUTCOME_MISSING_EXIT_PRICE
    from src.stocks.ml.data import assess_snapshot_outcome_readiness

    data = _readiness_data()
    early_id, early_session = _early_key(data)
    broken = data.status_by_horizon[3].with_columns(
        pl.when(
            (pl.col("instrument_id") == early_id)
            & (pl.col("session") == early_session)
        )
        .then(pl.lit(OUTCOME_MISSING_EXIT_PRICE))
        .otherwise(pl.col("outcome_status"))
        .alias("outcome_status")
    )
    data = replace(data, status_by_horizon={**data.status_by_horizon, 3: broken})
    data = _pin_readiness_data(data)
    report = assess_snapshot_outcome_readiness(data, (3, 5))
    assert report.passed
    horizon3 = next(h for h in report.horizon_results if h.horizon_sessions == 3)
    assert horizon3.passed
    assert horizon3.confirmed_no_bar_rows == 1
    assert horizon3.unresolved_status_counts.to_json() == {}


def test_snapshot_outcome_readiness_allows_terminal_tail_partial_only() -> None:
    """Only a PARTIAL_TAIL confined to the chronological terminal suffix passes."""
    from dataclasses import replace

    from src.stocks.ml.contracts import OUTCOME_PARTIAL_TAIL
    from src.stocks.ml.data import assess_snapshot_outcome_readiness

    data = _readiness_data()
    sessions = data.feature_frame["session"].unique().sort().to_list()
    # The sidecar marks a contiguous suffix: the final horizon sessions plus the
    # scheduled-entry offset session (entry executes at the next open).
    suffix = set(sessions[-4:])
    tail_only = data.status_by_horizon[3].with_columns(
        pl.when(pl.col("session").is_in(sorted(suffix)))
        .then(pl.lit(OUTCOME_PARTIAL_TAIL))
        .otherwise(pl.col("outcome_status"))
        .alias("outcome_status")
    )
    data = replace(data, status_by_horizon={**data.status_by_horizon, 3: tail_only})
    data = _pin_readiness_data(data)
    report = assess_snapshot_outcome_readiness(data, (3,))
    assert report.passed
    horizon3 = report.horizon_results[0]
    assert horizon3.terminal_tail_rows == 16
    assert horizon3.unresolved_status_counts.to_json() == {}


def test_snapshot_outcome_readiness_legacy_inferred_is_diagnostic_only() -> None:
    """SCENARIO_SNAPSHOT_PINS_OUTCOME_EVIDENCE: unpinned provenance fails before OOF."""
    from src.stocks.ml.data import assess_snapshot_outcome_readiness

    data = _readiness_data()
    report = assess_snapshot_outcome_readiness(data, (3, 5))
    assert not report.passed
    assert report.reason == "outcome-provenance-unpinned"
    assert all(not result.passed for result in report.horizon_results)
    assert report.to_json()["reason"] == "outcome-provenance-unpinned"
    assert report.to_json()["passed"] is False


def test_snapshot_outcome_readiness_fails_closed_on_structural_defects() -> None:
    """Duplicate/unknown/uncovered keys, absent horizons, and bad tails raise."""
    from dataclasses import replace

    from src.stocks.ml.contracts import OUTCOME_PARTIAL_TAIL
    from src.stocks.ml.data import assess_snapshot_outcome_readiness

    data = _pin_readiness_data(_readiness_data())
    status3 = data.status_by_horizon[3]
    status5 = data.status_by_horizon[5]

    duplicated = pl.concat([status3, status3.head(1)])
    with pytest.raises(ValueError, match="duplicate decision keys"):
        assess_snapshot_outcome_readiness(
            replace(data, status_by_horizon={3: duplicated, 5: status5}), (3, 5)
        )

    unknown = status3.with_columns(pl.lit("BOGUS").alias("outcome_status"))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        assess_snapshot_outcome_readiness(
            replace(data, status_by_horizon={3: unknown, 5: status5}), (3, 5)
        )

    with pytest.raises(ValueError, match="outcome-status sidecar for horizon 3"):
        assess_snapshot_outcome_readiness(
            replace(data, status_by_horizon={5: status5}), (3, 5)
        )

    sessions = data.feature_frame["session"].unique().sort().to_list()
    misplaced = status3.with_columns(
        pl.when(pl.col("session") == sessions[0])
        .then(pl.lit(OUTCOME_PARTIAL_TAIL))
        .otherwise(pl.col("outcome_status"))
        .alias("outcome_status")
    )
    with pytest.raises(ValueError, match="impossible terminal-tail layout"):
        assess_snapshot_outcome_readiness(
            replace(data, status_by_horizon={3: misplaced, 5: status5}), (3, 5)
        )


def test_scenario_unresolved_outcome_is_diagnostic_reconciliation_queue() -> None:
    """SCENARIO_UNRESOLVED_OUTCOME_IS_DIAGNOSTIC: source gap is a worklist, not cash."""
    from datetime import date, timedelta

    from src.stocks.data.outcome_evidence import (
        RECONCILIATION_ACTION_COLUMN,
        RECONCILIATION_SOURCE_UNAVAILABLE,
        build_missing_exit_reconciliation_report,
        resolve_policy_outcome,
    )
    from src.stocks.data.quality import KRXSessionCalendar
    from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1

    unavailable = {date(2024, 1, 5)}
    evidence = resolve_policy_outcome(
        pl.DataFrame(
            {
                "instrument_id": ["KRX:00001"] * 6,
                "session": [
                    (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i))
                    for i in range(6)
                ],
                "open": [100.0] * 4 + [None, 100.0],
                "sector": ["S1"] * 6,
                "adtv": [1.0e8] * 6,
                "market_cap": [1.0e11] * 6,
                "beta": [1.0] * 6,
                "volatility": [0.02] * 6,
            }
        ),
        KRXSessionCalendar(
            version="fixture",
            sessions=tuple(
                (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)).date()
                for i in range(6)
            ),
            generated_time=datetime(2024, 1, 1, tzinfo=UTC),
        ),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
        unavailable_sessions=unavailable,
    )
    report = build_missing_exit_reconciliation_report(evidence, (3,))
    assert report.source_unavailable_count == 1
    assert report.source_unavailable_requests.height == 1
    request_row = report.source_unavailable_requests.row(0, named=True)
    assert request_row["instrument_id"] == "KRX:00001"
    assert request_row["scheduled_exit_session"] == date(2024, 1, 5)
    action = report.rows.filter(
        pl.col(RECONCILIATION_ACTION_COLUMN) == RECONCILIATION_SOURCE_UNAVAILABLE
    )
    assert action.height == 1


def test_scenario_snapshot_pins_outcome_evidence_no_bar_is_unreconciled() -> None:
    """SCENARIO_SNAPSHOT_PINS_OUTCOME_EVIDENCE: no-bar is not a backfill or a return."""
    from datetime import date, timedelta

    from src.stocks.data.outcome_evidence import (
        RECONCILIATION_ACTION_COLUMN,
        RECONCILIATION_OFFICIAL_OPEN_BACKFILL,
        RECONCILIATION_VERIFIED_TRADING_HALT,
        build_missing_exit_reconciliation_report,
        resolve_policy_outcome,
    )
    from src.stocks.data.quality import KRXSessionCalendar
    from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1

    base = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"] * 6,
            "session": [
                (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i))
                for i in range(6)
            ],
            "open": [100.0] * 4 + [None, 100.0],
            "sector": ["S1"] * 6,
            "adtv": [1.0e8] * 6,
            "market_cap": [1.0e11] * 6,
            "beta": [1.0] * 6,
            "volatility": [0.02] * 6,
        }
    )
    evidence = resolve_policy_outcome(
        base,
        KRXSessionCalendar(
            version="fixture",
            sessions=tuple(
                (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)).date()
                for i in range(6)
            ),
            generated_time=datetime(2024, 1, 1, tzinfo=UTC),
        ),
        horizon_sessions=3,
        policy=SCHEDULED_OPEN_V1,
    )
    report = build_missing_exit_reconciliation_report(evidence, (3,))
    assert report.unreconciled_no_bar_count == 1
    assert report.source_unavailable_count == 0
    assert report.source_unavailable_requests.height == 0
    assert report.official_open_backfill_count == 0

    backfilled = build_missing_exit_reconciliation_report(
        evidence,
        (3,),
        official_bars=pl.DataFrame(
            {
                "instrument_id": ["KRX:00001"],
                "price_date": [date(2024, 1, 5)],
                "open": [105.0],
            }
        ),
    )
    assert backfilled.official_open_backfill_count == 1
    assert backfilled.unreconciled_no_bar_count == 0
    backfill_action = backfilled.rows.filter(
        pl.col(RECONCILIATION_ACTION_COLUMN) == RECONCILIATION_OFFICIAL_OPEN_BACKFILL
    )
    assert backfill_action.height == 1

    halted = build_missing_exit_reconciliation_report(
        evidence,
        (3,),
        corporate_events=pl.DataFrame(
            {
                "instrument_id": ["KRX:00001"],
                "session": [date(2024, 1, 5)],
                "event_kind": ["TRADING_HALT"],
                "corporate_action_event_id": ["halt-2024-01-05"],
            }
        ),
    )
    assert halted.verified_trading_halt_count == 1
    assert halted.unreconciled_no_bar_count == 0
    halt_action = halted.rows.filter(
        pl.col(RECONCILIATION_ACTION_COLUMN) == RECONCILIATION_VERIFIED_TRADING_HALT
    )
    assert halt_action["corporate_action_event_id"].to_list() == ["halt-2024-01-05"]


def test_scenario_snapshot_pins_outcome_evidence_rejects_foreign_evidence() -> None:
    """SCENARIO_SNAPSHOT_PINS_OUTCOME_EVIDENCE: foreign policy fails closed."""
    from src.stocks.ml.data import _horizon_evidence_frame

    data = _readiness_data()
    horizon = 3
    feature = data.feature_frame.select(
        pl.col("instrument_id"), pl.col("session")
    )
    status = data.status_by_horizon[horizon]
    evidence_cols = status.select(
        pl.col("instrument_id"),
        pl.col("session"),
        pl.lit(horizon, dtype=pl.Int64).alias("horizon_sessions"),
        pl.lit("policy-hash-v1").alias("policy_hash"),
        pl.lit("SCHEDULED_OPEN").alias("resolution_kind"),
        pl.col("outcome_status"),
        pl.col("session").dt.date().alias("scheduled_entry_session"),
        pl.col("session").dt.date().alias("scheduled_exit_session"),
        pl.lit("FILLED").alias("entry_disposition"),
        pl.lit("FILLED").alias("exit_disposition"),
    )
    frame = pl.concat(
        [
            evidence_cols,
            evidence_cols.head(1).with_columns(
                pl.lit("foreign-policy-hash").alias("policy_hash")
            ),
        ]
    )
    with pytest.raises(ValueError, match="multiple policy hashes"):
        _horizon_evidence_frame(frame, feature, horizon, has_evidence_columns=True)

    with pytest.raises(ValueError, match="no outcome-evidence partition"):
        _horizon_evidence_frame(
            evidence_cols, feature, 999, has_evidence_columns=True
        )
