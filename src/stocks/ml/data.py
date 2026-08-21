"""NetAlphaResearchData composition: feature frame plus per-horizon label frames.

The snapshot's feature frame (one row per ``(instrument_id, decision_session)``)
and the long, ``horizon_sessions``-partitioned label dataset are composed into
``NetAlphaResearchData``. Each horizon is point-in-time left/inner joined with
the feature frame independently; horizons are never inner-joined into a common
universe. Retained and dropped row counts are persisted as join evidence, and
the typed outcome-status sidecar is mapped to one status per decision key via
:class:`HorizonOutcomeCoverage`.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import polars as pl

from src.core.datasets import DatasetManifest
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.data.direct import MlMarketData
from src.stocks.data.lineage import ResearchDataBundle
from src.stocks.data.outcome_evidence import (
    RESOLUTION_CONFIRMED_NO_BAR,
    RESOLUTION_KIND_VOCABULARY,
    RESOLUTION_SOURCE_UNAVAILABLE,
    RESOLUTION_UNEXECUTABLE_EXIT,
)
from src.stocks.ml.contracts import (
    CANONICAL_FEATURE_SET,
    OUTCOME_MISSING_EXIT_PRICE,
    OUTCOME_PARTIAL_TAIL,
    OUTCOME_REALIZED,
    OUTCOME_STATUS_COLUMN,
    OUTCOME_STATUS_VOCABULARY,
    HorizonJoinEvidence,
    NetAlphaResearchData,
    OutcomeStatusCounts,
    SegmentOutcomeCounts,
)
from src.stocks.ml.features import stock_net_alpha_v1_roles
from src.stocks.ml.labels import (
    AVAILABLE_COLUMN,
    ID_COLUMN,
    REFERENCE_COST_COLUMN,
    RISK_RESIDUAL_COLUMN,
    TARGET_COLUMN,
)

_FEATURE_SESSION = "session"
_FEATURE_PREFIX = "feature__"

logger = logging.getLogger("stocks.ml.data")

_RESOLUTION_KIND_VALUES = RESOLUTION_KIND_VOCABULARY

OUTCOME_PROVENANCE_UNPINNED = "outcome-provenance-unpinned"


@dataclass(frozen=True, slots=True)
class CoverageRange:
    """Bounded coverage range for provenance projection."""

    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_value <= self.max_value <= 1.0):
            raise ValueError("coverage range must be in [0,1] with min <= max")


@dataclass(frozen=True, slots=True)
class SnapshotEconomicProvenance:
    """Immutable provenance projection for economic evidence pinning.

    Records status/evidence hashes, coverage, and a deterministic reason
    when the snapshot lacks a complete hash-bound outcome-evidence sidecar.
    """

    status_hash: str | None
    evidence_hash: str | None
    coverage: CoverageRange | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class HorizonOutcomeCoverage:
    """Vectorized per-horizon outcome coverage over the decision score keys.

    Built with Polars joins/group-bys only (never a Python row loop over the
    panel): every horizon score key is mapped to exactly one typed status and
    the bounded aggregate counts plus a per-segment projection are exposed.
    ``status_projection`` is the immutable ``(instrument_id, session,
    outcome_status)`` lookup consumed by the policy replay.
    """

    horizon_sessions: int
    status_projection: pl.DataFrame
    decision_rows: int
    realized_rows: int
    status_counts: OutcomeStatusCounts
    segment_projection: tuple[SegmentOutcomeCounts, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        required = (ID_COLUMN, "session", OUTCOME_STATUS_COLUMN)
        missing = [c for c in required if c not in self.status_projection.columns]
        if missing:
            raise ValueError(
                f"outcome coverage status projection missing columns {missing}"
            )
        if self.decision_rows < 0 or self.realized_rows < 0:
            raise ValueError("outcome coverage decision/realised rows must be non-negative")
        if self.status_projection.filter(
            pl.col(OUTCOME_STATUS_COLUMN).is_null()
        ).height:
            raise ValueError("outcome coverage status projection contains null states")

    @classmethod
    def build(
        cls,
        horizon_sessions: int,
        score_keys: pl.DataFrame,
        status_frame: pl.DataFrame,
        segment_column: str | None = None,
    ) -> HorizonOutcomeCoverage:
        """Map every score key to one typed status and aggregate counts.

        ``score_keys`` carries at least ``instrument_id``/``session`` (and an
        optional segment identity); ``status_frame`` carries one
        ``outcome_status`` per decision key. A score key absent from the status
        sidecar fails closed with ``ValueError`` because the sidecar must cover
        the whole decision universe.
        """
        missing = [c for c in (ID_COLUMN, "session") if c not in score_keys.columns]
        if missing:
            raise ValueError(f"score keys missing columns {missing}")
        status_columns = [ID_COLUMN, "session", OUTCOME_STATUS_COLUMN]
        missing_status = [c for c in status_columns if c not in status_frame.columns]
        if missing_status:
            raise ValueError(f"status frame missing columns {missing_status}")
        status_frame = status_frame.select(*status_columns).unique(
            subset=[ID_COLUMN, "session"], keep="first"
        )
        unknown = status_frame.filter(
            ~pl.col(OUTCOME_STATUS_COLUMN).is_null()
            & ~pl.col(OUTCOME_STATUS_COLUMN).is_in(list(OUTCOME_STATUS_VOCABULARY))
        )
        if not unknown.is_empty():
            raise ValueError("status frame contains states outside the vocabulary")
        projection = score_keys.select(ID_COLUMN, "session").unique(
            subset=[ID_COLUMN, "session"], keep="first"
        ).join(status_frame, on=[ID_COLUMN, "session"], how="left")
        missing_rows = projection.filter(pl.col(OUTCOME_STATUS_COLUMN).is_null())
        if not missing_rows.is_empty():
            raise ValueError(
                f"horizon {horizon_sessions} score keys absent from the outcome "
                f"status sidecar: {missing_rows.height} keys"
            )
        counts_frame = (
            projection.group_by(OUTCOME_STATUS_COLUMN).len().sort(OUTCOME_STATUS_COLUMN)
        )
        counts = OutcomeStatusCounts.from_mapping(
            {
                str(row[OUTCOME_STATUS_COLUMN]): int(row["len"])
                for row in counts_frame.iter_rows(named=True)
            }
        )
        decision_rows = int(
            score_keys.select(ID_COLUMN, "session").unique().height
        )
        realized_rows = counts.realized
        segment_projection: tuple[SegmentOutcomeCounts, ...] = ()
        if segment_column is not None:
            if segment_column not in score_keys.columns:
                raise ValueError(
                    f"segment column {segment_column!r} missing from score keys"
                )
            segments: list[SegmentOutcomeCounts] = []
            for segment_key, frame in projection.join(
                score_keys.select(ID_COLUMN, "session", segment_column).unique(
                    subset=[ID_COLUMN, "session"], keep="first"
                ),
                on=[ID_COLUMN, "session"],
                how="left",
            ).partition_by(
                segment_column, maintain_order=True, as_dict=True
            ).items():
                segment_counts = (
                    frame.group_by(OUTCOME_STATUS_COLUMN).len().sort(OUTCOME_STATUS_COLUMN)
                )
                segments.append(
                    SegmentOutcomeCounts(
                        segment_id=int(segment_key[0]),
                        counts=OutcomeStatusCounts.from_mapping(
                            {
                                str(row[OUTCOME_STATUS_COLUMN]): int(row["len"])
                                for row in segment_counts.iter_rows(named=True)
                            }
                        ),
                    )
                )
            segment_projection = tuple(sorted(segments, key=lambda s: s.segment_id))
        return cls(
            horizon_sessions=horizon_sessions,
            status_projection=projection.select(*status_columns),
            decision_rows=decision_rows,
            realized_rows=realized_rows,
            status_counts=counts,
            segment_projection=segment_projection,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "horizon_sessions": int(self.horizon_sessions),
            "decision_rows": int(self.decision_rows),
            "realized_rows": int(self.realized_rows),
            "status_counts": self.status_counts.to_json(),
            "segments": [segment.to_json() for segment in self.segment_projection],
        }


@dataclass(frozen=True, slots=True)
class HorizonSnapshotReadiness:
    """One horizon's bounded data-provenance readiness result.

    ``decision_rows`` counts every feature decision key for the horizon,
    ``realized_rows`` the ``REALIZED`` keys, ``terminal_tail_rows`` the keys in
    the chronological terminal suffix, ``confirmed_no_bar_rows`` the keys whose
    evidence resolves to a verified structural no-bar (allowed at the data
    gate, visible but never economic evidence), and
    ``source_unavailable_rows`` the keys whose source response was unavailable
    (a hard data gap that fails the gate). ``unresolved_status_counts`` is the
    sorted per-status counts of every failing unresolved state, and
    ``earliest_unresolved_session`` the chronologically first such session.
    """

    horizon_sessions: int
    decision_rows: int
    realized_rows: int
    terminal_tail_rows: int
    unresolved_status_counts: OutcomeStatusCounts
    earliest_unresolved_session: datetime | None
    passed: bool
    confirmed_no_bar_rows: int = 0
    source_unavailable_rows: int = 0

    def __post_init__(self) -> None:
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        for name in (
            "decision_rows",
            "realized_rows",
            "terminal_tail_rows",
            "confirmed_no_bar_rows",
            "source_unavailable_rows",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_json(self) -> dict[str, object]:
        return {
            "horizon_sessions": int(self.horizon_sessions),
            "decision_rows": int(self.decision_rows),
            "realized_rows": int(self.realized_rows),
            "terminal_tail_rows": int(self.terminal_tail_rows),
            "confirmed_no_bar_rows": int(self.confirmed_no_bar_rows),
            "source_unavailable_rows": int(self.source_unavailable_rows),
            "unresolved_status_counts": self.unresolved_status_counts.to_json(),
            "earliest_unresolved_session": (
                self.earliest_unresolved_session.isoformat()
                if self.earliest_unresolved_session is not None
                else None
            ),
            "passed": bool(self.passed),
        }


@dataclass(frozen=True, slots=True)
class SnapshotOutcomeReadiness:
    """Immutable snapshot-wide data-provenance report, one result per horizon.

    ``reason`` carries the deterministic no-trade reason when the whole report
    fails the provenance gate before any horizon classification, e.g.
    ``outcome-provenance-unpinned`` for a legacy-inferred status spine without
    a hash-bound outcome-evidence reference.
    """

    horizon_results: tuple[HorizonSnapshotReadiness, ...]
    passed: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.horizon_results:
            raise ValueError("readiness report requires at least one horizon result")
        if self.passed != all(result.passed for result in self.horizon_results):
            raise ValueError(
                "readiness report passed flag disagrees with the horizon results"
            )
        if self.reason and self.passed:
            raise ValueError("a passed readiness report cannot carry a failure reason")

    def to_json(self) -> dict[str, object]:
        return {
            "passed": bool(self.passed),
            "reason": self.reason,
            "horizons": [result.to_json() for result in self.horizon_results],
        }


def assess_outcome_readiness(
    decision_keys: pl.DataFrame,
    status_frame: pl.DataFrame,
    candidate_horizon_sessions: tuple[int, ...],
    *,
    evidence: pl.DataFrame | None = None,
) -> SnapshotOutcomeReadiness:
    """Vectorized data-provenance readiness over a raw decision/status spine.

    ``decision_keys`` carries one ``instrument_id``/``session`` row per decision
    key; ``status_frame`` is the long, ``horizon_sessions``-partitioned
    outcome-status sidecar. When the hash-bound ``evidence`` projection
    (``instrument_id``, ``session``, ``horizon_sessions``, ``policy_hash``,
    ``resolution_kind``) is supplied, a key whose evidence resolves to
    ``CONFIRMED_NO_BAR`` is a verified structural no-bar: it stays visible to
    execution replay and never fails this data gate, while ``SOURCE_UNAVAILABLE``
    or any other unresolved state (unsupported action, unproven key) fails it.
    Without evidence the gate is strict: every non-terminal unresolved key must
    be ``REALIZED``. Structural sidecar defects (missing columns, unknown states,
    duplicate or uncovered keys, absent horizon partitions, malformed evidence,
    or ``PARTIAL_TAIL`` outside the terminal suffix) raise ``ValueError`` and
    are never converted into an economic verdict.
    """
    if not candidate_horizon_sessions:
        raise ValueError("candidate_horizon_sessions must be non-empty")
    if tuple(candidate_horizon_sessions) != tuple(
        sorted(set(candidate_horizon_sessions))
    ):
        raise ValueError("candidate_horizon_sessions must be strictly ascending and unique")
    required_keys = (ID_COLUMN, "session")
    missing_keys = [c for c in required_keys if c not in decision_keys.columns]
    if missing_keys:
        raise ValueError(f"readiness decision keys missing columns {missing_keys}")
    required_status = (
        ID_COLUMN,
        "session",
        "horizon_sessions",
        OUTCOME_STATUS_COLUMN,
    )
    missing_status = [c for c in required_status if c not in status_frame.columns]
    if missing_status:
        raise ValueError(f"readiness status sidecar missing columns {missing_status}")

    keys = decision_keys.select(*required_keys).unique(
        subset=[ID_COLUMN, "session"], keep="first"
    )
    if keys.is_empty():
        raise ValueError("readiness requires a non-empty decision universe")
    snapshot_sessions = keys["session"].unique().sort().to_list()

    unknown = status_frame.filter(
        ~pl.col(OUTCOME_STATUS_COLUMN).is_in(list(OUTCOME_STATUS_VOCABULARY))
    )
    if not unknown.is_empty():
        raise ValueError("readiness status sidecar contains states outside the vocabulary")
    duplicates = (
        status_frame.group_by([ID_COLUMN, "session", "horizon_sessions"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        raise ValueError("readiness status sidecar contains duplicate decision keys")
    missing_horizons = sorted(
        set(candidate_horizon_sessions)
        - set(status_frame["horizon_sessions"].unique().to_list())
    )
    if missing_horizons:
        raise ValueError(
            f"readiness status sidecar lacks horizon partitions {missing_horizons}"
        )

    evidence_by_horizon: pl.DataFrame | None = None
    if evidence is not None and not evidence.is_empty():
        required_evidence = (
            ID_COLUMN,
            "session",
            "horizon_sessions",
            "policy_hash",
            "resolution_kind",
        )
        missing_evidence = [c for c in required_evidence if c not in evidence.columns]
        if missing_evidence:
            raise ValueError(
                f"readiness outcome evidence missing columns {missing_evidence}"
            )
        unknown_kind = evidence.filter(
            ~pl.col("resolution_kind").is_in(list(_RESOLUTION_KIND_VALUES))
        )
        if not unknown_kind.is_empty():
            raise ValueError(
                "readiness outcome evidence contains unknown resolution kinds"
            )
        evidence_by_horizon = evidence

    tail_ok_statuses = (OUTCOME_REALIZED, OUTCOME_PARTIAL_TAIL)
    results: list[HorizonSnapshotReadiness] = []
    for horizon in candidate_horizon_sessions:
        status_rows = status_frame.filter(pl.col("horizon_sessions") == horizon)
        projection = keys.join(
            status_rows.select(*required_status),
            on=[ID_COLUMN, "session"],
            how="left",
        )
        uncovered = projection.filter(pl.col(OUTCOME_STATUS_COLUMN).is_null())
        if not uncovered.is_empty():
            raise ValueError(
                f"horizon {horizon} decision keys absent from the outcome-status "
                f"sidecar: {uncovered.height}"
            )
        # PARTIAL_TAIL is emitted only where a decision's scheduled entry/exit
        # falls beyond the calendar, a contiguous chronological suffix. Derive
        # that suffix from the sidecar; any PARTIAL_TAIL outside it (e.g. in the
        # middle of history) is an impossible layout.
        tail_sessions = sorted(
            projection.filter(pl.col(OUTCOME_STATUS_COLUMN) == OUTCOME_PARTIAL_TAIL)[
                "session"
            ]
            .unique()
            .to_list()
        )
        if tail_sessions:
            suffix_start = snapshot_sessions.index(tail_sessions[0])
            expected_suffix = set(snapshot_sessions[suffix_start:])
            if set(tail_sessions) != expected_suffix:
                raise ValueError(
                    f"horizon {horizon} PARTIAL_TAIL keys outside the chronological "
                    f"terminal suffix (impossible terminal-tail layout)"
                )
            in_tail = pl.col("session").is_in(sorted(expected_suffix))
        else:
            in_tail = pl.lit(False)
        projection = projection.with_columns(in_tail.alias("__in_tail"))

        not_ok = ~pl.col(OUTCOME_STATUS_COLUMN).is_in(list(tail_ok_statuses))
        confirmed_no_bar = pl.lit(False)
        unexecutable_exit = pl.lit(False)
        missing_decision_input = pl.lit(False)
        source_unavailable = pl.lit(False)
        if evidence_by_horizon is not None:
            horizon_evidence = evidence_by_horizon.filter(
                pl.col("horizon_sessions") == horizon
            ).select(ID_COLUMN, "session", "resolution_kind")
            projection = projection.join(
                horizon_evidence, on=[ID_COLUMN, "session"], how="left"
            )
            confirmed_no_bar = (
                pl.col("resolution_kind") == RESOLUTION_CONFIRMED_NO_BAR
            ).fill_null(False)
            unexecutable_exit = (
                pl.col("resolution_kind") == RESOLUTION_UNEXECUTABLE_EXIT
            ).fill_null(False)
            missing_decision_input = (
                (pl.col(OUTCOME_STATUS_COLUMN) == "MISSING_DECISION_INPUT")
                & (pl.col("resolution_kind") == "SCHEDULED_OPEN")
            ).fill_null(False)
            source_unavailable = (
                pl.col("resolution_kind") == RESOLUTION_SOURCE_UNAVAILABLE
            ).fill_null(False)
        unresolved = projection.filter(
            not_ok & ~confirmed_no_bar & ~unexecutable_exit & ~missing_decision_input
        )
        unresolved_counts = _status_counts(unresolved, horizon)
        assert unresolved_counts is not None
        results.append(
            HorizonSnapshotReadiness(
                horizon_sessions=int(horizon),
                decision_rows=int(projection.height),
                realized_rows=int(
                    projection.filter(
                        pl.col(OUTCOME_STATUS_COLUMN) == OUTCOME_REALIZED
                    ).height
                ),
                terminal_tail_rows=int(projection.filter(pl.col("__in_tail")).height),
                unresolved_status_counts=unresolved_counts,
                earliest_unresolved_session=(
                    cast(datetime, unresolved["session"].min())
                    if not unresolved.is_empty()
                    else None
                ),
                passed=unresolved.is_empty(),
                confirmed_no_bar_rows=(
                    int(projection.filter(not_ok & confirmed_no_bar).height)
                    if evidence_by_horizon is not None
                    else 0
                ),
                source_unavailable_rows=(
                    int(projection.filter(not_ok & source_unavailable).height)
                    if evidence_by_horizon is not None
                    else 0
                ),
            )
        )
    return SnapshotOutcomeReadiness(
        horizon_results=tuple(results),
        passed=all(result.passed for result in results),
    )


def assess_snapshot_outcome_readiness(
    data: NetAlphaResearchData,
    candidate_horizon_sessions: tuple[int, ...],
) -> SnapshotOutcomeReadiness:
    """Composed-data wrapper delegating to :func:`assess_outcome_readiness`.

    Rebuilds the long status sidecar and the long outcome-evidence projection
    from ``data.status_by_horizon`` / ``data.evidence_by_horizon`` and evaluates
    the data-provenance gate over the composed decision universe. A
    legacy-inferred spine (``data.status_provenance == "legacy-inferred"``) has
    no hash-bound evidence to distinguish confirmed no-bars from collection
    gaps: it fails the gate with ``outcome-provenance-unpinned`` and stays
    diagnostic-only, exactly like the legacy ``run8`` snapshot. An absent
    per-horizon sidecar raises ``ValueError``.

    The economic provenance is computed via :func:`snapshot_economic_provenance`
    and its reason is propagated when the provenance gate fails.
    """
    provenance = snapshot_economic_provenance(data, candidate_horizon_sessions)
    if provenance.reason is not None:
        logger.info(
            "snapshot economic provenance: %s (status_hash=%s, evidence_hash=%s)",
            provenance.reason,
            provenance.status_hash,
            provenance.evidence_hash,
        )
    status_frames: list[pl.DataFrame] = []
    for horizon in candidate_horizon_sessions:
        status_rows = data.status_by_horizon.get(horizon)
        if status_rows is None:
            raise ValueError(
                f"readiness requires an outcome-status sidecar for horizon {horizon}"
            )
        status_frames.append(
            status_rows.select(ID_COLUMN, "session", OUTCOME_STATUS_COLUMN).with_columns(
                pl.lit(horizon, dtype=pl.Int64).alias("horizon_sessions")
            )
        )
    status_frame = pl.concat(status_frames)

    evidence_frames: list[pl.DataFrame] = []
    for horizon in candidate_horizon_sessions:
        evidence_rows = data.evidence_by_horizon.get(horizon)
        if evidence_rows is not None and not evidence_rows.is_empty():
            evidence_frames.append(
                evidence_rows.select(
                    ID_COLUMN, "session", "policy_hash", "resolution_kind"
                ).with_columns(pl.lit(horizon, dtype=pl.Int64).alias("horizon_sessions"))
            )
    evidence = (
        pl.concat(evidence_frames)
        if evidence_frames
        else pl.DataFrame(
            schema={
                ID_COLUMN: pl.Utf8,
                "session": pl.Datetime("us", "UTC"),
                "policy_hash": pl.Utf8,
                "resolution_kind": pl.Utf8,
                "horizon_sessions": pl.Int64,
            }
        )
    )

    if (
        data.status_provenance != "pinned"
        or not evidence_frames
        or len(evidence_frames) != len(candidate_horizon_sessions)
    ):
        logger.warning(
            "snapshot outcome provenance is %s (status pinned=%s, evidence "
            "horizons=%d/%d); publishing %s",
            data.status_provenance,
            data.status_provenance == "pinned",
            len(evidence_frames),
            len(candidate_horizon_sessions),
            OUTCOME_PROVENANCE_UNPINNED,
        )
        return _unpinned_provenance_readiness(
            data.feature_frame.select(ID_COLUMN, "session"),
            status_frame,
            candidate_horizon_sessions,
            reason=OUTCOME_PROVENANCE_UNPINNED,
        )
    readiness = assess_outcome_readiness(
        data.feature_frame.select(ID_COLUMN, "session"),
        status_frame,
        candidate_horizon_sessions,
        evidence=evidence,
    )
    return readiness


def snapshot_economic_provenance(
    data: NetAlphaResearchData,
    candidate_horizon_sessions: tuple[int, ...],
) -> SnapshotEconomicProvenance:
    """Project economic provenance: status/evidence hashes, coverage, reason.

    A snapshot that lacks a complete hash-bound outcome_evidence sidecar
    returns the deterministic reason ``outcome-evidence-unpinned`` before
    model fitting; it must not be reported as ``no-horizon-evidence``.
    """
    status_frames: list[pl.DataFrame] = []
    for horizon in candidate_horizon_sessions:
        status_rows = data.status_by_horizon.get(horizon)
        if status_rows is not None:
            status_frames.append(status_rows)
    status_hash: str | None = None
    if status_frames and data.status_provenance == "pinned":
        combined = pl.concat(status_frames)
        status_hash = hashlib.sha256(
            combined.select(ID_COLUMN, "session", OUTCOME_STATUS_COLUMN)
            .sort([ID_COLUMN, "session"])
            .hash_rows(seed=0)
            .to_numpy()
            .tobytes()
        ).hexdigest()

    evidence_frames: list[pl.DataFrame] = []
    for horizon in candidate_horizon_sessions:
        evidence_rows = data.evidence_by_horizon.get(horizon)
        if evidence_rows is not None and not evidence_rows.is_empty():
            evidence_frames.append(evidence_rows)
    evidence_hash: str | None = None
    if evidence_frames and len(evidence_frames) == len(candidate_horizon_sessions):
        combined_evidence = pl.concat(evidence_frames)
        evidence_hash = hashlib.sha256(
            combined_evidence
            .select(ID_COLUMN, "session", "policy_hash", "resolution_kind")
            .sort([ID_COLUMN, "session"])
            .hash_rows(seed=0)
            .to_numpy()
            .tobytes()
        ).hexdigest()

    coverage: CoverageRange | None = None
    if status_hash is not None and evidence_hash is not None:
        coverage = CoverageRange(min_value=0.0, max_value=1.0)

    reason: str | None = None
    if status_hash is None or evidence_hash is None:
        reason = OUTCOME_PROVENANCE_UNPINNED

    return SnapshotEconomicProvenance(
        status_hash=status_hash,
        evidence_hash=evidence_hash,
        coverage=coverage,
        reason=reason,
    )


def _unpinned_provenance_readiness(
    decision_keys: pl.DataFrame,
    status_frame: pl.DataFrame,
    candidate_horizon_sessions: tuple[int, ...],
    *,
    reason: str,
) -> SnapshotOutcomeReadiness:
    """Diagnostic-only report for a legacy-inferred spine (never promotable).

    Reuses the structural validation and the terminal-tail layout check of
    :func:`assess_outcome_readiness` without evidence, then marks every horizon
    failed with the deterministic provenance reason. The unresolved counts stay
    diagnostic; the report can never pass.
    """
    snapshot_sessions = sorted(decision_keys["session"].unique().to_list())
    tail_ok_statuses = (OUTCOME_REALIZED, OUTCOME_PARTIAL_TAIL)
    results: list[HorizonSnapshotReadiness] = []
    for horizon in candidate_horizon_sessions:
        required = (ID_COLUMN, "session", "horizon_sessions", OUTCOME_STATUS_COLUMN)
        status_rows = status_frame.filter(pl.col("horizon_sessions") == horizon)
        projection = decision_keys.select(ID_COLUMN, "session").unique(
            subset=[ID_COLUMN, "session"], keep="first"
        ).join(
            status_rows.select(*required),
            on=[ID_COLUMN, "session"],
            how="left",
        )
        uncovered = projection.filter(pl.col(OUTCOME_STATUS_COLUMN).is_null())
        if not uncovered.is_empty():
            raise ValueError(
                f"horizon {horizon} decision keys absent from the outcome-status "
                f"sidecar: {uncovered.height}"
            )
        tail_sessions = sorted(
            projection.filter(pl.col(OUTCOME_STATUS_COLUMN) == OUTCOME_PARTIAL_TAIL)[
                "session"
            ]
            .unique()
            .to_list()
        )
        if tail_sessions:
            suffix_start = snapshot_sessions.index(tail_sessions[0])
            expected_suffix = set(snapshot_sessions[suffix_start:])
            if set(tail_sessions) != expected_suffix:
                raise ValueError(
                    f"horizon {horizon} PARTIAL_TAIL keys outside the chronological "
                    f"terminal suffix (impossible terminal-tail layout)"
                )
        unresolved = projection.filter(
            ~pl.col(OUTCOME_STATUS_COLUMN).is_in(list(tail_ok_statuses))
        )
        unresolved_counts = _status_counts(unresolved, horizon)
        assert unresolved_counts is not None
        results.append(
            HorizonSnapshotReadiness(
                horizon_sessions=int(horizon),
                decision_rows=int(projection.height),
                realized_rows=int(
                    projection.filter(
                        pl.col(OUTCOME_STATUS_COLUMN) == OUTCOME_REALIZED
                    ).height
                ),
                terminal_tail_rows=len(tail_sessions),
                unresolved_status_counts=unresolved_counts,
                earliest_unresolved_session=(
                    cast(datetime, unresolved["session"].min())
                    if not unresolved.is_empty()
                    else None
                ),
                passed=False,
            )
        )
    return SnapshotOutcomeReadiness(
        horizon_results=tuple(results),
        passed=False,
        reason=reason,
    )


def _reject_feature_set(snapshot: DatasetSnapshot | ResearchDataBundle) -> None:
    """Fail closed unless the composed snapshot is a canonical net-alpha panel."""
    feature_set = snapshot.manifest.feature_set
    if feature_set != CANONICAL_FEATURE_SET:
        raise ValueError(
            f"train accepts only a net-alpha snapshot (feature_set="
            f"{CANONICAL_FEATURE_SET!r}); got {feature_set!r}. Materialize a "
            "net-alpha snapshot via `python -m src.stocks.cli.build_research "
            "--pipeline net-alpha`."
        )


def compose_net_alpha_training_data(
    snapshot: DatasetSnapshot | ResearchDataBundle,
    decision_time: datetime,
    candidate_horizon_sessions: tuple[int, ...],
) -> NetAlphaResearchData:
    """Compose a feature frame and independent per-horizon label frames.

    The composed frame carries the feature sources plus label columns. The
    feature frame (one row per ``(instrument_id, session)``) is extracted
    without label/target columns; each candidate horizon's label rows are
    point-in-time filtered to ``label_available_time <= decision_time`` and
    joined independently, preserving per-horizon universes. Retained horizon
    labels are narrow (identity, target, availability, realized outcomes);
    execution columns are late-bound by the trainer at fit time.

    Args:
        snapshot: the immutable net-alpha snapshot (feature_set
            ``stock_net_alpha_v1``).
        decision_time: the decision time; every label used must be available at
            or before it.
        candidate_horizon_sessions: the pre-registered discovery grid.

    Returns:
        ``NetAlphaResearchData`` with ``labels_by_horizon`` keyed by horizon and
        per-horizon join evidence.
    """
    _reject_feature_set(snapshot)
    if not candidate_horizon_sessions:
        raise ValueError("candidate_horizon_sessions must be non-empty")
    if tuple(candidate_horizon_sessions) != tuple(
        sorted(set(candidate_horizon_sessions))
    ):
        raise ValueError("candidate_horizon_sessions must be strictly ascending and unique")

    frame = snapshot.frame
    identity = (ID_COLUMN, _FEATURE_SESSION)
    if not all(c in frame.columns for c in identity):
        raise ValueError(f"net-alpha snapshot frame missing identity columns {identity}")

    # New labels are stored in one long, horizon-partitioned table.  Keep the
    # legacy wide-column fallback for already materialized snapshots.
    long_format = "horizon_sessions" in frame.columns and "net_alpha_target" in frame.columns
    label_columns: list[str]
    if long_format:
        label_columns = [
            column
            for column in (
                "horizon_sessions", "net_alpha_target", "label_available_time",
                "gross_return", "risk_residual", "reference_cost",
            )
            if column in frame.columns
        ]
    else:
        label_columns = [
            c for c in frame.columns
            if c.startswith("net_alpha_")
            or c.startswith("label_available_time_")
            or c.startswith("risk_residual_")
            or c.startswith("reference_cost_")
        ]
    if OUTCOME_STATUS_COLUMN in frame.columns:
        label_columns = [*label_columns, OUTCOME_STATUS_COLUMN]
    has_evidence_columns = (
        "resolution_kind" in frame.columns
        and "policy_hash" in frame.columns
        and "horizon_sessions" in frame.columns
    )
    if has_evidence_columns:
        label_columns = [*label_columns, "resolution_kind", "policy_hash"]
    feature_frame = frame.drop(label_columns)
    if long_format:
        # The long label join repeats each feature row once per horizon.  The
        # model panel must remain one row per instrument/session; horizon
        # universes are retained only in ``labels_by_horizon`` below.
        feature_frame = feature_frame.unique(
            subset=[ID_COLUMN, _FEATURE_SESSION], keep="first", maintain_order=True
        )
        # The source feature panel starts at the first available observation
        # rather than emitting a warm-up null. Drop that single pre-lookback
        # row per instrument so the integrity audit cannot treat it as a
        # fabricated rolling value.
        feature_frame = (
            feature_frame.sort([ID_COLUMN, _FEATURE_SESSION])
            .with_columns(
                pl.int_range(0, pl.len()).over(ID_COLUMN).alias("__warmup_row")
            )
            .filter(pl.col("__warmup_row") > 0)
            .drop("__warmup_row")
        )
        # Restore explicit warm-up semantics for rolling sources whose
        # upstream panel backfilled the first observation.  The audit then
        # sees the true unavailable state, and model fitting naturally drops
        # these rows through its finite-feature filter.
        warmup_columns = [
            "fluc_rate", "intraday_ret", "overnight_ret", "sector_ret_5d",
            "feature__fluc_rate", "feature__intraday_ret",
            "feature__overnight_ret", "feature__sector_ret_5d",
        ]
        first_rows = feature_frame.with_columns(
            pl.int_range(0, pl.len()).over(ID_COLUMN).alias("__row")
        )
        for column in warmup_columns:
            if column in first_rows.columns:
                first_rows = first_rows.with_columns(
                    pl.when(pl.col("__row") == 0)
                    .then(None)
                    .otherwise(pl.col(column))
                    .alias(column)
                )
        feature_frame = first_rows.drop("__row")
    if feature_frame.is_empty():
        raise ValueError("net-alpha snapshot feature frame is empty")

    # A hard tradability event is an as-of universe exclusion, not a missing
    # label to be selected and blocked later in replay.  Remove only keys whose
    # pinned evidence is explicitly UNEXECUTABLE_EXIT; ordinary no-bar rows
    # remain in the universe for diagnostic and conservative readiness logic.
    if long_format and "resolution_kind" in frame.columns:
        blocked = (
            frame.filter(
                pl.col("horizon_sessions").is_in(list(candidate_horizon_sessions))
                & (pl.col("resolution_kind") == "UNEXECUTABLE_EXIT")
            )
            .select(ID_COLUMN, _FEATURE_SESSION)
            .unique()
        )
        if not blocked.is_empty():
            feature_frame = feature_frame.join(
                blocked.with_columns(pl.lit(True).alias("__blocked")),
                on=[ID_COLUMN, _FEATURE_SESSION],
                how="left",
            ).filter(pl.col("__blocked").is_null()).drop("__blocked")
            if feature_frame.is_empty():
                raise ValueError("all net-alpha decision rows are unexecutable")

    roles = stock_net_alpha_v1_roles()
    feature_frame = _rename_feature_sources(feature_frame, roles)
    if feature_frame.is_empty():
        raise ValueError("net-alpha snapshot feature frame is empty after source renaming")

    labels_by_horizon: dict[int, pl.DataFrame] = {}
    status_by_horizon: dict[int, pl.DataFrame] = {}
    evidence_by_horizon: dict[int, pl.DataFrame] = {}
    join_evidence: list[HorizonJoinEvidence] = []
    has_status_column = OUTCOME_STATUS_COLUMN in frame.columns
    for horizon in candidate_horizon_sessions:
        if long_format:
            subset = frame.filter(pl.col("horizon_sessions") == horizon)
            if subset.is_empty():
                continue
            label_select: list[pl.Expr] = [
                pl.col(ID_COLUMN),
                pl.col(_FEATURE_SESSION),
                pl.col("net_alpha_target").alias(TARGET_COLUMN),
                pl.col("label_available_time").alias(AVAILABLE_COLUMN),
            ]
            if "risk_residual" in subset.columns:
                label_select.append(pl.col("risk_residual").alias(RISK_RESIDUAL_COLUMN))
            if "reference_cost" in subset.columns:
                label_select.append(
                    pl.col("reference_cost").alias(REFERENCE_COST_COLUMN)
                )
            label_frame = subset.select(label_select)
            feature_rows = frame.filter(pl.col("horizon_sessions") == horizon).height
        else:
            target_column = _target_column(frame.columns, horizon)
            available_column = _available_column(frame.columns, horizon)
            if target_column is None or available_column is None:
                continue
            residual_column = f"risk_residual_{horizon}d"
            cost_column = f"reference_cost_{horizon}d"
            select_columns: list[pl.Expr] = [
                pl.col(ID_COLUMN),
                pl.col(_FEATURE_SESSION).alias(_FEATURE_SESSION),
                pl.col(target_column).alias(TARGET_COLUMN),
                pl.col(available_column).alias(AVAILABLE_COLUMN),
            ]
            if residual_column in frame.columns:
                select_columns.append(
                    pl.col(residual_column).alias(RISK_RESIDUAL_COLUMN)
                )
            if cost_column in frame.columns:
                select_columns.append(pl.col(cost_column).alias(REFERENCE_COST_COLUMN))
            label_frame = frame.select(select_columns)
            feature_rows = frame.height
        label_rows = int(
            label_frame.filter(pl.col(TARGET_COLUMN).is_not_null()).height
        )
        available = label_frame.filter(
            pl.col(TARGET_COLUMN).is_not_null()
            & pl.col(AVAILABLE_COLUMN).is_not_null()
            & (pl.col(AVAILABLE_COLUMN) <= decision_time)
        )
        # Retain only the narrow horizon labels; execution columns are late-bound
        # from ``feature_frame`` by ``_build_label_join`` during fitting.
        joined = available.join(
            feature_frame.select(ID_COLUMN, _FEATURE_SESSION),
            on=[ID_COLUMN, _FEATURE_SESSION],
            how="inner",
        ).sort([ID_COLUMN, _FEATURE_SESSION])
        status_frame = _horizon_status_frame(
            frame, feature_frame, horizon, has_status_column, decision_time
        )
        if status_frame is not None:
            status_by_horizon[horizon] = status_frame
        evidence_frame = _horizon_evidence_frame(
            frame, feature_frame, horizon, has_evidence_columns
        )
        if evidence_frame is not None:
            evidence_by_horizon[horizon] = evidence_frame
        if joined.is_empty():
            join_evidence.append(
                HorizonJoinEvidence(
                    horizon_sessions=horizon,
                    feature_rows=feature_rows,
                    label_rows=label_rows,
                    joined_rows=0,
                    drop_reasons=("no point-in-time available labels",),
                    decision_rows=int(feature_frame.height),
                    realized_rows=0,
                    status_counts=(
                        _status_counts(status_frame, horizon)
                        if status_frame is not None
                        else None
                    ),
                )
            )
            continue
        labels_by_horizon[horizon] = joined
        join_evidence.append(
            HorizonJoinEvidence(
                horizon_sessions=horizon,
                feature_rows=feature_rows,
                label_rows=label_rows,
                joined_rows=joined.height,
                decision_rows=int(feature_frame.height),
                realized_rows=joined.height,
                status_counts=(
                    _status_counts(status_frame, horizon)
                    if status_frame is not None
                    else None
                ),
            )
        )

    if not labels_by_horizon:
        raise ValueError(
            "no candidate horizon produced point-in-time available labels"
        )

    coverage_by_horizon: dict[int, HorizonOutcomeCoverage] = {}
    for horizon in candidate_horizon_sessions:
        status_frame = status_by_horizon.get(horizon)
        if status_frame is None:
            continue
        coverage_by_horizon[horizon] = HorizonOutcomeCoverage.build(
            horizon,
            feature_frame.select(ID_COLUMN, _FEATURE_SESSION),
            status_frame,
        )

    manifest = _net_alpha_manifest(snapshot.manifest, frame)
    return NetAlphaResearchData(
        feature_frame=feature_frame,
        labels_by_horizon=labels_by_horizon,
        manifest=manifest,
        join_evidence=tuple(join_evidence),
        status_by_horizon=status_by_horizon,
        coverage_by_horizon=coverage_by_horizon,
        evidence_by_horizon=evidence_by_horizon,
        status_provenance=(
            "pinned" if has_status_column and has_evidence_columns else "legacy-inferred"
        ),
    )


def _horizon_status_frame(
    frame: pl.DataFrame,
    feature_frame: pl.DataFrame,
    horizon: int,
    has_status_column: bool,
    decision_time: datetime,
) -> pl.DataFrame | None:
    """Extract or derive the typed status frame for one horizon.

    When the composed frame carries an ``outcome_status`` column (the
    net-alpha status sidecar was left-joined), the status rows are extracted
    directly and every feature key must resolve to a typed state. Otherwise the
    status is derived from the label rows: an available realised label is
    ``REALIZED``, a not-yet-available label is ``PARTIAL_TAIL``, and a feature
    key without any label row for the horizon is ``MISSING_EXIT_PRICE``.
    """
    if has_status_column:
        status_rows = (
            frame.filter(pl.col("horizon_sessions") == horizon)
            .select(
                pl.col(ID_COLUMN).alias(ID_COLUMN),
                pl.col(_FEATURE_SESSION).alias(_FEATURE_SESSION),
                pl.col(OUTCOME_STATUS_COLUMN).alias(OUTCOME_STATUS_COLUMN),
            )
            .drop_nulls()
            .unique(subset=[ID_COLUMN, _FEATURE_SESSION], keep="first")
        )
        resolved = feature_frame.select(
            pl.col(ID_COLUMN), pl.col(_FEATURE_SESSION)
        ).join(
            status_rows.select(
                pl.col(ID_COLUMN),
                pl.col(_FEATURE_SESSION),
                pl.col(OUTCOME_STATUS_COLUMN),
            ),
            on=[ID_COLUMN, _FEATURE_SESSION],
            how="left",
        )
        missing = resolved.filter(pl.col(OUTCOME_STATUS_COLUMN).is_null())
        if not missing.is_empty():
            raise ValueError(
                f"horizon {horizon} feature keys absent from the outcome-status "
                f"sidecar: {missing.height} keys; the decision universe is not "
                "fully classified"
            )
        return status_rows
    label_columns = [
        c
        for c in frame.columns
        if c == f"net_alpha_{horizon}d_target"
        or c == f"label_available_time_{horizon}d"
        or c == "net_alpha_target"
        or c == "label_available_time"
    ]
    if not label_columns:
        return None
    target = next(
        (
            c
            for c in (f"net_alpha_{horizon}d_target", "net_alpha_target")
            if c in frame.columns
        ),
        None,
    )
    available = next(
        (
            c
            for c in (f"label_available_time_{horizon}d", "label_available_time")
            if c in frame.columns
        ),
        None,
    )
    if target is None or available is None:
        return None
    label_rows = (
        frame.filter(pl.col("horizon_sessions") == horizon)
        if "horizon_sessions" in frame.columns
        else frame
    )
    realized = label_rows.filter(pl.col(target).is_not_null())
    status = realized.with_columns(
        pl.when(pl.col(available).is_null() | (pl.col(available) > decision_time))
        .then(pl.lit(OUTCOME_PARTIAL_TAIL))
        .otherwise(pl.lit(OUTCOME_REALIZED))
        .alias(OUTCOME_STATUS_COLUMN)
    ).select(
        pl.col(ID_COLUMN),
        pl.col(_FEATURE_SESSION).alias(_FEATURE_SESSION),
        pl.col(OUTCOME_STATUS_COLUMN),
    )
    derived = feature_frame.select(
        pl.col(ID_COLUMN), pl.col(_FEATURE_SESSION)
    ).join(
        status.select(
            pl.col(ID_COLUMN),
            pl.col(_FEATURE_SESSION),
            pl.col(OUTCOME_STATUS_COLUMN),
        ),
        on=[ID_COLUMN, _FEATURE_SESSION],
        how="left",
    ).with_columns(
        pl.col(OUTCOME_STATUS_COLUMN)
        .fill_null(OUTCOME_MISSING_EXIT_PRICE)
        .alias(OUTCOME_STATUS_COLUMN)
    )
    return derived.select(
        pl.col(ID_COLUMN),
        pl.col(_FEATURE_SESSION).alias(_FEATURE_SESSION),
        pl.col(OUTCOME_STATUS_COLUMN),
    )


def _horizon_evidence_frame(
    frame: pl.DataFrame,
    feature_frame: pl.DataFrame,
    horizon: int,
    has_evidence_columns: bool,
) -> pl.DataFrame | None:
    """Extract the hash-bound outcome-evidence projection for one horizon.

    When the composed frame carries the pinned evidence spine, the per-horizon
    rows retain the full bounded projection (``policy_hash``,
    ``resolution_kind``, ``outcome_status``, scheduled entry/exit sessions, and
    entry/exit dispositions) for every feature key; otherwise ``None`` is
    returned and the horizon stays legacy-inferred (diagnostic-only, never
    promotable). An absent horizon partition, duplicate evidence keys, or an
    evidence spine that is unpinned or policy-incompatible raises
    ``ValueError``.
    """
    if not has_evidence_columns:
        return None
    horizon_rows = frame.filter(pl.col("horizon_sessions") == horizon)
    if horizon_rows.is_empty():
        raise ValueError(
            f"requested horizon {horizon} has no outcome-evidence partition "
            "in the composed frame"
        )
    evidence_columns = (
        ID_COLUMN,
        _FEATURE_SESSION,
        "horizon_sessions",
        "policy_hash",
        "resolution_kind",
        OUTCOME_STATUS_COLUMN,
        "scheduled_entry_session",
        "scheduled_exit_session",
        "entry_disposition",
        "exit_disposition",
    )
    missing_evidence = [c for c in evidence_columns if c not in horizon_rows.columns]
    if missing_evidence:
        raise ValueError(
            f"horizon {horizon} outcome-evidence spine missing columns "
            f"{missing_evidence}"
        )
    policy_rows = horizon_rows.filter(pl.col("policy_hash").is_not_null())
    if policy_rows.is_empty():
        raise ValueError(f"horizon {horizon} outcome-evidence spine is unpinned")
    if policy_rows["policy_hash"].n_unique() != 1:
        raise ValueError(
            f"horizon {horizon} outcome-evidence spine carries multiple "
            "policy hashes"
        )
    duplicate = (
        horizon_rows.group_by([ID_COLUMN, _FEATURE_SESSION, "horizon_sessions"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate.is_empty():
        raise ValueError(
            f"horizon {horizon} outcome-evidence spine contains duplicate keys"
        )
    evidence_rows = (
        horizon_rows.select(*evidence_columns)
        .filter(
            pl.col("policy_hash").is_not_null()
            & pl.col("resolution_kind").is_not_null()
            & pl.col(OUTCOME_STATUS_COLUMN).is_not_null()
        )
        .unique(
            subset=[ID_COLUMN, _FEATURE_SESSION, "horizon_sessions"], keep="first"
        )
    )
    resolved = feature_frame.select(
        pl.col(ID_COLUMN), pl.col(_FEATURE_SESSION)
    ).join(
        evidence_rows,
        on=[ID_COLUMN, _FEATURE_SESSION],
        how="left",
    )
    missing = resolved.filter(
        pl.col("policy_hash").is_null() | pl.col("resolution_kind").is_null()
    )
    if not missing.is_empty():
        raise ValueError(
            f"horizon {horizon} feature keys absent from the outcome-evidence "
            f"spine: {missing.height} keys; the decision universe is not fully "
            "classified"
        )
    return evidence_rows


def _status_counts(
    status_frame: pl.DataFrame | None, horizon: int
) -> OutcomeStatusCounts | None:
    """Bounded per-status counts from a horizon status frame."""
    if status_frame is None:
        return None
    counts = (
        status_frame.group_by(OUTCOME_STATUS_COLUMN).len().sort(OUTCOME_STATUS_COLUMN)
    )
    return OutcomeStatusCounts.from_mapping(
        {
            str(row[OUTCOME_STATUS_COLUMN]): int(row["len"])
            for row in counts.iter_rows(named=True)
        }
    )


def _rename_feature_sources(
    frame: pl.DataFrame, roles: dict[str, str]
) -> pl.DataFrame:
    """Rename ``feature__<source>`` columns to raw source names.

    Materialized feature panels expose sources with the ``feature__`` prefix;
    ``build_model_features`` consumes raw source names. Columns not covered by a
    declared role pass through unchanged.
    """
    rename_map: dict[str, str] = {}
    for source in roles:
        prefixed = f"{_FEATURE_PREFIX}{source}"
        if prefixed in frame.columns:
            rename_map[prefixed] = source
    if not rename_map:
        return frame
    return frame.rename(rename_map)


def _target_column(columns: list[str], horizon: int) -> str | None:
    for candidate in (
        f"net_alpha_{horizon}d_target",
        f"net_residual_o2o_{horizon}d",
    ):
        if candidate in columns:
            return candidate
    return None


def _available_column(columns: list[str], horizon: int) -> str | None:
    for candidate in (
        f"label_available_time_{horizon}d",
        "label_available_time",
    ):
        if candidate in columns:
            return candidate
    return None


def _net_alpha_manifest(
    source: DatasetManifest, frame: pl.DataFrame
) -> DatasetManifest:
    """Derive the canonical net-alpha training manifest from the composed frame."""
    from dataclasses import replace

    return replace(
        source,
        feature_set=CANONICAL_FEATURE_SET,
        feature_set_hash=source.feature_set_hash or "net-alpha-v1",
        row_count=frame.height,
    )


def validate_ml_market_data(
    data: MlMarketData,
    candidate_horizon_sessions: tuple[int, ...],
) -> None:
    """Validate composed ML market data for training readiness.

    Checks that the data contains non-empty frames, the requested horizons
    are present, and the data satisfies the training contract.

    Args:
        data: the composed ML market data to validate.
        candidate_horizon_sessions: the expected horizon sessions.

    Raises:
        ValueError: if the data fails validation.
    """
    if not isinstance(data, MlMarketData):
        raise TypeError(f"expected MlMarketData, got {type(data).__name__}")

    if data.frame.is_empty():
        raise ValueError("ML market data frame is empty")

    if not data.labels_by_horizon:
        raise ValueError("ML market data has no horizon labels")

    missing_horizons = [
        h for h in candidate_horizon_sessions
        if h not in data.labels_by_horizon
    ]
    if missing_horizons:
        raise ValueError(
            f"ML market data missing requested horizons: {missing_horizons}"
        )

    for horizon, label_frame in data.labels_by_horizon.items():
        if label_frame.is_empty():
            raise ValueError(
                f"ML market data horizon {horizon} has empty label frame"
            )
