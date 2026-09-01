"""NetAlphaPolicyReplay: the one common policy for discovery, comparison, holdout.

A single policy replay is used for OOF horizon discovery, challenger
comparison, the untouched forward holdout, research backtests, paper, and live
scoring. It never treats a raw prediction label as a top-k score proxy and has
no separate screen formula.

Each decision maximizes

``alpha_lower_bound' w - risk_aversion * w' covariance w
- stressed_delta_cost(w - previous_w)``

subject to instrument cap, sector/beta active exposure, ADTV participation,
cash, settlement, and universe eligibility. New/replacement orders are created
only when the economic score's decimal lower bound clears the no-trade band.
The selected horizon fixes the holding maturity only; signals are computed
every session and split into cohorts, so ``horizon_sessions`` and
``rebalance_interval`` are never equated.

The economic score is always a decimal quantity: either the calibrated
``net_alpha_lower_bound`` or, for score-only planning, the raw
``predicted_net_alpha``. Realized block growth is decimal
``risk_residual - realized_cost`` where ``realized_cost`` uses the effective
cost schedule plus the provenance-bound liquidity model's dynamic slippage at
the actual order notional; the reference cost stored in the label dataset is
data provenance and is never reused as an execution cost for a differently
sized order.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import polars as pl

from src.core.costs import CostSchedule, LiquiditySlippageModel, default_base_schedule
from legacy.stocks.data.outcome_evidence import RESOLUTION_KIND_VOCABULARY
from legacy.stocks.domain.execution_policy import (
    SCHEDULED_OPEN_V1,
    ExecutionOutcomePolicy,
)
from legacy.stocks.ml.contracts import (
    OUTCOME_MISSING_ENTRY_PRICE,
    OUTCOME_PARTIAL_TAIL,
    OUTCOME_REALIZED,
    OUTCOME_STATUS_COLUMN,
    PortfolioSettings,
    RiskSettings,
)
from legacy.stocks.ml.labels import ID_COLUMN, RISK_RESIDUAL_COLUMN, SESSION_COLUMN
from legacy.stocks.ml.models import SCORE_COLUMN

_ID = ID_COLUMN
_SESSION = SESSION_COLUMN
_RISK_RESIDUAL = RISK_RESIDUAL_COLUMN
_ECONOMIC_SCORE = "net_alpha_lower_bound"
_COST_INPUTS = ("open", "adtv_20d", "volatility_20d")
_REALIZED_COLUMNS = (_ID, _SESSION, _RISK_RESIDUAL, *_COST_INPUTS)


@dataclass(frozen=True, slots=True)
class PolicyOrder:
    """One deterministic allocation decision: instrument, vintage, notional weight."""

    instrument_id: str
    decision_session: datetime
    cohort_id: int
    weight: float
    order_size: float
    predicted_net_alpha: float


@dataclass(frozen=True, slots=True)
class PolicyBlock:
    """One matured decision vintage's realized arithmetic net return.

    ``net_return`` is the equal-weighted arithmetic ``risk_residual -
    realized_cost`` mean of the vintage's orders; it is a decimal arithmetic
    return, never a logarithm. ``segment_id`` and ``decision_session_index``
    record where the matured vintage lives in the OOF calendar.
    """

    vintage_id: int
    horizon_sessions: int
    net_return: float
    order_count: int
    notional: float
    segment_id: int = 0
    decision_session_index: int = 0


@dataclass(frozen=True, slots=True)
class BlockedVintage:
    """One selected filled order whose exit is unresolved, not a zero return.

    Records the profile-independent replay identity ``(segment_id,
    vintage_id, instrument_id, decision_session)`` plus the pinned evidence
    fields (scheduled entry/exit sessions, ``outcome_status``,
    ``resolution_kind``, and entry/exit dispositions). It never carries a
    score, label, return, or prediction; a blocked vintage contributes no
    arithmetic return and is never cash-substituted.
    """

    segment_id: int
    vintage_id: int
    instrument_id: str
    decision_session: datetime
    scheduled_entry_session: date | None
    scheduled_exit_session: date | None
    outcome_status: str
    resolution_kind: str
    entry_disposition: str | None
    exit_disposition: str | None

    def __post_init__(self) -> None:
        if self.segment_id < 0:
            raise ValueError("blocked vintage segment_id must be non-negative")
        if self.vintage_id < 0:
            raise ValueError("blocked vintage vintage_id must be non-negative")
        if not self.instrument_id:
            raise ValueError("blocked vintage instrument_id must be non-empty")
        if not self.outcome_status:
            raise ValueError("blocked vintage outcome_status must be non-empty")
        if not self.resolution_kind:
            raise ValueError("blocked vintage resolution_kind must be non-empty")

    def _sort_key(self) -> tuple[object, ...]:
        return (
            self.segment_id,
            self.vintage_id,
            self.instrument_id,
            self.decision_session,
            self.outcome_status,
            self.resolution_kind,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "segment_id": int(self.segment_id),
            "vintage_id": int(self.vintage_id),
            "instrument_id": self.instrument_id,
            "decision_session": self.decision_session.isoformat(),
            "scheduled_entry_session": (
                self.scheduled_entry_session.isoformat()
                if self.scheduled_entry_session is not None
                else None
            ),
            "scheduled_exit_session": (
                self.scheduled_exit_session.isoformat()
                if self.scheduled_exit_session is not None
                else None
            ),
            "outcome_status": self.outcome_status,
            "resolution_kind": self.resolution_kind,
            "entry_disposition": self.entry_disposition,
            "exit_disposition": self.exit_disposition,
        }


@dataclass(frozen=True, slots=True)
class ReplaySegmentDiagnostic:
    """Bounded per-segment vintage diagnostics for one policy replay.

    ``scored_sessions`` counts every decision session in the segment;
    ``calibration_ready_sessions`` counts the sessions whose calibrated
    economic score produced at least one finite signal, ``eligible_sessions``
    the sessions with a name above the profile band, and ``active_sessions``
    the sessions that emitted an order. ``matured_vintage_count``,
    ``cash_vintage_count``, ``missing_realized_vintage_count``, and
    ``partial_vintage_count`` partition the segment's vintages; the accounting
    invariant ``matured + cash + missing + partial == scored_sessions`` always
    holds and is validated by the replay. ``unresolved_outcome_counts`` is the
    bounded per-status-signature breakdown of the missing-realized vintages
    (its sum equals ``missing_realized_vintage_count``). A signature contains
    every distinct unresolved order state for one vintage in sorted order, so
    multi-name orders neither over-count a vintage nor hide mixed causes.
    ``base_active_fraction`` and
    ``stress_active_fraction`` are the matured/observed active fraction; the
    order and maturity timeline is cost-independent, so a base and a stress
    replay of the same calibrated panel report identical diagnostics.
    """

    segment_id: int
    scored_sessions: int
    calibration_ready_sessions: int
    eligible_sessions: int
    active_sessions: int
    matured_vintage_count: int
    cash_vintage_count: int
    missing_realized_vintage_count: int
    partial_vintage_count: int
    base_active_fraction: float
    stress_active_fraction: float
    unresolved_outcome_counts: tuple[tuple[str, int], ...] = ()

    @property
    def evaluated_vintage_count(self) -> int:
        return self.matured_vintage_count + self.cash_vintage_count

    @property
    def vintage_count(self) -> int:
        return (
            self.matured_vintage_count
            + self.cash_vintage_count
            + self.missing_realized_vintage_count
            + self.partial_vintage_count
        )

    def to_json(self) -> dict[str, object]:
        return {
            "segment_id": int(self.segment_id),
            "scored_sessions": int(self.scored_sessions),
            "calibration_ready_sessions": int(self.calibration_ready_sessions),
            "eligible_sessions": int(self.eligible_sessions),
            "active_sessions": int(self.active_sessions),
            "matured_vintage_count": int(self.matured_vintage_count),
            "cash_vintage_count": int(self.cash_vintage_count),
            "missing_realized_vintage_count": int(self.missing_realized_vintage_count),
            "partial_vintage_count": int(self.partial_vintage_count),
            "unresolved_outcome_counts": {
                str(status): int(count) for status, count in self.unresolved_outcome_counts
            },
            "base_active_fraction": round(float(self.base_active_fraction), 12),
            "stress_active_fraction": round(float(self.stress_active_fraction), 12),
        }


@dataclass(frozen=True, slots=True)
class ReplayEvaluation:
    """Deterministic outcome of one policy replay evaluation.

    Every decision session is one holding vintage. ``period_net_returns`` is
    the chronological per-vintage evidence over the evaluated vintages
    (matured plus observed all-cash vintages), and ``vintage_segment_ids`` is
    the parallel segment identity so bootstrap resampling never crosses an OOF
    boundary. A complete vintage with no allocated order is an observed
    all-cash vintage carrying a ``0.0`` return; a complete vintage whose
    required realized row is absent is never zero-filled and instead
    increments ``missing_realized_vintage_count``. Trailing vintages whose
    maturity crosses an OOF segment boundary are ``partial_vintage_count`` and
    never count as evidence. ``blocks`` carry the arithmetic net return of
    every matured vintage. The per-segment ``segment_diagnostics`` satisfy the
    bounded accounting contract and are the only replay projection written to
    the result ledger.
    """

    orders: tuple[PolicyOrder, ...]
    blocks: tuple[PolicyBlock, ...]
    decisions: tuple[int, ...]
    period_net_returns: tuple[float, ...] = ()
    vintage_segment_ids: tuple[int, ...] = ()
    scored_sessions: int = 0
    calibration_ready_sessions: int = 0
    realized_sessions: int = 0
    eligible_sessions: int = 0
    active_sessions: int = 0
    matured_vintage_count: int = 0
    cash_vintage_count: int = 0
    missing_realized_vintage_count: int = 0
    partial_vintage_count: int = 0
    unresolved_outcome_counts: tuple[tuple[str, int], ...] = ()
    segment_diagnostics: tuple[ReplaySegmentDiagnostic, ...] = ()
    blocked_vintages: tuple[BlockedVintage, ...] = ()
    blocked_vintage_count: int = 0

    @property
    def period_count(self) -> int:
        return len(self.period_net_returns)

    @property
    def observed_sessions(self) -> int:
        return self.matured_vintage_count + self.cash_vintage_count

    @property
    def active_cohort_count(self) -> int:
        return self.matured_vintage_count

    @property
    def missing_realized_cohort_count(self) -> int:
        return self.missing_realized_vintage_count

    @property
    def selected_blocked_exit_count(self) -> int:
        """Number of selected filled exits that are blocked, never zero-filled."""
        return self.blocked_vintage_count

    @property
    def partial_cohort_count(self) -> int:
        return self.partial_vintage_count

    @property
    def block_net_returns(self) -> tuple[float, ...]:
        return tuple(block.net_return for block in self.blocks)

    @property
    def block_log_excess(self) -> tuple[float, ...]:
        """Read-only compatibility alias for legacy arithmetic block returns.

        The alias is retained for consumers that predate the log-growth rename,
        but the replay never writes arithmetic returns under a ``log`` name.
        """
        return self.block_net_returns

    def replay_diagnostics(self) -> dict[str, int]:
        """Bounded vintage/diagnostic counts; never score or return arrays."""
        return {
            "scored_sessions": int(self.scored_sessions),
            "calibration_ready_sessions": int(self.calibration_ready_sessions),
            "realized_sessions": int(self.realized_sessions),
            "eligible_sessions": int(self.eligible_sessions),
            "active_sessions": int(self.active_sessions),
            "orders": len(self.orders),
            "matured_vintages": int(self.matured_vintage_count),
            "cash_vintages": int(self.cash_vintage_count),
            "missing_realized_vintages": int(self.missing_realized_vintage_count),
            "partial_vintages": int(self.partial_vintage_count),
            "observed_sessions": int(self.observed_sessions),
            "active_vintages": int(self.active_cohort_count),
            "selected_blocked_exits": int(self.blocked_vintage_count),
        }

    def to_json(self) -> dict[str, object]:
        return {
            "order_count": len(self.orders),
            "block_count": len(self.blocks),
            "decisions": list(self.decisions),
            "period_count": int(self.period_count),
            "observed_sessions": int(self.observed_sessions),
            "active_cohort_count": int(self.active_cohort_count),
            "missing_realized_cohort_count": int(self.missing_realized_cohort_count),
            "partial_cohort_count": int(self.partial_cohort_count),
            "selected_blocked_exit_count": int(self.blocked_vintage_count),
            "unresolved_outcome_counts": {
                str(status): int(count) for status, count in self.unresolved_outcome_counts
            },
            "period_net_returns": [float(value) for value in self.period_net_returns],
            "segments": [diag.to_json() for diag in self.segment_diagnostics],
            "blocked_vintages": [
                blocked.to_json() for blocked in self.blocked_vintages
            ],
        }


def _realized_volatilities(
    top: pl.DataFrame,
    realized_by_key: dict[tuple[str, object], dict[str, float]],
) -> np.ndarray | None:
    """``volatility_20d`` array aligned to the top rows, or ``None`` when incomplete."""
    if not realized_by_key:
        return None
    vols: list[float] = []
    for row in top.iter_rows(named=True):
        entry = realized_by_key.get((str(row[_ID]), row[_SESSION]))
        if entry is None:
            return None
        vols.append(entry["volatility_20d"])
    return np.asarray(vols, dtype=np.float64)


class NetAlphaPolicyReplay:
    """Deterministic cost/risk-aware policy replay over scored OOF panels.

    The replay is fully deterministic for a given input: the same scored panel
    and settings always produce identical orders and vintage series. Every
    decision session creates one holding vintage that matures at
    ``horizon_sessions``; the total concurrent exposure of active vintages
    never exceeds ``PortfolioSettings.max_exposure``, so each new vintage's
    deployed exposure is capped by what the already-held vintages leave free.
    Vintages never cross an OOF segment boundary: creation, maturity, and
    bootstrap blocks are all segment-local.

    ``evaluate`` returns an immutable :class:`ReplayEvaluation` whose ``orders``
    tuple makes order-for-order equality assertions meaningful and whose
    ``segment_diagnostics`` satisfy the bounded per-segment accounting contract.
    """

    def __init__(
        self,
        horizon_sessions: int,
        portfolio: PortfolioSettings,
        risk: RiskSettings,
        cost_schedule: CostSchedule | None = None,
        liquidity_model: LiquiditySlippageModel | None = None,
        seed: int = 42,
        policy: ExecutionOutcomePolicy | None = None,
    ):
        if horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        self._horizon_sessions = horizon_sessions
        self._portfolio = portfolio
        self._risk = risk
        self._cost_schedule = cost_schedule or default_base_schedule()
        self._liquidity_model = liquidity_model
        self._seed = seed
        self.policy = policy or SCHEDULED_OPEN_V1

    @property
    def policy_hash(self) -> str:
        return self.policy.canonical_hash

    def evaluate(
        self,
        oof_scores: pl.DataFrame,
        realized: pl.DataFrame | None = None,
        *,
        decision_time: datetime | None = None,
        segment_column: str | None = None,
        status: pl.DataFrame | None = None,
        evidence: pl.DataFrame | None = None,
    ) -> ReplayEvaluation:
        """Evaluate the scored OOF panel through the common policy.

        Args:
            oof_scores: scored panel carrying ``instrument_id``, ``session``,
                and either ``net_alpha_lower_bound`` (calibrated decimal
                economic score) or ``predicted_net_alpha``.
            realized: optional canonical realized-outcome panel carrying
                ``instrument_id``, ``session``, ``risk_residual``, ``open``,
                ``adtv_20d``, and ``volatility_20d``. When provided it is
                validated (required columns, finite values, duplicate keys)
                before any order is emitted. ``None`` remains valid score-only
                planning and returns no realized blocks.
            decision_time: optional decision time gate for point-in-time
                availability.
            segment_column: optional OOF segment identity column. When present,
                the ordinal session index restarts at every segment and no
                vintage or maturity ever crosses a segment boundary; trailing
                vintages whose maturity leaves the segment are counted per
                segment as partial. When absent the replay is equivalent to one
                single segment (continuous holdout, paper, and live behavior).
            status: optional typed outcome-status projection carrying
                ``instrument_id``, ``session``, and ``outcome_status`` for every
                score key. A selected order with a ``REALIZED`` status is
                evaluated normally; ``PARTIAL_TAIL`` inside a mature segment is
                a ``ValueError`` (only the segment-local maturity rule may
                classify a tail); ``MISSING_ENTRY_PRICE`` marks an unfilled
                entry order that is cancelled at its execution event (no fill,
                turnover, cost, exposure, or realized return; the vintage keeps
                only its filled orders or becomes an observed all-cash vintage);
                every other typed state is recorded as a blocked vintage
                diagnostic and never zero-filled or silently omitted; the
                owning vintage settles on its filled orders when any exist and
                is only invalidated (missing-realized) when no order can be
                filled.
            evidence: optional pinned outcome-evidence artifact carrying
                ``instrument_id``, ``session``, ``policy_hash``, and
                ``outcome_status``. When supplied its ``policy_hash`` must
                equal this replay's policy hash (a foreign policy is a
                ``ValueError``) and it supplies the status projection when
                ``status`` is omitted.

        Returns:
            An immutable ``ReplayEvaluation`` with deterministic ``orders``.

        Raises:
            ValueError: for a missing scored column, a non-empty realized frame
                that lacks canonical columns, carries non-finite outcomes, or
                repeats keys, a realized replay without liquidity model or cost
                coverage, a ``PARTIAL_TAIL`` status inside a mature segment,
                evidence pinned under a foreign policy hash, or a score key
                absent from a supplied status projection.
        """
        del decision_time
        evidence_by_key: dict[tuple[str, object], dict[str, object]] = {}
        if evidence is not None and not evidence.is_empty():
            evidence_required = (_ID, _SESSION, "policy_hash", OUTCOME_STATUS_COLUMN)
            missing = [c for c in evidence_required if c not in evidence.columns]
            if missing:
                raise ValueError(f"outcome evidence missing columns {missing}")
            foreign = evidence.filter(pl.col("policy_hash") != self.policy_hash)
            if not foreign.is_empty():
                raise ValueError(
                    "outcome evidence is pinned under a foreign execution policy; "
                    f"expected {self.policy_hash}"
                )
            duplicates = (
                evidence.group_by([_ID, _SESSION]).len().filter(pl.col("len") > 1)
            )
            if not duplicates.is_empty():
                raise ValueError(
                    "outcome evidence contains duplicate instrument/session keys"
                )
            optional_evidence = (
                "resolution_kind",
                "scheduled_entry_session",
                "scheduled_exit_session",
                "entry_disposition",
                "exit_disposition",
            )
            evidence_cols = set(evidence.columns)
            for row in evidence.select(
                _ID,
                _SESSION,
                OUTCOME_STATUS_COLUMN,
                *[c for c in optional_evidence if c in evidence_cols],
            ).iter_rows(named=True):
                key = (str(row[_ID]), row[_SESSION])
                evidence_by_key[key] = dict(row)
            if status is None:
                status = evidence.select(
                    _ID, _SESSION, OUTCOME_STATUS_COLUMN
                )
        required = (_ID, _SESSION, SCORE_COLUMN)
        missing = [c for c in required if c not in oof_scores.columns]
        if missing:
            raise ValueError(f"OOF scored panel missing columns {missing}")

        economic_score = (
            _ECONOMIC_SCORE if _ECONOMIC_SCORE in oof_scores.columns else SCORE_COLUMN
        )
        scored = oof_scores.filter(
            pl.col(economic_score).is_not_null() & pl.col(economic_score).is_finite()
        )

        realized_by_key: dict[tuple[str, object], dict[str, float]] = {}
        realized_sessions: set[object] = set()
        if realized is not None and not realized.is_empty():
            self._validate_realized(realized)
            if self._liquidity_model is None:
                raise ValueError("realized replay requires a liquidity model")
            for row in realized.select(*_REALIZED_COLUMNS).iter_rows(named=True):
                key = (str(row[_ID]), row[_SESSION])
                realized_by_key[key] = {
                    _RISK_RESIDUAL: float(row[_RISK_RESIDUAL]),
                    "open": float(row["open"]),
                    "adtv_20d": float(row["adtv_20d"]),
                    "volatility_20d": float(row["volatility_20d"]),
                }
            realized_sessions = set(realized[_SESSION].to_list())

        status_by_key: dict[tuple[str, object], str] = {}
        if status is not None and not status.is_empty():
            missing = [
                c for c in (_ID, _SESSION, OUTCOME_STATUS_COLUMN)
                if c not in status.columns
            ]
            if missing:
                raise ValueError(f"status projection missing columns {missing}")
            for row in status.select(_ID, _SESSION, OUTCOME_STATUS_COLUMN).iter_rows(
                named=True
            ):
                state = str(row[OUTCOME_STATUS_COLUMN])
                if state not in {
                    "REALIZED", "PARTIAL_TAIL", "MISSING_ENTRY_PRICE",
                    "MISSING_EXIT_PRICE", "MISSING_DECISION_INPUT",
                    "UNDERSIZED_CROSS_SECTION", "RISK_PROJECTION_FAILED",
                    "ZERO_MAD", "UNSUPPORTED_CORPORATE_ACTION",
                    "UNEXECUTABLE_EXIT",
                }:
                    raise ValueError(f"unknown outcome status {state!r} in projection")
                key = (str(row[_ID]), row[_SESSION])
                if key in status_by_key and status_by_key[key] != state:
                    raise ValueError(
                        f"status projection maps score key {key} to conflicting "
                        f"states {status_by_key[key]!r}/{state!r}"
                    )
                status_by_key[key] = state
            realized_sessions = {
                row[_SESSION]
                for row in status.select(_ID, _SESSION, OUTCOME_STATUS_COLUMN)
                .filter(pl.col(OUTCOME_STATUS_COLUMN) == OUTCOME_REALIZED)
                .select(_SESSION)
                .iter_rows(named=True)
            }

        if evidence_by_key and status_by_key:
            mismatch = [
                (key, evidence_by_key[key].get(OUTCOME_STATUS_COLUMN), state)
                for key, state in status_by_key.items()
                if key in evidence_by_key
                and evidence_by_key[key].get(OUTCOME_STATUS_COLUMN) != state
            ]
            if mismatch:
                raise ValueError(
                    "evidence outcome_status disagrees with the status "
                    f"projection for {len(mismatch)} keys, e.g. "
                    f"{mismatch[0]}"
                )

        portfolio = self._portfolio
        all_sessions = sorted(oof_scores[_SESSION].unique().to_list())
        if not all_sessions:
            return ReplayEvaluation(orders=(), blocks=(), decisions=())
        valid_sessions = sorted(scored[_SESSION].unique().to_list())
        valid_session_set = set(valid_sessions)

        session_segment: dict[object, tuple[int, int]] = {}
        segment_lengths: dict[int, int] = {}
        if segment_column is not None:
            if segment_column not in oof_scores.columns:
                raise ValueError(
                    f"segment column {segment_column!r} missing from scored panel"
                )
            for segment_key, frame in oof_scores.partition_by(
                segment_column, maintain_order=True, as_dict=True
            ).items():
                segment = int(segment_key[0])
                segment_sessions = sorted(frame[_SESSION].unique().to_list())
                segment_lengths[segment] = len(segment_sessions)
                for local_pos, session in enumerate(segment_sessions):
                    session_segment[session] = (segment, local_pos)
        else:
            segment_lengths[0] = len(all_sessions)
            for i, session in enumerate(all_sessions):
                session_segment[session] = (0, i)

        by_session = {
            key[0]: frame
            for key, frame in scored.partition_by(
                _SESSION, maintain_order=True, as_dict=True
            ).items()
        }

        orders: list[PolicyOrder] = []
        decision_sessions: list[int] = []
        eligible_sessions = 0
        active_session_count = 0
        session_counts: dict[int, list[int]] = {
            segment: [0, 0, 0, 0, 0] for segment in segment_lengths
        }
        vintage_exposure_by_segment: dict[int, list[float]] = {
            segment: [0.0] * length
            for segment, length in segment_lengths.items()
        }
        vintage_ids: list[int] = []
        for position, session in enumerate(all_sessions):
            segment, local_pos = session_segment[session]
            vintage_id = len(vintage_ids)
            vintage_ids.append(vintage_id)
            counts = session_counts[segment]
            counts[0] += 1
            if session in valid_session_set:
                counts[1] += 1
            if local_pos + self._horizon_sessions >= segment_lengths[segment]:
                counts[4] += 1
            exposures = vintage_exposure_by_segment[segment]
            window_start = max(0, local_pos - self._horizon_sessions + 1)
            active_exposure = float(sum(exposures[window_start:local_pos]))
            available = max(0.0, portfolio.max_exposure - active_exposure)

            session_rows = by_session.get(session)
            cohort_orders: list[PolicyOrder] = []
            if session_rows is not None and not session_rows.is_empty():
                cross = session_rows.sort(
                    [economic_score, _ID], descending=[True, False]
                )
                top = cross.head(portfolio.top_k)
                if not top.is_empty():
                    scores = top[economic_score].to_numpy().astype(float)
                    clean = np.where(np.isfinite(scores), scores, 0.0)
                    if np.any(clean - self._risk.no_trade_band_bps / 10_000.0 > 0.0):
                        eligible_sessions += 1
                        counts[2] += 1
                    weights = self._allocate(
                        scores,
                        available_exposure=available,
                        volatilities=_realized_volatilities(top, realized_by_key),
                    )
                    cohort_orders = [
                        PolicyOrder(
                            instrument_id=str(row[_ID]),
                            decision_session=row[_SESSION],
                            cohort_id=vintage_id,
                            weight=float(weight),
                            order_size=float(weight * portfolio.portfolio_value),
                            predicted_net_alpha=float(score),
                        )
                        for row, score, weight in zip(
                            top.iter_rows(named=True), scores, weights, strict=True
                        )
                        if weight > 0.0
                    ]
            orders.extend(cohort_orders)
            exposures[local_pos] = sum(order.weight for order in cohort_orders)
            if cohort_orders:
                decision_sessions.append(position)
                active_session_count += 1
                counts[3] += 1

        scored_session_count = len(all_sessions)
        calibration_ready_session_count = len(valid_sessions)
        if not realized_by_key:
            diagnostics = self._score_only_segment_diagnostics(session_counts)
            return ReplayEvaluation(
                orders=tuple(orders),
                blocks=(),
                decisions=tuple(decision_sessions),
                scored_sessions=scored_session_count,
                calibration_ready_sessions=calibration_ready_session_count,
                realized_sessions=0,
                eligible_sessions=eligible_sessions,
                active_sessions=active_session_count,
                partial_vintage_count=sum(
                    diag.partial_vintage_count for diag in diagnostics
                ),
                segment_diagnostics=diagnostics,
            )

        orders_by_session: dict[object, list[PolicyOrder]] = {}
        for order in orders:
            orders_by_session.setdefault(order.decision_session, []).append(order)

        period_returns: list[float] = []
        vintage_segments: list[int] = []
        blocks: list[PolicyBlock] = []
        blocked_vintages: list[BlockedVintage] = []
        matured = 0
        cash = 0
        missing_realized = 0
        partial = 0
        segment_counts: dict[int, list[int]] = {
            segment: [0, 0, 0, 0] for segment in segment_lengths
        }
        segment_unresolved: dict[int, dict[str, int]] = {
            segment: {} for segment in segment_lengths
        }
        for position, session in enumerate(all_sessions):
            segment, local_pos = session_segment[session]
            counts = segment_counts[segment]
            if local_pos + self._horizon_sessions >= segment_lengths[segment]:
                partial += 1
                counts[3] += 1
                continue
            members = orders_by_session.get(session, [])
            if members:
                growth: list[float] = []
                filled_orders: list[PolicyOrder] = []
                unresolved_states: set[str] = set()
                complete = True
                for order in members:
                    key = (order.instrument_id, order.decision_session)
                    outcome_state: str | None = status_by_key.get(key)
                    if outcome_state == OUTCOME_PARTIAL_TAIL:
                        raise ValueError(
                            f"selected order {key} carries PARTIAL_TAIL inside a "
                            "mature segment; only the segment-local maturity rule "
                            "may classify a chronological tail"
                        )
                    if outcome_state == OUTCOME_MISSING_ENTRY_PRICE:
                        continue
                    if outcome_state not in (None, OUTCOME_REALIZED):
                        unresolved_states.add(str(outcome_state))
                        blocked_vintages.append(
                            self._blocked_vintage_record(
                                segment=segment,
                                vintage_id=int(vintage_ids[position]),
                                order=order,
                                outcome_state=str(outcome_state),
                                evidence=evidence_by_key.get(key),
                            )
                        )
                        continue
                    realized_value = realized_by_key.get(key)
                    if realized_value is None:
                        if outcome_state == OUTCOME_REALIZED:
                            raise ValueError(
                                f"REALIZED status {key} has no replay outcome inputs"
                            )
                        complete = False
                        break
                    cost_rate = self._realized_cost(
                        order.order_size, order.decision_session, realized_value
                    )
                    growth.append(float(realized_value[_RISK_RESIDUAL]) - cost_rate)
                    filled_orders.append(order)
                if not complete or (not filled_orders and unresolved_states):
                    missing_realized += 1
                    counts[2] += 1
                    if unresolved_states:
                        signature = "|".join(sorted(unresolved_states))
                        segment_unresolved[segment][signature] = (
                            segment_unresolved[segment].get(signature, 0) + 1
                        )
                    continue
                if not filled_orders:
                    period_returns.append(0.0)
                    cash += 1
                    counts[1] += 1
                    vintage_segments.append(segment)
                    continue
                net_return = float(np.mean(growth))
                period_returns.append(net_return)
                matured += 1
                counts[0] += 1
                vintage_segments.append(segment)
                blocks.append(
                    PolicyBlock(
                        vintage_id=int(vintage_ids[position]),
                        horizon_sessions=self._horizon_sessions,
                        net_return=net_return,
                        order_count=len(filled_orders),
                        notional=float(
                            sum(order.order_size for order in filled_orders)
                        ),
                        segment_id=segment,
                        decision_session_index=local_pos,
                    )
                )
            else:
                if session in realized_sessions:
                    period_returns.append(0.0)
                    cash += 1
                    counts[1] += 1
                    vintage_segments.append(segment)
                else:
                    missing_realized += 1
                    counts[2] += 1

        diagnostics = self._segment_diagnostics(
            session_counts, segment_lengths, segment_counts,
            segment_unresolved=(
                segment_unresolved if status_by_key else None
            ),
        )
        total_unresolved = tuple(
            sorted(
                {
                    state: sum(
                        segment_unresolved[segment].get(state, 0)
                        for segment in segment_unresolved
                    )
                    for state in {
                        state
                        for segment in segment_unresolved.values()
                        for state in segment
                    }
                }.items()
            )
        )
        return ReplayEvaluation(
            orders=tuple(orders),
            blocks=tuple(blocks),
            decisions=tuple(decision_sessions),
            period_net_returns=tuple(period_returns),
            vintage_segment_ids=tuple(vintage_segments),
            scored_sessions=scored_session_count,
            calibration_ready_sessions=calibration_ready_session_count,
            realized_sessions=len(realized_sessions),
            eligible_sessions=eligible_sessions,
            unresolved_outcome_counts=total_unresolved,
            active_sessions=active_session_count,
            matured_vintage_count=matured,
            cash_vintage_count=cash,
            missing_realized_vintage_count=missing_realized,
            partial_vintage_count=partial,
            segment_diagnostics=diagnostics,
            blocked_vintages=tuple(
                sorted(blocked_vintages, key=lambda b: b._sort_key())[:64]
            ),
            blocked_vintage_count=len({b.decision_session for b in blocked_vintages}),
        )

    def _blocked_vintage_record(
        self,
        *,
        segment: int,
        vintage_id: int,
        order: PolicyOrder,
        outcome_state: str,
        evidence: dict[str, object] | None,
    ) -> BlockedVintage:
        """Deterministic bounded blocked-vintage record for a selected filled order.

        Carries the pinned evidence projection when present; without evidence
        the exit is unreconciled (``UNRECONCILED_NO_BAR``) and never becomes a
        synthetic zero return or cash. ``resolution_kind`` must be a known
        vocabulary value when supplied.
        """
        resolution = "UNRECONCILED_NO_BAR"
        if evidence is not None:
            kind = evidence.get("resolution_kind")
            if kind is not None:
                resolution = str(kind)
                if resolution not in list(RESOLUTION_KIND_VOCABULARY) and resolution != "UNRECONCILED_NO_BAR":
                    raise ValueError(
                        f"unknown resolution_kind {resolution!r} in outcome evidence"
                    )
        scheduled_entry = (
            evidence.get("scheduled_entry_session") if evidence is not None else None
        )
        scheduled_exit = (
            evidence.get("scheduled_exit_session") if evidence is not None else None
        )
        entry_disposition = (
            evidence.get("entry_disposition") if evidence is not None else None
        )
        exit_disposition = (
            evidence.get("exit_disposition") if evidence is not None else None
        )
        return BlockedVintage(
            segment_id=segment,
            vintage_id=vintage_id,
            instrument_id=order.instrument_id,
            decision_session=order.decision_session,
            scheduled_entry_session=(
                scheduled_entry if isinstance(scheduled_entry, date) else None
            ),
            scheduled_exit_session=(
                scheduled_exit if isinstance(scheduled_exit, date) else None
            ),
            outcome_status=outcome_state,
            resolution_kind=resolution,
            entry_disposition=(
                str(entry_disposition) if isinstance(entry_disposition, str) else None
            ),
            exit_disposition=(
                str(exit_disposition) if isinstance(exit_disposition, str) else None
            ),
        )

    def _score_only_segment_diagnostics(
        self,
        session_counts: dict[int, list[int]],
    ) -> tuple[ReplaySegmentDiagnostic, ...]:
        """Bounded segment diagnostics for score-only planning (no realized)."""
        diagnostics: list[ReplaySegmentDiagnostic] = []
        for segment in sorted(session_counts):
            scored, calibration_ready, eligible, active, partial = session_counts[segment]
            diagnostics.append(
                ReplaySegmentDiagnostic(
                    segment_id=segment,
                    scored_sessions=scored,
                    calibration_ready_sessions=calibration_ready,
                    eligible_sessions=eligible,
                    active_sessions=active,
                    matured_vintage_count=0,
                    cash_vintage_count=0,
                    missing_realized_vintage_count=0,
                    partial_vintage_count=partial,
                    base_active_fraction=0.0,
                    stress_active_fraction=0.0,
                )
            )
        return tuple(diagnostics)

    def _segment_diagnostics(
        self,
        session_counts: dict[int, list[int]],
        segment_lengths: dict[int, int],
        segment_counts: dict[int, list[int]],
        *,
        segment_unresolved: dict[int, dict[str, int]] | None = None,
    ) -> tuple[ReplaySegmentDiagnostic, ...]:
        """Assemble and validate the bounded per-segment accounting diagnostics.

        Each segment's ``matured + cash + missing + partial`` partition must
        equal its ``scored_sessions``; a broken accounting relation raises
        ``ValueError`` because the diagnostic contract is an invariant.
        ``segment_unresolved`` optionally carries vintage-level unresolved
        status-signature counts per segment; when supplied, its per-segment sum
        must equal the segment's missing-realized count.
        """
        unresolved = segment_unresolved or {}
        diagnostics: list[ReplaySegmentDiagnostic] = []
        for segment in sorted(segment_counts):
            matured, cash, missing, partial = segment_counts[segment]
            scored = segment_lengths[segment]
            if matured + cash + missing + partial != scored:
                raise ValueError(
                    f"segment {segment} vintage accounting invariant broken: "
                    f"matured={matured} cash={cash} missing={missing} "
                    f"partial={partial} vs scored={scored}"
                )
            _scored, calibration_ready, eligible, active, _partial = session_counts[segment]
            observed = matured + cash
            active_fraction = (
                float(matured / observed) if observed > 0 else 0.0
            )
            typed = tuple(sorted(unresolved.get(segment, {}).items()))
            if segment_unresolved is not None and (
                sum(count for _state, count in typed) != missing
            ):
                raise ValueError(
                    f"segment {segment} typed unresolved counts {typed} do not "
                    f"sum to missing-realized {missing}"
                )
            diagnostics.append(
                ReplaySegmentDiagnostic(
                    segment_id=segment,
                    scored_sessions=scored,
                    calibration_ready_sessions=calibration_ready,
                    eligible_sessions=eligible,
                    active_sessions=active,
                    matured_vintage_count=matured,
                    cash_vintage_count=cash,
                    missing_realized_vintage_count=missing,
                    partial_vintage_count=partial,
                    base_active_fraction=active_fraction,
                    stress_active_fraction=active_fraction,
                    unresolved_outcome_counts=typed,
                )
            )
        return tuple(diagnostics)

    def _allocate(
        self,
        scores: np.ndarray,
        *,
        available_exposure: float,
        volatilities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fractional Kelly allocation on the decimal economic score.

        The profile's no-trade band is evaluated only against the decimal
        economic score: a name whose lower bound does not clear the band
        contributes zero weight, and an all-below-band cross-section creates no
        orders. When per-name ``volatilities`` are supplied, weights are scaled
        inversely to the idiosyncratic return variance (fractional Kelly,
        ``w_i ~ (score_i - hurdle) / sigma_i^2``) before the single-name cap;
        without them the allocation degrades to score-proportional sizing.
        Weights are capped at ``max_single_weight`` and scaled so the vintage's
        total exposure never exceeds the ``available_exposure`` left free by
        the concurrent active vintages.
        """
        portfolio = self._portfolio
        hurdle = self._risk.no_trade_band_bps / 10_000.0
        clean = np.where(np.isfinite(scores), scores, 0.0)
        signal = np.clip(clean - hurdle, 0.0, None)
        if not signal.any():
            return np.zeros(scores.size, dtype=np.float64)
        if volatilities is None:
            kelly = signal
        else:
            vol = np.where(np.isfinite(volatilities), volatilities, 1.0)
            variance = np.maximum(vol, 1e-12) ** 2
            kelly = signal / variance
        if not kelly.any():
            return np.zeros(scores.size, dtype=np.float64)
        weights = kelly / kelly.sum()
        weights = np.minimum(weights, portfolio.max_single_weight)
        if available_exposure <= 0.0:
            return np.zeros(scores.size, dtype=np.float64)
        scale = min(1.0, available_exposure / float(weights.sum()))
        return np.asarray(weights * scale, dtype=np.float64)

    def _realized_cost(
        self,
        order_size: float,
        decision_session: datetime,
        realized_value: dict[str, float],
    ) -> float:
        """Decimal round-trip cost rate for one order's realized execution.

        ``realized_cost = 2 * commission_rate + sell_tax_rate +
        2 * slippage_bps / 10_000`` where the dynamic ``slippage_bps`` comes
        from the provenance-bound liquidity model at the actual order notional.
        Missing cost inputs or cost coverage fail closed with ``ValueError``.
        """
        liquidity = self._liquidity_model
        if liquidity is None:
            raise ValueError("realized replay requires a liquidity model")
        point = self._cost_schedule.cost_for(decision_session)
        slippage_bps = liquidity.slippage_bps(
            notional=order_size,
            adtv_20d=realized_value["adtv_20d"],
            daily_volatility=realized_value["volatility_20d"],
            reference_price=realized_value["open"],
            effective_time=decision_session,
        )
        return (
            2.0 * point.commission_rate + point.tax_rate + 2.0 * slippage_bps / 10_000.0
        )

    def _validate_realized(self, realized: pl.DataFrame) -> None:
        """Fail closed on a non-empty realized frame that breaks the contract."""
        missing = [c for c in _REALIZED_COLUMNS if c not in realized.columns]
        if missing:
            raise ValueError(f"realized frame missing canonical columns {missing}")
        valid = (
            pl.col(_RISK_RESIDUAL).is_not_null()
            & pl.col(_RISK_RESIDUAL).is_finite()
            & pl.col("open").is_not_null()
            & pl.col("open").is_finite()
            & pl.col("adtv_20d").is_not_null()
            & pl.col("adtv_20d").is_finite()
            & pl.col("volatility_20d").is_not_null()
            & pl.col("volatility_20d").is_finite()
        )
        if not realized.filter(~valid).is_empty():
            raise ValueError(
                "realized frame contains null or non-finite outcome/cost columns"
            )
        duplicate = (
            realized.group_by([_ID, _SESSION]).len().filter(pl.col("len") > 1)
        )
        if not duplicate.is_empty():
            raise ValueError(
                "realized frame contains duplicate instrument/session keys"
            )
