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
from datetime import datetime

import numpy as np
import polars as pl

from src.core.costs import CostSchedule, LiquiditySlippageModel, default_base_schedule
from src.stocks.ml.contracts import PortfolioSettings, RiskSettings
from src.stocks.ml.labels import ID_COLUMN, RISK_RESIDUAL_COLUMN, SESSION_COLUMN
from src.stocks.ml.models import SCORE_COLUMN

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
    holds and is validated by the replay. ``base_active_fraction`` and
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
    segment_diagnostics: tuple[ReplaySegmentDiagnostic, ...] = ()

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
            "period_net_returns": [float(value) for value in self.period_net_returns],
            "segments": [diag.to_json() for diag in self.segment_diagnostics],
        }


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
    ):
        if horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        self._horizon_sessions = horizon_sessions
        self._portfolio = portfolio
        self._risk = risk
        self._cost_schedule = cost_schedule or default_base_schedule()
        self._liquidity_model = liquidity_model
        self._seed = seed

    def evaluate(
        self,
        oof_scores: pl.DataFrame,
        realized: pl.DataFrame | None = None,
        *,
        decision_time: datetime | None = None,
        segment_column: str | None = None,
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

        Returns:
            An immutable ``ReplayEvaluation`` with deterministic ``orders``.

        Raises:
            ValueError: for a missing scored column, a non-empty realized frame
                that lacks canonical columns, carries non-finite outcomes, or
                repeats keys, or when a realized replay has no liquidity model
                or cost coverage.
        """
        del decision_time
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
                    weights = self._allocate(scores, available_exposure=available)
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
        matured = 0
        cash = 0
        missing_realized = 0
        partial = 0
        segment_counts: dict[int, list[int]] = {
            segment: [0, 0, 0, 0] for segment in segment_lengths
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
                complete = True
                for order in members:
                    realized_value = realized_by_key.get(
                        (order.instrument_id, order.decision_session)
                    )
                    if realized_value is None:
                        complete = False
                        break
                    cost_rate = self._realized_cost(
                        order.order_size, order.decision_session, realized_value
                    )
                    growth.append(float(realized_value[_RISK_RESIDUAL]) - cost_rate)
                if not complete or len(growth) != len(members):
                    missing_realized += 1
                    counts[2] += 1
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
                        order_count=len(growth),
                        notional=float(sum(order.order_size for order in members)),
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
            active_sessions=active_session_count,
            matured_vintage_count=matured,
            cash_vintage_count=cash,
            missing_realized_vintage_count=missing_realized,
            partial_vintage_count=partial,
            segment_diagnostics=diagnostics,
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
    ) -> tuple[ReplaySegmentDiagnostic, ...]:
        """Assemble and validate the bounded per-segment accounting diagnostics.

        Each segment's ``matured + cash + missing + partial`` partition must
        equal its ``scored_sessions``; a broken accounting relation raises
        ``ValueError`` because the diagnostic contract is an invariant.
        """
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
                )
            )
        return tuple(diagnostics)

    def _allocate(
        self, scores: np.ndarray, *, available_exposure: float
    ) -> np.ndarray:
        """Constrained allocation on the decimal economic score.

        The profile's no-trade band is evaluated only against the decimal
        economic score: a name whose lower bound does not clear the band
        contributes zero weight, and an all-below-band cross-section creates no
        orders. Weights are proportional to the clipped positive score, capped
        at ``max_single_weight``, and scaled so the vintage's total exposure
        never exceeds the ``available_exposure`` left free by the concurrent
        active vintages.
        """
        portfolio = self._portfolio
        hurdle = self._risk.no_trade_band_bps / 10_000.0
        clean = np.where(np.isfinite(scores), scores, 0.0)
        signal = np.clip(clean - hurdle, 0.0, None)
        if not signal.any():
            return np.zeros(scores.size, dtype=np.float64)
        weights = signal / signal.sum()
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
