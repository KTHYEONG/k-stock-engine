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
    """One deterministic allocation decision: instrument, cohort, notional weight."""

    instrument_id: str
    decision_session: datetime
    cohort_id: int
    weight: float
    order_size: float
    predicted_net_alpha: float


@dataclass(frozen=True, slots=True)
class PolicyBlock:
    """One active cohort's realized arithmetic net return.

    ``net_return`` is the equal-weighted arithmetic ``risk_residual -
    realized_cost`` mean of the cohort's orders; it is a decimal arithmetic
    return, never a logarithm. Geometric growth is formed later with ``log1p``.
    """

    cohort_id: int
    horizon_sessions: int
    net_return: float
    order_count: int
    notional: float


@dataclass(frozen=True, slots=True)
class ReplayEvaluation:
    """Deterministic outcome of one policy replay evaluation.

    The ``period_*`` evidence is the certification view over complete,
    non-overlapping holding cohorts in chronological order. A complete cohort
    with no allocated order is an observed all-cash cohort carrying a ``0.0``
    period return; a complete cohort whose required realized row is absent is
    never zero-filled and instead increments ``missing_realized_cohort_count``.
    ``blocks`` carry the arithmetic net return of every complete active cohort
    (used by OOF horizon discovery); trailing partial cohorts never count as
    evidence.
    """

    orders: tuple[PolicyOrder, ...]
    blocks: tuple[PolicyBlock, ...]
    decisions: tuple[int, ...]
    period_count: int = 0
    observed_sessions: int = 0
    active_cohort_count: int = 0
    missing_realized_cohort_count: int = 0
    period_net_returns: tuple[float, ...] = ()
    scored_sessions: int = 0
    realized_sessions: int = 0
    eligible_sessions: int = 0
    active_sessions: int = 0

    @property
    def block_net_returns(self) -> tuple[float, ...]:
        return tuple(block.net_return for block in self.blocks)

    @property
    def block_log_excess(self) -> tuple[float, ...]:
        """Read-side alias retained for OOF horizon discovery consumers."""
        return self.block_net_returns

    def replay_diagnostics(self) -> dict[str, int]:
        """Bounded cohort/diagnostic counts; never score or return arrays."""
        return {
            "scored_sessions": int(self.scored_sessions),
            "realized_sessions": int(self.realized_sessions),
            "eligible_sessions": int(self.eligible_sessions),
            "active_sessions": int(self.active_sessions),
            "orders": len(self.orders),
            "complete_cohorts": self.period_count + self.missing_realized_cohort_count,
            "active_cohorts": int(self.active_cohort_count),
            "observed_sessions": int(self.observed_sessions),
            "missing_realized_cohorts": int(self.missing_realized_cohort_count),
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
            "period_net_returns": [
                float(value) for value in self.period_net_returns
            ],
        }


class NetAlphaPolicyReplay:
    """Deterministic cost/risk-aware policy replay over scored OOF panels.

    The replay is fully deterministic for a given input: the same scored panel
    and settings always produce identical orders and block series. Decisions are
    grouped into staggered cohorts keyed by decision session; each cohort's
    realized arithmetic net return is the equal-weighted net (cost-after-risk)
    mean of its positions over the holding horizon.

    ``evaluate`` returns an immutable :class:`ReplayEvaluation` whose ``orders``
    tuple makes order-for-order equality assertions meaningful.
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
        if scored.is_empty():
            return ReplayEvaluation(orders=(), blocks=(), decisions=())

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
        sessions = sorted(scored[_SESSION].unique().to_list())
        session_index = {session: i for i, session in enumerate(sessions)}
        cohort_size = max(1, int(self._horizon_sessions))

        orders: list[PolicyOrder] = []
        decision_sessions: list[int] = []
        eligible_sessions = 0

        cohort_weights: dict[int, list[PolicyOrder]] = {}
        by_session = {
            key[0]: frame
            for key, frame in scored.partition_by(
                _SESSION, maintain_order=True, as_dict=True
            ).items()
        }
        for session in sessions:
            position = session_index[session]
            cohort = position // cohort_size
            cross = by_session[session].sort(
                [economic_score, _ID], descending=[True, False]
            )
            top = cross.head(portfolio.top_k)
            if top.is_empty():
                continue
            scores = top[economic_score].to_numpy().astype(float)
            clean = np.where(np.isfinite(scores), scores, 0.0)
            if np.any(clean - self._risk.no_trade_band_bps / 10_000.0 > 0.0):
                eligible_sessions += 1
            weights = self._allocate(scores)
            cohort_orders = [
                PolicyOrder(
                    instrument_id=str(row[_ID]),
                    decision_session=row[_SESSION],
                    cohort_id=cohort,
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
            if cohort_orders:
                cohort_weights.setdefault(cohort, []).extend(cohort_orders)
                decision_sessions.append(position)

        scored_session_count = len(sessions)
        active_session_count = len(decision_sessions)
        if not realized_by_key:
            return ReplayEvaluation(
                orders=tuple(orders),
                blocks=(),
                decisions=tuple(decision_sessions),
                scored_sessions=scored_session_count,
                realized_sessions=len(realized_sessions),
                eligible_sessions=eligible_sessions,
                active_sessions=active_session_count,
            )

        period_returns: list[float] = []
        blocks = []
        active_cohort_count = 0
        missing_realized_cohorts = 0
        complete_cohorts = scored_session_count // cohort_size
        for cohort in range(complete_cohorts):
            members = cohort_weights.get(cohort, [])
            if not members:
                start = cohort * cohort_size
                cohort_sessions = sessions[start : start + cohort_size]
                if all(s in realized_sessions for s in cohort_sessions):
                    period_returns.append(0.0)
                else:
                    missing_realized_cohorts += 1
                continue
            growth: list[float] = []
            for order in members:
                realized_value = realized_by_key.get(
                    (order.instrument_id, order.decision_session)
                )
                if realized_value is None:
                    break
                cost_rate = self._realized_cost(
                    order.order_size, order.decision_session, realized_value
                )
                growth.append(float(realized_value[_RISK_RESIDUAL]) - cost_rate)
            if len(growth) != len(members):
                missing_realized_cohorts += 1
                continue
            period_returns.append(float(np.mean(growth)))
            active_cohort_count += 1
            blocks.append(
                PolicyBlock(
                    cohort_id=cohort,
                    horizon_sessions=self._horizon_sessions,
                    net_return=float(np.mean(growth)),
                    order_count=len(growth),
                    notional=float(sum(order.order_size for order in members)),
                )
            )

        return ReplayEvaluation(
            orders=tuple(orders),
            blocks=tuple(blocks),
            decisions=tuple(decision_sessions),
            period_count=len(period_returns),
            observed_sessions=len(period_returns) * cohort_size,
            active_cohort_count=active_cohort_count,
            missing_realized_cohort_count=missing_realized_cohorts,
            period_net_returns=tuple(period_returns),
            scored_sessions=scored_session_count,
            realized_sessions=len(realized_sessions),
            eligible_sessions=eligible_sessions,
            active_sessions=active_session_count,
        )

    def _allocate(self, scores: np.ndarray) -> np.ndarray:
        """Constrained allocation on the decimal economic score.

        The 5-bps no-trade band is evaluated only against the decimal economic
        score: a name whose lower bound does not clear the band contributes zero
        weight, and an all-below-band cross-section creates no orders. Weights
        are proportional to the clipped positive score, capped at
        ``max_single_weight``, and normalized to ``max_exposure``.
        """
        portfolio = self._portfolio
        hurdle = self._risk.no_trade_band_bps / 10_000.0
        clean = np.where(np.isfinite(scores), scores, 0.0)
        signal = np.clip(clean - hurdle, 0.0, None)
        if not signal.any():
            return np.zeros(scores.size, dtype=np.float64)
        weights = signal / signal.sum()
        weights = np.minimum(weights, portfolio.max_single_weight)
        scale = min(
            1.0,
            portfolio.max_exposure / float(weights.sum()) if weights.sum() > 0 else 0.0,
        )
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
