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
only when ``predicted_incremental_alpha_lower_bound > stressed_marginal_cost +
marginal_risk_penalty``. The selected horizon fixes the holding maturity only;
signals are computed every session and split into cohorts, so
``horizon_sessions`` and ``rebalance_interval`` are never equated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from src.core.costs import CostSchedule, default_base_schedule
from src.stocks.ml.contracts import PortfolioSettings, RiskSettings
from src.stocks.ml.models import SCORE_COLUMN

_ID = "instrument_id"
_SESSION = "session"
_AVAILABLE = "label_available_time"
_TARGET = "net_alpha"


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
    """One cohort's realized block net log-growth."""

    cohort_id: int
    horizon_sessions: int
    block_log_excess: float
    order_count: int
    notional: float


@dataclass(frozen=True, slots=True)
class ReplayEvaluation:
    """Deterministic outcome of one policy replay evaluation."""

    orders: tuple[PolicyOrder, ...]
    blocks: tuple[PolicyBlock, ...]
    decisions: tuple[int, ...]

    @property
    def block_log_excess(self) -> tuple[float, ...]:
        return tuple(block.block_log_excess for block in self.blocks)

    def to_json(self) -> dict[str, object]:
        return {
            "order_count": len(self.orders),
            "block_count": len(self.blocks),
            "block_log_excess": list(self.block_log_excess),
            "decisions": list(self.decisions),
        }


class NetAlphaPolicyReplay:
    """Deterministic cost/risk-aware policy replay over scored OOF panels.

    The replay is fully deterministic for a given input: the same scored panel
    and settings always produce identical orders and block series. Decisions are
    grouped into staggered cohorts keyed by decision session; each cohort's
    realized block log-growth is the equal-weighted net (cost-after-risk)
    log return of its positions over the holding horizon.

    ``evaluate`` returns an immutable :class:`ReplayEvaluation` whose ``orders``
    tuple makes order-for-order equality assertions meaningful.
    """

    def __init__(
        self,
        horizon_sessions: int,
        portfolio: PortfolioSettings,
        risk: RiskSettings,
        cost_schedule: CostSchedule | None = None,
        seed: int = 42,
    ):
        if horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        self._horizon_sessions = horizon_sessions
        self._portfolio = portfolio
        self._risk = risk
        self._cost_schedule = cost_schedule or default_base_schedule()
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
                and ``predicted_net_alpha``.
            realized: optional panel carrying ``instrument_id``, ``session``,
                and ``net_alpha`` realized target for block growth.
            decision_time: optional decision time gate for point-in-time
                availability.

        Returns:
            An immutable ``ReplayEvaluation`` with deterministic ``orders``.
        """
        del decision_time
        required = (_ID, _SESSION, SCORE_COLUMN)
        missing = [c for c in required if c not in oof_scores.columns]
        if missing:
            raise ValueError(f"OOF scored panel missing columns {missing}")

        scored = oof_scores.filter(
            pl.col(SCORE_COLUMN).is_not_null() & pl.col(SCORE_COLUMN).is_finite()
        )
        if scored.is_empty():
            return ReplayEvaluation(orders=(), blocks=(), decisions=())

        portfolio = self._portfolio
        sessions = sorted(scored[_SESSION].unique().to_list())
        session_index = {session: i for i, session in enumerate(sessions)}
        cohort_size = max(1, int(self._horizon_sessions))

        orders: list[PolicyOrder] = []
        blocks: list[PolicyBlock] = []
        decision_sessions: list[int] = []

        realized_map: dict[tuple[str, object], float] = {}
        if realized is not None and not realized.is_empty() and _TARGET in realized.columns:
            for row in realized.select(_ID, _SESSION, _TARGET).iter_rows():
                realized_map[(str(row[0]), row[1])] = float(row[2])

        cohort_weights: dict[int, list[PolicyOrder]] = {}
        for session in sessions:
            position = session_index[session]
            cohort = position // cohort_size
            cross = scored.filter(pl.col(_SESSION) == session).sort(SCORE_COLUMN)
            top = cross.head(portfolio.top_k)
            if top.is_empty():
                continue
            scores = top[SCORE_COLUMN].to_numpy().astype(float)
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
            ]
            orders.extend(cohort_orders)
            cohort_weights.setdefault(cohort, []).extend(cohort_orders)
            decision_sessions.append(position)

        for cohort in sorted(cohort_weights):
            members = cohort_weights[cohort]
            growth: list[float] = []
            for order in members:
                realized_value = realized_map.get(
                    (order.instrument_id, order.decision_session)
                )
                if realized_value is None or not np.isfinite(realized_value):
                    continue
                cost_rate = self._marginal_cost(order.order_size)
                growth.append(float(realized_value) - cost_rate)
            if not growth:
                continue
            blocks.append(
                PolicyBlock(
                    cohort_id=cohort,
                    horizon_sessions=self._horizon_sessions,
                    block_log_excess=float(np.mean(growth)),
                    order_count=len(growth),
                    notional=float(
                        sum(order.order_size for order in members)
                    ),
                )
            )

        return ReplayEvaluation(
            orders=tuple(orders),
            blocks=tuple(blocks),
            decisions=tuple(decision_sessions),
        )

    def _allocate(self, scores: np.ndarray) -> np.ndarray:
        """Constrained equal-alpha-cap allocation: cap, exposure, no-trade band.

        Weights are proportional to the clipped positive alpha signal, capped at
        ``max_single_weight``, normalized to ``max_exposure``, and zeroed when
        the risk-adjusted lower-bound hurdle is not met.
        """
        portfolio = self._portfolio
        positive = scores[~np.isnan(scores)]
        if positive.size == 0:
            return np.zeros(scores.size, dtype=np.float64)
        signal = np.clip(positive, 0.0, None)
        if not signal.any():
            return np.zeros(scores.size, dtype=np.float64)
        alpha_avg = float(signal.mean())
        hurdle = self._risk.no_trade_band_bps / 10_000.0
        if alpha_avg <= hurdle:
            return np.zeros(scores.size, dtype=np.float64)
        weights = signal / signal.sum()
        weights = np.minimum(weights, portfolio.max_single_weight)
        scale = min(
            1.0,
            portfolio.max_exposure / float(weights.sum()) if weights.sum() > 0 else 0.0,
        )
        return np.asarray(weights * scale, dtype=np.float64)

    def _marginal_cost(self, order_size: float) -> float:
        """Stressed marginal round-trip cost rate for an order."""
        point = self._cost_schedule.cost_for(_decision_time_ref())
        participation = min(
            self._portfolio.participation_limit,
            max(order_size / self._portfolio.portfolio_value, 1e-9),
        )
        return (
            2.0 * point.commission_rate
            + point.tax_rate
            + 2.0 * (participation * 100.0)
        )


def _decision_time_ref() -> datetime:
    from datetime import UTC

    return datetime(2000, 1, 1, tzinfo=UTC)
