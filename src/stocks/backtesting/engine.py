"""Event-driven historical replay over the KRX session calendar.

``StockBacktester`` replays the *same* pure planner used by paper and live
paths: at every decision session it builds a point-in-time snapshot, loads the
scheduled artifact, calls ``run_trading_cycle``, and executes the returned
target intents at the next session's open with explicit cost, capacity,
halt/limit, tick/lot, partial-fill, open-order, and T+2 settlement semantics.

Settlement is keyed by the *due session index* and released exactly once at
that session. Unadjusted executable prices drive fills; economic continuity
across corporate actions is tracked in a separate ledger, never by adjusting
executable prices.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from math import ceil, floor
from typing import Any

import numpy as np
import polars as pl

from src.core.costs import CostSchedule, FillCostBreakdown, LiquiditySlippageModel
from src.core.datasets import DatasetManifest
from src.core.instruments import Instrument
from src.core.portfolio import PortfolioSnapshot, Position
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.data.costs import (
    CostEvidence,
    krx_market_for_code,
    resolve_fill_cost,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.trading_cycle import (
    CycleStatus,
    TradingCycleRequest,
    TradingCycleResult,
    run_trading_cycle,
)

REQUIRED_BACKTEST_COLUMNS = (
    "instrument_id",
    "session",
    "open",
    "close",
    "volume",
    "trading_value",
)


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise BacktestValidationError(f"non-datetime session value: {value!r}")


def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise BacktestValidationError(f"non-numeric value: {value!r}")


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise BacktestValidationError(f"boolean value is not a quantity: {value!r}")
    if isinstance(value, (int, float)):
        return int(value)
    raise BacktestValidationError(f"non-integer value: {value!r}")


class BacktestValidationError(ValueError):
    """Raised when a replay request or schedule is invalid."""


@dataclass(frozen=True, slots=True)
class ArtifactSlot:
    """One scheduled artifact eligibility range."""

    eligible_from: datetime
    eligible_to: datetime
    artifact_id: str


@dataclass(frozen=True, slots=True)
class ArtifactSchedule:
    """Immutable, sorted, non-overlapping artifact eligibility schedule."""

    slots: tuple[ArtifactSlot, ...]

    def __post_init__(self) -> None:
        if not self.slots:
            raise BacktestValidationError("artifact schedule must have at least one slot")
        prev_to: datetime | None = None
        for slot in self.slots:
            if slot.eligible_from.tzinfo is None or slot.eligible_to.tzinfo is None:
                raise BacktestValidationError("artifact slots must be timezone-aware")
            if slot.eligible_from > slot.eligible_to:
                raise BacktestValidationError("artifact slot eligible_from after eligible_to")
            if prev_to is not None and slot.eligible_from <= prev_to:
                raise BacktestValidationError("artifact slots must not overlap")
            prev_to = slot.eligible_to

    def artifact_for(self, decision_time: datetime) -> str:
        for slot in self.slots:
            if slot.eligible_from <= decision_time <= slot.eligible_to:
                return slot.artifact_id
        raise BacktestValidationError(
            f"no scheduled artifact eligible at {decision_time.isoformat()}"
        )


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    """Input contract for one historical replay run."""

    strategy_id: str
    start_time: datetime
    end_time: datetime
    decision_session_indices: tuple[int, ...]
    cost_schedule: CostSchedule
    stress_cost_schedule: CostSchedule
    risk_policy: StockRiskPolicy
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise BacktestValidationError("strategy_id must be non-empty")
        if self.start_time.tzinfo is None or self.end_time.tzinfo is None:
            raise BacktestValidationError("start_time and end_time must be timezone-aware")
        if self.start_time >= self.end_time:
            raise BacktestValidationError("start_time must be before end_time")
        if len(set(self.decision_session_indices)) != len(self.decision_session_indices):
            raise BacktestValidationError("decision_session_indices must not repeat")
        if list(self.decision_session_indices) != sorted(self.decision_session_indices):
            raise BacktestValidationError("decision_session_indices must be sorted ascending")
        if self.cost_schedule.name == self.stress_cost_schedule.name:
            raise BacktestValidationError("base and stress cost schedules must differ")
        self.cost_schedule.cost_for(self.start_time)
        self.cost_schedule.cost_for(self.end_time)


@dataclass(frozen=True, slots=True)
class BacktestLedgerRow:
    """One reconciled accounting snapshot at a session close."""

    session: datetime
    settled_cash: float
    unsettled_cash: float
    positions_value: float
    accrued_costs: float
    equity: float


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """One attempted fill, filled or unfilled with a reason."""

    session: datetime
    instrument_id: str
    side: str
    quantity: int
    price: float | None
    gross: float | None
    cost: float | None
    reason: str
    cost_breakdown: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Outcome of a historical replay: ledger, fills, and derived metrics."""

    ledger: tuple[BacktestLedgerRow, ...]
    trades: tuple[BacktestTrade, ...]
    final_value: float
    total_return: float
    metrics: dict[str, float]
    stress_final_value: float | None = None
    stress_metrics: dict[str, float] | None = None
    data_quality: dict[str, str] = field(default_factory=dict)
    planned_cycles: int = 0
    attempted_orders: int = 0
    filled_orders: int = 0
    no_trade_reasons: tuple[str, ...] = ()

    @property
    def equity_curve(self) -> list[float]:
        return [row.equity for row in self.ledger]


Planner = Callable[..., TradingCycleResult]

ReplayDecisionProvider = Callable[
    [datetime, datetime], "PreparedReplayDecision"
]
ReplayScenarioPlanner = Callable[
    ["PreparedReplayDecision", PortfolioSnapshot, TradingCycleRequest],
    TradingCycleResult,
]


@dataclass(frozen=True, slots=True)
class PreparedReplayDecision:
    """Immutable decision inputs prepared once per final decision timestamp.

    Carries the compact bounded allocation history (only allocation columns
    plus economic evidence columns), the frozen route-specific calibration
    state, and the decision/execution timestamps. Base and stress portfolio
    snapshots are advanced independently against this shared input.
    """

    decision_time: datetime
    execution_time: datetime
    visible: pl.DataFrame
    calibration_state: dict[str, object] | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedRow:
    """Array-backed replay row: ``get``/``[]`` resolve from the market."""

    market: PreparedReplayMarket
    index: int

    def get(self, key: str, default: object = None) -> object:
        return self.market.value_at(self.index, key, default)

    def __getitem__(self, key: str) -> object:
        return self.market.value_at(self.index, key)


@dataclass(frozen=True, slots=True)
class PreparedReplayMarket:
    """Immutable, array-backed replay market shared by every candidate.

    Built once per training snapshot, the market owns the canonical sorted
    ``session``/``instrument_id`` row index and aligned ``float64`` arrays for
    execution fields, close returns, trading value, rolling ADTV, and rolling
    volatility, so a candidate contributes only an aligned score overlay instead
    of re-partitioning frames and recomputing market-wide statistics per replay.
    ``session_ranges`` maps each session index to its contiguous row range;
    ``rows_by_key`` resolves ``(instrument_id, session)`` rows in ``O(1)``.
    """

    sessions: tuple[datetime, ...]
    session_ranges: Mapping[int, tuple[int, int]]
    rows_by_key: Mapping[tuple[str, datetime], _PreparedRow]
    instrument_ids: np.ndarray
    row_session_of: np.ndarray
    close: np.ndarray
    open_: np.ndarray
    volume: np.ndarray
    trading_value: np.ndarray
    adtv: np.ndarray
    volatility: np.ndarray
    has_volatility: bool
    available_time: np.ndarray
    limit_locked: np.ndarray | None
    action_interval_covered: np.ndarray | None
    close_returns: np.ndarray
    instruments: Mapping[str, Instrument]
    artifacts: ArtifactSchedule | None
    initial_portfolio: PortfolioSnapshot | None
    cache_bytes: int

    @property
    def row_count(self) -> int:
        return int(self.instrument_ids.size)

    def value_at(self, index: int, key: str, default: object = None) -> object:
        """Return the aligned column value for one market row."""
        if key == "instrument_id":
            return self.instrument_ids[index]
        if key == "session":
            return self.sessions[int(self.row_session_of[index])]
        if key == "open":
            return self.open_[index]
        if key == "close":
            return self.close[index]
        if key == "volume":
            return self.volume[index]
        if key == "trading_value":
            return self.trading_value[index]
        if key == "adtv":
            return self.adtv[index]
        if key == "feature__volatility_20d":
            if not self.has_volatility:
                return default
            return self.volatility[index]
        if key == "available_time":
            return self.available_time[index]
        if key == "limit_locked":
            if self.limit_locked is None:
                return default
            return bool(self.limit_locked[index])
        if key == "action_interval_covered":
            if self.action_interval_covered is None:
                return default
            return bool(self.action_interval_covered[index])
        return default

    @classmethod
    def build(
        cls,
        frame: pl.DataFrame,
        adtv_window: int,
        *,
        instruments: Mapping[str, Instrument] | None = None,
        artifacts: ArtifactSchedule | None = None,
        initial_portfolio: PortfolioSnapshot | None = None,
    ) -> PreparedReplayMarket:
        """Build the immutable market once from a validated replay frame.

        The frame must carry ``REQUIRED_BACKTEST_COLUMNS``; the causal 20-session
        rolling ADTV and per-instrument close-return series are computed once and
        aligned to the canonical ``(session, instrument_id)`` row order. Raises
        ``BacktestValidationError`` for missing columns or non-finite execution
        values.
        """
        missing = [c for c in REQUIRED_BACKTEST_COLUMNS if c not in frame.columns]
        if missing:
            raise BacktestValidationError(f"panel must carry {', '.join(missing)}")
        ordered = frame.sort(["session", "instrument_id"])
        if ordered.is_empty():
            raise BacktestValidationError("panel has no rows")
        with_adtv = ordered.with_columns(
            pl.col("trading_value")
            .rolling_mean(adtv_window, min_samples=1)
            .over("instrument_id")
            .alias("adtv")
        )
        return_series = (
            ordered.sort("session")
            .with_columns(
                (pl.col("close").log().diff().over("instrument_id")).alias("__logret")
            )["__logret"]
            .fill_null(0.0)
        )
        sessions = tuple(
            _as_datetime(s) for s in ordered["session"].unique().sort().to_list()
        )
        session_index_of = {
            session: i for i, session in enumerate(sessions)
        }
        row_sessions = [session_index_of[_as_datetime(s)] for s in ordered["session"].to_list()]
        ranges: dict[int, tuple[int, int]] = {}
        current = -1
        start = 0
        for i, session_idx in enumerate(row_sessions):
            if session_idx != current:
                if current != -1:
                    ranges[current] = (start, i)
                current = session_idx
                start = i
        ranges[current] = (start, len(row_sessions))

        instrument_ids = np.asarray(
            [str(i) for i in ordered["instrument_id"].to_list()], dtype=object
        )
        available_time = np.asarray(
            [
                _as_datetime(v)
                if v is not None
                else None
                for v in ordered.get_column("available_time").to_list()
            ],
            dtype=object,
        )
        limit_locked = (
            ordered["limit_locked"].to_numpy().astype(bool)
            if "limit_locked" in ordered.columns
            else None
        )
        action_interval_covered = (
            ordered["action_interval_covered"].to_numpy().astype(bool)
            if "action_interval_covered" in ordered.columns
            else None
        )
        rows_by_key: dict[tuple[str, datetime], _PreparedRow] = {}
        market = cls(
            sessions=sessions,
            session_ranges=ranges,
            rows_by_key=rows_by_key,
            instrument_ids=instrument_ids,
            row_session_of=np.asarray(row_sessions, dtype=np.int64),
            close=ordered["close"].to_numpy().astype(np.float64),
            open_=ordered["open"].to_numpy().astype(np.float64),
            volume=ordered["volume"].to_numpy().astype(np.float64),
            trading_value=ordered["trading_value"].to_numpy().astype(np.float64),
            adtv=with_adtv["adtv"].to_numpy().astype(np.float64),
            volatility=(
                ordered["feature__volatility_20d"].to_numpy().astype(np.float64)
                if "feature__volatility_20d" in ordered.columns
                else np.zeros(ordered.height, dtype=np.float64)
            ),
            has_volatility="feature__volatility_20d" in ordered.columns,
            available_time=available_time,
            limit_locked=limit_locked,
            action_interval_covered=action_interval_covered,
            close_returns=return_series.to_numpy().astype(np.float64),
            instruments=instruments or {},
            artifacts=artifacts,
            initial_portfolio=initial_portfolio,
            cache_bytes=int(ordered.estimated_size()),
        )
        for i in range(ordered.height):
            rows_by_key[
                (str(ordered["instrument_id"][i]), sessions[int(row_sessions[i])])
            ] = _PreparedRow(market, i)
        return market


@dataclass(slots=True)
class _ScenarioState:
    """Mutable base or stress replay state advanced in one paired session loop."""

    account_snapshot_id: str
    settled_cash: float
    unsettled_cash: float
    accrued_costs: float
    positions: dict[str, int]
    settlements: dict[int, float]
    pending_orders: list[dict[str, object]]
    trades: list[BacktestTrade]
    ledger: list[BacktestLedgerRow]
    last_close: dict[str, float]
    attempted_orders: int
    base_positions: tuple[Position, ...]

    @classmethod
    def from_initial(cls, initial_portfolio: PortfolioSnapshot) -> _ScenarioState:
        return cls(
            account_snapshot_id=initial_portfolio.account_snapshot_id,
            settled_cash=initial_portfolio.settled_cash,
            unsettled_cash=initial_portfolio.unsettled_cash,
            accrued_costs=0.0,
            positions={
                p.instrument.instrument_id: int(p.quantity)
                for p in initial_portfolio.positions
                if p.quantity > 0
            },
            settlements={},
            pending_orders=[],
            trades=[],
            ledger=[],
            last_close={},
            attempted_orders=0,
            base_positions=tuple(initial_portfolio.positions),
        )


class StockBacktester:
    """Replays the trading cycle over ordered sessions with explicit fills."""

    def __init__(
        self,
        *,
        planner: Planner = run_trading_cycle,
        registry: ModelArtifactRegistry,
        instruments: Mapping[str, Instrument],
        manifest: DatasetManifest,
        cost_schedule: CostSchedule,
        stress_cost_schedule: CostSchedule | None = None,
        cost_evidence: CostEvidence | None = None,
        adtv_window: int = 20,
        seed: int = 42,
        decision_provider: ReplayDecisionProvider | None = None,
        scenario_planner: ReplayScenarioPlanner | None = None,
    ):
        self.planner = planner
        self.registry = registry
        self.instruments = instruments
        self.manifest = manifest
        self.cost_schedule = cost_schedule
        self.stress_cost_schedule = stress_cost_schedule
        self.cost_evidence = cost_evidence
        self.adtv_window = adtv_window
        self.seed = seed
        self.decision_provider = decision_provider
        self.scenario_planner = scenario_planner
        self._last_cycles: dict[int, TradingCycleResult] = {}
        self.prepared_decision_count = 0

    def cycles_at(self, decision_session_indices: tuple[int, ...]) -> dict[int, TradingCycleResult]:
        """Return the pure planning results for replay decisions.

        Exposes the same planner output used by paper/live so a parity test can
        assert the replay step and a paper cycle produce identical targets for
        an identical snapshot.
        """
        return {index: self._last_cycles[index] for index in decision_session_indices if index in self._last_cycles}

    def run(
        self,
        panel: pl.DataFrame,
        artifacts: ArtifactSchedule,
        initial_portfolio: PortfolioSnapshot,
        request: BacktestRequest,
        *,
        adtv: pl.DataFrame | None = None,
    ) -> BacktestResult:
        missing = [c for c in REQUIRED_BACKTEST_COLUMNS if c not in panel.columns]
        if missing:
            raise BacktestValidationError(f"panel must carry {', '.join(missing)}")
        self._assert_eligible_and_covered(panel)
        sessions = [_as_datetime(s) for s in sorted(panel["session"].unique().to_list())]
        if not sessions:
            raise BacktestValidationError("panel has no sessions")
        if sessions[0].tzinfo is None:
            raise BacktestValidationError("panel sessions must be timezone-aware")
        if sessions[0] > request.end_time or sessions[-1] < request.start_time:
            raise BacktestValidationError("replay window does not overlap the panel")

        by_session = dict(
            zip(
                sessions,
                panel.sort("session").partition_by(["session"], maintain_order=True),
                strict=True,
            )
        )
        if self.decision_provider is not None and self.scenario_planner is not None:
            market = PreparedReplayMarket.build(
                panel,
                self.adtv_window,
                instruments=self.instruments,
                artifacts=artifacts,
                initial_portfolio=initial_portfolio,
            )
            return self.run_prepared(request, market, None)
        ledger, trades, attempted_orders = self._run_ledger(
            panel, by_session, sessions, artifacts, initial_portfolio, request,
            self.cost_schedule, self._liquidity_model(stress=False), adtv=adtv,
        )
        final_value = ledger[-1].equity if ledger else initial_portfolio.settled_cash
        metrics = self._metrics(ledger, trades)
        stress_final_value: float | None = None
        stress_metrics: dict[str, float] | None = None
        if self.stress_cost_schedule is not None:
            stress_ledger, _stress_trades, _ = self._run_ledger(
                panel, by_session, sessions, artifacts, initial_portfolio, request,
                self.stress_cost_schedule, self._liquidity_model(stress=True),
                adtv=adtv,
            )
            stress_metrics = self._metrics(stress_ledger, _stress_trades)
            stress_final_value = stress_ledger[-1].equity if stress_ledger else None
        planned_cycles = sum(
            1
            for index in sorted(self._last_cycles)
            if self._last_cycles[index].status is CycleStatus.PLANNED
        )
        no_trade_reasons = tuple(
            reason
            for index in sorted(self._last_cycles)
            if self._last_cycles[index].status is not CycleStatus.PLANNED
            for reason in self._last_cycles[index].reasons
        )
        filled_orders = sum(1 for trade in trades if trade.quantity > 0)
        return BacktestResult(
            ledger=tuple(ledger),
            trades=tuple(trades),
            final_value=final_value,
            total_return=(
                (final_value - initial_portfolio.settled_cash)
                / initial_portfolio.settled_cash
                if initial_portfolio.settled_cash > 0
                else 0.0
            ),
            metrics=metrics,
            stress_final_value=stress_final_value,
            stress_metrics=stress_metrics,
            data_quality=self._data_quality_evidence(),
            planned_cycles=planned_cycles,
            attempted_orders=attempted_orders,
            filled_orders=filled_orders,
            no_trade_reasons=no_trade_reasons,
        )

    def run_prepared(
        self,
        request: BacktestRequest,
        market: PreparedReplayMarket,
        score_overlay: np.ndarray | None,
    ) -> BacktestResult:
        """Replay against a shared immutable prepared market.

        Base and stress portfolio/settlement states are advanced independently
        over the one immutable ``PreparedReplayMarket``; row data is read from
        aligned arrays instead of per-session ``partition_by``/``to_dicts`` and
        market-wide ADTV/returns are computed once. ``score_overlay`` is the
        candidate's aligned ``float64`` score per market row (``NaN`` when the
        candidate has no score) and is validated for length; execution reads
        only market arrays while allocation decisions flow through the prepared
        decision provider. Raises ``BacktestValidationError`` when the paired
        decision provider is absent or the overlay length mismatches.
        """
        if self.decision_provider is None or self.scenario_planner is None:
            raise BacktestValidationError(
                "prepared replay requires a decision provider and scenario planner"
            )
        if score_overlay is not None and len(score_overlay) != market.row_count:
            raise BacktestValidationError(
                f"score overlay length {len(score_overlay)} does not match "
                f"market row count {market.row_count}"
            )
        return self._run_paired_prepared(request, market)

    def _run_paired_prepared(
        self,
        request: BacktestRequest,
        market: PreparedReplayMarket,
    ) -> BacktestResult:
        if self.decision_provider is None or self.scenario_planner is None:
            raise BacktestValidationError(
                "prepared replay requires a decision provider and scenario planner"
            )
        if market.initial_portfolio is None:
            raise BacktestValidationError("prepared market requires an initial portfolio")
        if market.artifacts is None:
            raise BacktestValidationError("prepared market requires an artifact schedule")
        decision_set = {int(i) for i in request.decision_session_indices}
        rows_by_key = market.rows_by_key
        base_state = _ScenarioState.from_initial(market.initial_portfolio)
        stress_state = _ScenarioState.from_initial(market.initial_portfolio)
        base_liquidity = self._liquidity_model(stress=False)
        stress_liquidity = self._liquidity_model(stress=True)
        base_schedule = self.cost_schedule
        stress_schedule = self.stress_cost_schedule
        self.prepared_decision_count = 0

        for index, session in enumerate(market.sessions):
            self._advance_prepared_scenario(
                base_state, market, rows_by_key, index, session, base_schedule,
                base_liquidity,
            )
            if stress_schedule is not None:
                self._advance_prepared_scenario(
                    stress_state, market, rows_by_key, index, session,
                    stress_schedule, stress_liquidity,
                )

            if index in decision_set and index + 1 < len(market.sessions):
                decision_time = self._prepared_decision_time(market, index, session)
                execution_time = market.sessions[index + 1]
                artifact_id = market.artifacts.artifact_for(decision_time)
                prepared = self.decision_provider(decision_time, execution_time)
                self.prepared_decision_count += 1
                base_cycle = self._plan_paired(
                    prepared, base_state, decision_time, execution_time,
                    artifact_id, request,
                )
                self._last_cycles[index] = base_cycle
                base_state.pending_orders = self._plan_orders(
                    base_cycle, rows_by_key, execution_time, base_state.positions,
                    base_state.settled_cash, base_schedule,
                )
                base_state.attempted_orders += len(base_state.pending_orders)
                if stress_schedule is not None:
                    stress_cycle = self._plan_paired(
                        prepared, stress_state, decision_time, execution_time,
                        artifact_id, request,
                    )
                    self._last_cycles[index] = stress_cycle
                    stress_state.pending_orders = self._plan_orders(
                        stress_cycle, rows_by_key, execution_time,
                        stress_state.positions, stress_state.settled_cash,
                        stress_schedule,
                    )
                    stress_state.attempted_orders += len(stress_state.pending_orders)

        return self._result_from_states(base_state, stress_state, market.initial_portfolio)

    def _advance_prepared_scenario(
        self,
        state: _ScenarioState,
        market: PreparedReplayMarket,
        rows_by_key: Mapping[tuple[str, datetime], Any],
        index: int,
        session: datetime,
        schedule: CostSchedule,
        liquidity_model: LiquiditySlippageModel | None,
    ) -> None:
        """Advance one scenario reading only aligned prepared-market arrays."""
        due = state.settlements.pop(index, 0.0)
        if due:
            state.settled_cash += due
            state.unsettled_cash -= due

        new_pending: list[dict[str, object]] = []
        for order in state.pending_orders:
            state.settled_cash, state.unsettled_cash, state.accrued_costs = (
                self._execute_order(
                    order, None, rows_by_key, state.positions, state.settled_cash,
                    state.unsettled_cash, state.accrued_costs, state.settlements,
                    schedule, liquidity_model, index, session, state.trades,
                )
            )
        state.pending_orders = new_pending

        start, stop = market.session_ranges[index]
        for row in range(start, stop):
            close = market.close[row]
            if close is not None and close == close:
                state.last_close[str(market.instrument_ids[row])] = float(close)
        positions_value = sum(
            state.positions[i] * state.last_close[i]
            for i in state.positions
            if i in state.last_close
        )
        equity = (
            state.settled_cash
            + state.unsettled_cash
            + positions_value
            - state.accrued_costs
        )
        state.ledger.append(
            BacktestLedgerRow(
                session=session,
                settled_cash=state.settled_cash,
                unsettled_cash=state.unsettled_cash,
                positions_value=positions_value,
                accrued_costs=state.accrued_costs,
                equity=equity,
            )
        )

    def _prepared_decision_time(self, market: PreparedReplayMarket, index: int, session: datetime) -> datetime:
        start, stop = market.session_ranges[index]
        values: list[datetime] = [
            value
            for value in market.available_time[start:stop]
            if value is not None
        ]
        if not values:
            raise BacktestValidationError("no available_time at decision session")
        return max(values)

    def _result_from_states(
        self,
        base_state: _ScenarioState,
        stress_state: _ScenarioState,
        initial_portfolio: PortfolioSnapshot,
    ) -> BacktestResult:
        """Derive the immutable result from independent base/stress states."""
        ledger = base_state.ledger
        trades = base_state.trades
        final_value = ledger[-1].equity if ledger else initial_portfolio.settled_cash
        metrics = self._metrics(ledger, trades)
        stress_final_value: float | None = None
        stress_metrics: dict[str, float] | None = None
        if self.stress_cost_schedule is not None:
            stress_metrics = self._metrics(
                stress_state.ledger, stress_state.trades
            )
            stress_final_value = (
                stress_state.ledger[-1].equity if stress_state.ledger else None
            )
        planned_cycles = sum(
            1
            for index in sorted(self._last_cycles)
            if self._last_cycles[index].status is CycleStatus.PLANNED
        )
        no_trade_reasons = tuple(
            reason
            for index in sorted(self._last_cycles)
            if self._last_cycles[index].status is not CycleStatus.PLANNED
            for reason in self._last_cycles[index].reasons
        )
        filled_orders = sum(1 for trade in trades if trade.quantity > 0)
        return BacktestResult(
            ledger=tuple(ledger),
            trades=tuple(trades),
            final_value=final_value,
            total_return=(
                (final_value - initial_portfolio.settled_cash)
                / initial_portfolio.settled_cash
                if initial_portfolio.settled_cash > 0
                else 0.0
            ),
            metrics=metrics,
            stress_final_value=stress_final_value,
            stress_metrics=stress_metrics,
            data_quality=self._data_quality_evidence(),
            planned_cycles=planned_cycles,
            attempted_orders=base_state.attempted_orders,
            filled_orders=filled_orders,
            no_trade_reasons=no_trade_reasons,
        )

    def _run_paired(
        self,
        panel: pl.DataFrame,
        by_session: dict[datetime, pl.DataFrame],
        sessions: list[datetime],
        artifacts: ArtifactSchedule,
        initial_portfolio: PortfolioSnapshot,
        request: BacktestRequest,
        *,
        adtv: pl.DataFrame | None = None,
    ) -> BacktestResult:
        """Advance independent base/stress states with one decision preparation.

        One chronological session loop advances separate base and stress
        portfolio/settlement states. At each decision timestamp the replay
        decision-input provider is asked exactly once for the immutable market
        and planning inputs; the scenario planner then runs separately for the
        base and stress portfolio snapshots. Execution stays fully
        scenario-specific while the prepared history and calibration are shared.
        """
        if self.decision_provider is None or self.scenario_planner is None:
            raise BacktestValidationError(
                "paired replay requires a decision provider and scenario planner"
            )
        decision_set = {int(i) for i in request.decision_session_indices}
        rows_frame = self._rows_frame_with_adtv(panel, adtv)
        rows_by_key: Mapping[tuple[str, datetime], Any] = {
            (str(r["instrument_id"]), _as_datetime(r["session"])): r
            for r in rows_frame.to_dicts()
        }

        base_state = _ScenarioState.from_initial(initial_portfolio)
        stress_state = _ScenarioState.from_initial(initial_portfolio)
        base_liquidity = self._liquidity_model(stress=False)
        stress_liquidity = self._liquidity_model(stress=True)
        base_schedule = self.cost_schedule
        stress_schedule = self.stress_cost_schedule
        self.prepared_decision_count = 0

        for index, session in enumerate(sessions):
            rows = by_session[session]
            self._advance_scenario(
                base_state, rows, rows_by_key, index, session, base_schedule,
                base_liquidity,
            )
            if stress_schedule is not None:
                self._advance_scenario(
                    stress_state, rows, rows_by_key, index, session,
                    stress_schedule, stress_liquidity,
                )

            if index in decision_set and index + 1 < len(sessions):
                decision_time = self._decision_time(session, rows)
                execution_time = sessions[index + 1]
                artifact_id = artifacts.artifact_for(decision_time)
                prepared = self.decision_provider(decision_time, execution_time)
                self.prepared_decision_count += 1
                base_cycle = self._plan_paired(
                    prepared, base_state, decision_time, execution_time,
                    artifact_id, request,
                )
                self._last_cycles[index] = base_cycle
                base_state.pending_orders = self._plan_orders(
                    base_cycle, rows_by_key, execution_time, base_state.positions,
                    base_state.settled_cash, base_schedule,
                )
                base_state.attempted_orders += len(base_state.pending_orders)
                if stress_schedule is not None:
                    stress_cycle = self._plan_paired(
                        prepared, stress_state, decision_time, execution_time,
                        artifact_id, request,
                    )
                    self._last_cycles[index] = stress_cycle
                    stress_state.pending_orders = self._plan_orders(
                        stress_cycle, rows_by_key, execution_time,
                        stress_state.positions, stress_state.settled_cash,
                        stress_schedule,
                    )
                    stress_state.attempted_orders += len(stress_state.pending_orders)

        return self._result_from_states(base_state, stress_state, initial_portfolio)

    def _advance_scenario(
        self,
        state: _ScenarioState,
        rows: pl.DataFrame,
        rows_by_key: Mapping[tuple[str, datetime], Any],
        index: int,
        session: datetime,
        schedule: CostSchedule,
        liquidity_model: LiquiditySlippageModel | None,
    ) -> None:
        """Advance one scenario's settlements, orders, and ledger for a session."""
        due = state.settlements.pop(index, 0.0)
        if due:
            state.settled_cash += due
            state.unsettled_cash -= due

        new_pending: list[dict[str, object]] = []
        for order in state.pending_orders:
            state.settled_cash, state.unsettled_cash, state.accrued_costs = (
                self._execute_order(
                    order, rows, rows_by_key, state.positions, state.settled_cash,
                    state.unsettled_cash, state.accrued_costs, state.settlements,
                    schedule, liquidity_model, index, session, state.trades,
                )
            )
        state.pending_orders = new_pending

        for r in rows.to_dicts():
            if r.get("close") is not None:
                state.last_close[str(r["instrument_id"])] = _as_float(r["close"])
        positions_value = sum(
            state.positions[i] * state.last_close[i]
            for i in state.positions
            if i in state.last_close
        )
        equity = (
            state.settled_cash
            + state.unsettled_cash
            + positions_value
            - state.accrued_costs
        )
        state.ledger.append(
            BacktestLedgerRow(
                session=session,
                settled_cash=state.settled_cash,
                unsettled_cash=state.unsettled_cash,
                positions_value=positions_value,
                accrued_costs=state.accrued_costs,
                equity=equity,
            )
        )

    def _plan_paired(
        self,
        prepared: PreparedReplayDecision,
        state: _ScenarioState,
        decision_time: datetime,
        execution_time: datetime,
        artifact_id: str,
        request: BacktestRequest,
    ) -> TradingCycleResult:
        """Run the scenario planner for one scenario against the prepared inputs."""
        portfolio = PortfolioSnapshot(
            account_snapshot_id=state.account_snapshot_id,
            as_of=decision_time,
            settled_cash=state.settled_cash,
            unsettled_cash=state.unsettled_cash,
            positions=self._snapshot_positions(state.positions, state.base_positions),
            open_order_ids=(),
        )
        cycle_request = TradingCycleRequest(
            strategy_id=request.strategy_id,
            artifact_id=artifact_id,
            dataset_id="backtest",
            decision_time=decision_time,
            execution_time=execution_time,
            risk_policy=request.risk_policy,
            mode="plan",
        )
        assert self.scenario_planner is not None
        return self.scenario_planner(prepared, portfolio, cycle_request)

    def _data_quality_evidence(self) -> dict[str, str]:
        """Immutable data-quality lineage recorded with every replay result."""
        m = self.manifest
        evidence = {
            "dataset_content_hash": m.content_hash,
            "quality_report_hash": m.quality_report_hash,
            "master_hash": m.master_hash,
            "calendar_hash": m.calendar_hash,
            "action_hash": m.corporate_action_hash,
            "cost_hash": m.cost_source_hash,
        }
        if self.cost_evidence is not None:
            evidence["cost_artifact_hash"] = self.cost_evidence.content_hash
            evidence["cost_model_id"] = self.cost_evidence.liquidity_model.model_id
            evidence["cost_params_hash"] = self.cost_evidence.base_liquidity_model.params_hash
        return evidence

    def _liquidity_model(self, *, stress: bool) -> LiquiditySlippageModel | None:
        """Resolve the explicit dynamic liquidity model for a ledger run."""
        if self.cost_evidence is None:
            return None
        return (
            self.cost_evidence.stress_liquidity_model
            if stress
            else self.cost_evidence.base_liquidity_model
        )

    def _assert_eligible_and_covered(self, panel: pl.DataFrame) -> None:
        """Fail closed unless every row is eligible and action-covered.

        A derived return across an uncovered action interval must not feed a
        replay. When the dataset is PROVISIONAL there is no verified action
        evidence, so only the eligibility gate applies; RESEARCH/PRODUCTION
        additionally require every row to carry an explicit action-coverage
        record.
        """
        status_column = "data_quality_status"
        if status_column in panel.columns:
            non_eligible = panel.filter(pl.col(status_column) != "eligible")
            if not non_eligible.is_empty():
                raise BacktestValidationError(
                    f"{non_eligible.height} non-eligible rows in replay panel"
                )
        if self.manifest.certification.value == "provisional":
            return
        coverage_column = "action_interval_covered"
        if coverage_column not in panel.columns:
            raise BacktestValidationError(
                f"{self.manifest.certification.value} replay requires action coverage"
            )
        uncovered = panel.filter(
            pl.col(coverage_column).is_null() | (~pl.col(coverage_column))
        )
        if not uncovered.is_empty():
            raise BacktestValidationError(
                f"{uncovered.height} rows cross an uncovered action interval"
            )

    def _run_ledger(
        self,
        panel: pl.DataFrame,
        by_session: dict[datetime, pl.DataFrame],
        sessions: list[datetime],
        artifacts: ArtifactSchedule,
        initial_portfolio: PortfolioSnapshot,
        request: BacktestRequest,
        schedule: CostSchedule,
        liquidity_model: LiquiditySlippageModel | None,
        *,
        adtv: pl.DataFrame | None = None,
    ) -> tuple[list[BacktestLedgerRow], list[BacktestTrade], int]:
        decision_set = {int(i) for i in request.decision_session_indices}
        rows_frame = self._rows_frame_with_adtv(panel, adtv)
        rows_by_key: Mapping[tuple[str, datetime], Any] = {
            (str(r["instrument_id"]), _as_datetime(r["session"])): r
            for r in rows_frame.to_dicts()
        }

        settled_cash = initial_portfolio.settled_cash
        unsettled_cash = initial_portfolio.unsettled_cash
        accrued_costs = 0.0
        positions: dict[str, int] = {
            p.instrument.instrument_id: int(p.quantity)
            for p in initial_portfolio.positions
            if p.quantity > 0
        }
        settlements: dict[int, float] = {}
        pending_orders: list[dict[str, object]] = []
        trades: list[BacktestTrade] = []
        ledger: list[BacktestLedgerRow] = []
        last_close: dict[str, float] = {}
        base_positions = tuple(initial_portfolio.positions)
        attempted_orders = 0

        for index, session in enumerate(sessions):
            rows = by_session[session]
            due = settlements.pop(index, 0.0)
            if due:
                settled_cash += due
                unsettled_cash -= due

            new_pending: list[dict[str, object]] = []
            for order in pending_orders:
                settled_cash, unsettled_cash, accrued_costs = self._execute_order(
                    order, rows, rows_by_key, positions, settled_cash, unsettled_cash,
                    accrued_costs, settlements, schedule, liquidity_model, index, session,
                    trades,
                )
            pending_orders = new_pending

            for r in rows.to_dicts():
                if r.get("close") is not None:
                    last_close[str(r["instrument_id"])] = _as_float(r["close"])
            positions_value = sum(
                positions[i] * last_close[i] for i in positions if i in last_close
            )
            equity = settled_cash + unsettled_cash + positions_value - accrued_costs
            ledger.append(
                BacktestLedgerRow(
                    session=session,
                    settled_cash=settled_cash,
                    unsettled_cash=unsettled_cash,
                    positions_value=positions_value,
                    accrued_costs=accrued_costs,
                    equity=equity,
                )
            )

            if index in decision_set and index + 1 < len(sessions):
                decision_time = self._decision_time(session, rows)
                artifact_id = artifacts.artifact_for(decision_time)
                portfolio = PortfolioSnapshot(
                    account_snapshot_id=initial_portfolio.account_snapshot_id,
                    as_of=decision_time,
                    settled_cash=settled_cash,
                    unsettled_cash=unsettled_cash,
                    positions=self._snapshot_positions(positions, base_positions),
                    open_order_ids=(),
                )
                cycle = self._plan(
                    panel, portfolio, artifact_id, decision_time,
                    sessions[index + 1], request,
                )
                self._last_cycles[index] = cycle
                pending_orders = self._plan_orders(
                    cycle, rows_by_key, sessions[index + 1], positions,
                    settled_cash, schedule,
                )
                attempted_orders += len(pending_orders)
        return ledger, trades, attempted_orders

    def _rows_frame_with_adtv(
        self,
        panel: pl.DataFrame,
        adtv: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """Build the execution row frame, reusing a validated supplied ADTV.

        When ``adtv`` is supplied it replaces any existing ``adtv`` column
        rather than recomputing the same causal rolling mean; otherwise the
        current calculation is preserved. A missing key column, a non-finite
        cached ADTV value, or a supplied ADTV that does not cover every replay
        row raises ``ValueError``.
        """
        if adtv is None:
            return panel.sort("session").with_columns(
                pl.col("trading_value")
                .rolling_mean(self.adtv_window, min_samples=1)
                .over("instrument_id")
                .alias("adtv")
            )
        missing = [c for c in ("instrument_id", "session", "adtv") if c not in adtv.columns]
        if missing:
            raise ValueError(f"supplied ADTV lookup must carry {', '.join(missing)}")
        non_finite = adtv.filter(
            pl.col("adtv").is_not_null() & ~pl.col("adtv").is_finite()
        )
        if not non_finite.is_empty():
            raise ValueError("non-finite cached ADTV input in replay rows")
        rows = panel.drop("adtv", strict=False).join(
            adtv, on=["instrument_id", "session"], how="left",
        )
        uncovered = rows.filter(pl.col("adtv").is_null())
        if not uncovered.is_empty():
            raise ValueError(
                f"supplied ADTV lookup does not cover {uncovered.height} replay rows"
            )
        return rows

    def _snapshot_positions(
        self,
        positions: dict[str, int],
        base_positions: tuple[Position, ...],
    ) -> tuple[Position, ...]:
        result: list[Position] = []
        for position in base_positions:
            instrument_id = position.instrument.instrument_id
            quantity = positions.get(instrument_id, 0)
            if quantity > 0:
                result.append(
                    Position(
                        instrument=position.instrument,
                        quantity=quantity,
                        average_cost=position.average_cost,
                    )
                )
        return tuple(result)

    def _plan(
        self,
        panel: pl.DataFrame,
        portfolio: PortfolioSnapshot,
        artifact_id: str,
        decision_time: datetime,
        execution_time: datetime,
        request: BacktestRequest,
    ) -> TradingCycleResult:
        visible = panel.filter(
            pl.col("available_time") <= decision_time
        ) if "available_time" in panel.columns else panel
        snapshot = DatasetSnapshot(manifest=self.manifest, frame=visible)
        cycle_request = TradingCycleRequest(
            strategy_id=request.strategy_id,
            artifact_id=artifact_id,
            dataset_id="backtest",
            decision_time=decision_time,
            execution_time=execution_time,
            risk_policy=request.risk_policy,
            mode="plan",
        )
        return self.planner(
            snapshot, self.registry, self.instruments, portfolio, cycle_request
        )

    def _decision_time(self, session: datetime, rows: pl.DataFrame) -> datetime:
        values = [
            r["available_time"]
            for r in rows.to_dicts()
            if r.get("available_time") is not None
        ]
        if not values:
            raise BacktestValidationError("no available_time at decision session")
        return max(_as_datetime(v) for v in values)

    def _plan_orders(
        self,
        cycle: TradingCycleResult,
        rows_by_key: Mapping[tuple[str, datetime], Any],
        execution_session: datetime,
        positions: dict[str, int],
        settled_cash: float,
        schedule: CostSchedule,
    ) -> list[dict[str, object]]:
        intents = list(cycle.intents)
        orders: list[dict[str, object]] = []
        for intent in intents:
            instrument_id = intent.instrument_id
            row = rows_by_key.get((instrument_id, execution_session))
            if row is None or row.get("open") is None or _as_float(row["open"]) <= 0:
                continue
            price = _as_float(row["open"])
            current = positions.get(instrument_id, 0)
            target_qty = int(intent.target_value / price)
            delta = target_qty - current
            if delta == 0:
                continue
            orders.append(
                {
                    "intent": intent,
                    "instrument_id": instrument_id,
                    "price": price,
                    "delta": delta,
                }
            )
        del settled_cash, schedule
        return orders

    def _execute_order(
        self,
        order: dict[str, object],
        rows: pl.DataFrame | None,
        rows_by_key: Mapping[tuple[str, datetime], Any],
        positions: dict[str, int],
        settled_cash: float,
        unsettled_cash: float,
        accrued_costs: float,
        settlements: dict[int, float],
        schedule: CostSchedule,
        liquidity_model: LiquiditySlippageModel | None,
        session_index: int,
        session: datetime,
        trades: list[BacktestTrade],
    ) -> tuple[float, float, float]:
        instrument_id = str(order["instrument_id"])
        row = rows_by_key.get((instrument_id, session))
        if row is None:
            trades.append(self._unfilled(session, instrument_id, order, "no-session-row"))
            return settled_cash, unsettled_cash, accrued_costs
        open_price = row.get("open")
        if open_price is None or _as_float(open_price) <= 0:
            trades.append(self._unfilled(session, instrument_id, order, "missing-open"))
            return settled_cash, unsettled_cash, accrued_costs
        if row.get("limit_locked") is True:
            trades.append(self._unfilled(session, instrument_id, order, "limit-locked"))
            return settled_cash, unsettled_cash, accrued_costs
        if row.get("action_interval_covered") is False:
            trades.append(self._unfilled(session, instrument_id, order, "no-action-coverage"))
            return settled_cash, unsettled_cash, accrued_costs

        cost_point = schedule.cost_for(session)
        reference_open = _as_float(open_price)
        delta = _as_int(order["delta"])
        evidence = self.cost_evidence
        stress = bool(liquidity_model is not None and liquidity_model.stress_multiplier > 1.0)
        adtv = _as_float(row.get("adtv") or 0.0)
        if adtv <= 0:
            trades.append(self._unfilled(session, instrument_id, order, "no-capacity-data"))
            return settled_cash, unsettled_cash, accrued_costs
        capacity_qty = int((0.005 * adtv) // reference_open)
        if capacity_qty <= 0:
            trades.append(self._unfilled(session, instrument_id, order, "insufficient-capacity"))
            return settled_cash, unsettled_cash, accrued_costs

        if delta > 0:
            quantity = min(delta, capacity_qty)
            volatility = None
            if evidence is not None:
                volatility = _as_float(row.get("feature__volatility_20d") or 0.0)
                if volatility <= 0:
                    trades.append(
                        self._unfilled(session, instrument_id, order, "missing-liquidity-input")
                    )
                    return settled_cash, unsettled_cash, accrued_costs
            estimate_gross = quantity * reference_open
            fill_price = self._adverse_fill_price(
                reference_open, estimate_gross, adtv, volatility, effective_time=session,
                side="BUY", stress=stress, schedule=schedule, liquidity_model=liquidity_model,
            )
            gross = quantity * fill_price
            buy_rate = self._fill_cost_rate(
                evidence, instrument_id, "BUY", session, gross, adtv, volatility, stress,
            )
            if gross * (1.0 + buy_rate) > settled_cash:
                affordable = int(settled_cash // (fill_price * (1.0 + buy_rate)))
                quantity = min(delta, capacity_qty, affordable)
                if quantity <= 0:
                    trades.append(self._unfilled(session, instrument_id, order, "insufficient-cash"))
                    return settled_cash, unsettled_cash, accrued_costs
                gross = quantity * fill_price
                fill_price = self._adverse_fill_price(
                    reference_open, gross, adtv, volatility, effective_time=session,
                    side="BUY", stress=stress, schedule=schedule, liquidity_model=liquidity_model,
                )
                gross = quantity * fill_price
                buy_rate = self._fill_cost_rate(
                    evidence, instrument_id, "BUY", session, gross, adtv, volatility, stress,
                )
            cost = gross * buy_rate
            settled_cash -= gross
            accrued_costs += cost
            positions[instrument_id] = positions.get(instrument_id, 0) + quantity
            trades.append(
                BacktestTrade(
                    session, instrument_id, "BUY", quantity, fill_price, gross, cost,
                    "filled" if quantity == delta else "partial",
                    self._cost_breakdown(
                        evidence, instrument_id, "BUY", session, fill_price, gross, adtv,
                        volatility, stress,
                    ),
                )
            )
        else:
            sell_qty = -delta
            held = positions.get(instrument_id, 0)
            if held <= 0:
                trades.append(self._unfilled(session, instrument_id, order, "no-holdings"))
                return settled_cash, unsettled_cash, accrued_costs
            sell_qty = min(sell_qty, held)
            volatility = None
            if evidence is not None:
                volatility = _as_float(row.get("feature__volatility_20d") or 0.0)
                if adtv <= 0 or volatility <= 0:
                    trades.append(
                        self._unfilled(session, instrument_id, order, "missing-liquidity-input")
                    )
                    return settled_cash, unsettled_cash, accrued_costs
            estimate_gross = sell_qty * reference_open
            fill_price = self._adverse_fill_price(
                reference_open, estimate_gross, adtv, volatility, effective_time=session,
                side="SELL", stress=stress, schedule=schedule, liquidity_model=liquidity_model,
            )
            gross = sell_qty * fill_price
            sell_rate = self._fill_cost_rate(
                evidence, instrument_id, "SELL", session, gross, adtv, volatility, stress,
            )
            cost = gross * sell_rate
            positions[instrument_id] = held - sell_qty
            accrued_costs += cost
            unsettled_cash += gross
            settlement_lag = (
                evidence.settlement_days if evidence is not None else cost_point.settlement_days
            )
            settlements[session_index + settlement_lag] = (
                settlements.get(session_index + settlement_lag, 0.0) + gross
            )
            trades.append(
                BacktestTrade(
                    session, instrument_id, "SELL", sell_qty, fill_price, gross, cost,
                    "filled" if sell_qty == -delta else "partial",
                    self._cost_breakdown(
                        evidence, instrument_id, "SELL", session, fill_price, gross, adtv,
                        volatility, stress,
                    ),
                )
            )
        return settled_cash, unsettled_cash, accrued_costs

    def _adverse_fill_price(
        self,
        reference_open: float,
        notional: float,
        adtv_20d: float,
        daily_volatility: float | None,
        *,
        effective_time: datetime,
        side: str,
        stress: bool,
        schedule: CostSchedule,
        liquidity_model: LiquiditySlippageModel | None,
    ) -> float:
        """Adverse price rounded to the effective tick (up for buys, down for sells).

        The one-way slippage (half spread plus square-root impact) is embedded
        in the fill price and never charged again as a separate cost.
        """
        if liquidity_model is not None:
            adverse_bps = liquidity_model.slippage_bps(
                notional=notional,
                adtv_20d=adtv_20d,
                daily_volatility=daily_volatility or 0.01,
                reference_price=reference_open,
                effective_time=effective_time,
            )
        else:
            adverse_bps = schedule.cost_for(effective_time).slippage_bps
        adverse = reference_open * (1.0 + adverse_bps / 10_000.0)
        tick = self._tick_for(reference_open, effective_time)
        if side == "BUY":
            return max(ceil(adverse / tick) * tick, tick)
        return max(floor(adverse / tick) * tick, tick)

    def _tick_for(self, price: float, effective_time: datetime) -> float:
        if self.cost_evidence is not None:
            return self.cost_evidence.tick_schedule.tick_size(price, effective_time)
        return 1.0

    def _fill_cost_rate(
        self,
        evidence: CostEvidence | None,
        instrument_id: str,
        side: str,
        effective_time: datetime,
        notional: float,
        adtv_20d: float,
        daily_volatility: float | None,
        stress: bool,
    ) -> float:
        """Commission plus statutory tax only; spread/impact live in the fill price."""
        if evidence is None:
            point = self.cost_schedule.cost_for(effective_time)
            rate = point.commission_rate
            if side == "SELL":
                rate += point.tax_rate
            return rate
        breakdown = self._resolved_cost(
            evidence, instrument_id, side, effective_time, 1.0, notional, adtv_20d,
            daily_volatility, stress,
        )[0]
        rate = breakdown.commission_rate
        if side == "SELL":
            rate += breakdown.sell_tax_rate
        return rate

    def _cost_breakdown(
        self,
        evidence: CostEvidence | None,
        instrument_id: str,
        side: str,
        effective_time: datetime,
        price: float,
        notional: float,
        adtv_20d: float,
        daily_volatility: float | None,
        stress: bool,
    ) -> dict[str, object] | None:
        if evidence is None:
            return None
        breakdown, artifact_hash = self._resolved_cost(
            evidence, instrument_id, side, effective_time, price, notional, adtv_20d,
            daily_volatility, stress,
        )
        return breakdown.to_dict(artifact_hash=artifact_hash)

    def _resolved_cost(
        self,
        evidence: CostEvidence,
        instrument_id: str,
        side: str,
        effective_time: datetime,
        price: float,
        notional: float,
        adtv_20d: float,
        daily_volatility: float | None,
        stress: bool,
    ) -> tuple[FillCostBreakdown, str]:
        breakdown, artifact_hash = resolve_fill_cost(
            evidence,
            side=side,
            market=krx_market_for_code(instrument_id),
            price=price,
            notional=notional,
            adtv_20d=adtv_20d,
            daily_volatility=daily_volatility or 0.01,
            effective_time=effective_time,
            stress=stress,
        )
        return breakdown, artifact_hash

    @staticmethod
    def _unfilled(
        session: datetime,
        instrument_id: str,
        order: dict[str, object],
        reason: str,
    ) -> BacktestTrade:
        return BacktestTrade(
            session=session,
            instrument_id=instrument_id,
            side="SELL" if _as_int(order["delta"]) < 0 else "BUY",
            quantity=0,
            price=None,
            gross=None,
            cost=None,
            reason=reason,
        )

    @staticmethod
    def _metrics(
        ledger: list[BacktestLedgerRow],
        trades: list[BacktestTrade],
    ) -> dict[str, float]:
        equity = np.asarray([row.equity for row in ledger], dtype=float)
        if equity.size < 2 or equity[0] <= 0:
            return {
                "cagr": 0.0,
                "annualized_volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "cost_drag": 0.0,
                "exposure": 0.0,
            }
        returns = np.diff(equity) / equity[:-1]
        mean_equity = float(np.mean(equity))
        cagr = (equity[-1] / equity[0]) ** (252.0 / equity.size) - 1.0
        volatility = float(np.std(returns, ddof=0) * np.sqrt(252.0))
        sharpe = (
            float(np.mean(returns) / np.std(returns, ddof=0)) * np.sqrt(252.0)
            if np.std(returns, ddof=0) > 0
            else 0.0
        )
        peaks = np.maximum.accumulate(equity)
        dd = (peaks - equity) / np.where(peaks > 0, peaks, 1.0)
        drawdown = float(np.max(dd)) if dd.size else 0.0
        gross_notional = sum(t.gross for t in trades if t.gross is not None)
        total_cost = sum(t.cost for t in trades if t.cost is not None)
        positions_value = sum(row.positions_value for row in ledger)
        return {
            "cagr": cagr,
            "annualized_volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "turnover": gross_notional / mean_equity,
            "cost_drag": total_cost / mean_equity,
            "exposure": (
                positions_value / mean_equity / equity.size if equity.size else 0.0
            ),
        }
