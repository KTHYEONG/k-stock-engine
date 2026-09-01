# mypy: ignore-errors
"""Screen calendar capacity, plan/settings resolution, route diagnostic records."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.stocks.ml.labels import SESSION_COLUMN


@dataclass(frozen=True, slots=True)
class ScreenCalendarCapacity:
    scheduled_decision_count: int
    names_per_session: int
    required_rows: int
    def __post_init__(self) -> None:
        if not isinstance(self.scheduled_decision_count, int) or self.scheduled_decision_count < 0:
            raise ValueError("scheduled_decision_count must be non-negative int")
        if not isinstance(self.names_per_session, int) or self.names_per_session < 1:
            raise ValueError("names_per_session must be positive int")
        if not isinstance(self.required_rows, int) or self.required_rows < 0:
            raise ValueError("required_rows must be non-negative int")
        if int(self.required_rows) != int(self.scheduled_decision_count) * int(self.names_per_session):
            raise ValueError("required_rows must equal scheduled_decision_count * names_per_session")

def resolve_screen_calendar_capacity(frame: pl.DataFrame, *, decision_cadence_sessions: int, names_per_session: int) -> ScreenCalendarCapacity:
    if not isinstance(decision_cadence_sessions, int) or decision_cadence_sessions < 1:
        raise ValueError("decision_cadence_sessions must be positive int")
    if not isinstance(names_per_session, int) or names_per_session < 1:
        raise ValueError("names_per_session must be positive int")
    session_col = SESSION_COLUMN if SESSION_COLUMN in frame.columns else ("session_index" if "session_index" in frame.columns else None)
    if session_col is None:
        raise ValueError("frame must carry session for calendar capacity")
    try:
        sessions_sorted = sorted(frame[session_col].unique().to_list())
    except Exception:
        sessions_sorted = frame[session_col].unique().sort().to_list()
    if not sessions_sorted:
        scheduled = []
    elif len(sessions_sorted) >= 2:
        try:
            from src.stocks.trading.rebalance_schedule import rebalance_session_indices
            idxs = rebalance_session_indices(tuple(sessions_sorted), min(sessions_sorted), max(sessions_sorted), int(decision_cadence_sessions), legacy_daily=False)
            scheduled = [sessions_sorted[i] for i in idxs if 0 <= i < len(sessions_sorted)]
        except Exception:
            scheduled = sessions_sorted[:: int(decision_cadence_sessions)]
    else:
        scheduled = sessions_sorted
    scheduled_count = len(scheduled)
    required = int(scheduled_count) * int(names_per_session)
    return ScreenCalendarCapacity(scheduled_decision_count=int(scheduled_count), names_per_session=int(names_per_session), required_rows=int(required))

@dataclass(frozen=True, slots=True)
class ResolvedModelSelectionPlan:
    horizon_sessions: int
    rebalance_frequency_sessions: int
    top_k: int
    policy_profile: object
    compute_budget: object
    def __post_init__(self) -> None:
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        if self.rebalance_frequency_sessions < 1:
            raise ValueError("rebalance_frequency_sessions must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.policy_profile is None:
            raise ValueError("policy_profile must be non-empty")

def resolve_model_selection_plan(request, settings) -> ResolvedModelSelectionPlan:
    if len(request.candidate_horizon_sessions) != 1:
        raise ValueError("research-only model-selection requires exactly one candidate horizon")
    if len(request.execution_frontier.candidate_horizon_sessions) != 1:
        raise ValueError("research-only model-selection plan requires exactly one frontier horizon")
    if len(request.execution_frontier.candidate_rebalance_frequency_sessions) != 1:
        raise ValueError("research-only model-selection plan requires exactly one C value")
    if len(request.execution_frontier.candidate_top_k) != 1:
        raise ValueError("research-only model-selection plan requires exactly one K value")
    horizon = int(request.candidate_horizon_sessions[0])
    cand_c = int(request.execution_frontier.candidate_rebalance_frequency_sessions[0])
    cand_k = int(request.execution_frontier.candidate_top_k[0])
    if int(settings.reference_rebalance_frequency_sessions) != cand_c or int(settings.reference_top_k) != cand_k:
        raise ValueError("reference execution settings do not match the bound frontier")
    target_profile_id = str(settings.reference_policy_profile_id)
    feasible = request.execution_frontier.feasible_cells(request.portfolio.max_exposure, request.portfolio.max_single_weight)
    found = any(h == horizon and c == cand_c and k == cand_k for h, c, k in feasible)
    if not found:
        raise ValueError(f"resolved execution cell (H={horizon},C={cand_c},K={cand_k}) is infeasible")
    profile = next((p for p in request.policy_profiles if str(p.profile_id) == target_profile_id), None)
    if profile is None:
        raise ValueError(f"reference policy profile {target_profile_id!r} not found")
    return ResolvedModelSelectionPlan(horizon_sessions=horizon, rebalance_frequency_sessions=cand_c, top_k=cand_k, policy_profile=profile, compute_budget=settings.compute_budget)
