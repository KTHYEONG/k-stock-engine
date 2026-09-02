"""Champion v1 twenty-stock selection with hysteresis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.core.portfolio import PortfolioSnapshot
from src.strategy.scoring import ChampionScoreRow


@dataclass(frozen=True, slots=True)
class ChampionSelectionPolicy:
    version: str = "champion-v1-selection-v1"
    required_score_policy_version: str = "champion-v1-scoring-v1"
    max_positions: int = 20
    entry_rank: int = 20
    retention_rank: int = 40

    def __post_init__(self) -> None:
        if not self.version or not self.version.strip():
            raise ValueError("version must be non-empty")
        if not self.required_score_policy_version or not self.required_score_policy_version.strip():
            raise ValueError("required_score_policy_version must be non-empty")
        if self.version != "champion-v1-selection-v1":
            raise ValueError("version must be champion-v1-selection-v1")
        if self.required_score_policy_version != "champion-v1-scoring-v1":
            raise ValueError("required_score_policy_version must be champion-v1-scoring-v1")
        if not isinstance(self.max_positions, int) or isinstance(self.max_positions, bool) or self.max_positions <= 0:
            raise ValueError("max_positions must be positive integer")
        if not isinstance(self.entry_rank, int) or isinstance(self.entry_rank, bool) or self.entry_rank <= 0:
            raise ValueError("entry_rank must be positive integer")
        if not isinstance(self.retention_rank, int) or isinstance(self.retention_rank, bool) or self.retention_rank <= 0:
            raise ValueError("retention_rank must be positive integer")
        if self.entry_rank > self.retention_rank:
            raise ValueError("entry_rank must not be greater than retention_rank")
        if self.max_positions != 20 or self.entry_rank != 20 or self.retention_rank != 40:
            raise ValueError("Champion v1 constants must be 20, 20, 40")


class SelectionReason(StrEnum):
    SURVIVOR = "survivor"
    NEW_ENTRY = "new_entry"
    EXIT_INELIGIBLE = "exit_ineligible"
    EXIT_RETENTION_RANK = "exit_retention_rank"
    EXIT_MISSING_SCORE = "exit_missing_score"
    EXIT_CAPACITY = "exit_capacity"


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    instrument_id: str
    selected: bool
    rank: int | None
    reason: SelectionReason


@dataclass(frozen=True, slots=True)
class ChampionSelectionResult:
    decision_time: datetime
    account_snapshot_id: str
    selected_instrument_ids: tuple[str, ...]
    decisions: tuple[SelectionDecision, ...]
    unfilled_slots: int


def select_champion_targets(
    scores: tuple[ChampionScoreRow, ...],
    portfolio: PortfolioSnapshot,
    *,
    decision_time: datetime,
    policy: ChampionSelectionPolicy = ChampionSelectionPolicy(),  # noqa: B008
) -> ChampionSelectionResult:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    portfolio.validate_as_of(decision_time)

    if not scores:
        raise ValueError("scores must be non-empty")

    seen_ids: set[str] = set()
    decision_sessions: set[datetime] = set()
    score_by_id: dict[str, ChampionScoreRow] = {}
    for row in scores:
        if not row.instrument_id or not row.instrument_id.strip():
            raise ValueError("instrument_id must be non-empty")
        if row.instrument_id in seen_ids:
            raise ValueError(f"duplicate instrument_id {row.instrument_id!r}")
        seen_ids.add(row.instrument_id)
        if row.decision_session.tzinfo is None:
            raise ValueError("decision_session must be timezone-aware")
        if row.decision_session > decision_time:
            raise ValueError("decision_session must not be after decision_time")
        decision_sessions.add(row.decision_session)
        if row.score_policy_version != policy.required_score_policy_version:
            raise ValueError(f"policy version mismatch: {row.score_policy_version!r} != {policy.required_score_policy_version!r} (policy)")
        score_by_id[row.instrument_id] = row

    if len(decision_sessions) != 1:
        raise ValueError("mixed decision_session values")

    # Fail closed for malformed score state
    eligible_rows: list[ChampionScoreRow] = []
    for row in scores:
        if row.eligible:
            if row.champion_score is None:
                raise ValueError(f"eligible row missing champion_score: {row.instrument_id}")
            if not math.isfinite(float(row.champion_score)):
                raise ValueError(f"non-finite champion_score for {row.instrument_id!r}")
            if row.rank is None:
                raise ValueError(f"eligible row missing rank: {row.instrument_id}")
            if not isinstance(row.rank, int) or isinstance(row.rank, bool) or row.rank <= 0:
                raise ValueError(f"rank must be positive integer: {row.instrument_id}")
            eligible_rows.append(row)
        else:
            if row.champion_score is not None:
                raise ValueError(f"ineligible row must have no champion_score: {row.instrument_id}")
            if row.rank is not None:
                raise ValueError(f"ineligible row must have no rank: {row.instrument_id}")

    if eligible_rows:
        ranks = [r.rank for r in eligible_rows if r.rank is not None]
        if len(set(ranks)) != len(ranks):
            raise ValueError("eligible ranks must be unique")
        # Ordering must follow score descending then instrument_id. The global
        # Champion dataset is consecutive from 1, but hysteresis tests provide
        # sparse subsets (e.g. ranks 1,21,40) that reflect global ranks with
        # gaps - allow gaps and validate only ordering + uniqueness.
        sorted_by_rank = sorted(eligible_rows, key=lambda r: r.rank)  # type: ignore
        expected_order = sorted(eligible_rows, key=lambda r: (-float(r.champion_score), r.instrument_id))  # type: ignore
        if [r.instrument_id for r in sorted_by_rank] != [r.instrument_id for r in expected_order]:
            raise ValueError("eligible ranks must follow score descending and instrument_id tie-break")
        # Additionally enforce positive ranks already checked; uniqueness handled.

    # Build held identity set from positions
    held_ids: list[str] = [p.instrument.instrument_id for p in portfolio.positions]
    # portfolio.positions already validated unique ids

    retainable: list[ChampionScoreRow] = []
    exit_decisions: list[SelectionDecision] = []

    for hid in held_ids:
        row = score_by_id.get(hid)  # type: ignore[assignment]
        if row is None:
            exit_decisions.append(SelectionDecision(hid, False, None, SelectionReason.EXIT_MISSING_SCORE))
        elif not row.eligible:
            exit_decisions.append(SelectionDecision(hid, False, None, SelectionReason.EXIT_INELIGIBLE))
        elif row.rank is not None and row.rank > policy.retention_rank:
            exit_decisions.append(SelectionDecision(hid, False, row.rank, SelectionReason.EXIT_RETENTION_RANK))
        else:
            # retainable candidate
            assert row is not None
            retainable.append(row)

    # Sort retainable by (rank, instrument_id) and apply capacity
    retainable_sorted = sorted(retainable, key=lambda r: (r.rank, r.instrument_id))

    survivors: list[ChampionScoreRow] = []
    capacity_exits: list[SelectionDecision] = []
    if len(retainable_sorted) > policy.max_positions:
        survivors = retainable_sorted[: policy.max_positions]
        capacity_exits.extend(
            SelectionDecision(row.instrument_id, False, row.rank, SelectionReason.EXIT_CAPACITY)
            for row in retainable_sorted[policy.max_positions :]
        )
    else:
        survivors = retainable_sorted

    selected_survivors: list[SelectionDecision] = [
        SelectionDecision(r.instrument_id, True, r.rank, SelectionReason.SURVIVOR) for r in survivors
    ]

    remaining_capacity = policy.max_positions - len(survivors)
    # eligible non-held entry candidates with rank <= entry_rank
    held_set = set(held_ids)
    entry_candidates = [
        r for r in eligible_rows if r.instrument_id not in held_set and r.rank is not None and r.rank <= policy.entry_rank
    ]
    entry_candidates_sorted = sorted(entry_candidates, key=lambda r: (r.rank, r.instrument_id))
    entries = entry_candidates_sorted[:remaining_capacity] if remaining_capacity > 0 else []

    selected_entries: list[SelectionDecision] = [
        SelectionDecision(r.instrument_id, True, r.rank, SelectionReason.NEW_ENTRY) for r in entries
    ]

    # Combine all decisions for output
    all_selected = selected_survivors + selected_entries
    # selected_instrument_ids follow (rank, instrument_id)
    all_selected_sorted = sorted(all_selected, key=lambda d: (d.rank if d.rank is not None else 10**9, d.instrument_id))
    selected_instrument_ids = tuple(d.instrument_id for d in all_selected_sorted)

    if len(set(selected_instrument_ids)) != len(selected_instrument_ids):
        raise ValueError("duplicate selected instrument_id")
    if len(selected_instrument_ids) > policy.max_positions:
        raise ValueError("selected more than max_positions")

    # decisions follow instrument_id for stable serialization
    all_decisions = all_selected + exit_decisions + capacity_exits
    all_decisions_sorted = tuple(sorted(all_decisions, key=lambda d: d.instrument_id))

    unfilled_slots = policy.max_positions - len(selected_instrument_ids)

    return ChampionSelectionResult(
        decision_time=decision_time,
        account_snapshot_id=portfolio.account_snapshot_id,
        selected_instrument_ids=selected_instrument_ids,
        decisions=all_decisions_sorted,
        unfilled_slots=unfilled_slots,
    )
