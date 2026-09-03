"""Walk-forward fold and benchmark contracts."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.core.ledger import LedgerNav

__all__ = [
    "BenchmarkConstituent",
    "BenchmarkKind",
    "EligibleUniverseSnapshot",
    "FoldReplay",
    "WalkForwardFold",
    "benchmark_target_weights",
    "build_walk_forward_folds",
    "stitch_oos_ledger_nav",
]


@dataclass(frozen=True, slots=True)
class BenchmarkConstituent:
    instrument_id: str
    close: float
    market_cap: float

    def __post_init__(self) -> None:
        if not self.instrument_id or not isinstance(self.instrument_id, str):
            raise ValueError("instrument_id must be non-empty")
        for name, val in (("close", self.close), ("market_cap", self.market_cap)):
            if isinstance(val, bool):
                raise ValueError(f"{name} must be finite")
            if not isinstance(val, (int, float)):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite")
            if float(val) <= 0:
                raise ValueError(f"{name} must be positive finite")


class BenchmarkKind(StrEnum):
    CAP_WEIGHT = "cap_weight"
    EQUAL_WEIGHT = "equal_weight"


@dataclass(frozen=True, slots=True)
class EligibleUniverseSnapshot:
    as_of: datetime
    constituents: tuple[BenchmarkConstituent, ...]

    def __init__(self, as_of: datetime, constituents: tuple[BenchmarkConstituent, ...]) -> None:
        # Use object.__setattr__ due to frozen
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "constituents", constituents)
        # Validation after setting to allow frozen check in post_init via manual
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, datetime):
            raise ValueError("as_of must be datetime")
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be aware")
        if not isinstance(self.constituents, tuple):
            raise ValueError("constituents must be tuple")
        if len(self.constituents) == 0:
            raise ValueError("constituents must be non-empty")
        seen: set[str] = set()
        for c in self.constituents:
            if not isinstance(c, BenchmarkConstituent):
                raise ValueError("constituents must be BenchmarkConstituent")
            # BenchmarkConstituent already validates close/market_cap but double-check message
            if not math.isfinite(float(c.close)) or float(c.close) <= 0:
                raise ValueError("close must be positive finite")
            if not math.isfinite(float(c.market_cap)) or float(c.market_cap) <= 0:
                raise ValueError("market_cap must be positive finite")
            if c.instrument_id in seen:
                raise ValueError(f"duplicate instrument_id {c.instrument_id!r}")
            seen.add(c.instrument_id)


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_id: int
    research_sessions: tuple[datetime, ...]
    oos_sessions: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if isinstance(self.fold_id, bool) or not isinstance(self.fold_id, int):
            raise ValueError("fold_id must be integer")
        if self.fold_id < 0:
            raise ValueError("fold_id must be non-negative")
        if not isinstance(self.research_sessions, tuple) or not isinstance(self.oos_sessions, tuple):
            raise ValueError("sessions must be tuple")
        if len(self.research_sessions) == 0 or len(self.oos_sessions) == 0:
            raise ValueError("research and oos must be non-empty")
        for s in (*self.research_sessions, *self.oos_sessions):
            if not isinstance(s, datetime) or s.tzinfo is None:
                raise ValueError("sessions must be timezone-aware datetime")


@dataclass(frozen=True, slots=True)
class FoldReplay:
    fold: WalkForwardFold
    result: object  # BacktestResult - kept as object to avoid circular
    ledger_run_hash: str
    dataset_hash: str
    universe_hash: str
    execution_contract_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.fold, WalkForwardFold):
            raise ValueError("fold must be WalkForwardFold")
        # result validation checked elsewhere but ensure has daily_nav and scenario
        for name in ("ledger_run_hash", "dataset_hash", "universe_hash", "execution_contract_hash"):
            val = getattr(self, name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{name} must be non-empty")


def build_walk_forward_folds(
    sessions: tuple[datetime, ...],
    *,
    research_sessions: int = 1260,
    oos_sessions: int = 252,
) -> tuple[WalkForwardFold, ...]:
    # Type and value checks for research/oos
    if isinstance(research_sessions, bool) or not isinstance(research_sessions, int):
        raise ValueError("research_sessions must be integer")
    if isinstance(oos_sessions, bool) or not isinstance(oos_sessions, int):
        raise ValueError("oos_sessions must be integer")
    if research_sessions <= 0 or oos_sessions <= 0:
        raise ValueError("research_sessions and oos_sessions must be positive")
    if not isinstance(sessions, tuple):
        raise ValueError("sessions must be tuple")
    if len(sessions) == 0:
        raise ValueError("sessions must be non-empty")
    # Check timezone-aware, strictly increasing, unique
    for s in sessions:
        if not isinstance(s, datetime):
            raise ValueError("sessions must be datetime")
        if s.tzinfo is None:
            raise ValueError("sessions must be timezone-aware")
    for i in range(len(sessions) - 1):
        if sessions[i] >= sessions[i + 1]:
            raise ValueError("sessions must be strictly increasing")
    if len(set(sessions)) != len(sessions):
        raise ValueError("sessions must be unique")
    if len(sessions) < research_sessions + oos_sessions:
        raise ValueError("insufficient sessions for at least one full research+OOS fold")
    num_folds = (len(sessions) - research_sessions) // oos_sessions
    if num_folds < 1:
        raise ValueError("insufficient sessions for at least one full fold")
    folds: list[WalkForwardFold] = []
    for idx in range(num_folds):
        start = idx * oos_sessions
        research = tuple(sessions[start : start + research_sessions])
        oos = tuple(sessions[start + research_sessions : start + research_sessions + oos_sessions])
        if len(research) != research_sessions or len(oos) != oos_sessions:
            raise ValueError("fold slice length mismatch")
        # Own research and OOS disjoint and ordered
        if max(research) >= min(oos):
            raise ValueError("research must be strictly before oos")
        if set(research).intersection(set(oos)):
            raise ValueError("research and oos must be disjoint")
        folds.append(WalkForwardFold(fold_id=idx, research_sessions=research, oos_sessions=oos))
    # Global invariants: OOS globally disjoint and ordered, and no OOS enters its own or prior research
    oos_all: list[datetime] = []
    for f in folds:
        oos_all.extend(f.oos_sessions)
    if len(set(oos_all)) != len(oos_all):
        raise ValueError("OOS session sets must be globally disjoint")
    # Ordered: since construction ensures order, check monotonic
    for i in range(len(oos_all) - 1):
        if oos_all[i] >= oos_all[i + 1]:
            raise ValueError("OOS sessions must be ordered")
    # No OOS session may enter its own or any prior research window
    # For each fold k, union of research up to k must not contain any oos of fold k (or any future? spec says no OOS session may enter prior research, so check each oos against prior researches)
    for k, fold in enumerate(folds):
        prior_research_union: set[datetime] = set()
        for j in range(k + 1):  # includes own research
            prior_research_union.update(folds[j].research_sessions)
        # fold's oos must be disjoint from prior research union
        if not set(fold.oos_sessions).isdisjoint(prior_research_union):
            raise ValueError("OOS session may not enter its own or any prior research window")
        # Also any earlier OOS may be in later research (allowed), so we don't check reverse
    return tuple(folds)


def stitch_oos_ledger_nav(replays: tuple[FoldReplay, ...]) -> tuple[LedgerNav, ...]:
    # Local imports to avoid circular
    from src.core.ledger import LedgerNav

    if not isinstance(replays, tuple):
        raise ValueError("replays must be tuple")
    if len(replays) == 0:
        raise ValueError("replays must be non-empty")
    for r in replays:
        if not isinstance(r, FoldReplay):
            raise ValueError("replays must be FoldReplay")
    # ledger_run_hash parity: exactly one hash
    hashes = {r.ledger_run_hash for r in replays}
    if len(hashes) != 1:
        raise ValueError("ledger_run_hash must be exactly one per candidate run")
    # Check exactly one per contiguous fold_id
    sorted_replays = tuple(sorted(replays, key=lambda x: x.fold.fold_id))
    fold_ids = [r.fold.fold_id for r in sorted_replays]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("duplicate fold_id in replays")
    for i in range(1, len(fold_ids)):
        if fold_ids[i] != fold_ids[i - 1] + 1:
            raise ValueError("fold_ids must be contiguous")
    # Validate each replay's daily_nav matches OOS sessions exactly once
    stitched: list[LedgerNav] = []
    for replay in sorted_replays:
        result = replay.result
        # Expect BacktestResult with daily_nav
        if not hasattr(result, "daily_nav"):
            raise ValueError("result must have daily_nav")
        daily_nav = result.daily_nav
        if not isinstance(daily_nav, tuple):
            raise ValueError("daily_nav must be tuple")
        oos = replay.fold.oos_sessions
        if len(daily_nav) != len(oos):
            raise ValueError(f"OOS session count mismatch for fold {replay.fold.fold_id}: expected {len(oos)} got {len(daily_nav)}")
        # Check timestamps match exactly in order
        nav_times = []
        for nav in daily_nav:
            if not isinstance(nav, LedgerNav):
                raise ValueError("daily_nav must contain LedgerNav marks")
            if not isinstance(nav.as_of, datetime) or nav.as_of.tzinfo is None:
                raise ValueError("LedgerNav as_of must be aware")
            if not isinstance(nav.nav, (int, float)) or isinstance(nav.nav, bool):
                raise ValueError("LedgerNav nav must be finite")
            if not math.isfinite(float(nav.nav)) or float(nav.nav) <= 0:
                raise ValueError("LedgerNav nav must be positive finite")
            nav_times.append(nav.as_of)
        if tuple(nav_times) != oos:
            # Provide OOS in message for test matching
            raise ValueError(f"OOS timestamp mismatch for fold {replay.fold.fold_id}")
        # Strictly increasing within fold already implied by OOS increasing, but check NAV finite etc.
        # Append in order
        stitched.extend(daily_nav)
    # Global strictly increasing timestamps
    for i in range(1, len(stitched)):
        if stitched[i].as_of <= stitched[i - 1].as_of:
            raise ValueError("stitched OOS Ledger NAV must be strictly time-increasing")
        if not math.isfinite(float(stitched[i].nav)) or float(stitched[i].nav) <= 0:
            raise ValueError("stitched nav must be positive finite")
    # First element also check
    if not math.isfinite(float(stitched[0].nav)) or float(stitched[0].nav) <= 0:
        raise ValueError("stitched nav must be positive finite")
    return tuple(stitched)


def benchmark_target_weights(
    snapshot: EligibleUniverseSnapshot,
    kind: BenchmarkKind,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(snapshot, EligibleUniverseSnapshot):
        raise ValueError("snapshot must be EligibleUniverseSnapshot")
    if not isinstance(kind, BenchmarkKind):
        raise ValueError("kind must be BenchmarkKind")
    constituents = snapshot.constituents
    # Snapshot already validated, but double-check ordering
    # Canonical instrument-id ordering
    sorted_const = sorted(constituents, key=lambda c: c.instrument_id)
    n = len(sorted_const)
    if n == 0:
        raise ValueError("constituents must be non-empty")
    weights: list[tuple[str, float]] = []
    if kind is BenchmarkKind.CAP_WEIGHT:
        total = sum(float(c.market_cap) for c in sorted_const)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("total market_cap must be positive finite")
        for c in sorted_const:
            w = float(c.market_cap) / total
            if not math.isfinite(w) or w < 0:
                raise ValueError("weight must be nonnegative finite")
            weights.append((c.instrument_id, float(w)))
    else:  # EQUAL_WEIGHT
        w = 1.0 / n
        for c in sorted_const:
            weights.append((c.instrument_id, float(w)))  # noqa: PERF401
    # Validate sum to 1.0 within 1e-12
    total_w = sum(w for _, w in weights)
    if not math.isfinite(total_w) or abs(total_w - 1.0) > 1e-12:
        raise ValueError(f"weights sum {total_w} not within 1e-12 of 1.0")
    # Validate canonical ordering already
    return tuple(weights)
