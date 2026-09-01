# mypy: ignore-errors
"""Horizon evidence, candidate blending and discovery diagnostics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HorizonEvidence:
    horizon_sessions: int
    realized_rows: int
    diagnostics: tuple[str, ...]


def _build_horizon_evidence(
    horizon_sessions: int | None = None,
    rows_by_horizon: Mapping[int, Sequence[object]] | None = None,
    **kwargs: object,
) -> HorizonEvidence:
    if horizon_sessions is None:
        candidate = kwargs.get("horizon_sessions", 0)
        horizon_sessions = int(candidate) if isinstance(candidate, (int, float)) else 0
    if not isinstance(horizon_sessions, int) or horizon_sessions < 1:
        raise ValueError("horizon_sessions must be positive")
    rows = (rows_by_horizon or {}).get(horizon_sessions, ())
    if not rows and "pre_holdout" in kwargs:
        candidate_rows = kwargs["pre_holdout"]
        rows = candidate_rows if isinstance(candidate_rows, Sequence) else ()
    diagnostics = () if rows else ("no-realized-rows",)
    return HorizonEvidence(horizon_sessions, len(rows), diagnostics)
