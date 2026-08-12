"""Deterministic multi-fidelity economic selection policy.

The redesign replaces the fixed ``screen -> top-8 -> every-fold full refit ->
full replay`` pipeline with an evidence-adaptive successive-promotion funnel:
for each route, ``route_budget`` screen trials are run once, the
``ceil(sqrt(route_budget))`` best positive-screen trials are promoted to a
fold-0 full refit, and the ``ceil(promotion_width / fold_count)`` best
all-positive candidates become exact economic finalists. Backfill keeps the
fail-closed property: a rejected finalist is deterministically replaced from
the promoted list until ``finalist_width`` all-positive candidates exist or the
list is exhausted.

The policy is versioned through :data:`SELECTION_POLICY_VERSION` so every
promotion decision, resume fingerprint, metric, and artifact provenance can be
traced to the exact selection rule that produced it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SELECTION_POLICY_VERSION = "economic-selection-v1"


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Evidence-adaptive promotion widths for one route.

    ``route_budget`` is the number of screen trials dedicated to the route.
    ``promotion_width`` is the number of positive-screen candidates promoted
    to a fold-0 full refit; ``finalist_width`` is the number of all-positive
    candidates taken to exact economic replay. ``widths`` exposes the
    ``(route_budget, promotion_width, finalist_width)`` triple that drives the
    81-trial three-route three-fold profile to a 27-to-6-to-2 funnel.
    """

    route_budget: int
    promotion_width: int
    finalist_width: int
    fold_count: int

    @classmethod
    def for_budget(
        cls,
        *,
        total_trials: int,
        route_count: int,
        fold_count: int,
    ) -> SelectionPolicy:
        """Derive the promotion funnel widths for an equal route budget.

        ``promotion_width`` grows with the square root of the route budget and
        never exceeds the number of positive-screen candidates; a degenerate
        budget of one trial still promotes at least two candidates so a screen
        quality ranking exists before economic evidence is spent.
        ``finalist_width`` shrinks the promotion list by one fold per remaining
        refit stage. Raises ``ValueError`` for non-positive inputs.
        """
        if total_trials < 1 or route_count < 1 or fold_count < 1:
            raise ValueError("total_trials, route_count, and fold_count must be positive")
        route_budget = max(1, total_trials // route_count)
        promotion_width = min(
            route_budget,
            max(2, math.ceil(math.sqrt(route_budget))),
        )
        finalist_width = min(
            promotion_width,
            max(1, math.ceil(promotion_width / fold_count)),
        )
        return cls(
            route_budget=route_budget,
            promotion_width=promotion_width,
            finalist_width=finalist_width,
            fold_count=fold_count,
        )

    @property
    def widths(self) -> tuple[int, int, int]:
        """The ``(route_budget, promotion_width, finalist_width)`` profile."""
        return (self.route_budget, self.promotion_width, self.finalist_width)

    def to_json_safe(self) -> dict[str, object]:
        """JSON-safe policy evidence for metrics and provenance."""
        return {
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "route_budget": int(self.route_budget),
            "promotion_width": int(self.promotion_width),
            "finalist_width": int(self.finalist_width),
            "fold_count": int(self.fold_count),
        }
