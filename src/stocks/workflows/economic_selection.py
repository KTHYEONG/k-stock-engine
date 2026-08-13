"""Deterministic multi-fidelity economic selection policy.

The v4 stability redesign replaces the v2/v3 ``screen -> promoted -> every-fold
full refit -> six-policy replay`` pipeline with an economically aligned proxy
screen funnel: for each route, ``route_budget`` screen trials run against every
purged fixed-stride proxy fold, the ``ceil(sqrt(route_budget))`` best
positive-screen trials are promoted to a full fold-0 refit, and the
``ceil(sqrt(promotion_width))`` all-positive candidates become the route's
economic finalists. Each finalist is replayed exactly once under the single
pre-registered default ``StockRiskPolicy``; the six-policy compounding grid is
removed from active selection. A rejected finalist is never silently replaced:
the route has no champion, which is conservative in the financial sense (an
additional ``NO_TRADE`` is acceptable, a relaxed gate is not).

The policy is versioned through :data:`SELECTION_POLICY_VERSION` so every
promotion decision, resume fingerprint, metric, and artifact provenance can be
traced to the exact selection rule that produced it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SELECTION_POLICY_VERSION = "economic-selection-v6-confirmed-recovery"


@dataclass(frozen=True, slots=True)
class ScreenFidelityPolicy:
    """Fixed-stride proxy screen funnel widths for one route.

    ``route_budget`` is the number of screen trials dedicated to the route.
    ``proxy_session_stride`` is the deterministic in-split session step used to
    build the proxy screen context (derived from ``ceil(sqrt(route_budget))``,
    so a larger search budget automatically keeps a proportional fixed temporal
    sample). ``promotion_width`` is the number of candidates confirmed on the
    remaining proxy folds and promoted to a full fold-0 refit;
    ``confirmation_width`` is the bounded confirmation funnel width
    (equal to ``promotion_width``); ``economic_finalist_width`` is the number
    of all-positive candidates taken to the exact single-policy compounding
    replay (``ceil(sqrt(promotion_width))`` per route); ``recovery_width`` is
    the pre-registered recovery ceiling (equal to ``economic_finalist_width``)
    for candidates admitted solely on a positive pooled mean log excess.
    ``widths`` exposes the ``(route_budget, proxy_session_stride,
    promotion_width, confirmation_width, economic_finalist_width,
    recovery_width)`` profile that drives the 81-trial three-route three-fold
    profile to a 27-to-6-to-6-to-6-to-3-to-3 funnel.
    """

    route_budget: int
    proxy_session_stride: int
    promotion_width: int
    confirmation_width: int
    economic_finalist_width: int
    recovery_width: int

    @classmethod
    def for_budget(
        cls,
        *,
        total_trials: int,
        route_count: int,
        fold_count: int,
    ) -> ScreenFidelityPolicy:
        """Derive the v4 proxy funnel widths for an equal route budget.

        ``route_budget`` is the equal per-route screen trial count;
        ``proxy_session_stride`` is ``ceil(sqrt(route_budget))`` so the proxy
        keeps a fixed temporal sample proportional to the search budget.
        ``promotion_width`` never exceeds the route budget and a degenerate
        budget of one trial still promotes two candidates so a screen quality
        ranking exists before economic evidence is spent.
        ``confirmation_width`` equals ``promotion_width`` and bounds the
        two-stage confirmation funnel: only the top fold-0 candidates are
        replayed on the remaining proxy folds before the pooled economic
        comparator runs. ``economic_finalist_width`` is
        ``ceil(sqrt(promotion_width))``: the route's top all-positive
        candidates are each replayed exactly once under the single default
        compounding policy, and a rejected finalist yields no champion.
        ``recovery_width`` equals ``economic_finalist_width`` and caps how many
        recovery candidates (positive pooled mean log excess, no positive
        pooled lower bound) may reach full refit. Raises ``ValueError`` for
        non-positive inputs.
        """
        if total_trials < 1 or route_count < 1 or fold_count < 1:
            raise ValueError("total_trials, route_count, and fold_count must be positive")
        route_budget = max(1, total_trials // route_count)
        proxy_session_stride = max(1, math.ceil(math.sqrt(route_budget)))
        promotion_width = min(
            route_budget,
            max(2, math.ceil(math.sqrt(route_budget))),
        )
        economic_finalist_width = min(
            promotion_width,
            max(1, math.ceil(math.sqrt(promotion_width))),
        )
        confirmation_width = promotion_width
        recovery_width = economic_finalist_width
        return cls(
            route_budget=route_budget,
            proxy_session_stride=proxy_session_stride,
            promotion_width=promotion_width,
            confirmation_width=confirmation_width,
            economic_finalist_width=economic_finalist_width,
            recovery_width=recovery_width,
        )

    @property
    def widths(self) -> tuple[int, int, int, int, int, int]:
        """The ``(route_budget, stride, promotion, confirmation, finalist, recovery)`` profile."""
        return (
            self.route_budget,
            self.proxy_session_stride,
            self.promotion_width,
            self.confirmation_width,
            self.economic_finalist_width,
            self.recovery_width,
        )

    def to_json_safe(self) -> dict[str, object]:
        """JSON-safe policy evidence for metrics and provenance."""
        return {
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "route_budget": int(self.route_budget),
            "proxy_session_stride": int(self.proxy_session_stride),
            "promotion_width": int(self.promotion_width),
            "confirmation_width": int(self.confirmation_width),
            "economic_finalist_width": int(self.economic_finalist_width),
            "recovery_width": int(self.recovery_width),
            "configured_compounding_policy_cells": 1,
        }


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Legacy v1 evidence-adaptive promotion widths for one route.

    Retained for reference; the active v2 policy is
    :class:`ScreenFidelityPolicy`. ``route_budget`` is the number of screen
    trials dedicated to the route, ``promotion_width`` the number of
    positive-screen candidates promoted to a fold-0 full refit, and
    ``finalist_width`` the number of all-positive candidates taken to exact
    economic replay.
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
        """Derive the legacy promotion funnel widths for an equal route budget."""
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
            "selection_policy_version": "economic-selection-v1",
            "route_budget": int(self.route_budget),
            "promotion_width": int(self.promotion_width),
            "finalist_width": int(self.finalist_width),
            "fold_count": int(self.fold_count),
        }
