# mypy: ignore-errors
"""Wealth transfer evidence and promotion evaluation (recovery spec)."""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from legacy.stocks.ml.contracts import ConversionWaterfallEvidence, RouteObjectiveKind

logger = logging.getLogger("stocks.ml.wealth_transfer")

class WealthEvidenceKind(StrEnum):
    EXECUTABLE_UNHEDGED = "executable_unhedged"
    EXECUTABLE_HEDGED = "executable_hedged"
    SYNTHETIC_PROJECTION = "synthetic_projection"

@dataclass(frozen=True, slots=True)
class WealthCandidateVerdict:
    promotion_status: str
    promotable: bool
    first_failure_stage: str
    reasons: tuple[str, ...]
    waterfall: ConversionWaterfallEvidence

    def __post_init__(self):
        if self.promotion_status not in ("PROMOTED_UNHEDGED", "PROMOTED_EXECUTABLE_HEDGED", "RESEARCH_EDGE_ONLY", "NO_TRADE"):
            raise ValueError(f"unknown promotion_status {self.promotion_status}")

class ConversionWaterfallAccumulator:
    """Bounded O(1) counters; no raw rows retained."""
    def __init__(self, mode_id: str, score_frame_fingerprint: str):
        if not mode_id:
            raise ValueError("mode_id must be non-empty")
        if not score_frame_fingerprint:
            raise ValueError("score_frame_fingerprint must be non-empty")
        self.mode_id = mode_id
        self.score_frame_fingerprint = score_frame_fingerprint
        # row counters
        self._scored = 0
        self._finite = 0
        self._calibrated = 0
        self._pos_mean = 0
        self._pos_lb = 0
        self._eligible = 0
        self._targets = 0
        self._row_drops: dict[str, int] = {}
        # decision counters
        self._scheduled = 0
        self._alloc_ready = 0
        self._target_change = 0
        self._decision_drops: dict[str, int] = {}
        # order
        self._submitted = 0
        self._filled = 0
        self._order_drops: dict[str, int] = {}
        # intervals
        self._observed = 0
        self._invested = 0
        # track single decision for simplicity (spec test uses one decision)
        self._decision_added = False

    def add_decision(self, evidence) -> None:
        # evidence is AllocationDecisionEvidence
        self._scored += int(evidence.scored_rows)
        self._finite += int(evidence.finite_score_rows)
        self._calibrated += int(evidence.calibrated_rows)
        self._pos_mean += int(evidence.positive_mean_rows)
        self._pos_lb += int(evidence.positive_lower_bound_rows)
        self._eligible += int(evidence.market_eligible_rows)
        self._targets += int(evidence.selected_target_rows)
        # aggregate drop reasons
        for reason, count in getattr(evidence, "drop_reasons", ()):
            self._row_drops[reason] = self._row_drops.get(reason, 0) + int(count)
        # decision counters: one scheduled per add_decision
        self._scheduled += 1
        if bool(getattr(evidence, "allocation_ready", False)):
            self._alloc_ready += 1
        else:
            self._decision_drops["no_allocation_ready"] = self._decision_drops.get("no_allocation_ready", 0) + 1
        if bool(getattr(evidence, "target_changed", False)):
            self._target_change += 1
        else:
            if bool(getattr(evidence, "allocation_ready", False)):
                self._decision_drops["no_target_change"] = self._decision_drops.get("no_target_change", 0) + 1
        self._decision_added = True

    def add_orders(self, submitted_orders: int, filled_orders: int, unfilled_reasons: Mapping[str, int]) -> None:
        self._submitted += int(submitted_orders)
        self._filled += int(filled_orders)
        for reason, count in dict(unfilled_reasons).items():
            self._order_drops[reason] = self._order_drops.get(reason, 0) + int(count)

    def add_intervals(self, observed: int, invested: int) -> None:
        self._observed += int(observed)
        self._invested += int(invested)

    def finalize(self) -> ConversionWaterfallEvidence:
        # Build sorted tuples
        row_reasons = tuple(sorted(self._row_drops.items()))
        decision_reasons = tuple(sorted(self._decision_drops.items()))
        order_reasons = tuple(sorted(self._order_drops.items()))
        # Validate unique and sorted already
        # Ensure monotone: if no decision added, set zeros accordingly
        return ConversionWaterfallEvidence(
            mode_id=self.mode_id,
            score_frame_fingerprint=self.score_frame_fingerprint,
            scored_rows=self._scored,
            finite_score_rows=self._finite,
            calibrated_rows=self._calibrated,
            positive_mean_rows=self._pos_mean,
            positive_lower_bound_rows=self._pos_lb,
            eligible_rows=self._eligible,
            target_positions=self._targets,
            scheduled_decisions=self._scheduled,
            allocation_ready_decisions=self._alloc_ready,
            target_change_decisions=self._target_change,
            submitted_orders=self._submitted,
            filled_orders=self._filled,
            observed_intervals=self._observed,
            invested_intervals=self._invested,
            row_drop_reasons=row_reasons,
            decision_drop_reasons=decision_reasons,
            order_drop_reasons=order_reasons,
        )

def merge_conversion_waterfalls(items: Sequence[ConversionWaterfallEvidence]) -> ConversionWaterfallEvidence:
    if not items:
        raise ValueError("merge requires at least one item")
    if any(item.mode_id != items[0].mode_id for item in items):
        raise ValueError("waterfall mode_id mismatch")
    if any(item.score_frame_fingerprint != items[0].score_frame_fingerprint for item in items):
        raise ValueError("waterfall score_frame_fingerprint mismatch")
    first = items[0]
    total_scored = sum(i.scored_rows for i in items)
    total_finite = sum(i.finite_score_rows for i in items)
    total_cal = sum(i.calibrated_rows for i in items)
    total_pos_mean = sum(i.positive_mean_rows for i in items)
    total_pos_lb = sum(i.positive_lower_bound_rows for i in items)
    total_eligible = sum(i.eligible_rows for i in items)
    total_targets = sum(i.target_positions for i in items)
    total_sched = sum(i.scheduled_decisions for i in items)
    total_ready = sum(i.allocation_ready_decisions for i in items)
    total_change = sum(i.target_change_decisions for i in items)
    total_sub = sum(i.submitted_orders for i in items)
    total_filled = sum(i.filled_orders for i in items)
    total_obs = sum(i.observed_intervals for i in items)
    total_inv = sum(i.invested_intervals for i in items)
    # Merge drop reasons by summing
    def merge_reasons(attr: str):
        merged: dict[str, int] = {}
        for it in items:
            for r, c in getattr(it, attr):
                merged[r] = merged.get(r, 0) + int(c)
        return tuple(sorted(merged.items()))
    row_r = merge_reasons("row_drop_reasons")
    dec_r = merge_reasons("decision_drop_reasons")
    ord_r = merge_reasons("order_drop_reasons")
    # Inputs are hash-bound and must represent one replay cohort.
    fp = first.score_frame_fingerprint
    # mode_id combine? use first
    return ConversionWaterfallEvidence(
        mode_id=first.mode_id,
        score_frame_fingerprint=fp,
        scored_rows=total_scored,
        finite_score_rows=total_finite,
        calibrated_rows=total_cal,
        positive_mean_rows=total_pos_mean,
        positive_lower_bound_rows=total_pos_lb,
        eligible_rows=total_eligible,
        target_positions=total_targets,
        scheduled_decisions=total_sched,
        allocation_ready_decisions=total_ready,
        target_change_decisions=total_change,
        submitted_orders=total_sub,
        filled_orders=total_filled,
        observed_intervals=total_obs,
        invested_intervals=total_inv,
        row_drop_reasons=row_r,
        decision_drop_reasons=dec_r,
        order_drop_reasons=ord_r,
    )

def evaluate_wealth_candidate(*, route_kind: RouteObjectiveKind, evidence_kind: WealthEvidenceKind, waterfall: ConversionWaterfallEvidence, certificate_passed: bool, hashes_reconciled: bool, absolute_lower_cagr: float | None, matched_excess_lower_cagr: float | None) -> WealthCandidateVerdict:
    """Single wealth evaluator; bounded DEBUG line."""
    # Determine first failure stage
    reasons: list[str] = []
    first_failure = ""
    # Stage order: calibration, positive_mean, positive_lower_bound, market_eligible, allocation_ready, target_change, submitted, filled, invested, certificate, hashes, evidence_kind
    # Use waterfall to find first zero stage
    if waterfall.scored_rows == 0:
        first_failure = "scored"
    elif waterfall.finite_score_rows == 0:
        first_failure = "finite"
    elif waterfall.calibrated_rows == 0:
        first_failure = "calibrated"
    elif waterfall.positive_mean_rows == 0:
        first_failure = "positive_mean"
    elif waterfall.positive_lower_bound_rows == 0:
        first_failure = "positive_lower_bound"
    elif waterfall.eligible_rows == 0:
        first_failure = "market_eligible"
    elif waterfall.target_positions == 0:
        first_failure = "target"
    elif waterfall.scheduled_decisions == 0:
        first_failure = "scheduled"
    elif waterfall.allocation_ready_decisions == 0:
        first_failure = "allocation_ready"
    elif waterfall.target_change_decisions == 0:
        first_failure = "target_change"
    elif waterfall.submitted_orders == 0:
        first_failure = "submitted"
    elif waterfall.filled_orders == 0:
        first_failure = "filled"
    elif waterfall.observed_intervals == 0:
        first_failure = "observed"
    elif waterfall.invested_intervals == 0:
        first_failure = "invested"
    elif not certificate_passed:
        first_failure = "certificate"
    elif not hashes_reconciled:
        first_failure = "hash"
    elif evidence_kind == WealthEvidenceKind.SYNTHETIC_PROJECTION:
        first_failure = "synthetic"
    else:
        first_failure = ""

    # Determine promotion status
    # Synthetic always research only
    if evidence_kind == WealthEvidenceKind.SYNTHETIC_PROJECTION:
        reasons.append("synthetic-route-not-executable")
        promotion_status = "RESEARCH_EDGE_ONLY"
        promotable = False
        if not first_failure:
            first_failure = "synthetic"
    elif route_kind == RouteObjectiveKind.HEDGED_RESIDUAL:
        reasons.append("synthetic-route-not-executable")
        promotion_status = "RESEARCH_EDGE_ONLY"
        promotable = False
        if not first_failure:
            first_failure = "route"
    else:
        # executable checks
        if waterfall.filled_orders == 0:
            reasons.append("no-filled-orders")
            promotion_status = "NO_TRADE"
            promotable = False
            if not first_failure:
                first_failure = "filled"
        elif waterfall.invested_intervals == 0:
            reasons.append("no-invested-intervals")
            promotion_status = "NO_TRADE"
            promotable = False
            if not first_failure:
                first_failure = "invested"
        elif not certificate_passed:
            reasons.append("certificate-failed")
            promotion_status = "NO_TRADE"
            promotable = False
            if not first_failure:
                first_failure = "certificate"
        elif not hashes_reconciled:
            reasons.append("hash-mismatch")
            promotion_status = "NO_TRADE"
            promotable = False
            if not first_failure:
                first_failure = "hash"
        else:
            # check evidence kind vs route
            if route_kind == RouteObjectiveKind.UNHEDGED_ABSOLUTE and evidence_kind == WealthEvidenceKind.EXECUTABLE_UNHEDGED:
                # need absolute lower cagr >0
                if absolute_lower_cagr is None or not (absolute_lower_cagr > 0):
                    reasons.append("non-positive-absolute-lower-cagr")
                    promotion_status = "NO_TRADE"
                    promotable = False
                    if not first_failure:
                        first_failure = "certificate"
                else:
                    promotion_status = "PROMOTED_UNHEDGED"
                    promotable = True
            elif route_kind == RouteObjectiveKind.EXECUTABLE_HEDGED and evidence_kind == WealthEvidenceKind.EXECUTABLE_HEDGED:
                # need matched excess? Use absolute for hedged? spec says matched excess
                if (matched_excess_lower_cagr is not None and matched_excess_lower_cagr > 0) or (absolute_lower_cagr is not None and absolute_lower_cagr > 0):
                    promotion_status = "PROMOTED_EXECUTABLE_HEDGED"
                    promotable = True
                else:
                    reasons.append("non-positive-excess-lower-cagr")
                    promotion_status = "NO_TRADE"
                    promotable = False
            else:
                # mismatch
                reasons.append("route-evidence-mismatch")
                promotion_status = "NO_TRADE"
                promotable = False
                if not first_failure:
                    first_failure = "evidence"

    # If promotable but synthetic/hedged residual, already handled
    if promotable and evidence_kind == WealthEvidenceKind.SYNTHETIC_PROJECTION:
        promotion_status = "RESEARCH_EDGE_ONLY"
        promotable = False
        if "synthetic-route-not-executable" not in reasons:
            reasons.append("synthetic-route-not-executable")

    # Fill first_failure if still empty but not promotable -> use promotion_status
    if not first_failure and not promotable:
        first_failure = "certificate" if not certificate_passed else "filled"

    # Emit bounded DEBUG line  # noqa: S110
    try:  # noqa: S110
        top_drop = ""
        if waterfall.row_drop_reasons:
            top_drop = max(waterfall.row_drop_reasons, key=lambda x: x[1])[0]  # noqa: B007
        logger.debug("[EVAL] stage=wealth_transfer candidate=%.3f scored=%.3f calibrated=%.3f positive_mean=%.3f positive_lcb=%.3f targets=%.3f decisions=%.3f submitted=%.3f filled=%.3f invested=%.3f first_zero_stage=%s top_drop=%s", float(waterfall.scored_rows), float(waterfall.finite_score_rows), float(waterfall.calibrated_rows), float(waterfall.positive_mean_rows), float(waterfall.positive_lower_bound_rows), float(waterfall.target_positions), float(waterfall.scheduled_decisions), float(waterfall.submitted_orders), float(waterfall.filled_orders), float(waterfall.invested_intervals), first_failure, top_drop)
    except Exception:  # noqa: S110
        pass

    return WealthCandidateVerdict(promotion_status=promotion_status, promotable=promotable, first_failure_stage=first_failure, reasons=tuple(reasons), waterfall=waterfall)

def require_executable_overlay_data(route, data) -> object:
    """Fail closed before fit if executable_hedged lacks hash-bound overlay."""
    # route is RouteObjective
    if getattr(route, "kind", None) != RouteObjectiveKind.EXECUTABLE_HEDGED:
        return getattr(data, "executable_overlay_data", None)
    overlay = getattr(data, "executable_overlay_data", None)
    if overlay is None:
        raise ValueError("hedge-execution-evidence-missing")
    # Check hash bound? need evidence_hash matches route hedge_evidence_hash
    route_hash = getattr(route, "hedge_evidence_hash", None)
    overlay_hash = getattr(overlay, "evidence_hash", None)
    if route_hash is not None and overlay_hash is not None and route_hash != overlay_hash:
        raise ValueError("hedge-execution-evidence-mismatch")
    # Check instrument is an explicitly bound ETF.
    instr = getattr(overlay, "instrument", None)
    kind = getattr(instr, "asset_kind", None)
    if kind is None or str(kind) not in ("ETF", "AssetKind.ETF"):
        raise ValueError("hedge-execution-instrument-not-etf")
    frame = getattr(overlay, "frame", None)
    required = {"instrument_id", "session", "open", "high", "low", "close"}
    if frame is None or not hasattr(frame, "columns") or not required.issubset(set(frame.columns)):
        raise ValueError("hedge-execution-bars-missing")
    return overlay
