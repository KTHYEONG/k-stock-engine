def test_conversion_waterfall_reconciles_rows_decisions_orders_and_intervals() -> None:
    from src.stocks.ml.wealth_transfer import ConversionWaterfallAccumulator
    from src.stocks.trading.portfolio_constructor import AllocationDecisionEvidence

    acc = ConversionWaterfallAccumulator(mode_id="h10:c10:k12:finite", score_frame_fingerprint="a" * 64)
    acc.add_decision(AllocationDecisionEvidence(scored_rows=10, finite_score_rows=8, calibrated_rows=6, positive_mean_rows=4, positive_lower_bound_rows=2, market_eligible_rows=2, selected_target_rows=1, allocation_ready=True, target_changed=True, drop_reasons=(("non_finite_score", 2), ("uncalibrated", 2), ("non_positive_mean", 2), ("non_positive_lower_bound", 2), ("rank_cap", 1))))
    acc.add_orders(2, 1, {"no-session-row": 1})
    acc.add_intervals(5, 4)
    evidence = acc.finalize()
    assert (evidence.scored_rows, evidence.target_positions) == (10, 1)
    assert (evidence.scheduled_decisions, evidence.target_change_decisions) == (1, 1)
    assert (evidence.submitted_orders, evidence.filled_orders) == (2, 1)
    assert (evidence.observed_intervals, evidence.invested_intervals) == (5, 4)
    assert sum(count for _, count in evidence.row_drop_reasons) == 9

def test_conversion_waterfall_rejects_nonconserving_drop_counts() -> None:
    import pytest
    from src.stocks.ml.contracts import ConversionWaterfallEvidence

    with pytest.raises(ValueError, match="row drop counts do not reconcile"):
        ConversionWaterfallEvidence(mode_id="m", score_frame_fingerprint="a" * 64, scored_rows=10, finite_score_rows=8, calibrated_rows=6, positive_mean_rows=4, positive_lower_bound_rows=2, eligible_rows=2, target_positions=1, scheduled_decisions=1, allocation_ready_decisions=1, target_change_decisions=1, submitted_orders=1, filled_orders=1, observed_intervals=2, invested_intervals=1, row_drop_reasons=(("non_finite_score", 2), ("uncalibrated", 2), ("non_positive_mean", 2), ("non_positive_lower_bound", 2)), decision_drop_reasons=(), order_drop_reasons=())

def test_synthetic_excess_is_research_only_even_with_positive_lower_bound() -> None:
    from src.stocks.ml.contracts import ConversionWaterfallEvidence, RouteObjectiveKind
    from src.stocks.ml.wealth_transfer import WealthEvidenceKind, evaluate_wealth_candidate

    waterfall = ConversionWaterfallEvidence(mode_id="m", score_frame_fingerprint="a" * 64, scored_rows=1, finite_score_rows=1, calibrated_rows=1, positive_mean_rows=1, positive_lower_bound_rows=1, eligible_rows=1, target_positions=1, scheduled_decisions=1, allocation_ready_decisions=1, target_change_decisions=1, submitted_orders=1, filled_orders=1, observed_intervals=2, invested_intervals=2, row_drop_reasons=(), decision_drop_reasons=(), order_drop_reasons=())
    verdict = evaluate_wealth_candidate(route_kind=RouteObjectiveKind.HEDGED_RESIDUAL, evidence_kind=WealthEvidenceKind.SYNTHETIC_PROJECTION, waterfall=waterfall, certificate_passed=True, hashes_reconciled=True, absolute_lower_cagr=-0.02, matched_excess_lower_cagr=0.06)
    assert verdict.promotion_status == "RESEARCH_EDGE_ONLY"
    assert verdict.promotable is False
    assert "synthetic-route-not-executable" in verdict.reasons

def test_executable_unhedged_requires_fills_investment_and_certificate() -> None:
    import dataclasses
    from src.stocks.ml.contracts import ConversionWaterfallEvidence, RouteObjectiveKind
    from src.stocks.ml.wealth_transfer import WealthEvidenceKind, evaluate_wealth_candidate

    evidence = ConversionWaterfallEvidence(mode_id="m", score_frame_fingerprint="a" * 64, scored_rows=1, finite_score_rows=1, calibrated_rows=1, positive_mean_rows=1, positive_lower_bound_rows=1, eligible_rows=1, target_positions=1, scheduled_decisions=1, allocation_ready_decisions=1, target_change_decisions=1, submitted_orders=1, filled_orders=1, observed_intervals=2, invested_intervals=2, row_drop_reasons=(), decision_drop_reasons=(), order_drop_reasons=())
    promoted = evaluate_wealth_candidate(route_kind=RouteObjectiveKind.UNHEDGED_ABSOLUTE, evidence_kind=WealthEvidenceKind.EXECUTABLE_UNHEDGED, waterfall=evidence, certificate_passed=True, hashes_reconciled=True, absolute_lower_cagr=0.01, matched_excess_lower_cagr=None)
    assert (promoted.promotion_status, promoted.promotable) == ("PROMOTED_UNHEDGED", True)
    no_fill = dataclasses.replace(evidence, filled_orders=0, order_drop_reasons=(("no-session-row", 1),))
    rejected = evaluate_wealth_candidate(route_kind=RouteObjectiveKind.UNHEDGED_ABSOLUTE, evidence_kind=WealthEvidenceKind.EXECUTABLE_UNHEDGED, waterfall=no_fill, certificate_passed=True, hashes_reconciled=True, absolute_lower_cagr=0.01, matched_excess_lower_cagr=None)
    assert rejected.promotion_status == "NO_TRADE"
    assert rejected.first_failure_stage == "filled"

def test_executable_hedged_route_fails_before_fit_without_hash_bound_overlay() -> None:
    import pytest
    from src.stocks.ml.contracts import RouteObjective, RouteObjectiveKind
    from src.stocks.ml.wealth_transfer import require_executable_overlay_data

    route = RouteObjective(kind=RouteObjectiveKind.EXECUTABLE_HEDGED, hedge_instrument="KRX:INVERSE_ETF", hedge_evidence_hash="b" * 64)
    data = type("Data", (), {"executable_overlay_data": None})()
    with pytest.raises(ValueError, match="hedge-execution-evidence-missing"):
        require_executable_overlay_data(route, data)
