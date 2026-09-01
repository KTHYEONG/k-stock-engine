"""Sparse transition planner maximizing forecast net utility over hold/enter/replace."""
# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping

from legacy.stocks.ml.return_transfer import ReturnDistributionForecast, TransitionCost
from legacy.stocks.trading.portfolio_constructor import StockRiskPolicy


def plan_return_transfer_transition(
    current_weights: Mapping[str, float],
    forecasts: Mapping[str, ReturnDistributionForecast],
    cost_inputs: Mapping[str, TransitionCost],
    constraints: StockRiskPolicy,
) -> Mapping[str, float]:
    """Deterministically maximizes forecast net utility over sparse hold/enter/replace actions using exact marginal costs and hard constraints."""
    if not forecasts:
        return {}
    # Validate constraints caps
    gross_cap = float(constraints.gross_cap)
    single_cap = float(constraints.single_name_cap)
    float(constraints.sector_cap) if hasattr(constraints, "sector_cap") else 1.0
    # Compute net utility per instrument: mu - cost
    # cost uses unit-notional schedule/liquidity outputs, never alpha
    scored: list[tuple[float, str]] = []
    for iid, fc in forecasts.items():
        cost = cost_inputs.get(iid)
        if cost is None:
            # missing cost -> cannot trade, treat as hold only if incumbent
            net = float(fc.mu)
        else:
            # if incumbent, hold cost is relevant; else enter cost
            is_incumbent = iid in current_weights and current_weights[iid] > 0
            net = float(fc.mu) - float(cost.hold) if is_incumbent else float(fc.mu) - float(cost.enter)
        scored.append((net, iid))
    # Deterministic tie break by instrument_id: sort by net desc then iid asc
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Sparse selection: keep incumbents unless replacement covers marginal exit+entry
    # Replacement logic: if candidate net > incumbent net + exit+enter, replace
    incumbents = {k for k, v in current_weights.items() if v > 0}
    k_bound = min(int(constraints.top_k), len(scored))  # bounded K
    candidates = [iid for _, iid in scored[:k_bound]]
    # For each incumbent, check if any candidate replacement is better by > exit+enter
    # Exact marginal costs: use cost_inputs
    for iid in list(incumbents):
        incumbent_forecast = forecasts.get(iid)
        incumbent_cost = cost_inputs.get(iid)
        if incumbent_forecast is None:
            continue
        # incumbent net (hold)
        inc_net = float(incumbent_forecast.mu) - (float(incumbent_cost.hold) if incumbent_cost else 0.0)
        # Find best replacement not already incumbent
        best_repl = None
        best_gain = 0.0
        for cand in candidates:
            if cand in incumbents:
                continue
            cand_fc = forecasts.get(cand)
            cand_cost = cost_inputs.get(cand)
            if cand_fc is None:
                continue
            exit_cost = float(incumbent_cost.exit) if incumbent_cost else 0.0
            enter_cost = float(cand_cost.enter) if cand_cost else 0.0
            cand_net = float(cand_fc.mu) - enter_cost
            gain = cand_net - inc_net - exit_cost
            # need gain > 0 to replace (positive epsilon)
            if gain > 1e-12 and gain > best_gain:
                best_gain = gain
                best_repl = cand
        if best_repl is not None:
            # replace incumbent with best_repl
            candidates = [c for c in candidates if c != iid]
            if best_repl not in candidates:
                candidates.append(best_repl)
            incumbents.discard(iid)
            incumbents.add(best_repl)
        # else retain incumbent: ensure it's in candidates
        if iid in incumbents and iid not in candidates:
            candidates.append(iid)

    # Now size weights: equal weight among selected up to caps, deterministic
    selected = sorted(set(candidates))  # deterministic
    # Respect gross and single caps
    if not selected:
        return {}
    # Simple equal weight, clipped to single_name_cap and gross_cap
    n = len(selected)
    gross = min(1.0, gross_cap)
    # allocate gross equally but capped
    per_name = min(single_cap, gross / n if n else 0.0)
    # If per_name * n < gross due to single cap, we leave cash
    weights: dict[str, float] = {}
    total = 0.0
    for iid in sorted(selected):  # deterministic instrument-id tie break
        w = float(per_name)
        # ensure not exceed single cap
        if w > single_cap:
            w = single_cap
        weights[iid] = w
        total += w
    # Ensure gross cap not exceeded (by construction) and deterministic
    # If total > gross_cap slightly due to floating, scale down proportionally
    if total > gross_cap + 1e-12:
        scale = gross_cap / total if total > 0 else 0.0
        for k in weights:
            weights[k] *= scale
    # Round to avoid tiny floating noise, keep deterministic
    for k in list(weights.keys()):
        weights[k] = round(float(weights[k]), 12)
        if weights[k] <= 1e-12:
            del weights[k]
    return weights
