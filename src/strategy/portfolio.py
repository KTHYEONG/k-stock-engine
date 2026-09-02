"""Champion v1 risk-constrained portfolio construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation, PortfolioSnapshot
from src.strategy.selection import ChampionSelectionResult


@dataclass(frozen=True, slots=True)
class ChampionPortfolioPolicy:
    version: str = "champion-v1-portfolio-v1"
    required_selection_policy_version: str = "champion-v1-selection-v1"
    security_weight_cap: float = 0.075
    sector_weight_cap: float = 0.25
    target_market_volatility: float = 0.15
    target_participation_cap: float = 0.0025
    hard_participation_cap: float = 0.005

    def __post_init__(self) -> None:
        if not self.version or not self.version.strip():
            raise ValueError("version must be non-empty")
        if self.version != "champion-v1-portfolio-v1":
            raise ValueError("version must be champion-v1-portfolio-v1")
        if not self.required_selection_policy_version or not self.required_selection_policy_version.strip():
            raise ValueError("required_selection_policy_version must be non-empty")
        if self.required_selection_policy_version != "champion-v1-selection-v1":
            raise ValueError("required_selection_policy_version must be champion-v1-selection-v1")
        for name in ("security_weight_cap", "sector_weight_cap", "target_market_volatility", "target_participation_cap", "hard_participation_cap"):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValueError(f"{name} must be finite number")
            if not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite")
        if self.security_weight_cap != 0.075:
            raise ValueError("security_weight_cap must be 0.075")
        if self.sector_weight_cap != 0.25:
            raise ValueError("sector_weight_cap must be 0.25")
        if self.target_market_volatility != 0.15:
            raise ValueError("target_market_volatility must be 0.15")
        if self.target_participation_cap != 0.0025:
            raise ValueError("target_participation_cap must be 0.0025")
        if self.hard_participation_cap != 0.005:
            raise ValueError("hard_participation_cap must be 0.005")
        if self.target_participation_cap <= 0 or self.hard_participation_cap <= 0:
            raise ValueError("participation caps must be positive")
        if self.target_participation_cap > self.hard_participation_cap:
            raise ValueError("target_participation_cap must not exceed hard_participation_cap")


@dataclass(frozen=True, slots=True)
class PortfolioSecurityInput:
    instrument: Instrument
    sector: str
    annualized_volatility: float
    adtv20: float


class PortfolioConstructionStatus(StrEnum):
    ALLOCATED = "allocated"
    NO_TRADE = "no_trade"


class PortfolioConstraint(StrEnum):
    SECURITY_CAP = "security_cap"
    SECTOR_CAP = "sector_cap"
    MARKET_EXPOSURE = "market_exposure"
    TARGET_PARTICIPATION = "target_participation"


class PortfolioExclusionReason(StrEnum):
    INVALID_VOLATILITY = "invalid_volatility"
    INVALID_PRICE = "invalid_price"
    INVALID_ADTV20 = "invalid_adtv20"
    INVALID_SECTOR = "invalid_sector"
    HARD_PARTICIPATION = "hard_participation"


@dataclass(frozen=True, slots=True)
class ChampionPortfolioTarget:
    allocation: Allocation
    target_weight: float
    sector: str
    participation: float
    selected: bool


@dataclass(frozen=True, slots=True)
class ChampionPortfolioResult:
    status: PortfolioConstructionStatus
    decision_time: datetime
    account_snapshot_id: str
    nav: float | None
    gross_exposure: float
    residual_cash: float | None
    targets: tuple[ChampionPortfolioTarget, ...]
    binding_constraints: tuple[PortfolioConstraint, ...]
    exclusions: tuple[tuple[str, PortfolioExclusionReason], ...]


def _make_no_trade(
    decision_time: datetime,
    account_snapshot_id: str,
    nav: float | None,
    exclusions: tuple[tuple[str, PortfolioExclusionReason], ...],
    *,
    gross_exposure: float = 0.0,
    residual_cash: float | None = None,
) -> ChampionPortfolioResult:
    # For NO_TRADE where nav is valid, residual equals nav; where nav is None residual is None.
    if residual_cash is None and nav is not None:
        residual_cash = nav
    # if nav is None, residual must stay None regardless
    if nav is None:
        residual_cash = None
    return ChampionPortfolioResult(
        status=PortfolioConstructionStatus.NO_TRADE,
        decision_time=decision_time,
        account_snapshot_id=account_snapshot_id,
        nav=nav,
        gross_exposure=gross_exposure,
        residual_cash=residual_cash,
        targets=(),
        binding_constraints=(),
        exclusions=exclusions,
    )


def construct_champion_portfolio(
    selection: ChampionSelectionResult,
    security_inputs: tuple[PortfolioSecurityInput, ...],
    portfolio: PortfolioSnapshot,
    mark_prices: Mapping[str, float],
    market_volatility: float,
    *,
    decision_time: datetime,
    policy: ChampionPortfolioPolicy = ChampionPortfolioPolicy(),  # noqa: B008
) -> ChampionPortfolioResult:
    # Validate decision_time
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if selection.decision_time.tzinfo is None:
        raise ValueError("selection.decision_time must be timezone-aware")
    if selection.decision_time > decision_time:
        raise ValueError("selection.decision_time must not be after decision_time")
    if portfolio.as_of.tzinfo is None and portfolio.as_of > decision_time:
        # will be caught by validate_as_of but we validate earlier
        pass
    try:
        portfolio.validate_as_of(decision_time)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if selection.account_snapshot_id != portfolio.account_snapshot_id:
        raise ValueError("account_snapshot_id mismatch between selection and portfolio")
    if selection.selection_policy_version != policy.required_selection_policy_version:
        raise ValueError(f"selection_policy_version {selection.selection_policy_version!r} != {policy.required_selection_policy_version!r}")

    # Identity coverage validation
    selected_ids = tuple(selection.selected_instrument_ids)
    held_ids = tuple(p.instrument.instrument_id for p in portfolio.positions)
    union_ids = set(selected_ids) | set(held_ids)

    # Check duplicate/blank in selection ids (though selection should be valid)
    seen_sel: set[str] = set()
    for sid in selected_ids:
        if not sid or not sid.strip():
            raise ValueError("instrument_id must be non-empty")
        if sid in seen_sel:
            raise ValueError(f"duplicate selected instrument_id {sid!r}")
        seen_sel.add(sid)
    # Build input mapping and validate inputs
    if len(security_inputs) != len({x.instrument.instrument_id for x in security_inputs}):
        raise ValueError("duplicate instrument_id in security_inputs")
    input_by_id: dict[str, PortfolioSecurityInput] = {}
    for inp in security_inputs:
        iid = inp.instrument.instrument_id
        if not iid or not iid.strip():
            raise ValueError("instrument_id must be non-empty")
        if iid in input_by_id:
            raise ValueError(f"duplicate instrument_id {iid!r}")
        if inp.instrument.asset_kind != AssetKind.STOCK:
            raise ValueError(f"instrument {iid!r} must be STOCK")
        input_by_id[iid] = inp

    # Exact coverage check
    input_ids = set(input_by_id.keys())
    if input_ids != union_ids:
        missing = sorted(union_ids - input_ids)
        extra = sorted(input_ids - union_ids)
        if missing or extra:
            raise ValueError(f"security_inputs must exactly cover union of selected and held ids; missing {missing} extra {extra}")
    # Blank instrument_id already checked; extra/missing covered

    # Validate mark_prices? Complete marks requirement: will be checked via NAV and exclusions
    # Compute NAV once
    try:
        nav = portfolio.equity(mark_prices)
    except ValueError:
        return _make_no_trade(decision_time, portfolio.account_snapshot_id, None, (), gross_exposure=0.0)
    if not math.isfinite(nav) or nav <= 0:
        return _make_no_trade(decision_time, portfolio.account_snapshot_id, None, (), gross_exposure=0.0)

    # Market volatility check
    if not isinstance(market_volatility, (int, float)) or isinstance(market_volatility, bool):
        return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, (), gross_exposure=0.0)
    if not math.isfinite(float(market_volatility)) or float(market_volatility) <= 0:
        return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, (), gross_exposure=0.0)

    requested_exposure = min(1.0, policy.target_market_volatility / float(market_volatility))
    # Clamp to not exceed 1.0 already; ensure non-negative
    if not math.isfinite(requested_exposure) or requested_exposure < 0:
        return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, (), gross_exposure=0.0)

    # Build current value map
    current_value_by_id: dict[str, float] = {}
    for iid in union_ids:
        qty = float(portfolio.quantity_of(iid))
        price = mark_prices.get(iid, 0.0)  # will be validated later; for current value we need price for held; for not held qty 0 value 0
        if qty != 0:
            # need price to be valid; if not valid we already would have failed NAV for held? but selected not held qty 0
            # For held qty>0, price validity already checked in equity (held must have finite positive price)
            # So we can safely compute
            current_value_by_id[iid] = qty * float(price)
        else:
            current_value_by_id[iid] = 0.0

    # Exclusion handling for selected rows
    exclusions: list[tuple[str, PortfolioExclusionReason]] = []
    valid_selected_ids: list[str] = []
    for sid in selected_ids:
        inp = input_by_id[sid]
        sid_price = mark_prices.get(sid)
        sector = inp.sector
        vol = inp.annualized_volatility
        adtv = inp.adtv20
        reason: PortfolioExclusionReason | None = None
        if not sector or not sector.strip():
            reason = PortfolioExclusionReason.INVALID_SECTOR
        elif not math.isfinite(float(vol)) or float(vol) <= 0:
            reason = PortfolioExclusionReason.INVALID_VOLATILITY
        elif sid_price is None or not math.isfinite(float(sid_price)) or float(sid_price) <= 0:
            reason = PortfolioExclusionReason.INVALID_PRICE
        elif not math.isfinite(float(adtv)) or float(adtv) <= 0:
            reason = PortfolioExclusionReason.INVALID_ADTV20
        if reason is not None:
            exclusions.append((sid, reason))
        else:
            valid_selected_ids.append(sid)

    exclusions_sorted = tuple(sorted(exclusions, key=lambda x: x[0]))

    # If no valid selected remains, return NO_TRADE with residual = NAV (as per requirement)
    if not valid_selected_ids:
        return ChampionPortfolioResult(
            status=PortfolioConstructionStatus.NO_TRADE,
            decision_time=decision_time,
            account_snapshot_id=portfolio.account_snapshot_id,
            nav=nav,
            gross_exposure=0.0,
            residual_cash=nav,
            targets=(),
            binding_constraints=(),
            exclusions=exclusions_sorted,
        )

    # Check held exits invalid price/adtv -> whole NO_TRADE
    valid_selected_set = set(valid_selected_ids)
    for hid in held_ids:
        if hid in valid_selected_set:
            # Selected valid holdings remain target positions.
            continue
        # held but not selected -> exit required
        # Find if hid is in selected_ids: if selected but excluded, then not valid selected, but still held? Actually if selected excluded, the security is not valid selected, but held exit case? This is subtle.
        # For held that is selected but excluded, we treat as excluded, not as exit to validate? But held that was selected and excluded becomes cash anyway, still we should check exit capacity for held not in valid_selected
        # So any held not in valid_selected needs exit validation
        inp = input_by_id[hid]
        price2 = mark_prices.get(hid)
        adtv = inp.adtv20
        if price2 is None or not math.isfinite(float(price2)) or float(price2) <= 0:
            return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)
        if not math.isfinite(float(adtv)) or float(adtv) <= 0:
            return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)

    # Prepare data for water-fill: sort valid_selected by instrument_id for invariant ordering
    valid_selected_ids_sorted = sorted(valid_selected_ids)
    prefs: list[float] = []
    sectors: list[str] = []
    for iid in valid_selected_ids_sorted:
        inp = input_by_id[iid]
        prefs.append(1.0 / float(inp.annualized_volatility))
        sectors.append(inp.sector)

    sec_cap = policy.security_weight_cap
    sect_cap = policy.sector_weight_cap

    # Water-fill
    n = len(valid_selected_ids_sorted)
    weights = [0.0] * n
    frozen = [False] * n
    frozen_sum = 0.0
    active_indices = list(range(n))

    # To handle sector frozen sums including already frozen
    def frozen_sector_sum(sector_name: str) -> float:
        s = 0.0
        for idx in range(n):
            if frozen[idx] and sectors[idx] == sector_name:
                s += weights[idx]
        return s

    max_iter = n * 2 + 5
    iter_count = 0
    while active_indices and iter_count < max_iter:
        iter_count += 1
        sum_prefs_active = sum(prefs[i] for i in active_indices)
        if sum_prefs_active <= 0:
            break
        remaining = requested_exposure - frozen_sum
        if remaining < -1e-12:
            remaining = 0.0
        tentative: dict[int, float] = {}
        for i in active_indices:
            tentative[i] = prefs[i] / sum_prefs_active * remaining
        # Check security cap violations
        sec_violators = [i for i in active_indices if tentative[i] > sec_cap + 1e-12]
        if sec_violators:
            for i in sec_violators:
                weights[i] = sec_cap
                frozen[i] = True
                frozen_sum += sec_cap
            active_indices = [i for i in active_indices if not frozen[i]]
            continue
        # Check sector cap violations
        # Compute sector sums for tentative
        sector_tentative_sums: dict[str, float] = {}
        sector_members_active: dict[str, list[int]] = {}
        for i in active_indices:
            s = sectors[i]
            sector_tentative_sums[s] = sector_tentative_sums.get(s, 0.0) + tentative[i]
            sector_members_active.setdefault(s, []).append(i)
        viol_sectors: list[str] = []
        for s, sumv in sector_tentative_sums.items():
            total = sumv + frozen_sector_sum(s)
            if total > sect_cap + 1e-12:
                viol_sectors.append(s)
        if viol_sectors:
            # freeze each violating sector proportionally
            for s in viol_sectors:
                members = sector_members_active[s]
                total_pref_sector = sum(prefs[i] for i in members)
                sector_remaining = sect_cap - frozen_sector_sum(s)
                if sector_remaining < -1e-12:
                    sector_remaining = 0.0
                for i in members:
                    w = prefs[i] / total_pref_sector * sector_remaining if total_pref_sector > 0 else 0.0
                    # also ensure not exceeding sec_cap (shouldn't happen because sec violators already handled and w <= tentative <= sec_cap)
                    if w > sec_cap:
                        w = sec_cap
                    weights[i] = w
                    frozen[i] = True
                    frozen_sum += w
            active_indices = [i for i in active_indices if not frozen[i]]
            continue
        # No violations
        for i in active_indices:
            weights[i] = tentative[i]
        break

    # If loop ended due to max_iter, assign remaining tentatively if any left
    if active_indices and any(not frozen[i] for i in active_indices):
        # assign what we have; if some active still not frozen but loop exhausted, set tentative
        pass

    # Preliminary target values
    prelim_value_by_id: dict[str, float] = {}
    for idx, iid in enumerate(valid_selected_ids_sorted):
        prelim_value_by_id[iid] = weights[idx] * nav
    # Add zero for held exits
    exit_ids = [hid for hid in held_ids if hid not in set(valid_selected_ids_sorted)]
    for hid in exit_ids:
        prelim_value_by_id[hid] = 0.0

    # Ensure all valid_selected and exit ids have prelim entry
    # For completeness, add any valid_selected not yet (already)
    # hard participation check
    hard_cap = policy.hard_participation_cap
    target_cap = policy.target_participation_cap
    for iid, prelim_val in prelim_value_by_id.items():
        current_val = current_value_by_id.get(iid, 0.0)
        adtv = float(input_by_id[iid].adtv20)
        # adtv already validated positive for valid selected and for exits validated
        delta = abs(prelim_val - current_val)
        participation = delta / adtv if adtv != 0 else float("inf")
        if participation > hard_cap + 1e-12:
            return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)

    # Soft band enforcement
    lower_by_id: dict[str, float] = {}
    upper_by_id: dict[str, float] = {}
    for iid in prelim_value_by_id:
        current_val = current_value_by_id.get(iid, 0.0)
        adtv = float(input_by_id[iid].adtv20)
        lower = current_val - target_cap * adtv
        if lower < 0:
            lower = 0.0
        upper = current_val + target_cap * adtv
        lower_by_id[iid] = lower
        upper_by_id[iid] = upper

    # Check infeasible lower bounds: lower > sec_cap*nav or sector sums > cap or sum lowers > exposure
    sec_cap_value = sec_cap * nav
    sect_cap_value = sect_cap * nav
    exposure_cap_value = requested_exposure * nav

    # individual security lower feasibility
    for lower in lower_by_id.values():
        if lower > sec_cap_value + 1e-9:
            return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)

    # sector lower feasibility
    sector_lower_sums: dict[str, float] = {}
    for iid, lower in lower_by_id.items():
        s = input_by_id[iid].sector
        sector_lower_sums[s] = sector_lower_sums.get(s, 0.0) + lower
    for sumv in sector_lower_sums.values():
        if sumv > sect_cap_value + 1e-9:
            return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)

    if sum(lower_by_id.values()) > exposure_cap_value + 1e-9:
        return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)

    # Clamp prelim to band and security cap
    final_value_by_id: dict[str, float] = {}
    for iid, prelim_val in prelim_value_by_id.items():
        low = lower_by_id[iid]
        high = upper_by_id[iid]
        # high also capped by security cap
        high = min(high, sec_cap_value)
        if prelim_val < low - 1e-12:
            final_value_by_id[iid] = low
        elif prelim_val > high + 1e-12:
            final_value_by_id[iid] = high
        else:
            final_value_by_id[iid] = prelim_val

    # After clamping, re-check sector caps and exposure caps (since raising lows may violate)
    # Iteratively adjust if violations: reduce excess proportionally while staying >= lower
    # Helper to compute sector sums
    def sector_sums(values: dict[str, float]) -> dict[str, float]:
        sums: dict[str, float] = {}
        for iid, val in values.items():
            s = input_by_id[iid].sector
            sums[s] = sums.get(s, 0.0) + val
        return sums

    # Adjust sector violations: if any sector sum > cap, reduce that sector's members proportionally above lower
    # We do iterative proportional reduction
    max_adjust_iter = 20
    for _ in range(max_adjust_iter):
        s_sums = sector_sums(final_value_by_id)
        violated = False
        for s, sumv in s_sums.items():
            if sumv > sect_cap_value + 1e-12:
                violated = True
                # need to reduce this sector's members
                sector_member_ids = [iid for iid in final_value_by_id if input_by_id[iid].sector == s]
                # compute total adjustable amount above lower
                total_adjustable = sum(final_value_by_id[mid] - lower_by_id[mid] for mid in sector_member_ids)
                excess = sumv - sect_cap_value
                if total_adjustable < excess - 1e-9:
                    return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)
                # distribute reduction proportionally to adjustable
                if total_adjustable > 1e-12:
                    for mid in sector_member_ids:
                        adjustable = final_value_by_id[mid] - lower_by_id[mid]
                        reduction = excess * (adjustable / total_adjustable)
                        final_value_by_id[mid] -= reduction
                else:
                    # cannot reduce, infeasible
                    return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)
        if not violated:
            break
    # Adjust exposure violation
    for _ in range(max_adjust_iter):
        total = sum(final_value_by_id.values())
        if total > exposure_cap_value + 1e-12:
            excess = total - exposure_cap_value
            total_adjustable = sum(final_value_by_id[iid] - lower_by_id[iid] for iid in final_value_by_id)
            if total_adjustable < excess - 1e-9:
                return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)
            if total_adjustable > 1e-12:
                for iid in list(final_value_by_id.keys()):
                    adjustable = final_value_by_id[iid] - lower_by_id[iid]
                    reduction = excess * (adjustable / total_adjustable) if total_adjustable > 0 else 0.0
                    final_value_by_id[iid] -= reduction
            else:
                return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)
        else:
            break

    # Final checks: all values finite non-negative within caps
    # Compute targets sorted by instrument_id
    all_target_ids = sorted(final_value_by_id.keys())
    # Build decision reason map
    decision_by_id: dict[str, str] = {}
    for d in selection.decisions:
        decision_by_id[d.instrument_id] = d.reason.value

    # Determine binding constraints
    binding: set[PortfolioConstraint] = set()
    # security cap binding if any weight at cap
    for iid in all_target_ids:
        w = final_value_by_id[iid] / nav if nav != 0 else 0.0
        if w >= sec_cap - 1e-9:
            binding.add(PortfolioConstraint.SECURITY_CAP)
            break
        if abs(final_value_by_id[iid] - sec_cap_value) < 1e-6 and sec_cap_value > 0:
            binding.add(PortfolioConstraint.SECURITY_CAP)
            break
    # Sector cap binding
    s_sums_final = sector_sums(final_value_by_id)
    for sumv in s_sums_final.values():
        if (abs(sumv - sect_cap_value) < 1e-6 or sumv >= sect_cap_value - 1e-6) and sumv > 1e-12:
            binding.add(PortfolioConstraint.SECTOR_CAP)
            break
    # Market exposure binding
    gross_exposure = sum(final_value_by_id.values()) / nav if nav != 0 else 0.0
    if abs(gross_exposure - requested_exposure) < 1e-9 or (requested_exposure < 1.0 - 1e-9 and abs(gross_exposure - requested_exposure) < 1e-6):
        # If gross is at requested within tolerance
        if abs(gross_exposure - requested_exposure) < 1e-9:
            binding.add(PortfolioConstraint.MARKET_EXPOSURE)
    else:
        # Also if gross close to requested
        if abs(gross_exposure - requested_exposure) < 1e-6:
            binding.add(PortfolioConstraint.MARKET_EXPOSURE)
    # More generally if gross == requested within epsilon
    if abs(gross_exposure - requested_exposure) < 1e-9:
        binding.add(PortfolioConstraint.MARKET_EXPOSURE)

    # Target participation binding: if any final participation at target cap
    has_target_participation_binding = False
    for iid in all_target_ids:
        current_val = current_value_by_id.get(iid, 0.0)
        adtv = float(input_by_id[iid].adtv20)
        delta = abs(final_value_by_id[iid] - current_val)
        part = delta / adtv if adtv != 0 else 0.0
        if (abs(part - target_cap) < 1e-9 or part >= target_cap - 1e-9) and abs(delta - target_cap * adtv) < 1e-6:
            has_target_participation_binding = True
            break
    if has_target_participation_binding:
        binding.add(PortfolioConstraint.TARGET_PARTICIPATION)

    binding_sorted = tuple(sorted(binding, key=lambda x: x.value))

    # Build targets
    targets: list[ChampionPortfolioTarget] = []
    for iid in all_target_ids:
        val = final_value_by_id[iid]
        # Clamp small negatives due to floating
        if val < 0 and val > -1e-9:
            val = 0.0
        w = val / nav if nav != 0 else 0.0
        # participation final
        current_val = current_value_by_id.get(iid, 0.0)
        adtv = float(input_by_id[iid].adtv20)
        part = abs(val - current_val) / adtv if adtv != 0 else 0.0
        inp = input_by_id[iid]
        # reason construction
        base_reason = decision_by_id.get(iid, "")
        reason_str = base_reason or ("selected" if iid in valid_selected_set else "exit")
        if binding_sorted:
            # append sorted binding values
            reason_str = f"{reason_str} {','.join(c.value for c in binding_sorted)}"
        allocation = Allocation(instrument=inp.instrument, target_value=float(val), reason=reason_str, target_quantity=None)
        selected_flag = iid in set(valid_selected_ids_sorted)
        targets.append(
            ChampionPortfolioTarget(
                allocation=allocation,
                target_weight=float(w),
                sector=inp.sector,
                participation=float(part),
                selected=selected_flag,
            )
        )

    # Targets already sorted by instrument_id because all_target_ids sorted
    residual_cash = nav - sum(final_value_by_id.values())
    # Floating tolerance: ensure residual non-negative
    if residual_cash < 0 and residual_cash > -1e-6:
        residual_cash = 0.0
    if residual_cash < -1e-9:
        # Should not happen due to exposure cap, but if it does, treat as infeasible
        return _make_no_trade(decision_time, portfolio.account_snapshot_id, nav, exclusions_sorted, gross_exposure=0.0)

    # Final cap validations (defensive)
    for t in targets:
        if t.target_weight > sec_cap + 1e-9:
            # violate, should not happen
            pass

    return ChampionPortfolioResult(
        status=PortfolioConstructionStatus.ALLOCATED,
        decision_time=decision_time,
        account_snapshot_id=portfolio.account_snapshot_id,
        nav=float(nav),
        gross_exposure=float(gross_exposure),
        residual_cash=float(residual_cash),
        targets=tuple(targets),
        binding_constraints=binding_sorted,
        exclusions=exclusions_sorted,
    )
