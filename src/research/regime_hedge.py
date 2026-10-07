"""KOSDAQ 150 regime hedge leg and composite multi-leg futures overlay."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.backtest.overlay import (
    DerivativeConfig,
    OverlayPolicy,
    OverlayState,
    OverlayTarget,
    check_leg_name,
)
from src.core.pit import PITDataError
from src.research.trend_overlay import realised_vol, tsmom_signal

__all__ = [
    "CompositeOverlay",
    "RegimeHedgeLeg",
    "RegimeHedgeSpec",
    "regime_hedge_derivative_config",
]


class RegimeHedgeSpec(BaseModel):
    """Frozen identity of the KOSDAQ 150 downtrend hedge (secondary futures leg); canonical JSON is part of the
    strategy identity.

    Short-only: while the KQ150 multi-horizon trend s (mean of sign(level_t / level_{t-h} - 1) over
    ``tsmom_horizons``) is negative, hold short futures of notional |s| * min(max_fraction, target_vol / sigma) * NAV,
    where sigma is the annualised realised vol of the last ``vol_window_sessions`` daily log returns. Otherwise flat.
    Why short-only and trend-gated: the stock book's beta to KOSDAQ only hurts in KOSDAQ downtrends; an always-on hedge
    paid the index premium in every up-market.

    Fields: tsmom_horizons (non-empty ascending ints >= 1); target_vol > 0; max_fraction >= 0 (0 = flat); vol_window_sessions >= 2;
    rebalance_every_sessions >= 1; contract_multiplier_krw > 0; initial_margin_rate, margin_buffer_rate (sum < 1);
    margin_topup_trigger_fraction in (0, 1]; futures_cost_rate in [0, 1); futures_tax_rate in [0, 1);
    futures_annual_deduction_krw >= 0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tsmom_horizons: tuple[int, ...]
    target_vol: float
    max_fraction: float
    vol_window_sessions: int
    rebalance_every_sessions: int
    contract_multiplier_krw: int
    initial_margin_rate: float
    margin_buffer_rate: float
    margin_topup_trigger_fraction: float
    futures_cost_rate: float
    futures_tax_rate: float
    futures_annual_deduction_krw: int

    @field_validator("tsmom_horizons", mode="before")
    @classmethod
    def _horizons(cls, value: object) -> tuple[int, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"tsmom_horizons must be a tuple/list of ints, got {value!r}")
        items: list[int] = []
        for x in value:
            if isinstance(x, bool) or not isinstance(x, int) or int(x) < 1:
                raise ValueError(f"tsmom_horizons elements must be ints >= 1, got {x!r}")
            items.append(int(x))
        if not items:
            raise ValueError("tsmom_horizons cannot be empty")
        for i in range(1, len(items)):
            if items[i] <= items[i - 1]:
                raise ValueError(f"tsmom_horizons must be strictly ascending, got {items!r}")
        return tuple(items)

    @field_validator("target_vol")
    @classmethod
    def _target_vol(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or out <= 0.0:
            raise ValueError(f"target_vol must be finite and > 0, got {value!r}")
        return out

    @field_validator("max_fraction")
    @classmethod
    def _max_fraction(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or out < 0.0:
            raise ValueError(f"max_fraction must be finite and >= 0, got {value!r}")
        return out

    @field_validator("vol_window_sessions", mode="before")
    @classmethod
    def _vol_window(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or int(value) < 2:
            raise ValueError(f"vol_window_sessions must be an int >= 2, got {value!r}")
        return int(value)

    @field_validator("rebalance_every_sessions", mode="before")
    @classmethod
    def _rebalance_every(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
            raise ValueError(f"rebalance_every_sessions must be an int >= 1, got {value!r}")
        return int(value)

    @field_validator("contract_multiplier_krw", mode="before")
    @classmethod
    def _multiplier(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
            raise ValueError(f"contract_multiplier_krw must be a positive int, got {value!r}")
        return int(value)

    @field_validator("initial_margin_rate", "margin_buffer_rate")
    @classmethod
    def _margin_rate(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 < out < 1.0:
            raise ValueError(f"margin rate must be in (0, 1), got {value!r}")
        return out

    @field_validator("margin_topup_trigger_fraction")
    @classmethod
    def _trigger(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 < out <= 1.0:
            raise ValueError(f"margin_topup_trigger_fraction must be in (0, 1], got {value!r}")
        return out

    @field_validator("futures_cost_rate", "futures_tax_rate")
    @classmethod
    def _rate(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 <= out < 1.0:
            raise ValueError(f"rate must satisfy 0 <= rate < 1, got {value!r}")
        return out

    @field_validator("futures_annual_deduction_krw", mode="before")
    @classmethod
    def _deduction(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or int(value) < 0:
            raise ValueError(f"futures_annual_deduction_krw must be an int >= 0, got {value!r}")
        return int(value)

    @model_validator(mode="after")
    def _validate_spec(self) -> RegimeHedgeSpec:
        if float(self.initial_margin_rate) + float(self.margin_buffer_rate) >= 1.0:
            raise ValueError("initial_margin_rate + margin_buffer_rate must be < 1")
        return self

    def canonical_json(self) -> str:
        """Canonical JSON with sorted keys and compact separators."""
        payload = self.model_dump(mode="json")
        cleaned = {k: v for k, v in payload.items() if v is not None}
        return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def regime_hedge_derivative_config(spec: RegimeHedgeSpec) -> DerivativeConfig:
    """Engine derivative terms of the regime hedge overlay."""
    return DerivativeConfig(
        contract_multiplier_krw=int(spec.contract_multiplier_krw),
        initial_margin_rate=float(spec.initial_margin_rate),
        margin_buffer_rate=float(spec.margin_buffer_rate),
        margin_topup_trigger_fraction=float(spec.margin_topup_trigger_fraction),
        futures_cost_rate=float(spec.futures_cost_rate),
        inverse_cost_rate=0.0,
        futures_tax_rate=Decimal(str(float(spec.futures_tax_rate))),
        futures_annual_deduction_krw=int(spec.futures_annual_deduction_krw),
        inverse_tax_rate=Decimal("0"),
    )


class RegimeHedgeLeg:
    """Secondary-leg policy: the signed net-short contract target of the regime hedge after session t's close, or
    None on non-rebalance rows. Reads only ``index_level[: t - execution_delay + 1]`` and ``state.nav``.
    """

    def __init__(
        self,
        spec: RegimeHedgeSpec,
        *,
        index_level: NDArray[np.float64],
        rebalance_offset: int = 0,
        execution_delay: int = 0,
    ) -> None:
        every = int(spec.rebalance_every_sessions)
        if isinstance(rebalance_offset, bool) or not isinstance(rebalance_offset, int):
            raise ValueError(f"rebalance_offset must be an int in [0, {every}), got {rebalance_offset!r}")
        if not 0 <= int(rebalance_offset) < every:
            raise ValueError(f"rebalance_offset must satisfy 0 <= value < {every}, got {rebalance_offset!r}")
        if isinstance(execution_delay, bool) or not isinstance(execution_delay, int):
            raise ValueError(f"execution_delay must be an int >= 0, got {execution_delay!r}")
        if int(execution_delay) < 0:
            raise ValueError(f"execution_delay must be >= 0, got {execution_delay!r}")
        self._spec = spec
        self._levels = np.ascontiguousarray(np.asarray(index_level, dtype=np.float64))
        if self._levels.ndim != 1:
            raise ValueError("index_level must be a 1-D array")
        self._offset = int(rebalance_offset)
        self._delay = int(execution_delay)
        self._start: int | None = None

    def target_contracts(self, state: OverlayState) -> int | None:
        """Net-short contracts of the regime hedge on rebalance rows, else None.

        Reads only ``state`` plus the causal prefix ``index_level[: t - execution_delay + 1]``.
        """
        if self._start is None:
            self._start = int(state.session_idx)
        run_row = int(state.session_idx) - int(self._start)
        if run_row < 0:
            raise ValueError(f"session_idx {state.session_idx} is before the run start {self._start}")
        every = int(self._spec.rebalance_every_sessions)
        if run_row % every != self._offset:
            return None
        if not float(self._spec.max_fraction) > 0.0:
            return 0
        idx = int(state.session_idx) - self._delay
        if idx < 0:
            return 0
        if idx >= int(self._levels.shape[0]):
            raise PITDataError(f"index level missing at row {state.session_idx}")
        level_t = float(self._levels[idx])
        if not math.isfinite(level_t) or level_t <= 0.0:
            raise PITDataError(f"index level missing at row {state.session_idx}")
        nav = float(state.nav)
        if not math.isfinite(nav) or nav <= 0.0:
            return 0

        signal = tsmom_signal(self._levels, idx, self._spec.tsmom_horizons)
        if signal is None or signal >= 0:
            return 0
        sigma = realised_vol(self._levels, idx, int(self._spec.vol_window_sessions))
        if sigma is None:
            return 0
        scale = Fraction(str(float(self._spec.max_fraction)))
        if sigma > 0.0:
            scale = min(scale, Fraction(str(float(self._spec.target_vol))) / Fraction(str(sigma)))
        notional = -signal * scale * state.nav / (self._spec.contract_multiplier_krw * Fraction(str(level_t)))
        whole, remainder = divmod(notional.numerator, notional.denominator)
        return int(whole + int(2 * remainder > notional.denominator))


class CompositeOverlay:
    """``OverlayPolicy`` combining a primary overlay with named secondary legs into one ``OverlayTarget``.

    A component returning None keeps its current position (the primary from ``state.contracts``, a leg from
    ``state.leg_contracts``). The composite returns None only when every component returns None.

    Raises ValueError when the primary is silent while an inverse-ETF position is held: the state carries units but
    no price, so the position cannot be restated and a zero target would silently liquidate it.
    """

    def __init__(self, primary: OverlayPolicy, legs: Mapping[str, RegimeHedgeLeg]) -> None:
        self._primary = primary
        self._legs = dict(legs)
        for name in self._legs:
            check_leg_name(name)

    def target(self, state: OverlayState) -> OverlayTarget | None:
        primary_target = self._primary.target(state)
        leg_targets: dict[str, int | None] = {
            name: leg.target_contracts(state) for name, leg in self._legs.items()
        }
        if primary_target is None and all(cnt is None for cnt in leg_targets.values()):
            return None

        if primary_target is None and int(state.inverse_units) != 0:
            raise ValueError("CompositeOverlay cannot carry an inverse-ETF position across a silent primary")
        contracts = primary_target.contracts if primary_target is not None else int(state.contracts)
        inverse_value = primary_target.inverse_value_krw if primary_target is not None else 0

        current_leg_contracts = dict(state.leg_contracts)
        out_legs: list[tuple[str, int]] = []
        for name in sorted(self._legs.keys()):
            cnt = leg_targets[name]
            if cnt is not None:
                out_legs.append((name, int(cnt)))
            else:
                out_legs.append((name, int(current_leg_contracts.get(name, 0))))

        return OverlayTarget(
            contracts=int(contracts),
            inverse_value_krw=int(inverse_value),
            legs=tuple(out_legs),
        )
