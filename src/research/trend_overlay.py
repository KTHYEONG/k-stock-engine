"""KOSPI 200 index trend overlay: signed futures by a time-series MA rule."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.core.pit import PITDataError

if TYPE_CHECKING:
    from src.backtest.overlay import DerivativeConfig, OverlayState, OverlayTarget

__all__ = ["TrendOverlay", "TrendOverlaySpec", "trend_derivative_config"]


class TrendOverlaySpec(BaseModel):
    """Frozen identity of the index trend overlay; canonical JSON is part of the strategy identity.

    The overlay holds integer index futures sized to ``long_fraction · NAV`` (long) while the underlying closes above
    its ``ma_sessions`` simple moving average, and ``short_fraction · NAV`` (short) otherwise. Why time-series trend
    on the large-cap index: the stock book's alpha is confined to small caps (no ML alpha inside the top-200 caps), so
    large-cap rallies can only be owned as index exposure, and an index trend filter keeps that exposure off in
    sustained declines.

    Fields: ma_sessions (int >= 2); long_fraction, short_fraction (finite, >= 0); rebalance_every_sessions (int >= 1);
    contract_multiplier_krw (int > 0); initial_margin_rate, margin_buffer_rate (sum < 1); margin_topup_trigger_fraction
    in (0, 1]; futures_cost_rate in [0, 1); futures_tax_rate in [0, 1); futures_annual_deduction_krw (int >= 0).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ma_sessions: int
    long_fraction: float
    short_fraction: float
    rebalance_every_sessions: int
    contract_multiplier_krw: int
    initial_margin_rate: float
    margin_buffer_rate: float
    margin_topup_trigger_fraction: float
    futures_cost_rate: float
    futures_tax_rate: float
    futures_annual_deduction_krw: int

    @field_validator("ma_sessions", mode="before")
    @classmethod
    def _ma(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or int(value) < 2:
            raise ValueError(f"ma_sessions must be an int >= 2, got {value!r}")
        return int(value)

    @field_validator("long_fraction", "short_fraction")
    @classmethod
    def _fraction(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or out < 0.0:
            raise ValueError(f"fraction must be finite and >= 0, got {value!r}")
        return out

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

    @field_validator("initial_margin_rate")
    @classmethod
    def _margin_rate(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 < out < 1.0:
            raise ValueError(f"initial_margin_rate must be in (0, 1), got {value!r}")
        return out

    @field_validator("margin_buffer_rate")
    @classmethod
    def _buffer_rate(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 < out < 1.0:
            raise ValueError(f"margin_buffer_rate must be in (0, 1), got {value!r}")
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
    def _check_margin_sum(self) -> TrendOverlaySpec:
        if float(self.initial_margin_rate) + float(self.margin_buffer_rate) >= 1.0:
            raise ValueError("initial_margin_rate + margin_buffer_rate must be < 1")
        return self

    def canonical_json(self) -> str:
        """Canonical JSON with sorted keys and compact separators."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def trend_derivative_config(spec: TrendOverlaySpec) -> DerivativeConfig:
    """Engine derivative terms of the overlay; inverse-ETF terms are zero because the overlay never holds the ETF."""
    from decimal import Decimal

    from src.backtest.overlay import DerivativeConfig

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


class TrendOverlay:
    """``OverlayPolicy`` holding signed index futures by the trend rule of ``TrendOverlaySpec``.

    ``index_level`` is the futures-underlying close aligned to the engine sessions (row = session index). The decision
    after session ``t``'s close reads only ``index_level[: t - execution_delay + 1]`` and ``state.nav``, so it is causal
    by construction. With fewer than ``ma_sessions`` finite positive levels in the window, the target is flat (no
    position). Why flat and not long: an unknown trend must not add leverage.

    Rebalances on run rows ``r % rebalance_every_sessions == rebalance_offset``; returns None on other rows.
    Contract count: the nearest integer to ``fraction · nav / (multiplier · level_t)``, ties toward zero. The sign
    follows the ledger convention: long = negative net-short count.
    """

    def __init__(
        self,
        spec: TrendOverlaySpec,
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

    def target(self, state: OverlayState) -> OverlayTarget | None:
        """New overlay target on rebalance rows, else None. Reads only ``state`` plus the causal prefix."""
        from src.backtest.overlay import OverlayTarget

        if self._start is None:
            self._start = int(state.session_idx)
        run_row = int(state.session_idx) - int(self._start)
        if run_row < 0:
            raise ValueError(f"session_idx {state.session_idx} is before the run start {self._start}")
        every = int(self._spec.rebalance_every_sessions)
        if run_row % every != self._offset:
            return None
        if not (float(self._spec.long_fraction) > 0.0 or float(self._spec.short_fraction) > 0.0):
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        idx = int(state.session_idx) - self._delay
        if idx < 0:
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        if idx >= int(self._levels.shape[0]):
            raise PITDataError(f"index level missing at row {state.session_idx}")
        level_t = float(self._levels[idx])
        if not math.isfinite(level_t) or level_t <= 0.0:
            raise PITDataError(f"index level missing at row {state.session_idx}")
        nav = float(state.nav)
        if not math.isfinite(nav) or nav <= 0.0:
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        ma = int(self._spec.ma_sessions)
        lo = max(idx - ma + 1, 0)
        window = self._levels[lo : idx + 1]
        finite = np.isfinite(window) & (window > 0.0)
        if int(np.count_nonzero(finite)) < ma:
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        scale = float(np.max(window))
        is_long = bool(level_t / scale > float(np.mean(window / scale)))
        fraction = float(self._spec.long_fraction) if is_long else float(self._spec.short_fraction)
        if not fraction > 0.0:
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        # Exact quoted decimals keep half-contract ties unbiased without an epsilon dead zone.
        notional = Fraction(str(fraction)) * state.nav / (self._spec.contract_multiplier_krw * Fraction(str(level_t)))
        whole, remainder = divmod(notional.numerator, notional.denominator)
        magnitude = whole + int(2 * remainder > notional.denominator)
        contracts = -magnitude if is_long else magnitude
        return OverlayTarget(contracts=int(contracts), inverse_value_krw=0)
