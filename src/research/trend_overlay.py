"""KOSPI 200 index trend overlay: signed futures by a time-series MA rule."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.core.pit import PITDataError

if TYPE_CHECKING:
    from src.backtest.overlay import DerivativeConfig, OverlayState, OverlayTarget

__all__ = ["TrendOverlay", "TrendOverlaySpec", "realised_vol", "trend_derivative_config", "tsmom_signal"]

_SESSIONS_PER_YEAR = 252  # KRX annualisation convention for the vol_cap estimate, not a tunable


class TrendOverlaySpec(BaseModel):
    """Frozen identity of the index trend overlay; canonical JSON is part of the strategy identity.

    The overlay holds integer index futures sized to ``long_fraction · NAV`` (long) or ``short_fraction · NAV`` (short).
    Why time-series trend on the large-cap index: the stock book's alpha is confined to small caps (no ML alpha inside
    the top-200 caps), so large-cap rallies can only be owned as index exposure, and an index trend filter keeps that
    exposure off in sustained declines.

    ``signal``: ``"ma"`` (level above its ``ma_sessions`` mean -> long, else short) or ``"tsmom"`` (s = the mean over
    ``tsmom_horizons`` of sign(level_t / level_{t-h} - 1), in [-1, 1]). Why the TSMOM average: it has no length to
    pick, and choosing the in-sample best MA length had no out-of-sample value (CSCV PBO 0.58).
    ``vol_cap`` (None = off): annualised realised volatility above which the position shrinks by vol_cap / vol. Why
    crisis-only and not vol targeting: always-on targeting cut normal-regime trend gains; the cap only acts in
    turbulent regimes, where a fixed notional multiplied risk 3.5x (2026).
    ``vol_window_sessions``: sessions of daily log returns in the realised-vol estimate (required with ``vol_cap``).

    Fields: signal ("ma" | "tsmom"); ma_sessions (int >= 2 when ma); tsmom_horizons (tuple[int, ...] when tsmom);
    vol_cap (float > 0 | None); vol_window_sessions (int >= 2 | None);
    long_fraction, short_fraction (finite, >= 0); rebalance_every_sessions (int >= 1);
    contract_multiplier_krw (int > 0); initial_margin_rate, margin_buffer_rate (sum < 1); margin_topup_trigger_fraction
    in (0, 1]; futures_cost_rate in [0, 1); futures_tax_rate in [0, 1); futures_annual_deduction_krw (int >= 0).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal: Literal["ma", "tsmom"] = "ma"
    ma_sessions: int | None = None
    tsmom_horizons: tuple[int, ...] | None = None
    vol_cap: float | None = None
    vol_window_sessions: int | None = None
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
    def _ma(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or int(value) < 2:
            raise ValueError(f"ma_sessions must be an int >= 2, got {value!r}")
        return int(value)

    @field_validator("tsmom_horizons", mode="before")
    @classmethod
    def _horizons(cls, value: object) -> tuple[int, ...] | None:
        if value is None:
            return None
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

    @field_validator("vol_cap")
    @classmethod
    def _vol_cap(cls, value: object) -> float | None:
        if value is None:
            return None
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or out <= 0.0:
            raise ValueError(f"vol_cap must be finite and > 0, got {value!r}")
        return out

    @field_validator("vol_window_sessions", mode="before")
    @classmethod
    def _vol_window(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or int(value) < 2:
            raise ValueError(f"vol_window_sessions must be an int >= 2, got {value!r}")
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
    def _validate_spec(self) -> TrendOverlaySpec:
        if float(self.initial_margin_rate) + float(self.margin_buffer_rate) >= 1.0:
            raise ValueError("initial_margin_rate + margin_buffer_rate must be < 1")
        if self.signal == "ma":
            if self.ma_sessions is None:
                raise ValueError("signal='ma' requires ma_sessions >= 2")
            if self.tsmom_horizons is not None:
                raise ValueError("signal='ma' requires tsmom_horizons is None")
        else:
            if self.tsmom_horizons is None:
                raise ValueError("signal='tsmom' requires non-empty tsmom_horizons")
            if self.ma_sessions is not None:
                raise ValueError("signal='tsmom' requires ma_sessions is None")
        if (self.vol_cap is None) != (self.vol_window_sessions is None):
            raise ValueError("vol_cap and vol_window_sessions must be set together")
        return self

    def canonical_json(self) -> str:
        """Canonical JSON with sorted keys and compact separators.

        ``signal`` is omitted when ``"ma"`` and every None-valued field is omitted, so pre-v6 specs keep their exact
        bytes.
        """
        payload = self.model_dump(mode="json")
        cleaned = {k: v for k, v in payload.items() if v is not None}
        if cleaned.get("signal") == "ma":
            del cleaned["signal"]
        return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def tsmom_signal(levels: NDArray[np.float64], idx: int, horizons: tuple[int, ...]) -> Fraction | None:
    """Mean over ``horizons`` of sign(levels[idx] / levels[idx - h] - 1), exact in [-1, 1]; reads rows <= ``idx``.

    Returns None when a horizon reaches before row 0 or a needed level is non-finite or non-positive.
    """
    level_t = float(levels[idx])
    if not math.isfinite(level_t) or level_t <= 0.0:
        return None
    total = 0
    for h in horizons:
        if idx - h < 0:
            return None
        past = float(levels[idx - h])
        if not math.isfinite(past) or past <= 0.0:
            return None
        total += (level_t > past) - (level_t < past)
    return Fraction(total, len(horizons))


def realised_vol(levels: NDArray[np.float64], idx: int, window: int) -> float | None:
    """Annualised population std (ddof 0) of the ``window`` daily log returns ending at row ``idx``; None when any
    level in ``[idx - window, idx]`` is missing, non-positive or before row 0."""
    if idx - window < 0:
        return None
    span = levels[idx - window : idx + 1]
    if not bool(np.all(np.isfinite(span) & (span > 0.0))):
        return None
    return float(np.std(np.diff(np.log(span)), ddof=0)) * math.sqrt(_SESSIONS_PER_YEAR)


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
    by construction. With insufficient finite positive levels for the signal or volatility cap, the target is flat
    (no position). Why flat and not long: an unknown trend must not add leverage.

    Rebalances on run rows ``r % rebalance_every_sessions == rebalance_offset``; returns None on other rows.
    Contract count: the nearest integer to ``fraction · |s| · c · nav / (multiplier · level_t)``, ties toward zero.
    The sign follows the ledger convention: long = negative net-short count.
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

        signal = self._signal_value(idx, level_t)
        if signal is None or signal == 0:
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        fraction = float(self._spec.long_fraction) if signal > 0 else float(self._spec.short_fraction)
        if not fraction > 0.0:
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        cap = self._vol_cap_multiplier(idx)
        if cap is None:
            return OverlayTarget(contracts=0, inverse_value_krw=0)

        # Exact quoted decimals keep half-contract ties unbiased without an epsilon dead zone; the TSMOM average is an
        # exact ratio so a 1/3 signal does not inherit binary rounding.
        size = Fraction(str(fraction)) * abs(signal) * cap
        notional = size * state.nav / (self._spec.contract_multiplier_krw * Fraction(str(level_t)))
        whole, remainder = divmod(notional.numerator, notional.denominator)
        magnitude = whole + int(2 * remainder > notional.denominator)
        contracts = -magnitude if signal > 0 else magnitude
        return OverlayTarget(contracts=int(contracts), inverse_value_krw=0)

    def _signal_value(self, idx: int, level_t: float) -> Fraction | None:
        """Trend signal in [-1, 1] from levels up to row ``idx``; None when the needed history is incomplete."""
        if self._spec.signal == "ma":
            ma = self._spec.ma_sessions
            assert ma is not None  # guaranteed by TrendOverlaySpec
            window = self._levels[max(idx - ma + 1, 0) : idx + 1]
            if int(np.count_nonzero(np.isfinite(window) & (window > 0.0))) < ma:
                return None
            scale = float(np.max(window))
            return Fraction(1) if level_t / scale > float(np.mean(window / scale)) else Fraction(-1)
        horizons = self._spec.tsmom_horizons
        assert horizons is not None  # guaranteed by TrendOverlaySpec
        return tsmom_signal(self._levels, idx, horizons)

    def _vol_cap_multiplier(self, idx: int) -> Fraction | None:
        """min(1, vol_cap / annualised realised vol) over the window ending at ``idx``; None when it is incomplete."""
        if self._spec.vol_cap is None:
            return Fraction(1)
        window = self._spec.vol_window_sessions
        assert window is not None  # guaranteed by TrendOverlaySpec
        sigma = realised_vol(self._levels, idx, window)
        if sigma is None:
            return None
        if not sigma > float(self._spec.vol_cap):
            return Fraction(1)
        return Fraction(str(float(self._spec.vol_cap))) / Fraction(str(sigma))
