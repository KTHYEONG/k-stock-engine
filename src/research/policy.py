"""Top-N selection policy with a per-name own-trend cash rule."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import date

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from src.research.cube import ResearchCube
from src.research.model import ScoreMatrix
from src.research.panel import FeaturePanel

__all__ = ["TrendCashPolicy", "UniverseRule", "build_targets", "decision_rows", "universe_mask"]


class UniverseRule(BaseModel):
    """Liquidity and price floors defining the tradable universe."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    min_adtv20_krw: int
    min_price_krw: int

    @field_validator("min_adtv20_krw", "min_price_krw")
    @classmethod
    def _non_negative_int(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"universe threshold must be a non-negative int, got {value!r}")
        return value


def universe_mask(cube: ResearchCube, rule: UniverseRule) -> NDArray[np.bool_]:
    """Tradable-name mask for every row: present, eligible, not entry-blocked, traded volume > 0,
    adtv20 >= floor, close >= price floor and finite 60-session volatility (unchanged from the research cube
    contract)."""
    present = np.asarray(cube.arrays["present"], dtype=bool)
    eligible = np.asarray(cube.arrays["eligible"], dtype=bool)
    blocked = np.asarray(cube.arrays["entry_blocked"], dtype=bool)
    volume = np.asarray(cube.arrays["volume"], dtype=np.float64)
    adtv20 = np.asarray(cube.arrays["adtv20"], dtype=np.float64)
    close = np.asarray(cube.arrays["close"], dtype=np.float64)
    vol60 = np.asarray(cube.arrays["ret_vol60"], dtype=np.float64)
    mask: NDArray[np.bool_] = (
        present
        & eligible
        & ~blocked
        & (volume > 0)
        & (adtv20 >= float(rule.min_adtv20_krw))
        & (close >= float(rule.min_price_krw))
        & np.isfinite(vol60)
    )
    return np.asarray(mask, dtype=bool)


class TrendCashPolicy(BaseModel):
    """Top-N selection by ensemble score with a per-name own-trend cash rule.

    Why: drawdowns are market-driven; picks whose own 1-month trend has failed are the earliest,
    bottom-up signal of a risk-off state, and holding cash for them cut the probe drawdown from -47% to -15%
    without any index timing. The rule only works on ML picks (random picks + the same rule lost money).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    family: str = "ml_trend_cash"
    n: int = 20
    keep_rank_multiple: float = 3.0
    rebalance_every_sessions: int = 5
    universe: UniverseRule = UniverseRule(min_adtv20_krw=500_000_000, min_price_krw=1_000)
    trend_min_dev_ma20: float | None = None
    trend_min_ret21: float | None = None
    min_units_per_slot: int = 3

    @field_validator("family")
    @classmethod
    def _non_empty_family(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("family must be a non-empty string")
        return value

    @field_validator("n")
    @classmethod
    def _positive_n(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"n must be an int >= 1, got {value!r}")
        return value

    @field_validator("keep_rank_multiple")
    @classmethod
    def _non_negative_multiple(cls, value: object) -> float:
        multiple = float(value)  # type: ignore[arg-type]
        if not math.isfinite(multiple) or multiple < 0.0:
            raise ValueError(f"keep_rank_multiple must be finite and >= 0, got {value!r}")
        return multiple

    @field_validator("rebalance_every_sessions")
    @classmethod
    def _positive_every(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"rebalance_every_sessions must be an int >= 1, got {value!r}")
        return value

    @field_validator("min_units_per_slot")
    @classmethod
    def _non_negative_units(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"min_units_per_slot must be an int >= 0, got {value!r}")
        return value

    @field_validator("trend_min_dev_ma20", "trend_min_ret21")
    @classmethod
    def _finite_threshold(cls, value: object) -> float | None:
        if value is None:
            return None
        threshold = float(value)  # type: ignore[arg-type]
        if not math.isfinite(threshold):
            raise ValueError(f"trend threshold must be finite, got {value!r}")
        return threshold

    def canonical_json(self) -> str:
        """Canonical JSON of the policy with sorted keys and compact separators."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        """SHA-256 hex of the canonical JSON; capital is excluded (evaluation scenario, not identity)."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def decision_rows(
    sessions: Sequence[date], *, lo: int, hi: int, every: int, phase: int
) -> tuple[int, ...]:
    """Decision rows ``lo - 1 + phase, + every, ...`` strictly inside ``[lo - 1, hi)``.

    Raises: ValueError for every < 1, phase outside [0, every), or a window outside the sessions.
    """
    total = len(sessions)
    if (
        isinstance(lo, bool)
        or not isinstance(lo, int)
        or isinstance(hi, bool)
        or not isinstance(hi, int)
        or not 0 <= lo <= hi <= total
    ):
        raise ValueError(f"window [{lo}, {hi}) is not within [0, {total})")
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError(f"every must be an int >= 1, got {every!r}")
    if isinstance(phase, bool) or not isinstance(phase, int) or not 0 <= phase < every:
        raise ValueError(f"phase must satisfy 0 <= phase < every, got {phase!r}")
    out: list[int] = []
    row = lo - 1 + phase
    while row < hi:
        if row >= 0:
            out.append(row)
        row += every
    return tuple(out)


def build_targets(
    policy: TrendCashPolicy,
    cube: ResearchCube,
    panel: FeaturePanel,
    scores: ScoreMatrix,
    universe: NDArray[np.bool_],
    *,
    rows: Sequence[int],
    capital_krw: int,
    cash_buffer: float,
) -> dict[int, NDArray[np.float64]]:
    """Target weights (fractions of NAV) per decision row; the remainder of NAV is cash.

    Per row: candidates are universe names with a finite score that satisfy ``min_units_per_slot``;
    rank by score descending (stable, ties by instrument index); keep previous selections whose rank is
    below ``n * keep_rank_multiple`` (up to n), fill the rest by rank; then a selected name receives
    weight 1/n when every enabled trend leg passes: ``dev_ma20 > trend_min_dev_ma20`` (enabled when
    the threshold is not None, feature must be finite) and ``ret_21 > trend_min_ret21`` (same). With
    both legs disabled every selected name receives 1/n; names are never dropped for a disabled leg,
    even when its feature is NaN. Others receive 0 (cash). Incumbency is tracked on the selection
    before the trend rule. Fewer than n candidates: all-zero row and incumbents reset.

    Raises: ValueError if a row has no score row in ``scores`` or ``capital_krw``/``cash_buffer`` invalid.
    """
    score_mat = np.asarray(scores.scores, dtype=np.float64)
    n_scored, n_names = score_mat.shape
    uni = np.asarray(universe, dtype=bool)
    dev_enabled = policy.trend_min_dev_ma20 is not None
    ret_enabled = policy.trend_min_ret21 is not None
    dev: NDArray[np.float64] | None = None
    ret: NDArray[np.float64] | None = None
    if dev_enabled:
        dev = np.asarray(panel.features["dev_ma20"], dtype=np.float64)
        if dev.shape != (n_scored, n_names):
            raise ValueError("universe, panel features and cube close must match the scores shape")
    if ret_enabled:
        ret = np.asarray(panel.features["ret_21"], dtype=np.float64)
        if ret.shape != (n_scored, n_names):
            raise ValueError("universe, panel features and cube close must match the scores shape")
    close = np.asarray(cube.arrays["close"], dtype=np.float64)
    if uni.shape != (n_scored, n_names) or close.shape != (n_scored, n_names):
        raise ValueError("universe, panel features and cube close must match the scores shape")
    for t in rows:
        if isinstance(t, bool) or not isinstance(t, int) or not 0 <= int(t) < n_scored:
            raise ValueError(f"row {t!r} has no score row in scores")
    if isinstance(capital_krw, bool) or not isinstance(capital_krw, int) or capital_krw < 1:
        raise ValueError(f"capital_krw must be an int >= 1, got {capital_krw!r}")
    buffer = float(cash_buffer)
    if not math.isfinite(buffer) or not 0.0 <= buffer < 1.0:
        raise ValueError(f"cash_buffer must satisfy 0 <= cash_buffer < 1, got {cash_buffer!r}")
    slot_krw = float(capital_krw) * (1.0 - buffer) / float(policy.n)
    units = int(policy.min_units_per_slot)
    keep_limit = int(float(policy.n) * float(policy.keep_rank_multiple))
    weight = 1.0 / float(policy.n)
    out: dict[int, NDArray[np.float64]] = {}
    incumbents: set[int] = set()
    for t in rows:
        row = int(t)
        uni_row = uni[row]
        score_row = score_mat[row]
        if units > 0:
            close_row = close[row]
            affordable = (
                np.isfinite(close_row) & (close_row > 0.0) & (close_row * float(units) <= slot_krw)
            )
        else:
            affordable = np.ones(n_names, dtype=bool)
        idx = np.flatnonzero(uni_row & np.isfinite(score_row) & affordable)
        weights = np.zeros(n_names, dtype=np.float64)
        if idx.size >= policy.n:
            ranked = idx[np.argsort(-score_row[idx], kind="stable")]
            position = {int(k): p for p, k in enumerate(ranked)}
            kept = [
                int(k) for k in ranked if int(k) in incumbents and position[int(k)] < keep_limit
            ][: policy.n]
            kept_set = set(kept)
            for k in ranked:
                if len(kept) >= policy.n:
                    break
                if int(k) not in kept_set:
                    kept_set.add(int(k))
                    kept.append(int(k))
            kept_arr = np.asarray(kept, dtype=np.int64)
            if dev_enabled and ret_enabled:
                assert dev is not None
                assert ret is not None
                passing = (
                    np.isfinite(dev[row, kept_arr])
                    & np.isfinite(ret[row, kept_arr])
                    & (dev[row, kept_arr] > float(policy.trend_min_dev_ma20))  # type: ignore[arg-type]
                    & (ret[row, kept_arr] > float(policy.trend_min_ret21))  # type: ignore[arg-type]
                )
            elif dev_enabled:
                assert dev is not None
                passing = np.isfinite(dev[row, kept_arr]) & (
                    dev[row, kept_arr] > float(policy.trend_min_dev_ma20)  # type: ignore[arg-type]
                )
            elif ret_enabled:
                assert ret is not None
                passing = np.isfinite(ret[row, kept_arr]) & (
                    ret[row, kept_arr] > float(policy.trend_min_ret21)  # type: ignore[arg-type]
                )
            else:
                passing = np.ones(len(kept), dtype=bool)
            for pos_k, k in enumerate(kept):
                if bool(passing[pos_k]):
                    weights[int(k)] = weight
            incumbents = set(kept)
        else:
            incumbents = set()
        out[row] = np.ascontiguousarray(weights)
    return out
