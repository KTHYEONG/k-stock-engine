"""Five phase-shifted stock sleeves and their combined book targets."""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from src.research.cube import ResearchCube
from src.research.model import ScoreMatrix
from src.research.panel import FeaturePanel
from src.research.policy import TrendCashPolicy, build_targets, decision_rows

__all__ = ["BookSpec", "build_sleeve_targets", "combine_sleeve_targets", "mean_sleeve_returns", "sleeve_capital_krw"]


class BookSpec(BaseModel):
    """Sleeve count, stock planning fraction and execution band; part of the strategy identity.

    Continuing holdings with positive targets skip resizes when the absolute value drift is
    below ``rebalance_band * target value``. Resizing pays slippage, impact and sell tax;
    entries, zero targets and forced exits are never suppressed. Zero disables the band.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sleeves: int
    stock_capital_fraction: float
    rebalance_band: float = 0.0

    @field_validator("sleeves")
    @classmethod
    def _positive_sleeves(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"sleeves must be an int >= 1, got {value!r}")
        return int(value)

    @field_validator("stock_capital_fraction")
    @classmethod
    def _fraction(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 < out <= 1.0:
            raise ValueError(f"stock_capital_fraction must satisfy 0 < f <= 1, got {value!r}")
        return out

    @field_validator("rebalance_band", mode="before")
    @classmethod
    def _band(cls, value: object) -> float:
        if isinstance(value, bool):
            raise ValueError(f"rebalance_band must satisfy 0 <= b < 1, got {value!r}")
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 <= out < 1.0:
            raise ValueError(f"rebalance_band must satisfy 0 <= b < 1, got {value!r}")
        return out

    def canonical_json(self) -> str:
        """Canonical JSON with sorted keys and compact separators.

        ``rebalance_band`` is emitted only when non-zero so pre-band specs keep their hash.
        """
        payload = self.model_dump(mode="json")
        if self.rebalance_band == 0.0:
            del payload["rebalance_band"]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sleeve_capital_krw(total_capital_krw: int, spec: BookSpec) -> int:
    """Capital each sleeve is simulated at: floor(total * stock_capital_fraction / sleeves).

    Why a fixed fraction: the hedge overlay later resizes the stock book to NAV / (1 + hedge ratio * beta);
    the fraction is the planning ratio (about 0.75 at beta 0.35) that sets unit affordability and impact.
    Raises ValueError for total_capital_krw < 1 or a result < 1.
    """
    if isinstance(total_capital_krw, bool) or not isinstance(total_capital_krw, int) or total_capital_krw < 1:
        raise ValueError(f"total_capital_krw must be an int >= 1, got {total_capital_krw!r}")
    result = math.floor(float(total_capital_krw) * float(spec.stock_capital_fraction) / float(spec.sleeves))
    if result < 1:
        raise ValueError(f"sleeve capital is below 1 KRW for total {total_capital_krw!r}")
    return result


def build_sleeve_targets(
    policy: TrendCashPolicy,
    cube: ResearchCube,
    panel: FeaturePanel,
    scores: ScoreMatrix,
    universe: NDArray[np.bool_],
    *,
    sessions: Sequence[date],
    sleeves: int,
    lo: int,
    hi: int,
    sleeve_capital_krw: int,
    cash_buffer: float,
) -> tuple[dict[int, NDArray[np.float64]], ...]:
    """Targets of each sleeve ``s`` in ``range(sleeves)``: ``build_targets`` on ``decision_rows`` at sleeve capital.

    Raises ValueError when ``sleeves != policy.rebalance_every_sessions`` (phases would not tile the sessions).
    """
    if isinstance(sleeves, bool) or not isinstance(sleeves, int) or sleeves < 1:
        raise ValueError(f"sleeves must be an int >= 1, got {sleeves!r}")
    if sleeves != int(policy.rebalance_every_sessions):
        raise ValueError(f"sleeves {sleeves!r} must equal the policy rebalance_every_sessions")
    every = int(policy.rebalance_every_sessions)
    out: list[dict[int, NDArray[np.float64]]] = []
    for phase in range(sleeves):
        rows = decision_rows(sessions, lo=lo, hi=hi, every=every, phase=phase)
        out.append(
            build_targets(
                policy, cube, panel, scores, universe,
                rows=list(rows), capital_krw=int(sleeve_capital_krw), cash_buffer=float(cash_buffer),
            )
        )
    return tuple(out)


def combine_sleeve_targets(
    sleeve_targets: Sequence[Mapping[int, NDArray[np.float64]]], *, lo: int, hi: int
) -> dict[int, NDArray[np.float64]]:
    """Stock-book target for every row ``r`` in ``[lo - 1, hi - 1]``: the mean of each sleeve's last target.

    A sleeve with no decision yet contributes zeros. Weights are fractions of stock-book NAV.
    Raises ValueError for an empty sleeve list, mismatched weight lengths, or lo < 1 / hi < lo.
    """
    if isinstance(lo, bool) or not isinstance(lo, int) or lo < 1:
        raise ValueError(f"lo must be an int >= 1, got {lo!r}")
    if isinstance(hi, bool) or not isinstance(hi, int) or hi < lo:
        raise ValueError(f"hi must satisfy hi >= lo, got {hi!r}")
    sleeves = list(sleeve_targets)
    if not sleeves:
        raise ValueError("sleeve_targets must be non-empty")
    width: int | None = None
    for mapping in sleeves:
        for weights in mapping.values():
            arr = np.asarray(weights, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError("sleeve weights must be 1-D")
            if width is None:
                width = int(arr.shape[0])
            elif int(arr.shape[0]) != width:
                raise ValueError("sleeve weight lengths must match")
    if width is None:
        raise ValueError("sleeve_targets contain no weights to infer width")
    n_sleeves = len(sleeves)
    sorted_rows: list[list[int]] = [sorted(mapping.keys()) for mapping in sleeves]
    for rows in sorted_rows:
        for row in rows:
            if isinstance(row, bool) or not isinstance(row, int):
                raise ValueError(f"sleeve target row must be an int, got {row!r}")
    out: dict[int, NDArray[np.float64]] = {}
    for row in range(lo - 1, hi):
        acc = np.zeros(width, dtype=np.float64)
        for mapping, rows in zip(sleeves, sorted_rows, strict=True):
            pos = bisect.bisect_right(rows, row) - 1
            if pos >= 0:
                acc += np.asarray(mapping[rows[pos]], dtype=np.float64)
        out[row] = np.ascontiguousarray(acc / float(n_sleeves), dtype=np.float64)
    return out


def mean_sleeve_returns(log_return_streams: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Simple-return stream of equal-weight sleeves: the mean of ``expm1`` of each log-return stream.

    Raises ValueError for an empty list, unequal lengths, or non-finite values.
    """
    streams = list(log_return_streams)
    if not streams:
        raise ValueError("log_return_streams must be non-empty")
    arrays = [np.asarray(stream, dtype=np.float64) for stream in streams]
    length = int(arrays[0].shape[0]) if arrays[0].ndim == 1 else -1
    if length < 0:
        raise ValueError("log-return streams must be 1-D")
    for arr in arrays:
        if arr.ndim != 1 or int(arr.shape[0]) != length:
            raise ValueError("log-return streams must have equal lengths")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("log-return streams must be finite")
    stacked = np.expm1(np.stack(arrays, axis=0))
    return np.ascontiguousarray(np.mean(stacked, axis=0), dtype=np.float64)
