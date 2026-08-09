"""Point-in-time stock universe policy with explicit exclusion reasons."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256

import polars as pl


@dataclass(frozen=True, slots=True)
class UniversePolicy:
    """Versioned, validated universe policy for common-stock membership.

    Criteria are structural invariants (six-digit common-stock code, positive
    close, no capital erosion, positive operating income) rather than magic
    numbers.
    """

    version: str
    min_close: float = 0.0
    require_operating_income: bool = True
    max_capital_erosion_pct: float = 50.0

    @property
    def fingerprint(self) -> str:
        payload = "\n".join(
            [
                self.version,
                str(self.min_close),
                str(self.require_operating_income),
                str(self.max_capital_erosion_pct),
            ]
        )
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class UniverseResult:
    decision_time: datetime
    policy: UniversePolicy
    members: tuple[str, ...]
    exclusions: dict[str, str] = field(default_factory=dict)


class PointInTimeUniverse:
    """Apply a universe policy at a decision time and return exclusion reasons."""

    def __init__(self, policy: UniversePolicy, code_column: str = "code", date_column: str = "date"):
        self.policy = policy
        self.code_column = code_column
        self.date_column = date_column

    @staticmethod
    def is_common_stock_code(code: object) -> bool:
        return isinstance(code, str) and code.isdigit() and len(code) == 6

    def apply(self, frame: pl.DataFrame, decision_time: datetime) -> UniverseResult:
        if self.date_column not in frame.columns:
            raise ValueError(f"missing date column {self.date_column!r}")
        available = frame.filter(pl.col(self.date_column) <= decision_time)
        if available.is_empty():
            return UniverseResult(
                decision_time=decision_time, policy=self.policy, members=(), exclusions={}
            )

        exclusions: dict[str, str] = {}
        codes = available[self.code_column].unique().to_list()

        for code in codes:
            rows = available.filter(pl.col(self.code_column) == code)
            if not self.is_common_stock_code(code):
                exclusions[str(code)] = "non-common-stock-code"
                continue
            last = rows.sort(self.date_column).tail(1)
            if self.policy.min_close > 0 and float(last["close"][0]) < self.policy.min_close:
                exclusions[str(code)] = "close-below-min"
                continue
            if "capital_erosion_rate" in last.columns:
                erosion = float(last["capital_erosion_rate"][0])
                if erosion > self.policy.max_capital_erosion_pct:
                    exclusions[str(code)] = "capital-erosion"
                    continue
            if self.policy.require_operating_income and "operating_income" in last.columns:
                oi = last["operating_income"][0]
                if oi is None or float(oi) <= 0:
                    exclusions[str(code)] = "no-positive-operating-income"
                    continue

        members = tuple(c for c in codes if c not in exclusions)
        return UniverseResult(
            decision_time=decision_time,
            policy=self.policy,
            members=members,
            exclusions=exclusions,
        )
