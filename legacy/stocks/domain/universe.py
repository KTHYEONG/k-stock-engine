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
    numbers. ``require_historical_master`` toggles whether listing/delisting and
    tradability intervals are mandatory for a promotable membership decision.
    """

    version: str
    min_close: float = 0.0
    require_operating_income: bool = True
    max_capital_erosion_pct: float = 50.0
    require_historical_master: bool = True

    @property
    def fingerprint(self) -> str:
        payload = "\n".join(
            [
                self.version,
                str(self.min_close),
                str(self.require_operating_income),
                str(self.max_capital_erosion_pct),
                str(self.require_historical_master),
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

    def __init__(
        self,
        policy: UniversePolicy,
        code_column: str = "code",
        available_column: str = "available_time",
    ):
        self.policy = policy
        self.code_column = code_column
        self.available_column = available_column

    @staticmethod
    def is_common_stock_code(code: object) -> bool:
        return isinstance(code, str) and code.isdigit() and len(code) == 6

    def apply(self, frame: pl.DataFrame, decision_time: datetime) -> UniverseResult:
        if self.available_column not in frame.columns:
            raise ValueError(f"missing availability column {self.available_column!r}")
        if self.code_column not in frame.columns:
            raise ValueError(f"missing code column {self.code_column!r}")
        available = frame.filter(pl.col(self.available_column) <= decision_time)
        if available.is_empty():
            return UniverseResult(
                decision_time=decision_time, policy=self.policy, members=(), exclusions={}
            )

        exclusions: dict[str, str] = {}
        codes = sorted(available[self.code_column].unique().to_list())

        for code in codes:
            if not self.is_common_stock_code(code):
                exclusions[str(code)] = "non-common-stock-code"
                continue
            rows = available.filter(pl.col(self.code_column) == code)
            last = rows.sort(self.available_column).tail(1)
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

        if self.policy.require_historical_master:
            self._apply_master_intervals(available, exclusions, decision_time)

        members = tuple(c for c in codes if c not in exclusions)
        return UniverseResult(
            decision_time=decision_time,
            policy=self.policy,
            members=members,
            exclusions=exclusions,
        )

    def _apply_master_intervals(
        self,
        available: pl.DataFrame,
        exclusions: dict[str, str],
        decision_time: datetime,
    ) -> None:
        """Enforce listing/tradability intervals when the master columns exist.

        ``listed_from``/``delisted_on`` and ``tradable_from``/``tradable_to``
        mark the historical security-master intervals; an instrument is excluded
        if it is delisted, not yet listed, or outside its tradable window at the
        decision time. Missing master columns are tolerated only because the
        legacy panel is provisional; promotable datasets must supply them.
        """
        interval_columns = (
            "listed_from",
            "delisted_on",
            "tradable_from",
            "tradable_to",
        )
        if not any(c in available.columns for c in interval_columns):
            return
        all_codes = sorted(available[self.code_column].unique().to_list())
        for code in all_codes:
            if not self.is_common_stock_code(code):
                continue
            rows = available.filter(pl.col(self.code_column) == code)
            if rows.is_empty():
                continue
            last = rows.sort(self.available_column).tail(1).to_dicts()[0]
            listed_dt = _as_datetime(last.get("listed_from"))
            delisted_dt = _as_datetime(last.get("delisted_on"))
            tradable_from_dt = _as_datetime(last.get("tradable_from", listed_dt))
            tradable_to_dt = _as_datetime(last.get("tradable_to", delisted_dt))
            if listed_dt is not None and decision_time < listed_dt:
                exclusions[code] = "not-yet-listed"
                continue
            if delisted_dt is not None and decision_time >= delisted_dt:
                exclusions[code] = "delisted"
                continue
            if tradable_from_dt is not None and decision_time < tradable_from_dt:
                exclusions[code] = "not-tradable-yet"
                continue
            if tradable_to_dt is not None and decision_time >= tradable_to_dt:
                exclusions[code] = "tradability-expired"
                continue

    @staticmethod
    def participation_capacity(proposed_notional: float, adtv: float) -> float:
        """Capacity as proposed notional over trailing ADTV (structural ratio).

        A ratio above the policy participation limit means excess demand must
        stay in cash; no fixed KRW liquidity cutoff lives in strategy code.
        """
        if proposed_notional < 0:
            raise ValueError("proposed_notional must be non-negative")
        if adtv <= 0:
            raise ValueError("adtv must be positive")
        return proposed_notional / adtv


def _as_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()  # type: ignore[no-any-return]
    return datetime.combine(value, datetime.min.time())  # type: ignore[arg-type]
