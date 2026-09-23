"""Single source of temporal and feature truth for a Korean swing-research release."""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, NonNegativeInt, PositiveFloat, PositiveInt, field_validator, model_validator

__all__ = [
    "OPENDART_DAILY_LIMIT",
    "CollectionBudget",
    "ResearchScope",
    "ScopeFeaturePolicy",
    "load_research_scope",
]

OPENDART_DAILY_LIMIT = 20000
_FISCAL_PATTERN = re.compile(r"\d{4}Q[1-4]")
_SCOPE_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_FISCAL_FLOOR = "2016Q1"


def _fiscal_key(period: str) -> int:
    year = int(period[:4])
    quarter = int(period[5])
    return year * 4 + quarter


class ScopeFeaturePolicy(BaseModel):
    """Validated feature availability rules for one immutable research scope."""

    price_lookback_sessions: PositiveInt
    fundamental_lookback_quarters: PositiveInt
    fundamental_fiscal_start: str
    investor_flow_enabled: bool
    industry_enabled: bool

    @field_validator("fundamental_fiscal_start")
    @classmethod
    def _check_fiscal_start(cls, value: str) -> str:
        if not _FISCAL_PATTERN.fullmatch(value):
            raise ValueError(f"invalid fundamental_fiscal_start {value!r}")
        if _fiscal_key(value) < _fiscal_key(_FISCAL_FLOOR):
            raise ValueError(f"invalid fundamental_fiscal_start {value!r}: floor is {_FISCAL_FLOOR}")
        return value


class CollectionBudget(BaseModel):
    """Provider request limits that preserve resumability and quota headroom."""

    dart_daily_budget: PositiveInt
    dart_batch_identities: PositiveInt
    dart_daily_reserve: NonNegativeInt = 400
    dart_request_min_interval_seconds: PositiveFloat = 1.0
    dart_max_workers: PositiveInt = 1

    @field_validator("dart_daily_budget")
    @classmethod
    def _check_daily_budget(cls, value: int) -> int:
        if value >= OPENDART_DAILY_LIMIT:
            raise ValueError(f"invalid dart_daily_budget {value!r}: must be below {OPENDART_DAILY_LIMIT}")
        return value

    @model_validator(mode="after")
    def _check_dart_reserve(self) -> CollectionBudget:
        if self.dart_daily_reserve >= self.dart_daily_budget:
            raise ValueError("dart_daily_reserve must be below dart_daily_budget")
        return self


class ResearchScope(BaseModel):
    """Single source of temporal and feature truth for a Korean swing-research release.

    The scope separates retained evidence, completed historical evaluation, and
    forward observation so no caller can turn an incomplete live period into a
    backtest sample.
    """

    scope_id: str
    evidence_start: date
    development_start: date
    development_end: date
    validation_start: date
    validation_end: date
    holdout_start: date
    holdout_end: date
    forward_start: date
    features: ScopeFeaturePolicy
    collection: CollectionBudget

    @field_validator("scope_id")
    @classmethod
    def _check_scope_id(cls, value: str) -> str:
        if "/" in value or "\\" in value or ".." in value:
            raise ValueError(f"invalid scope_id {value!r}")
        if any(ch.isspace() for ch in value):
            raise ValueError(f"invalid scope_id {value!r}")
        if not _SCOPE_PATTERN.fullmatch(value):
            raise ValueError(f"invalid scope_id {value!r}")
        return value

    @model_validator(mode="after")
    def _check_segments(self) -> ResearchScope:
        if self.evidence_start > self.development_start:
            raise ValueError("evidence_start must be on or before development_start")
        if self.development_start > self.development_end:
            raise ValueError("development_start must be on or before development_end")
        if self.validation_start > self.validation_end:
            raise ValueError("validation_start must be on or before validation_end")
        if self.holdout_start > self.holdout_end:
            raise ValueError("holdout_start must be on or before holdout_end")
        if self.validation_start != self.development_end + timedelta(days=1):
            raise ValueError("validation_start must immediately follow development_end")
        if self.holdout_start != self.validation_end + timedelta(days=1):
            raise ValueError("holdout_start must immediately follow validation_end")
        if self.forward_start != self.holdout_end + timedelta(days=1):
            raise ValueError("forward_start must immediately follow holdout_end")
        return self

    @property
    def content_hash(self) -> str:
        payload = {
            "collection": {
                "dart_batch_identities": self.collection.dart_batch_identities,
                "dart_daily_budget": self.collection.dart_daily_budget,
                "dart_daily_reserve": self.collection.dart_daily_reserve,
                "dart_max_workers": self.collection.dart_max_workers,
                "dart_request_min_interval_seconds": self.collection.dart_request_min_interval_seconds,
            },
            "development_end": self.development_end.isoformat(),
            "development_start": self.development_start.isoformat(),
            "evidence_start": self.evidence_start.isoformat(),
            "features": {
                "fundamental_fiscal_start": self.features.fundamental_fiscal_start,
                "fundamental_lookback_quarters": self.features.fundamental_lookback_quarters,
                "industry_enabled": self.features.industry_enabled,
                "investor_flow_enabled": self.features.investor_flow_enabled,
                "price_lookback_sessions": self.features.price_lookback_sessions,
            },
            "forward_start": self.forward_start.isoformat(),
            "holdout_end": self.holdout_end.isoformat(),
            "holdout_start": self.holdout_start.isoformat(),
            "scope_id": self.scope_id,
            "validation_end": self.validation_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def completed_start(self) -> date:
        return self.development_start

    @property
    def completed_end(self) -> date:
        return self.holdout_end

    def contains_evidence_date(self, value: date) -> bool:
        return value >= self.evidence_start

    def classify_completed_date(self, value: date) -> Literal["development", "validation", "holdout"]:
        if self.development_start <= value <= self.development_end:
            return "development"
        if self.validation_start <= value <= self.validation_end:
            return "validation"
        if self.holdout_start <= value <= self.holdout_end:
            return "holdout"
        raise ValueError(f"date {value.isoformat()} is outside completed segments")

    def require_fiscal_period(self, period: str) -> bool:
        if not _FISCAL_PATTERN.fullmatch(period):
            raise ValueError(f"invalid fiscal period {period!r}")
        return _fiscal_key(period) >= _fiscal_key(self.features.fundamental_fiscal_start)


def load_research_scope(path: Path) -> ResearchScope:
    """Load one TOML research scope and reject malformed or internally inconsistent policy."""
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    return ResearchScope.model_validate(raw)
