# mypy: ignore-errors
"""Route objective, capital-plan, compounding, overlay contracts."""
from __future__ import annotations

from legacy.stocks.ml.contracts import CompoundAlphaStudySettings as _BaseCompoundAlphaStudySettings


class CompoundAlphaStudySettings(_BaseCompoundAlphaStudySettings):
    """Compound-study settings type owned by the growth-contract boundary."""

    __slots__ = ()

__all__ = ["CompoundAlphaStudySettings"]
