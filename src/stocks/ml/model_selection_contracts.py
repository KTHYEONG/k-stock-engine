# mypy: ignore-errors
"""Screening, model-family, preflight, fold/OOF contracts."""
from __future__ import annotations

from src.stocks.ml.contracts import ModelSelectionStudySettings as _BaseModelSelectionStudySettings


class ModelSelectionStudySettings(_BaseModelSelectionStudySettings):
    """Model-selection settings type owned by the selection-contract boundary."""

    __slots__ = ()

__all__ = ["ModelSelectionStudySettings"]
