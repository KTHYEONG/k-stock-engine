# mypy: ignore-errors
"""Training request, policy, portfolio/risk, execution contracts."""
from __future__ import annotations

from src.stocks.ml.contracts import NetAlphaTrainingRequest as _BaseNetAlphaTrainingRequest


class NetAlphaTrainingRequest(_BaseNetAlphaTrainingRequest):
    """Training request type owned by the training-contract boundary."""

    __slots__ = ()

__all__ = ["NetAlphaTrainingRequest"]
