"""Facade for model-selection: delegates to canonical study."""
# mypy: ignore-errors
# ruff: noqa: I001, F401, F403
from __future__ import annotations

import sys

import src.stocks.ml.model_selection_study as _study
from src.stocks.ml.model_selection_study import *  # noqa: F403

evaluate_model_selection_study = _study.evaluate_model_selection_study
sys.modules[__name__] = _study
__all__ = ["evaluate_model_selection_study"]
