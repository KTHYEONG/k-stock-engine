"""Thin net-alpha training facade delegating to orchestrator."""
# mypy: ignore-errors
# ruff: noqa: I001, F401, F403, F811
from __future__ import annotations

import sys

import legacy.stocks.ml.training_orchestrator as _orch
from legacy.stocks.ml.training_orchestrator import *  # noqa: F403
from legacy.stocks.ml.training_orchestrator import (  # noqa: F401
    _build_horizon_evidence,
    _plan_training_allocation,
    _replay_costs_batch,
)

train_net_alpha_model = _orch.train_net_alpha_model  # noqa: F811
sys.modules[__name__] = _orch
__all__ = ["_build_horizon_evidence", "_plan_training_allocation", "_replay_costs_batch", "train_net_alpha_model"]
