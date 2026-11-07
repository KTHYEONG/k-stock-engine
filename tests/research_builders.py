"""Neutral research-config builders for tests; values are synthetic placeholders, not a tuned strategy."""

from __future__ import annotations

from typing import Any

from src.research.model import ScorerConfig
from src.research.policy import TrendCashPolicy, UniverseRule

__all__ = [
    "NEUTRAL_POLICY",
    "NEUTRAL_SCORER",
    "NEUTRAL_UNIVERSE",
    "policy_toml_lines",
    "scorer_config",
    "scorer_toml_lines",
    "trend_cash_policy",
]

NEUTRAL_UNIVERSE = UniverseRule(min_adtv20_krw=300_000_000, min_price_krw=2_000)

NEUTRAL_SCORER: dict[str, Any] = {
    "horizons": (5, 10),
    "num_boost_round": 120,
    "learning_rate": 0.05,
    "num_leaves": 23,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 2,
    "lambda_l2": 20.0,
    "seed": 11,
    "min_cross_section": 40,
    "winsor_low_pct": 2.0,
    "winsor_high_pct": 98.0,
    "purge_extra_sessions": 3,
    "min_train_rows": 15_000,
    "first_test_year": 2018,
}

NEUTRAL_POLICY: dict[str, Any] = {
    "keep_rank_multiple": 2.5,
    "rebalance_every_sessions": 5,
    "universe": NEUTRAL_UNIVERSE,
    "min_units_per_slot": 2,
}


def scorer_config(**overrides: Any) -> ScorerConfig:
    """``ScorerConfig`` from the neutral placeholders with ``overrides`` applied."""
    return ScorerConfig(**{**NEUTRAL_SCORER, **overrides})


def trend_cash_policy(**overrides: Any) -> TrendCashPolicy:
    """``TrendCashPolicy`` from the neutral placeholders with ``overrides`` applied."""
    return TrendCashPolicy(**{**NEUTRAL_POLICY, **overrides})


def _toml_lines(values: dict[str, Any]) -> str:
    def _scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        return repr(value)

    return "".join(
        f"{key} = [{', '.join(_scalar(item) for item in value)}]\n" if isinstance(value, (list, tuple))
        else f"{key} = {_scalar(value)}\n"
        for key, value in values.items()
    )


def scorer_toml_lines(**overrides: Any) -> str:
    """``[scorer]`` body lines (no header) from the neutral placeholders with ``overrides`` applied."""
    return _toml_lines({**NEUTRAL_SCORER, **overrides})


def policy_toml_lines(**overrides: Any) -> str:
    """``[policy]`` body lines (no header, no universe subtable) from the neutral placeholders with ``overrides``."""
    scalars = {key: value for key, value in NEUTRAL_POLICY.items() if key != "universe"}
    return _toml_lines({**scalars, **overrides})
