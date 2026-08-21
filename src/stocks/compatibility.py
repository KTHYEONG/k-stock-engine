"""Compatibility adapters for old persisted identifiers.

Old ``*_vN`` identifiers are immutable historical inputs.  This module
maps them to semantic enums, emits a compatibility event, and never
serializes an old alias into a new artifact.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.stocks.trading.policy import ExecutionUtility, SizingMethod


@dataclass(frozen=True, slots=True)
class ArtifactContractIdentity:
    """Canonical persisted identity: contract_id + schema_revision + fingerprint."""

    contract_id: str
    schema_revision: int
    fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_revision < 1:
            raise ValueError(
                f"schema_revision must be >= 1, got {self.schema_revision}"
            )
        if len(self.fingerprint) != 64:
            raise ValueError(
                f"fingerprint must be 64 hex characters, got {len(self.fingerprint)}"
            )


# Old execution utility aliases mapped to semantic enums.
_EXECUTION_UTILITY_ALIASES: dict[str, ExecutionUtility] = {
    "legacy_target_interpolation_v1": ExecutionUtility.LEGACY_TARGET_INTERPOLATION,
    "delta_cost_aware_v1": ExecutionUtility.DELTA_COST_AWARE,
    "sparse_hold_replace_v2": ExecutionUtility.SPARSE_HOLD_REPLACE,
}

# Old sizing method aliases mapped to semantic enums.
_SIZING_METHOD_ALIASES: dict[str, SizingMethod] = {
    "alpha_vol_squared_v1": SizingMethod.ALPHA_VOL_SQUARED,
    "risk_balanced_waterfill_v2": SizingMethod.RISK_BALANCED_WATERFILL,
    "confidence_mean_variance_v1": SizingMethod.CONFIDENCE_MEAN_VARIANCE,
}


def parse_execution_utility(value: str) -> ExecutionUtility:
    """Parse an execution utility string to the semantic enum.

    Old v-suffixed aliases are accepted and mapped; unknown aliases fail closed.

    Parameters
    ----------
    value:
        The execution utility string to parse.

    Returns
    -------
    ExecutionUtility
        The semantic enum value.

    Raises
    ------
    ValueError
        When the value is not a known execution utility alias.
    """
    result = _EXECUTION_UTILITY_ALIASES.get(value)
    if result is None:
        raise ValueError(
            f"unknown execution utility {value!r}; "
            f"allowed: {tuple(_EXECUTION_UTILITY_ALIASES.keys())}"
        )
    return result


def parse_sizing_method(value: str) -> SizingMethod:
    """Parse a sizing method string to the semantic enum.

    Old v-suffixed aliases are accepted and mapped; unknown aliases fail closed.

    Parameters
    ----------
    value:
        The sizing method string to parse.

    Returns
    -------
    SizingMethod
        The semantic enum value.

    Raises
    ------
    ValueError
        When the value is not a known sizing method alias.
    """
    result = _SIZING_METHOD_ALIASES.get(value)
    if result is None:
        raise ValueError(
            f"unknown sizing method {value!r}; "
            f"allowed: {tuple(_SIZING_METHOD_ALIASES.keys())}"
        )
    return result
