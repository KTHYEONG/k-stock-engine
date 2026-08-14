"""Net-alpha semantic feature registry and model-matrix builder.

The canonical production feature set is ``stock_net_alpha_v1``; ``v2``/``v3``
are implementation-step names and are never part of an external contract.
Every source carries exactly one economic role (``ALPHA``/``RISK``/
``LIQUIDITY``/``CONTROL``). Only ``ALPHA`` sources enter the learner matrix:
``RISK``/``LIQUIDITY``/``CONTROL`` are used for residualization, covariance,
tradeability, sizing, and audit only.

``build_model_features`` is the renamed and promoted successor of the legacy
``apply_v3_transforms``: it always returns the transformed frame together with
the model feature columns, so a trainer trains exclusively on the returned
frame and never on a mismatched raw frame.
"""
from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from src.stocks.data.feature_contracts import FeatureContractBook
from src.stocks.research.features import (
    STOCK_ALPHA_V2_ALLOWLIST,
    STOCK_ALPHA_V3_ROLES,
    apply_v3_transforms,
)

STOCK_NET_ALPHA_V1_FEATURE_SET = "stock_net_alpha_v1"

_ALPHA_ROLE = "ALPHA"
SESSION_COLUMN = "session"

# Reciprocal duplicate (``per``/``ep_ratio``, ``pbr``/``bp_ratio``) and raw
# total/flow duplicates (``net_purchase_total``) are excluded from the model
# registry; they never appear in the v1 canonical role map.
_RECIPROCAL_DUPLICATES = frozenset({"per", "pbr", "net_purchase_total"})


def stock_net_alpha_v1_roles() -> dict[str, str]:
    """Canonical ``(source -> role)`` map for ``stock_net_alpha_v1``.

    Inherits the semantic role declarations of the v3 experiment registry
    (same ALPHA/RISK/LIQUIDITY split) but drops reciprocal duplicates and raw
    totals so no two model columns are exact rank-equivalent or additive
    duplicates of the same economic flow.
    """
    return {
        source: role
        for source, role in STOCK_ALPHA_V3_ROLES.items()
        if source not in _RECIPROCAL_DUPLICATES
    }


def stock_net_alpha_v1_role_allowlist() -> tuple[tuple[str, str], ...]:
    """Ordered ``(source, role)`` pairs for the canonical v1 registry."""
    return tuple(
        (source, role)
        for source in STOCK_ALPHA_V2_ALLOWLIST
        if (role := stock_net_alpha_v1_roles().get(source)) is not None
    )


def stock_net_alpha_v1_allowlist() -> tuple[str, ...]:
    """Ordered ALPHA source allowlist for the canonical v1 feature set."""
    return tuple(
        source
        for source, role in stock_net_alpha_v1_role_allowlist()
        if role == _ALPHA_ROLE
    )


def stock_net_alpha_v1_semantic_contracts() -> tuple[dict[str, object], ...]:
    """Semantic per-feature contract declarations for the v1 role allowlist."""
    from src.stocks.research.features import (
        _v3_lookback_sessions,
    )

    contracts: list[dict[str, object]] = []
    for source, role in stock_net_alpha_v1_role_allowlist():
        formula_id = f"{STOCK_NET_ALPHA_V1_FEATURE_SET}:{source}:v1"
        contracts.append(
            {
                "name": source,
                "role": role,
                "source_field": source,
                "source_dataset_ids": ("base_panel",),
                "source_columns": (source,),
                "formula_id": formula_id,
                "lookback_sessions": _v3_lookback_sessions(source),
                "observation_rule": "session_close",
                "availability_rule": "next_session_open",
                "adjustment_basis": "split_adjusted",
                "null_policy": "retain_null",
                "stale_after_sessions": 0,
                "expected_frequency": "session",
            }
        )
    return tuple(contracts)


def stock_net_alpha_v1_contract_book() -> FeatureContractBook:
    """Feature contract book for the canonical v1 feature set."""
    from src.stocks.data.feature_contracts import semantic_feature_contract_book

    return semantic_feature_contract_book(
        STOCK_NET_ALPHA_V1_FEATURE_SET, stock_net_alpha_v1_semantic_contracts()
    )


def build_model_features(
    frame: pl.DataFrame,
    feature_roles: Mapping[str, str],
    *,
    session_column: str = SESSION_COLUMN,
    sector_column: str = "sector",
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Build the canonical learner matrix: role-aware ranks, sector ranks, missing flags.

    Only sources declared with role ``ALPHA`` enter the learner. Each canonical
    ALPHA source is ranked once within its session from its raw causal value;
    the sector-relative rank is derived from that rank (never re-ranked).
    Exact rank-equivalent families are reduced to one deterministic
    representative; a missing indicator is emitted only for a source with both
    missing and observed values in the frame, so no constant indicator is ever
    created.

    Returns:
        ``(transformed, model_feature_columns)`` where every model feature
        column is guaranteed present on the returned frame.

    Raises:
        ValueError: when the roles are invalid, declared sources are missing,
            non-finite, or a model feature is not ALPHA.
    """
    transformed, learner_columns = apply_v3_transforms(
        frame, feature_roles, session_column=session_column, sector_column=sector_column
    )
    missing = [c for c in learner_columns if c not in transformed.columns]
    if missing:
        raise ValueError(f"model feature columns missing from transformed frame: {missing}")
    return transformed, learner_columns
