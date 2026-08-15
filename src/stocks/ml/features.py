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

Feature transforms are split into a label-free schema fit and a deterministic
apply: ``fit_model_feature_schema`` freezes the rank-equivalent representatives
and missing-indicator decisions from a supplied fitting frame (the pre-holdout
partition) and ``apply_model_feature_schema`` applies that frozen schema to any
frame, so a holdout value or null mutation can never change the pre-holdout
learner matrix or schema fingerprint.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import polars as pl

from src.stocks.data.feature_contracts import FeatureContractBook
from src.stocks.research.features import (
    _ALPHA_ROLE,
    STOCK_ALPHA_V2_ALLOWLIST,
    STOCK_ALPHA_V3_ROLES,
    _rank_equivalent_cluster,
    _reject_target_columns,
    _validate_v3_roles,
)

STOCK_NET_ALPHA_V1_FEATURE_SET = "stock_net_alpha_v1"

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


@dataclass(frozen=True, slots=True)
class FeatureTransformSchema:
    """Immutable label-free feature-transform schema frozen from a fitting frame.

    ``representative_sources`` is the deterministic rank-equivalent-clustered
    ALPHA source order, ``missing_sources`` the sources that carry both missing
    and observed values in the fitting frame (so a missing indicator is
    emitted), and ``learner_columns`` the exact ordered model-matrix columns the
    schema emits. ``fingerprint`` is a JSON-safe deterministic hash of every
    schema decision; it must be unchanged by any holdout-only mutation.
    """

    representative_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]
    source_order: tuple[str, ...]
    learner_columns: tuple[str, ...]
    session_column: str
    sector_column: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.learner_columns:
            raise ValueError("feature transform schema must emit learner columns")
        if len(set(self.learner_columns)) != len(self.learner_columns):
            raise ValueError("learner_columns must be unique")
        for source in self.representative_sources:
            if source not in self.source_order:
                raise ValueError(
                    f"representative source {source!r} missing from source_order"
                )

    def to_json(self) -> dict[str, object]:
        return {
            "representative_sources": list(self.representative_sources),
            "missing_sources": list(self.missing_sources),
            "source_order": list(self.source_order),
            "learner_columns": list(self.learner_columns),
            "session_column": self.session_column,
            "sector_column": self.sector_column,
        }


def _schema_fingerprint(schema: FeatureTransformSchema) -> str:
    payload = json.dumps(
        schema.to_json(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def fit_model_feature_schema(
    frame: pl.DataFrame,
    feature_roles: Mapping[str, str],
    *,
    session_column: str = SESSION_COLUMN,
    sector_column: str = "sector",
) -> FeatureTransformSchema:
    """Fit every label-free schema decision from ``frame`` only.

    The rank-equivalent representatives and the missing-indicator set are frozen
    exclusively from the supplied fitting frame (the pre-holdout partition).
    Mutating any other partition (holdout) must leave the returned schema
    unchanged. Raises ``ValueError`` on invalid roles, missing declared sources,
    non-finite values, or a missing session/sector column.
    """
    _validate_v3_roles(feature_roles)
    if session_column not in frame.columns:
        raise ValueError(f"frame must carry {session_column!r}")
    if sector_column not in frame.columns:
        raise ValueError(f"frame must carry {sector_column!r} for sector-relative rank")
    sources = tuple(feature_roles)
    missing = [c for c in sources if c not in frame.columns]
    if missing:
        raise ValueError(f"v3 feature sources missing from frame: {missing}")
    _reject_target_columns(frame, sources)
    for column in sources:
        non_finite = frame.filter(
            pl.col(column).is_not_null() & ~pl.col(column).is_finite()
        )
        if not non_finite.is_empty():
            raise ValueError(f"non-finite value in v3 feature source {column}")
    alpha_sources = tuple(c for c in sources if feature_roles[c] == _ALPHA_ROLE)
    representative = _rank_equivalent_cluster(frame, alpha_sources, session_column)
    missing_sources = tuple(
        source
        for source in representative
        if int(frame[source].is_null().sum()) > 0
        and int(frame[source].is_not_null().sum()) > 0
    )
    learner_columns: list[str] = []
    for column in representative:
        learner_columns.append(f"{column}__rank")
        learner_columns.append(f"{column}__sector_rank")
        if column in missing_sources:
            learner_columns.append(f"{column}__missing")
    schema = FeatureTransformSchema(
        representative_sources=representative,
        missing_sources=missing_sources,
        source_order=sources,
        learner_columns=tuple(learner_columns),
        session_column=session_column,
        sector_column=sector_column,
        fingerprint="",
    )
    return FeatureTransformSchema(
        representative_sources=representative,
        missing_sources=missing_sources,
        source_order=sources,
        learner_columns=tuple(learner_columns),
        session_column=session_column,
        sector_column=sector_column,
        fingerprint=_schema_fingerprint(schema),
    )


def apply_model_feature_schema(
    frame: pl.DataFrame,
    schema: FeatureTransformSchema,
) -> pl.DataFrame:
    """Apply the frozen feature schema to any frame, holdout included.

    Within-session ranks, sector-demeaned ranks, and missing indicators are
    emitted from ``schema`` (never refit) so the pre-holdout learner matrix and
    the holdout transform are deterministic functions of the same frozen schema.
    """
    if schema.session_column not in frame.columns:
        raise ValueError(f"frame must carry {schema.session_column!r}")
    if schema.sector_column not in frame.columns:
        raise ValueError(f"frame must carry {schema.sector_column!r}")
    missing = [c for c in schema.source_order if c not in frame.columns]
    if missing:
        raise ValueError(f"v3 feature sources missing from frame: {missing}")
    _reject_target_columns(frame, tuple(schema.source_order))
    for column in schema.source_order:
        non_finite = frame.filter(
            pl.col(column).is_not_null() & ~pl.col(column).is_finite()
        )
        if not non_finite.is_empty():
            raise ValueError(f"non-finite value in v3 feature source {column}")

    rank_exprs: list[pl.Expr] = []
    for index, column in enumerate(schema.representative_sources):
        within = pl.col(column).count().over(schema.session_column)
        rank = (
            (pl.col(column).rank("average").over(schema.session_column) - 1.0)
            / (within - 1.0)
        )
        rank_exprs.append(
            rank.fill_null(0.5).cast(pl.Float32).alias(f"__v3_rank_{index}")
        )
    ranked = frame.with_columns(rank_exprs)

    sector_exprs: list[pl.Expr] = []
    for index, _ in enumerate(schema.representative_sources):
        sector_mean = pl.col(f"__v3_rank_{index}").mean().over(
            [schema.session_column, schema.sector_column]
        )
        sector_exprs.append(
            ((pl.col(f"__v3_rank_{index}") - sector_mean).cast(pl.Float32)).alias(
                f"__v3_sector_rank_{index}"
            )
        )
    expanded = ranked.with_columns(sector_exprs)

    out_exprs: list[pl.Expr] = []
    for column in expanded.columns:
        if column.startswith(("__v3_rank_", "__v3_sector_rank_")):
            continue
        if column in schema.source_order:
            continue
        out_exprs.append(pl.col(column))
    for index, column in enumerate(schema.representative_sources):
        missing_indicator = (
            pl.when(pl.col(column).is_null()).then(1.0).otherwise(0.0).cast(pl.Float32)
        )
        out_exprs.append(
            pl.col(f"__v3_rank_{index}").alias(f"{column}__rank")
        )
        out_exprs.append(
            pl.col(f"__v3_sector_rank_{index}").alias(f"{column}__sector_rank")
        )
        if column in schema.missing_sources:
            out_exprs.append(missing_indicator.alias(f"{column}__missing"))
    return expanded.select(out_exprs)


def build_model_features(
    frame: pl.DataFrame,
    feature_roles: Mapping[str, str],
    *,
    session_column: str = SESSION_COLUMN,
    sector_column: str = "sector",
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Build the canonical learner matrix: role-aware ranks, sector ranks, missing flags.

    Backward-compatible wrapper that fits the label-free schema on ``frame`` and
    applies it to the same frame. Only sources declared with role ``ALPHA``
    enter the learner; each canonical ALPHA source is ranked once within its
    session and the sector-relative rank is derived from that rank. Exact
    rank-equivalent families are reduced to one deterministic representative and
    a missing indicator is emitted only for a source with both missing and
    observed values in ``frame``.

    Returns:
        ``(transformed, model_feature_columns)`` where every model feature
        column is guaranteed present on the returned frame.

    Raises:
        ValueError: when the roles are invalid, declared sources are missing,
            non-finite, or a model feature is not ALPHA.
    """
    schema = fit_model_feature_schema(
        frame, feature_roles, session_column=session_column, sector_column=sector_column
    )
    transformed = apply_model_feature_schema(frame, schema)
    missing = [c for c in schema.learner_columns if c not in transformed.columns]
    if missing:
        raise ValueError(f"model feature columns missing from transformed frame: {missing}")
    return transformed, schema.learner_columns
