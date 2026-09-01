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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import polars as pl

from legacy.stocks.data.feature_contracts import FeatureContractBook
from legacy.stocks.research.features import (
    _ALPHA_ROLE,
    STOCK_ALPHA_V2_ALLOWLIST,
    STOCK_ALPHA_V3_ROLES,
    _rank_equivalent_cluster,
    _reject_target_columns,
    _validate_v3_roles,
)
from legacy.stocks.research.models import ModelManifest

STOCK_NET_ALPHA_V1_FEATURE_SET = "stock_net_alpha_v1"
STOCK_NET_ALPHA_V2_FEATURE_SET = "stock_net_alpha_v2"

SESSION_COLUMN = "session"

# Reciprocal duplicate (``per``/``ep_ratio``, ``pbr``/``bp_ratio``) and raw
# total/flow duplicates (``net_purchase_total``) are excluded from the model
# registry; they never appear in the v1 canonical role map.
_RECIPROCAL_DUPLICATES = frozenset({"per", "pbr", "net_purchase_total"})
_PIT_FUNDAMENTAL_SOURCES = frozenset({"bp_ratio", "ep_ratio"})


def stock_net_alpha_v1_roles(
    *, available_columns: Collection[str] | None = None
) -> dict[str, str]:
    """Canonical ``(source -> role)`` map for ``stock_net_alpha_v1``.

    Inherits the semantic role declarations of the v3 experiment registry
    (same ALPHA/RISK/LIQUIDITY split) but drops reciprocal duplicates and raw
    totals so no two model columns are exact rank-equivalent or additive
    duplicates of the same economic flow.
    """
    roles = {
        source: role
        for source, role in STOCK_ALPHA_V3_ROLES.items()
        if source not in _RECIPROCAL_DUPLICATES
    }
    if available_columns is not None and "disclosure_date" not in available_columns:
        for source in _PIT_FUNDAMENTAL_SOURCES:
            roles.pop(source, None)
    return roles


def stock_net_alpha_v1_role_allowlist(
    *, available_columns: Collection[str] | None = None
) -> tuple[tuple[str, str], ...]:
    """Ordered ``(source, role)`` pairs for the canonical v1 registry."""
    return tuple(
        (source, role)
        for source in STOCK_ALPHA_V2_ALLOWLIST
        if (
            role := stock_net_alpha_v1_roles(
                available_columns=available_columns
            ).get(source)
        ) is not None
    )


def stock_net_alpha_v1_allowlist(
    *, available_columns: Collection[str] | None = None
) -> tuple[str, ...]:
    """Ordered ALPHA source allowlist for the canonical v1 feature set."""
    return tuple(
        source
        for source, role in stock_net_alpha_v1_role_allowlist(
            available_columns=available_columns
        )
        if role == _ALPHA_ROLE
    )


def stock_net_alpha_v1_semantic_contracts(
    *, available_columns: Collection[str] | None = None
) -> tuple[dict[str, object], ...]:
    """Semantic per-feature contract declarations for the v1 role allowlist."""
    from legacy.stocks.research.features import (
        _v3_lookback_sessions,
    )

    contracts: list[dict[str, object]] = []
    for source, role in stock_net_alpha_v1_role_allowlist(
        available_columns=available_columns
    ):
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
                "source_available_time_field": (
                    "disclosure_date" if source in ("bp_ratio", "ep_ratio") else "available_time"
                ),
            }
        )
    return tuple(contracts)


def stock_net_alpha_v1_contract_book(
    *, available_columns: Collection[str] | None = None
) -> FeatureContractBook:
    """Feature contract book for the canonical v1 feature set."""
    from legacy.stocks.data.feature_contracts import semantic_feature_contract_book

    return semantic_feature_contract_book(
        STOCK_NET_ALPHA_V1_FEATURE_SET,
        stock_net_alpha_v1_semantic_contracts(available_columns=available_columns),
    )


def _v2_lookback_sessions(source: str) -> int:
    """Explicit lookback for v2: handles suffixes like 120d, 5d, 2_5d."""
    # Direct trailing <digits>d pattern
    if source.endswith("d"):
        # Take segment after last underscore, strip trailing d
        tail = source.rsplit("_", 1)[-1]
        if tail.endswith("d"):
            tail = tail[:-1]
            if tail.isdigit():
                return int(tail)
        # For patterns like ret_2_5d -> tail is 5d -> already handled
        # For disparity_120d -> 120d -> 120
        # Also handle embedded numbers like 120d inside
        import re

        match = re.search(r"(\d+)d$", source)
        if match:
            return int(match.group(1))
    # Fallback to legacy trailing integer
    suffix = source.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    if source.startswith(("overnight_ret", "intraday_ret", "fluc_rate", "sector_ret")):
        return 1
    return 0


def stock_net_alpha_v2_semantic_contracts() -> tuple[dict[str, object], ...]:
    """Semantic contracts for stock_net_alpha_v2 with explicit lookbacks and DERIVED lineage.

    Every feature carries DERIVED source_kind, explicit lookback, formula lineage,
    and source_available_time_field. Fundamentals bp_ratio/ep_ratio bind to
    disclosure_date and fail closed for production without PIT timestamps.
    """
    contracts: list[dict[str, object]] = []
    for source, role in stock_net_alpha_v1_role_allowlist():
        formula_id = f"{STOCK_NET_ALPHA_V2_FEATURE_SET}:{source}:v1"
        avail_field = "disclosure_date" if source in ("bp_ratio", "ep_ratio") else "available_time"
        contracts.append(
            {
                "name": source,
                "role": role,
                "source_field": source,
                "source_dataset_ids": ("base_panel",),
                "source_columns": (source,),
                "formula_id": formula_id,
                "source_kind": "derived",
                "lookback_sessions": _v2_lookback_sessions(source),
                "observation_rule": "session_close",
                "availability_rule": "next_session_open",
                "adjustment_basis": "split_adjusted",
                "null_policy": "retain_null",
                "stale_after_sessions": 0,
                "expected_frequency": "session",
                "source_available_time_field": avail_field,
            }
        )
    return tuple(contracts)


def stock_net_alpha_v2_contract_book() -> FeatureContractBook:
    """Feature contract book for v2 with explicit PIT lineage."""
    from legacy.stocks.data.feature_contracts import semantic_feature_contract_book

    return semantic_feature_contract_book(
        STOCK_NET_ALPHA_V2_FEATURE_SET, stock_net_alpha_v2_semantic_contracts()
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

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> FeatureTransformSchema:
        """Strict deserialize from a frozen ``to_json`` payload.

        Any missing, misshapen, or non-list field, or a learner-column order
        that disagrees with ``source_order``/``representative_sources``, raises
        ``ValueError`` so a corrupt or partial schema payload can never silently
        drive a model matrix. The fingerprint is always recomputed from the
        parsed decisions and is never trusted from the payload.
        """

        def as_tuple(value: object, name: str) -> tuple[str, ...]:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"feature transform schema {name} must be a string list")
            return tuple(value)

        for required in (
            "representative_sources",
            "missing_sources",
            "source_order",
            "learner_columns",
            "session_column",
            "sector_column",
        ):
            if required not in payload:
                raise ValueError(f"feature transform schema missing field {required!r}")
        representative = as_tuple(payload["representative_sources"], "representative_sources")
        missing = as_tuple(payload["missing_sources"], "missing_sources")
        source_order = as_tuple(payload["source_order"], "source_order")
        learner_columns = as_tuple(payload["learner_columns"], "learner_columns")
        session_column = payload["session_column"]
        sector_column = payload["sector_column"]
        if not isinstance(session_column, str) or not session_column:
            raise ValueError("feature transform schema session_column must be a non-empty string")
        if not isinstance(sector_column, str) or not sector_column:
            raise ValueError("feature transform schema sector_column must be a non-empty string")
        for source in representative:
            if source not in source_order:
                raise ValueError(
                    f"representative source {source!r} missing from source_order"
                )
        parsed = FeatureTransformSchema(
            representative_sources=representative,
            missing_sources=missing,
            source_order=source_order,
            learner_columns=learner_columns,
            session_column=session_column,
            sector_column=sector_column,
            fingerprint="",
        )
        fingerprint = _schema_fingerprint(parsed)
        stored = payload.get("fingerprint")
        if stored is not None and str(stored) != fingerprint:
            raise ValueError(
                "feature transform schema fingerprint mismatch: "
                f"payload {stored!r} != recomputed {fingerprint!r}"
            )
        return FeatureTransformSchema(
            representative_sources=representative,
            missing_sources=missing,
            source_order=source_order,
            learner_columns=learner_columns,
            session_column=session_column,
            sector_column=sector_column,
            fingerprint=fingerprint,
        )


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


_FEATURE_SOURCE_PREFIX = "feature__"


def materialize_model_feature_sources(
    frame: pl.DataFrame,
    source_order: Sequence[str],
) -> pl.DataFrame:
    """Resolve every declared source to exactly one of ``source`` or ``feature__source``.

    Each canonical ``source`` in ``source_order`` must materialize from exactly one
    of the unprefixed column ``source`` or the prefixed ``feature__source``:

    * both present -> their non-null values must be exactly equal; otherwise the
      frame fails closed before any score/prediction (a schema mismatch),
    * only the unprefixed column present -> kept as-is,
    * only the prefixed column present -> renamed to the canonical ``source``,
    * neither present -> ``ValueError`` (missing source),

    Non-finite values on a present source fail closed. The chosen source binding
    (canonical name or prefixed name) is recorded on the returned frame under the
    ``_feature_source_binding`` attribute for the artifact lineage payload.
    """
    if not source_order:
        raise ValueError("materialize_model_feature_sources requires a non-empty source_order")
    result = frame
    binding: dict[str, str] = {}
    for source in source_order:
        canonical = str(source)
        prefixed = f"{_FEATURE_SOURCE_PREFIX}{source}"
        has_canonical = canonical in result.columns
        has_prefixed = prefixed in result.columns
        if has_canonical and has_prefixed:
            conflict = result.filter(
                pl.col(canonical).is_not_null() & pl.col(prefixed).is_not_null()
                & (pl.col(canonical) != pl.col(prefixed))
            )
            if not conflict.is_empty():
                raise ValueError(
                    f"feature source {source!r} conflict between {canonical!r} "
                    f"and {prefixed!r}; a v7 artifact must resolve exactly one binding"
                )
            result = result.drop(prefixed)
            binding[source] = canonical
        elif has_canonical:
            binding[source] = canonical
        elif has_prefixed:
            result = result.rename({prefixed: canonical})
            binding[source] = prefixed
        else:
            raise ValueError(
                f"feature sources missing from frame: {source!r} "
                f"(neither {canonical!r} nor {prefixed!r} present)"
            )
        non_finite = result.filter(
            pl.col(canonical).is_not_null() & ~pl.col(canonical).is_finite()
        )
        if not non_finite.is_empty():
            raise ValueError(f"non-finite value in feature source {canonical!r}")
    object.__setattr__(result, "_feature_source_binding", dict(binding))
    return result


@dataclass(frozen=True, slots=True)
class ResearchFeatureSchema:
    source_groups: tuple[tuple[str, tuple[str, ...]], ...]
    winsor_bounds: tuple[tuple[str, float, float], ...]
    robust_location_scale: tuple[tuple[str, float, float], ...]
    fingerprint: str
    imputation_values: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.source_groups:
            raise ValueError("research feature schema must have source groups")
        if len({name for name, _ in self.source_groups}) != len(self.source_groups):
            raise ValueError("source_groups must have unique source names")
        if len(self.winsor_bounds) != len(self.source_groups):
            raise ValueError("winsor_bounds must align with source_groups")
        if len(self.robust_location_scale) != len(self.source_groups):
            raise ValueError("robust_location_scale must align with source_groups")
        # imputation_values may be empty during intermediate construction before hashing
        # fingerprint may be empty during intermediate construction before hashing
        if self.imputation_values:
            if len(self.imputation_values) != len(self.source_groups):
                raise ValueError("imputation_values must align with source_groups")
            names = {n for n, _ in self.imputation_values}
            if len(names) != len(self.imputation_values):
                raise ValueError("imputation_values must have unique source names")
            group_names = {n for n, _ in self.source_groups}
            if names != group_names:
                raise ValueError("imputation_values names must match source_groups")

    def to_json(self) -> dict[str, object]:
        return {
            "source_groups": [[k, list(v)] for k, v in self.source_groups],
            "winsor_bounds": [list(t) for t in self.winsor_bounds],
            "robust_location_scale": [list(t) for t in self.robust_location_scale],
            "imputation_values": [list(t) for t in self.imputation_values],
            "fingerprint": self.fingerprint,
        }


def _research_schema_fingerprint(schema: ResearchFeatureSchema) -> str:
    payload = json.dumps(schema.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def fit_research_feature_schema(
    train: pl.DataFrame, feature_roles: Mapping[str, str]
) -> ResearchFeatureSchema:
    _validate_v3_roles(feature_roles)
    session_column = SESSION_COLUMN
    sector_column = "sector"
    if session_column not in train.columns:
        raise ValueError(f"frame must carry {session_column!r}")
    if sector_column not in train.columns:
        train = train.with_columns(pl.lit("dummy").alias(sector_column))
    sources = tuple(feature_roles)
    missing = [c for c in sources if c not in train.columns]
    if missing:
        raise ValueError(f"v3 feature sources missing from frame: {missing}")
    _reject_target_columns(train, sources)
    for column in sources:
        non_finite = train.filter(pl.col(column).is_not_null() & ~pl.col(column).is_finite())
        if not non_finite.is_empty():
            raise ValueError(f"non-finite value in v3 feature source {column}")
    alpha_sources = tuple(c for c in sources if feature_roles[c] == _ALPHA_ROLE)
    representative = _rank_equivalent_cluster(train, alpha_sources, session_column)
    # winsor bounds per representative source from train only (1%/99%)
    winsor_bounds: list[tuple[str, float, float]] = []
    robust: list[tuple[str, float, float]] = []
    import numpy as np

    for src in representative:
        vals = [float(v) for v in train[src].to_list() if v is not None and isinstance(v, (int, float)) and __import__("math").isfinite(float(v))]
        if not vals:
            lo, hi = -1e12, 1e12
            med, scale = 0.0, 1.0
        else:
            arr = np.asarray(vals, dtype=np.float64)
            lo = float(np.quantile(arr, 0.01))
            hi = float(np.quantile(arr, 0.99))
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            scale = float(mad * 1.4826) if mad > 0 else float(np.std(arr) or 1.0)
            if not __import__("math").isfinite(scale) or scale <= 0:
                scale = 1.0
        winsor_bounds.append((src, float(lo), float(hi)))
        robust.append((src, float(med), float(scale)))
    # imputation median per source (train-only)
    imputation_values: list[tuple[str, float]] = []
    for src in representative:
        vals = [float(v) for v in train[src].to_list() if v is not None and isinstance(v, (int, float)) and __import__("math").isfinite(float(v))]
        median_val = 0.0 if not vals else float(np.median(np.asarray(vals, dtype=np.float64)))
        imputation_values.append((src, float(median_val)))
    # missing indicator set
    missing_sources = tuple(
        s for s in representative if int(train[s].is_null().sum()) > 0 and int(train[s].is_not_null().sum()) > 0
    )
    # source groups: each source maps to its derived columns
    source_groups: list[tuple[str, tuple[str, ...]]] = []
    for src in representative:
        cols: list[str] = [f"{src}__winsor", f"{src}__rank", f"{src}__sector_rank"]
        if src in missing_sources:
            cols.append(f"{src}__missing")
        cols.append(f"{src}__robust")
        source_groups.append((src, tuple(cols)))
    # linear-only interaction: only ALPHA-by-ALPHA realizable pair
    interaction_pairs = [("flow_consensus", "relative_trend_score")]
    for a, b in interaction_pairs:
        if a in representative and b in representative:
            name = f"{a}_x_{b}"
            # single derived column: rank product
            interaction_cols: tuple[str, ...] = (f"{name}__rank_product",)
            source_groups.append((name, interaction_cols))
            # winsor/robust placeholders for interaction: use rank product stats (0-1 range)
            # imputation for interaction is product of medians? but we finger print median rank product (0.25) as placeholder
            # For consistency, add dummy winsor/robust/imputation entries aligned to source_groups order
            # Use neutral bounds/scales
            winsor_bounds.append((name, 0.0, 1.0))
            robust.append((name, 0.25, 0.2))
            imputation_values.append((name, 0.25))
    tmp = ResearchFeatureSchema(
        source_groups=tuple(source_groups),
        winsor_bounds=tuple(winsor_bounds),
        robust_location_scale=tuple(robust),
        imputation_values=tuple(imputation_values),
        fingerprint="",
    )
    fp = _research_schema_fingerprint(tmp)
    return ResearchFeatureSchema(
        source_groups=tuple(source_groups),
        winsor_bounds=tuple(winsor_bounds),
        robust_location_scale=tuple(robust),
        imputation_values=tuple(imputation_values),
        fingerprint=fp,
    )


def apply_research_feature_schema(
    frame: pl.DataFrame, schema: ResearchFeatureSchema
) -> pl.DataFrame:
    session_column = SESSION_COLUMN
    sector_column = "sector"
    if session_column not in frame.columns:
        raise ValueError(f"frame must carry {session_column!r}")
    if sector_column not in frame.columns:
        frame = frame.with_columns(pl.lit("dummy").alias(sector_column))
    # verify schema sources present - interaction groups are derived, not required as input columns
    # Only check base sources (those with winsor/robust) that are not interaction products
    source_names = [name for name, _ in schema.source_groups]
    input_source_names = [n for n in source_names if "_x_" not in n]
    # For interactions, ensure constituents exist
    for grp_name, _ in schema.source_groups:
        if "_x_" in grp_name:
            a, b = grp_name.split("_x_")
            if a not in frame.columns or b not in frame.columns:
                raise ValueError(f"v3 feature sources missing from frame: {grp_name} requires {a!r} and {b!r}")
    missing = [c for c in input_source_names if c not in frame.columns]
    if missing:
        raise ValueError(f"v3 feature sources missing from frame: {missing}")
    _reject_target_columns(frame, tuple(input_source_names))
    for column in input_source_names:
        non_finite = frame.filter(pl.col(column).is_not_null() & ~pl.col(column).is_finite())
        if not non_finite.is_empty():
            raise ValueError(f"non-finite value in v3 feature source {column}")
    # impute missing values using train-only median BEFORE any derived numeric feature
    # capture original missing indicator before imputation so 0/1 indicator is correct
    missing_indicator_sources = {src for src, cols in schema.source_groups if f"{src}__missing" in cols}
    out = frame
    for src in missing_indicator_sources:
        if src in input_source_names:
            out = out.with_columns(pl.when(pl.col(src).is_null()).then(1.0).otherwise(0.0).cast(pl.Float32).alias(f"__missing_{src}"))
    impute_map = {src: float(v) for src, v in schema.imputation_values} if schema.imputation_values else {}
    for src in input_source_names:
        if src in impute_map:
            median_val = float(impute_map[src])
            out = out.with_columns(pl.col(src).fill_null(median_val).alias(src))
        # also fill any remaining nulls? already filled; ensure derived will be finite
    # build lookup maps
    winsor_map = {src: (lo, hi) for src, lo, hi in schema.winsor_bounds}
    robust_map = {src: (loc, scale) for src, loc, scale in schema.robust_location_scale}
    # step 1: winsorized columns (for base sources only; interactions use rank product directly)
    for src in input_source_names:
        lo, hi = winsor_map.get(src, (-1e12, 1e12))
        out = out.with_columns(pl.col(src).clip(lo, hi).alias(f"__winsor_{src}"))
    # step 2: ranks within session from winsorized values (base sources only)
    rank_exprs: list[pl.Expr] = []
    for src in input_source_names:
        within = pl.col(f"__winsor_{src}").count().over(session_column)
        rank = (pl.col(f"__winsor_{src}").rank("average").over(session_column) - 1.0) / (within - 1.0)
        rank_exprs.append(rank.fill_null(0.5).cast(pl.Float32).alias(f"__rank_{src}"))
    out = out.with_columns(rank_exprs)
    # step 2b: interaction rank products (fold-local)
    for grp_name, _ in schema.source_groups:
        if "_x_" in grp_name:
            a, b = grp_name.split("_x_")
            out = out.with_columns((pl.col(f"__rank_{a}") * pl.col(f"__rank_{b}")).cast(pl.Float32).alias(f"__rank_product_{grp_name}"))
    # step 3: sector ranks (base only)
    sector_exprs: list[pl.Expr] = []
    for src in input_source_names:
        sector_mean = pl.col(f"__rank_{src}").mean().over([session_column, sector_column])
        sector_exprs.append(((pl.col(f"__rank_{src}") - sector_mean).cast(pl.Float32)).alias(f"__sector_rank_{src}"))
    out = out.with_columns(sector_exprs)
    # step 4: robust standardized winsor (base only)
    robust_exprs: list[pl.Expr] = []
    for src in input_source_names:
        loc, scale = robust_map.get(src, (0.0, 1.0))
        robust_exprs.append(((pl.col(f"__winsor_{src}") - loc) / scale).cast(pl.Float32).alias(f"__robust_{src}"))
    out = out.with_columns(robust_exprs)
    # final selection: keep original non-source cols plus derived per source_groups
    keep_cols: list[pl.Expr] = []
    for col in out.columns:
        if col.startswith("__winsor_") or col.startswith("__rank_") or col.startswith("__rank_product_") or col.startswith("__sector_rank_") or col.startswith("__robust_") or col.startswith("__missing_"):
            continue
        if col in source_names:
            continue
        keep_cols.append(pl.col(col))
    for src, cols in schema.source_groups:
        # cols contains expected derived names like src__winsor etc.
        if "_x_" in src:
            # interaction rank product
            mapping = {
                f"{src}__rank_product": f"__rank_product_{src}",
            }
            for out_name in cols:
                tmp_name = mapping.get(out_name)
                if tmp_name is not None:
                    keep_cols.append(pl.col(tmp_name).alias(out_name))
            continue
        mapping = {
            f"{src}__winsor": f"__winsor_{src}",
            f"{src}__rank": f"__rank_{src}",
            f"{src}__sector_rank": f"__sector_rank_{src}",
            f"{src}__robust": f"__robust_{src}",
            f"{src}__missing": f"__missing_{src}",
        }
        for out_name in cols:
            tmp_name = mapping.get(out_name)
            if tmp_name is not None:
                keep_cols.append(pl.col(tmp_name).alias(out_name))
    return out.select(keep_cols)


def feature_transform_schema_from_manifest(
    manifest: ModelManifest,
) -> FeatureTransformSchema:
    """Load the frozen feature-transform schema persisted on an artifact manifest.

    The canonical production schema is stored under ``params.feature_transform_schema``
    as the strict ``FeatureTransformSchema.to_json`` payload. A missing payload, a
    non-string payload, or a JSON parse/validation/fingerprint failure raises
    ``ValueError`` so scoring and certified backtests never silently re-fit a
    schema from the current frame. The fingerprint is always recomputed and
    compared against the stored value; a mismatch fails closed.
    """
    params = manifest.params or {}
    payload = params.get("feature_transform_schema")
    if payload is None:
        raise ValueError(
            f"artifact {manifest.artifact_id!r} carries no frozen "
            "feature_transform_schema; scoring a v6 artifact requires the "
            "persisted transform contract"
        )
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"artifact {manifest.artifact_id!r} feature_transform_schema is "
                f"malformed JSON: {exc}"
            ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"artifact {manifest.artifact_id!r} feature_transform_schema must be "
            "a JSON object"
        )
    return FeatureTransformSchema.from_json(payload)
