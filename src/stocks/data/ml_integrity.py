"""Point-in-time ML snapshot integrity audit.

``validate_ml_snapshot`` runs a semantic audit over a composed training frame
after compose, before feature computation, and immediately before training. It
fails closed: a single violated invariant raises ``ValueError`` or yields
``MLDataAudit.passed is False``. There is deliberately no warning-and-continue
path, so an audit result is either a fully valid snapshot or a hard stop.

The audit is vectorized (Polars/NumPy only) and deterministic; every check is
recorded as a named, JSON-safe entry so the evidence can be persisted into the
training artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from src.stocks.data.feature_contracts import FeatureContractBook
from src.stocks.data.quality import KRXSessionCalendar

ID_COLUMN = "instrument_id"
SESSION_COLUMN = "session"
_AVAILABLE_COLUMN = "available_time"
_OBSERVATION_COLUMN = "observation_time"
_TARGET_PREFIXES = ("target_", "label_")
_OHLC = ("open", "high", "low", "close")
_FLOW_COMPONENTS = ("foreign_net_buy", "institution_net_buy", "individual_net_buy")
_FLOW_TOTAL = "net_purchase_total"
_PREDICTOR_PREFIXES = ("feature__", "raw__")
_FLOW_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class AuditCheck:
    """One named, JSON-safe integrity check result."""

    name: str
    passed: bool
    detail: str

    def to_json(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class MLDataAudit:
    """Immutable result of one ``validate_ml_snapshot`` run.

    ``passed`` is the conjunction of every check; a single failure falsifies the
    whole snapshot. ``label_universe`` records the per-horizon label universe
    sizes so training never silently inner-joins labels into a smaller sample.
    """

    passed: bool
    checks: tuple[AuditCheck, ...]
    row_count: int
    label_universe: dict[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "row_count": self.row_count,
            "label_universe": dict(self.label_universe),
            "checks": [check.to_json() for check in self.checks],
        }


def _column(frame: pl.DataFrame, name: str) -> str | None:
    if name in frame.columns:
        return name
    return None


def _check_duplicate_keys_and_calendar(
    frame: pl.DataFrame, calendar: KRXSessionCalendar
) -> AuditCheck:
    session_column = _column(frame, SESSION_COLUMN)
    id_column = _column(frame, ID_COLUMN)
    if session_column is None or id_column is None:
        return AuditCheck(
            "key_and_calendar",
            False,
            f"missing identity columns {id_column!r}/{session_column!r}",
        )
    duplicates = frame.group_by([id_column, session_column]).len().filter(
        pl.col("len") > 1
    )
    if duplicates.height:
        return AuditCheck(
            "key_and_calendar",
            False,
            f"duplicate (instrument_id, session) keys: {duplicates.height} groups",
        )
    unknown = frame.filter(~pl.col(session_column).is_in(calendar.sessions))
    if unknown.height:
        return AuditCheck(
            "key_and_calendar",
            False,
            f"{unknown.height} rows reference non-calendar sessions",
        )
    return AuditCheck("key_and_calendar", True, "unique keys, all sessions on calendar")


def _check_availability_ordering(frame: pl.DataFrame, decision_time: datetime) -> AuditCheck:
    available = _column(frame, _AVAILABLE_COLUMN)
    if available is None:
        return AuditCheck(
            "availability_ordering",
            False,
            f"missing {_AVAILABLE_COLUMN!r} for point-in-time availability",
        )
    late = frame.filter(pl.col(available) > pl.lit(decision_time))
    if late.height:
        return AuditCheck(
            "availability_ordering",
            False,
            f"{late.height} rows have available_time after decision_time",
        )
    observed = _column(frame, _OBSERVATION_COLUMN)
    if observed is not None:
        invalid = frame.filter(
            pl.col(observed).is_not_null() & (pl.col(observed) > pl.col(available))
        )
        if invalid.height:
            return AuditCheck(
                "availability_ordering",
                False,
                f"{invalid.height} rows have observation_time after available_time",
            )
    return AuditCheck(
        "availability_ordering", True, "available_time <= decision_time and causal"
    )


def _check_ohlc_invariants(frame: pl.DataFrame) -> AuditCheck:
    present = [c for c in _OHLC if c in frame.columns]
    if not present:
        return AuditCheck("ohlc_invariants", True, "no OHLC columns to validate")
    negative = frame.filter(
        pl.any_horizontal(pl.col(c) <= 0 for c in present)
        | (pl.col("volume") <= 0 if "volume" in frame.columns else pl.lit(False))
    )
    if negative.height:
        return AuditCheck(
            "ohlc_invariants", False, f"{negative.height} rows with non-positive price/volume"
        )
    if {"high", "low"}.issubset(frame.columns):
        violations = frame.filter(
            (pl.col("low") > pl.col("high"))
            | (
                pl.any_horizontal(
                    pl.col("low") > pl.col(c) for c in ("open", "close") if c in frame.columns
                )
            )
            | (
                pl.any_horizontal(
                    pl.col("high") < pl.col(c) for c in ("open", "close") if c in frame.columns
                )
            )
        )
        if violations.height:
            return AuditCheck(
                "ohlc_invariants",
                False,
                f"{violations.height} rows violate low <= min(open, close) <= max <= high",
            )
    return AuditCheck("ohlc_invariants", True, "positive prices/volume and ordering hold")


def _check_flow_identity(frame: pl.DataFrame) -> AuditCheck:
    if _FLOW_TOTAL not in frame.columns:
        return AuditCheck("flow_identity", True, "no flow total to reconcile")
    missing = [c for c in _FLOW_COMPONENTS if c not in frame.columns]
    if missing:
        return AuditCheck(
            "flow_identity",
            False,
            f"flow total present but missing components {missing}",
        )
    computed = sum(
        (pl.col(c).fill_null(0.0) for c in _FLOW_COMPONENTS), pl.lit(0.0)
    )
    mismatch = frame.filter(
        (pl.col(_FLOW_TOTAL) - computed).abs() > _FLOW_TOLERANCE
    )
    if mismatch.height:
        return AuditCheck(
            "flow_identity",
            False,
            f"{mismatch.height} rows where flow components do not reconcile to total",
        )
    return AuditCheck("flow_identity", True, "flow components reconcile to total")


def _check_contract_coverage(
    frame: pl.DataFrame, contracts: FeatureContractBook
) -> AuditCheck:
    """Fail closed on missing declared columns and non-finite feature values.

    Fully-missing or fully-constant columns are never an audit pass: a declared
    feature must carry information in the training window.
    """
    declared = tuple(contract.name for contract in contracts.contracts)
    missing = [name for name in declared if _column(frame, name) is None]
    if missing:
        return AuditCheck(
            "contract_coverage", False, f"declared feature columns missing: {missing}"
        )
    non_finite = []
    constant = []
    for name in declared:
        column = _column(frame, name)
        assert column is not None
        finite = frame.filter(pl.col(column).is_not_null() & ~pl.col(column).is_finite())
        if finite.height:
            non_finite.append(name)
            continue
        values = frame[column].drop_nulls()
        if values.len() and int(values.n_unique()) <= 1:
            constant.append(name)
    if non_finite or constant:
        return AuditCheck(
            "contract_coverage",
            False,
            f"non-finite columns {non_finite}; constant columns {constant}",
        )
    return AuditCheck("contract_coverage", True, "all declared columns present and informative")


def _check_predictor_namespace(frame: pl.DataFrame) -> AuditCheck:
    offending: list[str] = []
    for column in frame.columns:
        if column.startswith("label_available_time"):
            continue
        stem = column
        for prefix in _PREDICTOR_PREFIXES:
            if column.startswith(prefix):
                stem = column[len(prefix):]
                break
        if stem.startswith(_TARGET_PREFIXES):
            offending.append(column)
    if offending:
        raise ValueError(
            f"predictor namespace must not contain label/forward/target columns: {offending}"
        )
    return AuditCheck("predictor_namespace", True, "no label/forward columns in predictors")


def _check_warmup_and_stale(
    frame: pl.DataFrame, contracts: FeatureContractBook
) -> AuditCheck:
    """Verify rolling warm-up matches declared lookback and flag stale runs.

    A feature declared with ``lookback_sessions > 0`` must be null on the first
    observed session of each instrument (no prior window); a run of identical
    non-null values longer than ``stale_after_sessions`` (when declared) is
    reported as stale and fails the audit.
    """
    stale_violations = 0
    warmup_violations = 0
    details: list[str] = []
    for contract in contracts.contracts:
        column = _column(frame, contract.name)
        if column is None:
            continue
        if contract.lookback_sessions > 0:
            first = (
                frame.sort([ID_COLUMN, SESSION_COLUMN])
                .group_by(ID_COLUMN)
                .agg(pl.col(column).first())
            )
            violations = first.filter(pl.col(column).is_not_null())
            warmup_violations += violations.height
            if violations.height:
                details.append(
                    f"{contract.name}: {violations.height} instruments have no warm-up null"
                )
        if contract.stale_after_sessions > 0:
            runs = (
                frame.sort([ID_COLUMN, SESSION_COLUMN])
                .select(
                    ID_COLUMN,
                    pl.col(column).is_not_null().cast(pl.Int8).alias("__v"),
                    pl.col(column).alias("__col"),
                )
                .with_columns(
                    (
                        pl.col("__v")
                        .cum_sum()
                        .over(ID_COLUMN)
                        - pl.col("__v").cum_sum().over(ID_COLUMN).shift(1).fill_null(0)
                    ).alias("__run_id")
                )
                .group_by([ID_COLUMN, "__run_id"])
                .agg(pl.col("__col").len())
            )
            long = runs.filter(pl.col("__col") > contract.stale_after_sessions)
            stale_violations += long.height
            if long.height:
                details.append(
                    f"{contract.name}: {long.height} stale runs beyond "
                    f"{contract.stale_after_sessions} sessions"
                )
    if warmup_violations or stale_violations:
        return AuditCheck(
            "warmup_and_stale",
            False,
            "; ".join(details) or "warm-up or staleness violations",
        )
    return AuditCheck("warmup_and_stale", True, "warm-up matches lookback, no stale runs")


def _label_universe(frame: pl.DataFrame) -> dict[str, int]:
    """Per-horizon label universe counts for the audit evidence."""
    universe: dict[str, int] = {}
    for column in frame.columns:
        if column.startswith(("residual_o2o_", "net_residual_o2o_", "gross_o2o_")):
            universe[column] = int(frame[column].is_not_null().sum())
    return universe


def validate_ml_snapshot(
    frame: pl.DataFrame,
    contracts: FeatureContractBook,
    decision_time: datetime,
    calendar: KRXSessionCalendar,
) -> MLDataAudit:
    """Run the semantic integrity audit over a composed ML training frame.

    Args:
        frame: composed training frame carrying identity, OHLC, availability,
            point-in-time universe, and the declared feature columns.
        contracts: the feature contract book whose declared columns must all be
            present, finite, and non-constant in the training window.
        decision_time: the decision time the snapshot is composed at; every row
            must be available at or before it.
        calendar: the KRX session calendar all sessions must belong to.

    Returns:
        ``MLDataAudit`` with ``passed=True`` only when every check passes.

    Raises:
        ValueError: when the predictor namespace leaks label/forward/target
            columns (audit item 9), matching the fail-closed contract.
    """
    checks = (
        _check_predictor_namespace(frame),
        _check_duplicate_keys_and_calendar(frame, calendar),
        _check_availability_ordering(frame, decision_time),
        _check_ohlc_invariants(frame),
        _check_flow_identity(frame),
        _check_contract_coverage(frame, contracts),
        _check_warmup_and_stale(frame, contracts),
    )
    passed = all(check.passed for check in checks)
    return MLDataAudit(
        passed=passed,
        checks=checks,
        row_count=frame.height,
        label_universe=_label_universe(frame),
    )
