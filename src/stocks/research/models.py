"""Model protocol and deterministic baseline.

A model is an interchangeable implementation behind a protocol, not the
architecture. The platform first supports a deterministic baseline for pipeline
and metric validation; candidates are added through the same protocol only after
research approval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

from src.core.instruments import AssetKind


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Immutable model metadata; no ``latest`` alias is permitted."""

    artifact_id: str
    asset_kind: AssetKind
    feature_set: str
    feature_schema_hash: str
    universe_policy_hash: str
    label_definition: str
    label_horizon_sessions: int
    eligible_from: str
    eligible_to: str
    model_type: str = "baseline"
    params: dict[str, str] | None = None

    @property
    def eligible_time_range(self) -> tuple[str, str]:
        return (self.eligible_from, self.eligible_to)


@runtime_checkable
class Model(Protocol):
    """Interchangeable model implementation contract."""

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None: ...

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    def manifest(self) -> ModelManifest: ...


class DeterministicBaseline:
    """Deterministic, data-independent ranking model.

    Scores instruments by a configured ranking feature so the full pipeline
    (dataset -> folds -> artifact -> score -> portfolio) can be validated without
    any fitted parameters.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        ranking_feature: str = "feature_momentum_5d",
        descending: bool = True,
    ):
        self._manifest = manifest
        self._ranking_feature = ranking_feature
        self._descending = descending

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        if self._ranking_feature not in train.columns:
            raise ValueError(f"missing ranking feature {self._ranking_feature!r} in training fold")

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self._ranking_feature not in frame.columns:
            raise ValueError(f"missing ranking feature {self._ranking_feature!r} in frame")
        score_expr = (
            pl.col(self._ranking_feature)
            if self._descending
            else -pl.col(self._ranking_feature)
        )
        return frame.with_columns(score_expr.alias("pred_score"))

    def manifest(self) -> ModelManifest:
        return self._manifest
