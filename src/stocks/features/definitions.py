"""Versioned, dependency-declared stock feature definitions."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import polars as pl


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """A single versioned feature with declared input dependencies."""

    name: str
    version: int
    inputs: tuple[str, ...] = ()
    description: str = ""

    def render(self, frame: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError(f"{self.name} has no renderer; subclass it")

    @property
    def fingerprint(self) -> str:
        return sha256(
            f"{self.name}@{self.version}:{','.join(self.inputs)}".encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class MomentumFeature(FeatureDefinition):
    """(close - close_shift_n) / close_shift_n momentum feature."""

    lookback: int = 5

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

    def render(self, frame: pl.DataFrame) -> pl.DataFrame:
        if "close" not in frame.columns:
            raise ValueError("missing close column")
        shifted = pl.col("close").shift(self.lookback)
        return frame.with_columns(((pl.col("close") - shifted) / shifted).alias(self.name))


def feature_set_fingerprint(features: list[FeatureDefinition]) -> str:
    """Deterministic fingerprint of an ordered feature set definition."""
    return sha256(
        "\n".join(f.fingerprint for f in features).encode("utf-8")
    ).hexdigest()


def build_features(frame: pl.DataFrame, features: list[FeatureDefinition]) -> pl.DataFrame:
    out = frame
    for feature in features:
        out = feature.render(out)
    return out
