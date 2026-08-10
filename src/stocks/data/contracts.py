"""Stock data-layer contracts.

``DatasetSnapshot`` is the validated input passed into stock workflows: it
bundles the manifest with the frame it describes, so a workflow never has to
read Parquet or manufacture a manifest. Research contracts (coverage ranges,
windows, timing conventions) are the shared vocabulary of the immutable catalog
and its snapshot resolver.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import polars as pl

from src.core.datasets import DatasetManifest


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """A validated stock dataset: manifest plus the frame it describes."""

    manifest: DatasetManifest
    frame: pl.DataFrame


class TimingConvention(StrEnum):
    """Declared decision/execution timing of a research snapshot."""

    DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN = "decision_after_close_execute_next_open"


@dataclass(frozen=True, slots=True)
class CoverageRange:
    """Inclusive ``[start, end]`` date coverage of a catalog entry."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("coverage start must not be after end")

    def contains(self, other: CoverageRange) -> bool:
        return self.start <= other.start and self.end >= other.end

    def intersects(self, other: CoverageRange) -> bool:
        return self.start <= other.end and self.end >= other.start

    def to_json(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> CoverageRange:
        return cls(
            start=date.fromisoformat(str(data["start"])),
            end=date.fromisoformat(str(data["end"])),
        )


@dataclass(frozen=True, slots=True)
class ResearchWindows:
    """Inclusive train/validation/test windows of a research snapshot."""

    train: CoverageRange
    validation: CoverageRange
    test: CoverageRange

    def __post_init__(self) -> None:
        if self.train.end >= self.validation.start:
            raise ValueError("validation must start after train ends")
        if self.validation.end >= self.test.start:
            raise ValueError("test must start after validation ends")

    @property
    def research_range(self) -> CoverageRange:
        return CoverageRange(start=self.train.start, end=self.test.end)

    def to_json(self) -> dict[str, object]:
        return {
            "train": self.train.to_json(),
            "validation": self.validation.to_json(),
            "test": self.test.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> ResearchWindows:
        train = data["train"]
        validation = data["validation"]
        test = data["test"]
        if not (
            isinstance(train, dict)
            and isinstance(validation, dict)
            and isinstance(test, dict)
        ):
            raise ValueError("research windows must be JSON objects")
        return cls(
            train=CoverageRange.from_json(train),
            validation=CoverageRange.from_json(validation),
            test=CoverageRange.from_json(test),
        )
