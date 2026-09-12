"""Informativeness gate blocking uninformative Gold datasets."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from src.data.schemas import PITDataError


@dataclass(frozen=True, slots=True)
class FeatureCoverageFloor:
    """Minimum non-null coverage required for one feature column."""

    column: str
    min_non_null_ratio: float

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("column must be non-empty")
        if not 0.0 <= self.min_non_null_ratio <= 1.0:
            raise ValueError("min_non_null_ratio must be within [0.0, 1.0]")


@dataclass(frozen=True, slots=True)
class FeatureCoverageReport:
    """Measured non-null ratios with below-floor violations."""

    row_count: int
    ratios: Mapping[str, float]
    violations: tuple[str, ...]


CHAMPION_SCORE_COVERAGE_FLOORS: tuple[FeatureCoverageFloor, ...] = (
    FeatureCoverageFloor(column="champion_score", min_non_null_ratio=0.2),
    FeatureCoverageFloor(column="rank", min_non_null_ratio=0.2),
)


def evaluate_feature_coverage(*, frame: pl.DataFrame, floors: tuple[FeatureCoverageFloor, ...]) -> FeatureCoverageReport:
    """Measure non-null ratios for declared feature columns.

    Args:
        frame: Gold feature frame under review.
        floors: Required coverage floors keyed by column name.

    Returns:
        Coverage report with per-column ratios and ascending violations.

    Raises:
        PITDataError: If the frame has no rows or a declared column is absent.
    """
    row_count = frame.height
    if row_count == 0:
        raise PITDataError("gold frame has no rows")
    minimums = {floor.column: floor.min_non_null_ratio for floor in floors}
    ratios: dict[str, float] = {}
    for floor in floors:
        if floor.column not in frame.columns:
            raise PITDataError(f"missing declared feature column: {floor.column}")
        # 비율은 선언 컬럼별 non-null 건수 기준.
        ratios[floor.column] = (row_count - frame[floor.column].null_count()) / row_count
    violations = tuple(sorted(column for column, ratio in ratios.items() if ratio < minimums[column]))
    return FeatureCoverageReport(row_count=row_count, ratios=ratios, violations=violations)


def certify_informative_gold(
    *, frame: pl.DataFrame, floors: tuple[FeatureCoverageFloor, ...], dataset_label: str
) -> FeatureCoverageReport:
    """Fail closed when a Gold dataset carries no usable feature signal.

    Args:
        frame: Gold feature frame under review.
        floors: Required coverage floors keyed by column name.
        dataset_label: Dataset label recorded in the failure message.

    Returns:
        Coverage report when every floor is satisfied.

    Raises:
        PITDataError: If any declared column falls below its floor, or the
            frame is empty or misses a declared column.
    """
    report = evaluate_feature_coverage(frame=frame, floors=floors)
    if report.violations:
        # 위반 컬럼·실측 비율·데이터셋 라벨을 메시지에 포함한다.
        details = ", ".join(f"{column}={round(report.ratios[column], 4)}" for column in report.violations)
        raise PITDataError(f"uninformative gold {dataset_label}: {details} below coverage floor")
    return report
