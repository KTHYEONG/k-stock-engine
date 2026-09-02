"""Certified decision-time bounded snapshot reader."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import polars as pl

from src.data.schemas import PITDataError, PITSnapshotRequest, SilverTable

_FORBIDDEN_SUBSTRINGS = ("data/evidence", "data/canonical", "data/derived", "data/catalog")


class PITSnapshotRepository:
    """Bounded PIT snapshot over certified Silver frames."""

    def __init__(self, frames: Mapping[SilverTable, pl.DataFrame], root: Path) -> None:
        self._frames = dict(frames)
        self._root = Path(root)
        # Never resolve forbidden paths
        root_str = str(self._root)
        for forbidden in _FORBIDDEN_SUBSTRINGS:
            if forbidden in root_str:
                raise PITDataError(f"forbidden path in snapshot repository: {forbidden}")

    @classmethod
    def from_frames(
        cls, frames: Mapping[SilverTable, pl.DataFrame], *, root: Path
    ) -> PITSnapshotRepository:
        # Validate that frames don't contain forbidden resolution? Just store.
        return cls(frames, root)

    def snapshot(self, request: PITSnapshotRequest) -> dict[SilverTable, pl.DataFrame]:
        if request.decision_time.tzinfo is None:
            raise PITDataError("decision_time must be timezone-aware")
        result: dict[SilverTable, pl.DataFrame] = {}
        for table in request.required_tables:
            frame = self._frames.get(table)
            if frame is None:
                raise PITDataError(f"required Silver table is missing: {table.value}")
            if "available_at" not in frame.columns:
                raise PITDataError(f"Silver table lacks available_at: {table.value}")
            available_at = frame["available_at"]
            if getattr(available_at.dtype, "time_zone", None) is None:
                raise PITDataError(f"available_at must be timezone-aware: {table.value}")
            # Use Polars lazy filter for multi-million rows
            filtered = frame.filter(pl.col("available_at") <= request.decision_time)
            result[table] = filtered
        return result
