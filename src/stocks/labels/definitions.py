"""Entry/exit-semantic stock labels with declared horizon and cost treatment."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import polars as pl


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """Declares entry/exit fields, holding horizon, and cost assumption.

    Labels are built after the point-in-time universe and never recomputed
    inside a trainer.
    """

    name: str
    entry_field: str
    exit_field: str
    horizon_sessions: int
    corporate_action_treatment: str = "none"
    cost_assumption: str = "no-cost"

    def __post_init__(self) -> None:
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be positive")

    @property
    def fingerprint(self) -> str:
        return sha256(
            f"{self.name}:{self.entry_field}:{self.exit_field}:"
            f"{self.horizon_sessions}:{self.corporate_action_treatment}:{self.cost_assumption}".encode()
        ).hexdigest()

    def apply(self, frame: pl.DataFrame, id_column: str = "instrument_id") -> pl.DataFrame:
        if self.entry_field not in frame.columns or self.exit_field not in frame.columns:
            raise ValueError(
                f"label {self.name} requires {self.entry_field!r} and {self.exit_field!r} columns"
            )
        return frame.with_columns(
            (
                (pl.col(self.exit_field).shift(-self.horizon_sessions) / pl.col(self.entry_field)) - 1.0
            ).over(id_column).alias(self.name)
        )
