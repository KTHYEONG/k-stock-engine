"""Entry/exit-semantic stock labels with declared horizon and cost treatment."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import polars as pl

ID_COLUMN = "instrument_id"
SESSION_COLUMN = "session"


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """Declares entry/exit fields, holding horizon, and cost assumption.

    The canonical phase-1 label is ``log(close[T+horizon] / open[T+1])``: a
    decision after the close of session ``T`` enters at the next eligible open
    and liquidates at the close of ``T + horizon``. Labels are built after the
    point-in-time universe and never recomputed inside a trainer.
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

    def apply(
        self,
        frame: pl.DataFrame,
        id_column: str = ID_COLUMN,
        session_column: str = SESSION_COLUMN,
    ) -> pl.DataFrame:
        """Sort by instrument then session and compute next-open forward-close labels.

        Labels are emitted only when both the next-session entry price and the
        forward exit price exist; terminal rows carry a null label so they can be
        excluded from training rather than guessed.
        """
        required = (id_column, session_column, self.entry_field, self.exit_field)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"label {self.name} requires {', '.join(missing)}")
        ordered = frame.sort([id_column, session_column])
        entry = pl.col(self.entry_field).shift(-1)
        exit_price = pl.col(self.exit_field).shift(-self.horizon_sessions)
        label = (
            pl.when(entry.is_not_null() & exit_price.is_not_null())
            .then(exit_price.log() - entry.log())
            .otherwise(None)
            .over(id_column)
            .alias(self.name)
        )
        return ordered.with_columns(label)
