"""Shared LS ``t1702`` fixtures that satisfy the share-unit contract.

The LS adapter requires every ``tjj0000..tjj0018`` group and its aggregate
identities, so a fixture must be a complete, zero-sum row rather than the three
fields a test happens to read.
"""
from __future__ import annotations

from typing import Any


def ls_investor_row(session: str, *, scale: int = 1) -> dict[str, Any]:
    """Return one identity-consistent LS row (individual +927, foreign -828, other +129, institution -228)."""
    row: dict[str, Any] = {
        "date": session,
        "tjj0000": -100, "tjj0001": -50, "tjj0002": -30, "tjj0003": -20,
        "tjj0004": -10, "tjj0005": -10, "tjj0006": -8, "tjj0007": 100,
        "tjj0008": 927, "tjj0009": -800, "tjj0010": -28, "tjj0011": 29,
        "tjj0016": -828, "tjj0017": 129, "tjj0018": -228,
    }
    return {key: (value * scale if key.startswith("tjj") else value) for key, value in row.items()}


def ls_share_record(ticker: str, session: str, *, foreign: int = -828) -> dict[str, Any]:
    """Return one normalized LS share-unit record as the adapter emits it."""
    return {
        "ticker": ticker,
        "session": session,
        "unit": "shares",
        "individual_net_shares": 927,
        "foreign_net_shares": foreign,
        "institution_net_shares": -228,
        "other_net_shares": 129,
    }
