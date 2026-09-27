"""Shared provider answer shape for range-planned collection jobs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["RawResponse"]


@dataclass(frozen=True, slots=True)
class RawResponse:
    """One provider answer exactly as returned, with the query that produced it."""

    query: Mapping[str, str]
    rows: tuple[Mapping[str, object], ...]
