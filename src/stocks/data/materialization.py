"""Canonical net-alpha materialization entry point.

Re-exports ``NetAlphaMaterializationRequest``, ``NetAlphaMaterializationResult``,
and ``materialize_net_alpha_snapshot`` from the existing research_v2 module.
"""
from __future__ import annotations

from src.stocks.data.research_v2 import (
    NetAlphaMaterializationRequest as _NetAlphaMaterializationRequest,
)
from src.stocks.data.research_v2 import (
    NetAlphaMaterializationResult,
)
from src.stocks.data.research_v2 import (
    materialize_net_alpha_snapshot as _materialize_net_alpha_snapshot,
)


class NetAlphaMaterializationRequest(_NetAlphaMaterializationRequest):
    """Canonical materialization request type."""


def materialize_net_alpha_snapshot(
    request: NetAlphaMaterializationRequest,
) -> NetAlphaMaterializationResult:
    """Materialize the canonical net-alpha snapshot."""
    return _materialize_net_alpha_snapshot(request)

__all__ = [
    "NetAlphaMaterializationRequest",
    "NetAlphaMaterializationResult",
    "materialize_net_alpha_snapshot",
]
