"""Tests for canonical data materialization compatibility.

Scenarios:
- MATERIALIZATION_10: Canonical materialization preserves the research contract.
"""
from __future__ import annotations

from src.stocks.data.materialization import (
    NetAlphaMaterializationRequest,
    materialize_net_alpha_snapshot,
)
from src.stocks.data.research_v2 import (
    NetAlphaMaterializationRequest as ResearchV2Request,
    materialize_net_alpha_snapshot as research_v2_materialize,
)


class TestMaterializationCompatibility:
    """Materialization adapter preserves the legacy request contract."""

    def test_request_class_is_same(self) -> None:
        assert issubclass(NetAlphaMaterializationRequest, ResearchV2Request)

    def test_function_is_same(self) -> None:
        assert callable(materialize_net_alpha_snapshot)
        assert materialize_net_alpha_snapshot.__name__ == research_v2_materialize.__name__


class TestResearchV2BackwardCompatibility:
    """ResearchV2 module retains backward compatibility."""

    def test_request_class_exists(self) -> None:
        assert hasattr(ResearchV2Request, "source_snapshot_id")

    def test_materialize_function_exists(self) -> None:
        assert callable(research_v2_materialize)


def test_materialization_uses_one_request_type() -> None:  # noqa: N802
    from src.stocks.data.materialization import NetAlphaMaterializationRequest as MatRequest  # noqa: N812
    from src.stocks.data.research_v2 import NetAlphaMaterializationRequest as V2Request  # noqa: N812
    from src.stocks.data.materialization import materialize_net_alpha_snapshot as MatFunc  # noqa: N812
    from src.stocks.data.research_v2 import materialize_net_alpha_snapshot as V2Func  # noqa: N812

    assert MatRequest is V2Request
    assert MatFunc is V2Func
    # no subclass-only behavior: MatRequest should not be subclass of V2Request separately
    assert not (issubclass(MatRequest, V2Request) and MatRequest is not V2Request)
